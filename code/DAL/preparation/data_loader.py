import os
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from DAL.preparation.config import (
    ALL_LABELS,
    IMG_SIZE,
    BATCH_SIZE,
    NUM_WORKERS,
    MAX_PATCHES,
)
from DAL.preparation.lesion_utils import extract_lesion_candidates


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.15,
            hue=0.02,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_eval_transforms(img_size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _safe_label_list(value):
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None:
        return []
    if isinstance(value, float) and np.isnan(value):
        return []
    text = str(value).strip()
    return text.split() if text else []


def _one_hot_from_label_list(label_list, all_labels):
    label_set = {str(x) for x in label_list}
    return np.array(
        [1.0 if cls in label_set else 0.0 for cls in all_labels],
        dtype=np.float32,
    )


def _target_from_row(row: pd.Series, all_labels):
    encoded_cols = [f"y_{cls}" for cls in all_labels]

    if all(col in row.index for col in encoded_cols):
        return torch.tensor(
            [float(row[col]) for col in encoded_cols],
            dtype=torch.float32,
        )

    if "label_list" in row.index:
        return torch.tensor(
            _one_hot_from_label_list(_safe_label_list(row["label_list"]), all_labels),
            dtype=torch.float32,
        )

    if "labels" in row.index:
        return torch.tensor(
            _one_hot_from_label_list(_safe_label_list(row["labels"]), all_labels),
            dtype=torch.float32,
        )

    return torch.zeros(len(all_labels), dtype=torch.float32)


def _patch_rgb_to_pil(patch_rgb: np.ndarray) -> Image.Image:
    patch_uint8 = (np.clip(patch_rgb, 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(patch_uint8)


def _resize_binary_mask(mask: np.ndarray, out_size: int = IMG_SIZE) -> torch.Tensor:
    """
    Convert variable-size lesion mask to fixed [1, out_size, out_size]
    باستخدام nearest-neighbor حتى نحافظ على binary structure.
    """
    mask_uint8 = (mask > 0).astype(np.uint8) * 255
    resized = cv2.resize(
        mask_uint8,
        (out_size, out_size),
        interpolation=cv2.INTER_NEAREST,
    )
    resized = (resized > 0).astype(np.float32)
    return torch.from_numpy(resized).unsqueeze(0)  # [1, H, W]


def _compute_patch_lesion_ratio(mask: np.ndarray, box) -> float:
    """
    نسبة تغطية الـ lesion داخل patch box.
    هذا مفيد جدًا لاحقًا لو بدك lesion sparsity / patch attention guidance.
    """
    x1, y1, x2, y2 = box

    h, w = mask.shape[:2]
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))

    if x2 <= x1 or y2 <= y1:
        return 0.0

    crop = mask[y1:y2, x1:x2]
    area = float((y2 - y1) * (x2 - x1))
    if area <= 0:
        return 0.0

    lesion_pixels = float((crop > 0).sum())
    return lesion_pixels / area


class ImagePatchDataset(Dataset):
    """
    Image-level dataset
    - reads full image normally
    - extracts lesion candidate patches
    - returns fixed patch tensors
    - keeps lesion information in a training-compatible form
    """

    def __init__(
        self,
        image_df: pd.DataFrame,
        image_root: str,
        transform=None,
        all_labels=None,
        return_targets: bool = True,
        max_patches: int = MAX_PATCHES,
    ):
        self.df = image_df.reset_index(drop=True).copy()
        self.image_root = image_root
        self.transform = transform if transform is not None else get_eval_transforms()
        self.all_labels = all_labels if all_labels is not None else ALL_LABELS
        self.return_targets = return_targets
        self.max_patches = max_patches

        if "image" not in self.df.columns:
            raise ValueError("image_df must contain 'image' column.")

    def __len__(self):
        return len(self.df)

    def _read_image(self, image_name: str):
        image_path = os.path.join(self.image_root, str(image_name))
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image_bgr, image_path

    def _build_patch_stack(self, image_bgr):
        patches_rgb, boxes, lesion_mask = extract_lesion_candidates(
            image_bgr=image_bgr,
            max_patches=self.max_patches,
        )

        patch_tensors = []
        patch_boxes = []
        patch_valid_mask = []
        patch_lesion_ratio = []

        for patch_rgb, box in zip(patches_rgb, boxes):
            patch_pil = _patch_rgb_to_pil(patch_rgb)
            patch_tensor = self.transform(patch_pil)

            patch_tensors.append(patch_tensor)
            patch_boxes.append(torch.tensor(box, dtype=torch.int64))
            patch_valid_mask.append(1.0)
            patch_lesion_ratio.append(float(_compute_patch_lesion_ratio(lesion_mask, box)))

        while len(patch_tensors) < self.max_patches:
            patch_tensors.append(torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32))
            patch_boxes.append(torch.tensor([-1, -1, -1, -1], dtype=torch.int64))
            patch_valid_mask.append(0.0)
            patch_lesion_ratio.append(0.0)

        patch_stack = torch.stack(patch_tensors, dim=0)               # [K,3,H,W]
        patch_boxes = torch.stack(patch_boxes, dim=0)                 # [K,4]
        patch_valid_mask = torch.tensor(patch_valid_mask, dtype=torch.float32)     # [K]
        patch_lesion_ratio = torch.tensor(patch_lesion_ratio, dtype=torch.float32) # [K]

        lesion_mask_resized = _resize_binary_mask(lesion_mask, IMG_SIZE)  # [1,H,W]

        return {
            "patches": patch_stack,
            "patch_boxes": patch_boxes,
            "patch_mask": patch_valid_mask,
            "patch_lesion_ratio": patch_lesion_ratio,
            "lesion_mask_resized": lesion_mask_resized,
            "lesion_mask_raw": lesion_mask.astype(np.uint8),  # metadata only
            "num_valid_patches": int(len(patches_rgb)),
        }

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image"]

        image_bgr, image_path = self._read_image(image_name)
        h0, w0 = image_bgr.shape[:2]

        patch_bundle = self._build_patch_stack(image_bgr)

        item = {
            "patches": patch_bundle["patches"],
            "patch_boxes": patch_bundle["patch_boxes"],
            "patch_mask": patch_bundle["patch_mask"],
            "patch_lesion_ratio": patch_bundle["patch_lesion_ratio"],
            "lesion_mask_resized": patch_bundle["lesion_mask_resized"],

            # metadata
            "lesion_mask_raw": patch_bundle["lesion_mask_raw"],
            "num_valid_patches": patch_bundle["num_valid_patches"],
            "image_name": str(image_name),
            "source_path": str(image_path),
            "original_hw": (int(h0), int(w0)),
        }

        if self.return_targets:
            item["target"] = _target_from_row(row, self.all_labels)

        return item


def image_patch_collate_fn(batch: List[dict]):
    """
    stack only fixed-size tensors
    keep variable-size metadata as lists
    """
    out = {}

    tensor_keys = [
        "patches",
        "patch_boxes",
        "patch_mask",
        "patch_lesion_ratio",
        "lesion_mask_resized",
    ]

    optional_tensor_keys = ["target"]

    list_keys = [
        "lesion_mask_raw",
        "num_valid_patches",
        "image_name",
        "source_path",
        "original_hw",
    ]

    for key in tensor_keys:
        out[key] = torch.stack([sample[key] for sample in batch], dim=0)

    for key in optional_tensor_keys:
        if key in batch[0]:
            out[key] = torch.stack([sample[key] for sample in batch], dim=0)

    for key in list_keys:
        if key in batch[0]:
            out[key] = [sample[key] for sample in batch]

    return out


def make_loader(
    image_df: pd.DataFrame,
    image_root: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = NUM_WORKERS,
    train_mode: bool = False,
    all_labels=None,
    return_targets: bool = True,
):
    dataset = ImagePatchDataset(
        image_df=image_df,
        image_root=image_root,
        transform=get_train_transforms() if train_mode else get_eval_transforms(),
        all_labels=all_labels if all_labels is not None else ALL_LABELS,
        return_targets=return_targets,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=image_patch_collate_fn,
    )
    return loader


def create_image_patch_dataloaders(
    pipeline_bundle: Dict,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    image_root = pipeline_bundle["train_dir"]
    all_labels = pipeline_bundle["all_labels"]

    loaders = {}

    if "pool_labeled_df" in pipeline_bundle and len(pipeline_bundle["pool_labeled_df"]) > 0:
        loaders["train"] = make_loader(
            image_df=pipeline_bundle["pool_labeled_df"],
            image_root=image_root,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            train_mode=True,
            all_labels=all_labels,
            return_targets=True,
        )

    if "pool_unlabeled_df" in pipeline_bundle and len(pipeline_bundle["pool_unlabeled_df"]) > 0:
        loaders["unlabeled"] = make_loader(
            image_df=pipeline_bundle["pool_unlabeled_df"],
            image_root=image_root,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            train_mode=False,
            all_labels=all_labels,
            return_targets=False,
        )

    if "pool_val_df" in pipeline_bundle and len(pipeline_bundle["pool_val_df"]) > 0:
        loaders["val"] = make_loader(
            image_df=pipeline_bundle["pool_val_df"],
            image_root=image_root,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            train_mode=False,
            all_labels=all_labels,
            return_targets=True,
        )

    if "pool_test_df" in pipeline_bundle and len(pipeline_bundle["pool_test_df"]) > 0:
        loaders["test"] = make_loader(
            image_df=pipeline_bundle["pool_test_df"],
            image_root=image_root,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            train_mode=False,
            all_labels=all_labels,
            return_targets=True,
        )

    return loaders