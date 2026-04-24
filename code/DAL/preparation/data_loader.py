import os
from typing import Dict

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


class ImagePatchDataset(Dataset):
    """
    Image-level dataset:
    - reads one full image
    - extracts 1..K lesion patches on-the-fly
    - returns fixed-size patch tensor stack [K, 3, H, W]
    - returns image-level multi-label target
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
        patch_mask = []

        for patch_rgb, box in zip(patches_rgb, boxes):
            patch_pil = _patch_rgb_to_pil(patch_rgb)
            patch_tensor = self.transform(patch_pil)

            patch_tensors.append(patch_tensor)
            patch_boxes.append(torch.tensor(box, dtype=torch.int64))
            patch_mask.append(1)

        while len(patch_tensors) < self.max_patches:
            patch_tensors.append(torch.zeros((3, IMG_SIZE, IMG_SIZE), dtype=torch.float32))
            patch_boxes.append(torch.tensor([-1, -1, -1, -1], dtype=torch.int64))
            patch_mask.append(0)

        patch_stack = torch.stack(patch_tensors, dim=0)            # [K, 3, H, W]
        patch_boxes = torch.stack(patch_boxes, dim=0)              # [K, 4]
        patch_mask = torch.tensor(patch_mask, dtype=torch.float32) # [K]

        lesion_mask_tensor = torch.from_numpy(lesion_mask.astype(np.uint8))

        return patch_stack, patch_boxes, patch_mask, lesion_mask_tensor, len(patches_rgb)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_name = row["image"]

        image_bgr, image_path = self._read_image(image_name)
        patch_stack, patch_boxes, patch_mask, lesion_mask_tensor, num_valid_patches = self._build_patch_stack(image_bgr)

        item = {
            "patches": patch_stack,
            "patch_boxes": patch_boxes,
            "patch_mask": patch_mask,
            "lesion_mask": lesion_mask_tensor,
            "num_valid_patches": int(num_valid_patches),
            "image_name": image_name,
            "source_path": image_path,
        }

        if self.return_targets:
            item["target"] = _target_from_row(row, self.all_labels)

        return item


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