import os
import ast
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
    LESION_CONTEXT_PAD_RATIO,
    USE_SAVED_LESION_CROPS,
)

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


def _safe_literal_list(value):
    if isinstance(value, list):
        return [str(x) for x in value]

    if value is None:
        return []

    if isinstance(value, float) and np.isnan(value):
        return []

    text = str(value).strip()
    if not text:
        return []

    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except Exception:
            pass

    return text.split()


def _one_hot_from_label_list(label_list, all_labels):
    label_set = {str(x) for x in label_list}
    return np.array(
        [1.0 if cls in label_set else 0.0 for cls in all_labels],
        dtype=np.float32,
    )


def _target_from_row(row: pd.Series, all_labels):
    """
    Priority:
    1) lesion-level annotation_label
    2) lesion_label
    3) encoded image-level columns y_*
    4) label_list / labels fallback
    """

    if "annotation_label" in row.index:
        ann = row["annotation_label"]
        if ann is not None and not (isinstance(ann, float) and np.isnan(ann)) and str(ann).strip() != "":
            return torch.tensor(
                _one_hot_from_label_list([str(ann)], all_labels),
                dtype=torch.float32,
            )

    if "lesion_label" in row.index:
        ann = row["lesion_label"]
        if ann is not None and not (isinstance(ann, float) and np.isnan(ann)) and str(ann).strip() != "":
            return torch.tensor(
                _one_hot_from_label_list([str(ann)], all_labels),
                dtype=torch.float32,
            )

    encoded_cols = [f"y_{cls}" for cls in all_labels]
    if all(col in row.index for col in encoded_cols):
        return torch.tensor(
            [float(row[col]) for col in encoded_cols],
            dtype=torch.float32,
        )

    if "label_list" in row.index:
        return torch.tensor(
            _one_hot_from_label_list(_safe_literal_list(row["label_list"]), all_labels),
            dtype=torch.float32,
        )

    if "labels" in row.index:
        return torch.tensor(
            _one_hot_from_label_list(_safe_literal_list(row["labels"]), all_labels),
            dtype=torch.float32,
        )

    return torch.zeros(len(all_labels), dtype=torch.float32)


def _has_annotation(row: pd.Series) -> bool:
    for key in ["annotation_label", "lesion_label"]:
        if key in row.index:
            value = row[key]
            if value is not None and not (isinstance(value, float) and np.isnan(value)) and str(value).strip() != "":
                return True

    if "is_annotated" in row.index:
        value = row["is_annotated"]
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes"}
        return bool(value)

    return False


class LesionDataset(Dataset):
    """
    Required manifest columns:
    - image
    - x1, y1, x2, y2

    Optional columns:
    - lesion_id
    - crop_path
    - annotation_label
    - lesion_label
    - label_list / labels
    - is_annotated
    - split
    - pool
    """

    def __init__(
        self,
        manifest_df: pd.DataFrame,
        image_root: str,
        transform=None,
        all_labels=None,
        return_targets: bool = True,
        include_context_pad_ratio: float = LESION_CONTEXT_PAD_RATIO,
    ):
        self.df = manifest_df.reset_index(drop=True).copy()
        self.image_root = image_root
        self.transform = transform if transform is not None else get_eval_transforms()
        self.all_labels = all_labels if all_labels is not None else ALL_LABELS
        self.return_targets = return_targets
        self.include_context_pad_ratio = include_context_pad_ratio
        self.use_saved_lesion_crops = USE_SAVED_LESION_CROPS

        required = {"image", "x1", "y1", "x2", "y2"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    def __len__(self):
        return len(self.df)

    def _read_full_image(self, image_name: str):
        image_path = os.path.join(self.image_root, str(image_name))
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        return image_bgr, image_path

    def _read_crop_from_disk(self, crop_path: str):
        image_bgr = cv2.imread(crop_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read crop image: {crop_path}")
        return image_bgr

    def _crop_from_row(self, row: pd.Series):
        if self.use_saved_lesion_crops and "crop_path" in row.index:
            crop_path = row["crop_path"]
            if crop_path is not None and str(crop_path).strip() != "" and os.path.exists(str(crop_path)):
                crop_bgr = self._read_crop_from_disk(str(crop_path))
                crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
                return Image.fromarray(crop_rgb), str(crop_path), None

        image_bgr, image_path = self._read_full_image(row["image"])
        h, w = image_bgr.shape[:2]

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        pad_x = int(self.include_context_pad_ratio * bw)
        pad_y = int(self.include_context_pad_ratio * bh)

        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)

        crop_bgr = image_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        return Image.fromarray(crop_rgb), image_path, (x1, y1, x2, y2)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_pil, source_path, effective_box = self._crop_from_row(row)
        image_tensor = self.transform(image_pil)

        item = {
            "image": image_tensor,
            "lesion_id": row["lesion_id"] if "lesion_id" in row.index else f"lesion_{idx}",
            "image_name": row["image"],
            "source_path": source_path,
            "x1": int(row["x1"]),
            "y1": int(row["y1"]),
            "x2": int(row["x2"]),
            "y2": int(row["y2"]),
            "has_annotation": _has_annotation(row),
        }

        if effective_box is None:
            item["effective_crop_box"] = torch.tensor([-1, -1, -1, -1], dtype=torch.int64)
        else:
            item["effective_crop_box"] = torch.tensor(effective_box, dtype=torch.int64)

        if self.return_targets:
            item["target"] = _target_from_row(row, self.all_labels)

        return item


def make_loader(
    manifest_df: pd.DataFrame,
    image_root: str,
    batch_size: int = BATCH_SIZE,
    shuffle: bool = False,
    num_workers: int = NUM_WORKERS,
    train_mode: bool = False,
    all_labels=None,
    return_targets: bool = True,
):
    dataset = LesionDataset(
        manifest_df=manifest_df,
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


def create_lesion_dataloaders(
    manifest_bundle: Dict[str, pd.DataFrame],
    data_bundle: Dict,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """
    Expected manifest_bundle keys:
    - labeled_lesions_df
    - unlabeled_lesions_df
    - val_lesions_df
    - test_lesions_df

    data_bundle must contain:
    - train_dir
    - all_labels
    """
    image_root = data_bundle["train_dir"]
    all_labels = data_bundle["all_labels"]

    loaders = {}

    if "labeled_lesions_df" in manifest_bundle and len(manifest_bundle["labeled_lesions_df"]) > 0:
        loaders["train"] = make_loader(
            manifest_df=manifest_bundle["labeled_lesions_df"],
            image_root=image_root,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            train_mode=True,
            all_labels=all_labels,
            return_targets=True,
        )

    if "unlabeled_lesions_df" in manifest_bundle and len(manifest_bundle["unlabeled_lesions_df"]) > 0:
        loaders["unlabeled"] = make_loader(
            manifest_df=manifest_bundle["unlabeled_lesions_df"],
            image_root=image_root,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            train_mode=False,
            all_labels=all_labels,
            return_targets=False,
        )

    if "val_lesions_df" in manifest_bundle and len(manifest_bundle["val_lesions_df"]) > 0:
        loaders["val"] = make_loader(
            manifest_df=manifest_bundle["val_lesions_df"],
            image_root=image_root,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            train_mode=False,
            all_labels=all_labels,
            return_targets=True,
        )

    if "test_lesions_df" in manifest_bundle and len(manifest_bundle["test_lesions_df"]) > 0:
        loaders["test"] = make_loader(
            manifest_df=manifest_bundle["test_lesions_df"],
            image_root=image_root,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            train_mode=False,
            all_labels=all_labels,
            return_targets=True,
        )

    return loaders


def load_manifest_csv(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing manifest CSV: {csv_path}")
    return pd.read_csv(csv_path)


def create_lesion_dataloaders_from_csvs(
    csv_paths: Dict[str, str],
    data_bundle: Dict,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
):
    """
    Example csv_paths:
    {
        "labeled_lesions_df": ".../labeled_lesions.csv",
        "unlabeled_lesions_df": ".../unlabeled_lesions.csv",
        "val_lesions_df": ".../val_lesions.csv",
        "test_lesions_df": ".../test_lesions.csv",
    }
    """
    manifest_bundle = {
        key: load_manifest_csv(path)
        for key, path in csv_paths.items()
        if path is not None
    }

    return create_lesion_dataloaders(
        manifest_bundle=manifest_bundle,
        data_bundle=data_bundle,
        batch_size=batch_size,
        num_workers=num_workers,
    )