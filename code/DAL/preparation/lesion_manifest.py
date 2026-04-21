import os
import json
import ast
from typing import Dict, List

import cv2
import numpy as np
import pandas as pd

from DAL.preparation.config import (
    ALL_LABELS,
    REPORTS_DIR,
    SAVE_LESION_CROPS,
    ASSIGN_SEED_ANNOTATIONS,
    ANNOTATE_EVAL_SPLITS_WITH_WEAK_LABELS,
    ALLOW_MULTILABEL_SEED_ANNOTATION,
    MIN_LESION_AREA_RATIO,
)
from DAL.preparation.lesion_utils import extract_lesion_candidates


def ensure_lesion_manifest_directories(reports_dir: str = REPORTS_DIR):
    lesion_root = os.path.join(reports_dir, "lesions")
    manifest_dir = os.path.join(lesion_root, "manifests")
    crop_dir = os.path.join(lesion_root, "crops")
    meta_dir = os.path.join(lesion_root, "metadata")

    os.makedirs(lesion_root, exist_ok=True)
    os.makedirs(manifest_dir, exist_ok=True)
    os.makedirs(crop_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)

    return {
        "lesion_root": lesion_root,
        "manifest_dir": manifest_dir,
        "crop_dir": crop_dir,
        "meta_dir": meta_dir,
    }


def _normalize_label_list(value) -> List[str]:
    if value is None:
        return []

    if isinstance(value, list):
        return [str(x) for x in value]

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


def _get_label_list_from_row(row: pd.Series) -> List[str]:
    if "label_list" in row.index:
        return _normalize_label_list(row["label_list"])
    if "labels" in row.index:
        return _normalize_label_list(row["labels"])
    return []


def _get_primary_label(row: pd.Series) -> str:
    if "primary_label" in row.index and str(row["primary_label"]).strip():
        return str(row["primary_label"]).strip()

    label_list = _get_label_list_from_row(row)
    return label_list[0] if len(label_list) > 0 else "unknown"


def _is_multilabel(row: pd.Series) -> bool:
    if "is_multilabel" in row.index:
        value = row["is_multilabel"]
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes"}
        return bool(value)

    label_list = _get_label_list_from_row(row)
    return len(label_list) > 1


def _row_class_vector(row: pd.Series, all_labels: List[str]) -> Dict[str, float]:
    out = {}
    label_list = set(_get_label_list_from_row(row))

    for cls in all_labels:
        col = f"y_{cls}"
        if col in row.index:
            out[col] = float(row[col])
        else:
            out[col] = 1.0 if cls in label_list else 0.0

    return out


def _save_patch_rgb(patch_rgb: np.ndarray, save_path: str):
    patch_bgr = cv2.cvtColor(
        (np.clip(patch_rgb, 0, 1) * 255).astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )
    cv2.imwrite(save_path, patch_bgr)


def _empty_manifest_df():
    return pd.DataFrame(columns=[
        "lesion_id", "image", "image_path", "split", "pool", "crop_path",
        "candidate_index", "num_candidates_from_image",
        "x1", "y1", "x2", "y2", "box_w", "box_h", "box_area",
        "label_list", "primary_label", "is_multilabel",
        "annotation_label", "is_annotated", "annotation_source",
        "query_round", "selected_for_query",
        *[f"y_{cls}" for cls in ALL_LABELS],
    ])


def _build_records_for_split(
    split_df: pd.DataFrame,
    image_root: str,
    split_name: str,
    pool_name: str,
    crops_root: str,
    save_crops: bool = SAVE_LESION_CROPS,
    assign_seed_annotations: bool = False,
    annotate_eval_splits: bool = False,
    allow_multilabel_seed_annotation: bool = ALLOW_MULTILABEL_SEED_ANNOTATION,
):
    if split_df is None or len(split_df) == 0:
        return _empty_manifest_df()

    records = []

    split_crop_dir = os.path.join(crops_root, split_name)
    os.makedirs(split_crop_dir, exist_ok=True)

    for _, row in split_df.iterrows():
        image_name = str(row["image"])
        image_path = os.path.join(image_root, image_name)

        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            print(f"[WARN] Could not read image, skipping: {image_path}")
            continue

        label_list = _get_label_list_from_row(row)
        primary_label = _get_primary_label(row)
        is_multilabel = _is_multilabel(row)
        image_class_vector = _row_class_vector(row, ALL_LABELS)

        try:
            patches, boxes, _ = extract_lesion_candidates(
                image_bgr,
                min_area_ratio=MIN_LESION_AREA_RATIO,
            )
        except Exception as e:
            print(f"[WARN] lesion extraction failed for {image_name}: {e}")
            continue

        image_stem = os.path.splitext(os.path.basename(image_name))[0]

        for lesion_idx, (patch_rgb, box) in enumerate(zip(patches, boxes)):
            x1, y1, x2, y2 = [int(v) for v in box]
            lesion_id = f"{image_stem}_lesion_{lesion_idx}"

            crop_path = ""
            if save_crops:
                crop_filename = f"{lesion_id}.jpg"
                crop_path = os.path.join(split_crop_dir, crop_filename)
                _save_patch_rgb(patch_rgb, crop_path)

            annotation_label = ""
            is_annotated = 0
            annotation_source = ""

            if assign_seed_annotations:
                if (not is_multilabel) or allow_multilabel_seed_annotation:
                    annotation_label = primary_label
                    is_annotated = 1
                    annotation_source = "weak_image_label"

            if annotate_eval_splits and split_name in {"val", "test"}:
                if (not is_multilabel) or allow_multilabel_seed_annotation:
                    annotation_label = primary_label
                    is_annotated = 1
                    annotation_source = "weak_image_label_eval"

            record = {
                "lesion_id": lesion_id,
                "image": image_name,
                "image_path": image_path,
                "split": split_name,
                "pool": pool_name,
                "crop_path": crop_path,
                "candidate_index": lesion_idx,
                "num_candidates_from_image": len(boxes),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "box_w": max(1, x2 - x1),
                "box_h": max(1, y2 - y1),
                "box_area": max(1, x2 - x1) * max(1, y2 - y1),
                "label_list": json.dumps(label_list, ensure_ascii=False),
                "primary_label": primary_label,
                "is_multilabel": int(is_multilabel),
                "annotation_label": annotation_label,
                "is_annotated": int(is_annotated),
                "annotation_source": annotation_source,
                "query_round": -1,
                "selected_for_query": 0,
            }

            record.update(image_class_vector)
            records.append(record)

    if len(records) == 0:
        return _empty_manifest_df()

    return pd.DataFrame(records)


def _save_manifest(df: pd.DataFrame, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)


def _summarize_manifest(df: pd.DataFrame) -> Dict:
    if len(df) == 0:
        return {
            "num_lesions": 0,
            "num_images": 0,
            "annotated_lesions": 0,
            "unannotated_lesions": 0,
        }

    annotated = int(df["is_annotated"].sum()) if "is_annotated" in df.columns else 0

    return {
        "num_lesions": int(len(df)),
        "num_images": int(df["image"].nunique()) if "image" in df.columns else 0,
        "annotated_lesions": annotated,
        "unannotated_lesions": int(len(df) - annotated),
    }


def build_lesion_manifest_bundle(
    data_bundle: Dict,
    reports_dir: str = REPORTS_DIR,
    save_crops: bool = SAVE_LESION_CROPS,
    assign_seed_annotations: bool = ASSIGN_SEED_ANNOTATIONS,
    annotate_eval_splits: bool = ANNOTATE_EVAL_SPLITS_WITH_WEAK_LABELS,
    allow_multilabel_seed_annotation: bool = ALLOW_MULTILABEL_SEED_ANNOTATION,
):
    train_dir = data_bundle["train_dir"]
    effective_reports_dir = reports_dir or data_bundle.get("reports_dir", REPORTS_DIR)

    dirs = ensure_lesion_manifest_directories(effective_reports_dir)
    manifest_dir = dirs["manifest_dir"]
    crop_dir = dirs["crop_dir"]
    meta_dir = dirs["meta_dir"]

    train_pool_lesions_df = _build_records_for_split(
        split_df=data_bundle["train_pool_df"],
        image_root=train_dir,
        split_name="train_pool",
        pool_name="train_pool",
        crops_root=crop_dir,
        save_crops=save_crops,
        assign_seed_annotations=False,
        annotate_eval_splits=False,
        allow_multilabel_seed_annotation=allow_multilabel_seed_annotation,
    )

    labeled_lesions_df = _build_records_for_split(
        split_df=data_bundle["initial_labeled_df"],
        image_root=train_dir,
        split_name="train_seed",
        pool_name="labeled",
        crops_root=crop_dir,
        save_crops=save_crops,
        assign_seed_annotations=assign_seed_annotations,
        annotate_eval_splits=False,
        allow_multilabel_seed_annotation=allow_multilabel_seed_annotation,
    )

    unlabeled_lesions_df = _build_records_for_split(
        split_df=data_bundle["unlabeled_pool_df"],
        image_root=train_dir,
        split_name="train_unlabeled",
        pool_name="unlabeled",
        crops_root=crop_dir,
        save_crops=save_crops,
        assign_seed_annotations=False,
        annotate_eval_splits=False,
        allow_multilabel_seed_annotation=allow_multilabel_seed_annotation,
    )

    val_lesions_df = _build_records_for_split(
        split_df=data_bundle["val_df"],
        image_root=train_dir,
        split_name="val",
        pool_name="val",
        crops_root=crop_dir,
        save_crops=save_crops,
        assign_seed_annotations=False,
        annotate_eval_splits=annotate_eval_splits,
        allow_multilabel_seed_annotation=allow_multilabel_seed_annotation,
    )

    test_lesions_df = _build_records_for_split(
        split_df=data_bundle["test_df"],
        image_root=train_dir,
        split_name="test",
        pool_name="test",
        crops_root=crop_dir,
        save_crops=save_crops,
        assign_seed_annotations=False,
        annotate_eval_splits=annotate_eval_splits,
        allow_multilabel_seed_annotation=allow_multilabel_seed_annotation,
    )

    csv_paths = {
        "train_pool_lesions_df": os.path.join(manifest_dir, "train_pool_lesions.csv"),
        "labeled_lesions_df": os.path.join(manifest_dir, "labeled_lesions.csv"),
        "unlabeled_lesions_df": os.path.join(manifest_dir, "unlabeled_lesions.csv"),
        "val_lesions_df": os.path.join(manifest_dir, "val_lesions.csv"),
        "test_lesions_df": os.path.join(manifest_dir, "test_lesions.csv"),
    }

    _save_manifest(train_pool_lesions_df, csv_paths["train_pool_lesions_df"])
    _save_manifest(labeled_lesions_df, csv_paths["labeled_lesions_df"])
    _save_manifest(unlabeled_lesions_df, csv_paths["unlabeled_lesions_df"])
    _save_manifest(val_lesions_df, csv_paths["val_lesions_df"])
    _save_manifest(test_lesions_df, csv_paths["test_lesions_df"])

    summary = {
        "train_pool": _summarize_manifest(train_pool_lesions_df),
        "labeled": _summarize_manifest(labeled_lesions_df),
        "unlabeled": _summarize_manifest(unlabeled_lesions_df),
        "val": _summarize_manifest(val_lesions_df),
        "test": _summarize_manifest(test_lesions_df),
        "assign_seed_annotations": bool(assign_seed_annotations),
        "annotate_eval_splits": bool(annotate_eval_splits),
        "allow_multilabel_seed_annotation": bool(allow_multilabel_seed_annotation),
    }

    summary_path = os.path.join(meta_dir, "lesion_manifest_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== LESION MANIFEST PIPELINE =====")
    print("Manifest dir:", manifest_dir)
    print("Crop dir:", crop_dir)
    print("Summary path:", summary_path)
    for key, stats in summary.items():
        if isinstance(stats, dict):
            print(f"{key}: {stats}")

    return {
        "train_pool_lesions_df": train_pool_lesions_df,
        "labeled_lesions_df": labeled_lesions_df,
        "unlabeled_lesions_df": unlabeled_lesions_df,
        "val_lesions_df": val_lesions_df,
        "test_lesions_df": test_lesions_df,
        "csv_paths": csv_paths,
        "summary": summary,
        "summary_path": summary_path,
    }


def load_lesion_manifest_bundle(csv_paths: Dict[str, str]):
    out = {}
    for key, path in csv_paths.items():
        if path is None:
            continue
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing lesion manifest CSV: {path}")
        out[key] = pd.read_csv(path)
    return out