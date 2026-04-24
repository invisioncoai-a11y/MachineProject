import os
import json
from typing import Dict, Optional

import pandas as pd

from DAL.preparation.config import (
    ANNOTATIONS_DIR,
    QUERY_BATCH_SIZE,
)


def ensure_pool_manager_directories(annotations_dir: str = ANNOTATIONS_DIR):
    rounds_dir = os.path.join(annotations_dir, "rounds")
    management_dir = os.path.join(annotations_dir, "management")

    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(rounds_dir, exist_ok=True)
    os.makedirs(management_dir, exist_ok=True)

    return {
        "annotations_dir": annotations_dir,
        "rounds_dir": rounds_dir,
        "management_dir": management_dir,
    }


def _align_columns(df: pd.DataFrame, columns):
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out[list(columns)]


def _save_csv(df: pd.DataFrame, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def _save_json(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _require_columns(df: pd.DataFrame, required_cols, df_name: str):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def initialize_image_pool_state_from_split_bundle(
    split_bundle: Dict[str, pd.DataFrame],
    annotations_dir: str = ANNOTATIONS_DIR,
    save_round0_snapshot: bool = True,
):
    """
    Initialize image-level active learning pools from split bundle.
    Required keys:
    - initial_labeled_df
    - unlabeled_pool_df
    - val_df
    - test_df
    """
    dirs = ensure_pool_manager_directories(annotations_dir)
    rounds_dir = dirs["rounds_dir"]
    management_dir = dirs["management_dir"]

    labeled_df = split_bundle.get("initial_labeled_df", pd.DataFrame()).copy()
    unlabeled_df = split_bundle.get("unlabeled_pool_df", pd.DataFrame()).copy()
    val_df = split_bundle.get("val_df", pd.DataFrame()).copy()
    test_df = split_bundle.get("test_df", pd.DataFrame()).copy()

    if len(labeled_df) > 0:
        _require_columns(labeled_df, ["image"], "initial_labeled_df")
    if len(unlabeled_df) > 0:
        _require_columns(unlabeled_df, ["image"], "unlabeled_pool_df")

    base_paths = {
        "labeled": os.path.join(annotations_dir, "labeled_images.csv"),
        "unlabeled": os.path.join(annotations_dir, "unlabeled_images.csv"),
        "val": os.path.join(annotations_dir, "val_images.csv"),
        "test": os.path.join(annotations_dir, "test_images.csv"),
    }

    _save_csv(labeled_df, base_paths["labeled"])
    _save_csv(unlabeled_df, base_paths["unlabeled"])
    _save_csv(val_df, base_paths["val"])
    _save_csv(test_df, base_paths["test"])

    summary = {
        "num_labeled_images": int(len(labeled_df)),
        "num_unlabeled_images": int(len(unlabeled_df)),
        "num_val_images": int(len(val_df)),
        "num_test_images": int(len(test_df)),
    }

    _save_json(summary, os.path.join(management_dir, "pool_state_summary.json"))

    if save_round0_snapshot:
        round0_dir = os.path.join(rounds_dir, "round_00")
        os.makedirs(round0_dir, exist_ok=True)

        _save_csv(labeled_df, os.path.join(round0_dir, "labeled_before_training.csv"))
        _save_csv(unlabeled_df, os.path.join(round0_dir, "unlabeled_before_query.csv"))
        _save_json(summary, os.path.join(round0_dir, "round_00_summary.json"))

    print("\n===== IMAGE POOL MANAGER: INITIALIZED =====")
    print("Annotations dir:", annotations_dir)
    print("Labeled images:", len(labeled_df))
    print("Unlabeled images:", len(unlabeled_df))
    print("Val images:", len(val_df))
    print("Test images:", len(test_df))

    return {
        "labeled_df": labeled_df,
        "unlabeled_df": unlabeled_df,
        "val_df": val_df,
        "test_df": test_df,
        "paths": base_paths,
        "summary": summary,
    }


def load_pool_state(annotations_dir: str = ANNOTATIONS_DIR):
    labeled_path = os.path.join(annotations_dir, "labeled_images.csv")
    unlabeled_path = os.path.join(annotations_dir, "unlabeled_images.csv")
    val_path = os.path.join(annotations_dir, "val_images.csv")
    test_path = os.path.join(annotations_dir, "test_images.csv")

    if not os.path.exists(labeled_path):
        raise FileNotFoundError(f"Missing labeled pool file: {labeled_path}")
    if not os.path.exists(unlabeled_path):
        raise FileNotFoundError(f"Missing unlabeled pool file: {unlabeled_path}")

    labeled_df = pd.read_csv(labeled_path)
    unlabeled_df = pd.read_csv(unlabeled_path)
    val_df = pd.read_csv(val_path) if os.path.exists(val_path) else pd.DataFrame()
    test_df = pd.read_csv(test_path) if os.path.exists(test_path) else pd.DataFrame()

    return {
        "labeled_df": labeled_df,
        "unlabeled_df": unlabeled_df,
        "val_df": val_df,
        "test_df": test_df,
        "paths": {
            "labeled": labeled_path,
            "unlabeled": unlabeled_path,
            "val": val_path,
            "test": test_path,
        },
    }


def save_pool_state(
    labeled_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    val_df: Optional[pd.DataFrame] = None,
    test_df: Optional[pd.DataFrame] = None,
    annotations_dir: str = ANNOTATIONS_DIR,
):
    _save_csv(labeled_df, os.path.join(annotations_dir, "labeled_images.csv"))
    _save_csv(unlabeled_df, os.path.join(annotations_dir, "unlabeled_images.csv"))

    if val_df is not None:
        _save_csv(val_df, os.path.join(annotations_dir, "val_images.csv"))
    if test_df is not None:
        _save_csv(test_df, os.path.join(annotations_dir, "test_images.csv"))


def select_query_batch(
    unlabeled_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    round_id: int,
    k: int = QUERY_BATCH_SIZE,
    score_col: str = "query_score",
    annotations_dir: str = ANNOTATIONS_DIR,
):
    """
    Select top-k images from unlabeled pool using image-level scores.
    scored_df must contain:
    - image
    - query_score (or provided score_col)
    """
    if len(unlabeled_df) == 0:
        raise ValueError("unlabeled_df is empty.")

    _require_columns(unlabeled_df, ["image"], "unlabeled_df")
    _require_columns(scored_df, ["image", score_col], "scored_df")

    dirs = ensure_pool_manager_directories(annotations_dir)
    round_dir = os.path.join(dirs["rounds_dir"], f"round_{round_id:02d}")
    os.makedirs(round_dir, exist_ok=True)

    base_cols = list(unlabeled_df.columns)

    score_cols = [c for c in scored_df.columns if c != "image"]
    merged = unlabeled_df.merge(
        scored_df[["image"] + score_cols],
        on="image",
        how="left",
    )

    if merged[score_col].isna().all():
        raise ValueError(f"All values in '{score_col}' are NaN after merge.")

    merged = merged.sort_values(score_col, ascending=False).reset_index(drop=True)

    k = min(int(k), len(merged))
    selected_df = merged.head(k).copy()
    selected_df["selected_for_query"] = 1
    selected_df["query_round"] = int(round_id)
    selected_df["pool"] = "queried"

    remaining_unlabeled_df = merged.iloc[k:].copy()
    remaining_unlabeled_df = remaining_unlabeled_df[base_cols].copy()

    _save_csv(merged, os.path.join(round_dir, "scored_unlabeled.csv"))
    _save_csv(selected_df, os.path.join(round_dir, "selected_for_query.csv"))
    _save_csv(remaining_unlabeled_df, os.path.join(round_dir, "unlabeled_after_selection.csv"))

    summary = {
        "round_id": int(round_id),
        "selection_size": int(len(selected_df)),
        "remaining_unlabeled": int(len(remaining_unlabeled_df)),
        "score_col": score_col,
    }
    _save_json(summary, os.path.join(round_dir, "selection_summary.json"))

    print(f"\n===== IMAGE POOL MANAGER: ROUND {round_id} SELECTION =====")
    print("Selected images:", len(selected_df))
    print("Remaining unlabeled:", len(remaining_unlabeled_df))

    return {
        "selected_df": selected_df,
        "remaining_unlabeled_df": remaining_unlabeled_df,
        "round_dir": round_dir,
        "summary": summary,
    }


def commit_selected_batch(
    labeled_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    selected_df: pd.DataFrame,
    round_id: int,
    annotations_dir: str = ANNOTATIONS_DIR,
):
    """
    Move selected images from unlabeled -> labeled.
    """
    if len(selected_df) == 0:
        raise ValueError("selected_df is empty.")

    _require_columns(labeled_df, ["image"], "labeled_df")
    _require_columns(unlabeled_df, ["image"], "unlabeled_df")
    _require_columns(selected_df, ["image"], "selected_df")

    dirs = ensure_pool_manager_directories(annotations_dir)
    round_dir = os.path.join(dirs["rounds_dir"], f"round_{round_id:02d}")
    os.makedirs(round_dir, exist_ok=True)

    selected = selected_df.copy()
    selected["query_round"] = int(round_id)
    selected["selected_for_query"] = 1
    selected["pool"] = "labeled"

    remaining_unlabeled_df = unlabeled_df[
        ~unlabeled_df["image"].isin(selected["image"])
    ].copy()

    union_cols = list(dict.fromkeys(list(labeled_df.columns) + list(selected.columns)))
    labeled_aligned = _align_columns(labeled_df, union_cols)
    selected_aligned = _align_columns(selected, union_cols)

    updated_labeled_df = pd.concat(
        [labeled_aligned, selected_aligned],
        ignore_index=True,
    ).drop_duplicates(subset=["image"], keep="last")

    save_pool_state(
        labeled_df=updated_labeled_df,
        unlabeled_df=remaining_unlabeled_df,
        annotations_dir=annotations_dir,
    )

    _save_csv(selected, os.path.join(round_dir, "selected_committed.csv"))
    _save_csv(updated_labeled_df, os.path.join(round_dir, "labeled_after_update.csv"))
    _save_csv(remaining_unlabeled_df, os.path.join(round_dir, "unlabeled_after_update.csv"))

    summary = {
        "round_id": int(round_id),
        "newly_added": int(len(selected)),
        "total_labeled_after_update": int(len(updated_labeled_df)),
        "total_unlabeled_after_update": int(len(remaining_unlabeled_df)),
    }
    _save_json(summary, os.path.join(round_dir, "update_summary.json"))

    print(f"\n===== IMAGE POOL MANAGER: ROUND {round_id} UPDATE =====")
    print("Newly added images:", len(selected))
    print("Total labeled:", len(updated_labeled_df))
    print("Total unlabeled:", len(remaining_unlabeled_df))

    return {
        "updated_labeled_df": updated_labeled_df,
        "remaining_unlabeled_df": remaining_unlabeled_df,
        "selected_df": selected,
        "round_dir": round_dir,
        "summary": summary,
    }


def save_round_metrics(
    round_id: int,
    metrics: Dict,
    annotations_dir: str = ANNOTATIONS_DIR,
    filename: str = "metrics.json",
):
    dirs = ensure_pool_manager_directories(annotations_dir)
    round_dir = os.path.join(dirs["rounds_dir"], f"round_{round_id:02d}")
    os.makedirs(round_dir, exist_ok=True)

    metrics_path = os.path.join(round_dir, filename)
    _save_json(metrics, metrics_path)
    return metrics_path