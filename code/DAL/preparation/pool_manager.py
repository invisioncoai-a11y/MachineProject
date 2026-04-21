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


def initialize_pool_state_from_manifest_bundle(
    manifest_bundle: Dict[str, pd.DataFrame],
    annotations_dir: str = ANNOTATIONS_DIR,
    save_round0_snapshot: bool = True,
):
    """
    Save the initial pool state from lesion manifests.

    Expected keys in manifest_bundle:
    - labeled_lesions_df
    - unlabeled_lesions_df
    - val_lesions_df
    - test_lesions_df
    """
    dirs = ensure_pool_manager_directories(annotations_dir)
    rounds_dir = dirs["rounds_dir"]
    management_dir = dirs["management_dir"]

    labeled_df = manifest_bundle.get("labeled_lesions_df", pd.DataFrame()).copy()
    unlabeled_df = manifest_bundle.get("unlabeled_lesions_df", pd.DataFrame()).copy()
    val_df = manifest_bundle.get("val_lesions_df", pd.DataFrame()).copy()
    test_df = manifest_bundle.get("test_lesions_df", pd.DataFrame()).copy()

    if len(labeled_df) > 0:
        _require_columns(labeled_df, ["lesion_id"], "labeled_lesions_df")
    if len(unlabeled_df) > 0:
        _require_columns(unlabeled_df, ["lesion_id"], "unlabeled_lesions_df")

    base_paths = {
        "labeled": os.path.join(annotations_dir, "labeled_lesions.csv"),
        "unlabeled": os.path.join(annotations_dir, "unlabeled_lesions.csv"),
        "val": os.path.join(annotations_dir, "val_lesions.csv"),
        "test": os.path.join(annotations_dir, "test_lesions.csv"),
    }

    _save_csv(labeled_df, base_paths["labeled"])
    _save_csv(unlabeled_df, base_paths["unlabeled"])
    _save_csv(val_df, base_paths["val"])
    _save_csv(test_df, base_paths["test"])

    summary = {
        "num_labeled_lesions": int(len(labeled_df)),
        "num_unlabeled_lesions": int(len(unlabeled_df)),
        "num_val_lesions": int(len(val_df)),
        "num_test_lesions": int(len(test_df)),
    }

    _save_json(summary, os.path.join(management_dir, "pool_state_summary.json"))

    if save_round0_snapshot:
        round0_dir = os.path.join(rounds_dir, "round_00")
        os.makedirs(round0_dir, exist_ok=True)

        _save_csv(labeled_df, os.path.join(round0_dir, "labeled_before_training.csv"))
        _save_csv(unlabeled_df, os.path.join(round0_dir, "unlabeled_before_query.csv"))
        _save_json(summary, os.path.join(round0_dir, "round_00_summary.json"))

    print("\n===== POOL MANAGER: INITIALIZED =====")
    print("Annotations dir:", annotations_dir)
    print("Labeled lesions:", len(labeled_df))
    print("Unlabeled lesions:", len(unlabeled_df))
    print("Val lesions:", len(val_df))
    print("Test lesions:", len(test_df))

    return {
        "labeled_df": labeled_df,
        "unlabeled_df": unlabeled_df,
        "val_df": val_df,
        "test_df": test_df,
        "paths": base_paths,
        "summary": summary,
    }


def load_pool_state(annotations_dir: str = ANNOTATIONS_DIR):
    labeled_path = os.path.join(annotations_dir, "labeled_lesions.csv")
    unlabeled_path = os.path.join(annotations_dir, "unlabeled_lesions.csv")
    val_path = os.path.join(annotations_dir, "val_lesions.csv")
    test_path = os.path.join(annotations_dir, "test_lesions.csv")

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
    _save_csv(labeled_df, os.path.join(annotations_dir, "labeled_lesions.csv"))
    _save_csv(unlabeled_df, os.path.join(annotations_dir, "unlabeled_lesions.csv"))

    if val_df is not None:
        _save_csv(val_df, os.path.join(annotations_dir, "val_lesions.csv"))
    if test_df is not None:
        _save_csv(test_df, os.path.join(annotations_dir, "test_lesions.csv"))


def select_query_batch(
    unlabeled_df: pd.DataFrame,
    scored_df: pd.DataFrame,
    round_id: int,
    k: int = QUERY_BATCH_SIZE,
    score_col: str = "query_score",
    annotations_dir: str = ANNOTATIONS_DIR,
):
    """
    Select top-k lesions from unlabeled pool using scores.

    Required:
    - unlabeled_df contains lesion_id
    - scored_df contains lesion_id and score_col

    Returns:
    {
        "selected_df": ...,
        "remaining_unlabeled_df": ...,
        "round_dir": ...
    }
    """
    if len(unlabeled_df) == 0:
        raise ValueError("unlabeled_df is empty.")

    _require_columns(unlabeled_df, ["lesion_id"], "unlabeled_df")
    _require_columns(scored_df, ["lesion_id", score_col], "scored_df")

    dirs = ensure_pool_manager_directories(annotations_dir)
    round_dir = os.path.join(dirs["rounds_dir"], f"round_{round_id:02d}")
    os.makedirs(round_dir, exist_ok=True)

    base_cols = list(unlabeled_df.columns)

    score_cols = [c for c in scored_df.columns if c != "lesion_id"]
    merged = unlabeled_df.merge(
        scored_df[["lesion_id"] + score_cols],
        on="lesion_id",
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

    print(f"\n===== POOL MANAGER: ROUND {round_id} SELECTION =====")
    print("Selected:", len(selected_df))
    print("Remaining unlabeled:", len(remaining_unlabeled_df))

    return {
        "selected_df": selected_df,
        "remaining_unlabeled_df": remaining_unlabeled_df,
        "round_dir": round_dir,
    }


def build_simulated_annotations_from_primary_label(
    selected_df: pd.DataFrame,
    round_id: int,
):
    """
    Useful for early experiments when you do not yet have true lesion-level annotations.
    Uses primary_label as a weak/simulated lesion annotation.
    """
    if len(selected_df) == 0:
        return selected_df.copy()

    _require_columns(selected_df, ["lesion_id", "primary_label"], "selected_df")

    out = selected_df.copy()
    out["annotation_label"] = out["primary_label"].astype(str)
    out["is_annotated"] = 1
    out["annotation_source"] = "simulated_from_primary_label"
    out["query_round"] = int(round_id)
    out["selected_for_query"] = 1
    out["pool"] = "labeled"
    return out


def commit_annotated_batch(
    labeled_df: pd.DataFrame,
    unlabeled_df: pd.DataFrame,
    annotated_selected_df: pd.DataFrame,
    round_id: int,
    annotations_dir: str = ANNOTATIONS_DIR,
):
    """
    Move newly annotated lesions from unlabeled -> labeled and save round artifacts.

    annotated_selected_df must contain:
    - lesion_id
    - annotation_label
    """
    if len(annotated_selected_df) == 0:
        raise ValueError("annotated_selected_df is empty.")

    _require_columns(labeled_df, ["lesion_id"], "labeled_df")
    _require_columns(unlabeled_df, ["lesion_id"], "unlabeled_df")
    _require_columns(
        annotated_selected_df,
        ["lesion_id", "annotation_label"],
        "annotated_selected_df",
    )

    dirs = ensure_pool_manager_directories(annotations_dir)
    round_dir = os.path.join(dirs["rounds_dir"], f"round_{round_id:02d}")
    os.makedirs(round_dir, exist_ok=True)

    annotated = annotated_selected_df.copy()

    annotated = annotated[
        annotated["annotation_label"].notna()
        & (annotated["annotation_label"].astype(str).str.strip() != "")
    ].copy()

    if len(annotated) == 0:
        raise ValueError("No valid annotation_label values found in annotated_selected_df.")

    annotated["is_annotated"] = 1
    annotated["query_round"] = int(round_id)
    annotated["selected_for_query"] = 1
    annotated["pool"] = "labeled"

    remaining_unlabeled_df = unlabeled_df[
        ~unlabeled_df["lesion_id"].isin(annotated["lesion_id"])
    ].copy()

    union_cols = list(dict.fromkeys(list(labeled_df.columns) + list(annotated.columns)))
    labeled_aligned = _align_columns(labeled_df, union_cols)
    annotated_aligned = _align_columns(annotated, union_cols)

    updated_labeled_df = pd.concat(
        [labeled_aligned, annotated_aligned],
        ignore_index=True,
    ).drop_duplicates(subset=["lesion_id"], keep="last")

    save_pool_state(
        labeled_df=updated_labeled_df,
        unlabeled_df=remaining_unlabeled_df,
        annotations_dir=annotations_dir,
    )

    _save_csv(annotated, os.path.join(round_dir, "annotated_selected.csv"))
    _save_csv(updated_labeled_df, os.path.join(round_dir, "labeled_after_update.csv"))
    _save_csv(remaining_unlabeled_df, os.path.join(round_dir, "unlabeled_after_update.csv"))

    summary = {
        "round_id": int(round_id),
        "newly_annotated": int(len(annotated)),
        "total_labeled_after_update": int(len(updated_labeled_df)),
        "total_unlabeled_after_update": int(len(remaining_unlabeled_df)),
    }
    _save_json(summary, os.path.join(round_dir, "update_summary.json"))

    print(f"\n===== POOL MANAGER: ROUND {round_id} UPDATE =====")
    print("Newly annotated:", len(annotated))
    print("Total labeled:", len(updated_labeled_df))
    print("Total unlabeled:", len(remaining_unlabeled_df))

    return {
        "updated_labeled_df": updated_labeled_df,
        "remaining_unlabeled_df": remaining_unlabeled_df,
        "annotated_df": annotated,
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