import os
import json
import pandas as pd

from DAL.preparation.config import (
    ALL_LABELS,
    REPORTS_DIR,
    SEED,
)

from DAL.preparation.paths import (
    ensure_report_directories,
    get_dataset_paths,
)

from DAL.preparation.split_data import (
    prepare_train_dataframe,
    fit_label_binarizer,
    create_data_splits,
    save_split_csvs,
)

from DAL.preparation.lesion_manifest import (
    build_lesion_manifest_bundle,
)

from DAL.preparation.pool_manager import (
    initialize_pool_state_from_manifest_bundle,
)

from DAL.eda.explore_dataset import run_eda_pipeline
from DAL.eda.explore_lesions import run_lesion_eda_pipeline


def _shape_or_empty(df):
    if df is None:
        return [0, 0]
    return list(df.shape)


def _save_metadata(
    data_bundle: dict,
    lesion_bundle: dict,
    pool_state: dict,
    reports_dir: str,
):
    meta_dir = os.path.join(reports_dir, "metadata")
    os.makedirs(meta_dir, exist_ok=True)

    meta_path = os.path.join(meta_dir, "pipeline_metadata.json")

    serializable = {
        "project_root": data_bundle["project_root"],
        "dataset_root": data_bundle["dataset_root"],
        "train_csv": data_bundle["train_csv"],
        "sample_submission_csv": data_bundle["sample_submission_csv"],
        "train_dir": data_bundle["train_dir"],
        "test_dir": data_bundle["test_dir"],
        "reports_dir": data_bundle["reports_dir"],
        "all_labels": data_bundle["all_labels"],

        "image_level_shapes": {
            "train_df": _shape_or_empty(data_bundle.get("train_df")),
            "train_pool_df": _shape_or_empty(data_bundle.get("train_pool_df")),
            "val_df": _shape_or_empty(data_bundle.get("val_df")),
            "test_df": _shape_or_empty(data_bundle.get("test_df")),
            "initial_labeled_df": _shape_or_empty(data_bundle.get("initial_labeled_df")),
            "unlabeled_pool_df": _shape_or_empty(data_bundle.get("unlabeled_pool_df")),
            "sample_df": _shape_or_empty(data_bundle.get("sample_df")),
        },

        "lesion_level_shapes": {
            "train_pool_lesions_df": _shape_or_empty(lesion_bundle.get("train_pool_lesions_df")),
            "labeled_lesions_df": _shape_or_empty(lesion_bundle.get("labeled_lesions_df")),
            "unlabeled_lesions_df": _shape_or_empty(lesion_bundle.get("unlabeled_lesions_df")),
            "val_lesions_df": _shape_or_empty(lesion_bundle.get("val_lesions_df")),
            "test_lesions_df": _shape_or_empty(lesion_bundle.get("test_lesions_df")),
        },

        "lesion_csv_paths": lesion_bundle.get("csv_paths", {}),
        "lesion_summary": lesion_bundle.get("summary", {}),

        "pool_state_summary": pool_state.get("summary", {}),
        "pool_paths": pool_state.get("paths", {}),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    return meta_path


def run_data_pipeline():
    """
    Full data-stage pipeline:

    1) Ensure report directories
    2) Resolve dataset paths
    3) Read train/sample CSVs
    4) Build image-level dataframe
    5) Fit label binarizer
    6) Create image-level splits
    7) Save image-level split CSVs
    8) Run image-level EDA
    9) Build lesion manifests from image-level splits
    10) Run lesion-level EDA
    11) Initialize labeled/unlabeled lesion pool state
    12) Save combined metadata

    Returns a single bundle that RUN.py can consume later.
    """
    ensure_report_directories()

    # -------------------------
    # Resolve dataset paths
    # -------------------------
    dataset_paths = get_dataset_paths()
    dataset_root = dataset_paths["dataset_root"]
    train_csv = dataset_paths["train_csv"]
    sample_submission_csv = dataset_paths["sample_submission_csv"]
    train_dir = dataset_paths["train_dir"]
    test_dir = dataset_paths["test_dir"]

    # -------------------------
    # Read CSVs
    # -------------------------
    train_df_raw = pd.read_csv(train_csv)
    sample_df = pd.read_csv(sample_submission_csv)

    # -------------------------
    # Image-level preparation
    # -------------------------
    train_df = prepare_train_dataframe(train_df_raw)
    mlb, y_all = fit_label_binarizer(train_df, ALL_LABELS)

    split_bundle = create_data_splits(
        train_df=train_df,
        y_all=y_all,
        seed=SEED,
    )

    save_split_csvs(split_bundle, REPORTS_DIR)

    data_bundle = {
        "project_root": os.environ.get("PROJECT_ROOT", os.getcwd()),
        "dataset_root": dataset_root,
        "train_csv": train_csv,
        "sample_submission_csv": sample_submission_csv,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "reports_dir": REPORTS_DIR,
        "all_labels": list(mlb.classes_),
        "mlb": mlb,
        "sample_df": sample_df,
        **split_bundle,
    }

    # -------------------------
    # Image-level EDA
    # -------------------------
    run_eda_pipeline(data_bundle)

    # -------------------------
    # Lesion-level manifest generation
    # -------------------------
    lesion_bundle = build_lesion_manifest_bundle(
        data_bundle=data_bundle,
        reports_dir=REPORTS_DIR,
    )

    # -------------------------
    # Lesion-level EDA
    # -------------------------
    run_lesion_eda_pipeline(
        lesion_bundle=lesion_bundle,
        reports_dir=REPORTS_DIR,
    )

    # -------------------------
    # Initialize lesion pool state
    # -------------------------
    pool_state = initialize_pool_state_from_manifest_bundle(
        manifest_bundle=lesion_bundle,
    )

    # -------------------------
    # Save combined metadata
    # -------------------------
    metadata_path = _save_metadata(
        data_bundle=data_bundle,
        lesion_bundle=lesion_bundle,
        pool_state=pool_state,
        reports_dir=REPORTS_DIR,
    )

    # -------------------------
    # Final combined output
    # -------------------------
    pipeline_bundle = {
        "project_root": data_bundle["project_root"],
        "dataset_root": data_bundle["dataset_root"],
        "train_csv": data_bundle["train_csv"],
        "sample_submission_csv": data_bundle["sample_submission_csv"],
        "train_dir": data_bundle["train_dir"],
        "test_dir": data_bundle["test_dir"],
        "reports_dir": data_bundle["reports_dir"],
        "all_labels": data_bundle["all_labels"],
        "mlb": data_bundle["mlb"],
        "sample_df": data_bundle["sample_df"],

        # image-level
        "train_df": data_bundle["train_df"],
        "train_pool_df": data_bundle["train_pool_df"],
        "val_df": data_bundle["val_df"],
        "test_df": data_bundle["test_df"],
        "initial_labeled_df": data_bundle["initial_labeled_df"],
        "unlabeled_pool_df": data_bundle["unlabeled_pool_df"],
        "y_all": data_bundle["y_all"],

        # lesion-level bundle
        "lesion_bundle": lesion_bundle,
        "train_pool_lesions_df": lesion_bundle["train_pool_lesions_df"],
        "labeled_lesions_df": lesion_bundle["labeled_lesions_df"],
        "unlabeled_lesions_df": lesion_bundle["unlabeled_lesions_df"],
        "val_lesions_df": lesion_bundle["val_lesions_df"],
        "test_lesions_df": lesion_bundle["test_lesions_df"],
        "lesion_csv_paths": lesion_bundle["csv_paths"],
        "lesion_summary": lesion_bundle["summary"],
        "lesion_summary_path": lesion_bundle["summary_path"],

        # pool state
        "pool_state": pool_state,
        "pool_labeled_df": pool_state["labeled_df"],
        "pool_unlabeled_df": pool_state["unlabeled_df"],
        "pool_val_df": pool_state["val_df"],
        "pool_test_df": pool_state["test_df"],
        "pool_paths": pool_state["paths"],

        # metadata
        "metadata_path": metadata_path,
    }

    print("\n===== FULL DATA PIPELINE =====")
    print("Project root:", pipeline_bundle["project_root"])
    print("Dataset root:", pipeline_bundle["dataset_root"])
    print("Train CSV:", pipeline_bundle["train_csv"])
    print("Train dir:", pipeline_bundle["train_dir"])
    print("Test dir:", pipeline_bundle["test_dir"])
    print("Reports dir:", pipeline_bundle["reports_dir"])
    print("Metadata path:", pipeline_bundle["metadata_path"])

    print("\n--- Image-level shapes ---")
    print("Train:", pipeline_bundle["train_df"].shape)
    print("Train pool:", pipeline_bundle["train_pool_df"].shape)
    print("Validation:", pipeline_bundle["val_df"].shape)
    print("Test:", pipeline_bundle["test_df"].shape)
    print("Initial labeled:", pipeline_bundle["initial_labeled_df"].shape)
    print("Unlabeled pool:", pipeline_bundle["unlabeled_pool_df"].shape)

    print("\n--- Lesion-level shapes ---")
    print("Train pool lesions:", pipeline_bundle["train_pool_lesions_df"].shape)
    print("Labeled lesions:", pipeline_bundle["labeled_lesions_df"].shape)
    print("Unlabeled lesions:", pipeline_bundle["unlabeled_lesions_df"].shape)
    print("Val lesions:", pipeline_bundle["val_lesions_df"].shape)
    print("Test lesions:", pipeline_bundle["test_lesions_df"].shape)

    print("\n--- Pool state ---")
    print("Pool labeled:", pipeline_bundle["pool_labeled_df"].shape)
    print("Pool unlabeled:", pipeline_bundle["pool_unlabeled_df"].shape)

    return pipeline_bundle