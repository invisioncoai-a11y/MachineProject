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
from DAL.eda.explore_dataset import run_eda_pipeline


def _save_metadata(data_bundle: dict, reports_dir: str):
    meta_path = os.path.join(reports_dir, "dataset_metadata.json")

    serializable = {
        "project_root": data_bundle["project_root"],
        "dataset_root": data_bundle["dataset_root"],
        "train_csv": data_bundle["train_csv"],
        "sample_submission_csv": data_bundle["sample_submission_csv"],
        "train_dir": data_bundle["train_dir"],
        "test_dir": data_bundle["test_dir"],
        "reports_dir": data_bundle["reports_dir"],
        "all_labels": data_bundle["all_labels"],
        "train_shape": list(data_bundle["train_df"].shape),
        "sample_submission_shape": list(data_bundle["sample_df"].shape),
        "split_shapes": {
            "full_train_df": list(data_bundle["train_df"].shape),
            "train_pool_df": list(data_bundle["train_pool_df"].shape),
            "val_df": list(data_bundle["val_df"].shape),
            "test_df": list(data_bundle["test_df"].shape),
            "initial_labeled_df": list(data_bundle["initial_labeled_df"].shape),
            "unlabeled_pool_df": list(data_bundle["unlabeled_pool_df"].shape),
        },
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)


def run_data_pipeline():
    ensure_report_directories()

    dataset_paths = get_dataset_paths()
    train_csv = dataset_paths["train_csv"]
    sample_submission_csv = dataset_paths["sample_submission_csv"]
    train_dir = dataset_paths["train_dir"]
    test_dir = dataset_paths["test_dir"]
    dataset_root = dataset_paths["dataset_root"]

    train_df_raw = pd.read_csv(train_csv)
    sample_df = pd.read_csv(sample_submission_csv)

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

    _save_metadata(data_bundle, REPORTS_DIR)
    run_eda_pipeline(data_bundle)

    print("\n===== DATA PIPELINE =====")
    print("Project root:", data_bundle["project_root"])
    print("Dataset root:", dataset_root)
    print("Train CSV:", train_csv)
    print("Train dir:", train_dir)
    print("Test dir:", test_dir)
    print("Reports dir:", REPORTS_DIR)
    print("Train shape:", data_bundle["train_df"].shape)
    print("Sample submission shape:", data_bundle["sample_df"].shape)
    print("Columns:", list(data_bundle["train_df"].columns))
    print("Train pool:", data_bundle["train_pool_df"].shape)
    print("Validation:", data_bundle["val_df"].shape)
    print("Test:", data_bundle["test_df"].shape)
    print("Initial labeled:", data_bundle["initial_labeled_df"].shape)
    print("Unlabeled pool:", data_bundle["unlabeled_pool_df"].shape)

    return data_bundle