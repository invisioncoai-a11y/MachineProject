import os
import pandas as pd


def _resolve_dataset_root(extracted_root: str) -> str:
    """
    بعد فك الضغط داخل /content/local_dataset
    أحيانًا الملفات تكون مباشرة داخل المجلد،
    وأحيانًا تكون داخل مجلد فرعي واحد.
    """
    train_csv_direct = os.path.join(extracted_root, "train.csv")
    sample_csv_direct = os.path.join(extracted_root, "sample_submission.csv")

    if os.path.exists(train_csv_direct) and os.path.exists(sample_csv_direct):
        return extracted_root

    for item in os.listdir(extracted_root):
        candidate = os.path.join(extracted_root, item)
        if os.path.isdir(candidate):
            train_csv = os.path.join(candidate, "train.csv")
            sample_csv = os.path.join(candidate, "sample_submission.csv")
            if os.path.exists(train_csv) and os.path.exists(sample_csv):
                return candidate

    raise FileNotFoundError(
        f"Could not find dataset files inside: {extracted_root}"
    )


def run_data_pipeline():
    project_root = os.environ.get("PROJECT_ROOT", os.getcwd())
    extracted_dataset_dir = os.environ.get("EXTRACTED_DATASET_DIR", "")
    reports_dir = os.environ.get("REPORTS_DIR", os.path.join(project_root, "DAL", "reports"))

    if not extracted_dataset_dir:
        raise ValueError("EXTRACTED_DATASET_DIR is not set.")

    os.makedirs(reports_dir, exist_ok=True)

    dataset_root = _resolve_dataset_root(extracted_dataset_dir)

    train_csv = os.path.join(dataset_root, "train.csv")
    sample_submission_csv = os.path.join(dataset_root, "sample_submission.csv")
    train_dir = os.path.join(dataset_root, "train_images")
    test_dir = os.path.join(dataset_root, "test_images")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found: {train_csv}")
    if not os.path.exists(sample_submission_csv):
        raise FileNotFoundError(f"sample_submission.csv not found: {sample_submission_csv}")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"train_images not found: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"test_images not found: {test_dir}")

    train_df = pd.read_csv(train_csv)
    sample_df = pd.read_csv(sample_submission_csv)

    if "labels" in train_df.columns:
        train_df["label_list"] = train_df["labels"].apply(lambda x: x.split())

    print("===== DATA PIPELINE =====")
    print("Project root:", project_root)
    print("Dataset root:", dataset_root)
    print("Train CSV:", train_csv)
    print("Train dir:", train_dir)
    print("Test dir:", test_dir)
    print("Reports dir:", reports_dir)
    print("Train shape:", train_df.shape)
    print("Sample submission shape:", sample_df.shape)
    print("Columns:", list(train_df.columns))

    data_bundle = {
        "project_root": project_root,
        "reports_dir": reports_dir,
        "dataset_root": dataset_root,
        "train_csv": train_csv,
        "sample_submission_csv": sample_submission_csv,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "train_df": train_df,
        "sample_df": sample_df,
    }

    return data_bundle