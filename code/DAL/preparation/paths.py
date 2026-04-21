import os

from DAL.preparation.config import (
    EXTRACTED_DATASET_DIR,
    REPORTS_DIR,
)


def ensure_report_directories():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPORTS_DIR, "splits"), exist_ok=True)
    os.makedirs(os.path.join(REPORTS_DIR, "eda"), exist_ok=True)
    os.makedirs(os.path.join(REPORTS_DIR, "metadata"), exist_ok=True)


def _resolve_dataset_root(extracted_root: str) -> str:
    if not extracted_root:
        raise ValueError("EXTRACTED_DATASET_DIR is empty.")

    train_csv_direct = os.path.join(extracted_root, "train.csv")
    sample_csv_direct = os.path.join(extracted_root, "sample_submission.csv")

    if os.path.exists(train_csv_direct) and os.path.exists(sample_csv_direct):
        return extracted_root

    if not os.path.exists(extracted_root):
        raise FileNotFoundError(f"Extracted dataset root does not exist: {extracted_root}")

    for item in os.listdir(extracted_root):
        candidate = os.path.join(extracted_root, item)
        if not os.path.isdir(candidate):
            continue

        train_csv = os.path.join(candidate, "train.csv")
        sample_csv = os.path.join(candidate, "sample_submission.csv")
        if os.path.exists(train_csv) and os.path.exists(sample_csv):
            return candidate

    raise FileNotFoundError(
        f"Could not find dataset files inside: {extracted_root}"
    )


def get_dataset_paths():
    dataset_root = _resolve_dataset_root(EXTRACTED_DATASET_DIR)

    train_csv = os.path.join(dataset_root, "train.csv")
    sample_submission_csv = os.path.join(dataset_root, "sample_submission.csv")
    train_dir = os.path.join(dataset_root, "train_images")
    test_dir = os.path.join(dataset_root, "test_images")

    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"Missing train.csv: {train_csv}")
    if not os.path.exists(sample_submission_csv):
        raise FileNotFoundError(f"Missing sample_submission.csv: {sample_submission_csv}")
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"Missing train_images: {train_dir}")
    if not os.path.exists(test_dir):
        raise FileNotFoundError(f"Missing test_images: {test_dir}")

    return {
        "dataset_root": dataset_root,
        "train_csv": train_csv,
        "sample_submission_csv": sample_submission_csv,
        "train_dir": train_dir,
        "test_dir": test_dir,
    }