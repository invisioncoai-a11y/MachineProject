import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer

from DAL.preparation.config import (
    TEST_RATIO,
    VAL_RATIO,
    INITIAL_LABEL_RATIO,
    ALL_LABELS,
)


def prepare_train_dataframe(train_df_raw: pd.DataFrame) -> pd.DataFrame:
    df = train_df_raw.copy()

    if "image" not in df.columns or "labels" not in df.columns:
        raise ValueError("train.csv must contain 'image' and 'labels' columns.")

    df["label_list"] = df["labels"].apply(lambda x: str(x).split())
    df["num_labels"] = df["label_list"].apply(len)
    df["label_combo"] = df["label_list"].apply(lambda x: " | ".join(sorted(x)))
    df["primary_label"] = df["label_list"].apply(
        lambda x: x[0] if len(x) > 0 else "unknown"
    )
    df["is_multilabel"] = df["num_labels"] > 1

    return df


def fit_label_binarizer(train_df: pd.DataFrame, all_labels: list[str]):
    mlb = MultiLabelBinarizer(classes=all_labels)
    y_all = mlb.fit_transform(train_df["label_list"]).astype("float32")
    return mlb, y_all


def _safe_stratify_series(series: pd.Series):
    value_counts = series.value_counts()
    if (value_counts >= 2).all():
        return series
    return None


def _attach_encoded_columns(
    df: pd.DataFrame,
    y: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    out = df.copy()
    for i, cls in enumerate(class_names):
        out[f"y_{cls}"] = y[:, i] if len(y) > 0 else np.array([], dtype=np.float32)
    return out


def create_data_splits(train_df: pd.DataFrame, y_all: np.ndarray, seed: int = 42):
    class_names = list(ALL_LABELS)

    stratify_full = _safe_stratify_series(train_df["label_combo"])

    idx_all = np.arange(len(train_df))
    idx_train_pool, idx_test = train_test_split(
        idx_all,
        test_size=TEST_RATIO,
        random_state=seed,
        shuffle=True,
        stratify=stratify_full if stratify_full is not None else None,
    )

    train_pool_df = train_df.iloc[idx_train_pool].reset_index(drop=True)
    test_df = train_df.iloc[idx_test].reset_index(drop=True)

    y_train_pool = y_all[idx_train_pool]
    y_test = y_all[idx_test]

    relative_val_ratio = VAL_RATIO / (1.0 - TEST_RATIO)
    stratify_train_pool = _safe_stratify_series(train_pool_df["label_combo"])

    idx_pool = np.arange(len(train_pool_df))
    idx_train_final, idx_val = train_test_split(
        idx_pool,
        test_size=relative_val_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_train_pool if stratify_train_pool is not None else None,
    )

    final_train_df = train_pool_df.iloc[idx_train_final].reset_index(drop=True)
    val_df = train_pool_df.iloc[idx_val].reset_index(drop=True)

    y_final_train = y_train_pool[idx_train_final]
    y_val = y_train_pool[idx_val]

    stratify_final_train = _safe_stratify_series(final_train_df["label_combo"])
    idx_final = np.arange(len(final_train_df))

    initial_label_count = max(1, int(round(INITIAL_LABEL_RATIO * len(final_train_df))))

    if initial_label_count >= len(final_train_df):
        idx_initial_labeled = idx_final
        idx_unlabeled = np.array([], dtype=int)
    else:
        idx_initial_labeled, idx_unlabeled = train_test_split(
            idx_final,
            train_size=initial_label_count,
            random_state=seed,
            shuffle=True,
            stratify=stratify_final_train if stratify_final_train is not None else None,
        )

    initial_labeled_df = final_train_df.iloc[idx_initial_labeled].reset_index(drop=True)
    unlabeled_pool_df = final_train_df.iloc[idx_unlabeled].reset_index(drop=True)

    y_initial_labeled = y_final_train[idx_initial_labeled]
    y_unlabeled_pool = y_final_train[idx_unlabeled]

    train_df_encoded = _attach_encoded_columns(train_df, y_all, class_names)
    train_pool_df_encoded = _attach_encoded_columns(train_pool_df, y_train_pool, class_names)
    val_df_encoded = _attach_encoded_columns(val_df, y_val, class_names)
    test_df_encoded = _attach_encoded_columns(test_df, y_test, class_names)
    initial_labeled_df_encoded = _attach_encoded_columns(
        initial_labeled_df, y_initial_labeled, class_names
    )
    unlabeled_pool_df_encoded = _attach_encoded_columns(
        unlabeled_pool_df, y_unlabeled_pool, class_names
    )

    return {
        "train_df": train_df_encoded,
        "train_pool_df": train_pool_df_encoded,
        "val_df": val_df_encoded,
        "test_df": test_df_encoded,
        "initial_labeled_df": initial_labeled_df_encoded,
        "unlabeled_pool_df": unlabeled_pool_df_encoded,
        "y_all": y_all,
    }


def save_split_csvs(split_bundle: dict, reports_dir: str):
    split_dir = os.path.join(reports_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    split_bundle["train_df"].to_csv(os.path.join(split_dir, "train_full.csv"), index=False)
    split_bundle["train_pool_df"].to_csv(os.path.join(split_dir, "train_pool.csv"), index=False)
    split_bundle["val_df"].to_csv(os.path.join(split_dir, "validation.csv"), index=False)
    split_bundle["test_df"].to_csv(os.path.join(split_dir, "test.csv"), index=False)
    split_bundle["initial_labeled_df"].to_csv(os.path.join(split_dir, "initial_labeled.csv"), index=False)
    split_bundle["unlabeled_pool_df"].to_csv(os.path.join(split_dir, "unlabeled_pool.csv"), index=False)