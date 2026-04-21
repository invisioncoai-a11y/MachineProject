import os
import json
import matplotlib.pyplot as plt
import pandas as pd

from DAL.preparation.config import ALL_LABELS, REPORTS_DIR


def _get_df(bundle: dict, key: str) -> pd.DataFrame:
    df = bundle.get(key, pd.DataFrame())
    if df is None:
        return pd.DataFrame()
    return df.copy()


def _save_empty_plot(save_path: str, title: str):
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.text(0.5, 0.5, "No data available", ha="center", va="center")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_lesion_split_sizes(lesion_bundle: dict, save_path: str):
    names = ["train_pool", "labeled", "unlabeled", "val", "test"]
    sizes = [
        len(_get_df(lesion_bundle, "train_pool_lesions_df")),
        len(_get_df(lesion_bundle, "labeled_lesions_df")),
        len(_get_df(lesion_bundle, "unlabeled_lesions_df")),
        len(_get_df(lesion_bundle, "val_lesions_df")),
        len(_get_df(lesion_bundle, "test_lesions_df")),
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(names, sizes)
    plt.title("Lesion Split Sizes")
    plt.ylabel("Number of Lesion Candidates")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_annotation_status(lesion_bundle: dict, save_path: str):
    split_keys = {
        "labeled": "labeled_lesions_df",
        "unlabeled": "unlabeled_lesions_df",
        "val": "val_lesions_df",
        "test": "test_lesions_df",
    }

    names = []
    annotated_counts = []
    unannotated_counts = []

    for name, key in split_keys.items():
        df = _get_df(lesion_bundle, key)
        names.append(name)

        if len(df) == 0 or "is_annotated" not in df.columns:
            annotated_counts.append(0)
            unannotated_counts.append(len(df))
        else:
            annotated = int(df["is_annotated"].fillna(0).astype(int).sum())
            annotated_counts.append(annotated)
            unannotated_counts.append(int(len(df) - annotated))

    x = range(len(names))
    plt.figure(figsize=(9, 5))
    plt.bar(x, annotated_counts, label="Annotated")
    plt.bar(x, unannotated_counts, bottom=annotated_counts, label="Unannotated")
    plt.xticks(list(x), names)
    plt.ylabel("Number of Lesions")
    plt.title("Annotation Status by Split")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_labeled_class_distribution(labeled_df: pd.DataFrame, save_path: str):
    if len(labeled_df) == 0:
        _save_empty_plot(save_path, "Labeled Lesion Class Distribution")
        return

    counts = []
    if "annotation_label" in labeled_df.columns:
        for cls in ALL_LABELS:
            counts.append(int((labeled_df["annotation_label"].astype(str) == cls).sum()))
    elif "primary_label" in labeled_df.columns:
        for cls in ALL_LABELS:
            counts.append(int((labeled_df["primary_label"].astype(str) == cls).sum()))
    else:
        counts = [0 for _ in ALL_LABELS]

    plt.figure(figsize=(10, 5))
    plt.bar(ALL_LABELS, counts)
    plt.title("Labeled Lesion Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_candidates_per_image(train_pool_lesions_df: pd.DataFrame, save_path: str):
    if len(train_pool_lesions_df) == 0 or "image" not in train_pool_lesions_df.columns:
        _save_empty_plot(save_path, "Lesion Candidates per Image")
        return

    counts = train_pool_lesions_df.groupby("image").size().sort_values(ascending=False)

    plt.figure(figsize=(10, 5))
    plt.hist(counts.values, bins=min(20, max(5, counts.nunique())))
    plt.title("Lesion Candidates per Image")
    plt.xlabel("Number of Lesion Candidates")
    plt.ylabel("Number of Images")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_box_area_distribution(train_pool_lesions_df: pd.DataFrame, save_path: str):
    if len(train_pool_lesions_df) == 0 or "box_area" not in train_pool_lesions_df.columns:
        _save_empty_plot(save_path, "Lesion Box Area Distribution")
        return

    box_area = train_pool_lesions_df["box_area"].dropna()
    if len(box_area) == 0:
        _save_empty_plot(save_path, "Lesion Box Area Distribution")
        return

    plt.figure(figsize=(10, 5))
    plt.hist(box_area.values, bins=30)
    plt.title("Lesion Box Area Distribution")
    plt.xlabel("Box Area (pixels)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_top_images_by_lesion_count(train_pool_lesions_df: pd.DataFrame, save_path: str, top_k: int = 15):
    if len(train_pool_lesions_df) == 0 or "image" not in train_pool_lesions_df.columns:
        _save_empty_plot(save_path, "Top Images by Lesion Count")
        return

    counts = train_pool_lesions_df.groupby("image").size().sort_values(ascending=False).head(top_k)

    plt.figure(figsize=(12, 5))
    plt.bar(counts.index.astype(str), counts.values)
    plt.title(f"Top {top_k} Images by Lesion Candidate Count")
    plt.xlabel("Image")
    plt.ylabel("Number of Lesion Candidates")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _save_lesion_summary_json(lesion_bundle: dict, save_path: str):
    train_pool_df = _get_df(lesion_bundle, "train_pool_lesions_df")
    labeled_df = _get_df(lesion_bundle, "labeled_lesions_df")
    unlabeled_df = _get_df(lesion_bundle, "unlabeled_lesions_df")
    val_df = _get_df(lesion_bundle, "val_lesions_df")
    test_df = _get_df(lesion_bundle, "test_lesions_df")

    def split_summary(df: pd.DataFrame):
        summary = {
            "num_lesions": int(len(df)),
            "num_images": int(df["image"].nunique()) if "image" in df.columns else 0,
        }

        if "is_annotated" in df.columns:
            annotated = int(df["is_annotated"].fillna(0).astype(int).sum())
            summary["annotated"] = annotated
            summary["unannotated"] = int(len(df) - annotated)

        if "box_area" in df.columns and len(df) > 0:
            summary["avg_box_area"] = float(df["box_area"].dropna().mean()) if df["box_area"].dropna().shape[0] > 0 else 0.0

        return summary

    summary = {
        "all_labels": list(ALL_LABELS),
        "train_pool": split_summary(train_pool_df),
        "labeled": split_summary(labeled_df),
        "unlabeled": split_summary(unlabeled_df),
        "val": split_summary(val_df),
        "test": split_summary(test_df),
    }

    if len(labeled_df) > 0 and "annotation_label" in labeled_df.columns:
        class_counts = {
            cls: int((labeled_df["annotation_label"].astype(str) == cls).sum())
            for cls in ALL_LABELS
        }
        summary["labeled_class_distribution"] = class_counts

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_lesion_eda_pipeline(lesion_bundle: dict, reports_dir: str = REPORTS_DIR):
    lesion_eda_dir = os.path.join(reports_dir, "eda", "lesions")
    os.makedirs(lesion_eda_dir, exist_ok=True)

    train_pool_lesions_df = _get_df(lesion_bundle, "train_pool_lesions_df")
    labeled_lesions_df = _get_df(lesion_bundle, "labeled_lesions_df")

    _plot_lesion_split_sizes(
        lesion_bundle=lesion_bundle,
        save_path=os.path.join(lesion_eda_dir, "lesion_split_sizes.png"),
    )

    _plot_annotation_status(
        lesion_bundle=lesion_bundle,
        save_path=os.path.join(lesion_eda_dir, "annotation_status.png"),
    )

    _plot_labeled_class_distribution(
        labeled_df=labeled_lesions_df,
        save_path=os.path.join(lesion_eda_dir, "labeled_lesion_class_distribution.png"),
    )

    _plot_candidates_per_image(
        train_pool_lesions_df=train_pool_lesions_df,
        save_path=os.path.join(lesion_eda_dir, "lesion_candidates_per_image.png"),
    )

    _plot_box_area_distribution(
        train_pool_lesions_df=train_pool_lesions_df,
        save_path=os.path.join(lesion_eda_dir, "lesion_box_area_distribution.png"),
    )

    _plot_top_images_by_lesion_count(
        train_pool_lesions_df=train_pool_lesions_df,
        save_path=os.path.join(lesion_eda_dir, "top_images_by_lesion_count.png"),
    )

    _save_lesion_summary_json(
        lesion_bundle=lesion_bundle,
        save_path=os.path.join(lesion_eda_dir, "lesion_summary.json"),
    )

    print("\n===== LESION EDA PIPELINE =====")
    print("Lesion EDA outputs saved to:", lesion_eda_dir)