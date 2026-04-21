import os
import json
import matplotlib.pyplot as plt


def _plot_class_distribution(train_df, all_labels, save_path):
    counts = []
    for cls in all_labels:
        col = f"y_{cls}"
        if col in train_df.columns:
            counts.append(int(train_df[col].sum()))
        else:
            counts.append(0)

    plt.figure(figsize=(10, 5))
    plt.bar(all_labels, counts)
    plt.title("Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_label_combo_distribution(train_df, save_path, top_k=12):
    combo_counts = train_df["label_combo"].value_counts().head(top_k)

    plt.figure(figsize=(12, 5))
    plt.bar(combo_counts.index.astype(str), combo_counts.values)
    plt.title(f"Top {top_k} Label Combinations")
    plt.xlabel("Label Combination")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _plot_split_sizes(data_bundle, save_path):
    names = ["train_pool", "validation", "test", "initial_labeled", "unlabeled_pool"]
    sizes = [
        len(data_bundle["train_pool_df"]),
        len(data_bundle["val_df"]),
        len(data_bundle["test_df"]),
        len(data_bundle["initial_labeled_df"]),
        len(data_bundle["unlabeled_pool_df"]),
    ]

    plt.figure(figsize=(9, 5))
    plt.bar(names, sizes)
    plt.title("Split Sizes")
    plt.ylabel("Number of Samples")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def _save_summary_json(data_bundle, save_path):
    train_df = data_bundle["train_df"]

    summary = {
        "num_images": int(len(train_df)),
        "num_classes": int(len(data_bundle["all_labels"])),
        "all_labels": list(data_bundle["all_labels"]),
        "num_multilabel_images": int(train_df["is_multilabel"].sum()),
        "num_singlelabel_images": int((~train_df["is_multilabel"]).sum()),
        "num_unique_label_combinations": int(train_df["label_combo"].nunique()),
        "split_sizes": {
            "train_pool": int(len(data_bundle["train_pool_df"])),
            "validation": int(len(data_bundle["val_df"])),
            "test": int(len(data_bundle["test_df"])),
            "initial_labeled": int(len(data_bundle["initial_labeled_df"])),
            "unlabeled_pool": int(len(data_bundle["unlabeled_pool_df"])),
        }
    }

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_eda_pipeline(data_bundle: dict):
    eda_dir = os.path.join(data_bundle["reports_dir"], "eda")
    os.makedirs(eda_dir, exist_ok=True)

    _plot_class_distribution(
        train_df=data_bundle["train_df"],
        all_labels=data_bundle["all_labels"],
        save_path=os.path.join(eda_dir, "class_distribution.png"),
    )

    _plot_label_combo_distribution(
        train_df=data_bundle["train_df"],
        save_path=os.path.join(eda_dir, "label_combo_distribution.png"),
    )

    _plot_split_sizes(
        data_bundle=data_bundle,
        save_path=os.path.join(eda_dir, "split_sizes.png"),
    )

    _save_summary_json(
        data_bundle=data_bundle,
        save_path=os.path.join(eda_dir, "dataset_summary.json"),
    )

    print("\n===== EDA PIPELINE =====")
    print("EDA outputs saved to:", eda_dir)