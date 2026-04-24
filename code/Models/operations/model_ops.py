import os
import json
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm.auto import tqdm

from DAL.preparation.config import (
    BATCH_SIZE,
    NUM_WORKERS,
    NUM_EPOCHS,
    LEARNING_RATE,
    WEIGHT_DECAY,
    THRESHOLD,
)
from DAL.preparation.data_loader import create_image_patch_dataloaders
from Models.classes import HybridPlantDiseaseModel


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_model(
    num_classes: int,
    embedding_dim: int = 256,
    pretrained: bool = True,
    freeze_backbone: bool = False,
):
    model = HybridPlantDiseaseModel(
        num_classes=num_classes,
        embedding_dim=embedding_dim,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    )
    return model


def build_dataloaders(pipeline_bundle: dict):
    loaders = create_image_patch_dataloaders(
        pipeline_bundle=pipeline_bundle,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )
    return loaders


def build_optimizer(model: nn.Module):
    return AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


def build_scheduler(optimizer):
    return ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
        verbose=True,
    )


def _ensure_training_dirs(reports_dir: str, round_name: str = "round_00"):
    training_root = os.path.join(reports_dir, "training")
    round_dir = os.path.join(training_root, round_name)
    ckpt_dir = os.path.join(round_dir, "checkpoints")
    plots_dir = os.path.join(round_dir, "plots")
    metrics_dir = os.path.join(round_dir, "metrics")

    os.makedirs(training_root, exist_ok=True)
    os.makedirs(round_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)

    return {
        "training_root": training_root,
        "round_dir": round_dir,
        "ckpt_dir": ckpt_dir,
        "plots_dir": plots_dir,
        "metrics_dir": metrics_dir,
    }


def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, tuple):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    return obj


def _save_json(obj: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_to_serializable(obj), f, indent=2, ensure_ascii=False)


def _save_history(history: list, save_dir: str):
    history_df = pd.DataFrame(history)
    csv_path = os.path.join(save_dir, "history.csv")
    json_path = os.path.join(save_dir, "history.json")
    history_df.to_csv(csv_path, index=False)
    _save_json(history_df.to_dict(orient="records"), json_path)
    return history_df, csv_path, json_path


def _plot_history(history_df: pd.DataFrame, plots_dir: str):
    if len(history_df) == 0:
        return

    # loss curves
    plt.figure(figsize=(8, 5))
    plt.plot(history_df["epoch"], history_df["train_loss"], label="train_loss")
    plt.plot(history_df["epoch"], history_df["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training / Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "loss_curve.png"), dpi=200)
    plt.close()

    # validation metrics
    plt.figure(figsize=(9, 5))
    plt.plot(history_df["epoch"], history_df["val_accuracy"], label="val_accuracy")
    plt.plot(history_df["epoch"], history_df["val_macro_f1"], label="val_macro_f1")
    plt.plot(history_df["epoch"], history_df["val_macro_precision"], label="val_macro_precision")
    plt.plot(history_df["epoch"], history_df["val_macro_recall"], label="val_macro_recall")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "val_metrics_curve.png"), dpi=200)
    plt.close()


def _compute_multilabel_metrics(y_true, y_prob, class_names, threshold=THRESHOLD):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(np.float32)
    y_pred = (y_prob >= threshold).astype(int)

    # label-wise binary accuracy over all labels
    label_accuracy = float((y_true == y_pred).mean())

    # exact-match / subset accuracy
    subset_acc = float(accuracy_score(y_true, y_pred))

    macro_precision = float(
        precision_score(y_true, y_pred, average="macro", zero_division=0)
    )
    macro_recall = float(
        recall_score(y_true, y_pred, average="macro", zero_division=0)
    )
    macro_f1 = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )

    per_class_precision = precision_score(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_recall = recall_score(
        y_true, y_pred, average=None, zero_division=0
    )
    per_class_f1 = f1_score(
        y_true, y_pred, average=None, zero_division=0
    )

    return {
        "accuracy": label_accuracy,
        "subset_accuracy": subset_acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "threshold": float(threshold),
        "per_class": {
            cls: {
                "precision": float(per_class_precision[i]),
                "recall": float(per_class_recall[i]),
                "f1": float(per_class_f1[i]),
            }
            for i, cls in enumerate(class_names)
        },
    }


def _compute_losses(outputs, targets, bce_criterion):
    logits = outputs["logits"]
    main_loss = bce_criterion(logits, targets)

    proto_loss = torch.tensor(0.0, device=logits.device)
    consistency_loss = torch.tensor(0.0, device=logits.device)

    if "prototype_logits" in outputs and outputs["prototype_logits"] is not None:
        proto_logits = outputs["prototype_logits"]
        proto_loss = bce_criterion(proto_logits, targets)

        consistency_loss = torch.mean(
            (torch.sigmoid(logits) - torch.sigmoid(proto_logits)) ** 2
        )

    total_loss = main_loss + 0.2 * proto_loss + 0.05 * consistency_loss

    return total_loss, {
        "main_loss": float(main_loss.detach().item()),
        "proto_loss": float(proto_loss.detach().item()),
        "consistency_loss": float(consistency_loss.detach().item()),
    }


def _run_one_epoch(
    model,
    loader,
    device,
    class_names,
    optimizer=None,
    threshold=THRESHOLD,
):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    bce_criterion = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    running_main_loss = 0.0
    running_proto_loss = 0.0
    running_consistency_loss = 0.0
    total_samples = 0

    all_targets = []
    all_probs = []

    progress = tqdm(loader, leave=False)

    for batch in progress:
        patches = batch["patches"].to(device)         # [B, K, 3, 224, 224]
        patch_mask = batch["patch_mask"].to(device)   # [B, K]
        targets = batch["target"].to(device)          # [B, C]

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            outputs = model(
                patches=patches,
                patch_mask=patch_mask,
                return_patch_features=True,
            )

            total_loss, loss_parts = _compute_losses(
                outputs=outputs,
                targets=targets,
                bce_criterion=bce_criterion,
            )

            if is_train:
                total_loss.backward()
                optimizer.step()

        probs = torch.sigmoid(outputs["logits"])

        batch_size = targets.size(0)
        total_samples += batch_size

        running_loss += total_loss.detach().item() * batch_size
        running_main_loss += loss_parts["main_loss"] * batch_size
        running_proto_loss += loss_parts["proto_loss"] * batch_size
        running_consistency_loss += loss_parts["consistency_loss"] * batch_size

        all_targets.append(targets.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

        progress.set_description(
            f"{'train' if is_train else 'eval'} loss={total_loss.detach().item():.4f}"
        )

    epoch_loss = running_loss / max(total_samples, 1)
    epoch_main_loss = running_main_loss / max(total_samples, 1)
    epoch_proto_loss = running_proto_loss / max(total_samples, 1)
    epoch_consistency_loss = running_consistency_loss / max(total_samples, 1)

    all_targets = np.concatenate(all_targets, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)

    metrics = _compute_multilabel_metrics(
        y_true=all_targets,
        y_prob=all_probs,
        class_names=class_names,
        threshold=threshold,
    )

    metrics.update({
        "loss": float(epoch_loss),
        "main_loss": float(epoch_main_loss),
        "proto_loss": float(epoch_proto_loss),
        "consistency_loss": float(epoch_consistency_loss),
    })

    return metrics


def _save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    best_val_macro_f1,
    metrics,
    path,
):
    ckpt = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "best_val_macro_f1": float(best_val_macro_f1),
        "metrics": _to_serializable(metrics),
    }
    torch.save(ckpt, path)


def _load_checkpoint(model, optimizer, scheduler, path, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])

    return ckpt


def run_model_pipeline(pipeline_bundle: dict):
    """
    Round-00 supervised baseline training:
    - build loaders
    - sanity check
    - train for NUM_EPOCHS
    - validate each epoch
    - save best checkpoint by val_macro_f1
    - evaluate on test
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_names = pipeline_bundle["all_labels"]
    reports_dir = pipeline_bundle["reports_dir"]

    loaders = build_dataloaders(pipeline_bundle)

    if "train" not in loaders:
        raise ValueError("Train loader is missing.")
    if "val" not in loaders:
        raise ValueError("Validation loader is missing.")

    model = build_model(
        num_classes=len(class_names),
        embedding_dim=256,
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    optimizer = build_optimizer(model)
    scheduler = build_scheduler(optimizer)

    print("\n===== MODEL PIPELINE =====")
    print("Device:", device)
    print("Classes:", class_names)
    print("Trainable params:", count_trainable_parameters(model))

    for loader_name, loader in loaders.items():
        print(f"{loader_name} loader size:", len(loader.dataset))

    # Sanity check
    batch = next(iter(loaders["train"]))
    patches = batch["patches"].to(device)
    patch_mask = batch["patch_mask"].to(device)

    with torch.no_grad():
        outputs = model(
            patches=patches,
            patch_mask=patch_mask,
            return_patch_features=True,
        )

    print("\n===== SANITY CHECK =====")
    print("Input patches shape:", tuple(patches.shape))
    print("Patch mask shape:", tuple(patch_mask.shape))
    print("Logits shape:", tuple(outputs["logits"].shape))
    print("Embeddings shape:", tuple(outputs["embeddings"].shape))
    print("Prototype logits shape:", tuple(outputs["prototype_logits"].shape))
    print("Aggregated features shape:", tuple(outputs["aggregated_features"].shape))
    print("Patch features shape:", tuple(outputs["patch_features"].shape))
    if "target" in batch:
        print("Target shape:", tuple(batch["target"].shape))

    # Training dirs
    dirs = _ensure_training_dirs(reports_dir, round_name="round_00")
    best_ckpt_path = os.path.join(dirs["ckpt_dir"], "best_model.pt")
    last_ckpt_path = os.path.join(dirs["ckpt_dir"], "last_model.pt")

    history = []
    best_val_macro_f1 = -1.0
    best_epoch = -1
    best_metrics = None

    print("\n===== START TRAINING: ROUND 00 =====")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n--- Epoch {epoch}/{NUM_EPOCHS} ---")

        train_metrics = _run_one_epoch(
            model=model,
            loader=loaders["train"],
            device=device,
            class_names=class_names,
            optimizer=optimizer,
            threshold=THRESHOLD,
        )

        val_metrics = _run_one_epoch(
            model=model,
            loader=loaders["val"],
            device=device,
            class_names=class_names,
            optimizer=None,
            threshold=THRESHOLD,
        )

        scheduler.step(val_metrics["macro_f1"])

        current_lr = float(optimizer.param_groups[0]["lr"])

        epoch_record = {
            "epoch": int(epoch),
            "lr": current_lr,

            "train_loss": train_metrics["loss"],
            "train_main_loss": train_metrics["main_loss"],
            "train_proto_loss": train_metrics["proto_loss"],
            "train_consistency_loss": train_metrics["consistency_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_subset_accuracy": train_metrics["subset_accuracy"],
            "train_macro_precision": train_metrics["macro_precision"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_macro_f1": train_metrics["macro_f1"],

            "val_loss": val_metrics["loss"],
            "val_main_loss": val_metrics["main_loss"],
            "val_proto_loss": val_metrics["proto_loss"],
            "val_consistency_loss": val_metrics["consistency_loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_subset_accuracy": val_metrics["subset_accuracy"],
            "val_macro_precision": val_metrics["macro_precision"],
            "val_macro_recall": val_metrics["macro_recall"],
            "val_macro_f1": val_metrics["macro_f1"],
        }

        history.append(epoch_record)

        print(
            f"train_loss={train_metrics['loss']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_acc={val_metrics['accuracy']:.4f} | "
            f"val_macro_f1={val_metrics['macro_f1']:.4f} | "
            f"val_macro_precision={val_metrics['macro_precision']:.4f} | "
            f"val_macro_recall={val_metrics['macro_recall']:.4f}"
        )

        # save last checkpoint each epoch
        _save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_val_macro_f1=max(best_val_macro_f1, val_metrics["macro_f1"]),
            metrics={
                "train": train_metrics,
                "val": val_metrics,
            },
            path=last_ckpt_path,
        )

        # save best checkpoint by val_macro_f1
        if val_metrics["macro_f1"] > best_val_macro_f1:
            best_val_macro_f1 = float(val_metrics["macro_f1"])
            best_epoch = int(epoch)
            best_metrics = deepcopy(val_metrics)

            _save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_val_macro_f1=best_val_macro_f1,
                metrics={
                    "train": train_metrics,
                    "val": val_metrics,
                },
                path=best_ckpt_path,
            )

            print(f"[BEST] New best checkpoint saved at epoch {epoch}")

    history_df, history_csv_path, history_json_path = _save_history(
        history=history,
        save_dir=dirs["round_dir"],
    )
    _plot_history(history_df, dirs["plots_dir"])

    summary_train_path = os.path.join(dirs["metrics_dir"], "training_summary.json")
    _save_json({
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_val_metrics": best_metrics,
        "history_csv": history_csv_path,
        "history_json": history_json_path,
        "best_checkpoint": best_ckpt_path,
        "last_checkpoint": last_ckpt_path,
    }, summary_train_path)

    print("\n===== TRAINING FINISHED =====")
    print("Best epoch:", best_epoch)
    print("Best val macro F1:", best_val_macro_f1)
    print("History CSV:", history_csv_path)
    print("Best checkpoint:", best_ckpt_path)

    # Load best checkpoint before test
    _load_checkpoint(
        model=model,
        optimizer=None,
        scheduler=None,
        path=best_ckpt_path,
        device=device,
    )

    test_metrics = None
    if "test" in loaders:
        print("\n===== FINAL TEST EVALUATION =====")
        test_metrics = _run_one_epoch(
            model=model,
            loader=loaders["test"],
            device=device,
            class_names=class_names,
            optimizer=None,
            threshold=THRESHOLD,
        )

        test_metrics_path = os.path.join(dirs["metrics_dir"], "test_metrics.json")
        _save_json(test_metrics, test_metrics_path)

        print(
            f"test_loss={test_metrics['loss']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.4f} | "
            f"test_macro_f1={test_metrics['macro_f1']:.4f} | "
            f"test_macro_precision={test_metrics['macro_precision']:.4f} | "
            f"test_macro_recall={test_metrics['macro_recall']:.4f}"
        )
        print("Test metrics saved to:", test_metrics_path)

    return {
        "model": model,
        "loaders": loaders,
        "device": device,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_macro_f1,
        "best_checkpoint_path": best_ckpt_path,
        "last_checkpoint_path": last_ckpt_path,
        "test_metrics": test_metrics,
        "training_dir": dirs["round_dir"],
    }