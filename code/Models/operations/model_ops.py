import torch
import torch.nn as nn

from DAL.preparation.config import BATCH_SIZE, NUM_WORKERS
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


def run_model_pipeline(pipeline_bundle: dict):
    """
    First model-stage integration:
    - build image-patch dataloaders
    - build model
    - run one forward pass sanity check

    No full training yet.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaders = build_dataloaders(pipeline_bundle)

    model = build_model(
        num_classes=len(pipeline_bundle["all_labels"]),
        embedding_dim=256,
        pretrained=True,
        freeze_backbone=False,
    ).to(device)

    print("\n===== MODEL PIPELINE =====")
    print("Device:", device)
    print("Classes:", pipeline_bundle["all_labels"])
    print("Trainable params:", count_trainable_parameters(model))

    for loader_name, loader in loaders.items():
        print(f"{loader_name} loader size:", len(loader.dataset))

    if "train" in loaders and len(loaders["train"].dataset) > 0:
        batch = next(iter(loaders["train"]))

        patches = batch["patches"].to(device)       # [B, K, C, H, W]
        patch_mask = batch["patch_mask"].to(device) # [B, K]

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

    return {
        "model": model,
        "loaders": loaders,
        "device": device,
    }