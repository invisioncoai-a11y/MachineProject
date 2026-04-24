import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights


class HybridPlantDiseaseModel(nn.Module):
    """
    ConvNeXt-Tiny backbone
    + image-level aggregation over lesion patches
    + classification head
    + prototype / embedding head
    """

    def __init__(
        self,
        num_classes: int,
        embedding_dim: int = 256,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = convnext_tiny(weights=weights)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        in_features = backbone.classifier[2].in_features

        self.dropout = nn.Dropout(0.2)
        self.classification_head = nn.Linear(in_features, num_classes)

        self.embedding_head = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

        self.prototypes = nn.Parameter(torch.randn(num_classes, embedding_dim))

        self._reset_custom_heads()

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

    def _reset_custom_heads(self):
        nn.init.xavier_uniform_(self.classification_head.weight)
        nn.init.zeros_(self.classification_head.bias)

        for m in self.embedding_head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

        nn.init.normal_(self.prototypes, mean=0.0, std=0.02)

    def _encode_patches(self, patches):
        """
        patches: [B, K, C, H, W]
        returns patch_features: [B, K, D]
        """
        b, k, c, h, w = patches.shape

        flat = patches.view(b * k, c, h, w)
        feat_map = self.features(flat)
        pooled = self.avgpool(feat_map)
        pooled = torch.flatten(pooled, 1)
        pooled = self.dropout(pooled)

        patch_features = pooled.view(b, k, -1)
        return patch_features

    def _aggregate_patch_features(self, patch_features, patch_mask):
        """
        patch_features: [B, K, D]
        patch_mask: [B, K]
        """
        mask = patch_mask.unsqueeze(-1).float()  # [B, K, 1]
        denom = mask.sum(dim=1).clamp(min=1.0)
        aggregated = (patch_features * mask).sum(dim=1) / denom
        return aggregated

    def forward(self, patches, patch_mask, return_patch_features: bool = False):
        """
        patches: [B, K, C, H, W]
        patch_mask: [B, K]
        """
        patch_features = self._encode_patches(patches)        # [B, K, D]
        aggregated_features = self._aggregate_patch_features(
            patch_features, patch_mask
        )                                                     # [B, D]

        logits = self.classification_head(aggregated_features)

        embeddings = self.embedding_head(aggregated_features)
        embeddings = F.normalize(embeddings, p=2, dim=1)

        prototypes = F.normalize(self.prototypes, p=2, dim=1)
        prototype_logits = embeddings @ prototypes.t()

        outputs = {
            "logits": logits,
            "embeddings": embeddings,
            "prototype_logits": prototype_logits,
            "aggregated_features": aggregated_features,
        }

        if return_patch_features:
            outputs["patch_features"] = patch_features

        return outputs