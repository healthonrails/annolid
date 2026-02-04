from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn as nn

from annolid.behavior.data_loading.transforms import (
    IdentityTransform,
    ResizeCenterCropNormalize,
)
from annolid.behavior.models.classifier import BehaviorClassifier
from annolid.behavior.models.feature_extractors import (
    CLIPFeatureExtractor,
    Dinov3BehaviorFeatureExtractor,
    ResNetFeatureExtractor,
)

BACKBONE_CHOICES = ("clip", "resnet18", "dinov3")
DEFAULT_DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"


def build_transform(backbone: str):
    # DINOv3 feature extractor performs its own normalization/resize.
    return IdentityTransform() if backbone == "dinov3" else ResizeCenterCropNormalize()


def build_feature_extractor(
    backbone: str,
    device: torch.device,
    *,
    dinov3_model: str,
    feature_dim: Optional[int],
    unfreeze_dino: bool = False,
) -> Tuple[nn.Module, int]:
    """Factory for feature extractors."""
    if backbone == "clip":
        extractor = CLIPFeatureExtractor()
        if feature_dim is not None and feature_dim != extractor.feature_dim:
            raise ValueError("CLIP backbone does not support overriding feature_dim.")
        return extractor.to(device), extractor.feature_dim
    if backbone == "resnet18":
        target_dim = feature_dim or 512
        extractor = ResNetFeatureExtractor(feature_dim=target_dim)
        return extractor.to(device), extractor.feature_dim
    if backbone == "dinov3":
        target_dim = feature_dim or 768
        extractor = Dinov3BehaviorFeatureExtractor(
            model_name=dinov3_model,
            feature_dim=target_dim,
            freeze=not unfreeze_dino,
            device=device.type,
        )
        return extractor.to(device), extractor.feature_dim
    raise ValueError(
        f"Unsupported backbone '{backbone}'. Valid options: {BACKBONE_CHOICES}"
    )


def build_classifier(
    *,
    num_classes: int,
    backbone: str,
    device: torch.device,
    dinov3_model: str,
    feature_dim: Optional[int],
    transformer_dim: int,
    unfreeze_dinov3: bool = False,
) -> BehaviorClassifier:
    feature_extractor, backbone_dim = build_feature_extractor(
        backbone,
        device,
        dinov3_model=dinov3_model,
        feature_dim=feature_dim,
        unfreeze_dino=unfreeze_dinov3,
    )
    return BehaviorClassifier(
        feature_extractor,
        num_classes=int(num_classes),
        d_model=int(transformer_dim),
        feature_dim=int(backbone_dim),
    ).to(device)


def load_classifier(
    *,
    checkpoint_path: str,
    num_classes: int,
    backbone: str,
    device: torch.device,
    dinov3_model: str,
    feature_dim: Optional[int],
    transformer_dim: int,
) -> BehaviorClassifier:
    model = build_classifier(
        num_classes=num_classes,
        backbone=backbone,
        device=device,
        dinov3_model=dinov3_model,
        feature_dim=feature_dim,
        transformer_dim=transformer_dim,
        unfreeze_dinov3=False,
    )
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model
