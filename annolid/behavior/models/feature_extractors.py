import logging
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from transformers import CLIPModel, CLIPProcessor

from annolid.features import Dinov3Config, Dinov3FeatureExtractor

logger = logging.getLogger(__name__)


class CLIPFeatureExtractor(nn.Module):
    """
    A class to extract features from images using the CLIP vision encoder.

    Args:
        model_name (str): The name of the pre-trained CLIP model (e.g., 'openai/clip-vit-base-patch32').
    """

    def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        self.clip_model = CLIPModel.from_pretrained(model_name)
        self.vision_encoder = self.clip_model.vision_model
        self.processor = CLIPProcessor.from_pretrained(model_name)
        hidden_size = getattr(self.vision_encoder.config, "hidden_size", None)
        if hidden_size is None:
            raise RuntimeError(
                "Could not determine CLIP vision hidden size from configuration.")
        self.feature_dim = int(hidden_size)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input images.

        Args:
            images (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Extracted features of shape (batch_size, feature_dim).
        """
        # Process the images using CLIP's vision encoder
        with torch.no_grad():  # Optional: Avoid backpropagation during feature extraction
            features = self.vision_encoder(pixel_values=images).pooler_output

        return features  # Output shape: (batch_size, feature_dim)


class ResNetFeatureExtractor(nn.Module):
    """
    Extracts features from images using a ResNet backbone.
    """

    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        # Use torchvision.models directly for easier weight handling
        if pretrained:
            # or .IMAGENET1K_V1 if you specifically need that
            self.resnet = models.resnet18(
                weights=models.ResNet18_Weights.DEFAULT)
        else:
            # Explicitly set weights to None if not pretrained
            self.resnet = models.resnet18(weights=None)

        self.resnet_in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()  # Remove the classification head

        self.project_layer = nn.Linear(
            self.resnet_in_features, feature_dim) if feature_dim != self.resnet_in_features else None
        self.feature_dim = feature_dim if self.project_layer else self.resnet_in_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.resnet(x)
        if self.project_layer:
            features = self.project_layer(features)
        return features


class Dinov3BehaviorFeatureExtractor(nn.Module):
    """
    Wrapper around the dense DINOv3 extractor to produce clip-level embeddings.

    Args:
        model_name (str): Hugging Face checkpoint id.
        feature_dim (int): Dimensionality of output embeddings expected by the classifier.
        freeze (bool): If True, prevent gradients through DINOv3 weights.
        device (Optional[torch.device]): Preferred device for inference.
    """

    def __init__(
        self,
        model_name: str = "facebook/dinov3-vits16-pretrain-lvd1689m",
        feature_dim: int = 768,
        freeze: bool = True,
        device: Optional[str] = None,
    ):
        super().__init__()
        cfg = Dinov3Config(model_name=model_name,
                           device=device, return_layer="last")
        self.extractor = Dinov3FeatureExtractor(cfg)
        hidden_size = getattr(self.extractor.model.config, "hidden_size", None)
        embed_dim = getattr(self.extractor.model.config,
                            "embed_dim", hidden_size)
        if embed_dim is None:
            raise RuntimeError(
                "Could not determine DINOv3 embedding dimension from model config.")
        self.backbone_dim = int(embed_dim)
        self.project_layer = (
            nn.Linear(
                self.backbone_dim, feature_dim) if feature_dim != self.backbone_dim else nn.Identity()
        )
        if freeze:
            for param in self.extractor.model.parameters():
                param.requires_grad = False

        self.feature_dim = feature_dim

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Projected embeddings of shape (batch_size, feature_dim).
        """
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor (B, C, H, W), got shape {tuple(x.shape)}")

        device = x.device
        batch_features = []
        for frame in x:
            # Convert to HxWxC uint8 for the DINO extractor.
            frame_np = frame.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()
            frame_uint8 = (frame_np * 255.0).astype(np.uint8)
            feat_grid = self.extractor.extract(
                frame_uint8, return_type="torch", normalize=True)
            # Global average pooling over patch grid.
            pooled = feat_grid.mean(dim=(-2, -1))
            batch_features.append(pooled)

        features = torch.stack(batch_features, dim=0).to(device)
        return self.project_layer(features)
