import torch
import torch.nn as nn
import torchvision.models as models
import logging

logger = logging.getLogger(__name__)


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.resnet(x)
        if self.project_layer:
            features = self.project_layer(features)
        return features
