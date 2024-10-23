import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ResNetFeatureExtractor(nn.Module):
    """
    Extracts features from images using a ResNet backbone.

    Args:
        pretrained (bool, optional): Whether to use a pre-trained ResNet model. Defaults to True.
        feature_dim (int, optional): The desired dimension of the output features. Defaults to 512.
    """

    def __init__(self, pretrained: bool = True, feature_dim: int = 512):
        super().__init__()
        self.resnet = torch.hub.load(
            'pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
        self.resnet_in_features = self.resnet.fc.in_features  # Store the in_features
        self.resnet.fc = nn.Identity()

        # Use self.resnet_in_features  for the projection layer
        self.project_layer = nn.Linear(
            self.resnet_in_features, feature_dim) if feature_dim != self.resnet_in_features else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.

        Args:
            x (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The extracted feature tensor.
        """
        features = self.resnet(x)
        if self.project_layer:  # Apply projection if feature_dim is different
            features = self.project_layer(features)
        return features
