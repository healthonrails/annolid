import torch
import torch.nn as nn
import torchvision.models as models
import logging
from transformers import CLIPModel, CLIPProcessor

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        features = self.resnet(x)
        if self.project_layer:
            features = self.project_layer(features)
        return features
