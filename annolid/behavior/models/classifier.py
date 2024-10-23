import torch
import torch.nn as nn
import math
import logging

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Adds positional encoding to the input tensor.

    Args:
        d_model (int): The dimension of the input embeddings.
        max_len (int, optional): The maximum sequence length. Defaults to 5000.
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Make (max_len, 1, d_model) for proper broadcasting
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)  # Register as a buffer, not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for positional encoding.

        Args:
            x (torch.Tensor): The input tensor (seq_len, batch_size, d_model).

        Returns:
            torch.Tensor: The output tensor with positional encoding added.
        """
        x = x + \
            self.pe[:x.size(
                0), :]  # Use slicing for correct positional encoding length.
        return x


class BehaviorClassifier(nn.Module):
    """
    Classifies animal behavior using a Transformer architecture.

    Args:
        feature_extractor (nn.Module): The feature extraction module.
        d_model (int, optional): The embedding dimension. Defaults to 512.
        nhead (int, optional): The number of attention heads. Defaults to 8.
        num_layers (int, optional): The number of transformer encoder layers. Defaults to 6.
        dim_feedforward (int, optional): The dimension of the feedforward network. Defaults to 2048.
        dropout (float, optional): The dropout probability. Defaults to 0.1.
        num_classes (int, optional): The number of behavior classes. Defaults to 5.
    """

    def __init__(self, feature_extractor: nn.Module, d_model: int = 512, nhead: int = 8,
                 num_layers: int = 6, dim_feedforward: int = 2048, dropout: float = 0.1,
                 num_classes: int = 5):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.positional_encoding = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the behavior classifier.

        Args:
            x (torch.Tensor): The input video tensor (batch_size, frames, c, h, w).

        Returns:
            torch.Tensor: The classification output (batch_size, num_classes).
        """
        batch_size, frames, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        features = self.feature_extractor(x)

        # Reshape and transpose
        # (frames, batch_size, feature_dim)
        features = features.view(batch_size, frames, -1).transpose(0, 1)

        features = self.positional_encoding(features)
        encoded_features = self.transformer_encoder(features)
        pooled_features = encoded_features.mean(
            dim=0)  # Global average pooling
        output = self.classifier(pooled_features)
        return output
