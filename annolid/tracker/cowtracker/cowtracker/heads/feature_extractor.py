# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Feature extraction: DPT + ResNet side features."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cowtracker.dependencies import get_vggt_dpt_head_cls
from cowtracker.layers.resnet_deconv import ResNet18Deconv


class FeatureExtractor(nn.Module):
    """
    Combined DPT and ResNet feature extractor.

    Takes aggregated tokens from backbone and raw images,
    outputs combined features for tracking.
    """

    DIM_IN = 2048  # 2 * embed_dim (1024)
    PATCH_SIZE = 14
    INTERMEDIATE_LAYER_IDX = [4, 11, 17, 23]

    def __init__(
        self,
        features: int = 128,
        down_ratio: int = 2,
        side_resnet_channels: int = 128,
    ):
        """
        Args:
            features: Number of DPT output features.
            down_ratio: Downsampling ratio relative to input image.
            side_resnet_channels: Number of ResNet side feature channels.
        """
        super().__init__()
        # Keep VGGT's DPTHead here for checkpoint/key compatibility with
        # upstream CowTracker weights.
        DPTHead = get_vggt_dpt_head_cls()

        self.features = features
        self.down_ratio = down_ratio

        # DPT head for backbone features
        self.dpt_head = DPTHead(
            dim_in=self.DIM_IN,
            patch_size=self.PATCH_SIZE,
            features=features,
            feature_only=True,
            down_ratio=down_ratio,
            pos_embed=False,
            intermediate_layer_idx=self.INTERMEDIATE_LAYER_IDX,
        )

        # ResNet for raw image features
        self.fnet = ResNet18Deconv(3, side_resnet_channels)

        self.out_dim = features + side_resnet_channels

    def forward(
        self,
        aggregated_tokens_list: list,
        images: torch.Tensor,
        patch_start_idx: int,
    ) -> torch.Tensor:
        """
        Extract combined features from backbone tokens and raw images.

        Args:
            aggregated_tokens_list: List of tokens from aggregator.
            images: Input images [B, S, 3, H, W].
            patch_start_idx: Patch start index for DPT.

        Returns:
            combined_features: [B, S, C, H_out, W_out] where C = features + side_resnet_channels.
        """
        B, S, _, H_img, W_img = images.shape

        # DPT features from backbone tokens
        backbone_features = self.dpt_head(
            aggregated_tokens_list, images, patch_start_idx
        )
        _, _, _, H_out, W_out = backbone_features.shape

        # Side ResNet features from raw images
        images_flat = images.view(B * S, 3, H_img, W_img)
        side_features = self.fnet(images_flat)[0]
        _, side_channels, H_side, W_side = side_features.shape

        # Resize side features to match backbone output if needed
        if H_side != H_out or W_side != W_out:
            side_features = F.interpolate(
                side_features, size=(H_out, W_out), mode="bilinear", align_corners=True
            )

        side_features = side_features.view(B, S, side_channels, H_out, W_out)

        return torch.cat([backbone_features, side_features], dim=2)
