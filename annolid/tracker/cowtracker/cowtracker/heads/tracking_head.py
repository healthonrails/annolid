# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CowTracker tracking head - Warping-based iterative refinement."""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from cowtracker.layers.video_transformer import MODEL_CONFIGS, VisionTransformerVideo
from cowtracker.utils.ops import bilinear_sampler, coords_grid


class CowTrackingHead(nn.Module):
    """
    Warping-based iterative refinement module.

    Responsibility: features -> (tracks, visibility, confidence)
    Does NOT handle: feature extraction, windowing
    """

    TEMPORAL_INTERLEAVE_STRIDE = 2
    MAX_FRAMES = 256
    MLP_RATIO = 4.0
    REFINE_PATCH_SIZE = 4

    def __init__(
        self,
        feature_dim: int,
        down_ratio: int = 2,
        warp_iters: int = 5,
        warp_model: str = "vits",
        warp_vit_num_blocks: int = None,
    ):
        """
        Args:
            feature_dim: Input feature dimension (features + side_resnet_channels).
            down_ratio: Feature downsampling ratio relative to original image.
            warp_iters: Number of Warping-based iterative refinement iterations.
            warp_model: Model configuration for video transformer.
            warp_vit_num_blocks: Number of transformer blocks (None = use default).
        """
        super().__init__()

        self.warp_iters = warp_iters
        self.down_ratio = down_ratio

        # Warping-based iterative refinement iteration dimension
        self.iter_dim = MODEL_CONFIGS[warp_model]["features"]

        # Video transformer for temporal attention
        self.refine_net = VisionTransformerVideo(
            warp_model,
            self.iter_dim,
            patch_size=self.REFINE_PATCH_SIZE,
            temporal_interleave_stride=self.TEMPORAL_INTERLEAVE_STRIDE,
            max_frames=self.MAX_FRAMES,
            mlp_ratio=self.MLP_RATIO,
            attn_dropout=0.0,
            proj_dropout=0.0,
            drop_path=0.0,
            num_blocks=warp_vit_num_blocks,
        )

        # Feature processing layers
        self.fmap_conv = nn.Conv2d(feature_dim, self.iter_dim, 1, 1, 0, bias=True)
        self.hidden_conv = nn.Conv2d(
            self.iter_dim * 2, self.iter_dim, 1, 1, 0, bias=True
        )
        self.warp_linear = nn.Conv2d(
            3 * self.iter_dim + 2, self.iter_dim, 1, 1, 0, bias=True
        )
        self.refine_transform = nn.Conv2d(
            self.iter_dim // 2 * 3, self.iter_dim, 1, 1, 0, bias=True
        )

        # Upsampling weights
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2 * self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, (down_ratio**2) * 9, 1, padding=0, bias=True),
        )

        # Flow + visibility + confidence head
        self.flow_head = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2 * self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2 * self.iter_dim, 4, 1, padding=0, bias=True),
        )

        print(
            f"CowTrackingHead initialized: iter_dim={self.iter_dim}, warp_iters={warp_iters}"
        )

    def forward(
        self,
        features: torch.Tensor,
        image_size: Tuple[int, int],
        first_frame_features: torch.Tensor = None,
    ) -> dict:
        """
        Run Warping-based iterative refinement.

        Args:
            features: Extracted features [B, S, C, H, W].
            image_size: Original image size (H_img, W_img) for upsampling.
            first_frame_features: Optional first frame features [B, 1, C, H, W]
                for cross-window tracking.

        Returns:
            dict with:
                - track: Dense tracks [B, S, H_img, W_img, 2].
                - vis: Visibility scores [B, S, H_img, W_img].
                - conf: Confidence scores [B, S, H_img, W_img].
        """
        B, S, _, H, W = features.shape
        H_img, W_img = image_size

        # Project features to iteration dimension
        fmap = self.fmap_conv(features.view(B * S, -1, H, W)).view(B, S, -1, H, W)

        # Frame 0 reference features
        if first_frame_features is not None:
            frame0_fmap = self.fmap_conv(first_frame_features.view(B, -1, H, W)).view(
                B, 1, -1, H, W
            )
        else:
            frame0_fmap = fmap[:, 0:1]
        frame0_expanded = frame0_fmap.expand(B, S, -1, H, W)

        # Initialize hidden state from concatenation of frame0 and current features
        net = self.hidden_conv(
            torch.cat([frame0_expanded, fmap], dim=2).view(B * S, -1, H, W)
        ).view(B, S, -1, H, W)

        # Initialize flow to zero
        flow = torch.zeros(B, S, 2, H, W, device=features.device, dtype=features.dtype)

        # Iterative refinement
        for _ in range(self.warp_iters):
            flow = flow.detach()

            # Compute warped coordinates
            coords = (
                coords_grid(B * S, H, W, device=features.device)
                .to(fmap.dtype)
                .view(B, S, 2, H, W)
            )
            coords_warped = coords + flow

            # Warp features using current flow estimate
            warped_fmap = bilinear_sampler(
                fmap.view(B * S, -1, H, W),
                coords_warped.view(B * S, 2, H, W).permute(0, 2, 3, 1),
            ).view(B, S, -1, H, W)

            # Build refinement input
            refine_inp = self.warp_linear(
                torch.cat([frame0_expanded, warped_fmap, net, flow], dim=2).view(
                    B * S, -1, H, W
                )
            ).view(B, S, -1, H, W)

            # Apply video transformer with temporal attention
            refine_out = self.refine_net(refine_inp)["out"]

            # Update hidden state
            net = self.refine_transform(
                torch.cat(
                    [refine_out.view(B * S, -1, H, W), net.view(B * S, -1, H, W)], dim=1
                )
            ).view(B, S, -1, H, W)

            # Predict flow and info update
            update = self.flow_head(net.view(B * S, -1, H, W)).view(B, S, 4, H, W)
            flow = flow + update[:, :, :2]
            info = update[:, :, 2:]

        # Upsample to original resolution
        weight = 0.25 * self.upsample_weight(net.view(B * S, -1, H, W)).view(
            B, S, -1, H, W
        )
        flow_up, info_up = self._upsample_predictions(flow, info, weight)

        # Convert flow to absolute track coordinates
        tracks = self._flow_to_tracks(flow_up, H_img, W_img)

        return {
            "track": tracks,
            "vis": torch.sigmoid(info_up[..., 0]),
            "conf": torch.sigmoid(info_up[..., 1]),
        }

    def _upsample_predictions(
        self,
        flow: torch.Tensor,
        info: torch.Tensor,
        weight: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Upsample flow and info using learned convex combination."""
        B, S, _, H, W = flow.shape

        flow_ups, info_ups = [], []
        for t in range(S):
            f_up, i_up = self._upsample_single(flow[:, t], info[:, t], weight[:, t])
            flow_ups.append(f_up)
            info_ups.append(i_up)

        return torch.stack(flow_ups, dim=1), torch.stack(info_ups, dim=1)

    def _upsample_single(
        self,
        flow: torch.Tensor,
        info: torch.Tensor,
        mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Upsample single frame using soft convex combination."""
        N, _, H, W = flow.shape
        C = info.shape[1]
        factor = self.down_ratio

        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1).view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1).view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2).permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2).permute(0, 1, 4, 2, 5, 3)

        return (
            up_flow.reshape(N, 2, factor * H, factor * W).permute(0, 2, 3, 1),
            up_info.reshape(N, C, factor * H, factor * W).permute(0, 2, 3, 1),
        )

    def _flow_to_tracks(
        self,
        flow: torch.Tensor,
        H_img: int,
        W_img: int,
    ) -> torch.Tensor:
        """Convert flow to absolute track coordinates."""
        B, S = flow.shape[:2]
        device, dtype = flow.device, flow.dtype

        # Create coordinate grid
        y, x = torch.meshgrid(
            torch.arange(H_img, device=device, dtype=dtype),
            torch.arange(W_img, device=device, dtype=dtype),
            indexing="ij",
        )
        coords = (
            torch.stack([x, y], dim=-1)
            .unsqueeze(0)
            .unsqueeze(0)
            .expand(B, S, -1, -1, -1)
        )

        # Normalize flow relative to frame 0 during inference
        if not self.training:
            flow = flow - flow[:, 0:1]
            flow[:, 0] = 0

        return coords + flow
