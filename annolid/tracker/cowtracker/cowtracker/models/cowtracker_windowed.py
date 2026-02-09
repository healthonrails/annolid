# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""CoWTracker Windowed: Full version for long videos."""

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from cowtracker.models.cowtracker import CoWTracker
from cowtracker.inference.windowed import WindowedInference


class CoWTrackerWindowed(nn.Module, PyTorchModelHubMixin):
    """
    CoWTracker windowed version: processes video in sliding windows.

    Suitable for: long videos / limited GPU memory.
    Composes CoWTracker with WindowedInference.
    """

    def __init__(
        self,
        # Window parameters
        window_len: int = 100,
        stride: int = 100,
        num_memory_frames: int = 10,
        # CoWTracker parameters
        **cow_tracker_kwargs,
    ):
        """
        Args:
            window_len: Number of frames per window.
            stride: Step size between windows.
            num_memory_frames: Maximum number of memory frames.
            **cow_tracker_kwargs: Arguments passed to CoWTracker.
        """
        super().__init__()

        print(
            f"Initializing CoWTrackerWindowed: window_len={window_len}, stride={stride}"
        )

        self.model = CoWTracker(**cow_tracker_kwargs)
        self.windowed = WindowedInference(
            window_len=window_len,
            stride=stride,
            num_memory_frames=num_memory_frames,
        )

    def forward(self, video: torch.Tensor, queries: torch.Tensor = None) -> dict:
        """
        Forward pass with windowed inference.

        Args:
            video: Input video [B, S, 3, H, W] or [S, 3, H, W] in range [0, 255].
            queries: Optional query points (unused, for API compatibility).

        Returns:
            dict with:
                - track: Dense tracks [B, T, H, W, 2].
                - vis: Visibility scores [B, T, H, W].
                - conf: Confidence scores [B, T, H, W].
        """
        # Normalize input
        images = video / 255.0
        if images.ndim == 4:
            images = images.unsqueeze(0)

        B, T, C, H, W = images.shape
        device, dtype = images.device, images.dtype

        # Initialize accumulated outputs
        accumulated = {
            "track": torch.zeros((B, T, H, W, 2), device=device, dtype=dtype),
            "vis": torch.zeros((B, T, H, W), device=device, dtype=dtype),
            "conf": torch.zeros((B, T, H, W), device=device, dtype=dtype),
        }

        windows = self.windowed.compute_windows(T)
        first_frame = images[:, 0:1]
        first_frame_features = None

        for window_idx, (start, end) in enumerate(windows):
            if not self.training:
                print(
                    f"Processing window {window_idx + 1}/{len(windows)}: frames [{start}, {end})"
                )

            # Get memory frame indices
            memory_indices = self.windowed.select_memory_frames(window_idx, start)
            if not self.training and memory_indices:
                print(f"  Memory frames: {memory_indices}")

            # Gather frames: first_frame + memory + window
            frames = self._gather_frames(
                images, first_frame, start, end, memory_indices
            )

            # Extract backbone tokens
            tokens, patch_idx = self.model.aggregator(frames)

            # Extract combined features
            features = self.model.feature_extractor(tokens, frames, patch_idx)

            # Split features: first_frame | memory | window
            first_frame_features = features[:, 0:1]
            num_memory = len(memory_indices)

            # Run tracking on extended features (memory + window), using first_frame as reference
            extended_features = features[:, 1:]  # Exclude first_frame from input
            pred = self.model.tracking_head(
                extended_features,
                image_size=(H, W),
                first_frame_features=first_frame_features,
            )

            # Extract window predictions (remove memory frames from output)
            window_pred = {
                "track": pred["track"][:, num_memory:],
                "vis": pred["vis"][:, num_memory:],
                "conf": pred["conf"][:, num_memory:],
            }

            # Merge into accumulated results
            self.windowed.merge_predictions(
                window_idx, start, end, window_pred, accumulated
            )

            # Cleanup for memory efficiency
            if not self.training:
                del features, tokens, pred
                torch.cuda.empty_cache()

        if not self.training:
            accumulated["images"] = images

        return accumulated

    def _gather_frames(
        self,
        images: torch.Tensor,
        first_frame: torch.Tensor,
        start: int,
        end: int,
        memory_indices: list,
    ) -> torch.Tensor:
        """Gather first_frame + memory + window frames."""
        parts = [first_frame]

        if memory_indices:
            parts.append(images[:, memory_indices])

        parts.append(images[:, start:end])

        return torch.cat(parts, dim=1)

    # Proxy properties for convenient access to internal components
    @property
    def aggregator(self):
        return self.model.aggregator

    @property
    def feature_extractor(self):
        return self.model.feature_extractor

    @property
    def tracking_head(self):
        return self.model.tracking_head

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str = None,
        window_len: int = 100,
        stride: int = 100,
        device: str = "cuda",
        dtype=torch.bfloat16,
    ):
        """
        Load model from checkpoint (local path or HuggingFace Hub).

        Args:
            checkpoint_path: Path to local checkpoint file.
                             If None, downloads from default HuggingFace repo.
            window_len: Number of frames per window.
            stride: Step size between windows.
            device: Target device.
            dtype: Target dtype.

        Returns:
            Loaded model in eval mode.
        """
        model = cls(window_len=window_len, stride=stride)

        # Use CoWTracker's checkpoint loading method (handles local path and HuggingFace download)
        ckpt = CoWTracker._load_checkpoint(checkpoint_path)
        state_dict = ckpt.get("model", ckpt)

        # Remap legacy checkpoint keys if needed (delegate to CoWTracker)
        legacy_prefixes = [
            "tracking_head.feature_extractor.",
            "tracking_head.aggregator.",
            "tracking_head.fnet.",
        ]
        if any(k.startswith(p) for k in state_dict.keys() for p in legacy_prefixes):
            print("Detected legacy checkpoint format, remapping keys...")
            state_dict = CoWTracker._remap_legacy_state_dict(state_dict)

        # Add "model." prefix if checkpoint is from CoWTracker (no prefix)
        # CoWTrackerWindowed wraps CoWTracker as self.model, so keys need "model." prefix
        if not any(k.startswith("model.") for k in state_dict.keys()):
            print("Adding 'model.' prefix to state dict keys...")
            state_dict = {f"model.{k}": v for k, v in state_dict.items()}

        msg = model.load_state_dict(state_dict, strict=False)
        print(f"Load message: {msg}")

        model = model.to(device).to(dtype)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        print("Model loaded successfully!")
        return model
