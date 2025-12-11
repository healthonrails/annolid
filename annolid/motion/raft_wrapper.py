"""
Lightweight RAFT optical-flow wrapper using torchvision models.

Usage:
    from annolid.motion.raft_wrapper import RAFTOpticalFlow
    raft = RAFTOpticalFlow(model_type="small", device="cuda")
    flow = raft(prev_frame, curr_frame)  # flow: torch.Tensor (B, 2, H, W)

Notes:
- Requires torch and torchvision>=0.15 with optical_flow models.
- Pads inputs to multiples of 8 (as RAFT expects), then unpads outputs.
- Accepts grayscale or RGB; grayscale is repeated to 3 channels.
"""
from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F

try:
    from torchvision.models.optical_flow import (
        raft_large,
        raft_small,
        Raft_Large_Weights,
        Raft_Small_Weights,
    )
except Exception as exc:  # pragma: no cover - optional dependency
    raft_large = raft_small = None
    Raft_Large_Weights = Raft_Small_Weights = None
    _import_error = exc
else:
    _import_error = None


class RAFTOpticalFlow(torch.nn.Module):
    """
    Thin wrapper over torchvision RAFT. Choose `model_type` = 'small' (faster)
    or 'large' (higher accuracy). Returns flow as (B, 2, H, W) torch Tensor.
    """

    def __init__(self, model_type: str = "small", device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        if raft_small is None or Raft_Small_Weights is None:
            raise ImportError(
                f"torchvision optical_flow models not available: {_import_error}"
            )

        self.device = torch.device(device) if device is not None else (
            torch.device("cuda") if torch.cuda.is_available()
            else torch.device("mps") if hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            else torch.device("cpu")
        )

        if model_type == "large":
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(
                weights=weights, progress=False).to(self.device)
        else:
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(
                weights=weights, progress=False).to(self.device)

        self.transforms = weights.transforms()
        self.model.eval()

    @torch.inference_mode()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1, img2: (B, C, H, W) or (C, H, W) tensors, values in [0,1] or [0,255].
        Returns:
            flow: (B, 2, H, W) torch.Tensor on the same device as the model.
        """
        if img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 3:
            img2 = img2.unsqueeze(0)

        # Ensure 3 channels
        if img1.shape[1] == 1:
            img1 = img1.repeat(1, 3, 1, 1)
            img2 = img2.repeat(1, 3, 1, 1)

        original_h, original_w = img1.shape[-2:]
        pad_h = (8 - original_h % 8) % 8
        pad_w = (8 - original_w % 8) % 8
        if pad_h or pad_w:
            img1 = F.pad(img1, (0, pad_w, 0, pad_h))
            img2 = F.pad(img2, (0, pad_w, 0, pad_h))

        # Normalize using torchvision RAFT weights transforms
        img1, img2 = self.transforms(img1, img2)

        flows = self.model(img1.to(self.device), img2.to(self.device))
        flow = flows[-1]  # final refined flow

        if pad_h or pad_w:
            flow = flow[:, :, :original_h, :original_w]

        return flow


def compute_raft_flow(
    prev_frame: torch.Tensor,
    curr_frame: torch.Tensor,
    model_type: str = "small",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convenience function to compute RAFT flow and return (B,2,H,W) tensor."""
    raft = RAFTOpticalFlow(model_type=model_type, device=device)
    with torch.inference_mode():
        return raft(prev_frame, curr_frame)
