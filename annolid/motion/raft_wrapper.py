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

from collections import OrderedDict
import threading
from typing import Optional, Union

import torch
import torch.nn.functional as F

from annolid.utils.devices import clear_device_cache

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

_MAX_CACHED_RAFT_MODELS = 4
_RAFT_MODEL_CACHE: OrderedDict[tuple[str, str], "RAFTOpticalFlow"] = OrderedDict()
_RAFT_CACHE_LOCK = threading.Lock()


def _normalize_raft_model_type(model_type: str) -> str:
    return "large" if str(model_type or "").strip().lower() == "large" else "small"


def _resolve_raft_device(device: Optional[Union[str, torch.device]]) -> torch.device:
    if isinstance(device, torch.device):
        requested = device
    else:
        requested_value = str(device).strip() if device is not None else ""
        requested = (
            None
            if not requested_value or requested_value.lower() == "auto"
            else torch.device(requested_value)
        )

    if requested is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested.type == "cuda":
        if torch.cuda.is_available() and (
            requested.index is None or requested.index < int(torch.cuda.device_count())
        ):
            return requested
        return _resolve_raft_device(None)

    if requested.type == "mps":
        mps_backend = getattr(torch.backends, "mps", None)
        if mps_backend is not None and mps_backend.is_available():
            return requested
        return _resolve_raft_device(None)

    return requested


class RAFTOpticalFlow(torch.nn.Module):
    """
    Thin wrapper over torchvision RAFT. Choose `model_type` = 'small' (faster)
    or 'large' (higher accuracy). Returns flow as (B, 2, H, W) torch Tensor.
    """

    def __init__(
        self,
        model_type: str = "small",
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()
        if raft_small is None or Raft_Small_Weights is None:
            raise ImportError(
                f"torchvision optical_flow models not available: {_import_error}"
            )

        self.model_type = _normalize_raft_model_type(model_type)
        self.device = _resolve_raft_device(device)

        if self.model_type == "large":
            weights = Raft_Large_Weights.DEFAULT
            self.model = raft_large(weights=weights, progress=False).to(self.device)
        else:
            weights = Raft_Small_Weights.DEFAULT
            self.model = raft_small(weights=weights, progress=False).to(self.device)

        self.transforms = weights.transforms()
        self.model.eval()

    @staticmethod
    def _prepare_pair(
        img1: torch.Tensor, img2: torch.Tensor, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0).unsqueeze(0)
        elif img1.dim() == 3:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 2:
            img2 = img2.unsqueeze(0).unsqueeze(0)
        elif img2.dim() == 3:
            img2 = img2.unsqueeze(0)

        if img1.dim() != 4 or img2.dim() != 4:
            raise ValueError(
                f"Expected 2D, 3D, or 4D tensors for RAFT, got {tuple(img1.shape)} and {tuple(img2.shape)}"
            )

        # Ensure 3 channels (expand grayscale, drop alpha on RGBA/BGRA).
        if img1.shape[1] == 1:
            img1 = img1.expand(-1, 3, -1, -1)
        elif img1.shape[1] > 3:
            img1 = img1[:, :3]
        if img2.shape[1] == 1:
            img2 = img2.expand(-1, 3, -1, -1)
        elif img2.shape[1] > 3:
            img2 = img2[:, :3]

        img1 = img1.to(device=device, non_blocking=False)
        img2 = img2.to(device=device, non_blocking=False)

        # torchvision's OpticalFlow transform expects uint8 in [0, 255] or float in [0, 1].
        # Rescale [0, 255] float inputs into [0, 1] without forcing a GPU->CPU sync.
        if img1.is_floating_point() or img2.is_floating_point():
            img1 = img1.float()
            img2 = img2.float()
            scale = torch.where(
                (img1.amax() > 1.5) | (img2.amax() > 1.5),
                img1.new_tensor(1.0 / 255.0),
                img1.new_tensor(1.0),
            )
            img1 = img1 * scale
            img2 = img2 * scale

        return img1, img2

    @torch.inference_mode()
    def forward(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            img1, img2: (B, C, H, W), (C, H, W), or (H, W) tensors, values in [0,1] or [0,255].
        Returns:
            flow: (B, 2, H, W) torch.Tensor on the same device as the model.
        """
        img1, img2 = self._prepare_pair(img1, img2, self.device)

        # Normalize to [-1, 1] float32 before replicate padding so padding works on all devices.
        img1, img2 = self.transforms(img1, img2)

        original_h, original_w = img1.shape[-2:]
        pad_h = (8 - original_h % 8) % 8
        pad_w = (8 - original_w % 8) % 8
        if pad_h or pad_w:
            img1 = F.pad(img1, (0, pad_w, 0, pad_h), mode="replicate")
            img2 = F.pad(img2, (0, pad_w, 0, pad_h), mode="replicate")

        flows = self.model(img1, img2)
        flow = flows[-1]  # final refined flow

        if pad_h or pad_w:
            flow = flow[:, :, :original_h, :original_w]

        return flow


def get_raft_model(
    model_type: str = "small",
    device: Optional[Union[str, torch.device]] = None,
) -> RAFTOpticalFlow:
    """Return a cached RAFTOpticalFlow instance keyed by (model_type, resolved_device)."""
    normalized_type = _normalize_raft_model_type(model_type)
    resolved_device = _resolve_raft_device(device)
    cache_key = (normalized_type, str(resolved_device))

    with _RAFT_CACHE_LOCK:
        cached = _RAFT_MODEL_CACHE.get(cache_key)
        if cached is not None:
            _RAFT_MODEL_CACHE.move_to_end(cache_key)
            return cached

    model = RAFTOpticalFlow(model_type=normalized_type, device=resolved_device)

    evicted_device = None
    with _RAFT_CACHE_LOCK:
        existing = _RAFT_MODEL_CACHE.get(cache_key)
        if existing is not None:
            _RAFT_MODEL_CACHE.move_to_end(cache_key)
            return existing
        _RAFT_MODEL_CACHE[cache_key] = model
        while len(_RAFT_MODEL_CACHE) > _MAX_CACHED_RAFT_MODELS:
            _, evicted = _RAFT_MODEL_CACHE.popitem(last=False)
            evicted_device = evicted.device

    if evicted_device is not None:
        clear_device_cache(torch_module=torch, device=evicted_device)
    return model


def clear_raft_model_cache() -> None:
    """Release cached RAFT models and clear accelerator caches."""
    with _RAFT_CACHE_LOCK:
        devices = [model.device for model in _RAFT_MODEL_CACHE.values()]
        _RAFT_MODEL_CACHE.clear()
    for dev in devices:
        clear_device_cache(torch_module=torch, device=dev)


def compute_raft_flow(
    prev_frame: torch.Tensor,
    curr_frame: torch.Tensor,
    model_type: str = "small",
    device: Optional[Union[str, torch.device]] = None,
) -> torch.Tensor:
    """Convenience function to compute RAFT flow using a cached model and return (B,2,H,W) tensor."""
    raft = get_raft_model(model_type=model_type, device=device)
    with torch.inference_mode():
        return raft(prev_frame, curr_frame)
