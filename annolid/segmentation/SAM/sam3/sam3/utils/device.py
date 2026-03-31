from __future__ import annotations

from contextlib import nullcontext
from typing import Optional, Union

import torch
from torch import nn


DeviceLike = Optional[Union[str, torch.device]]


def _mps_available() -> bool:
    """Return True if a usable MPS device is present."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def cuda_is_available() -> bool:
    """Return True when CUDA is usable in the current process."""
    return torch.cuda.is_available()


def cuda_device_major(device_index: int = 0) -> Optional[int]:
    """Return the major compute capability for a CUDA device, if available."""
    if not cuda_is_available():
        return None
    return torch.cuda.get_device_properties(device_index).major


def supports_tf32(device_index: int = 0) -> bool:
    """Return True when the current CUDA device should enable TF32."""
    major = cuda_device_major(device_index)
    return major is not None and major >= 8


def supports_flash_attention(device_index: int = 0) -> bool:
    """Return True when the current CUDA device is new enough for flash attention."""
    major = cuda_device_major(device_index)
    return major is not None and major >= 8


def cuda_runtime_summary(device_index: int = 0) -> str:
    """Return a human-readable CUDA runtime summary."""
    if not cuda_is_available():
        return "CUDA unavailable"
    return (
        f"CUDA arch {torch.cuda.get_arch_list()}, "
        f"GPU device: {torch.cuda.get_device_properties(device_index)}"
    )


def select_device(preferred: DeviceLike = None) -> torch.device:
    """
    Pick the best available torch device.

    Priority: explicitly requested -> CUDA -> MPS -> CPU.
    """
    if preferred is not None:
        return torch.device(preferred)

    if torch.cuda.is_available():
        return torch.device("cuda")
    if _mps_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_default_device(preferred: DeviceLike = None) -> torch.device:
    """
    Resolve and set the torch default device, returning the resolved device.
    """
    resolved = select_device(preferred)
    try:
        torch.set_default_device(resolved)
    except Exception:
        # Best-effort: some environments may disallow setting the default device.
        pass
    return resolved


def safe_autocast(
    *,
    device: DeviceLike = None,
    dtype: torch.dtype = torch.bfloat16,
):
    """
    Return CUDA autocast context only when the resolved device is CUDA.
    """
    resolved = select_device(device)
    if resolved.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def module_device(module: nn.Module) -> torch.device:
    """
    Return the first-parameter device for a module, defaulting to CPU.
    """
    try:
        return next(module.parameters()).device
    except Exception:
        return torch.device("cpu")


def module_dtype(module: nn.Module) -> torch.dtype:
    """
    Return the first-parameter dtype for a module, defaulting to float32.
    """
    try:
        return next(module.parameters()).dtype
    except Exception:
        return torch.float32


def to_device(
    tensor: torch.Tensor,
    device: DeviceLike,
    *,
    non_blocking: bool = False,
) -> torch.Tensor:
    """
    Move a tensor to a target device with consistent API.
    """
    return tensor.to(device=select_device(device), non_blocking=non_blocking)


def host_to_device(
    tensor: torch.Tensor,
    device: DeviceLike,
    *,
    non_blocking: bool = True,
) -> torch.Tensor:
    """
    Move host/device tensor to target device safely.

    - Uses pinned-memory fast path only for CPU->CUDA transfers.
    - Avoids pin_memory on MPS/CPU tensors to prevent cross-device storage errors.
    """
    target = select_device(device)
    if tensor.device.type == "cpu" and target.type == "cuda":
        try:
            tensor = tensor.pin_memory()
        except Exception:
            pass
    return tensor.to(device=target, non_blocking=bool(non_blocking))


def tensor_to_module(
    tensor: torch.Tensor,
    module: nn.Module,
    *,
    non_blocking: bool = False,
) -> torch.Tensor:
    """
    Move a tensor to match a module's device and dtype.
    """
    return tensor.to(
        device=module_device(module),
        dtype=module_dtype(module),
        non_blocking=non_blocking,
    )
