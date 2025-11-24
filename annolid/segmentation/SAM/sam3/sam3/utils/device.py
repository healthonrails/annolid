from __future__ import annotations

from typing import Optional, Union

import torch


DeviceLike = Optional[Union[str, torch.device]]


def _mps_available() -> bool:
    """Return True if a usable MPS device is present."""
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


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
