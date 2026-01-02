from __future__ import annotations

from typing import Tuple

import torch


def parse_layers(value: str) -> Tuple[int, ...]:
    raw = str(value or "").strip()
    if not raw:
        return (-1,)
    items = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        items.append(int(token))
    return tuple(items) if items else (-1,)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def normalize_device(device: str | None) -> str:
    """Normalize CLI-style device strings into valid torch device strings.

    Supports common patterns:
      - "0" -> "cuda:0" (when CUDA is available)
      - "cuda0" -> "cuda:0"
      - "auto" / "best" / "" / None -> best available device
    """
    if device is None:
        return default_device()
    raw = str(device).strip()
    if not raw:
        return default_device()

    lowered = raw.lower()
    if lowered in ("auto", "best"):
        return default_device()

    if lowered.isdigit():
        if torch.cuda.is_available():
            return f"cuda:{int(lowered)}"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if lowered.startswith("cuda") and ":" not in lowered:
        suffix = lowered[4:]
        if suffix.isdigit():
            return f"cuda:{int(suffix)}"

    return raw
