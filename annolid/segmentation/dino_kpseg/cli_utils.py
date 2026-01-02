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
