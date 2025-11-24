"""Minimal RLE helpers for inference-only builds."""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import torch

try:  # optional; only used if available
    from pycocotools import mask as mask_util
except Exception:  # pragma: no cover - optional dependency
    mask_util = None


def _encode_single_mask(mask: np.ndarray) -> dict:
    """Return COCO-style uncompressed RLE for a single binary mask."""
    h, w = mask.shape
    pixels = mask.astype(np.uint8).flatten(order="F")
    counts: List[int] = []
    prev_val = 0
    count = 0
    for pix in pixels:
        if pix == prev_val:
            count += 1
        else:
            counts.append(count)
            count = 1
            prev_val = int(pix)
    counts.append(count)
    rle = {"counts": counts, "size": [h, w]}
    return rle


def rle_encode(masks: torch.Tensor, return_areas: bool = False) -> Sequence[dict]:
    """
    Encode a stack of boolean masks into COCO-style RLE dictionaries.

    This is intentionally lightweight and avoids training dependencies; if
    pycocotools is available we use it to produce compressed counts, otherwise
    we fall back to a simple uncompressed RLE.
    """
    if masks.ndim != 3:
        raise ValueError("Mask tensor must have shape (N, H, W)")
    masks = masks.bool()
    if masks.numel() == 0:
        return []

    masks_np = masks.detach().cpu().numpy()
    rles = []
    for mask in masks_np:
        if mask_util is not None:  # use optimized path when available
            rle = mask_util.encode(np.asfortranarray(mask.astype(np.uint8)))
            # mask_util.encode returns a list; take the first element
            rle = rle[0]
            rle["counts"] = rle["counts"].decode("utf-8")
            if return_areas:
                rle["area"] = int(mask.sum())
        else:
            rle = _encode_single_mask(mask)
            if return_areas:
                rle["area"] = int(mask.sum())
        rles.append(rle)
    return rles


def robust_rle_encode(masks: torch.Tensor) -> Sequence[dict]:
    """Compatibility wrapper mirroring the training helper signature."""
    return rle_encode(masks)
