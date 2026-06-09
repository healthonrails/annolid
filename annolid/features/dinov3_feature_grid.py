"""Shared DINOv3 feature-grid helpers for tracking and segmentation."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from annolid.utils.logger import logger


def normalize_feature_grid(feats: torch.Tensor) -> torch.Tensor:
    """L2-normalize a dense feature grid over the channel dimension."""
    return feats / (torch.sqrt((feats * feats).sum(dim=0, keepdim=True)) + 1e-12)


def extract_feature_grid(extractor: object, image: Image.Image) -> torch.Tensor:
    """Extract and normalize a single dense DINO feature grid."""
    feats = extractor.extract(image, return_layer="all", normalize=True)
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    if feats.dim() == 4:
        feats = feats[-2:].mean(dim=0)
    return normalize_feature_grid(feats.to(torch.float32))


def svd_positional_basis(
    *,
    extractor: object,
    feature_shape: Tuple[int, int, int],
    patch_size: int,
    components: int,
    cache: Dict[Tuple[str, int, int, int, int], torch.Tensor],
    model_key: str = "",
    image_size: Optional[Tuple[int, int]] = None,
) -> Optional[torch.Tensor]:
    """Estimate an INSID3-style positional channel basis from a blank response."""
    channels, grid_h, grid_w = feature_shape
    if components <= 0 or channels <= 0 or grid_h * grid_w <= 2:
        return None
    if not hasattr(extractor, "model"):
        return None

    cache_key = (
        str(model_key or ""),
        int(channels),
        int(grid_h),
        int(grid_w),
        int(components),
    )
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    if image_size is None:
        width = max(1, int(grid_w * patch_size))
        height = max(1, int(grid_h * patch_size))
    else:
        width, height = int(image_size[0]), int(image_size[1])
    try:
        blank_feats = extract_feature_grid(extractor, Image.new("RGB", (width, height)))
    except Exception as exc:  # pragma: no cover - backend-specific fallback
        logger.debug("DINOv3 positional basis unavailable: %s", exc)
        return None
    if tuple(blank_feats.shape) != tuple(feature_shape):
        logger.debug(
            "DINOv3 positional basis skipped for mismatched blank grid: got %s expected %s",
            tuple(blank_feats.shape),
            tuple(feature_shape),
        )
        return None

    flat = blank_feats.reshape(channels, -1)
    flat = flat - flat.mean(dim=1, keepdim=True)
    try:
        u, _s, _vh = torch.linalg.svd(flat.cpu(), full_matrices=False)
    except RuntimeError as exc:  # pragma: no cover - backend-specific fallback
        logger.debug("DINOv3 positional SVD failed: %s", exc)
        return None
    rank = min(int(components), int(u.shape[1]))
    if rank <= 0:
        return None
    basis = u[:, :rank].contiguous()
    cache[cache_key] = basis
    return basis


def apply_channel_debias_basis(
    feats: torch.Tensor,
    basis: torch.Tensor,
    *,
    strength: float = 1.0,
) -> torch.Tensor:
    """Project feature channels away from a positional basis and renormalize."""
    if feats.dim() != 3 or basis.dim() != 2 or basis.shape[0] != feats.shape[0]:
        return feats
    flat = feats.reshape(feats.shape[0], -1)
    basis = basis.to(device=flat.device, dtype=flat.dtype)
    positional = basis @ (basis.transpose(0, 1) @ flat)
    debiased = flat - float(strength) * positional
    return normalize_feature_grid(debiased.reshape_as(feats))


def coordinate_debias_basis(
    grid_h: int,
    grid_w: int,
    *,
    components: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    """Build a spatial coordinate basis used as a fallback when SVD is unavailable."""
    if grid_h <= 0 or grid_w <= 0 or components <= 0:
        return None
    y_coords = torch.linspace(-1.0, 1.0, grid_h, device=device, dtype=dtype)
    x_coords = torch.linspace(-1.0, 1.0, grid_w, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")
    raw_terms = (
        xx,
        yy,
        xx * yy,
        xx.square(),
        yy.square(),
        xx.square() - yy.square(),
        torch.sin(math.pi * xx),
        torch.sin(math.pi * yy),
        torch.cos(math.pi * xx),
        torch.cos(math.pi * yy),
    )
    columns: List[torch.Tensor] = []
    for term in raw_terms:
        col = term.reshape(-1)
        col = col - col.mean()
        norm = col.norm()
        if float(norm.item()) <= 1e-12:
            continue
        columns.append(col / norm)
        if len(columns) >= components:
            break
    if not columns:
        return None
    orthonormal: List[torch.Tensor] = []
    for col in columns:
        vec = col
        for basis_col in orthonormal:
            vec = vec - torch.dot(vec, basis_col) * basis_col
        norm = vec.norm()
        if float(norm.item()) <= 1e-12:
            continue
        orthonormal.append(vec / norm)
        if len(orthonormal) >= components:
            break
    if not orthonormal:
        return None
    return torch.stack(orthonormal, dim=1)
