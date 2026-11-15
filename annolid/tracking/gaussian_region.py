"""Gaussian-weighted local-region descriptors and drift-free pixel template refinement.

Provides three complementary spatial-region mechanisms for keypoint tracking:

1. **Mask-aware Gaussian feature-grid smoothing** (`gaussian_smooth_feature_grid`):
   aggregates a small isotropic Gaussian neighborhood around each patch cell while
   normalizing strictly over foreground animal patches when an instance mask is
   present, preventing background bleed on contour landmarks (nose, ears, paws,
   tail tip).
2. **Continuous sub-patch Gaussian sampling** (`sample_subpatch_gaussian_descriptor`):
   samples feature grids at continuous ``(r, c)`` sub-patch coordinates rather than
   floor-quantized integer cells so landmarks straddling patch boundaries retain
   smooth appearance representations.
3. **Drift-free Gaussian-weighted pixel-region ZNCC refinement**
   (`GaussianPixelTemplate`, `extract_gaussian_pixel_template`,
   `match_gaussian_pixel_template`): anchors a small Gaussian-windowed pixel +
   gradient template at manual seed/correction frames and evaluates vectorized
   zero-mean normalized cross-correlation around each frame's coarse estimate to
   prevent cumulative frame-to-frame optical-flow drift.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from annolid.features.dinov3_feature_grid import normalize_feature_grid

__all__ = [
    "GaussianPixelTemplate",
    "extract_gaussian_pixel_template",
    "gaussian_kernel_1d",
    "gaussian_kernel_2d",
    "gaussian_smooth_feature_grid",
    "match_gaussian_pixel_template",
    "pixel_to_continuous_patch",
    "refine_patch_region_gaussian",
    "sample_subpatch_gaussian_descriptor",
]


@dataclass(frozen=True)
class GaussianPixelTemplate:
    """Zero-weighted-mean, unit-weighted-norm pixel/gradient region anchor."""

    radius_px: int
    sigma_px: float
    weights: torch.Tensor
    norm_intensity_kernel: torch.Tensor
    norm_grad_kernel: Optional[torch.Tensor] = None
    min_std: float = 1.5


def gaussian_kernel_1d(
    radius: int,
    sigma: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized 1D symmetric Gaussian kernel of size ``2 * radius + 1``."""
    r = max(0, int(radius))
    if r == 0:
        return torch.ones((1,), device=device, dtype=dtype)
    s = max(1e-4, float(sigma))
    coords = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    weights = torch.exp(-0.5 * (coords / s) ** 2)
    weights = weights / weights.sum().clamp_min(1e-12)
    return weights.to(dtype=dtype)


def gaussian_kernel_2d(
    radius: int,
    sigma: float,
    *,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Build a normalized 2D isotropic Gaussian kernel of shape ``[2r+1, 2r+1]``."""
    k1d = gaussian_kernel_1d(radius, sigma, device=device, dtype=torch.float32)
    k2d = torch.outer(k1d, k1d)
    k2d = k2d / k2d.sum().clamp_min(1e-12)
    return k2d.to(dtype=dtype)


def _separable_conv2d(x: torch.Tensor, k1d: torch.Tensor, radius: int) -> torch.Tensor:
    """Apply separable 2D depthwise convolution to ``[B, C, H, W]``."""
    channels = int(x.shape[1])
    kv = k1d.view(1, 1, -1, 1).expand(channels, 1, -1, 1)
    kh = k1d.view(1, 1, 1, -1).expand(channels, 1, 1, -1)
    y = F.conv2d(x, kv, padding=(radius, 0), groups=channels)
    y = F.conv2d(y, kh, padding=(0, radius), groups=channels)
    return y


def gaussian_smooth_feature_grid(
    feats: torch.Tensor,
    *,
    radius: int,
    sigma: float,
    patch_mask: Optional[np.ndarray | torch.Tensor] = None,
    mask_aware: bool = True,
    blend: float = 1.0,
    normalize: bool = True,
) -> torch.Tensor:
    """Aggregate local Gaussian patch neighborhoods with optional foreground mask gating.

    Parameters
    ----------
    feats : torch.Tensor
        Feature grid of shape ``[C, H, W]``.
    radius : int
        Neighborhood radius in patch units (``0`` returns ``feats`` normalized).
    sigma : float
        Spatial Gaussian standard deviation in patch units.
    patch_mask : np.ndarray or torch.Tensor, optional
        Boolean or ``[0, 1]`` foreground mask of shape ``[H, W]``.
    mask_aware : bool
        When ``True`` and ``patch_mask`` is provided, normalizes convolution over
        foreground cells ``((F * M) * G) / (M * G + eps)`` wherever foreground
        support exists, avoiding background bleed at animal contours.
    blend : float
        Interpolation weight in ``[0, 1]`` between center patch features (``0``)
        and the Gaussian regional descriptor (``1``).
    normalize : bool
        If ``True``, L2-normalizes each spatial cell along ``dim=0``.
    """
    if feats.dim() != 3:
        return feats
    r = max(0, int(radius))
    alpha = float(np.clip(blend, 0.0, 1.0))
    channels, grid_h, grid_w = feats.shape
    if r == 0 or alpha <= 0.0 or channels == 0 or (grid_h <= 1 and grid_w <= 1):
        return normalize_feature_grid(feats) if normalize else feats

    work_dtype = (
        torch.float32 if feats.dtype in (torch.float16, torch.bfloat16) else feats.dtype
    )
    x = feats.to(dtype=work_dtype).unsqueeze(0)
    k1d = gaussian_kernel_1d(r, sigma, device=feats.device, dtype=work_dtype)

    ones = torch.ones((1, 1, grid_h, grid_w), device=feats.device, dtype=work_dtype)
    border_norm = _separable_conv2d(ones, k1d, r).clamp_min(1e-8)
    unmasked_smooth = _separable_conv2d(x, k1d, r) / border_norm

    smoothed = unmasked_smooth
    if mask_aware and patch_mask is not None:
        if isinstance(patch_mask, np.ndarray):
            mask_t = torch.from_numpy(patch_mask.astype(np.float32, copy=False))
        else:
            mask_t = patch_mask.detach().to(dtype=torch.float32)
        if tuple(mask_t.shape) == (grid_h, grid_w):
            mask_4d = mask_t.to(device=feats.device, dtype=work_dtype).view(
                1, 1, grid_h, grid_w
            )
            if float(mask_4d.max().item()) > 0.0:
                mask_weight = _separable_conv2d(mask_4d, k1d, r)
                fg_num = _separable_conv2d(x * mask_4d, k1d, r)
                fg_smooth = fg_num / mask_weight.clamp_min(1e-6)
                has_fg_support = mask_weight > 1e-4
                smoothed = torch.where(has_fg_support, fg_smooth, unmasked_smooth)

    smoothed_3d = smoothed.squeeze(0).to(dtype=feats.dtype)
    if normalize:
        smoothed_3d = normalize_feature_grid(smoothed_3d)
    if alpha < 1.0:
        base = normalize_feature_grid(feats) if normalize else feats
        mixed = (1.0 - alpha) * base + alpha * smoothed_3d
        return normalize_feature_grid(mixed) if normalize else mixed
    return smoothed_3d


def pixel_to_continuous_patch(
    x: float,
    y: float,
    *,
    roi_offset: Tuple[float, float],
    roi_size: Tuple[int, int],
    scale_x: float,
    scale_y: float,
    patch_size: float,
    grid_hw: Tuple[int, int],
) -> Tuple[float, float]:
    """Map image pixel ``(x, y)`` to continuous ``(row, col)`` patch coordinates.

    Integer coordinate ``(r, c)`` corresponds to the geometric center of patch
    ``(r, c)``, i.e. ``((r + 0.5) * patch_size, (c + 0.5) * patch_size)`` in
    resized ROI pixels.
    """
    grid_h, grid_w = grid_hw
    local_x = float(x) - float(roi_offset[0])
    local_y = float(y) - float(roi_offset[1])
    roi_w, roi_h = roi_size
    if roi_w > 0:
        local_x = min(max(local_x, 0.0), float(roi_w - 1))
    if roi_h > 0:
        local_y = min(max(local_y, 0.0), float(roi_h - 1))
    ps = max(1e-6, float(patch_size))
    cont_c = (local_x * float(scale_x) / ps) - 0.5
    cont_r = (local_y * float(scale_y) / ps) - 0.5
    cont_c = min(max(cont_c, 0.0), float(max(0, grid_w - 1)))
    cont_r = min(max(cont_r, 0.0), float(max(0, grid_h - 1)))
    return (cont_r, cont_c)


def sample_subpatch_gaussian_descriptor(
    feats: torch.Tensor,
    *,
    cont_rc: Tuple[float, float],
    radius: int = 1,
    sigma: float = 0.75,
    patch_mask: Optional[np.ndarray] = None,
    mask_aware: bool = True,
) -> torch.Tensor:
    """Sample a continuous sub-patch Gaussian-weighted descriptor from ``[C, H, W]``."""
    channels, grid_h, grid_w = feats.shape
    if grid_h <= 0 or grid_w <= 0 or channels <= 0:
        return torch.zeros((max(1, channels),), device=feats.device, dtype=feats.dtype)

    r_f = float(np.clip(cont_rc[0], 0.0, max(0.0, float(grid_h - 1))))
    c_f = float(np.clip(cont_rc[1], 0.0, max(0.0, float(grid_w - 1))))
    center_r = int(round(r_f))
    center_c = int(round(c_f))
    win_r = max(1, int(radius))
    s = max(1e-4, float(sigma))

    r0 = max(0, center_r - win_r)
    r1 = min(grid_h - 1, center_r + win_r)
    c0 = max(0, center_c - win_r)
    c1 = min(grid_w - 1, center_c + win_r)

    rows = torch.arange(r0, r1 + 1, device=feats.device, dtype=torch.float32) - r_f
    cols = torch.arange(c0, c1 + 1, device=feats.device, dtype=torch.float32) - c_f
    dr, dc = torch.meshgrid(rows, cols, indexing="ij")
    weights = torch.exp(-0.5 * (dr * dr + dc * dc) / (s * s))

    if (
        mask_aware
        and patch_mask is not None
        and tuple(patch_mask.shape) == (grid_h, grid_w)
    ):
        local_mask = patch_mask[r0 : r1 + 1, c0 : c1 + 1]
        if bool(np.any(local_mask)):
            mask_t = torch.from_numpy(local_mask.astype(np.float32, copy=False)).to(
                device=feats.device
            )
            masked_weights = weights * mask_t
            if float(masked_weights.sum().item()) > 1e-8:
                weights = masked_weights

    w_sum = weights.sum().clamp_min(1e-12)
    weights = (weights / w_sum).to(dtype=feats.dtype)
    patch_slice = feats[:, r0 : r1 + 1, c0 : c1 + 1]
    desc = (patch_slice * weights.unsqueeze(0)).sum(dim=(-2, -1)).detach()
    return desc / (desc.norm() + 1e-12)


def _extract_subpixel_patch_with_border(
    gray: np.ndarray,
    center_xy: Tuple[float, float],
    size_hw: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Extract a sub-pixel centered float32 grayscale patch with border replication."""
    if gray is None or gray.ndim != 2 or gray.size == 0:
        return None
    h, w = gray.shape
    cx, cy = float(center_xy[0]), float(center_xy[1])
    if not (math.isfinite(cx) and math.isfinite(cy)):
        return None
    cx = float(np.clip(cx, 0.0, max(0.0, float(w - 1))))
    cy = float(np.clip(cy, 0.0, max(0.0, float(h - 1))))

    patch_h, patch_w = int(size_hw[0]), int(size_hw[1])
    pad = max(patch_h, patch_w) // 2 + 2
    gray_f = gray.astype(np.float32, copy=False)
    if cx - pad < 0 or cy - pad < 0 or cx + pad >= w or cy + pad >= h:
        padded = cv2.copyMakeBorder(
            gray_f, pad, pad, pad, pad, borderType=cv2.BORDER_REPLICATE
        )
        return cv2.getRectSubPix(padded, (patch_w, patch_h), (cx + pad, cy + pad))
    return cv2.getRectSubPix(gray_f, (patch_w, patch_h), (cx, cy))


def _sobel_gradient_xy_interior(patch_padded: np.ndarray) -> np.ndarray:
    """Compute interior ``(2, H-2, W-2)`` signed Sobel ``(gx, gy)`` from a 1px-padded patch."""
    gx = cv2.Sobel(patch_padded, cv2.CV_32F, 1, 0, ksize=3)[1:-1, 1:-1]
    gy = cv2.Sobel(patch_padded, cv2.CV_32F, 0, 1, ksize=3)[1:-1, 1:-1]
    return np.stack([gx, gy], axis=0).astype(np.float32, copy=False)


def extract_gaussian_pixel_template(
    gray: Optional[np.ndarray],
    xy: Tuple[float, float],
    *,
    window_px: int = 11,
    sigma_px: float = 3.0,
    min_std: float = 1.5,
) -> Optional[GaussianPixelTemplate]:
    """Extract a Gaussian-weighted zero-mean unit-norm pixel/gradient anchor template.

    Returns ``None`` on uniform/blank patches so synthetic uniform frames are
    untouched while textured anatomical landmarks get a drift-free pixel anchor.
    """
    if gray is None or gray.ndim != 2:
        return None
    win = max(5, int(window_px))
    if win % 2 == 0:
        win += 1
    radius = win // 2
    sigma = max(0.75, float(sigma_px))

    padded_patch = _extract_subpixel_patch_with_border(gray, xy, (win + 2, win + 2))
    if padded_patch is None:
        return None

    intensity = torch.from_numpy(padded_patch[1:-1, 1:-1].copy())
    weights_2d = gaussian_kernel_2d(radius, sigma, dtype=torch.float32)

    mean_i = float((weights_2d * intensity).sum().item())
    centered_i = intensity - mean_i
    norm_i = float(torch.sqrt((weights_2d * centered_i * centered_i).sum()).item())
    if not math.isfinite(norm_i) or norm_i < float(min_std):
        return None

    norm_intensity_kernel = ((weights_2d * centered_i) / norm_i).view(1, 1, win, win)

    grad_xy_t = torch.from_numpy(_sobel_gradient_xy_interior(padded_patch))
    norm_g = float(
        torch.sqrt((weights_2d.unsqueeze(0) * grad_xy_t * grad_xy_t).sum()).item()
    )
    norm_grad_kernel: Optional[torch.Tensor] = None
    if math.isfinite(norm_g) and norm_g >= max(1.0, 0.5 * float(min_std)):
        norm_grad_kernel = ((weights_2d.unsqueeze(0) * grad_xy_t) / norm_g).unsqueeze(0)

    return GaussianPixelTemplate(
        radius_px=radius,
        sigma_px=sigma,
        weights=weights_2d.view(1, 1, win, win),
        norm_intensity_kernel=norm_intensity_kernel,
        norm_grad_kernel=norm_grad_kernel,
        min_std=float(min_std),
    )


def _weighted_zncc_surface(
    roi_4d: torch.Tensor,
    norm_template_kernel: torch.Tensor,
    weights_4d: torch.Tensor,
    min_std: float,
) -> torch.Tensor:
    """Vectorized Gaussian-weighted ZNCC surface over all shifts in ``roi_4d``."""
    numerator = F.conv2d(roi_4d, norm_template_kernel)
    local_mean = F.conv2d(roi_4d, weights_4d)
    local_mean_sq = F.conv2d(roi_4d * roi_4d, weights_4d)
    local_var = (local_mean_sq - local_mean * local_mean).clamp_min(0.0)
    min_var = float(min_std * min_std)
    denom = torch.sqrt(local_var.clamp_min(1e-8))
    zncc = (numerator / denom).clamp(-1.0, 1.0)
    zncc = torch.where(local_var >= min_var, zncc, torch.zeros_like(zncc))
    return zncc.squeeze(0).squeeze(0)


def _weighted_directional_grad_surface(
    grad_xy_roi_4d: torch.Tensor,
    norm_grad_kernel: torch.Tensor,
    weights_4d: torch.Tensor,
    min_grad_norm: float,
) -> torch.Tensor:
    """Vectorized Gaussian-weighted cosine similarity on signed ``(gx, gy)``."""
    numerator = F.conv2d(grad_xy_roi_4d, norm_grad_kernel)
    grad_sq = (grad_xy_roi_4d * grad_xy_roi_4d).sum(dim=1, keepdim=True)
    local_energy = F.conv2d(grad_sq, weights_4d).clamp_min(0.0)
    min_energy = float(min_grad_norm * min_grad_norm)
    denom = torch.sqrt(local_energy.clamp_min(1e-8))
    cos_sim = (numerator / denom).clamp(-1.0, 1.0)
    cos_sim = torch.where(
        local_energy >= min_energy, cos_sim, torch.zeros_like(cos_sim)
    )
    return cos_sim.squeeze(0).squeeze(0)


def match_gaussian_pixel_template(
    gray: Optional[np.ndarray],
    template: Optional[GaussianPixelTemplate],
    *,
    center_xy: Tuple[float, float],
    search_radius_px: int = 4,
    spatial_prior_sigma_px: Optional[float] = None,
) -> Optional[Tuple[float, float, float]]:
    """Locate ``template`` around ``center_xy`` with sub-pixel parabolic refinement.

    Returns ``(refined_x, refined_y, peak_zncc)`` or ``None`` when no valid
    textured response exists in the search window.
    """
    if gray is None or template is None:
        return None
    search_r = max(1, int(search_radius_px))
    win = 2 * int(template.radius_px) + 1
    roi_inner = win + 2 * search_r
    padded_roi = _extract_subpixel_patch_with_border(
        gray, center_xy, (roi_inner + 2, roi_inner + 2)
    )
    if padded_roi is None:
        return None

    intensity_roi = torch.from_numpy(padded_roi[1:-1, 1:-1].copy()).view(
        1, 1, roi_inner, roi_inner
    )
    zncc_surface = _weighted_zncc_surface(
        intensity_roi,
        template.norm_intensity_kernel,
        template.weights,
        template.min_std,
    )
    if template.norm_grad_kernel is not None:
        grad_roi = torch.from_numpy(_sobel_gradient_xy_interior(padded_roi)).unsqueeze(
            0
        )
        zncc_grad = _weighted_directional_grad_surface(
            grad_roi,
            template.norm_grad_kernel,
            template.weights,
            max(1.0, 0.5 * template.min_std),
        )
        zncc_surface = 0.70 * zncc_surface + 0.30 * zncc_grad

    surf_np = zncc_surface.numpy()
    if surf_np.size == 0 or not np.isfinite(surf_np).any():
        return None

    prior_sigma = (
        float(spatial_prior_sigma_px)
        if spatial_prior_sigma_px is not None
        else max(2.0, float(search_r) * 1.5)
    )
    offsets = np.arange(-search_r, search_r + 1, dtype=np.float32)
    dy_grid, dx_grid = np.meshgrid(offsets, offsets, indexing="ij")
    spatial_prior = np.exp(
        -0.5 * (dx_grid * dx_grid + dy_grid * dy_grid) / (prior_sigma * prior_sigma)
    )
    # Soft spatial tie-breaker keeps periodic textures anchored near the coarse prior.
    guided_surface = surf_np - 0.06 * (1.0 - spatial_prior)

    flat_idx = int(np.argmax(guided_surface))
    side = 2 * search_r + 1
    best_r = flat_idx // side
    best_c = flat_idx % side
    raw_peak_zncc = float(surf_np[best_r, best_c])
    if not math.isfinite(raw_peak_zncc) or raw_peak_zncc <= 0.0:
        return None

    sub_dc = 0.0
    sub_dr = 0.0
    solved_2d = False
    if 0 < best_c < side - 1 and 0 < best_r < side - 1:
        s00 = float(surf_np[best_r, best_c])
        s_xm = float(surf_np[best_r, best_c - 1])
        s_xp = float(surf_np[best_r, best_c + 1])
        s_ym = float(surf_np[best_r - 1, best_c])
        s_yp = float(surf_np[best_r + 1, best_c])
        s_mm = float(surf_np[best_r - 1, best_c - 1])
        s_mp = float(surf_np[best_r - 1, best_c + 1])
        s_pm = float(surf_np[best_r + 1, best_c - 1])
        s_pp = float(surf_np[best_r + 1, best_c + 1])

        gx = 0.5 * (s_xp - s_xm)
        gy = 0.5 * (s_yp - s_ym)
        hxx = s_xp - 2.0 * s00 + s_xm
        hyy = s_yp - 2.0 * s00 + s_ym
        hxy = 0.25 * (s_pp - s_mp - s_pm + s_mm)
        det = hxx * hyy - hxy * hxy
        if hxx < -1e-5 and hyy < -1e-5 and det > 1e-8:
            raw_dc = -(hyy * gx - hxy * gy) / det
            raw_dr = -(-hxy * gx + hxx * gy) / det
            if abs(raw_dc) <= 0.75 and abs(raw_dr) <= 0.75:
                sub_dc = float(np.clip(raw_dc, -0.6, 0.6))
                sub_dr = float(np.clip(raw_dr, -0.6, 0.6))
                solved_2d = True

    if not solved_2d:
        if 0 < best_c < side - 1:
            s_left = float(surf_np[best_r, best_c - 1])
            s_mid = float(surf_np[best_r, best_c])
            s_right = float(surf_np[best_r, best_c + 1])
            denom_x = s_left - 2.0 * s_mid + s_right
            if denom_x < -1e-5:
                sub_dc = float(np.clip(0.5 * (s_left - s_right) / denom_x, -0.5, 0.5))

        if 0 < best_r < side - 1:
            s_up = float(surf_np[best_r - 1, best_c])
            s_mid = float(surf_np[best_r, best_c])
            s_down = float(surf_np[best_r + 1, best_c])
            denom_y = s_up - 2.0 * s_mid + s_down
            if denom_y < -1e-5:
                sub_dr = float(np.clip(0.5 * (s_up - s_down) / denom_y, -0.5, 0.5))

    dx = float(best_c - search_r) + sub_dc
    dy = float(best_r - search_r) + sub_dr
    h, w = gray.shape
    refined_x = float(np.clip(float(center_xy[0]) + dx, 0.0, max(0.0, float(w - 1))))
    refined_y = float(np.clip(float(center_xy[1]) + dy, 0.0, max(0.0, float(h - 1))))
    return (refined_x, refined_y, raw_peak_zncc)


def refine_patch_region_gaussian(
    logits: np.ndarray,
    *,
    center_rc: Tuple[int, int],
    r_min: int,
    c_min: int,
    valid_mask: Optional[np.ndarray],
    radius: int,
    sigma: float,
    temperature: float,
    patch_centers_x: np.ndarray,
    patch_centers_y: np.ndarray,
    fallback_xy: Tuple[float, float],
) -> Tuple[float, float, float]:
    """Vectorized Gaussian-softmax centroid over a local patch similarity window."""
    r = int(radius)
    if r <= 0 or logits.size == 0:
        return (float(fallback_xy[0]), float(fallback_xy[1]), 0.0)

    local_r = int(center_rc[0] - r_min)
    local_c = int(center_rc[1] - c_min)
    reg_h, reg_w = int(logits.shape[0]), int(logits.shape[1])
    if local_r < 0 or local_c < 0 or local_r >= reg_h or local_c >= reg_w:
        return (float(fallback_xy[0]), float(fallback_xy[1]), 0.0)

    r_start = max(0, local_r - r)
    r_end = min(reg_h - 1, local_r + r)
    c_start = max(0, local_c - r)
    c_end = min(reg_w - 1, local_c + r)

    sub_logits = logits[r_start : r_end + 1, c_start : c_end + 1].astype(
        np.float64, copy=False
    )
    if valid_mask is not None:
        sub_valid = valid_mask[r_start : r_end + 1, c_start : c_end + 1].astype(
            bool, copy=False
        )
        if not np.any(sub_valid):
            return (float(fallback_xy[0]), float(fallback_xy[1]), 0.0)
    else:
        sub_valid = None

    global_rows = np.arange(r_min + r_start, r_min + r_end + 1, dtype=np.int64)
    global_cols = np.arange(c_min + c_start, c_min + c_end + 1, dtype=np.int64)
    dr = (global_rows - int(center_rc[0])).astype(np.float64)[:, None]
    dc = (global_cols - int(center_rc[1])).astype(np.float64)[None, :]

    inv_sigma_sq = 1.0 / (max(1e-4, float(sigma)) ** 2)
    temp = max(1e-4, float(temperature))
    spatial = np.exp(-0.5 * (dr * dr + dc * dc) * inv_sigma_sq)

    max_sim = (
        float(np.max(sub_logits[sub_valid]))
        if sub_valid is not None
        else float(np.max(sub_logits))
    )
    score_w = np.exp((sub_logits - max_sim) / temp)
    weights = spatial * score_w
    if sub_valid is not None:
        weights = np.where(sub_valid, weights, 0.0)

    total = float(np.sum(weights))
    if not math.isfinite(total) or total <= 1e-12:
        return (float(fallback_xy[0]), float(fallback_xy[1]), 0.0)

    xs = patch_centers_x[global_cols].astype(np.float64)[None, :]
    ys = patch_centers_y[global_rows].astype(np.float64)[:, None]
    sum_x = float(np.sum(weights * xs))
    sum_y = float(np.sum(weights * ys))
    max_weight = float(np.max(weights))
    confidence = float(np.clip(max_weight / total, 0.0, 1.0))
    return (sum_x / total, sum_y / total, confidence)
