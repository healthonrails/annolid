# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Visualization utilities for point tracking."""

import colorsys
import os
from typing import List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch


# Bremm 2D colormap for position-based coloring
# This creates a smooth 2D color gradient based on x,y position
BREMM_COLORMAP = None  # Lazy loaded


def _create_bremm_colormap():
    """Create a 2D colormap programmatically (Bremm-style).

    This creates a smooth 2D color gradient where:
    - X position maps to hue variation
    - Y position maps to saturation/value variation
    """
    size = 256
    colormap = np.zeros((size, size, 3), dtype=np.uint8)

    for y in range(size):
        for x in range(size):
            # Normalize to [0, 1]
            nx = x / (size - 1)
            ny = y / (size - 1)

            # Create a 2D color mapping using HSV
            # Hue varies with x, saturation/value with y
            hue = (nx * 0.8 + ny * 0.2) % 1.0  # Mix of x and y for hue
            saturation = 0.6 + 0.4 * (1 - ny)  # Higher saturation at top
            value = 0.7 + 0.3 * nx  # Higher value on right

            # Convert HSV to RGB
            rgb = colorsys.hsv_to_rgb(hue, saturation, value)
            colormap[y, x] = [int(c * 255) for c in rgb]

    return colormap


def _get_bremm_colormap():
    """Get or create the bremm colormap."""
    global BREMM_COLORMAP
    if BREMM_COLORMAP is None:
        # Try to load from file first
        colormap_file = os.path.join(os.path.dirname(__file__), "bremm.png")
        if os.path.exists(colormap_file):
            BREMM_COLORMAP = (plt.imread(colormap_file) * 255).astype(np.uint8)
            if BREMM_COLORMAP.shape[2] == 4:  # RGBA
                BREMM_COLORMAP = BREMM_COLORMAP[:, :, :3]
        else:
            BREMM_COLORMAP = _create_bremm_colormap()
    return BREMM_COLORMAP


def get_2d_colors(xys: np.ndarray, H: int, W: int) -> np.ndarray:
    """Get colors based on 2D position using Bremm colormap.

    This creates position-dependent colors where nearby points have
    similar colors, useful for visualizing spatial coherence.

    Args:
        xys: Point coordinates [N, 2] in pixel space (x, y)
        H: Image height
        W: Image width

    Returns:
        Array of RGB colors [N, 3] as uint8
    """
    colormap = _get_bremm_colormap()
    height, width = colormap.shape[:2]

    N = xys.shape[0]
    output = np.zeros((N, 3), dtype=np.uint8)

    # Normalize coordinates to [0, 1]
    xys_norm = xys.copy().astype(np.float32)
    xys_norm[:, 0] = xys_norm[:, 0] / max(W - 1, 1)
    xys_norm[:, 1] = xys_norm[:, 1] / max(H - 1, 1)

    # Clip to valid range
    xys_norm = np.clip(xys_norm, 0, 1)

    # Map to colormap coordinates
    for i in range(N):
        x, y = xys_norm[i]
        xp = int((width - 1) * x)
        yp = int((height - 1) * y)
        output[i] = colormap[yp, xp]

    return output


def get_colors_from_cmap(num_colors: int, cmap: str = "gist_rainbow") -> np.ndarray:
    """Gets colormap for points using matplotlib colormap.

    Args:
        num_colors: Number of colors to generate
        cmap: Matplotlib colormap name (e.g., "gist_rainbow", "jet", "turbo")

    Returns:
        Array of RGB colors [num_colors, 3] as uint8
    """
    cmap_ = matplotlib.colormaps.get_cmap(cmap)
    colors = []
    for i in range(num_colors):
        c = cmap_(i / float(num_colors))
        colors.append((int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)))
    return np.array(colors)


def paint_point_track(
    frames: np.ndarray,
    point_tracks: np.ndarray,
    visibility: np.ndarray,
    colormap: Optional[Union[List[Tuple[int, int, int]], np.ndarray]] = None,
    rate: int = 1,
    show_bkg: bool = True,
) -> np.ndarray:
    """Paint point tracks on video frames using GPU-accelerated scatter.

    Args:
        frames: Video frames [T, H, W, C] in uint8
        point_tracks: Track coordinates [P, T, 2] (x, y)
        visibility: Visibility mask [P, T]
        colormap: Optional list/array of RGB colors for each point
        rate: Subsampling rate for visualization (affects point size)
        show_bkg: Whether to show background (True) or black out (False)

    Returns:
        Painted frames [T, H, W, C] in uint8
    """
    print("Starting visualization...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    frames_t = (
        torch.from_numpy(frames).float().permute(0, 3, 1, 2).to(device)
    )  # [T,C,H,W]

    if show_bkg:
        frames_t = frames_t * 0.5  # darken to see tracks better
    else:
        frames_t = frames_t * 0.0  # black out background

    point_tracks_t = torch.from_numpy(point_tracks).to(device)  # [P,T,2]
    visibility_t = torch.from_numpy(visibility).to(device)  # [P,T]
    T, C, H, W = frames_t.shape
    P = point_tracks.shape[0]

    # Use gist_rainbow colormap (matching app3.py behavior)
    if colormap is None:
        colormap = get_colors_from_cmap(P, "gist_rainbow")
    colors = torch.tensor(colormap, dtype=torch.float32, device=device)  # [P,3]

    # Adjust radius based on rate
    if rate == 1:
        radius = 1
    elif rate == 2:
        radius = 1
    elif rate == 4:
        radius = 2
    elif rate == 8:
        radius = 4
    else:
        radius = 6

    sharpness = 0.15 + 0.05 * np.log2(rate)

    D = radius * 2 + 1
    y = torch.arange(D, device=device).float()[:, None] - radius
    x = torch.arange(D, device=device).float()[None, :] - radius
    dist2 = x**2 + y**2
    icon = torch.clamp(1 - (dist2 - (radius**2) / 2.0) / (radius * 2 * sharpness), 0, 1)
    icon = icon.view(1, D, D)
    dx = torch.arange(-radius, radius + 1, device=device)
    dy = torch.arange(-radius, radius + 1, device=device)
    disp_y, disp_x = torch.meshgrid(dy, dx, indexing="ij")

    for t in range(T):
        mask = visibility_t[:, t]
        if mask.sum() == 0:
            continue
        xy = point_tracks_t[mask, t] + 0.5
        xy[:, 0] = xy[:, 0].clamp(0, W - 1)
        xy[:, 1] = xy[:, 1].clamp(0, H - 1)
        colors_now = colors[mask]
        N = xy.shape[0]
        cx = xy[:, 0].long()
        cy = xy[:, 1].long()
        x_grid = cx[:, None, None] + disp_x
        y_grid = cy[:, None, None] + disp_y
        valid = (x_grid >= 0) & (x_grid < W) & (y_grid >= 0) & (y_grid < H)
        x_valid = x_grid[valid]
        y_valid = y_grid[valid]
        icon_weights = icon.expand(N, D, D)[valid]
        colors_valid = (
            colors_now[:, :, None, None]
            .expand(N, 3, D, D)
            .permute(1, 0, 2, 3)[:, valid]
        )
        idx_flat = (y_valid * W + x_valid).long()

        accum = torch.zeros_like(frames_t[t])
        weight = torch.zeros(1, H * W, device=device)
        img_flat = accum.view(C, -1)
        weighted_colors = colors_valid * icon_weights
        img_flat.scatter_add_(1, idx_flat.unsqueeze(0).expand(C, -1), weighted_colors)
        weight.scatter_add_(1, idx_flat.unsqueeze(0), icon_weights.unsqueeze(0))
        weight = weight.view(1, H, W)

        alpha = weight.clamp(0, 1)
        accum = accum / (weight + 1e-6)
        frames_t[t] = frames_t[t] * (1 - alpha) + accum * alpha

    print("Visualization done.")
    return frames_t.clamp(0, 255).byte().permute(0, 2, 3, 1).cpu().numpy()
