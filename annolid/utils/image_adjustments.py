from __future__ import annotations

from typing import Any, Tuple

import numpy as np


def normalize_brightness_contrast_value(value: Any) -> int:
    """Normalize GUI/runtime brightness/contrast values to [-100, 100]."""
    try:
        parsed = int(round(float(value)))
    except (TypeError, ValueError):
        parsed = 0
    return max(-100, min(100, parsed))


def compute_brightness_contrast_linear_transform(
    brightness: Any,
    contrast: Any,
) -> Tuple[int, int, float, float, bool]:
    """Return normalized values + linear transform params for uint8 images.

    Pixel transform:
      out = clip(alpha * in + beta, 0, 255)
    """
    normalized_brightness = normalize_brightness_contrast_value(brightness)
    normalized_contrast = normalize_brightness_contrast_value(contrast)
    brightness_factor = max(0.0, 1.0 + (float(normalized_brightness) / 100.0))
    contrast_factor = max(0.0, 1.0 + (float(normalized_contrast) / 100.0))
    alpha = float(brightness_factor * contrast_factor)
    beta = float(127.5 * (1.0 - contrast_factor))
    enabled = bool(alpha != 1.0 or beta != 0.0)
    return normalized_brightness, normalized_contrast, alpha, beta, enabled


def apply_linear_intensity_transform_uint8(
    image: np.ndarray | None,
    *,
    alpha: float,
    beta: float,
    enabled: bool,
) -> np.ndarray | None:
    """Apply precomputed brightness/contrast transform on uint8-like arrays."""
    if image is None:
        return None
    if not enabled:
        return image

    adjusted = image.astype(np.float32) * float(alpha) + float(beta)
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def apply_brightness_contrast_uint8(
    image: np.ndarray | None,
    *,
    brightness: Any,
    contrast: Any,
) -> np.ndarray | None:
    """Convenience wrapper that computes and applies linear transform."""
    _, _, alpha, beta, enabled = compute_brightness_contrast_linear_transform(
        brightness,
        contrast,
    )
    return apply_linear_intensity_transform_uint8(
        image, alpha=alpha, beta=beta, enabled=enabled
    )
