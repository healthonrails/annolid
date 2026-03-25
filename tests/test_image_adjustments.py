from __future__ import annotations

import numpy as np

from annolid.utils.image_adjustments import (
    apply_brightness_contrast_uint8,
    apply_linear_intensity_transform_uint8,
    compute_brightness_contrast_linear_transform,
    normalize_brightness_contrast_value,
)


def test_normalize_brightness_contrast_value_clamps_and_defaults() -> None:
    assert normalize_brightness_contrast_value(None) == 0
    assert normalize_brightness_contrast_value("bad") == 0
    assert normalize_brightness_contrast_value(120) == 100
    assert normalize_brightness_contrast_value(-150) == -100


def test_compute_linear_transform_returns_expected_identity() -> None:
    b, c, alpha, beta, enabled = compute_brightness_contrast_linear_transform(0, 0)
    assert b == 0
    assert c == 0
    assert alpha == 1.0
    assert beta == 0.0
    assert enabled is False


def test_apply_linear_transform_uint8_noop_returns_same_reference() -> None:
    image = np.full((3, 3, 3), 77, dtype=np.uint8)
    output = apply_linear_intensity_transform_uint8(
        image,
        alpha=1.0,
        beta=0.0,
        enabled=False,
    )
    assert output is image


def test_apply_brightness_contrast_uint8_adjusts_pixels() -> None:
    image = np.full((2, 2, 3), 80, dtype=np.uint8)
    output = apply_brightness_contrast_uint8(
        image,
        brightness=25,
        contrast=20,
    )
    assert output is not None
    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert float(output.mean()) > float(image.mean())
