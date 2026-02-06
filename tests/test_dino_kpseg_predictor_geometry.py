from __future__ import annotations

import numpy as np
import pytest

from annolid.segmentation.dino_kpseg.predictor import DinoKPSEGPredictor


def test_resized_to_original_xy_clips_to_frame_bounds() -> None:
    x, y = DinoKPSEGPredictor._resized_to_original_xy(
        999.0,
        -20.0,
        resized_h=64,
        resized_w=64,
        frame_h=32,
        frame_w=48,
    )
    assert x == pytest.approx(47.0)
    assert y == pytest.approx(0.0)


def test_mask_to_uint8_normalizes_float_mask() -> None:
    mask = np.array([[0.0, 0.5], [1.0, 2.0]], dtype=np.float32)
    out = DinoKPSEGPredictor._mask_to_uint8(mask, frame_shape=(2, 2))
    assert out.dtype == np.uint8
    assert int(out[0, 0]) == 0
    assert int(out[0, 1]) == 127
    assert int(out[1, 0]) == 255
    assert int(out[1, 1]) == 255


def test_mask_to_uint8_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        _ = DinoKPSEGPredictor._mask_to_uint8(
            np.zeros((2, 3), dtype=np.uint8),
            frame_shape=(3, 2),
        )
