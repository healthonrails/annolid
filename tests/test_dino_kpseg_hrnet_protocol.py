from __future__ import annotations

import math

import pytest

from annolid.segmentation.dino_kpseg.eval import default_oks_sigmas, oks_from_error
from annolid.segmentation.dino_kpseg.train import _resolve_hrnet_protocol_params


def test_resolve_hrnet_protocol_defaults_for_conv_head() -> None:
    lr, epochs, freeze_bn = _resolve_hrnet_protocol_params(
        head_type_norm="conv",
        num_train_images=256,
        requested_lr=3e-4,
        requested_epochs=120,
        freeze_bn=None,
    )
    assert lr == pytest.approx(5e-4)
    assert epochs == 210
    assert freeze_bn is False


def test_resolve_hrnet_protocol_small_data_enables_freeze_bn_and_lower_lr() -> None:
    lr, epochs, freeze_bn = _resolve_hrnet_protocol_params(
        head_type_norm="conv",
        num_train_images=32,
        requested_lr=5e-4,
        requested_epochs=210,
        freeze_bn=None,
    )
    assert freeze_bn is True
    assert lr == pytest.approx(5e-5)
    assert epochs == 210


def test_default_oks_sigmas_prefers_nose_value() -> None:
    sigmas = default_oks_sigmas(
        kpt_count=3,
        keypoint_names=["nose_tip", "left_ear", "right_ear"],
    )
    assert sigmas[0] == pytest.approx(0.026)
    assert sigmas[1] == pytest.approx(0.067)
    assert sigmas[2] == pytest.approx(0.067)


def test_oks_from_error_matches_formula() -> None:
    value = oks_from_error(error_px=10.0, scale_px=100.0, sigma=0.067)
    expected = math.exp(-(10.0**2) / (2.0 * (100.0**2) * (0.067**2)))
    assert value == pytest.approx(expected)
