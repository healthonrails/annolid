"""Tests for OKS-based AP computation — validates that gt-absent entries
don't pollute AP and that the relaxed sigma produces non-zero AP for
reasonable predictions.
"""

from __future__ import annotations

import math

import pytest
import torch

from annolid.segmentation.dino_kpseg.train import (
    _average_precision,
    _oks_from_distance,
)


# ── _oks_from_distance ──────────────────────────────────────────────


def test_oks_perfect_match() -> None:
    """Zero distance → OKS = 1.0."""
    dist = torch.tensor([0.0])
    oks = _oks_from_distance(dist, sigma_px=14.0)
    assert oks.item() == pytest.approx(1.0, abs=1e-6)


def test_oks_moderate_distance() -> None:
    """Distance equal to sigma → OKS = exp(-0.5) ≈ 0.6065."""
    sigma = 28.0
    dist = torch.tensor([sigma])
    oks = _oks_from_distance(dist, sigma_px=sigma)
    expected = math.exp(-0.5)
    assert oks.item() == pytest.approx(expected, abs=1e-4)


def test_oks_large_distance_becomes_small() -> None:
    """Distance = 3*sigma → OKS ≈ 0.011, very small."""
    sigma = 14.0
    dist = torch.tensor([sigma * 3.0])
    oks = _oks_from_distance(dist, sigma_px=sigma)
    assert oks.item() < 0.02


def test_oks_doubled_sigma_more_lenient() -> None:
    """Doubling sigma makes OKS much more lenient for same distance."""
    dist = torch.tensor([20.0])
    oks_strict = _oks_from_distance(dist, sigma_px=14.0)
    oks_lenient = _oks_from_distance(dist, sigma_px=28.0)
    assert oks_lenient.item() > oks_strict.item()
    # At 20px with sigma=28, OKS should be well above 0.5
    assert oks_lenient.item() > 0.5


# ── _average_precision with gt-present filtering ────────────────────


def test_ap_perfect_detections() -> None:
    """All detections are TP → AP = 1.0."""
    scores = [0.9, 0.8, 0.7]
    tp_flags = [True, True, True]
    ap = _average_precision(scores, tp_flags, num_gt=3)
    assert ap == pytest.approx(1.0, abs=1e-4)


def test_ap_all_fp() -> None:
    """All detections are FP → AP = 0.0."""
    scores = [0.9, 0.8, 0.7]
    tp_flags = [False, False, False]
    ap = _average_precision(scores, tp_flags, num_gt=3)
    assert ap == pytest.approx(0.0, abs=1e-6)


def test_ap_not_polluted_by_gt_absent_phantom_fps() -> None:
    """Key test: when gt-absent entries are properly filtered OUT,
    AP should reflect only the real detections.

    Before the fix, gt-absent entries with conf>0 and OKS=0 would be
    included.  When the phantom FPs have higher confidence than real TPs
    (realistic in early training), they push down precision at high recall.
    """
    # Simulate the BROKEN case: 5 entries total — 3 real TPs interleaved
    # with 2 gt-absent phantoms that have HIGHER confidence (typical when
    # the model confidently predicts on patches without GT annotations).
    scores_broken = [0.95, 0.85, 0.8, 0.7, 0.6]
    # phantoms at positions 0 and 1 (highest conf) → oks=0 → FP
    tp_broken = [False, False, True, True, True]
    ap_broken = _average_precision(scores_broken, tp_broken, num_gt=3)

    # NEW (fixed): only 3 entries with GT present
    scores_fixed = [0.8, 0.7, 0.6]
    tp_fixed = [True, True, True]
    ap_fixed = _average_precision(scores_fixed, tp_fixed, num_gt=3)

    # The fixed version should have perfect AP
    assert ap_fixed == pytest.approx(1.0, abs=1e-4)
    # The broken version has AP < 1.0 due to phantom FPs ranked higher
    assert ap_broken < ap_fixed


def test_ap_with_mixed_results_after_filtering() -> None:
    """After filtering, some detections are TP and some are FP.
    AP should still be non-zero unlike the all-zeros we saw."""
    scores = [0.9, 0.7, 0.5, 0.3]
    tp_flags = [True, False, True, False]
    ap = _average_precision(scores, tp_flags, num_gt=3)
    # With 2 TP out of 3 GT, AP should be non-zero
    assert ap > 0.0


# ── Integration: OKS thresholding for AP ────────────────────────────


def test_oks_based_tp_flags_with_doubled_sigma() -> None:
    """Simulate the training loop's AP computation with doubled sigma.
    Even moderate prediction errors should produce non-zero OKS-based AP."""
    patch_size = 14
    # sigma_px = patch_size * 2.0 (the fix)
    sigma = float(patch_size) * 2.0

    # Predictions with moderate errors: 10px, 15px, 25px
    distances = torch.tensor([10.0, 15.0, 25.0])
    oks_vals = _oks_from_distance(distances, sigma_px=sigma)

    # At IoU threshold 0.50, check how many are TP
    tp_at_50 = [float(v) >= 0.50 for v in oks_vals.tolist()]
    # All three should pass at 0.50 with sigma=28
    assert all(tp_at_50), (
        f"Expected all TP at IoU=0.50, got {tp_at_50}, OKS={oks_vals.tolist()}"
    )

    scores = [0.8, 0.6, 0.4]
    ap = _average_precision(scores, tp_at_50, num_gt=3)
    assert ap == pytest.approx(1.0, abs=1e-4)
