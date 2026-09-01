from __future__ import annotations

import numpy as np

from annolid.tracking.identity_continuity import (
    detect_centroid_assignment_ambiguity,
    detect_identity_assignment_ambiguity,
    find_cross_instance_mask_overlaps,
)


def _square_mask(center_x: int, center_y: int) -> np.ndarray:
    mask = np.zeros((120, 120), dtype=bool)
    mask[center_y - 3 : center_y + 4, center_x - 3 : center_x + 4] = True
    return mask


def test_find_cross_instance_mask_overlaps_reports_exact_pair() -> None:
    first = np.zeros((8, 8), dtype=bool)
    second = np.zeros((8, 8), dtype=bool)
    third = np.zeros((8, 8), dtype=bool)
    first[1:5, 1:5] = True
    second[4:7, 4:7] = True
    third[6:8, 0:2] = True

    overlaps = find_cross_instance_mask_overlaps(
        {"fish_0": first, "fish_1": second, "fish_2": third}
    )

    assert len(overlaps) == 1
    assert overlaps[0].first_label == "fish_0"
    assert overlaps[0].second_label == "fish_1"
    assert overlaps[0].pixel_count == 1


def test_detect_identity_assignment_ambiguity_detects_strong_permutation() -> None:
    previous = {
        "fish_3": _square_mask(25, 60),
        "fish_4": _square_mask(95, 60),
    }
    current = {
        "fish_3": _square_mask(90, 60),
        "fish_4": _square_mask(30, 60),
    }

    ambiguity = detect_identity_assignment_ambiguity(previous, current)

    assert ambiguity is not None
    assert ambiguity.changed_assignments == (
        ("fish_3", "fish_4"),
        ("fish_4", "fish_3"),
    )
    assert ambiguity.baseline_cost_px == 130.0
    assert ambiguity.alternative_cost_px == 10.0
    assert ambiguity.relative_improvement > 0.9


def test_detect_identity_assignment_ambiguity_ignores_normal_motion() -> None:
    previous = {
        "fish_3": _square_mask(25, 60),
        "fish_4": _square_mask(95, 60),
    }
    current = {
        "fish_3": _square_mask(30, 60),
        "fish_4": _square_mask(90, 60),
    }

    assert detect_identity_assignment_ambiguity(previous, current) is None


def test_detect_identity_assignment_ambiguity_ignores_marginal_alternative() -> None:
    previous = {
        "fish_3": _square_mask(50, 60),
        "fish_4": _square_mask(70, 60),
    }
    current = {
        "fish_3": _square_mask(61, 60),
        "fish_4": _square_mask(59, 60),
    }

    assert detect_identity_assignment_ambiguity(previous, current) is None


def test_centroid_detector_catches_fish_crossing_regression() -> None:
    previous = {
        "fish_3": (577.488, 177.665),
        "fish_4": (574.932, 224.490),
    }
    current = {
        "fish_3": (573.110, 228.487),
        "fish_4": (578.762, 197.322),
    }

    ambiguity = detect_centroid_assignment_ambiguity(
        previous,
        current,
        frame_shape=(720, 1280),
    )

    assert ambiguity is not None
    assert ambiguity.changed_assignments == (
        ("fish_3", "fish_4"),
        ("fish_4", "fish_3"),
    )
    assert ambiguity.relative_improvement > 0.65
