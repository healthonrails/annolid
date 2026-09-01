from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from shapely.geometry import Polygon

from annolid.annotation.polygon_constraints import (
    make_instance_masks_exclusive,
    resolve_polygon_shape_conflicts,
)


def _polygon(shape: SimpleNamespace) -> Polygon:
    return Polygon(shape.points)


def test_resolve_polygon_shape_conflicts_removes_touching_geometry() -> None:
    first = SimpleNamespace(
        shape_type="polygon",
        points=[[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]],
    )
    second = SimpleNamespace(
        shape_type="polygon",
        points=[[4.0, 4.0], [8.0, 4.0], [8.0, 8.0], [4.0, 8.0]],
    )

    resolution = resolve_polygon_shape_conflicts([first, second])

    assert resolution.shapes == (first, second)
    assert resolution.adjusted_shape_indices == (1,)
    assert resolution.dropped_shape_indices == ()
    assert first.points == [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]
    assert set(map(tuple, first.points)).isdisjoint(set(map(tuple, second.points)))
    assert _polygon(first).disjoint(_polygon(second))


def test_resolve_polygon_shape_conflicts_honors_priority() -> None:
    lower_confidence = SimpleNamespace(
        shape_type="polygon",
        points=[[0.0, 0.0], [6.0, 0.0], [6.0, 6.0], [0.0, 6.0]],
    )
    higher_confidence = SimpleNamespace(
        shape_type="polygon",
        points=[[4.0, 2.0], [8.0, 2.0], [8.0, 5.0], [4.0, 5.0]],
    )
    higher_points = [point[:] for point in higher_confidence.points]

    resolution = resolve_polygon_shape_conflicts(
        [lower_confidence, higher_confidence],
        priorities=[0.2, 0.9],
    )

    assert resolution.adjusted_shape_indices == (0,)
    assert resolution.dropped_shape_indices == ()
    assert higher_confidence.points == higher_points
    assert _polygon(lower_confidence).disjoint(_polygon(higher_confidence))


def test_resolve_polygon_shape_conflicts_reports_fully_occluded_shape() -> None:
    first = SimpleNamespace(
        shape_type="polygon",
        points=[[0.0, 0.0], [5.0, 0.0], [5.0, 5.0], [0.0, 5.0]],
    )
    duplicate = SimpleNamespace(
        shape_type="polygon",
        points=[point[:] for point in first.points],
    )

    resolution = resolve_polygon_shape_conflicts([first, duplicate])

    assert resolution.shapes == (first,)
    assert resolution.adjusted_shape_indices == ()
    assert resolution.dropped_shape_indices == (1,)


def test_resolve_polygon_shape_conflicts_preserves_contained_owner() -> None:
    surrounding = SimpleNamespace(
        shape_type="polygon",
        points=[[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
    )
    contained_owner = SimpleNamespace(
        shape_type="polygon",
        points=[[4.0, 4.0], [6.0, 4.0], [6.0, 6.0], [4.0, 6.0]],
    )

    resolution = resolve_polygon_shape_conflicts(
        [surrounding, contained_owner],
        priorities=[0.1, 0.9],
    )

    surrounding_geometry = _polygon(surrounding)
    contained_geometry = _polygon(contained_owner)
    assert resolution.adjusted_shape_indices == (0,)
    assert not surrounding_geometry.interiors
    assert surrounding_geometry.is_valid
    assert surrounding_geometry.disjoint(contained_geometry)


def test_resolve_polygon_shape_conflicts_normalizes_zero_area_spike() -> None:
    spiked = SimpleNamespace(
        shape_type="polygon",
        points=[
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [3.0, 5.0],
            [4.0, 4.0],
            [0.0, 4.0],
        ],
    )
    original_area = _polygon(spiked).area

    resolution = resolve_polygon_shape_conflicts([spiked])

    normalized = _polygon(spiked)
    assert resolution.adjusted_shape_indices == (0,)
    assert normalized.is_valid
    assert normalized.area == original_area
    assert len(spiked.points) == 4


def test_make_instance_masks_exclusive_prefers_higher_confidence() -> None:
    first = np.zeros((6, 6), dtype=bool)
    second = np.zeros((6, 6), dtype=bool)
    first[1:4, 1:4] = True
    second[3:5, 3:5] = True

    first_out, second_out = make_instance_masks_exclusive(
        [first, second],
        scores=[0.7, 0.9],
    )

    assert not np.any(first_out & second_out)
    assert second_out[3, 3]
    assert not first_out[3, 3]
    assert np.array_equal(first_out | second_out, first | second)


def test_make_instance_masks_exclusive_rejects_misaligned_scores() -> None:
    masks = [np.zeros((2, 2), dtype=bool), np.zeros((2, 2), dtype=bool)]

    with pytest.raises(ValueError, match="Expected 2 priorities, received 1"):
        make_instance_masks_exclusive(masks, scores=[0.5])
