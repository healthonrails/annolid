"""Identity-safety checks for multi-instance mask tracking."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class MaskOverlap:
    """Cross-instance pixel ownership conflict."""

    first_label: str
    second_label: str
    pixel_count: int


@dataclass(frozen=True)
class IdentityAssignmentAmbiguity:
    """A substantially better non-identity temporal assignment."""

    changed_assignments: tuple[tuple[str, str], ...]
    baseline_cost_px: float
    alternative_cost_px: float
    improvement_px: float
    relative_improvement: float


def find_cross_instance_mask_overlaps(
    masks: Mapping[str, np.ndarray],
) -> tuple[MaskOverlap, ...]:
    """Return every pair of differently labeled masks sharing raster pixels."""

    normalized: dict[str, np.ndarray] = {}
    expected_shape: tuple[int, ...] | None = None
    for label, mask in masks.items():
        mask_bool = np.asarray(mask, dtype=bool)
        if mask_bool.ndim != 2:
            raise ValueError(
                f"Instance mask for {label!r} must be 2D, got {mask_bool.shape!r}."
            )
        if expected_shape is None:
            expected_shape = mask_bool.shape
        elif mask_bool.shape != expected_shape:
            raise ValueError(
                "Instance masks must share one shape, got "
                f"{expected_shape!r} and {mask_bool.shape!r}."
            )
        normalized[str(label)] = mask_bool

    labels = sorted(normalized)
    overlaps: list[MaskOverlap] = []
    for left_index, left_label in enumerate(labels):
        for right_label in labels[left_index + 1 :]:
            pixel_count = int(
                np.count_nonzero(normalized[left_label] & normalized[right_label])
            )
            if pixel_count:
                overlaps.append(
                    MaskOverlap(
                        first_label=left_label,
                        second_label=right_label,
                        pixel_count=pixel_count,
                    )
                )
    return tuple(overlaps)


def _mask_centroid(mask: np.ndarray) -> tuple[float, float] | None:
    rows, columns = np.nonzero(mask)
    if not columns.size:
        return None
    return float(columns.mean()), float(rows.mean())


def detect_centroid_assignment_ambiguity(
    previous_centroids: Mapping[str, tuple[float, float]],
    current_centroids: Mapping[str, tuple[float, float]],
    *,
    frame_shape: tuple[int, int],
    min_relative_improvement: float = 0.35,
    min_absolute_improvement_px: float | None = None,
) -> IdentityAssignmentAmbiguity | None:
    """Detect a materially better non-identity centroid assignment.

    The function deliberately reports ambiguity instead of rewriting labels.
    Position alone cannot prove identity through a complete occlusion, but a
    strong non-identity assignment is sufficient reason to stop persistence and
    request review.
    """

    previous_by_label = {
        str(label): centroid for label, centroid in previous_centroids.items()
    }
    current_by_label = {
        str(label): centroid for label, centroid in current_centroids.items()
    }
    labels = sorted(previous_by_label)
    if len(labels) < 2 or set(labels) != set(current_by_label):
        return None

    height, width = (int(frame_shape[0]), int(frame_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"Frame shape must be positive, got {frame_shape!r}.")

    previous_points: list[tuple[float, float]] = []
    current_points: list[tuple[float, float]] = []
    for label in labels:
        previous_point = tuple(float(value) for value in previous_by_label[label])
        current_point = tuple(float(value) for value in current_by_label[label])
        if len(previous_point) != 2 or len(current_point) != 2:
            raise ValueError("Identity centroids must contain exactly two values.")
        if not all(math.isfinite(value) for value in previous_point + current_point):
            return None
        previous_points.append(previous_point)
        current_points.append(current_point)

    cost_matrix = np.empty((len(labels), len(labels)), dtype=np.float64)
    for previous_index, previous_centroid in enumerate(previous_points):
        for current_index, current_centroid in enumerate(current_points):
            cost_matrix[previous_index, current_index] = math.dist(
                previous_centroid,
                current_centroid,
            )

    row_indices, column_indices = linear_sum_assignment(cost_matrix)
    if len(row_indices) != len(labels):
        return None

    changed_assignments = tuple(
        sorted(
            (labels[current_index], labels[previous_index])
            for previous_index, current_index in zip(row_indices, column_indices)
            if previous_index != current_index
        )
    )
    if not changed_assignments:
        return None

    baseline_cost = float(np.trace(cost_matrix))
    alternative_cost = float(cost_matrix[row_indices, column_indices].sum())
    improvement = baseline_cost - alternative_cost
    if improvement <= 0:
        return None

    relative_improvement = improvement / max(baseline_cost, 1e-9)
    if min_absolute_improvement_px is None:
        absolute_threshold = max(8.0, math.hypot(width, height) * 0.01)
    else:
        absolute_threshold = max(0.0, float(min_absolute_improvement_px))

    if improvement < absolute_threshold:
        return None
    if relative_improvement < max(0.0, float(min_relative_improvement)):
        return None

    return IdentityAssignmentAmbiguity(
        changed_assignments=changed_assignments,
        baseline_cost_px=baseline_cost,
        alternative_cost_px=alternative_cost,
        improvement_px=improvement,
        relative_improvement=relative_improvement,
    )


def detect_identity_assignment_ambiguity(
    previous_masks: Mapping[str, np.ndarray],
    current_masks: Mapping[str, np.ndarray],
    *,
    min_relative_improvement: float = 0.35,
    min_absolute_improvement_px: float | None = None,
) -> IdentityAssignmentAmbiguity | None:
    """Detect a likely identity permutation without guessing which one is correct.

    CUTIE channels usually preserve identity, but visually similar objects can
    exchange channel content during a direct occlusion. This compares the current
    label assignment with the globally optimal centroid assignment. A result is
    returned only when the alternative is materially better; callers should pause
    for review rather than silently relabel an inherently ambiguous crossing.
    """

    previous_by_label = {str(label): mask for label, mask in previous_masks.items()}
    current_by_label = {str(label): mask for label, mask in current_masks.items()}
    labels = sorted(previous_by_label)
    if len(labels) < 2 or set(labels) != set(current_by_label):
        return None

    previous_centroids: dict[str, tuple[float, float]] = {}
    current_centroids: dict[str, tuple[float, float]] = {}
    expected_shape: tuple[int, int] | None = None
    for label in labels:
        previous = np.asarray(previous_by_label[label], dtype=bool)
        current = np.asarray(current_by_label[label], dtype=bool)
        if previous.ndim != 2 or current.ndim != 2 or previous.shape != current.shape:
            return None
        if expected_shape is None:
            expected_shape = previous.shape
        elif previous.shape != expected_shape:
            return None
        previous_centroid = _mask_centroid(previous)
        current_centroid = _mask_centroid(current)
        if previous_centroid is None or current_centroid is None:
            return None
        previous_centroids[label] = previous_centroid
        current_centroids[label] = current_centroid

    if expected_shape is None:
        return None
    return detect_centroid_assignment_ambiguity(
        previous_centroids,
        current_centroids,
        frame_shape=expected_shape,
        min_relative_improvement=min_relative_improvement,
        min_absolute_improvement_px=min_absolute_improvement_px,
    )
