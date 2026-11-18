"""Identity-safety checks for multi-instance mask tracking."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from annolid.utils.body_part_shapes import split_body_part_csv_instance_name
from annolid.utils.mask_geometry import (
    BBoxXYXY,
    compute_mask_centroid_and_area,
    tight_mask_bbox,
)


def _semantic_track_role(label: str) -> str:
    """Group labels so whole instances only permute with whole instances and parts with matching parts."""
    _parent, part = split_body_part_csv_instance_name(label)
    if part:
        part_base = part.split("#", 1)[0].strip().lower()
        return f"part:{part_base or part.lower()}"
    return "__whole__"


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
    bboxes: dict[str, BBoxXYXY | None] = {
        label: tight_mask_bbox(normalized[label]) for label in labels
    }
    overlaps: list[MaskOverlap] = []
    for left_index, left_label in enumerate(labels):
        left_bbox = bboxes[left_label]
        if left_bbox is None:
            continue
        lx0, ly0, lx1, ly1 = left_bbox
        left_mask = normalized[left_label]
        for right_label in labels[left_index + 1 :]:
            right_bbox = bboxes[right_label]
            if right_bbox is None:
                continue
            rx0, ry0, rx1, ry1 = right_bbox
            ix0 = max(lx0, rx0)
            iy0 = max(ly0, ry0)
            ix1 = min(lx1, rx1)
            iy1 = min(ly1, ry1)
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            pixel_count = int(
                np.count_nonzero(
                    left_mask[iy0:iy1, ix0:ix1]
                    & normalized[right_label][iy0:iy1, ix0:ix1]
                )
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
    stats = compute_mask_centroid_and_area(mask)
    if stats is None:
        return None
    return stats[0], stats[1]


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

    prev_arr = np.asarray(previous_points, dtype=np.float64)
    curr_arr = np.asarray(current_points, dtype=np.float64)

    role_groups: dict[str, list[int]] = {}
    for idx, label in enumerate(labels):
        role_groups.setdefault(_semantic_track_role(label), []).append(idx)

    if min_absolute_improvement_px is None:
        absolute_threshold = max(8.0, math.hypot(width, height) * 0.01)
    else:
        absolute_threshold = max(0.0, float(min_absolute_improvement_px))
    rel_threshold = max(0.0, float(min_relative_improvement))

    changed_list: list[tuple[str, str]] = []
    total_baseline = 0.0
    total_alternative = 0.0

    for group_indices in role_groups.values():
        if len(group_indices) < 2:
            continue
        g_idx = np.asarray(group_indices, dtype=np.intp)
        p_sub = prev_arr[g_idx]
        c_sub = curr_arr[g_idx]
        sub_cost = np.hypot(
            p_sub[:, None, 0] - c_sub[None, :, 0],
            p_sub[:, None, 1] - c_sub[None, :, 1],
        )
        r_ind, c_ind = linear_sum_assignment(sub_cost)
        if len(r_ind) != len(group_indices):
            continue
        group_changes = [
            (labels[group_indices[int(c_i)]], labels[group_indices[int(r_i)]])
            for r_i, c_i in zip(r_ind, c_ind)
            if int(r_i) != int(c_i)
        ]
        if not group_changes:
            continue
        g_base = float(np.trace(sub_cost))
        g_alt = float(sub_cost[r_ind, c_ind].sum())
        g_imp = g_base - g_alt
        if g_imp <= 0.0:
            continue
        g_rel = g_imp / max(g_base, 1e-9)
        if g_imp < absolute_threshold or g_rel < rel_threshold:
            continue
        changed_list.extend(group_changes)
        total_baseline += g_base
        total_alternative += g_alt

    if not changed_list:
        return None

    improvement = total_baseline - total_alternative
    relative_improvement = improvement / max(total_baseline, 1e-9)
    return IdentityAssignmentAmbiguity(
        changed_assignments=tuple(sorted(changed_list)),
        baseline_cost_px=total_baseline,
        alternative_cost_px=total_alternative,
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
