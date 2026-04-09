from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from typing import Sequence

import numpy as np
from qtpy import QtCore


def _shape_point_list(shape) -> list[QtCore.QPointF]:
    points = []
    for point in list(getattr(shape, "points", []) or []):
        try:
            points.append(QtCore.QPointF(point))
        except Exception:
            try:
                points.append(QtCore.QPointF(float(point[0]), float(point[1])))
            except Exception:
                continue
    return points


def _shape_centroid(shape) -> tuple[float, float]:
    points = _shape_point_list(shape)
    if not points:
        return (0.0, 0.0)
    xs = [float(point.x()) for point in points]
    ys = [float(point.y()) for point in points]
    return (sum(xs) / len(xs), sum(ys) / len(ys))


def polygon_edit_state(shape) -> dict:
    other = dict(getattr(shape, "other_data", {}) or {})
    state = other.get("polygon_edit", {})
    if isinstance(state, dict):
        return dict(state)
    return {}


def set_polygon_edit_state(shape, state: str, **metadata) -> bool:
    if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
        return False
    other = dict(getattr(shape, "other_data", {}) or {})
    edit_state = dict(other.get("polygon_edit") or {})
    normalized_state = str(state or "manual").strip().lower() or "manual"
    edit_state["state"] = normalized_state
    for key, value in metadata.items():
        if value is None:
            edit_state.pop(key, None)
        else:
            edit_state[key] = value
    if normalized_state in {"manual", "restored"}:
        for key in (
            "collapsed_points",
            "collapsed_shared_vertex_ids",
            "collapsed_shared_edge_ids",
            "source_pages",
            "interpolation_ratio",
        ):
            edit_state.pop(key, None)
    other["polygon_edit"] = edit_state
    shape.other_data = other
    return True


def is_collapsed_polygon(shape) -> bool:
    if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
        return False
    edit_state = polygon_edit_state(shape)
    return str(edit_state.get("state") or "").lower() == "collapsed"


def is_inferred_polygon(shape) -> bool:
    if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
        return False
    edit_state = polygon_edit_state(shape)
    return str(edit_state.get("state") or "").lower() == "inferred"


def polygon_identity_key(shape) -> tuple[str, str, str]:
    label = str(getattr(shape, "label", "") or "").strip().lower()
    group_id = str(getattr(shape, "group_id", "") or "").strip()
    description = str(getattr(shape, "description", "") or "").strip().lower()
    return (label, group_id, description)


def is_inferable_polygon(shape) -> bool:
    if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
        return False
    if not bool(getattr(shape, "visible", True)):
        return False
    if is_collapsed_polygon(shape):
        return False
    if is_inferred_polygon(shape):
        return False
    return len(_shape_point_list(shape)) >= 3


def _points_to_array(points: Sequence[QtCore.QPointF]) -> np.ndarray:
    arr = np.asarray([[float(p.x()), float(p.y())] for p in points], dtype=np.float64)
    return arr


def _resample_closed_polygon(
    points: Sequence[QtCore.QPointF], count: int
) -> np.ndarray:
    count = max(3, int(count or 0))
    pts = _points_to_array(points)
    if pts.shape[0] == 0:
        return np.zeros((count, 2), dtype=np.float64)
    if pts.shape[0] == 1:
        return np.repeat(pts, count, axis=0)
    if pts.shape[0] == 2:
        pts = np.vstack([pts, pts[-1]])
    closed = np.vstack([pts, pts[0]])
    edges = np.diff(closed, axis=0)
    lengths = np.linalg.norm(edges, axis=1)
    perimeter = float(np.sum(lengths))
    if perimeter <= 1e-12:
        return np.repeat(pts[:1], count, axis=0)
    target = np.linspace(0.0, perimeter, count, endpoint=False)
    cumulative = np.insert(np.cumsum(lengths), 0, 0.0)
    resampled = []
    for distance in target:
        idx = int(np.searchsorted(cumulative, distance, side="right") - 1)
        idx = max(0, min(idx, len(edges) - 1))
        segment_length = float(lengths[idx])
        if segment_length <= 1e-12:
            resampled.append(pts[idx].copy())
            continue
        ratio = float((distance - cumulative[idx]) / segment_length)
        ratio = max(0.0, min(1.0, ratio))
        point = pts[idx] + (edges[idx] * ratio)
        resampled.append(point)
    return np.asarray(resampled, dtype=np.float64)


def _best_cyclic_alignment(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError("Point sequences must have the same length.")
    if reference.shape[0] == 0:
        return candidate
    best = candidate
    best_score = None
    for flipped in (candidate, candidate[::-1]):
        for shift in range(candidate.shape[0]):
            rolled = np.roll(flipped, -shift, axis=0)
            score = float(np.sum((reference - rolled) ** 2))
            if best_score is None or score < best_score:
                best_score = score
                best = rolled
    return best


def interpolate_closed_polygon_points(
    points_a: Sequence[QtCore.QPointF],
    points_b: Sequence[QtCore.QPointF],
    ratio: float,
    *,
    point_count: int | None = None,
) -> list[QtCore.QPointF]:
    count = max(
        3,
        int(point_count or 0) or max(len(points_a or []), len(points_b or []), 3),
    )
    resampled_a = _resample_closed_polygon(points_a, count)
    resampled_b = _resample_closed_polygon(points_b, count)
    aligned_b = _best_cyclic_alignment(resampled_a, resampled_b)
    blended = (resampled_a * (1.0 - float(ratio))) + (aligned_b * float(ratio))
    return [QtCore.QPointF(float(x), float(y)) for x, y in blended]


def _clone_polygon_shape(
    source_shape,
    points: Sequence[QtCore.QPointF],
    *,
    state: str,
    source_pages: Sequence[int] | None = None,
    interpolation_ratio: float | None = None,
):
    from annolid.gui.shape import Shape

    clone = Shape(
        label=getattr(source_shape, "label", None),
        line_color=getattr(source_shape, "line_color", None),
        shape_type="polygon",
        flags=deepcopy(getattr(source_shape, "flags", None)),
        group_id=getattr(source_shape, "group_id", None),
        description=getattr(source_shape, "description", None),
        visible=bool(getattr(source_shape, "visible", True)),
    )
    clone.other_data = deepcopy(dict(getattr(source_shape, "other_data", {}) or {}))
    clone.fill = bool(getattr(source_shape, "fill", False))
    clone.selected = False
    clone.points = [QtCore.QPointF(point) for point in points]
    clone.point_labels = [1] * len(clone.points)
    clone.close()
    set_polygon_edit_state(
        clone,
        str(state or "inferred"),
        source_pages=[int(page) for page in list(source_pages or [])],
        interpolation_ratio=float(interpolation_ratio)
        if interpolation_ratio is not None
        else None,
    )
    if str(state or "").lower() == "collapsed":
        clone.visible = False
    return clone


def _polygon_groups(shapes: Sequence) -> dict[tuple[str, str, str], list]:
    groups: dict[tuple[str, str, str], list] = defaultdict(list)
    for shape in list(shapes or []):
        if not is_inferable_polygon(shape):
            continue
        groups[polygon_identity_key(shape)].append(shape)
    for key in list(groups.keys()):
        groups[key].sort(key=lambda shape: _shape_centroid(shape))
    return groups


def infer_polygon_shapes_between_pages(
    previous_shapes: Sequence | None,
    next_shapes: Sequence | None,
    *,
    target_page: int,
    previous_page: int | None = None,
    next_page: int | None = None,
) -> list:
    prev_groups = _polygon_groups(previous_shapes)
    next_groups = _polygon_groups(next_shapes)
    if not prev_groups and not next_groups:
        return []

    result = []
    all_keys = sorted(set(prev_groups.keys()) | set(next_groups.keys()))
    if (
        previous_page is not None
        and next_page is not None
        and next_page != previous_page
    ):
        ratio = (float(target_page) - float(previous_page)) / float(
            next_page - previous_page
        )
    else:
        ratio = 0.5
    ratio = max(0.0, min(1.0, float(ratio)))

    for key in all_keys:
        prev_shapes_for_key = list(prev_groups.get(key, []))
        next_shapes_for_key = list(next_groups.get(key, []))
        if prev_shapes_for_key and next_shapes_for_key:
            pair_count = max(len(prev_shapes_for_key), len(next_shapes_for_key))
            for index in range(pair_count):
                prev_shape = prev_shapes_for_key[
                    min(index, len(prev_shapes_for_key) - 1)
                ]
                next_shape = next_shapes_for_key[
                    min(index, len(next_shapes_for_key) - 1)
                ]
                prev_points = _shape_point_list(prev_shape)
                next_points = _shape_point_list(next_shape)
                if len(prev_points) < 3 and len(next_points) < 3:
                    continue
                if len(prev_points) < 3:
                    points = [QtCore.QPointF(point) for point in next_points]
                    source = next_shape
                elif len(next_points) < 3:
                    points = [QtCore.QPointF(point) for point in prev_points]
                    source = prev_shape
                else:
                    points = interpolate_closed_polygon_points(
                        prev_points,
                        next_points,
                        ratio,
                    )
                    source = prev_shape
                inferred = _clone_polygon_shape(
                    source,
                    points,
                    state="inferred",
                    source_pages=[
                        p for p in (previous_page, next_page) if p is not None
                    ],
                    interpolation_ratio=ratio,
                )
                result.append(inferred)
            continue
        source_shapes = prev_shapes_for_key or next_shapes_for_key
        source = source_shapes[0]
        points = _shape_point_list(source)
        if len(points) < 3:
            continue
        inferred = _clone_polygon_shape(
            source,
            points,
            state="inferred",
            source_pages=[p for p in (previous_page, next_page) if p is not None],
            interpolation_ratio=ratio,
        )
        result.append(inferred)
    return result


def collapse_polygon_shape(shape) -> bool:
    if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
        return False
    if len(_shape_point_list(shape)) < 3:
        return False
    edit_state = dict(polygon_edit_state(shape))
    if edit_state.get("state") == "collapsed":
        return True
    edit_state.pop("state", None)
    edit_state["collapsed_points"] = [
        [float(point.x()), float(point.y())] for point in _shape_point_list(shape)
    ]
    edit_state["collapsed_shared_vertex_ids"] = list(
        getattr(shape, "shared_vertex_ids", []) or []
    )
    edit_state["collapsed_shared_edge_ids"] = list(
        getattr(shape, "shared_edge_ids", []) or []
    )
    set_polygon_edit_state(shape, "collapsed", **edit_state)
    shape.visible = False
    return True


def restore_polygon_shape(shape) -> bool:
    edit_state = polygon_edit_state(shape)
    if str(edit_state.get("state") or "").lower() != "collapsed":
        return False
    set_polygon_edit_state(shape, "manual")
    shape.visible = True
    return True


def mark_polygon_shape_manual(shape) -> bool:
    return set_polygon_edit_state(shape, "manual")
