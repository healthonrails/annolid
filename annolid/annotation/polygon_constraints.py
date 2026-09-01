"""Deterministic ownership constraints for generated instance geometry."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from shapely import make_valid
from shapely.geometry import GeometryCollection, LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import nearest_points


# LabelMe polygons have no explicit topology or shared-edge ownership. A small
# clearance makes that ownership unambiguous while remaining well below one
# raster pixel, minimizing its effect when the polygon is rasterized.
DEFAULT_POLYGON_CLEARANCE_PX = 1e-3


@dataclass(frozen=True)
class PolygonConflictResolution:
    """Result of enforcing exclusive ownership across a shape batch."""

    shapes: tuple[object, ...]
    adjusted_shape_indices: tuple[int, ...]
    dropped_shape_indices: tuple[int, ...]


def _normalize_priorities(
    count: int,
    priorities: Sequence[float] | None,
) -> list[float]:
    if priorities is not None:
        try:
            priority_count = len(priorities)
        except TypeError as exc:
            raise ValueError("Polygon priorities must be a sized sequence.") from exc
        if priority_count != count:
            raise ValueError(f"Expected {count} priorities, received {priority_count}.")

    normalized: list[float] = []
    for index in range(count):
        try:
            value = float(priorities[index]) if priorities is not None else 0.0
        except (TypeError, ValueError):
            value = 0.0
        normalized.append(value if math.isfinite(value) else 0.0)
    return normalized


def make_instance_masks_exclusive(
    masks: Sequence[np.ndarray],
    scores: Sequence[float] | None = None,
) -> list[np.ndarray]:
    """Assign every foreground pixel to at most one instance.

    Text-prompt segmentation predicts each detection independently, so two masks
    can claim the same pixels. Higher-confidence instances win those pixels;
    stable input order breaks ties. Returned masks preserve input order and never
    share foreground pixels.
    """

    normalized = [np.asarray(mask).astype(bool) for mask in masks]
    if not normalized:
        return []

    expected_shape = normalized[0].shape
    if len(expected_shape) != 2:
        raise ValueError(f"Instance masks must be 2D, got {expected_shape!r}.")
    for mask in normalized[1:]:
        if mask.shape != expected_shape:
            raise ValueError(
                "Instance masks must share one shape, got "
                f"{expected_shape!r} and {mask.shape!r}."
            )

    normalized_scores = _normalize_priorities(len(normalized), scores)
    priority_order = sorted(
        range(len(normalized)),
        key=lambda index: (-normalized_scores[index], index),
    )
    claimed = np.zeros(expected_shape, dtype=bool)
    exclusive = [np.zeros(expected_shape, dtype=bool) for _ in normalized]
    for index in priority_order:
        exclusive[index] = normalized[index] & ~claimed
        claimed |= normalized[index]
    return exclusive


def _polygon_from_shape(shape: object) -> tuple[Polygon | None, bool, bool]:
    """Return normalized geometry, whether it is a polygon, and whether it changed."""

    shape_type = str(getattr(shape, "shape_type", "polygon") or "polygon")
    points = getattr(shape, "points", None)
    if shape_type != "polygon":
        return None, False, False
    if not isinstance(points, list) or len(points) < 3:
        return None, True, False

    coordinates: list[tuple[float, float]] = []
    try:
        for point in points:
            if hasattr(point, "x") and hasattr(point, "y"):
                coordinates.append((float(point.x()), float(point.y())))
            else:
                coordinates.append((float(point[0]), float(point[1])))
    except (IndexError, TypeError, ValueError):
        return None, True, False

    if not all(math.isfinite(value) for point in coordinates for value in point):
        return None, True, False
    original = Polygon(coordinates)
    polygon = _largest_polygon(make_valid(original))
    return polygon, True, not original.is_valid


def _largest_polygon(geometry: BaseGeometry) -> Polygon | None:
    """Return the largest polygonal component supported by LabelMe shapes."""

    if geometry.is_empty:
        return None
    if isinstance(geometry, Polygon):
        return geometry if geometry.area > 0 else None
    if isinstance(geometry, MultiPolygon):
        candidates = list(geometry.geoms)
    elif isinstance(geometry, GeometryCollection):
        candidates = [
            part for part in geometry.geoms if isinstance(part, (Polygon, MultiPolygon))
        ]
        candidates = [
            polygon
            for part in candidates
            for polygon in (part.geoms if isinstance(part, MultiPolygon) else [part])
        ]
    else:
        return None
    return max(candidates, key=lambda polygon: polygon.area, default=None)


def _replace_shape_points(shape: object, polygon: Polygon) -> None:
    old_points = getattr(shape, "points")
    coordinates = list(polygon.exterior.coords)[:-1]
    template = old_points[0]

    if hasattr(template, "x") and hasattr(template, "y"):
        point_type = type(template)
        new_points = [point_type(float(x), float(y)) for x, y in coordinates]
    elif isinstance(template, tuple):
        new_points = [(float(x), float(y)) for x, y in coordinates]
    else:
        new_points = [[float(x), float(y)] for x, y in coordinates]

    shape.points = new_points
    if hasattr(shape, "point_labels"):
        shape.point_labels = [1] * len(new_points)
    if hasattr(shape, "point_shared_ids"):
        shape.point_shared_ids = [None] * len(new_points)


def _open_polygon_interiors(
    polygon: Polygon,
    clearance: float,
) -> Polygon | None:
    """Convert holes to narrow exterior channels for a single-ring format.

    LabelMe polygons serialize one exterior ring and cannot preserve interior
    rings. Clipping a lower-priority polygon around a contained owner can create
    such a ring. Opening each hole to the exterior retains a single valid polygon
    without silently filling the claimed region again during serialization.
    """

    resolved: Polygon | None = polygon
    remaining_holes = len(polygon.interiors)
    while resolved is not None and resolved.interiors:
        if remaining_holes <= 0:  # Defensive guard against invalid topology.
            return None
        exterior_point, interior_point = nearest_points(
            LineString(resolved.exterior.coords),
            LineString(resolved.interiors[0].coords),
        )
        channel = LineString([exterior_point, interior_point]).buffer(
            clearance,
            cap_style="square",
            join_style="mitre",
        )
        resolved = _largest_polygon(make_valid(resolved.difference(channel)))
        remaining_holes -= 1
    return resolved


def resolve_polygon_shape_conflicts(
    shapes: Sequence[object],
    priorities: Sequence[float] | None = None,
    *,
    clearance_px: float = DEFAULT_POLYGON_CLEARANCE_PX,
) -> PolygonConflictResolution:
    """Make generated polygon shapes geometrically disjoint.

    Shapes are processed from highest to lowest priority, with stable input order
    breaking ties. Each polygon owns its accepted geometry; later polygons are
    clipped against that geometry plus a sub-pixel clearance. This resolves both
    area overlap and shared edges or vertices using a deterministic set operation,
    rather than perturbing individual coordinates.

    Non-polygon shapes pass through unchanged. Invalid polygons are normalized;
    malformed, degenerate, or fully occluded polygons are omitted and their
    original indices are reported. When a difference creates multiple components,
    only the largest is retained. Any interior ring is opened to the exterior with
    a narrow channel because the current LabelMe shape contract stores one simple
    ring per instance.
    """

    clearance = float(clearance_px)
    if not math.isfinite(clearance) or clearance <= 0:
        raise ValueError("Polygon clearance must be a finite positive value.")

    shape_list = list(shapes)
    normalized_priorities = _normalize_priorities(len(shape_list), priorities)
    priority_order = sorted(
        range(len(shape_list)),
        key=lambda index: (-normalized_priorities[index], index),
    )

    accepted_by_index: dict[int, object] = {}
    claimed: BaseGeometry = GeometryCollection()
    adjusted: list[int] = []
    dropped: list[int] = []

    for index in priority_order:
        shape = shape_list[index]
        polygon, is_polygon, needs_replacement = _polygon_from_shape(shape)
        if polygon is None:
            if is_polygon:
                dropped.append(index)
            else:
                accepted_by_index[index] = shape
            continue

        resolved = polygon
        if resolved.interiors:
            resolved = _open_polygon_interiors(resolved, clearance)
            needs_replacement = True
            if resolved is None:
                dropped.append(index)
                continue
        if not claimed.is_empty and not resolved.disjoint(claimed):
            exclusion = claimed.buffer(
                clearance,
                quad_segs=1,
                join_style="mitre",
            )
            resolved = _largest_polygon(make_valid(resolved.difference(exclusion)))
            if resolved is not None and resolved.interiors:
                resolved = _open_polygon_interiors(resolved, clearance)
            if resolved is None:
                dropped.append(index)
                continue
            needs_replacement = True

        if needs_replacement:
            _replace_shape_points(shape, resolved)
            adjusted.append(index)

        accepted_by_index[index] = shape
        claimed = resolved if claimed.is_empty else claimed.union(resolved)

    return PolygonConflictResolution(
        shapes=tuple(
            accepted_by_index[index]
            for index in range(len(shape_list))
            if index in accepted_by_index
        ),
        adjusted_shape_indices=tuple(sorted(adjusted)),
        dropped_shape_indices=tuple(sorted(dropped)),
    )
