#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from annolid.utils.annotation_store import (
    AnnotationStore,
    AnnotationStoreError,
    load_labelme_json,
)
from annolid.utils.logger import logger

Point = Sequence[float]
Polygon = Sequence[Point]

__all__ = [
    "load_annotation",
    "resample_polygon",
    "flatten_points",
    "polygon_area",
    "polygon_centroid",
    "polygon_perimeter",
    "normalize_polygon",
    "behavior_label",
    "list_polygons",
    "polygons_by_label",
    "frame_number_from_filename",
    "Point",
    "Polygon",
]


def load_annotation(json_path: Union[str, Path]) -> Optional[Dict[str, object]]:
    """Load a LabelMe-style annotation with annotation store support."""
    path = Path(json_path)
    try:
        return load_labelme_json(path)
    except (
        AnnotationStoreError,
        FileNotFoundError,
        json.JSONDecodeError,
        ValueError,
    ) as exc:
        logger.error("Failed to load annotation %s: %s", path, exc)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Unexpected error loading %s: %s", path, exc)
    return None


def resample_polygon(
    points: Polygon, num_points: int, fill_value: float = 0.0
) -> List[List[float]]:
    """Resample a polygon to exactly ``num_points`` vertices using linear interpolation."""
    if num_points <= 0:
        return []
    if not points:
        return [[fill_value, fill_value]] * num_points

    pts = np.asarray(points, dtype=float)
    if len(pts) == 1:
        return np.repeat(pts, num_points, axis=0).tolist()

    distances = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cumulative = np.insert(np.cumsum(distances), 0, 0.0)
    total_length = cumulative[-1]
    if total_length == 0:
        return np.repeat(pts[:1], num_points, axis=0).tolist()

    target_distances = np.linspace(0, total_length, num_points)
    resampled: List[List[float]] = []
    for distance in target_distances:
        idx = np.searchsorted(cumulative, distance)
        if idx == 0 or cumulative[idx] == distance:
            resampled.append(pts[idx].tolist())
            continue

        denom = cumulative[idx] - cumulative[idx - 1]
        ratio = (distance - cumulative[idx - 1]) / denom if denom else 0.0
        interpolated = (1 - ratio) * pts[idx - 1] + ratio * pts[idx]
        resampled.append(interpolated.tolist())
    return resampled


def flatten_points(points: Iterable[Iterable[float]]) -> List[float]:
    """Flatten a list of point pairs into a single list."""
    return [coord for point in points for coord in point]


def polygon_area(points: Polygon) -> float:
    """Compute polygon area via the shoelace formula."""
    if len(points) < 3:
        return 0.0
    pts = np.asarray(points, dtype=float)
    x = pts[:, 0]
    y = pts[:, 1]
    return 0.5 * float(np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def polygon_centroid(points: Polygon) -> List[float]:
    """Return centroid of polygon vertices."""
    if not points:
        return [0.0, 0.0]
    pts = np.asarray(points, dtype=float)
    return np.mean(pts, axis=0).tolist()


def polygon_perimeter(points: Polygon) -> float:
    """Compute polygon perimeter (sum of consecutive segment lengths)."""
    if len(points) < 2:
        return 0.0
    pts = np.asarray(points, dtype=float)
    return float(np.sum(np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))))


def normalize_polygon(points: Polygon) -> List[List[float]]:
    """Center polygon points around the origin to reduce translation variance."""
    if not points:
        return []
    pts = np.asarray(points, dtype=float)
    centroid = np.mean(pts, axis=0)
    return (pts - centroid).tolist()


def behavior_label(flags: Dict[str, object], *, context: str = "") -> Optional[str]:
    """Return the first truthy flag as the behavior label, logging when missing or ambiguous."""
    true_flags = [key for key, value in flags.items() if bool(value)]
    if not true_flags:
        if context:
            logger.warning("No behavior flag found for %s", context)
        return None
    if len(true_flags) > 1 and context:
        logger.warning(
            "Multiple behavior flags %s for %s; using %s",
            true_flags,
            context,
            true_flags[0],
        )
    return true_flags[0]


def list_polygons(
    shapes: Iterable[Dict[str, object]],
    *,
    max_polygons: Optional[int] = None,
    context: str = "",
) -> List[List[List[float]]]:
    """Return polygons in the order they appear, optionally limited."""
    polygons: List[List[List[float]]] = []
    for shape in shapes:
        if shape.get("shape_type") != "polygon":
            continue
        polygons.append(shape.get("points") or [])
        if max_polygons and len(polygons) >= max_polygons:
            break

    if not polygons and context:
        logger.warning("No polygons found for %s", context)
    return polygons


def polygons_by_label(
    shapes: Iterable[Dict[str, object]],
    labels: Sequence[str],
    *,
    context: str = "",
) -> Dict[str, List[List[float]]]:
    """Return a mapping of label -> polygon points, defaulting to empty lists when missing."""
    found: Dict[str, Optional[List[List[float]]]] = {label: None for label in labels}
    for shape in shapes:
        if shape.get("shape_type") != "polygon":
            continue
        label = shape.get("label")
        if label not in found:
            continue
        points = shape.get("points") or []
        if found[label] is None:
            found[label] = points
        elif context:
            logger.warning(
                "Multiple '%s' polygons found for %s; using the first one",
                label,
                context,
            )

    result: Dict[str, List[List[float]]] = {}
    for label in labels:
        if found[label] is None and context:
            logger.warning("Polygon '%s' missing for %s", label, context)
        result[label] = found[label] or []
    return result


def frame_number_from_filename(path: Union[str, Path]) -> Optional[int]:
    """Extract frame number from a LabelMe-style filename using the annotation store helper."""
    try:
        return AnnotationStore.frame_number_from_path(path)
    except Exception:
        return None
