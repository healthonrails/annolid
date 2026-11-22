"""Lossless geometric union for single-ring vector shapes and mask fragments."""

from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
from qtpy.QtCore import QPointF
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from annolid.gui.shape import Shape

_VECTOR_MERGE_TYPES = frozenset({"polygon", "rectangle", "oriented_rectangle"})


def _normalized_metadata(shape: Shape) -> tuple:
    return (
        shape.label,
        shape.group_id,
        dict(shape.flags or {}),
        shape.description or "",
        dict(shape.other_data or {}),
    )


def _merge_mask_shapes(shapes: list[Shape]) -> Shape:
    """Merge >=2 visible unlocked mask shapes into one tight-cropped mask shape."""
    first = shapes[0]
    first_meta = _normalized_metadata(first)
    placements: list[tuple[int, int, np.ndarray]] = []
    min_x = None
    min_y = None
    max_x = None
    max_y = None

    for shape in shapes:
        if shape.shape_type != "mask" or shape.mask is None:
            raise ValueError("Cannot mix mask shapes with vector shapes in one merge.")
        if not shape.visible:
            raise ValueError("Select only visible shapes to merge.")
        if shape.is_locked():
            raise ValueError("Unlock locked shapes before merging.")
        if _normalized_metadata(shape) != first_meta:
            raise ValueError(
                "Selected shapes must have matching labels, group IDs, flags, "
                "descriptions, and custom metadata. Edit them to match first."
            )
        if not shape.points:
            raise ValueError("Each mask shape must have a valid anchor position.")
        mask_arr = np.asarray(shape.mask)
        if mask_arr.ndim > 2:
            mask_arr = mask_arr[..., 0]
        mask_bool = mask_arr.astype(bool)
        if mask_bool.ndim != 2 or mask_bool.size == 0 or not mask_bool.any():
            raise ValueError("Repair empty mask shapes before merging.")
        x0 = float(shape.points[0].x())
        y0 = float(shape.points[0].y())
        if not (math.isfinite(x0) and math.isfinite(y0)):
            raise ValueError("Shape coordinates must be finite.")
        ix0 = int(round(x0))
        iy0 = int(round(y0))
        mh, mw = mask_bool.shape
        ix1 = ix0 + mw
        iy1 = iy0 + mh
        placements.append((ix0, iy0, mask_bool))
        min_x = ix0 if min_x is None else min(min_x, ix0)
        min_y = iy0 if min_y is None else min(min_y, iy0)
        max_x = ix1 if max_x is None else max(max_x, ix1)
        max_y = iy1 if max_y is None else max(max_y, iy1)

    assert (
        min_x is not None
        and min_y is not None
        and max_x is not None
        and max_y is not None
    )
    canvas_w = max_x - min_x
    canvas_h = max_y - min_y
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("Merged mask has empty bounds.")

    union_mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    for ix0, iy0, mask_bool in placements:
        ox = ix0 - min_x
        oy = iy0 - min_y
        mh, mw = mask_bool.shape
        union_mask[oy : oy + mh, ox : ox + mw] |= mask_bool

    rows = np.where(union_mask.any(axis=1))[0]
    cols = np.where(union_mask.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        raise ValueError("Merged mask is empty.")
    ry0, ry1 = int(rows[0]), int(rows[-1]) + 1
    cx0, cx1 = int(cols[0]), int(cols[-1]) + 1
    tight_mask = np.ascontiguousarray(union_mask[ry0:ry1, cx0:cx1], dtype=bool)
    out_x0 = float(min_x + cx0)
    out_y0 = float(min_y + ry0)
    out_x1 = float(min_x + cx1)
    out_y1 = float(min_y + ry1)

    result = Shape(
        label=first.label,
        shape_type="mask",
        group_id=first.group_id,
        description=first.description,
        mask=tight_mask,
    )
    result.flags = deepcopy(first.flags)
    result.other_data = deepcopy(first.other_data)
    result.line_color = deepcopy(first.line_color)
    result.fill_color = deepcopy(first.fill_color)
    result.points = [QPointF(out_x0, out_y0), QPointF(out_x1, out_y1)]
    result.point_labels = [1, 1]
    result.close()
    return result


def merge_shapes(shapes):
    """Return a merged polygon or mask shape without mutating inputs; reject lossy unions."""
    if len(shapes) < 2:
        raise ValueError("Select at least two polygons, rectangles, or masks to merge.")
    shape_list = list(shapes)
    if all(s.shape_type == "mask" for s in shape_list):
        return _merge_mask_shapes(shape_list)
    if any(s.shape_type == "mask" for s in shape_list):
        raise ValueError("Cannot mix mask shapes with polygon/rectangle shapes.")

    first = shape_list[0]
    first_meta = _normalized_metadata(first)
    geometries = []
    for shape in shape_list:
        if shape.shape_type not in _VECTOR_MERGE_TYPES:
            raise ValueError(
                "Only polygons, rectangles, oriented rectangles, and masks can be merged."
            )
        if not shape.visible or shape.mask is not None:
            raise ValueError("Select visible shapes without attached masks.")
        if shape.is_locked():
            raise ValueError("Unlock locked shapes before merging.")
        if _normalized_metadata(shape) != first_meta:
            raise ValueError(
                "Selected shapes must have matching labels, group IDs, flags, "
                "descriptions, and custom metadata. Edit them to match first."
            )
        if any(label != 1 for label in shape.point_labels):
            raise ValueError(
                "Shapes with vertex labels cannot be merged without losing labels."
            )
        points = [(p.x(), p.y()) for p in shape.points]
        if not all(math.isfinite(v) for point in points for v in point):
            raise ValueError("Shape coordinates must be finite.")
        if shape.shape_type == "rectangle":
            if len(points) != 2:
                raise ValueError("Each rectangle must have two corners.")
            geometry = box(*points[0], *points[1])
        else:
            if len(points) < 3:
                raise ValueError("Each polygon must have at least three vertices.")
            geometry = Polygon(points)
        if not geometry.is_valid or geometry.is_empty or geometry.area <= 0:
            raise ValueError("Repair invalid or zero-area shapes before merging.")
        geometries.append(geometry)
    union = unary_union(geometries)
    if union.geom_type != "Polygon":
        raise ValueError(
            "Shapes must overlap or share an edge to form one connected polygon."
        )
    if union.interiors:
        raise ValueError(
            "The merged shape would contain holes, which a single annotation cannot store."
        )
    result = Shape(
        label=first.label,
        shape_type="polygon",
        group_id=first.group_id,
        description=first.description,
    )
    result.flags = deepcopy(first.flags)
    result.other_data = deepcopy(first.other_data)
    result.line_color = deepcopy(first.line_color)
    result.fill_color = deepcopy(first.fill_color)
    result.points = [QPointF(x, y) for x, y in list(union.exterior.coords)[:-1]]
    result.point_labels = [1] * len(result.points)
    result.close()
    return result
