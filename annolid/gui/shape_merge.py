"""Lossless geometric union for single-ring canvas annotations."""

import math
from copy import deepcopy

from qtpy.QtCore import QPointF
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from annolid.gui.shape import Shape


def merge_shapes(shapes):
    """Return a new polygon without mutating inputs; reject lossy unions."""
    if len(shapes) < 2:
        raise ValueError("Select at least two polygons or rectangles to merge.")
    first = shapes[0]
    metadata = ("label", "group_id", "flags", "description", "other_data")
    geometries = []
    for shape in shapes:
        if shape.shape_type not in {"polygon", "rectangle"}:
            raise ValueError("Only polygons and rectangles can be merged.")
        if not shape.visible or shape.mask is not None:
            raise ValueError("Select visible shapes without attached masks.")
        if any(getattr(shape, key) != getattr(first, key) for key in metadata):
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
