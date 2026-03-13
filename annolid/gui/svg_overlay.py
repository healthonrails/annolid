from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from qtpy import QtCore

from annolid.gui.overlay import OverlayTransform, overlay_transform_to_dict
from annolid.gui.shape import Shape
from annolid.io.vector.svg_import import (
    ImportedVectorDocument,
    flatten_svg_path,
)
from annolid.io.vector import import_vector_document


@dataclass(frozen=True)
class SvgImportResult:
    shapes: list[Shape]
    metadata: dict[str, object]


def _imported_path_to_shape(
    imported, source_path: Path, overlay_id: str
) -> Shape | None:
    points = list(imported.points or [])
    if imported.kind == "text":
        shape = Shape(
            label=imported.label or imported.text or imported.id,
            shape_type="point",
            flags={},
        )
        if points:
            x, y = points[0]
            shape.addPoint(QtCore.QPointF(float(x), float(y)))
        shape.other_data = {
            "overlay_id": overlay_id,
            "overlay_source": str(source_path),
            "overlay_element": imported.source_tag,
            "overlay_element_id": imported.id,
            "overlay_transform": list(imported.transform or []),
            "overlay_text": imported.text,
            "overlay_layer": imported.layer_name,
            "overlay_stroke": imported.stroke,
            "overlay_fill": imported.fill,
            "overlay_visible": True,
            "overlay_opacity": 0.5,
            "overlay_z_order": 0,
        }
        return shape
    if len(points) < 2:
        return None
    if points[0] == points[-1] and len(points) > 2:
        points = points[:-1]
    if imported.kind == "polygon" and len(points) >= 3:
        shape_type = "polygon"
    elif len(points) == 2:
        shape_type = "line"
    else:
        shape_type = "linestrip"
    shape = Shape(label=imported.label or imported.id, shape_type=shape_type, flags={})
    for x, y in points:
        shape.addPoint(QtCore.QPointF(float(x), float(y)))
    if shape_type == "polygon":
        shape.close()
    shape.other_data = {
        "overlay_id": overlay_id,
        "overlay_source": str(source_path),
        "overlay_element": imported.source_tag,
        "overlay_element_id": imported.id,
        "overlay_transform": list(imported.transform or []),
        "overlay_text": imported.text,
        "overlay_layer": imported.layer_name,
        "overlay_stroke": imported.stroke,
        "overlay_fill": imported.fill,
        "overlay_visible": True,
        "overlay_opacity": 0.5,
        "overlay_z_order": 0,
    }
    return shape


def import_svg_shapes(path: str | Path) -> SvgImportResult:
    document: ImportedVectorDocument = import_vector_document(path)
    source_path = Path(document.source_path)
    overlay_id = f"svg_{uuid4().hex}"
    initial_transform = OverlayTransform()
    shapes = [
        shape
        for shape in (
            _imported_path_to_shape(imported, source_path, overlay_id)
            for imported in document.shapes
        )
        if shape is not None
    ]
    metadata = {
        "id": overlay_id,
        "source": document.source_path,
        "document_width": document.width,
        "document_height": document.height,
        "view_box": document.view_box,
        "shape_count": len(shapes),
        "transform": overlay_transform_to_dict(initial_transform),
    }
    return SvgImportResult(shapes=shapes, metadata=metadata)


def import_vector_shapes(path: str | Path) -> SvgImportResult:
    return import_svg_shapes(path)


__all__ = [
    "SvgImportResult",
    "flatten_svg_path",
    "import_svg_shapes",
    "import_vector_shapes",
]
