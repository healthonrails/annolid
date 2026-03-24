from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from qtpy import QtCore

from annolid.gui.overlay import OverlayTransform, VectorShape, overlay_transform_to_dict
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


def _normalize_shape_labels(shapes: list[Shape]) -> None:
    counts: dict[str, int] = {}
    total: dict[str, int] = {}
    for shape in list(shapes or []):
        label = str(getattr(shape, "label", "") or "").strip()
        if not label:
            label = "shape"
        total[label] = int(total.get(label, 0) or 0) + 1
    for shape in list(shapes or []):
        label = str(getattr(shape, "label", "") or "").strip()
        if not label:
            label = "shape"
        if int(total.get(label, 0) or 0) <= 1:
            shape.label = label
            continue
        index = int(counts.get(label, 0) or 0) + 1
        counts[label] = index
        shape.label = f"{label}_{index}"


def _looks_like_axis_aligned_rectangle(points: list[tuple[float, float]]) -> bool:
    pts = list(points or [])
    if len(pts) < 4:
        return False
    if len(pts) >= 2 and pts[0] == pts[-1]:
        pts = pts[:-1]
    if len(pts) != 4:
        return False
    xs = sorted({round(float(x), 6) for x, _ in pts})
    ys = sorted({round(float(y), 6) for _, y in pts})
    if len(xs) != 2 or len(ys) != 2:
        return False
    corners = {(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])}
    normalized = {(round(float(x), 6), round(float(y), 6)) for x, y in pts}
    return normalized == corners


def _bbox_only_overlay(shapes: list[Shape]) -> bool:
    if not shapes:
        return False
    polygon_shapes = [
        s for s in shapes if str(getattr(s, "shape_type", "")).lower() == "polygon"
    ]
    if not polygon_shapes or len(polygon_shapes) != len(shapes):
        return False
    return all(
        _looks_like_axis_aligned_rectangle(
            [
                (float(point.x()), float(point.y()))
                for point in list(getattr(shape, "points", []) or [])
            ]
        )
        for shape in polygon_shapes
    )


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
            "overlay_role": "correction",
            "overlay_locked": False,
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
        "overlay_role": "correction",
        "overlay_locked": False,
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
    source_kind = str(getattr(document, "source_kind", "svg") or "svg").lower()

    def _should_render_imported_shape(imported_path) -> bool:
        # For PDF-compatible AI imports, only render region polygons by default.
        # These files often include dense guide lines and text outlines that
        # clutter the overlay and interfere with annotation workflows.
        if source_kind in {"ai", "pdf"}:
            return str(getattr(imported_path, "kind", "") or "").lower() == "polygon"
        return True

    shapes = [
        shape
        for shape in (
            _imported_path_to_shape(imported, source_path, overlay_id)
            for imported in document.shapes
            if _should_render_imported_shape(imported)
        )
        if shape is not None
    ]
    _normalize_shape_labels(shapes)
    metadata = {
        "id": overlay_id,
        "source": document.source_path,
        "source_kind": source_kind,
        "document_width": document.width,
        "document_height": document.height,
        "view_box": document.view_box,
        "page_box": list(document.page_box) if document.page_box else None,
        "art_box": list(document.art_box) if document.art_box else None,
        "shape_count": len(shapes),
        "source_shapes": [
            VectorShape(
                id=str(imported.id or f"shape_{index}"),
                kind=str(imported.kind or "polyline"),
                points=[(float(x), float(y)) for x, y in list(imported.points or [])],
                label=str(imported.label or "") or None,
                stroke=imported.stroke,
                fill=imported.fill,
                text=imported.text,
                locked=True,
                source_tag=imported.source_tag,
                layer_name=imported.layer_name,
                source_path=document.source_path,
            )
            for index, imported in enumerate(document.shapes)
        ],
        "transform": overlay_transform_to_dict(initial_transform),
        "locked_source": True,
        "editable_layer_name": "Corrections",
    }
    if metadata["source_kind"] in {"ai", "pdf"} and _bbox_only_overlay(shapes):
        metadata["bbox_only_overlay"] = True
        metadata["import_warning"] = (
            "Imported overlay appears to contain only rectangular page/image bounds. "
            "This AI/PDF may not include editable vector region paths."
        )
    return SvgImportResult(shapes=shapes, metadata=metadata)


def import_vector_shapes(path: str | Path) -> SvgImportResult:
    return import_svg_shapes(path)


__all__ = [
    "SvgImportResult",
    "flatten_svg_path",
    "import_svg_shapes",
    "import_vector_shapes",
]
