from __future__ import annotations

import json
from pathlib import Path
from xml.etree import ElementTree as ET

from annolid.gui.overlay import OverlayDocument, VectorShape, overlay_transform_to_dict


def _shape_points_attribute(shape: VectorShape) -> str:
    return " ".join(f"{float(x):.6g},{float(y):.6g}" for x, y in shape.points)


def export_overlay_document_json(document: OverlayDocument, path: str | Path) -> Path:
    resolved = Path(path)
    payload = {
        "source_path": document.source_path,
        "layer_name": document.layer_name,
        "transform": overlay_transform_to_dict(document.transform),
        "shapes": [
            {
                "id": shape.id,
                "kind": shape.kind,
                "points": [(float(x), float(y)) for x, y in shape.points],
                "label": shape.label,
                "stroke": shape.stroke,
                "fill": shape.fill,
                "text": shape.text,
                "locked": bool(shape.locked),
                "source_tag": shape.source_tag,
                "layer_name": shape.layer_name,
                "source_path": shape.source_path,
            }
            for shape in document.shapes
        ],
    }
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved


def export_overlay_document_labelme(
    document: OverlayDocument, path: str | Path
) -> Path:
    resolved = Path(path)
    shapes = []
    for shape in document.shapes:
        if shape.kind == "polygon":
            shape_type = "polygon"
        elif shape.kind == "point":
            shape_type = "point"
        else:
            shape_type = "line"
        other_data = {}
        if shape.stroke is not None:
            other_data["overlay_stroke"] = shape.stroke
        if shape.fill is not None:
            other_data["overlay_fill"] = shape.fill
        if shape.text is not None:
            other_data["overlay_text"] = shape.text
        if shape.layer_name is not None:
            other_data["overlay_layer"] = shape.layer_name
        if shape.source_tag is not None:
            other_data["overlay_element"] = shape.source_tag
        if shape.source_path is not None:
            other_data["overlay_source"] = shape.source_path
        shapes.append(
            {
                "label": shape.label or shape.id,
                "points": [[float(x), float(y)] for x, y in shape.points],
                "group_id": None,
                "shape_type": shape_type,
                "flags": {},
                "description": "",
                "mask": None,
                "visible": True,
                **other_data,
            }
        )
    payload = {
        "version": "annolid-overlay-export",
        "flags": {},
        "shapes": shapes,
        "imagePath": "",
        "imageData": None,
        "imageHeight": 0,
        "imageWidth": 0,
        "otherData": {
            "overlay": {
                "source_path": document.source_path,
                "layer_name": document.layer_name,
                "transform": overlay_transform_to_dict(document.transform),
            }
        },
    }
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved


def export_overlay_document_svg(document: OverlayDocument, path: str | Path) -> Path:
    resolved = Path(path)
    svg = ET.Element(
        "svg",
        {
            "xmlns": "http://www.w3.org/2000/svg",
            "version": "1.1",
        },
    )
    group = ET.SubElement(
        svg,
        "g",
        {
            "id": document.layer_name or "overlay",
            "data-source-path": str(document.source_path or ""),
        },
    )
    transform = overlay_transform_to_dict(document.transform)
    group.set("data-transform", json.dumps(transform, separators=(",", ":")))
    for shape in document.shapes:
        attrs = {
            "id": str(shape.id or shape.label or "shape"),
        }
        if shape.stroke:
            attrs["stroke"] = str(shape.stroke)
        if shape.fill:
            attrs["fill"] = str(shape.fill)
        elif shape.kind != "polygon":
            attrs.setdefault("fill", "none")
        if shape.kind == "polygon":
            attrs["points"] = _shape_points_attribute(shape)
            ET.SubElement(group, "polygon", attrs)
        elif shape.kind == "polyline":
            attrs["points"] = _shape_points_attribute(shape)
            attrs.setdefault("fill", "none")
            ET.SubElement(group, "polyline", attrs)
        elif shape.kind == "point":
            x, y = shape.points[0] if shape.points else (0.0, 0.0)
            attrs["cx"] = f"{float(x):.6g}"
            attrs["cy"] = f"{float(y):.6g}"
            attrs["r"] = "3"
            ET.SubElement(group, "circle", attrs)
        elif shape.kind == "text":
            x, y = shape.points[0] if shape.points else (0.0, 0.0)
            attrs["x"] = f"{float(x):.6g}"
            attrs["y"] = f"{float(y):.6g}"
            node = ET.SubElement(group, "text", attrs)
            node.text = shape.text or shape.label or shape.id
        else:
            attrs["points"] = _shape_points_attribute(shape)
            attrs.setdefault("fill", "none")
            ET.SubElement(group, "polyline", attrs)
    ET.ElementTree(svg).write(resolved, encoding="utf-8", xml_declaration=True)
    return resolved
