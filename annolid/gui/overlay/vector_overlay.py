from __future__ import annotations

from dataclasses import dataclass, field

import math
from typing import Any

import numpy as np


@dataclass
class OverlayTransform:
    tx: float = 0.0
    ty: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    rotation_deg: float = 0.0
    opacity: float = 0.5
    visible: bool = True
    z_order: int = 0


@dataclass
class VectorShape:
    id: str
    kind: str
    points: list[tuple[float, float]]
    label: str | None = None
    stroke: str | None = None
    fill: str | None = None
    text: str | None = None
    locked: bool = False
    source_tag: str | None = None
    layer_name: str | None = None
    source_path: str | None = None


def vector_shape_to_dict(shape: VectorShape) -> dict[str, Any]:
    return {
        "id": str(shape.id or ""),
        "kind": str(shape.kind or ""),
        "points": [(float(x), float(y)) for x, y in list(shape.points or [])],
        "label": str(shape.label or "") or None,
        "stroke": str(shape.stroke or "") or None,
        "fill": str(shape.fill or "") or None,
        "text": str(shape.text or "") or None,
        "locked": bool(shape.locked),
        "source_tag": str(shape.source_tag or "") or None,
        "layer_name": str(shape.layer_name or "") or None,
        "source_path": str(shape.source_path or "") or None,
    }


def vector_shape_from_dict(
    value: dict[str, Any] | VectorShape | None,
) -> VectorShape | None:
    if isinstance(value, VectorShape):
        return value
    data = dict(value or {})
    shape_id = str(data.get("id", "") or "")
    kind = str(data.get("kind", "") or "")
    if not shape_id or not kind:
        return None
    points = [
        (float(point[0]), float(point[1]))
        for point in list(data.get("points") or [])
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    return VectorShape(
        id=shape_id,
        kind=kind,
        points=points,
        label=str(data.get("label", "") or "") or None,
        stroke=str(data.get("stroke", "") or "") or None,
        fill=str(data.get("fill", "") or "") or None,
        text=str(data.get("text", "") or "") or None,
        locked=bool(data.get("locked", False)),
        source_tag=str(data.get("source_tag", "") or "") or None,
        layer_name=str(data.get("layer_name", "") or "") or None,
        source_path=str(data.get("source_path", "") or "") or None,
    )


@dataclass
class OverlayDocument:
    source_path: str
    layer_name: str | None = None
    transform: OverlayTransform = field(default_factory=OverlayTransform)
    shapes: list[VectorShape] = field(default_factory=list)
    source_kind: str = "svg"
    source_shapes: list[VectorShape] = field(default_factory=list)
    landmark_pairs: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OverlayLandmarkPairState:
    pair_id: str
    overlay_label: str | None = None
    image_label: str | None = None
    overlay_element_id: str | None = None
    image_shape_key: str | None = None


@dataclass
class OverlayRecordModel:
    id: str
    source_path: str
    source_kind: str = "svg"
    shape_count: int = 0
    transform: OverlayTransform = field(default_factory=OverlayTransform)
    metadata: dict[str, Any] = field(default_factory=dict)
    source_shapes: list[VectorShape] = field(default_factory=list)
    editable_shapes: list[VectorShape] = field(default_factory=list)
    landmark_pairs: list[OverlayLandmarkPairState] = field(default_factory=list)


def overlay_transform_to_dict(transform: OverlayTransform) -> dict[str, Any]:
    return {
        "tx": float(transform.tx),
        "ty": float(transform.ty),
        "sx": float(transform.sx),
        "sy": float(transform.sy),
        "rotation_deg": float(transform.rotation_deg),
        "opacity": float(transform.opacity),
        "visible": bool(transform.visible),
        "z_order": int(transform.z_order),
    }


def overlay_transform_from_dict(
    value: dict[str, Any] | None,
    *,
    default: OverlayTransform | None = None,
) -> OverlayTransform:
    base = default or OverlayTransform()
    data = dict(value or {})
    return OverlayTransform(
        tx=float(data.get("tx", base.tx)),
        ty=float(data.get("ty", base.ty)),
        sx=float(data.get("sx", base.sx)),
        sy=float(data.get("sy", base.sy)),
        rotation_deg=float(data.get("rotation_deg", base.rotation_deg)),
        opacity=float(data.get("opacity", base.opacity)),
        visible=bool(data.get("visible", base.visible)),
        z_order=int(data.get("z_order", base.z_order)),
    )


def overlay_landmark_pair_to_dict(
    pair: OverlayLandmarkPairState,
) -> dict[str, Any]:
    return {
        "pair_id": str(pair.pair_id or ""),
        "overlay_label": str(pair.overlay_label or "") or None,
        "image_label": str(pair.image_label or "") or None,
        "overlay_element_id": str(pair.overlay_element_id or "") or None,
        "image_shape_key": str(pair.image_shape_key or "") or None,
    }


def overlay_landmark_pair_from_dict(
    value: dict[str, Any] | None,
) -> OverlayLandmarkPairState | None:
    data = dict(value or {})
    pair_id = str(data.get("pair_id", "") or "")
    if not pair_id:
        return None
    return OverlayLandmarkPairState(
        pair_id=pair_id,
        overlay_label=str(data.get("overlay_label", "") or "") or None,
        image_label=str(data.get("image_label", "") or "") or None,
        overlay_element_id=str(data.get("overlay_element_id", "") or "") or None,
        image_shape_key=str(data.get("image_shape_key", "") or "") or None,
    )


def overlay_record_from_dict(value: dict[str, Any] | None) -> OverlayRecordModel | None:
    data = dict(value or {})
    overlay_id = str(data.get("id", "") or "")
    if not overlay_id:
        return None
    metadata = dict(data.get("metadata") or {})
    landmark_pairs = [
        pair
        for pair in (
            overlay_landmark_pair_from_dict(item)
            for item in list(
                data.get("landmark_pairs") or metadata.get("landmark_pairs") or []
            )
        )
        if pair is not None
    ]
    source_shapes = [
        shape
        for shape in (
            vector_shape_from_dict(item)
            for item in list(
                data.get("source_shapes") or metadata.get("source_shapes") or []
            )
        )
        if shape is not None
    ]
    editable_shapes = [
        shape
        for shape in (
            vector_shape_from_dict(item)
            for item in list(data.get("editable_shapes") or [])
        )
        if shape is not None
    ]
    source_kind = str(data.get("source_kind") or metadata.get("source_kind") or "svg")
    return OverlayRecordModel(
        id=overlay_id,
        source_path=str(data.get("source", "") or ""),
        source_kind=source_kind,
        shape_count=int(data.get("shape_count", 0) or 0),
        transform=overlay_transform_from_dict(data.get("transform")),
        metadata=metadata,
        source_shapes=list(source_shapes),
        editable_shapes=list(editable_shapes),
        landmark_pairs=landmark_pairs,
    )


def overlay_record_to_dict(record: OverlayRecordModel) -> dict[str, Any]:
    metadata = dict(record.metadata or {})
    metadata.setdefault("source_kind", str(record.source_kind or "svg"))
    metadata["source_shapes"] = [
        vector_shape_to_dict(shape) for shape in list(record.source_shapes or [])
    ]
    metadata["landmark_pairs"] = [
        overlay_landmark_pair_to_dict(pair)
        for pair in list(record.landmark_pairs or [])
    ]
    return {
        "id": str(record.id or ""),
        "source": str(record.source_path or ""),
        "source_kind": str(record.source_kind or "svg"),
        "shape_count": int(record.shape_count or len(record.editable_shapes or [])),
        "metadata": metadata,
        "transform": overlay_transform_to_dict(record.transform),
        "source_shapes": [
            vector_shape_to_dict(shape) for shape in list(record.source_shapes or [])
        ],
        "editable_shapes": [
            vector_shape_to_dict(shape) for shape in list(record.editable_shapes or [])
        ],
        "landmark_pairs": [
            overlay_landmark_pair_to_dict(pair)
            for pair in list(record.landmark_pairs or [])
        ],
    }


def overlay_transform_to_matrix(transform: OverlayTransform) -> np.ndarray:
    radians = math.radians(float(transform.rotation_deg))
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)
    matrix = np.array(
        [
            [transform.sx * cos_theta, -transform.sy * sin_theta, transform.tx],
            [transform.sx * sin_theta, transform.sy * cos_theta, transform.ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return matrix


def points_bounds_center(points: list[tuple[float, float]]) -> tuple[float, float]:
    if not points:
        return (0.0, 0.0)
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    return ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)


def overlay_delta_matrix(
    current: OverlayTransform,
    target: OverlayTransform,
    *,
    pivot: tuple[float, float] | None = None,
) -> np.ndarray:
    pivot_x, pivot_y = pivot if pivot is not None else (0.0, 0.0)
    delta_tx = float(target.tx) - float(current.tx)
    delta_ty = float(target.ty) - float(current.ty)
    if abs(float(current.sx)) < 1e-8 or abs(float(current.sy)) < 1e-8:
        raise ValueError("Current overlay scale must be non-zero")
    scale_x = float(target.sx) / float(current.sx)
    scale_y = float(target.sy) / float(current.sy)
    rotation_deg = float(target.rotation_deg) - float(current.rotation_deg)
    radians = math.radians(rotation_deg)
    cos_theta = math.cos(radians)
    sin_theta = math.sin(radians)

    translate_to_origin = np.array(
        [[1.0, 0.0, -pivot_x], [0.0, 1.0, -pivot_y], [0.0, 0.0, 1.0]], dtype=float
    )
    scale_matrix = np.array(
        [[scale_x, 0.0, 0.0], [0.0, scale_y, 0.0], [0.0, 0.0, 1.0]], dtype=float
    )
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta, 0.0], [sin_theta, cos_theta, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    translate_from_origin = np.array(
        [[1.0, 0.0, pivot_x], [0.0, 1.0, pivot_y], [0.0, 0.0, 1.0]], dtype=float
    )
    translate_delta = np.array(
        [[1.0, 0.0, delta_tx], [0.0, 1.0, delta_ty], [0.0, 0.0, 1.0]], dtype=float
    )
    return (
        translate_delta
        @ translate_from_origin
        @ rotation_matrix
        @ scale_matrix
        @ translate_to_origin
    )
