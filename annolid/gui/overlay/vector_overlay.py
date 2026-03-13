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


@dataclass
class OverlayDocument:
    source_path: str
    layer_name: str | None = None
    transform: OverlayTransform = field(default_factory=OverlayTransform)
    shapes: list[VectorShape] = field(default_factory=list)


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
