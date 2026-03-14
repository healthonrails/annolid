from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from annolid.gui.overlay.vector_overlay import (
    OverlayTransform,
    VectorShape,
    overlay_transform_from_dict,
)


@dataclass
class AffineTransform:
    tx: float = 0.0
    ty: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    rotation_deg: float = 0.0


@dataclass
class ViewerLayerModel:
    id: str
    name: str
    visible: bool = True
    opacity: float = 1.0
    locked: bool = False
    z_index: int = 0
    transform: AffineTransform = field(default_factory=AffineTransform)


@dataclass
class RasterImageLayer(ViewerLayerModel):
    backend_page_index: int = 0
    channel: int | None = None


@dataclass
class RasterLabelLayer(ViewerLayerModel):
    mapping_table: dict[int, dict] = field(default_factory=dict)
    source_path: str = ""
    page_index: int = 0


@dataclass
class VectorOverlayLayer(ViewerLayerModel):
    shapes: list[VectorShape] = field(default_factory=list)
    source_path: str = ""
    source_kind: str = "svg"
    shape_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LandmarkPair:
    pair_id: str
    source: tuple[float, float]
    target: tuple[float, float]
    key: tuple[str, str] = ("", "")


@dataclass
class LandmarkLayer(ViewerLayerModel):
    pairs: list[LandmarkPair] = field(default_factory=list)


@dataclass
class AnnotationLayer(ViewerLayerModel):
    shapes: list[Any] = field(default_factory=list)


def affine_transform_from_overlay(transform: OverlayTransform) -> AffineTransform:
    return AffineTransform(
        tx=float(transform.tx),
        ty=float(transform.ty),
        sx=float(transform.sx),
        sy=float(transform.sy),
        rotation_deg=float(transform.rotation_deg),
    )


def affine_transform_from_dict(value: dict[str, Any] | None) -> AffineTransform:
    return affine_transform_from_overlay(overlay_transform_from_dict(value))


def raster_label_layer_from_state(
    state: dict[str, Any] | None,
    *,
    mapping_table: dict[int, dict] | None = None,
) -> RasterLabelLayer | None:
    data = dict(state or {})
    source_path = str(data.get("source_path", "") or "")
    if not source_path:
        return None
    name = Path(source_path).stem or "Label Overlay"
    return RasterLabelLayer(
        id="label_image_overlay",
        name=name,
        visible=bool(data.get("visible", True)),
        opacity=float(data.get("opacity", 0.45) or 0.45),
        locked=True,
        z_index=20,
        transform=affine_transform_from_dict(data.get("transform")),
        mapping_table=dict(mapping_table or {}),
        source_path=source_path,
        page_index=int(data.get("page_index", 0) or 0),
    )


def vector_overlay_layer_from_record(
    record: dict[str, Any] | None,
    *,
    shapes: list[VectorShape] | None = None,
) -> VectorOverlayLayer | None:
    data = dict(record or {})
    overlay_id = str(data.get("id", "") or "")
    if not overlay_id:
        return None
    transform = overlay_transform_from_dict(data.get("transform"))
    source_path = str(data.get("source", "") or "")
    metadata = dict(data.get("metadata") or {})
    name = str(metadata.get("layer_name") or Path(source_path).stem or overlay_id)
    shape_items = list(shapes or [])
    return VectorOverlayLayer(
        id=overlay_id,
        name=name,
        visible=bool(transform.visible),
        opacity=float(transform.opacity),
        locked=False,
        z_index=int(transform.z_order),
        transform=affine_transform_from_overlay(transform),
        shapes=shape_items,
        source_path=source_path,
        source_kind=str(
            data.get("source_kind") or metadata.get("source_kind") or "svg"
        ),
        shape_count=int(data.get("shape_count", len(shape_items)) or len(shape_items)),
        metadata=metadata,
    )
