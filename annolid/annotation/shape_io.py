"""Canonical LabelMe-compatible shape serialization and normalization.

Unifies shape-to-dict and dict-to-shape record conversion across GUI saves,
LabelFile loading, keypoint writers, and shape propagation utilities.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import numpy as np

from annolid.utils.annotation_compat import utils as labelme_utils
from annolid.utils.labelme_flags import sanitize_labelme_shape_dict

CANONICAL_SHAPE_KEYS = frozenset(
    {
        "label",
        "points",
        "group_id",
        "shape_type",
        "flags",
        "description",
        "mask",
        "visible",
        "locked",
        "shared_vertex_ids",
        "shared_edge_ids",
        "point_labels",
    }
)


def normalize_group_id(value: Any) -> Optional[int]:
    """Normalize a LabelMe ``group_id`` safely (rejecting booleans)."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if np.isfinite(value) and float(value).is_integer():
            return int(value)
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return int(text)
        except ValueError:
            return None
    return None


def serialize_point(point: Any) -> list[float]:
    """Serialize a Qt point or 2-element sequence to ``[float, float]``."""
    if hasattr(point, "x") and hasattr(point, "y") and callable(point.x):
        return [float(point.x()), float(point.y())]
    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 2:
        return [float(point[0]), float(point[1])]
    raise TypeError(f"Unsupported point format: {point!r}")


def with_manual_description_tag(description: Optional[str]) -> str:
    """Append Annolid's backward-compatible ``manul`` marker when saving manually."""
    text = (description or "").strip()
    if not text:
        return "manul"
    lowered_tokens = {token.strip().lower() for token in text.split(",")}
    if "manul" in lowered_tokens:
        return text
    return f"{text}, manul"


def _encode_shape_mask(mask: Any) -> Optional[str]:
    if mask is None:
        return None
    if isinstance(mask, str):
        return mask
    arr = np.asarray(mask)
    if arr.size == 0:
        return None
    if arr.ndim > 2:
        arr = arr[..., 0]
    return labelme_utils.img_arr_to_b64(arr.astype(bool).astype(np.uint8))


def _decode_shape_mask(raw_mask: Any) -> Optional[np.ndarray]:
    if raw_mask is None:
        return None
    if isinstance(raw_mask, np.ndarray):
        arr = raw_mask
        if arr.ndim > 2:
            arr = arr[..., 0]
        return arr.astype(bool)
    if isinstance(raw_mask, (str, bytes)):
        if not raw_mask:
            return None
        return labelme_utils.img_b64_to_arr(raw_mask).astype(bool)
    return None


def serialize_shape(
    shape: Any,
    *,
    include_manual_tag: bool = False,
    include_point_labels: bool = False,
    sanitize_flags: bool = False,
    include_null_mask: bool = False,
) -> Dict[str, Any]:
    """Convert a ``Shape`` instance (or dict) to a JSON-safe LabelMe shape dict.

    Never mutates the input ``shape`` instance.
    """
    if isinstance(shape, dict):
        data = dict(shape)
        if sanitize_flags:
            return sanitize_labelme_shape_dict(data)
        return data

    other_data = getattr(shape, "other_data", None)
    data: Dict[str, Any] = dict(other_data) if isinstance(other_data, Mapping) else {}
    # Remove any stale mask key carried over in other_data for non-mask shapes.
    data.pop("mask", None)

    points = [serialize_point(pt) for pt in (getattr(shape, "points", None) or [])]
    shape_type = str(getattr(shape, "shape_type", None) or "polygon")
    raw_mask = getattr(shape, "mask", None)
    is_mask_shape = shape_type.lower() == "mask"

    if not is_mask_shape and len(points) == 1 and shape_type == "polygon":
        shape_type = "point"

    raw_flags = getattr(shape, "flags", None)
    flags = dict(raw_flags) if isinstance(raw_flags, Mapping) else {}
    description = getattr(shape, "description", None)
    if include_manual_tag:
        description = with_manual_description_tag(description)

    data.update(
        {
            "label": getattr(shape, "label", ""),
            "points": points,
            "group_id": normalize_group_id(getattr(shape, "group_id", None)),
            "shape_type": shape_type,
            "flags": flags,
            "visible": bool(getattr(shape, "visible", True)),
            "description": description,
        }
    )

    if is_mask_shape and raw_mask is not None:
        encoded_mask = _encode_shape_mask(raw_mask)
        if encoded_mask is not None:
            data["mask"] = encoded_mask
        elif include_null_mask:
            data["mask"] = None
    elif include_null_mask:
        data["mask"] = None

    locked = bool(getattr(shape, "locked", False))
    if locked:
        data["locked"] = True
    else:
        data.pop("locked", None)
        data.pop("overlay_locked", None)

    shared_vertex_ids = list(getattr(shape, "shared_vertex_ids", None) or [])
    if any(shared_vertex_ids):
        data["shared_vertex_ids"] = [str(v or "") for v in shared_vertex_ids]
    else:
        data.pop("shared_vertex_ids", None)

    shared_edge_ids = list(getattr(shape, "shared_edge_ids", None) or [])
    if any(shared_edge_ids):
        data["shared_edge_ids"] = [str(v or "") for v in shared_edge_ids]
    else:
        data.pop("shared_edge_ids", None)

    if include_point_labels:
        point_labels = getattr(shape, "point_labels", None)
        if point_labels:
            data["point_labels"] = list(point_labels)

    if sanitize_flags:
        return sanitize_labelme_shape_dict(data)
    return data


def deserialize_shape_record(raw: Mapping[str, Any]) -> Dict[str, Any]:
    """Normalize a raw LabelMe JSON shape dict into a canonical in-memory record."""
    shape_type = str(raw.get("shape_type") or "polygon")
    raw_flags = raw.get("flags")
    flags = dict(raw_flags) if isinstance(raw_flags, Mapping) else {}
    mask = (
        _decode_shape_mask(raw.get("mask"))
        if shape_type.lower() == "mask" and raw.get("mask") is not None
        else None
    )
    visible = True if raw.get("visible") is None else bool(raw.get("visible"))
    locked = bool(raw.get("locked", False)) or bool(raw.get("overlay_locked", False))
    other_data = {k: v for k, v in raw.items() if k not in CANONICAL_SHAPE_KEYS}

    return {
        "label": raw["label"],
        "points": raw["points"],
        "shape_type": shape_type,
        "flags": flags,
        "description": raw.get("description"),
        "group_id": normalize_group_id(raw.get("group_id")),
        "mask": mask,
        "visible": visible,
        "locked": locked,
        "shared_vertex_ids": list(raw.get("shared_vertex_ids") or []),
        "shared_edge_ids": list(raw.get("shared_edge_ids") or []),
        "point_labels": list(raw.get("point_labels") or []),
        "other_data": other_data,
    }
