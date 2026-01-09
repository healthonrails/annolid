from __future__ import annotations

from enum import IntEnum
from typing import Any, Mapping, MutableMapping, Optional


class KeypointVisibility(IntEnum):
    """YOLO/COCO-style keypoint visibility.

    - 0: not labeled / missing
    - 1: labeled but occluded
    - 2: labeled and visible
    """

    MISSING = 0
    OCCLUDED = 1
    VISIBLE = 2


FLAG_KP_VISIBLE = "kp_visible"
FLAG_KP_VISIBILITY = "kp_visibility"
LEGACY_FLAG_VISIBILITY = "visibility"


def normalize_visibility(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(KeypointVisibility.VISIBLE if value else KeypointVisibility.OCCLUDED)
    if isinstance(value, (int, float)):
        v = int(value)
    elif isinstance(value, str):
        try:
            v = int(float(value.strip()))
        except ValueError:
            return None
    else:
        return None
    if v in (0, 1, 2):
        return v
    return None


def visibility_from_flags(flags: object) -> Optional[int]:
    if not isinstance(flags, Mapping):
        return None
    if FLAG_KP_VISIBILITY in flags:
        v = normalize_visibility(flags.get(FLAG_KP_VISIBILITY))
        if v is not None:
            return v
    if LEGACY_FLAG_VISIBILITY in flags:
        v = normalize_visibility(flags.get(LEGACY_FLAG_VISIBILITY))
        if v is not None:
            return v
    if FLAG_KP_VISIBLE in flags:
        v = normalize_visibility(flags.get(FLAG_KP_VISIBLE))
        if v is not None:
            return v
    return None


def visibility_from_labelme_shape(shape: Mapping[str, Any]) -> Optional[int]:
    flags = shape.get("flags")
    v = visibility_from_flags(flags)
    if v is not None:
        return v
    # Legacy: some pipelines stored YOLO visibility in `description`.
    return normalize_visibility(shape.get("description"))


def ensure_flags(flags: object) -> MutableMapping[str, Any]:
    if isinstance(flags, MutableMapping):
        return flags
    if isinstance(flags, Mapping):
        return dict(flags)
    return {}


def set_visibility_flags(flags: MutableMapping[str, Any], visibility: int) -> None:
    v = normalize_visibility(visibility)
    if v is None:
        raise ValueError(f"Unsupported visibility value: {visibility!r}")
    flags[FLAG_KP_VISIBILITY] = int(v)
    flags[FLAG_KP_VISIBLE] = bool(v == KeypointVisibility.VISIBLE)


def keypoint_visibility_from_shape_object(shape: object) -> int:
    flags = ensure_flags(getattr(shape, "flags", None))
    v = visibility_from_flags(flags)
    if v is not None:
        return int(v)
    return int(KeypointVisibility.VISIBLE)


def set_keypoint_visibility_on_shape_object(shape: object, visibility: int) -> None:
    flags = ensure_flags(getattr(shape, "flags", None))
    set_visibility_flags(flags, visibility)
    setattr(shape, "flags", dict(flags))
