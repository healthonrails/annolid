from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _parse_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, float) and value in (0.0, 1.0):
        return bool(int(value))
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "1", "yes", "y", "on"):
            return True
        if text in ("false", "0", "no", "n", "off"):
            return False
    return None


def sanitize_labelme_flags(flags: Any) -> Dict[str, bool]:
    """Coerce LabelMe image-level flags into a safe `dict[str, bool]`.

    LabelMe treats image-level `flags` as boolean toggles. Annolid also uses these
    flags for behavior pinning. Some datasets may contain non-boolean values
    (e.g. `{"instance_label": "mouse"}`), which will crash Qt checkbox widgets.
    """
    if not isinstance(flags, dict):
        return {}

    sanitized: Dict[str, bool] = {}
    for key, value in flags.items():
        name = str(key).strip() if key is not None else ""
        if not name:
            continue
        parsed = _parse_bool(value)
        if parsed is None:
            continue
        sanitized[name] = parsed
    return sanitized


def sanitize_labelme_flags_with_meta(
    flags: Any,
) -> Tuple[Dict[str, bool], Dict[str, Any]]:
    """Return `(sanitized_flags, meta)` for LabelMe image-level flags.

    LabelMe expects `flags` to be boolean toggles. Any non-boolean values can
    crash Qt checkbox widgets. This helper keeps boolean-like values (coerced to
    `bool`) and returns the remaining entries as `meta` so callers can preserve
    them elsewhere (e.g. `otherData`).
    """
    if not isinstance(flags, dict):
        return {}, {}

    sanitized: Dict[str, bool] = {}
    meta: Dict[str, Any] = {}
    for key, value in flags.items():
        name = str(key).strip() if key is not None else ""
        if not name:
            continue
        parsed = _parse_bool(value)
        if parsed is None:
            meta[name] = value
            continue
        sanitized[name] = parsed
    return sanitized, meta


def sanitize_labelme_shape_dict(shape_data: Any) -> Any:
    """Ensure LabelMe per-shape `flags` contain boolean-only values.

    LabelMe renders per-shape flags as checkbox toggles and expects `bool`
    values. Non-bool values (floats, strings, lists, ...) can crash the LabelMe
    editor. This helper moves non-bool entries out of `shape["flags"]` and into
    extra shape keys so downstream tools can still access the metadata.
    """
    if not isinstance(shape_data, dict):
        return shape_data

    flags = shape_data.get("flags")
    if not isinstance(flags, dict):
        flags = {}

    safe_flags: Dict[str, bool] = {}
    moved_meta: Dict[str, Any] = {}
    for key, value in flags.items():
        name = str(key).strip() if key is not None else ""
        if not name:
            continue
        if isinstance(value, bool):
            safe_flags[name] = value
        else:
            moved_meta[name] = value

    shape_data["flags"] = safe_flags

    reserved_keys = {
        "label",
        "points",
        "group_id",
        "shape_type",
        "flags",
        "visible",
        "description",
        "mask",
        "point_labels",
    }
    for key, value in moved_meta.items():
        if key in reserved_keys:
            meta = shape_data.get("annolid_meta")
            if not isinstance(meta, dict):
                meta = {}
                shape_data["annolid_meta"] = meta
            meta.setdefault(key, value)
        else:
            shape_data.setdefault(key, value)

    return shape_data
