from __future__ import annotations

from typing import Any, Mapping


def is_zone_shape_payload(shape: Mapping[str, Any] | None) -> bool:
    """Return True when a LabelMe-style shape payload semantically represents a zone."""
    if not isinstance(shape, Mapping):
        return False

    flags = shape.get("flags")
    if not isinstance(flags, Mapping):
        flags = {}

    semantic_type = str(flags.get("semantic_type") or "").strip().lower()
    if semantic_type == "zone":
        return True

    shape_category = str(flags.get("shape_category") or "").strip().lower()
    if shape_category == "zone":
        return True

    if bool(flags.get("zone")):
        return True

    zone_kind = flags.get("zone_kind")
    if isinstance(zone_kind, str) and zone_kind.strip():
        return True

    zone_label = flags.get("zone_label")
    if isinstance(zone_label, str) and zone_label.strip():
        return True

    label_text = str(shape.get("label") or "").strip().lower()
    description_text = str(shape.get("description") or "").strip().lower()
    return ("zone" in label_text) or ("zone" in description_text)
