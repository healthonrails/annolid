from __future__ import annotations

from typing import Any, Dict, Optional


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
