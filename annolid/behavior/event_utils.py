"""Shared helpers for behavior event parsing and normalization.

These utilities are intentionally GUI-free so they can be reused by CLI tools
and analytics modules without pulling in Qt dependencies.
"""

from __future__ import annotations

from typing import Optional


def normalize_event_label(event_label: Optional[str]) -> Optional[str]:
    """Map arbitrary event labels onto canonical ``"start"``/``"end"`` tokens.

    Labels that contain "start", "begin", or "onset" (case insensitive) are
    treated as start markers. Labels that contain "end", "stop", or "offset"
    are treated as end markers. Unknown labels return ``None`` so callers can
    decide how to handle them.
    """
    if not event_label:
        return None

    label = event_label.lower()
    if "start" in label or "begin" in label or "onset" in label:
        return "start"
    if "end" in label or "stop" in label or "offset" in label:
        return "end"
    return None
