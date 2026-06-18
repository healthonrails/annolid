"""Shared behavior-label normalization helpers."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, List, Sequence

from annolid.core.behavior.catalog import normalize_behavior_code

NO_BEHAVIOR_LABEL = "no_behavior"

_LABEL_SPLIT_RE = re.compile(r"[,;\r\n]+")
_NO_BEHAVIOR_KEYS = {
    normalize_behavior_code(value)
    for value in (
        NO_BEHAVIOR_LABEL,
        "no behavior",
        "no behaviours",
        "no behaviors",
        "no listed behavior",
        "no listed behaviors",
        "none",
        "none of the above",
        "not applicable",
        "n/a",
        "other",
        "background",
    )
}
_NO_BEHAVIOR_TEXT_PHRASES = (
    "no behavior",
    "no behaviour",
    "no listed behavior",
    "no listed behaviour",
    "none of the listed behaviors",
    "none of the listed behaviours",
    "none of these behaviors",
    "none of these behaviours",
)


def behavior_label_key(value: Any) -> str:
    """Return the comparison key used for behavior labels."""

    return normalize_behavior_code(value)


def iter_behavior_label_values(values: Any) -> Iterable[str]:
    """Yield non-empty behavior labels from strings or nested collections."""

    if values is None:
        return
    if isinstance(values, str):
        for part in _LABEL_SPLIT_RE.split(values):
            label = str(part or "").strip()
            if label:
                yield label
        return
    if isinstance(values, (bytes, bytearray)):
        label = values.decode("utf-8", errors="ignore").strip()
        if label:
            yield label
        return
    if isinstance(values, Iterable):
        for item in values:
            yield from iter_behavior_label_values(item)
        return
    label = str(values or "").strip()
    if label:
        yield label


def is_no_behavior_label(value: Any) -> bool:
    """Return true when a model/user label means the reserved no-behavior class."""

    key = behavior_label_key(value)
    return bool(key and key in _NO_BEHAVIOR_KEYS)


def text_indicates_no_behavior(value: Any) -> bool:
    """Return true for sentence-level no-behavior statements."""

    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()
    if not text:
        return False
    return any(phrase in text for phrase in _NO_BEHAVIOR_TEXT_PHRASES)


def normalize_behavior_label_list(
    values: Any,
    *,
    include_no_behavior: bool = False,
) -> List[str]:
    """Normalize label lists while preserving the first human-readable spelling."""

    labels: List[str] = []
    seen: set[str] = set()
    for label in iter_behavior_label_values(values):
        if is_no_behavior_label(label):
            continue
        key = behavior_label_key(label) or label.casefold()
        if key in seen:
            continue
        seen.add(key)
        labels.append(label)
    if include_no_behavior:
        labels.append(NO_BEHAVIOR_LABEL)
    return labels


def allowed_behavior_labels(values: Any) -> List[str]:
    """Return behavior labels plus the reserved no-behavior sentinel once."""

    return [*normalize_behavior_label_list(values), NO_BEHAVIOR_LABEL]


def canonicalize_behavior_label(value: Any, allowed_labels: Sequence[str]) -> str:
    """Map a model-provided label back to an allowed behavior label."""

    raw = str(value or "").strip()
    if not raw:
        return ""
    if is_no_behavior_label(raw):
        return NO_BEHAVIOR_LABEL

    labels = normalize_behavior_label_list(allowed_labels)
    by_case = {label.casefold(): label for label in labels}
    by_key = {behavior_label_key(label): label for label in labels}
    case_match = by_case.get(raw.casefold())
    if case_match:
        return case_match
    key_match = by_key.get(behavior_label_key(raw))
    if key_match:
        return key_match
    return ""
