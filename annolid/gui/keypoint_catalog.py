from __future__ import annotations

from typing import Iterable, List

from qtpy import QtCore


def normalize_keypoint_names(values: Iterable[object] | None) -> List[str]:
    """Normalize keypoint labels with stable order and case-insensitive dedupe."""
    if values is None:
        return []
    out: List[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


def merge_keypoint_lists(*sources: Iterable[object] | None) -> List[str]:
    merged: List[str] = []
    for source in sources:
        merged.extend(normalize_keypoint_names(source))
    return normalize_keypoint_names(merged)


def extract_labels_from_uniq_label_list(widget) -> List[str]:
    if widget is None:
        return []
    labels: List[str] = []
    try:
        role = int(QtCore.Qt.UserRole)
        for i in range(widget.count()):
            item = widget.item(i)
            if item is None:
                continue
            # AnnolidUniqLabelListWidget stores canonical label in UserRole and
            # display text may include counts (e.g. "nose [42]").
            raw = item.data(role)
            text = str(raw or "").strip()
            if not text:
                text = str(item.text() or "").strip()
            if text:
                labels.append(text)
    except Exception:
        return []
    return normalize_keypoint_names(labels)
