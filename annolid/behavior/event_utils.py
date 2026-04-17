"""Shared helpers for behavior event parsing and normalization.

These utilities are intentionally GUI-free so they can be reused by CLI tools
and analytics modules without pulling in Qt dependencies.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional, Sequence


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


AGGRESSION_BOUT_LABEL = "aggression_bout"
AGGRESSION_SUB_EVENT_CODES: tuple[str, ...] = (
    "slap_in_face",
    "run_away",
    "fight_initiation",
)

_AGGRESSION_ALIAS_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "slap_in_face",
        re.compile(r"\bslap(?:\s+in(?:\s+the)?\s+face)?\b", flags=re.IGNORECASE),
    ),
    (
        "run_away",
        re.compile(
            r"\b(run[\s_-]?away|runs?[\s_-]?away|ran[\s_-]?away)\b", flags=re.IGNORECASE
        ),
    ),
    (
        "fight_initiation",
        re.compile(
            r"\b(fight[\s_-]?initiation|initiation\s+of\s+(?:a\s+)?(?:big(?:ger)?)?\s*fights?|initiat(?:e|es|ed|ing)\s+(?:a\s+)?(?:big(?:ger)?)?\s*fights?)\b",
            flags=re.IGNORECASE,
        ),
    ),
)


def aggression_sub_event_schema() -> Dict[str, object]:
    """Return the canonical schema used for aggression bout sub-event scoring."""
    return {
        "schema": "aggression_sub_events/v1",
        "bout_label": AGGRESSION_BOUT_LABEL,
        "sub_event_codes": list(AGGRESSION_SUB_EVENT_CODES),
    }


def normalize_aggression_sub_event_code(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    if normalized in AGGRESSION_SUB_EVENT_CODES:
        return normalized
    for code, pattern in _AGGRESSION_ALIAS_PATTERNS:
        if pattern.search(text):
            return code
    return None


def parse_aggression_sub_event_counts(value: object) -> Dict[str, int]:
    """Parse user/model-provided sub-event structures into canonical counts."""
    counts: Dict[str, int] = {}

    def _add(raw_code: object, raw_count: object = 1) -> None:
        code = normalize_aggression_sub_event_code(
            None if raw_code is None else str(raw_code)
        )
        if code is None:
            return
        try:
            parsed = int(raw_count)
        except Exception:
            parsed = 1
        if parsed <= 0:
            return
        counts[code] = int(counts.get(code, 0)) + parsed

    if value is None:
        return {}

    if isinstance(value, Mapping):
        for raw_code, raw_count in value.items():
            _add(raw_code, raw_count)
        return _ordered_aggression_counts(counts)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for item in value:
            if isinstance(item, Mapping):
                _add(
                    item.get("event") or item.get("code") or item.get("name"),
                    item.get("count", 1),
                )
            else:
                _add(item, 1)
        return _ordered_aggression_counts(counts)

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return {}
        inferred = infer_aggression_sub_event_counts_from_text(text)
        for code, count in inferred.items():
            counts[code] = int(counts.get(code, 0)) + int(count)
        return _ordered_aggression_counts(counts)

    return {}


def infer_aggression_sub_event_counts_from_text(text: Optional[str]) -> Dict[str, int]:
    """Infer canonical aggression sub-event counts from free-form text."""
    raw = str(text or "").strip()
    if not raw:
        return {}
    counts: Dict[str, int] = {}
    for code, pattern in _AGGRESSION_ALIAS_PATTERNS:
        matches = list(pattern.finditer(raw))
        if matches:
            counts[code] = int(len(matches))
    return _ordered_aggression_counts(counts)


def aggregate_aggression_bout_summary(
    predictions: Sequence[Mapping[str, Any]],
    *,
    bout_label: str = AGGRESSION_BOUT_LABEL,
) -> Dict[str, object]:
    """Aggregate stable bout and sub-event counts from segment predictions."""
    label_norm = str(bout_label or AGGRESSION_BOUT_LABEL).strip().lower()

    bout_count = 0
    sub_event_counts: Dict[str, int] = {code: 0 for code in AGGRESSION_SUB_EVENT_CODES}
    sub_event_bout_counts: Dict[str, int] = {
        code: 0 for code in AGGRESSION_SUB_EVENT_CODES
    }

    for item in predictions:
        label = str(item.get("label") or "").strip().lower()
        if label != label_norm:
            continue
        bout_count += 1

        parsed = parse_aggression_sub_event_counts(
            item.get("aggression_sub_events")
            or item.get("sub_events")
            or item.get("subevents")
        )
        if not parsed:
            parsed = infer_aggression_sub_event_counts_from_text(
                str(item.get("description") or "")
            )

        for code in AGGRESSION_SUB_EVENT_CODES:
            count = int(parsed.get(code, 0))
            if count <= 0:
                continue
            sub_event_counts[code] += count
            sub_event_bout_counts[code] += 1

    return {
        "schema": "aggression_bout_summary/v1",
        "bout_label": label_norm,
        "bout_count": int(bout_count),
        "sub_event_counts": _ordered_aggression_counts(sub_event_counts),
        "sub_event_bout_counts": _ordered_aggression_counts(sub_event_bout_counts),
        "bouts_with_initiation": int(sub_event_bout_counts.get("fight_initiation", 0)),
    }


def _ordered_aggression_counts(raw_counts: Mapping[str, int]) -> Dict[str, int]:
    ordered: Dict[str, int] = {}
    for code in AGGRESSION_SUB_EVENT_CODES:
        count = int(raw_counts.get(code, 0))
        if count > 0:
            ordered[code] = count
    return ordered
