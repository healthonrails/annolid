"""Shared helpers for behavior event parsing and normalization.

These utilities are intentionally GUI-free so they can be reused by CLI tools
and analytics modules without pulling in Qt dependencies.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_SUBJECT_LABEL = "Subject 1"

_POINT_EVENT_TOKENS = frozenset(
    {
        "point",
        "point event",
        "point_event",
        "instant",
        "instantaneous",
        "occurrence",
        "event",
        "click",
        "mark",
    }
)


@dataclass(frozen=True)
class ParsedEventRow:
    """Normalized representation of a single row in a behavior event CSV."""

    line_number: int
    recording_time: Optional[float]
    raw_recording_time: str
    trial_time: Optional[float]
    subject: str
    has_explicit_subject: bool
    behavior: str
    raw_event: str
    event_kind: Optional[str]  # "start", "end", "point", or None
    modifiers: Tuple[str, ...] = ()


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


def classify_event_label(event_label: Optional[str]) -> Optional[str]:
    """Classify an event label as ``"start"``, ``"end"``, ``"point"``, or ``None``."""
    boundary = normalize_event_label(event_label)
    if boundary is not None:
        return boundary
    if not event_label:
        return None
    cleaned = " ".join(str(event_label).strip().lower().split())
    if cleaned in _POINT_EVENT_TOKENS:
        return "point"
    return None


def parse_recording_seconds(value: object) -> Optional[float]:
    """Parse float seconds or ``HH:MM:SS(.mmm)`` / ``MM:SS(.mmm)`` into finite seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    text = str(value).strip()
    if not text:
        return None
    try:
        numeric = float(text)
        return numeric if math.isfinite(numeric) else None
    except ValueError:
        pass

    if ":" in text:
        parts = text.split(":")
        if len(parts) in (2, 3):
            try:
                nums = [float(part) for part in parts]
            except ValueError:
                return None
            if any(not math.isfinite(num) or num < 0 for num in nums):
                return None
            if len(nums) == 2:
                minutes, seconds = nums
                return minutes * 60.0 + seconds
            hours, minutes, seconds = nums
            return hours * 3600.0 + minutes * 60.0 + seconds
    return None


def _split_modifiers(raw_value: object) -> Tuple[str, ...]:
    if raw_value is None:
        return ()
    if isinstance(raw_value, (list, tuple)):
        items = [str(item).strip() for item in raw_value if str(item).strip()]
        return tuple(items)
    text = str(raw_value).strip()
    if not text or text.lower() in {"none", "na", "n/a"}:
        return ()
    parts = re.split(r"[|,;]", text)
    return tuple(part.strip() for part in parts if part.strip())


def parse_event_mapping_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
    start_line_number: int = 1,
) -> List[ParsedEventRow]:
    """Convert raw dictionary rows into :class:`ParsedEventRow` records."""
    parsed: List[ParsedEventRow] = []
    for offset, raw_row in enumerate(rows):
        lowered: Dict[str, object] = {
            str(key).strip().lower(): val
            for key, val in raw_row.items()
            if key is not None
        }
        raw_rec = (
            lowered.get("recording time")
            if "recording time" in lowered
            else lowered.get(
                "timestamp",
                lowered.get("time"),
            )
        )
        raw_rec_str = "" if raw_rec is None else str(raw_rec).strip()
        rec_sec = parse_recording_seconds(raw_rec)

        raw_trial = lowered.get("trial time")
        trial_sec = parse_recording_seconds(raw_trial)

        raw_subject = lowered.get("subject")
        subj_str = "" if raw_subject is None else str(raw_subject).strip()
        has_explicit_subject = bool(subj_str)
        subject = subj_str if subj_str else default_subject

        raw_behavior = lowered.get("behavior")
        behavior = "" if raw_behavior is None else str(raw_behavior).strip()

        raw_event_val = lowered.get("event")
        raw_event = "" if raw_event_val is None else str(raw_event_val).strip()
        event_kind = classify_event_label(raw_event)

        raw_modifiers = (
            lowered.get("modifiers")
            if "modifiers" in lowered
            else lowered.get("modifier")
        )
        modifiers = _split_modifiers(raw_modifiers)

        line_no = int(lowered.get("_line_number", start_line_number + offset))
        parsed.append(
            ParsedEventRow(
                line_number=line_no,
                recording_time=rec_sec,
                raw_recording_time=raw_rec_str,
                trial_time=trial_sec,
                subject=subject,
                has_explicit_subject=has_explicit_subject,
                behavior=behavior,
                raw_event=raw_event,
                event_kind=event_kind,
                modifiers=modifiers,
            )
        )
    return parsed


def read_parsed_event_rows(
    path: Path,
    *,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
) -> List[ParsedEventRow]:
    """Read a behavior event CSV file into :class:`ParsedEventRow` records.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When required headers (recording time, behavior, event) are absent.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Behavior events CSV not found: {csv_path}")

    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        normalized_headers = {
            str(name).strip().lower() for name in reader.fieldnames if name is not None
        }
        has_time_col = bool(
            normalized_headers & {"recording time", "timestamp", "time"}
        )
        missing: List[str] = []
        if not has_time_col:
            missing.append("Recording time")
        if "behavior" not in normalized_headers:
            missing.append("Behavior")
        if "event" not in normalized_headers:
            missing.append("Event")
        if missing:
            raise ValueError(
                f"Behavior CSV {csv_path} is missing required column(s): {', '.join(missing)}"
            )

        raw_rows: List[Dict[str, object]] = []
        for row_idx, row in enumerate(reader, start=2):
            entry: Dict[str, object] = dict(row)
            entry["_line_number"] = row_idx
            raw_rows.append(entry)

    return parse_event_mapping_rows(
        raw_rows,
        default_subject=default_subject,
        start_line_number=2,
    )


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
