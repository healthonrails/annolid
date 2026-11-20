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
    """Normalized representation of a single row in a behavior event CSV/TSV."""

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
    comment: str = ""


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


def _extract_modifiers_from_lowered_row(
    lowered: Mapping[str, object],
) -> Tuple[str, ...]:
    collected: List[str] = []
    for key in ("modifiers", "modifier", "behavior modifier(s)", "behavior modifiers"):
        if key in lowered and lowered[key] is not None:
            collected.extend(_split_modifiers(lowered[key]))
    # BORIS tabular/aggregated exports can emit numbered modifier columns:
    # "modifier #1", "modifier #2", "modifier 1", etc.
    numbered_keys = sorted(
        k
        for k in lowered
        if re.match(r"^modifier\s*#?\s*\d+$", k) and lowered[k] is not None
    )
    for k in numbered_keys:
        collected.extend(_split_modifiers(lowered[k]))
    deduped: List[str] = []
    for item in collected:
        if item and item not in deduped:
            deduped.append(item)
    return tuple(deduped)


def _extract_comment_from_lowered_row(lowered: Mapping[str, object]) -> str:
    parts: List[str] = []
    for key in (
        "comment",
        "comments",
        "notes",
        "note",
        "comment start",
        "comment stop",
    ):
        raw = lowered.get(key)
        if raw is None:
            continue
        txt = str(raw).strip()
        if txt and txt.lower() not in {"none", "nan"} and txt not in parts:
            parts.append(txt)
    return "; ".join(parts)


def parse_event_mapping_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
    start_line_number: int = 1,
) -> List[ParsedEventRow]:
    """Convert raw dictionary rows into :class:`ParsedEventRow` records.

    Supports Annolid CSV rows, BORIS Tabular rows (``Time``, ``Status`` = START/STOP/POINT),
    and BORIS Aggregated rows (``Start (s)``, ``Stop (s)``, ``Behavior type`` = STATE/POINT).
    """
    parsed: List[ParsedEventRow] = []
    for offset, raw_row in enumerate(rows):
        lowered: Dict[str, object] = {
            str(key).strip().lower(): val
            for key, val in raw_row.items()
            if key is not None
        }
        line_no = int(lowered.get("_line_number", start_line_number + offset))

        raw_subject = (
            lowered.get("subject")
            if "subject" in lowered
            else lowered.get("subject name")
        )
        subj_str = "" if raw_subject is None else str(raw_subject).strip()
        if subj_str.lower() in {"no focal subject", "none"}:
            subj_str = ""
        has_explicit_subject = bool(subj_str)
        subject = subj_str if subj_str else default_subject

        raw_behavior = (
            lowered.get("behavior")
            if "behavior" in lowered
            else lowered.get("behavior code")
        )
        behavior = "" if raw_behavior is None else str(raw_behavior).strip()

        modifiers = _extract_modifiers_from_lowered_row(lowered)
        comment = _extract_comment_from_lowered_row(lowered)

        has_event_or_status = any(
            k in lowered for k in ("event", "status", "event type")
        )
        has_aggregated_bounds = any(
            k in lowered for k in ("start (s)", "start")
        ) and any(k in lowered for k in ("stop (s)", "stop"))

        if not has_event_or_status and has_aggregated_bounds:
            raw_start = (
                lowered.get("start (s)")
                if "start (s)" in lowered
                else lowered.get("start")
            )
            raw_stop = (
                lowered.get("stop (s)")
                if "stop (s)" in lowered
                else lowered.get("stop")
            )
            start_str = "" if raw_start is None else str(raw_start).strip()
            stop_str = "" if raw_stop is None else str(raw_stop).strip()
            start_sec = parse_recording_seconds(raw_start)
            stop_sec = parse_recording_seconds(raw_stop)
            btype = (
                str(lowered.get("behavior type") or lowered.get("type") or "")
                .strip()
                .lower()
            )
            is_point_row = "point" in btype or (
                not btype and not stop_str and start_sec is not None
            )
            if is_point_row:
                parsed.append(
                    ParsedEventRow(
                        line_number=line_no,
                        recording_time=start_sec,
                        raw_recording_time=start_str,
                        trial_time=start_sec,
                        subject=subject,
                        has_explicit_subject=has_explicit_subject,
                        behavior=behavior,
                        raw_event="POINT",
                        event_kind="point",
                        modifiers=modifiers,
                        comment=comment,
                    )
                )
            else:
                comment_start = (
                    str(lowered.get("comment start") or "").strip() or comment
                )
                comment_stop = str(lowered.get("comment stop") or "").strip() or ""
                parsed.append(
                    ParsedEventRow(
                        line_number=line_no,
                        recording_time=start_sec,
                        raw_recording_time=start_str,
                        trial_time=start_sec,
                        subject=subject,
                        has_explicit_subject=has_explicit_subject,
                        behavior=behavior,
                        raw_event="START",
                        event_kind="start",
                        modifiers=modifiers,
                        comment=comment_start,
                    )
                )
                parsed.append(
                    ParsedEventRow(
                        line_number=line_no,
                        recording_time=stop_sec,
                        raw_recording_time=stop_str,
                        trial_time=stop_sec,
                        subject=subject,
                        has_explicit_subject=has_explicit_subject,
                        behavior=behavior,
                        raw_event="STOP",
                        event_kind="end",
                        modifiers=modifiers,
                        comment=comment_stop,
                    )
                )
            continue

        raw_rec = None
        for time_key in ("recording time", "timestamp", "time", "time (s)"):
            if time_key in lowered:
                raw_rec = lowered.get(time_key)
                break
        raw_rec_str = "" if raw_rec is None else str(raw_rec).strip()
        rec_sec = parse_recording_seconds(raw_rec)

        raw_trial = lowered.get("trial time")
        trial_sec = parse_recording_seconds(raw_trial)

        raw_event_val = None
        for ev_key in ("event", "status", "event type"):
            if ev_key in lowered:
                raw_event_val = lowered.get(ev_key)
                break
        raw_event = "" if raw_event_val is None else str(raw_event_val).strip()
        event_kind = classify_event_label(raw_event)

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
                comment=comment,
            )
        )
    return parsed


_TIME_HEADER_TOKENS = frozenset({"recording time", "timestamp", "time", "time (s)"})
_EVENT_HEADER_TOKENS = frozenset({"event", "status", "event type"})
_BEHAVIOR_HEADER_TOKENS = frozenset({"behavior", "behavior code"})
_AGG_START_TOKENS = frozenset({"start (s)", "start"})
_AGG_STOP_TOKENS = frozenset({"stop (s)", "stop"})


def _is_valid_event_header_set(headers: set[str]) -> bool:
    has_behavior = bool(headers & _BEHAVIOR_HEADER_TOKENS)
    if not has_behavior:
        return False
    has_tabular = bool(headers & _TIME_HEADER_TOKENS) and bool(
        headers & _EVENT_HEADER_TOKENS
    )
    has_aggregated = bool(headers & _AGG_START_TOKENS) and bool(
        headers & _AGG_STOP_TOKENS
    )
    return has_tabular or has_aggregated


def read_parsed_event_rows(
    path: Path,
    *,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
) -> List[ParsedEventRow]:
    """Read an Annolid or BORIS behavior event CSV/TSV file into :class:`ParsedEventRow` records.

    Raises:
        FileNotFoundError: When ``path`` does not exist.
        ValueError: When required headers are absent.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Behavior events CSV not found: {csv_path}")

    text = csv_path.read_text(encoding="utf-8-sig")
    if not text.strip():
        return []

    raw_lines = text.splitlines()
    delimiter = (
        "\t"
        if csv_path.suffix.lower() == ".tsv"
        or ("\t" in raw_lines[0] and "," not in raw_lines[0])
        else ","
    )

    # Locate the table header row (skipping optional BORIS observation metadata preamble).
    header_line_idx = 0
    found_header = False
    max_scan = min(len(raw_lines), 40)
    for idx in range(max_scan):
        line = raw_lines[idx]
        if not line.strip():
            continue
        cand_delim = "\t" if ("\t" in line and "," not in line) else delimiter
        cells = next(csv.reader([line], delimiter=cand_delim), [])
        norm = {
            str(c).strip().lower() for c in cells if c is not None and str(c).strip()
        }
        if _is_valid_event_header_set(norm):
            header_line_idx = idx
            delimiter = cand_delim
            found_header = True
            break

    sliced_text = "\n".join(raw_lines[header_line_idx:])
    reader = csv.DictReader(sliced_text.splitlines(), delimiter=delimiter)
    if reader.fieldnames is None:
        return []

    normalized_headers = {
        str(name).strip().lower() for name in reader.fieldnames if name is not None
    }
    if not found_header and not _is_valid_event_header_set(normalized_headers):
        has_time_col = bool(normalized_headers & _TIME_HEADER_TOKENS)
        missing: List[str] = []
        if not has_time_col:
            missing.append("Recording time")
        if not (normalized_headers & _BEHAVIOR_HEADER_TOKENS):
            missing.append("Behavior")
        if not (normalized_headers & _EVENT_HEADER_TOKENS):
            missing.append("Event")
        raise ValueError(
            f"Behavior CSV {csv_path} is missing required column(s): {', '.join(missing)}"
        )

    raw_rows: List[Dict[str, object]] = []
    start_row_no = header_line_idx + 2
    for row_idx, row in enumerate(reader, start=start_row_no):
        if not any(str(v or "").strip() for v in row.values()):
            continue
        entry: Dict[str, object] = dict(row)
        entry["_line_number"] = row_idx
        raw_rows.append(entry)

    return parse_event_mapping_rows(
        raw_rows,
        default_subject=default_subject,
        start_line_number=start_row_no,
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
