"""Compute behavior time-budget summaries from Annolid event exports.

The implementation is inspired by BORIS' reporting features and is intended
to give Annolid users quick access to descriptive statistics without
round-tripping through external tools.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, stdev
import math
from typing import Deque, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from annolid.behavior.event_utils import (
    ParsedEventRow,
    classify_event_label,
    parse_event_mapping_rows,
    parse_recording_seconds,
    read_parsed_event_rows,
)
from annolid.core.behavior.spec import ProjectSchema, load_behavior_spec
from annolid.utils.atomic_io import atomic_write_text

__all__ = [
    "BehaviorInterval",
    "TimeBudgetRow",
    "BinnedTimeBudgetRow",
    "BehaviorCooccurrenceRow",
    "BoutDefinition",
    "BoutSummaryRow",
    "aggression_bout_definition",
    "extract_behavior_intervals",
    "summarize_intervals",
    "compute_time_budget",
    "compute_binned_time_budget",
    "compute_behavior_cooccurrence",
    "compute_bout_summary",
    "format_time_budget_table",
    "format_binned_time_budget_table",
    "format_cooccurrence_table",
    "format_bout_summary_table",
    "write_time_budget_csv",
    "write_cooccurrence_csv",
    "write_bout_summary_csv",
    "TimeBudgetComputationError",
]


DefaultLabel = "Subject 1"


@dataclass(frozen=True)
class BehaviorInterval:
    """Closed interval describing a single behavior occurrence."""

    subject: str
    behavior: str
    start: float
    end: float
    modifiers: Tuple[str, ...] = ()
    comment: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class TimeBudgetRow:
    """Aggregate metrics for a subject/behavior pair (BORIS-compatible)."""

    subject: str
    behavior: str
    occurrences: int
    total_duration: float
    mean_duration: float
    median_duration: float
    min_duration: float
    max_duration: float
    percent_of_session: float
    std_duration: float = 0.0
    rate_per_min: float = 0.0
    latency_to_first: float = 0.0
    inter_event_interval_mean: Optional[float] = None
    inter_event_interval_std: Optional[float] = None


class TimeBudgetComputationError(Exception):
    """Raised when time-budget analysis fails due to malformed data."""


@dataclass(frozen=True)
class BinnedTimeBudgetRow:
    """Duration and occurrence count of a behavior within a specific time bin."""

    subject: str
    behavior: str
    bin_start: float
    bin_end: float
    duration: float
    occurrences: int = 0


@dataclass(frozen=True)
class BehaviorCooccurrenceRow:
    """Temporal co-occurrence metrics between two subject/behavior streams."""

    subject_a: str
    behavior_a: str
    subject_b: str
    behavior_b: str
    overlap_bouts: int
    overlap_duration: float
    duration_a: float
    duration_b: float
    pct_of_a: float
    pct_of_b: float
    jaccard_index: float


@dataclass(frozen=True)
class BoutDefinition:
    """Describe how event intervals should be grouped into bouts."""

    name: str
    behaviors: Tuple[str, ...]
    max_gap_seconds: float = 0.0
    initiation_behaviors: Tuple[str, ...] = ()


@dataclass(frozen=True)
class BoutSummaryRow:
    """Per-subject bout-level summary with behavior counts."""

    subject: str
    bout_name: str
    bout_count: int
    total_duration: float
    mean_duration: float
    median_duration: float
    min_duration: float
    max_duration: float
    behavior_counts: Dict[str, int]
    initiation_count: int


def aggression_bout_definition(
    *,
    max_gap_seconds: float = 2.0,
) -> BoutDefinition:
    """Default aggression-bout grouping used for social-conflict analysis."""

    return BoutDefinition(
        name="aggression_bout",
        behaviors=("slap_in_face", "run_away", "fight_initiation"),
        max_gap_seconds=float(max_gap_seconds),
        initiation_behaviors=("fight_initiation",),
    )


def _safe_float(value: object) -> Optional[float]:
    return parse_recording_seconds(value)


def _normalize_subject(value: Optional[str]) -> str:
    value = (str(value) if value is not None else "").strip()
    return value if value else DefaultLabel


def _normalize_row_keys(row: Dict[str, object]) -> Dict[str, object]:
    return {
        str(key).strip().lower(): (val.strip() if isinstance(val, str) else val)
        for key, val in row.items()
        if key is not None
    }


def _extract_recording_time(row: Dict[str, object]) -> Optional[float]:
    for key in ("recording time", "timestamp", "time", "time (s)"):
        if key in row:
            return _safe_float(row.get(key))
    return None


def _read_event_rows(path: Path) -> List[Dict[str, str]]:
    resolved = Path(path)
    try:
        parsed_rows = read_parsed_event_rows(resolved)
    except ValueError:
        parsed_rows = []
    if parsed_rows:
        out: List[Dict[str, str]] = []
        for r in parsed_rows:
            out.append(
                {
                    "recording time": r.raw_recording_time
                    if r.recording_time is None
                    else str(r.recording_time),
                    "subject": r.subject,
                    "behavior": r.behavior,
                    "event": r.raw_event,
                    "modifiers": "|".join(r.modifiers),
                    "comment": r.comment,
                }
            )
        return out

    with resolved.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = []
        for row in reader:
            cleaned = {
                str(key).strip().lower(): (val.strip() if isinstance(val, str) else val)
                for key, val in row.items()
                if key is not None
            }
            rows.append(cleaned)  # type: ignore[arg-type]
    return rows


def extract_behavior_intervals(
    rows: str | Path | Iterable[Dict[str, object] | ParsedEventRow],
) -> Tuple[List[BehaviorInterval], List[str]]:
    """Convert exported rows (or a CSV/TSV path) into balanced behavior intervals."""

    if isinstance(rows, (str, Path)):
        rows = _read_event_rows(Path(rows))

    raw_list = list(rows)
    if raw_list and all(isinstance(x, ParsedEventRow) for x in raw_list):
        dict_rows: List[Dict[str, object]] = [
            {
                "recording time": r.raw_recording_time
                if r.recording_time is None
                else r.recording_time,
                "subject": r.subject,
                "behavior": r.behavior,
                "event": r.raw_event,
                "modifiers": r.modifiers,
                "comment": r.comment,
            }
            for r in raw_list  # type: ignore[union-attr]
        ]
    else:
        mapping_candidates = [
            dict(r) if isinstance(r, Mapping) else {} for r in raw_list
        ]
        # Expand BORIS Aggregated rows (Start (s), Stop (s), Behavior type) if present.
        has_agg = any(
            any(str(k).strip().lower() in {"start (s)", "start"} for k in m)
            and not any(
                str(k).strip().lower() in {"event", "status", "event type"} for k in m
            )
            for m in mapping_candidates
        )
        if has_agg:
            expanded = parse_event_mapping_rows(mapping_candidates)
            dict_rows = [
                {
                    "recording time": r.raw_recording_time
                    if r.recording_time is None
                    else r.recording_time,
                    "subject": r.subject,
                    "behavior": r.behavior,
                    "event": r.raw_event,
                    "modifiers": r.modifiers,
                    "comment": r.comment,
                }
                for r in expanded
            ]
        else:
            dict_rows = mapping_candidates

    intervals: List[BehaviorInterval] = []
    warnings: List[str] = []

    valid_by_key: Dict[
        Tuple[str, str], List[Tuple[float, str, int, Tuple[str, ...], str]]
    ] = defaultdict(list)

    normalized_rows = [_normalize_row_keys(dict(row)) for row in dict_rows]
    sorted_rows = sorted(
        enumerate(normalized_rows),
        key=lambda item: (
            _extract_recording_time(item[1]) or 0.0,
            item[0],
        ),
    )

    for order_idx, row in sorted_rows:
        raw_behavior = (
            row.get("behavior") if "behavior" in row else row.get("behavior code")
        )
        behavior = (
            str(raw_behavior).strip()
            if raw_behavior is not None and str(raw_behavior).strip()
            else None
        )
        raw_subj = row.get("subject") if "subject" in row else row.get("subject name")
        subject = _normalize_subject(raw_subj)  # type: ignore[arg-type]
        recording_time = _extract_recording_time(row)
        label = (
            row.get("event")
            if "event" in row
            else row.get("status", row.get("event type"))
        )

        raw_mods = row.get("modifiers", row.get("modifier"))
        if isinstance(raw_mods, (list, tuple)):
            mods = tuple(str(m).strip() for m in raw_mods if str(m).strip())
        elif isinstance(raw_mods, str) and raw_mods.strip():
            mods = tuple(
                p.strip()
                for p in raw_mods.replace("|", ",").replace(";", ",").split(",")
                if p.strip()
            )
        else:
            mods = ()
        comment = str(row.get("comment") or "").strip()

        if behavior is None or recording_time is None or label is None:
            warnings.append(
                f"Skipping row with missing fields: behavior={raw_behavior}, recording_time={recording_time}, label={label}"
            )
            continue

        event_type = classify_event_label(str(label))
        if event_type is None:
            warnings.append(
                f"Ignoring unrecognised event label '{label}' for subject '{subject}', behavior '{behavior}'."
            )
            continue
        if event_type == "point":
            intervals.append(
                BehaviorInterval(
                    subject=subject,
                    behavior=behavior,
                    start=recording_time,
                    end=recording_time,
                    modifiers=mods,
                    comment=comment,
                )
            )
            continue

        valid_by_key[(subject, behavior)].append(
            (recording_time, event_type, order_idx, mods, comment)
        )

    epsilon = 1e-9
    for (subject, behavior), series in valid_by_key.items():
        series.sort(key=lambda item: (item[0], item[2]))
        open_starts: Deque[Tuple[float, Tuple[str, ...], str]] = deque()
        idx = 0
        n_items = len(series)
        while idx < n_items:
            ts = series[idx][0]
            starts_at_ts: Deque[Tuple[float, Tuple[str, ...], str]] = deque()
            ends_at_ts: Deque[Tuple[float, Tuple[str, ...], str]] = deque()
            while idx < n_items and abs(series[idx][0] - ts) <= epsilon:
                rec_t, ev_kind, _, mods, comment = series[idx]
                if ev_kind == "start":
                    starts_at_ts.append((rec_t, mods, comment))
                else:
                    ends_at_ts.append((rec_t, mods, comment))
                idx += 1

            # Close intervals started prior to `ts` before opening abutting starts at `ts`.
            while open_starts and ends_at_ts:
                start_time, s_mods, s_comment = open_starts.popleft()
                end_time, e_mods, e_comment = ends_at_ts.popleft()
                intervals.append(
                    BehaviorInterval(
                        subject=subject,
                        behavior=behavior,
                        start=start_time,
                        end=end_time,
                        modifiers=s_mods or e_mods,
                        comment=s_comment or e_comment,
                    )
                )

            # Pair same-timestamp zero-duration (start, end) events.
            while starts_at_ts and ends_at_ts:
                start_time, s_mods, s_comment = starts_at_ts.popleft()
                end_time, e_mods, e_comment = ends_at_ts.popleft()
                intervals.append(
                    BehaviorInterval(
                        subject=subject,
                        behavior=behavior,
                        start=start_time,
                        end=end_time,
                        modifiers=s_mods or e_mods,
                        comment=s_comment or e_comment,
                    )
                )

            while ends_at_ts:
                orphan_t, _, _ = ends_at_ts.popleft()
                warnings.append(
                    f"Encountered end event without matching start for subject '{subject}', behavior '{behavior}' at {orphan_t:.3f}s."
                )

            while starts_at_ts:
                open_starts.append(starts_at_ts.popleft())

        if open_starts:
            warnings.append(
                f"{len(open_starts)} unclosed start event(s) for subject '{subject}', behavior '{behavior}'."
            )

    intervals.sort(key=lambda item: (item.start, item.end, item.subject, item.behavior))
    return intervals, warnings


def _merge_overlapping_intervals(
    items: Sequence[BehaviorInterval],
) -> List[Tuple[float, float]]:
    positive = sorted(
        [(float(i.start), float(i.end)) for i in items if i.end > i.start],
        key=lambda pair: (pair[0], pair[1]),
    )
    if not positive:
        return []
    merged: List[Tuple[float, float]] = [positive[0]]
    for s_val, e_val in positive[1:]:
        prev_s, prev_e = merged[-1]
        if s_val <= prev_e + 1e-9:
            merged[-1] = (prev_s, max(prev_e, e_val))
        else:
            merged.append((s_val, e_val))
    return merged


def _clip_intervals_to_window(
    intervals: Sequence[BehaviorInterval],
    *,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> List[BehaviorInterval]:
    if start_time is None and end_time is None:
        return list(intervals)
    lo = float("-inf") if start_time is None else float(start_time)
    hi = float("inf") if end_time is None else float(end_time)
    if hi < lo - 1e-9:
        raise ValueError(f"end_time ({hi}) must not precede start_time ({lo}).")
    clipped: List[BehaviorInterval] = []
    epsilon = 1e-9
    for iv in intervals:
        if abs(iv.end - iv.start) <= epsilon:
            # Point event
            if lo - epsilon <= iv.start <= hi + epsilon:
                clamped_t = min(max(iv.start, lo), hi)
                clipped.append(
                    BehaviorInterval(
                        subject=iv.subject,
                        behavior=iv.behavior,
                        start=clamped_t,
                        end=clamped_t,
                        modifiers=iv.modifiers,
                        comment=iv.comment,
                    )
                )
        else:
            cs = max(lo, iv.start)
            ce = min(hi, iv.end)
            if ce - cs > epsilon:
                clipped.append(
                    BehaviorInterval(
                        subject=iv.subject,
                        behavior=iv.behavior,
                        start=cs,
                        end=ce,
                        modifiers=iv.modifiers,
                        comment=iv.comment,
                    )
                )
    return clipped


def summarize_intervals(
    intervals: Iterable[BehaviorInterval],
    *,
    session_duration: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exclude_behaviors: Optional[Sequence[str]] = None,
    omit_excluded_rows: bool = False,
) -> List[TimeBudgetRow]:
    """Aggregate BORIS-compatible time-budget statistics per subject/behavior.

    Args:
        intervals: Balanced behavior intervals (state spans and zero-duration points).
        session_duration: Explicit total session duration in seconds.
        start_time: Optional analysis window start in seconds (clips state intervals).
        end_time: Optional analysis window end in seconds (clips state intervals).
        exclude_behaviors: Optional behavior codes (e.g. ``["out_of_sight"]``) whose
            per-subject union duration is subtracted from the observation denominator
            when computing ``percent_of_session`` and ``rate_per_min`` (BORIS parity).
        omit_excluded_rows: When True, omit ``exclude_behaviors`` from returned rows.
    """
    raw_list = list(intervals)
    if not raw_list:
        return []

    interval_list = _clip_intervals_to_window(
        raw_list, start_time=start_time, end_time=end_time
    )
    if not interval_list:
        return []

    grouped: Dict[Tuple[str, str], List[BehaviorInterval]] = defaultdict(list)
    for interval in interval_list:
        grouped[(interval.subject, interval.behavior)].append(interval)

    win_start = (
        float(start_time)
        if start_time is not None
        else min(i.start for i in interval_list)
    )
    win_end = (
        float(end_time) if end_time is not None else max(i.end for i in interval_list)
    )
    observed_span = max(0.0, win_end - win_start)
    base_session_duration = (
        max(0.0, float(session_duration))
        if session_duration is not None and float(session_duration) > 0.0
        else observed_span
    )

    excl_set = {str(b).strip() for b in (exclude_behaviors or ()) if str(b).strip()}
    excluded_duration_by_subject: Dict[str, float] = {}
    if excl_set:
        by_subj_excl: Dict[str, List[BehaviorInterval]] = defaultdict(list)
        for iv in interval_list:
            if iv.behavior in excl_set:
                by_subj_excl[iv.subject].append(iv)
        for subj, excl_ivs in by_subj_excl.items():
            merged_excl = _merge_overlapping_intervals(excl_ivs)
            excluded_duration_by_subject[subj] = sum(
                e_val - s_val for s_val, e_val in merged_excl
            )

    summary: List[TimeBudgetRow] = []
    for (subject, behavior), items in sorted(grouped.items()):
        if omit_excluded_rows and behavior in excl_set:
            continue
        ordered_items = sorted(items, key=lambda x: (x.start, x.end))
        durations = [i.duration for i in ordered_items if i.duration >= 0.0]
        if not durations:
            continue
        total_duration = sum(durations)
        std_dur = float(stdev(durations)) if len(durations) >= 2 else 0.0
        merged_spans = _merge_overlapping_intervals(ordered_items)
        union_duration = sum(e_val - s_val for s_val, e_val in merged_spans)

        # Subtract out-of-sight / excluded time for non-excluded behaviors
        subj_excl_dur = (
            0.0
            if behavior in excl_set
            else excluded_duration_by_subject.get(subject, 0.0)
        )
        eff_duration = max(0.0, base_session_duration - subj_excl_dur)

        percent = (
            min(100.0, union_duration / eff_duration * 100.0)
            if eff_duration > 0
            else 0.0
        )
        rate_min = (len(durations) / (eff_duration / 60.0)) if eff_duration > 0 else 0.0
        latency_first = max(0.0, float(ordered_items[0].start) - win_start)

        iei_mean: Optional[float] = None
        iei_std: Optional[float] = None
        if len(ordered_items) >= 2:
            ieis = [
                max(0.0, float(nxt.start) - float(prv.end))
                for prv, nxt in zip(ordered_items, ordered_items[1:])
            ]
            iei_mean = float(mean(ieis))
            iei_std = float(stdev(ieis)) if len(ieis) >= 2 else 0.0

        summary.append(
            TimeBudgetRow(
                subject=subject,
                behavior=behavior,
                occurrences=len(durations),
                total_duration=total_duration,
                mean_duration=mean(durations),
                median_duration=median(durations),
                min_duration=min(durations),
                max_duration=max(durations),
                percent_of_session=percent,
                std_duration=std_dur,
                rate_per_min=rate_min,
                latency_to_first=latency_first,
                inter_event_interval_mean=iei_mean,
                inter_event_interval_std=iei_std,
            )
        )

    return summary


def compute_time_budget(
    rows: str | Path | Iterable[Dict[str, object] | ParsedEventRow],
    *,
    session_duration: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exclude_behaviors: Optional[Sequence[str]] = None,
    omit_excluded_rows: bool = False,
) -> Tuple[List[TimeBudgetRow], List[str]]:
    """Compute aggregated time-budget metrics from raw event rows or a CSV/TSV path."""
    intervals, warnings = extract_behavior_intervals(rows)
    summary = summarize_intervals(
        intervals,
        session_duration=session_duration,
        start_time=start_time,
        end_time=end_time,
        exclude_behaviors=exclude_behaviors,
        omit_excluded_rows=omit_excluded_rows,
    )
    return summary, warnings


def summarize_by_category(
    rows: Iterable[TimeBudgetRow], schema: ProjectSchema
) -> List[Tuple[str, float, int]]:
    """Aggregate time-budget rows by category using the provided schema."""
    category_map = schema.category_map()
    behavior_map = schema.behavior_map()
    totals: Dict[str, Tuple[float, int]] = {}

    for row in rows:
        behavior = behavior_map.get(row.behavior)
        category_id = behavior.category_id if behavior else None
        if category_id is None:
            category_id = "uncategorized"
        total_duration, occurrences = totals.get(category_id, (0.0, 0))
        totals[category_id] = (
            total_duration + row.total_duration,
            occurrences + row.occurrences,
        )

    summary: List[Tuple[str, float, int]] = []
    for category_id, (total_duration, occurrences) in totals.items():
        if category_id == "uncategorized":
            name = "Uncategorized"
        else:
            category = category_map.get(category_id)
            name = category.name if category else category_id
        summary.append((name, total_duration, occurrences))
    summary.sort(key=lambda item: item[0].lower())
    return summary


def compute_binned_time_budget(
    intervals: Iterable[BehaviorInterval],
    bin_size: float,
    *,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> List[BinnedTimeBudgetRow]:
    """Compute per-bin durations and occurrence counts for state AND point behaviors."""

    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    raw_list = list(intervals)
    if not raw_list:
        return []

    interval_list = _clip_intervals_to_window(
        raw_list, start_time=start_time, end_time=end_time
    )
    if not interval_list:
        return []

    session_start = (
        float(start_time)
        if start_time is not None
        else min(i.start for i in interval_list)
    )
    session_end = (
        float(end_time) if end_time is not None else max(i.end for i in interval_list)
    )
    epsilon = 1e-9
    if session_end <= session_start + epsilon:
        # Single-instant session (e.g., point events all at one timestamp)
        session_end = session_start + bin_size

    n_bins = max(1, int(math.ceil((session_end - session_start - epsilon) / bin_size)))
    bin_totals: Dict[Tuple[str, str, float, float], float] = defaultdict(float)
    bin_counts: Dict[Tuple[str, str, float, float], int] = defaultdict(int)

    grouped_for_bins: Dict[Tuple[str, str], List[BehaviorInterval]] = defaultdict(list)
    for interval in interval_list:
        grouped_for_bins[(interval.subject, interval.behavior)].append(interval)

    for (subject, behavior), items in grouped_for_bins.items():
        # 1) Union durations from positive-width state intervals
        for span_start, span_end in _merge_overlapping_intervals(items):
            start_idx = min(
                n_bins - 1,
                max(0, int(math.floor((span_start - session_start) / bin_size))),
            )
            end_idx = min(
                n_bins - 1,
                max(
                    0,
                    int(math.floor((span_end - session_start - epsilon) / bin_size)),
                ),
            )
            for idx in range(start_idx, end_idx + 1):
                bin_start = session_start + idx * bin_size
                bin_end = min(bin_start + bin_size, session_end)
                overlap_start = max(span_start, bin_start)
                overlap_end = min(span_end, bin_end)
                duration = overlap_end - overlap_start
                if duration > epsilon:
                    key = (subject, behavior, bin_start, bin_end)
                    bin_totals[key] += duration

        # 2) Occurrence counts (both state bouts overlapping a bin and point events in a bin)
        for item in items:
            if item.end - item.start <= epsilon:
                idx = min(
                    n_bins - 1,
                    max(0, int(math.floor((item.start - session_start) / bin_size))),
                )
                bin_start = session_start + idx * bin_size
                bin_end = min(bin_start + bin_size, session_end)
                key = (subject, behavior, bin_start, bin_end)
                bin_counts[key] += 1
            else:
                start_idx = min(
                    n_bins - 1,
                    max(0, int(math.floor((item.start - session_start) / bin_size))),
                )
                end_idx = min(
                    n_bins - 1,
                    max(
                        0,
                        int(
                            math.floor((item.end - session_start - epsilon) / bin_size)
                        ),
                    ),
                )
                for idx in range(start_idx, end_idx + 1):
                    bin_start = session_start + idx * bin_size
                    bin_end = min(bin_start + bin_size, session_end)
                    if min(item.end, bin_end) - max(item.start, bin_start) > epsilon:
                        key = (subject, behavior, bin_start, bin_end)
                        bin_counts[key] += 1

    all_keys = set(bin_totals.keys()) | set(bin_counts.keys())
    rows = [
        BinnedTimeBudgetRow(
            subject=subject,
            behavior=behavior,
            bin_start=bin_start,
            bin_end=bin_end,
            duration=bin_totals.get((subject, behavior, bin_start, bin_end), 0.0),
            occurrences=bin_counts.get((subject, behavior, bin_start, bin_end), 0),
        )
        for (subject, behavior, bin_start, bin_end) in all_keys
        if bin_totals.get((subject, behavior, bin_start, bin_end), 0.0) > epsilon
        or bin_counts.get((subject, behavior, bin_start, bin_end), 0) > 0
    ]
    rows.sort(key=lambda r: (r.subject, r.behavior, r.bin_start))
    return rows


def _intersect_span_lists(
    spans_a: Sequence[Tuple[float, float]],
    spans_b: Sequence[Tuple[float, float]],
) -> List[Tuple[float, float]]:
    out: List[Tuple[float, float]] = []
    i = 0
    j = 0
    epsilon = 1e-9
    while i < len(spans_a) and j < len(spans_b):
        sa, ea = spans_a[i]
        sb, eb = spans_b[j]
        lo = max(sa, sb)
        hi = min(ea, eb)
        if hi - lo > epsilon:
            out.append((lo, hi))
        if ea < eb:
            i += 1
        else:
            j += 1
    return out


def compute_behavior_cooccurrence(
    intervals: Iterable[BehaviorInterval],
    *,
    within_subject_only: bool = False,
) -> List[BehaviorCooccurrenceRow]:
    """Compute pairwise temporal overlap across (subject, behavior) state streams."""
    grouped: Dict[Tuple[str, str], List[BehaviorInterval]] = defaultdict(list)
    for iv in intervals:
        if iv.end > iv.start + 1e-9:
            grouped[(iv.subject, iv.behavior)].append(iv)

    keys = sorted(grouped.keys())
    merged_by_key = {k: _merge_overlapping_intervals(grouped[k]) for k in keys}
    dur_by_key = {k: sum(e - s for s, e in spans) for k, spans in merged_by_key.items()}

    rows: List[BehaviorCooccurrenceRow] = []
    for idx_a in range(len(keys)):
        ka = keys[idx_a]
        for idx_b in range(idx_a + 1, len(keys)):
            kb = keys[idx_b]
            if within_subject_only and ka[0] != kb[0]:
                continue
            overlaps = _intersect_span_lists(merged_by_key[ka], merged_by_key[kb])
            if not overlaps:
                continue
            ov_dur = sum(e - s for s, e in overlaps)
            da = dur_by_key[ka]
            db = dur_by_key[kb]
            union_dur = max(1e-12, da + db - ov_dur)
            rows.append(
                BehaviorCooccurrenceRow(
                    subject_a=ka[0],
                    behavior_a=ka[1],
                    subject_b=kb[0],
                    behavior_b=kb[1],
                    overlap_bouts=len(overlaps),
                    overlap_duration=ov_dur,
                    duration_a=da,
                    duration_b=db,
                    pct_of_a=(100.0 * ov_dur / da) if da > 0 else 0.0,
                    pct_of_b=(100.0 * ov_dur / db) if db > 0 else 0.0,
                    jaccard_index=ov_dur / union_dur,
                )
            )
    rows.sort(key=lambda r: (-r.overlap_duration, r.subject_a, r.behavior_a))
    return rows


def compute_bout_summary(
    intervals: Iterable[BehaviorInterval],
    *,
    definition: BoutDefinition,
) -> List[BoutSummaryRow]:
    """Group selected behavior intervals into bouts and summarise counts."""

    behavior_filter = {
        str(name).strip() for name in definition.behaviors if str(name).strip()
    }
    if not behavior_filter:
        return []

    max_gap = max(0.0, float(definition.max_gap_seconds))
    initiation_filter = {
        str(name).strip()
        for name in definition.initiation_behaviors
        if str(name).strip()
    }

    per_subject: Dict[str, List[BehaviorInterval]] = defaultdict(list)
    for interval in intervals:
        if interval.behavior in behavior_filter:
            per_subject[interval.subject].append(interval)

    summary_rows: List[BoutSummaryRow] = []
    for subject, subject_intervals in sorted(per_subject.items()):
        if not subject_intervals:
            continue
        ordered = sorted(
            subject_intervals, key=lambda item: (item.start, item.end, item.behavior)
        )

        bouts: List[List[BehaviorInterval]] = []
        current: List[BehaviorInterval] = []
        current_end = 0.0
        for interval in ordered:
            if not current:
                current = [interval]
                current_end = interval.end
                continue
            if interval.start <= current_end + max_gap:
                current.append(interval)
                current_end = max(current_end, interval.end)
            else:
                bouts.append(current)
                current = [interval]
                current_end = interval.end
        if current:
            bouts.append(current)

        if not bouts:
            continue

        durations = [
            max(0.0, max(item.end for item in bout) - min(item.start for item in bout))
            for bout in bouts
        ]
        behavior_counts: Dict[str, int] = {name: 0 for name in definition.behaviors}
        initiation_count = 0
        for bout in bouts:
            in_this_bout = set()
            for item in bout:
                if item.behavior in behavior_counts:
                    behavior_counts[item.behavior] += 1
                in_this_bout.add(item.behavior)
            if initiation_filter and any(
                name in in_this_bout for name in initiation_filter
            ):
                initiation_count += 1

        summary_rows.append(
            BoutSummaryRow(
                subject=subject,
                bout_name=definition.name,
                bout_count=len(bouts),
                total_duration=sum(durations),
                mean_duration=mean(durations),
                median_duration=median(durations),
                min_duration=min(durations),
                max_duration=max(durations),
                behavior_counts=behavior_counts,
                initiation_count=initiation_count,
            )
        )

    return summary_rows


def format_time_budget_table(
    rows: Sequence[TimeBudgetRow], schema: Optional[ProjectSchema] = None
) -> str:
    if not rows:
        return "No completed behavior intervals were found."

    include_category = schema is not None
    headers = [
        "Subject",
        "Behavior",
        "Occurrences",
        "Total (s)",
        "Mean (s)",
        "SD (s)",
        "Median (s)",
        "Min (s)",
        "Max (s)",
        "% Session",
        "Rate (/min)",
        "Latency (s)",
        "Mean IEI (s)",
    ]
    if include_category:
        headers.insert(2, "Category")

    behavior_map = schema.behavior_map() if schema else {}
    category_map = schema.category_map() if schema else {}

    str_rows: List[List[str]] = [
        [
            row.subject,
            row.behavior,
            str(row.occurrences),
            f"{row.total_duration:.2f}",
            f"{row.mean_duration:.2f}",
            f"{row.std_duration:.2f}",
            f"{row.median_duration:.2f}",
            f"{row.min_duration:.2f}",
            f"{row.max_duration:.2f}",
            f"{row.percent_of_session:.2f}",
            f"{row.rate_per_min:.2f}",
            f"{row.latency_to_first:.2f}",
            f"{row.inter_event_interval_mean:.2f}"
            if row.inter_event_interval_mean is not None
            else "—",
        ]
        for row in rows
    ]

    if include_category:
        for row, formatted in zip(rows, str_rows):
            behavior = behavior_map.get(row.behavior)
            category_name = None
            if behavior and behavior.category_id:
                category = category_map.get(behavior.category_id)
                category_name = category.name if category else behavior.category_id
            formatted.insert(2, category_name or "—")

    col_widths = [len(header) for header in headers]
    for r in str_rows:
        for idx, value in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(value))

    def fmt_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(col_widths[idx]) for idx, value in enumerate(values)
        )

    lines = [fmt_row(headers), fmt_row(["-" * len(h) for h in headers])]
    lines.extend(fmt_row(r) for r in str_rows)
    return "\n".join(lines)


def format_binned_time_budget_table(rows: Sequence[BinnedTimeBudgetRow]) -> str:
    if not rows:
        return "No activity fell within the requested time bins."
    headers = [
        "Subject",
        "Behavior",
        "Bin Start (s)",
        "Bin End (s)",
        "Occurrences",
        "Duration (s)",
    ]
    str_rows: List[List[str]] = [
        [
            row.subject,
            row.behavior,
            f"{row.bin_start:.2f}",
            f"{row.bin_end:.2f}",
            str(row.occurrences),
            f"{row.duration:.2f}",
        ]
        for row in rows
    ]

    col_widths = [len(header) for header in headers]
    for r in str_rows:
        for idx, value in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(value))

    def fmt_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(col_widths[idx]) for idx, value in enumerate(values)
        )

    lines = [fmt_row(headers), fmt_row(["-" * len(h) for h in headers])]
    lines.extend(fmt_row(r) for r in str_rows)
    return "\n".join(lines)


def format_cooccurrence_table(rows: Sequence[BehaviorCooccurrenceRow]) -> str:
    if not rows:
        return "No overlapping behavior intervals were found."
    headers = [
        "Subject A",
        "Behavior A",
        "Subject B",
        "Behavior B",
        "Overlap Bouts",
        "Overlap (s)",
        "% of A",
        "% of B",
        "Jaccard",
    ]
    str_rows = [
        [
            r.subject_a,
            r.behavior_a,
            r.subject_b,
            r.behavior_b,
            str(r.overlap_bouts),
            f"{r.overlap_duration:.2f}",
            f"{r.pct_of_a:.1f}%",
            f"{r.pct_of_b:.1f}%",
            f"{r.jaccard_index:.3f}",
        ]
        for r in rows
    ]
    col_widths = [len(header) for header in headers]
    for r in str_rows:
        for idx, value in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(value))

    def fmt_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(col_widths[idx]) for idx, value in enumerate(values)
        )

    lines = [fmt_row(headers), fmt_row(["-" * len(h) for h in headers])]
    lines.extend(fmt_row(r) for r in str_rows)
    return "\n".join(lines)


def format_bout_summary_table(
    rows: Sequence[BoutSummaryRow],
    *,
    behavior_order: Optional[Sequence[str]] = None,
) -> str:
    if not rows:
        return "No bouts were detected for the requested behavior set."

    if behavior_order:
        behavior_columns = [str(name) for name in behavior_order if str(name).strip()]
    else:
        seen: List[str] = []
        for row in rows:
            for name in row.behavior_counts.keys():
                if name not in seen:
                    seen.append(name)
        behavior_columns = seen

    headers = [
        "Subject",
        "Bout",
        "Bouts",
        "Total (s)",
        "Mean (s)",
        "Median (s)",
        "Min (s)",
        "Max (s)",
    ]
    headers.extend(f"{name} Count" for name in behavior_columns)
    headers.append("Initiation Bouts")

    str_rows: List[List[str]] = []
    for row in rows:
        values = [
            row.subject,
            row.bout_name,
            str(row.bout_count),
            f"{row.total_duration:.2f}",
            f"{row.mean_duration:.2f}",
            f"{row.median_duration:.2f}",
            f"{row.min_duration:.2f}",
            f"{row.max_duration:.2f}",
        ]
        values.extend(
            str(int(row.behavior_counts.get(name, 0))) for name in behavior_columns
        )
        values.append(str(int(row.initiation_count)))
        str_rows.append(values)

    col_widths = [len(header) for header in headers]
    for row in str_rows:
        for idx, value in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(value))

    def fmt_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(col_widths[idx]) for idx, value in enumerate(values)
        )

    lines = [fmt_row(headers), fmt_row(["-" * len(h) for h in headers])]
    lines.extend(fmt_row(row) for row in str_rows)
    return "\n".join(lines)


def format_category_summary(rows: Sequence[Tuple[str, float, int]]) -> str:
    if not rows:
        return "No category assignments were found."
    headers = ["Category", "Total (s)", "Occurrences"]
    str_rows = [
        [name, f"{total:.2f}", str(occurrences)] for name, total, occurrences in rows
    ]
    col_widths = [len(header) for header in headers]
    for r in str_rows:
        for idx, value in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(value))

    def fmt_row(values: Sequence[str]) -> str:
        return "  ".join(
            value.ljust(col_widths[idx]) for idx, value in enumerate(values)
        )

    lines = [fmt_row(headers), fmt_row(["-" * len(h) for h in headers])]
    lines.extend(fmt_row(r) for r in str_rows)
    return "\n".join(lines)


def write_time_budget_csv(
    rows: Sequence[TimeBudgetRow], path: Path, schema: Optional[ProjectSchema] = None
) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    headers = [
        "Subject",
        "Behavior",
        "Occurrences",
        "TotalSeconds",
        "MeanSeconds",
        "MedianSeconds",
        "MinSeconds",
        "MaxSeconds",
        "PercentOfSession",
        "StdSeconds",
        "RatePerMin",
        "LatencySeconds",
        "MeanIEISeconds",
        "StdIEISeconds",
    ]
    include_category = schema is not None
    behavior_map = schema.behavior_map() if schema else {}
    category_map = schema.category_map() if schema else {}
    if include_category:
        headers.insert(2, "Category")
    writer.writerow(headers)
    for row in rows:
        item = [
            row.subject,
            row.behavior,
            row.occurrences,
            f"{row.total_duration:.6f}",
            f"{row.mean_duration:.6f}",
            f"{row.median_duration:.6f}",
            f"{row.min_duration:.6f}",
            f"{row.max_duration:.6f}",
            f"{row.percent_of_session:.6f}",
            f"{row.std_duration:.6f}",
            f"{row.rate_per_min:.6f}",
            f"{row.latency_to_first:.6f}",
            f"{row.inter_event_interval_mean:.6f}"
            if row.inter_event_interval_mean is not None
            else "",
            f"{row.inter_event_interval_std:.6f}"
            if row.inter_event_interval_std is not None
            else "",
        ]
        if include_category:
            behavior = behavior_map.get(row.behavior)
            category_name = None
            if behavior and behavior.category_id:
                category = category_map.get(behavior.category_id)
                category_name = category.name if category else behavior.category_id
            item.insert(2, category_name or "")
        writer.writerow(item)
    atomic_write_text(Path(path), buffer.getvalue(), encoding="utf-8")


def write_cooccurrence_csv(
    rows: Sequence[BehaviorCooccurrenceRow],
    path: Path,
) -> None:
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "SubjectA",
            "BehaviorA",
            "SubjectB",
            "BehaviorB",
            "OverlapBouts",
            "OverlapSeconds",
            "DurationASeconds",
            "DurationBSeconds",
            "PercentOfA",
            "PercentOfB",
            "JaccardIndex",
        ]
    )
    for r in rows:
        writer.writerow(
            [
                r.subject_a,
                r.behavior_a,
                r.subject_b,
                r.behavior_b,
                int(r.overlap_bouts),
                f"{r.overlap_duration:.6f}",
                f"{r.duration_a:.6f}",
                f"{r.duration_b:.6f}",
                f"{r.pct_of_a:.6f}",
                f"{r.pct_of_b:.6f}",
                f"{r.jaccard_index:.6f}",
            ]
        )
    atomic_write_text(Path(path), buffer.getvalue(), encoding="utf-8")


def write_bout_summary_csv(
    rows: Sequence[BoutSummaryRow],
    path: Path,
    *,
    behavior_order: Optional[Sequence[str]] = None,
) -> None:
    if behavior_order:
        behavior_columns = [str(name) for name in behavior_order if str(name).strip()]
    else:
        seen: List[str] = []
        for row in rows:
            for name in row.behavior_counts.keys():
                if name not in seen:
                    seen.append(name)
        behavior_columns = seen

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    headers = [
        "Subject",
        "Bout",
        "Bouts",
        "TotalSeconds",
        "MeanSeconds",
        "MedianSeconds",
        "MinSeconds",
        "MaxSeconds",
    ]
    headers.extend(f"{name}Count" for name in behavior_columns)
    headers.append("InitiationBouts")
    writer.writerow(headers)

    for row in rows:
        values: List[object] = [
            row.subject,
            row.bout_name,
            int(row.bout_count),
            f"{row.total_duration:.6f}",
            f"{row.mean_duration:.6f}",
            f"{row.median_duration:.6f}",
            f"{row.min_duration:.6f}",
            f"{row.max_duration:.6f}",
        ]
        values.extend(
            int(row.behavior_counts.get(name, 0)) for name in behavior_columns
        )
        values.append(int(row.initiation_count))
        writer.writerow(values)
    atomic_write_text(Path(path), buffer.getvalue(), encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute time-budget summaries from Annolid behavior event CSV exports."
    )
    parser.add_argument(
        "events_csv",
        type=Path,
        help="Path to a CSV file produced by the Annolid behavior export.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Optional path to write the summary as CSV. When omitted the table is printed to stdout.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail with non-zero exit code if unmatched start/end events are encountered.",
    )
    parser.add_argument(
        "--bin-size",
        type=float,
        help="Optional bin width (in seconds) for a secondary time-budget table.",
    )
    parser.add_argument(
        "--schema",
        type=Path,
        help="Optional project schema file used to enrich summaries (JSON or YAML).",
    )
    parser.add_argument(
        "--bout-profile",
        choices=("aggression",),
        help="Optional bout preset. 'aggression' groups slap/run-away/fight-initiation intervals into bouts.",
    )
    parser.add_argument(
        "--bout-name",
        default="custom_bout",
        help="Name for a custom bout definition (used with --bout-behaviors).",
    )
    parser.add_argument(
        "--bout-behaviors",
        help="Comma-separated behaviors to include in bout grouping (for example: slap_in_face,run_away,fight_initiation).",
    )
    parser.add_argument(
        "--bout-initiation-behaviors",
        help="Comma-separated behaviors that count as bout initiations.",
    )
    parser.add_argument(
        "--bout-gap-seconds",
        type=float,
        default=None,
        help="Maximum allowed gap between selected behavior intervals within the same bout.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if not args.events_csv.exists():
        parser.error(f"Events file not found: {args.events_csv}")

    try:
        rows = _read_event_rows(args.events_csv)
        intervals, warnings = extract_behavior_intervals(rows)
    except OSError as exc:
        raise TimeBudgetComputationError(f"Failed to read events file: {exc}") from exc

    schema: Optional[ProjectSchema] = None
    if args.schema is not None:
        if not args.schema.exists():
            parser.error(f"Schema file not found: {args.schema}")
            return 2
        try:
            schema, _ = load_behavior_spec(path=args.schema)
        except Exception as exc:
            parser.error(f"Failed to load schema {args.schema}: {exc}")
            return 2

    summary = summarize_intervals(intervals)

    try:
        binned_rows = (
            compute_binned_time_budget(intervals, args.bin_size)
            if args.bin_size is not None
            else None
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    bout_definition: Optional[BoutDefinition] = None
    if args.bout_profile == "aggression":
        profile_gap = (
            2.0 if args.bout_gap_seconds is None else float(args.bout_gap_seconds)
        )
        profile_gap = max(0.0, profile_gap)
        bout_definition = aggression_bout_definition(max_gap_seconds=profile_gap)
    elif args.bout_behaviors:
        behavior_names = tuple(
            item.strip() for item in str(args.bout_behaviors).split(",") if item.strip()
        )
        if not behavior_names:
            parser.error(
                "--bout-behaviors must contain at least one non-empty behavior."
            )
            return 2
        initiation_names = tuple(
            item.strip()
            for item in str(args.bout_initiation_behaviors or "").split(",")
            if item.strip()
        )
        bout_definition = BoutDefinition(
            name=str(args.bout_name).strip() or "custom_bout",
            behaviors=behavior_names,
            max_gap_seconds=max(
                0.0,
                0.0 if args.bout_gap_seconds is None else float(args.bout_gap_seconds),
            ),
            initiation_behaviors=initiation_names,
        )
    elif args.bout_initiation_behaviors:
        parser.error(
            "--bout-initiation-behaviors requires --bout-behaviors or --bout-profile."
        )
        return 2

    bout_rows: Optional[List[BoutSummaryRow]] = None
    if bout_definition is not None:
        bout_rows = compute_bout_summary(intervals, definition=bout_definition)

    for warning in warnings:
        print(f"[warning] {warning}", file=sys.stderr)

    if args.strict and warnings:
        return 2

    if args.output:
        write_time_budget_csv(summary, args.output, schema=schema)
    else:
        print(format_time_budget_table(summary, schema=schema))

    if schema is not None and summary:
        category_summary = summarize_by_category(summary, schema)
        if args.output:
            category_path = args.output.with_name(
                args.output.stem + "_categories" + args.output.suffix
            )
            with category_path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["Category", "TotalSeconds", "Occurrences"])
                for name, total, occurrences in category_summary:
                    writer.writerow([name, f"{total:.6f}", occurrences])
        else:
            print()
            print("Category Summary")
            print(format_category_summary(category_summary))

    if binned_rows is not None:
        if not args.output:
            print()
        print(f"Binned time-budget (bin size = {args.bin_size:.2f}s)")
        print(format_binned_time_budget_table(binned_rows))

    if bout_rows is not None:
        if args.output:
            bout_path = args.output.with_name(
                args.output.stem + "_bouts" + args.output.suffix
            )
            write_bout_summary_csv(
                bout_rows,
                bout_path,
                behavior_order=bout_definition.behaviors,
            )
        else:
            print()
            print(
                f"Bout summary ({bout_definition.name}, gap <= {bout_definition.max_gap_seconds:.2f}s)"
            )
            print(
                format_bout_summary_table(
                    bout_rows,
                    behavior_order=bout_definition.behaviors,
                )
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
