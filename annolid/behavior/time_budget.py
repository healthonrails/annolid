"""Compute behavior time-budget summaries from Annolid event exports.

The implementation is inspired by BORIS' reporting features and is intended
to give Annolid users quick access to descriptive statistics without
round-tripping through external tools.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median
import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from annolid.behavior.event_utils import normalize_event_label
from annolid.behavior.project_schema import ProjectSchema, load_schema as load_project_schema

__all__ = [
    "BehaviorInterval",
    "TimeBudgetRow",
    "BinnedTimeBudgetRow",
    "extract_behavior_intervals",
    "summarize_intervals",
    "compute_time_budget",
    "compute_binned_time_budget",
    "format_time_budget_table",
    "format_binned_time_budget_table",
    "write_time_budget_csv",
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

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(frozen=True)
class TimeBudgetRow:
    """Aggregate metrics for a subject/behavior pair."""

    subject: str
    behavior: str
    occurrences: int
    total_duration: float
    mean_duration: float
    median_duration: float
    min_duration: float
    max_duration: float
    percent_of_session: float


class TimeBudgetComputationError(Exception):
    """Raised when time-budget analysis fails due to malformed data."""


@dataclass(frozen=True)
class BinnedTimeBudgetRow:
    """Duration of a behavior within a specific time bin."""

    subject: str
    behavior: str
    bin_start: float
    bin_end: float
    duration: float


def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_subject(value: Optional[str]) -> str:
    value = (value or "").strip()
    return value if value else DefaultLabel


def _read_event_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows: List[Dict[str, str]] = []
        for row in reader:
            cleaned = {key.strip().lower(): (val.strip() if isinstance(
                val, str) else val) for key, val in row.items()}
            rows.append(cleaned)
    return rows


def extract_behavior_intervals(rows: Iterable[Dict[str, object]]) -> Tuple[List[BehaviorInterval], List[str]]:
    """Convert exported rows into balanced behavior intervals."""

    intervals: List[BehaviorInterval] = []
    warnings: List[str] = []

    stacks: Dict[Tuple[str, str], List[float]] = defaultdict(list)

    sorted_rows = sorted(
        rows,
        key=lambda r: (_safe_float(r.get("recording time"))
                       or 0.0, str(r.get("event") or "")),
    )

    for row in sorted_rows:
        behavior = row.get("behavior")
        subject = _normalize_subject(row.get("subject"))
        recording_time = _safe_float(row.get("recording time"))
        label = row.get("event")

        if behavior is None or recording_time is None or label is None:
            warnings.append(
                f"Skipping row with missing fields: behavior={behavior}, recording_time={recording_time}, label={label}"
            )
            continue

        event_type = normalize_event_label(str(label))
        if event_type is None:
            warnings.append(
                f"Ignoring unrecognised event label '{label}' for subject '{subject}', behavior '{behavior}'."
            )
            continue

        key = (subject, str(behavior))
        if event_type == "start":
            stacks[key].append(recording_time)
        else:
            if not stacks[key]:
                warnings.append(
                    f"Encountered end event without matching start for subject '{subject}', behavior '{behavior}' at {recording_time:.3f}s."
                )
                continue
            start_time = stacks[key].pop()
            if recording_time < start_time:
                warnings.append(
                    f"End time {recording_time:.3f}s precedes start time {start_time:.3f}s for subject '{subject}', behavior '{behavior}'."
                )
                continue
            intervals.append(
                BehaviorInterval(
                    subject=subject,
                    behavior=str(behavior),
                    start=start_time,
                    end=recording_time,
                )
            )

    for (subject, behavior), pending in stacks.items():
        if pending:
            warnings.append(
                f"{len(pending)} unclosed start event(s) for subject '{subject}', behavior '{behavior}'."
            )

    return intervals, warnings


def summarize_intervals(intervals: Iterable[BehaviorInterval]) -> List[TimeBudgetRow]:
    """Aggregate durations per subject/behavior for the provided intervals."""

    interval_list = list(intervals)
    if not interval_list:
        return []

    grouped: Dict[Tuple[str, str], List[BehaviorInterval]] = defaultdict(list)
    for interval in interval_list:
        grouped[(interval.subject, interval.behavior)].append(interval)

    session_start = min(i.start for i in interval_list)
    session_end = max(i.end for i in interval_list)
    session_duration = max(0.0, session_end - session_start)

    summary: List[TimeBudgetRow] = []
    for (subject, behavior), items in sorted(grouped.items()):
        durations = [i.duration for i in items if i.duration >= 0.0]
        if not durations:
            continue
        total_duration = sum(durations)
        percent = (total_duration / session_duration *
                   100.0) if session_duration > 0 else 0.0
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
            )
        )

    return summary


def compute_time_budget(rows: Iterable[Dict[str, object]]) -> Tuple[List[TimeBudgetRow], List[str]]:
    """Compute aggregated time-budget metrics from raw event rows."""
    intervals, warnings = extract_behavior_intervals(rows)
    summary = summarize_intervals(intervals)
    return summary, warnings


def summarize_by_category(rows: Iterable[TimeBudgetRow], schema: ProjectSchema) -> List[Tuple[str, float, int]]:
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
        totals[category_id] = (total_duration + row.total_duration, occurrences + row.occurrences)

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


def compute_binned_time_budget(intervals: Iterable[BehaviorInterval], bin_size: float) -> List[BinnedTimeBudgetRow]:
    """Compute per-bin durations for each subject/behavior."""

    if bin_size <= 0:
        raise ValueError("bin_size must be positive.")

    interval_list = list(intervals)
    if not interval_list:
        return []

    session_start = min(i.start for i in interval_list)
    session_end = max(i.end for i in interval_list)
    if session_end <= session_start:
        return []

    epsilon = 1e-9
    bin_totals: Dict[Tuple[str, str, float, float], float] = defaultdict(float)

    for interval in interval_list:
        start_idx = max(
            0, int(math.floor((interval.start - session_start) / bin_size)))
        end_idx = max(
            0,
            int(math.floor((interval.end - session_start - epsilon) / bin_size)),
        )
        for idx in range(start_idx, end_idx + 1):
            bin_start = session_start + idx * bin_size
            bin_end = min(bin_start + bin_size, session_end)
            overlap_start = max(interval.start, bin_start)
            overlap_end = min(interval.end, bin_end)
            duration = overlap_end - overlap_start
            if duration > epsilon:
                key = (interval.subject, interval.behavior, bin_start, bin_end)
                bin_totals[key] += duration

    rows = [
        BinnedTimeBudgetRow(
            subject=subject,
            behavior=behavior,
            bin_start=bin_start,
            bin_end=bin_end,
            duration=duration,
        )
        for (subject, behavior, bin_start, bin_end), duration in bin_totals.items()
        if duration > epsilon
    ]
    rows.sort(key=lambda r: (r.subject, r.behavior, r.bin_start))
    return rows


def format_time_budget_table(rows: Sequence[TimeBudgetRow], schema: Optional[ProjectSchema] = None) -> str:
    if not rows:
        return "No completed behavior intervals were found."

    include_category = schema is not None
    headers = [
        "Subject",
        "Behavior",
        "Occurrences",
        "Total (s)",
        "Mean (s)",
        "Median (s)",
        "Min (s)",
        "Max (s)",
        "% Session",
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
            f"{row.median_duration:.2f}",
            f"{row.min_duration:.2f}",
            f"{row.max_duration:.2f}",
            f"{row.percent_of_session:.2f}",
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
        return "  ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(values))

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
        "Duration (s)",
    ]
    str_rows: List[List[str]] = [
        [
            row.subject,
            row.behavior,
            f"{row.bin_start:.2f}",
            f"{row.bin_end:.2f}",
            f"{row.duration:.2f}",
        ]
        for row in rows
    ]

    col_widths = [len(header) for header in headers]
    for r in str_rows:
        for idx, value in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(value))

    def fmt_row(values: Sequence[str]) -> str:
        return "  ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(values))

    lines = [fmt_row(headers), fmt_row(["-" * len(h) for h in headers])]
    lines.extend(fmt_row(r) for r in str_rows)
    return "\n".join(lines)


def format_category_summary(rows: Sequence[Tuple[str, float, int]]) -> str:
    if not rows:
        return "No category assignments were found."
    headers = ["Category", "Total (s)", "Occurrences"]
    str_rows = [
        [name, f"{total:.2f}", str(occurrences)]
        for name, total, occurrences in rows
    ]
    col_widths = [len(header) for header in headers]
    for r in str_rows:
        for idx, value in enumerate(r):
            col_widths[idx] = max(col_widths[idx], len(value))

    def fmt_row(values: Sequence[str]) -> str:
        return "  ".join(value.ljust(col_widths[idx]) for idx, value in enumerate(values))

    lines = [fmt_row(headers), fmt_row(["-" * len(h) for h in headers])]
    lines.extend(fmt_row(r) for r in str_rows)
    return "\n".join(lines)


def write_time_budget_csv(rows: Sequence[TimeBudgetRow], path: Path, schema: Optional[ProjectSchema] = None) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
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
            ]
            if include_category:
                behavior = behavior_map.get(row.behavior)
                category_name = None
                if behavior and behavior.category_id:
                    category = category_map.get(behavior.category_id)
                    category_name = category.name if category else behavior.category_id
                item.insert(2, category_name or "")
            writer.writerow(item)


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
        raise TimeBudgetComputationError(
            f"Failed to read events file: {exc}") from exc

    schema: Optional[ProjectSchema] = None
    if args.schema is not None:
        if not args.schema.exists():
            parser.error(f"Schema file not found: {args.schema}")
            return 2
        try:
            schema = load_project_schema(args.schema)
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

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
