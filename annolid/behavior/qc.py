"""Ethogram consistency and schema quality-control checks for behavior event logs.

Validates Annolid/BORIS-style behavior event CSV exports and in-memory event rows
for structural integrity, balanced state intervals, point-vs-state event usage,
and mutual-exclusion (`exclusive_with`) violations defined in ``ProjectSchema``.
"""

from __future__ import annotations

from collections import defaultdict, deque
import csv
from dataclasses import dataclass
import io
from pathlib import Path
from typing import (
    Any,
    Deque,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from annolid.behavior.event_utils import (
    DEFAULT_SUBJECT_LABEL,
    ParsedEventRow,
    parse_event_mapping_rows,
    read_parsed_event_rows,
)
from annolid.behavior.project_schema import (
    BehaviorDefinition,
    ProjectSchema,
    validate_schema,
)
from annolid.behavior.time_budget import BehaviorInterval
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text

__all__ = [
    "PointBehaviorEvent",
    "EthogramIssue",
    "EthogramQCReport",
    "validate_behavior_events",
    "validate_behavior_csv",
    "format_qc_report_table",
    "write_qc_report_json",
    "write_qc_issues_csv",
]

_EPSILON = 1e-9


@dataclass(frozen=True)
class PointBehaviorEvent:
    """Instantaneous behavior occurrence."""

    subject: str
    behavior: str
    timestamp: float
    line_number: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject,
            "behavior": self.behavior,
            "timestamp": float(self.timestamp),
            "line_number": self.line_number,
        }


@dataclass(frozen=True)
class EthogramIssue:
    """Single quality-control finding on a behavior event log."""

    severity: str  # "error" or "warning"
    code: str
    message: str
    subject: Optional[str] = None
    behavior: Optional[str] = None
    timestamp: Optional[float] = None
    end_timestamp: Optional[float] = None
    line_number: Optional[int] = None
    related_behavior: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
        }
        if self.subject is not None:
            payload["subject"] = self.subject
        if self.behavior is not None:
            payload["behavior"] = self.behavior
        if self.timestamp is not None:
            payload["timestamp"] = round(float(self.timestamp), 6)
        if self.end_timestamp is not None:
            payload["end_timestamp"] = round(float(self.end_timestamp), 6)
        if self.line_number is not None:
            payload["line_number"] = int(self.line_number)
        if self.related_behavior is not None:
            payload["related_behavior"] = self.related_behavior
        return payload


@dataclass(frozen=True)
class EthogramQCReport:
    """Structured summary of ethogram consistency validation."""

    source_path: Optional[str]
    total_rows: int
    valid_interval_count: int
    point_event_count: int
    subjects: Tuple[str, ...]
    behaviors: Tuple[str, ...]
    issues: Tuple[EthogramIssue, ...]
    intervals: Tuple[BehaviorInterval, ...] = ()
    point_events: Tuple[PointBehaviorEvent, ...] = ()

    @property
    def error_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for issue in self.issues if issue.severity == "warning")

    @property
    def passed(self) -> bool:
        return self.error_count == 0

    def is_clean(self, *, strict: bool = False) -> bool:
        if strict:
            return self.error_count == 0 and self.warning_count == 0
        return self.error_count == 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "behavior_qc_report/v1",
            "source_path": self.source_path,
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "total_rows": self.total_rows,
            "valid_interval_count": self.valid_interval_count,
            "point_event_count": self.point_event_count,
            "subjects": list(self.subjects),
            "behaviors": list(self.behaviors),
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _build_schema_lookups(
    schema: ProjectSchema,
) -> Tuple[
    Dict[str, BehaviorDefinition],
    Set[str],
    Dict[str, str],
    Set[Tuple[str, str]],
    List[EthogramIssue],
]:
    issues: List[EthogramIssue] = []
    for warning_msg in validate_schema(schema):
        issues.append(
            EthogramIssue(
                severity="warning",
                code="schema_warning",
                message=warning_msg,
            )
        )

    behavior_lookup: Dict[str, BehaviorDefinition] = {}
    for beh in schema.behaviors:
        if beh.code:
            behavior_lookup[beh.code] = beh
        if beh.name and beh.name not in behavior_lookup:
            behavior_lookup[beh.name] = beh

    known_subjects: Set[str] = set()
    for subj in schema.subjects:
        if subj.id:
            known_subjects.add(subj.id)
        if subj.name:
            known_subjects.add(subj.name)

    modifier_lookup: Dict[str, str] = {}
    for mod in schema.modifiers:
        if mod.id:
            modifier_lookup[mod.id] = mod.id
        if mod.name and mod.name not in modifier_lookup:
            modifier_lookup[mod.name] = mod.id

    exclusion_pairs: Set[Tuple[str, str]] = set()
    for beh in schema.behaviors:
        canon_a = beh.code
        for raw_other in beh.exclusive_with:
            target = behavior_lookup.get(raw_other)
            if target is None:
                issues.append(
                    EthogramIssue(
                        severity="warning",
                        code="schema_warning",
                        behavior=beh.code,
                        related_behavior=str(raw_other),
                        message=(
                            f"Behavior '{beh.code}' lists unknown exclusive_with "
                            f"target '{raw_other}'."
                        ),
                    )
                )
                canon_b = str(raw_other)
            else:
                canon_b = target.code
            if canon_a == canon_b:
                continue
            exclusion_pairs.add((canon_a, canon_b))
            exclusion_pairs.add((canon_b, canon_a))

    return (
        behavior_lookup,
        known_subjects,
        modifier_lookup,
        exclusion_pairs,
        issues,
    )


def validate_behavior_events(
    rows: Iterable[ParsedEventRow | Mapping[str, object]],
    *,
    schema: Optional[ProjectSchema] = None,
    strict_schema: bool = False,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
    source_path: Optional[Path | str] = None,
) -> EthogramQCReport:
    """Validate behavior event rows for consistency and schema compliance."""
    raw_items = list(rows)
    if raw_items and all(isinstance(item, ParsedEventRow) for item in raw_items):
        parsed_rows: List[ParsedEventRow] = list(raw_items)  # type: ignore[arg-type]
    else:
        mapping_items = [
            item if isinstance(item, Mapping) else {} for item in raw_items
        ]
        parsed_rows = parse_event_mapping_rows(
            mapping_items,
            default_subject=default_subject,
        )

    issues: List[EthogramIssue] = []
    behavior_lookup: Dict[str, BehaviorDefinition] = {}
    known_subjects: Set[str] = set()
    modifier_lookup: Dict[str, str] = {}
    exclusion_pairs: Set[Tuple[str, str]] = set()

    if schema is not None:
        (
            behavior_lookup,
            known_subjects,
            modifier_lookup,
            exclusion_pairs,
            schema_issues,
        ) = _build_schema_lookups(schema)
        issues.extend(schema_issues)

    schema_violation_severity = "error" if strict_schema else "warning"
    seen_subjects: Set[str] = set()
    seen_behaviors: Set[str] = set()
    reported_unknown_behaviors: Set[str] = set()
    reported_unknown_subjects: Set[str] = set()

    valid_state_rows: List[ParsedEventRow] = []
    point_events: List[PointBehaviorEvent] = []

    prev_timestamp: Optional[float] = None

    for row in parsed_rows:
        if not row.behavior or not row.raw_event or not row.raw_recording_time:
            missing_parts: List[str] = []
            if not row.raw_recording_time:
                missing_parts.append("Recording time")
            if not row.behavior:
                missing_parts.append("Behavior")
            if not row.raw_event:
                missing_parts.append("Event")
            issues.append(
                EthogramIssue(
                    severity="error",
                    code="missing_required_field",
                    subject=row.subject if row.has_explicit_subject else None,
                    behavior=row.behavior or None,
                    line_number=row.line_number,
                    message=(
                        f"Line {row.line_number}: missing required value(s) for "
                        f"{', '.join(missing_parts)}."
                    ),
                )
            )
            continue

        seen_subjects.add(row.subject)
        seen_behaviors.add(row.behavior)

        if row.recording_time is None:
            issues.append(
                EthogramIssue(
                    severity="error",
                    code="invalid_timestamp",
                    subject=row.subject,
                    behavior=row.behavior,
                    line_number=row.line_number,
                    message=(
                        f"Line {row.line_number}: invalid recording timestamp "
                        f"{row.raw_recording_time!r} for behavior '{row.behavior}'."
                    ),
                )
            )
            continue

        if row.recording_time < -_EPSILON:
            issues.append(
                EthogramIssue(
                    severity="error",
                    code="invalid_timestamp",
                    subject=row.subject,
                    behavior=row.behavior,
                    timestamp=row.recording_time,
                    line_number=row.line_number,
                    message=(
                        f"Line {row.line_number}: negative recording timestamp "
                        f"{row.recording_time:.3f}s for behavior '{row.behavior}'."
                    ),
                )
            )
            continue

        if (
            prev_timestamp is not None
            and row.recording_time < prev_timestamp - _EPSILON
        ):
            issues.append(
                EthogramIssue(
                    severity="warning",
                    code="non_monotonic_row_order",
                    subject=row.subject,
                    behavior=row.behavior,
                    timestamp=row.recording_time,
                    line_number=row.line_number,
                    message=(
                        f"Line {row.line_number}: timestamp {row.recording_time:.3f}s "
                        f"precedes previous row timestamp {prev_timestamp:.3f}s."
                    ),
                )
            )
        prev_timestamp = row.recording_time

        if row.event_kind is None:
            issues.append(
                EthogramIssue(
                    severity="error",
                    code="unrecognized_event_label",
                    subject=row.subject,
                    behavior=row.behavior,
                    timestamp=row.recording_time,
                    line_number=row.line_number,
                    message=(
                        f"Line {row.line_number}: unrecognized event label "
                        f"{row.raw_event!r} for subject '{row.subject}', "
                        f"behavior '{row.behavior}'."
                    ),
                )
            )
            continue

        beh_def: Optional[BehaviorDefinition] = None
        if schema is not None:
            if (
                row.has_explicit_subject
                and known_subjects
                and row.subject not in known_subjects
                and row.subject not in reported_unknown_subjects
            ):
                reported_unknown_subjects.add(row.subject)
                issues.append(
                    EthogramIssue(
                        severity=schema_violation_severity,
                        code="unknown_subject",
                        subject=row.subject,
                        timestamp=row.recording_time,
                        line_number=row.line_number,
                        message=(
                            f"Subject '{row.subject}' (first seen on line {row.line_number}) "
                            "is not declared in the project schema."
                        ),
                    )
                )

            beh_def = behavior_lookup.get(row.behavior)
            if beh_def is None and behavior_lookup:
                if row.behavior not in reported_unknown_behaviors:
                    reported_unknown_behaviors.add(row.behavior)
                    issues.append(
                        EthogramIssue(
                            severity=schema_violation_severity,
                            code="unknown_behavior",
                            subject=row.subject,
                            behavior=row.behavior,
                            timestamp=row.recording_time,
                            line_number=row.line_number,
                            message=(
                                f"Behavior '{row.behavior}' (first seen on line {row.line_number}) "
                                "is not declared in the project schema."
                            ),
                        )
                    )

            if row.modifiers:
                for mod in row.modifiers:
                    canon_mod = modifier_lookup.get(mod)
                    if canon_mod is None:
                        issues.append(
                            EthogramIssue(
                                severity=schema_violation_severity,
                                code="unknown_modifier",
                                subject=row.subject,
                                behavior=row.behavior,
                                timestamp=row.recording_time,
                                line_number=row.line_number,
                                message=(
                                    f"Line {row.line_number}: modifier '{mod}' on behavior "
                                    f"'{row.behavior}' is not declared in the project schema."
                                ),
                            )
                        )
                    elif (
                        beh_def is not None
                        and beh_def.modifier_ids
                        and canon_mod not in beh_def.modifier_ids
                    ):
                        issues.append(
                            EthogramIssue(
                                severity="warning",
                                code="disallowed_modifier",
                                subject=row.subject,
                                behavior=row.behavior,
                                timestamp=row.recording_time,
                                line_number=row.line_number,
                                message=(
                                    f"Line {row.line_number}: modifier '{mod}' is not listed "
                                    f"in modifier_ids for behavior '{beh_def.code}'."
                                ),
                            )
                        )

        if beh_def is not None and not beh_def.is_state:
            if row.event_kind in {"start", "end"}:
                issues.append(
                    EthogramIssue(
                        severity="error",
                        code="point_behavior_state_event",
                        subject=row.subject,
                        behavior=row.behavior,
                        timestamp=row.recording_time,
                        line_number=row.line_number,
                        message=(
                            f"Line {row.line_number}: point behavior '{row.behavior}' "
                            f"(is_state=False) cannot use '{row.raw_event}' boundary marker."
                        ),
                    )
                )
                continue

        if row.event_kind == "point":
            if beh_def is not None and beh_def.is_state:
                issues.append(
                    EthogramIssue(
                        severity="warning",
                        code="state_behavior_point_event",
                        subject=row.subject,
                        behavior=row.behavior,
                        timestamp=row.recording_time,
                        line_number=row.line_number,
                        message=(
                            f"Line {row.line_number}: state behavior '{row.behavior}' "
                            f"(is_state=True) recorded as instantaneous '{row.raw_event}'."
                        ),
                    )
                )
            point_events.append(
                PointBehaviorEvent(
                    subject=row.subject,
                    behavior=row.behavior,
                    timestamp=float(row.recording_time),
                    line_number=row.line_number,
                )
            )
            continue

        valid_state_rows.append(row)

    # Group by (subject, behavior) and process chronologically per timestamp bucket so:
    # 1) Back-to-back abutting intervals [0.0, 1.0] + [1.0, 2.0] close at 1.0 before
    #    opening the next interval at 1.0 regardless of CSV tie order.
    # 2) Standalone zero-duration intervals [1.0, 1.0] (with no open interval < 1.0)
    #    pair cleanly with a zero_duration_interval warning rather than orphan/unclosed errors.
    by_series: Dict[Tuple[str, str], List[ParsedEventRow]] = defaultdict(list)
    for row in valid_state_rows:
        by_series[(row.subject, row.behavior)].append(row)

    intervals: List[BehaviorInterval] = []

    for (subject, behavior), series_rows in sorted(by_series.items()):
        series_rows.sort(key=lambda r: (float(r.recording_time or 0.0), r.line_number))
        queue: Deque[Tuple[float, int]] = deque()

        idx = 0
        n_rows = len(series_rows)
        while idx < n_rows:
            ts = float(series_rows[idx].recording_time or 0.0)
            bucket_starts: Deque[ParsedEventRow] = deque()
            bucket_ends: Deque[ParsedEventRow] = deque()
            while (
                idx < n_rows
                and abs(float(series_rows[idx].recording_time or 0.0) - ts) <= _EPSILON
            ):
                curr = series_rows[idx]
                if curr.event_kind == "start":
                    bucket_starts.append(curr)
                else:
                    bucket_ends.append(curr)
                idx += 1

            # 1) Close open intervals started before `ts` first.
            while queue and bucket_ends:
                start_ts, _start_line = queue.popleft()
                end_row = bucket_ends.popleft()
                end_ts = float(end_row.recording_time or ts)
                if abs(end_ts - start_ts) <= _EPSILON:
                    issues.append(
                        EthogramIssue(
                            severity="warning",
                            code="zero_duration_interval",
                            subject=subject,
                            behavior=behavior,
                            timestamp=start_ts,
                            end_timestamp=end_ts,
                            line_number=end_row.line_number,
                            message=(
                                f"Line {end_row.line_number}: zero-duration interval "
                                f"({start_ts:.3f}s -> {end_ts:.3f}s) for subject '{subject}', "
                                f"behavior '{behavior}'."
                            ),
                        )
                    )
                intervals.append(
                    BehaviorInterval(
                        subject=subject,
                        behavior=behavior,
                        start=start_ts,
                        end=end_ts,
                    )
                )

            # 2) Pair same-timestamp (start, end) entries as zero-duration intervals.
            while bucket_starts and bucket_ends:
                start_row = bucket_starts.popleft()
                end_row = bucket_ends.popleft()
                start_ts = float(start_row.recording_time or ts)
                end_ts = float(end_row.recording_time or ts)
                issues.append(
                    EthogramIssue(
                        severity="warning",
                        code="zero_duration_interval",
                        subject=subject,
                        behavior=behavior,
                        timestamp=start_ts,
                        end_timestamp=end_ts,
                        line_number=end_row.line_number,
                        message=(
                            f"Line {end_row.line_number}: zero-duration interval "
                            f"({start_ts:.3f}s -> {end_ts:.3f}s) for subject '{subject}', "
                            f"behavior '{behavior}'."
                        ),
                    )
                )
                intervals.append(
                    BehaviorInterval(
                        subject=subject,
                        behavior=behavior,
                        start=start_ts,
                        end=end_ts,
                    )
                )

            # 3) Any leftover ends at `ts` have no matching start.
            while bucket_ends:
                end_row = bucket_ends.popleft()
                end_ts = float(end_row.recording_time or ts)
                issues.append(
                    EthogramIssue(
                        severity="error",
                        code="orphan_state_end",
                        subject=subject,
                        behavior=behavior,
                        timestamp=end_ts,
                        line_number=end_row.line_number,
                        message=(
                            f"Line {end_row.line_number}: end event for subject '{subject}', "
                            f"behavior '{behavior}' at {end_ts:.3f}s has no matching start."
                        ),
                    )
                )

            # 4) Push new starts at `ts`, flagging duplicates when already active.
            while bucket_starts:
                start_row = bucket_starts.popleft()
                start_ts = float(start_row.recording_time or ts)
                if queue:
                    active_start, active_line = queue[-1]
                    issues.append(
                        EthogramIssue(
                            severity="error",
                            code="duplicate_state_start",
                            subject=subject,
                            behavior=behavior,
                            timestamp=start_ts,
                            line_number=start_row.line_number,
                            message=(
                                f"Line {start_row.line_number}: start event for subject '{subject}', "
                                f"behavior '{behavior}' at {start_ts:.3f}s while state started at "
                                f"{active_start:.3f}s (line {active_line}) is still active."
                            ),
                        )
                    )
                queue.append((start_ts, start_row.line_number))

        while queue:
            start_ts, start_line = queue.popleft()
            issues.append(
                EthogramIssue(
                    severity="error",
                    code="unclosed_state_start",
                    subject=subject,
                    behavior=behavior,
                    timestamp=start_ts,
                    line_number=start_line,
                    message=(
                        f"Line {start_line}: unclosed start event for subject '{subject}', "
                        f"behavior '{behavior}' at {start_ts:.3f}s."
                    ),
                )
            )

    intervals.sort(key=lambda item: (item.subject, item.start, item.end, item.behavior))
    point_events.sort(key=lambda item: (item.subject, item.timestamp, item.behavior))

    # Check mutual exclusions per subject when schema defines exclusive_with.
    if exclusion_pairs and intervals:
        by_subject: Dict[str, List[BehaviorInterval]] = defaultdict(list)
        for interval in intervals:
            by_subject[interval.subject].append(interval)

        for subject, subj_intervals in sorted(by_subject.items()):
            ordered = sorted(
                subj_intervals, key=lambda item: (item.start, item.end, item.behavior)
            )
            active: List[BehaviorInterval] = []
            for current in ordered:
                active = [
                    prev for prev in active if prev.end > current.start + _EPSILON
                ]
                curr_def = behavior_lookup.get(current.behavior)
                curr_canon = curr_def.code if curr_def is not None else current.behavior
                for prev in active:
                    prev_def = behavior_lookup.get(prev.behavior)
                    prev_canon = (
                        prev_def.code if prev_def is not None else prev.behavior
                    )
                    if (prev_canon, curr_canon) in exclusion_pairs:
                        overlap_start = max(prev.start, current.start)
                        overlap_end = min(prev.end, current.end)
                        if overlap_end - overlap_start > _EPSILON:
                            issues.append(
                                EthogramIssue(
                                    severity="error",
                                    code="exclusive_behavior_overlap",
                                    subject=subject,
                                    behavior=prev.behavior,
                                    related_behavior=current.behavior,
                                    timestamp=overlap_start,
                                    end_timestamp=overlap_end,
                                    message=(
                                        f"Mutually exclusive behaviors '{prev.behavior}' "
                                        f"({prev.start:.3f}s-{prev.end:.3f}s) and "
                                        f"'{current.behavior}' ({current.start:.3f}s-{current.end:.3f}s) "
                                        f"overlap for subject '{subject}' during "
                                        f"{overlap_start:.3f}s-{overlap_end:.3f}s."
                                    ),
                                )
                            )
                active.append(current)

    issues.sort(
        key=lambda item: (
            0 if item.severity == "error" else 1,
            float("inf") if item.timestamp is None else item.timestamp,
            0 if item.line_number is None else item.line_number,
            item.code,
        )
    )

    return EthogramQCReport(
        source_path=str(source_path) if source_path is not None else None,
        total_rows=len(parsed_rows),
        valid_interval_count=len(intervals),
        point_event_count=len(point_events),
        subjects=tuple(sorted(seen_subjects)),
        behaviors=tuple(sorted(seen_behaviors)),
        issues=tuple(issues),
        intervals=tuple(intervals),
        point_events=tuple(point_events),
    )


def validate_behavior_csv(
    csv_path: Path | str,
    *,
    schema: Optional[ProjectSchema] = None,
    strict_schema: bool = False,
    default_subject: str = DEFAULT_SUBJECT_LABEL,
) -> EthogramQCReport:
    """Read and validate a behavior event CSV export."""
    resolved = Path(csv_path)
    parsed_rows = read_parsed_event_rows(resolved, default_subject=default_subject)
    return validate_behavior_events(
        parsed_rows,
        schema=schema,
        strict_schema=strict_schema,
        default_subject=default_subject,
        source_path=resolved,
    )


def format_qc_report_table(report: EthogramQCReport) -> str:
    """Format an :class:`EthogramQCReport` as a human-readable summary and table."""
    status = "PASSED" if report.passed else "FAILED"
    lines = [
        f"Ethogram QC Status: {status}",
        (
            f"Rows: {report.total_rows}  |  Valid Intervals: {report.valid_interval_count}  "
            f"|  Point Events: {report.point_event_count}  "
            f"|  Errors: {report.error_count}  |  Warnings: {report.warning_count}"
        ),
    ]
    if report.source_path:
        lines.insert(1, f"Source: {report.source_path}")

    if not report.issues:
        lines.append("No consistency issues detected.")
        return "\n".join(lines)

    lines.append("")
    headers = ["Severity", "Code", "Line", "Subject", "Behavior", "Time (s)", "Details"]
    str_rows: List[List[str]] = []
    for issue in report.issues:
        if issue.timestamp is not None and issue.end_timestamp is not None:
            time_str = f"{issue.timestamp:.3f}-{issue.end_timestamp:.3f}"
        elif issue.timestamp is not None:
            time_str = f"{issue.timestamp:.3f}"
        else:
            time_str = "—"
        str_rows.append(
            [
                issue.severity.upper(),
                issue.code,
                str(issue.line_number) if issue.line_number is not None else "—",
                issue.subject or "—",
                issue.behavior or "—",
                time_str,
                issue.message,
            ]
        )

    col_widths = [len(header) for header in headers]
    for r in str_rows:
        for idx in range(len(headers) - 1):
            col_widths[idx] = max(col_widths[idx], len(r[idx]))

    def _fmt(values: Sequence[str]) -> str:
        fixed = [values[idx].ljust(col_widths[idx]) for idx in range(len(headers) - 1)]
        fixed.append(values[-1])
        return "  ".join(fixed)

    lines.append(_fmt(headers))
    lines.append(_fmt(["-" * len(h) for h in headers]))
    lines.extend(_fmt(r) for r in str_rows)
    return "\n".join(lines)


def write_qc_report_json(report: EthogramQCReport, path: Path | str) -> None:
    """Atomically write an :class:`EthogramQCReport` JSON file."""
    atomic_write_json(Path(path), report.to_dict(), indent=2, ensure_ascii=False)


def write_qc_issues_csv(report: EthogramQCReport, path: Path | str) -> None:
    """Atomically write QC issues to a CSV file."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "Severity",
            "Code",
            "LineNumber",
            "Subject",
            "Behavior",
            "RelatedBehavior",
            "TimestampSeconds",
            "EndTimestampSeconds",
            "Message",
        ]
    )
    for issue in report.issues:
        writer.writerow(
            [
                issue.severity,
                issue.code,
                "" if issue.line_number is None else issue.line_number,
                issue.subject or "",
                issue.behavior or "",
                issue.related_behavior or "",
                "" if issue.timestamp is None else f"{issue.timestamp:.6f}",
                "" if issue.end_timestamp is None else f"{issue.end_timestamp:.6f}",
                issue.message,
            ]
        )
    atomic_write_text(Path(path), buffer.getvalue(), encoding="utf-8")
