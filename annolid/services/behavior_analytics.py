"""Service-layer orchestration for ethogram QC, Inter-Rater Reliability, cohort analysis, and time budgets."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

from annolid.behavior.cohort import (
    CohortBehaviorReport,
    CohortComputationError,
    CohortGroupSummaryRow,
    CohortRecordingSpec,
    CohortSubjectRow,
    TransitionEdge,
    analyze_behavior_cohort,
    apply_cohort_metadata_csv,
    format_cohort_report_table,
    load_cohort_manifest,
    write_cohort_group_summary_csv,
    write_cohort_subject_rows_csv,
    write_transition_matrix_csv,
)
from annolid.behavior.irr import (
    BehaviorIRRRow,
    IRRComputationError,
    IRRReport,
    compute_irr_from_csvs,
    format_irr_report_table,
    write_irr_report_json,
    write_irr_summary_csv,
)
from annolid.behavior.qc import (
    EthogramIssue,
    EthogramQCReport,
    format_qc_report_table,
    validate_behavior_csv,
    write_qc_issues_csv,
    write_qc_report_json,
)
from annolid.behavior.time_budget import (
    BoutDefinition,
    BoutSummaryRow,
    TimeBudgetComputationError,
    _read_event_rows,
    aggression_bout_definition,
    compute_binned_time_budget,
    compute_bout_summary,
    extract_behavior_intervals,
    format_binned_time_budget_table,
    format_bout_summary_table,
    format_category_summary,
    format_time_budget_table,
    summarize_by_category,
    summarize_intervals,
    write_bout_summary_csv,
    write_time_budget_csv,
)
from annolid.core.behavior.spec import ProjectSchema, load_behavior_spec
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text
from annolid.version import get_version

__all__ = [
    "EthogramIssue",
    "EthogramQCReport",
    "BehaviorIRRRow",
    "IRRReport",
    "IRRComputationError",
    "CohortComputationError",
    "CohortRecordingSpec",
    "CohortSubjectRow",
    "CohortGroupSummaryRow",
    "TransitionEdge",
    "CohortBehaviorReport",
    "run_behavior_qc",
    "run_behavior_qc_cli",
    "run_behavior_irr",
    "run_behavior_irr_cli",
    "run_behavior_cohort",
    "run_behavior_cohort_cli",
    "run_behavior_time_budget_cli",
]


def _load_optional_schema(schema_path: Optional[Path | str]) -> Optional[ProjectSchema]:
    if schema_path is None:
        return None
    resolved = Path(schema_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Schema file not found: {resolved}")
    schema, _ = load_behavior_spec(path=resolved)
    return schema


def run_behavior_qc(
    events_csv: Path | str,
    *,
    schema_path: Optional[Path | str] = None,
    strict_schema: bool = False,
) -> EthogramQCReport:
    """Validate a behavior event CSV file against ethogram and schema rules."""
    schema = _load_optional_schema(schema_path)
    return validate_behavior_csv(
        Path(events_csv),
        schema=schema,
        strict_schema=strict_schema,
    )


def run_behavior_qc_cli(
    *,
    events_csv: Path | str,
    schema_path: Optional[Path | str] = None,
    strict: bool = False,
    strict_schema: bool = False,
    json_stdout: bool = False,
    out_json: Optional[Path | str] = None,
    csv_out: Optional[Path | str] = None,
) -> int:
    """Execute ``annolid-run behavior-qc`` and return a process exit code."""
    try:
        report = run_behavior_qc(
            events_csv,
            schema_path=schema_path,
            strict_schema=strict_schema,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    if out_json is not None:
        write_qc_report_json(report, Path(out_json))
    if csv_out is not None:
        write_qc_issues_csv(report, Path(csv_out))

    if json_stdout:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_qc_report_table(report))

    return 0 if report.is_clean(strict=bool(strict)) else 2


def run_behavior_irr(
    csv_paths: Sequence[Path | str],
    *,
    rater_names: Optional[Sequence[str]] = None,
    schema_path: Optional[Path | str] = None,
    bin_seconds: float = 0.1,
    onset_tolerance_seconds: float = 0.5,
    subject: Optional[str] = None,
    pool_subjects: bool = False,
    session_start: Optional[float] = None,
    session_end: Optional[float] = None,
    session_duration: Optional[float] = None,
    include_empty_schema_behaviors: bool = False,
) -> IRRReport:
    """Compute Inter-Rater Reliability across 2+ behavior event CSV exports."""
    schema = _load_optional_schema(schema_path)
    return compute_irr_from_csvs(
        csv_paths,
        rater_names=rater_names,
        bin_seconds=bin_seconds,
        onset_tolerance_seconds=onset_tolerance_seconds,
        schema=schema,
        subject=subject,
        pool_subjects=pool_subjects,
        session_start=session_start,
        session_end=session_end,
        session_duration=session_duration,
        include_empty_schema_behaviors=include_empty_schema_behaviors,
    )


def run_behavior_irr_cli(
    *,
    csv_paths: Sequence[Path | str],
    rater_names: Optional[Sequence[str]] = None,
    schema_path: Optional[Path | str] = None,
    bin_seconds: float = 0.1,
    onset_tolerance_seconds: float = 0.5,
    subject: Optional[str] = None,
    pool_subjects: bool = False,
    session_start: Optional[float] = None,
    session_end: Optional[float] = None,
    session_duration: Optional[float] = None,
    include_empty_schema_behaviors: bool = False,
    min_kappa: Optional[float] = None,
    json_stdout: bool = False,
    out_json: Optional[Path | str] = None,
    csv_out: Optional[Path | str] = None,
) -> int:
    """Execute ``annolid-run behavior-irr`` and return a process exit code."""
    if len(csv_paths) < 2:
        print(
            "[annolid-run] behavior-irr requires at least 2 behavior event CSV files.",
            file=sys.stderr,
        )
        return 2

    try:
        report = run_behavior_irr(
            csv_paths,
            rater_names=rater_names,
            schema_path=schema_path,
            bin_seconds=bin_seconds,
            onset_tolerance_seconds=onset_tolerance_seconds,
            subject=subject,
            pool_subjects=pool_subjects,
            session_start=session_start,
            session_end=session_end,
            session_duration=session_duration,
            include_empty_schema_behaviors=include_empty_schema_behaviors,
        )
    except (FileNotFoundError, ValueError, OSError, IRRComputationError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    if out_json is not None:
        write_irr_report_json(report, Path(out_json))
    if csv_out is not None:
        write_irr_summary_csv(report, Path(csv_out))

    if json_stdout:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_irr_report_table(report))

    if min_kappa is not None and report.macro_cohens_kappa < float(min_kappa):
        print(
            f"[annolid-run] Macro Cohen's kappa {report.macro_cohens_kappa:.4f} "
            f"is below --min-kappa {float(min_kappa):.4f}.",
            file=sys.stderr,
        )
        return 2
    return 0


def run_behavior_time_budget_cli(
    *,
    events_csv: Path | str,
    output_csv: Optional[Path | str] = None,
    json_out: Optional[Path | str] = None,
    json_stdout: bool = False,
    strict: bool = False,
    bin_size: Optional[float] = None,
    schema_path: Optional[Path | str] = None,
    bout_profile: Optional[str] = None,
    bout_name: str = "custom_bout",
    bout_behaviors: Optional[str] = None,
    bout_initiation_behaviors: Optional[str] = None,
    bout_gap_seconds: Optional[float] = None,
) -> int:
    """Execute ``annolid-run time-budget`` and return a process exit code."""
    csv_path = Path(events_csv)
    if not csv_path.exists():
        print(f"[annolid-run] Events file not found: {csv_path}", file=sys.stderr)
        return 2

    try:
        schema = _load_optional_schema(schema_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    try:
        rows = _read_event_rows(csv_path)
        intervals, warnings = extract_behavior_intervals(rows)
    except (OSError, TimeBudgetComputationError) as exc:
        print(f"[annolid-run] Failed to read events file: {exc}", file=sys.stderr)
        return 2

    summary = summarize_intervals(intervals)

    try:
        binned_rows = (
            compute_binned_time_budget(intervals, float(bin_size))
            if bin_size is not None
            else None
        )
    except ValueError as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    bout_definition: Optional[BoutDefinition] = None
    if bout_profile == "aggression":
        profile_gap = (
            2.0 if bout_gap_seconds is None else max(0.0, float(bout_gap_seconds))
        )
        bout_definition = aggression_bout_definition(max_gap_seconds=profile_gap)
    elif bout_behaviors:
        behavior_names = tuple(
            item.strip() for item in str(bout_behaviors).split(",") if item.strip()
        )
        if not behavior_names:
            print(
                "[annolid-run] --bout-behaviors must contain at least one non-empty behavior.",
                file=sys.stderr,
            )
            return 2
        initiation_names = tuple(
            item.strip()
            for item in str(bout_initiation_behaviors or "").split(",")
            if item.strip()
        )
        bout_definition = BoutDefinition(
            name=str(bout_name).strip() or "custom_bout",
            behaviors=behavior_names,
            max_gap_seconds=max(
                0.0, 0.0 if bout_gap_seconds is None else float(bout_gap_seconds)
            ),
            initiation_behaviors=initiation_names,
        )
    elif bout_initiation_behaviors:
        print(
            "[annolid-run] --bout-initiation-behaviors requires --bout-behaviors or --bout-profile.",
            file=sys.stderr,
        )
        return 2

    bout_rows: Optional[List[BoutSummaryRow]] = None
    if bout_definition is not None:
        bout_rows = compute_bout_summary(intervals, definition=bout_definition)

    category_summary = (
        summarize_by_category(summary, schema)
        if (schema is not None and summary)
        else []
    )

    for warning in warnings:
        print(f"[warning] {warning}", file=sys.stderr)

    if strict and warnings:
        return 2

    payload: Dict[str, Any] = {
        "schema": "behavior_time_budget/v1",
        "source_path": str(csv_path),
        "warnings": list(warnings),
        "rows": [
            {
                "subject": r.subject,
                "behavior": r.behavior,
                "occurrences": int(r.occurrences),
                "total_duration": round(float(r.total_duration), 6),
                "mean_duration": round(float(r.mean_duration), 6),
                "median_duration": round(float(r.median_duration), 6),
                "min_duration": round(float(r.min_duration), 6),
                "max_duration": round(float(r.max_duration), 6),
                "percent_of_session": round(float(r.percent_of_session), 6),
            }
            for r in summary
        ],
    }
    if category_summary:
        payload["categories"] = [
            {
                "category": name,
                "total_duration": round(float(total), 6),
                "occurrences": int(occ),
            }
            for name, total, occ in category_summary
        ]
    if binned_rows is not None:
        payload["binned_rows"] = [
            {
                "subject": r.subject,
                "behavior": r.behavior,
                "bin_start": round(float(r.bin_start), 6),
                "bin_end": round(float(r.bin_end), 6),
                "duration": round(float(r.duration), 6),
            }
            for r in binned_rows
        ]
    if bout_rows is not None:
        payload["bouts"] = [
            {
                "subject": r.subject,
                "bout_name": r.bout_name,
                "bout_count": int(r.bout_count),
                "total_duration": round(float(r.total_duration), 6),
                "mean_duration": round(float(r.mean_duration), 6),
                "median_duration": round(float(r.median_duration), 6),
                "min_duration": round(float(r.min_duration), 6),
                "max_duration": round(float(r.max_duration), 6),
                "behavior_counts": dict(r.behavior_counts),
                "initiation_count": int(r.initiation_count),
            }
            for r in bout_rows
        ]

    if json_out is not None:
        atomic_write_json(Path(json_out), payload, indent=2, ensure_ascii=False)

    out_path = Path(output_csv) if output_csv is not None else None
    if out_path is not None:
        write_time_budget_csv(summary, out_path, schema=schema)
        if category_summary:
            category_path = out_path.with_name(
                out_path.stem + "_categories" + out_path.suffix
            )
            cat_buf = io.StringIO()
            writer = csv.writer(cat_buf)
            writer.writerow(["Category", "TotalSeconds", "Occurrences"])
            for name, total, occurrences in category_summary:
                writer.writerow([name, f"{total:.6f}", occurrences])
            atomic_write_text(category_path, cat_buf.getvalue(), encoding="utf-8")
        if bout_rows is not None and bout_definition is not None:
            bout_path = out_path.with_name(out_path.stem + "_bouts" + out_path.suffix)
            write_bout_summary_csv(
                bout_rows,
                bout_path,
                behavior_order=bout_definition.behaviors,
            )

    if json_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif out_path is None:
        print(format_time_budget_table(summary, schema=schema))
        if category_summary:
            print()
            print("Category Summary")
            print(format_category_summary(category_summary))
        if binned_rows is not None and bin_size is not None:
            print()
            print(f"Binned time-budget (bin size = {float(bin_size):.2f}s)")
            print(format_binned_time_budget_table(binned_rows))
        if bout_rows is not None and bout_definition is not None:
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


def _resolve_cohort_specs(
    *,
    manifest_path: Optional[Path | str] = None,
    events_csvs: Optional[Sequence[Path | str]] = None,
    metadata_csv: Optional[Path | str] = None,
    excluded_paths: Optional[Sequence[Path]] = None,
) -> List[CohortRecordingSpec]:
    specs: List[CohortRecordingSpec] = []
    excluded_resolved = {p.expanduser().resolve() for p in (excluded_paths or ())}
    if manifest_path is not None:
        manifest_resolved = Path(manifest_path).expanduser().resolve()
        excluded_resolved.add(manifest_resolved)
        specs.extend(load_cohort_manifest(manifest_resolved))
    if metadata_csv is not None:
        excluded_resolved.add(Path(metadata_csv).expanduser().resolve())

    if events_csvs:
        seen_csvs = {Path(s.events_csv).expanduser().resolve() for s in specs}
        for raw_item in events_csvs:
            candidate = Path(raw_item).expanduser().resolve()
            if not candidate.exists():
                raise FileNotFoundError(f"Cohort events input not found: {candidate}")
            if candidate.is_dir():
                dir_csvs = sorted(
                    p.resolve()
                    for p in candidate.glob("*.csv")
                    if p.is_file() and p.resolve() not in excluded_resolved
                )
                if not dir_csvs:
                    raise CohortComputationError(
                        f"Directory '{candidate}' contains no behavior event CSV files."
                    )
                for csv_file in dir_csvs:
                    if csv_file not in seen_csvs:
                        seen_csvs.add(csv_file)
                        specs.append(
                            CohortRecordingSpec(
                                recording_id=csv_file.stem,
                                events_csv=csv_file,
                            )
                        )
            else:
                if candidate not in seen_csvs:
                    seen_csvs.add(candidate)
                    specs.append(
                        CohortRecordingSpec(
                            recording_id=candidate.stem,
                            events_csv=candidate,
                        )
                    )

    if not specs:
        raise CohortComputationError(
            "Provide --manifest <cohort.csv|cohort.json> and/or one or more event CSV paths."
        )

    if metadata_csv is not None:
        specs = apply_cohort_metadata_csv(specs, metadata_csv)
    return specs


def run_behavior_cohort(
    *,
    manifest_path: Optional[Path | str] = None,
    events_csvs: Optional[Sequence[Path | str]] = None,
    metadata_csv: Optional[Path | str] = None,
    schema_path: Optional[Path | str] = None,
    session_duration: Optional[float] = None,
    max_transition_gap: Optional[float] = None,
    include_self_transitions: bool = True,
    include_schema_behaviors: bool = False,
    strict: bool = False,
    excluded_paths: Optional[Sequence[Path]] = None,
) -> CohortBehaviorReport:
    """Aggregate behavioral endpoints and Markov transitions across a cohort."""
    schema = _load_optional_schema(schema_path)
    specs = _resolve_cohort_specs(
        manifest_path=manifest_path,
        events_csvs=events_csvs,
        metadata_csv=metadata_csv,
        excluded_paths=excluded_paths,
    )
    return analyze_behavior_cohort(
        specs,
        schema=schema,
        session_duration_override_s=session_duration,
        max_transition_gap_s=max_transition_gap,
        include_self_transitions=include_self_transitions,
        include_schema_behaviors=include_schema_behaviors,
        strict=strict,
    )


def run_behavior_cohort_cli(
    *,
    manifest_path: Optional[Path | str] = None,
    events_csvs: Optional[Sequence[Path | str]] = None,
    metadata_csv: Optional[Path | str] = None,
    schema_path: Optional[Path | str] = None,
    session_duration: Optional[float] = None,
    max_transition_gap: Optional[float] = None,
    exclude_self_transitions: bool = False,
    include_schema_behaviors: bool = False,
    strict: bool = False,
    json_stdout: bool = False,
    out_json: Optional[Path | str] = None,
    summary_csv: Optional[Path | str] = None,
    subjects_csv: Optional[Path | str] = None,
    transitions_csv: Optional[Path | str] = None,
) -> int:
    """Execute ``annolid-run behavior-cohort`` and return a process exit code."""
    excluded: List[Path] = [
        Path(p)
        for p in (out_json, summary_csv, subjects_csv, transitions_csv)
        if p is not None
    ]
    try:
        report = run_behavior_cohort(
            manifest_path=manifest_path,
            events_csvs=events_csvs,
            metadata_csv=metadata_csv,
            schema_path=schema_path,
            session_duration=session_duration,
            max_transition_gap=max_transition_gap,
            include_self_transitions=not bool(exclude_self_transitions),
            include_schema_behaviors=bool(include_schema_behaviors),
            strict=bool(strict),
            excluded_paths=excluded,
        )
    except (FileNotFoundError, ValueError, OSError, CohortComputationError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    for warning in report.warnings:
        print(f"[warning] {warning}", file=sys.stderr)
    if strict and report.warnings:
        return 2

    provenance: Dict[str, Any] = {
        "schema": "behavior_cohort/v1",
        "annolid_version": str(get_version() or "0.0.0"),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "parameters": {
            "manifest_path": str(manifest_path) if manifest_path is not None else None,
            "metadata_csv": str(metadata_csv) if metadata_csv is not None else None,
            "session_duration": session_duration,
            "max_transition_gap": max_transition_gap,
            "include_self_transitions": not bool(exclude_self_transitions),
            "include_schema_behaviors": bool(include_schema_behaviors),
            "strict": bool(strict),
        },
    }

    payload = report.to_dict()
    payload["provenance"] = provenance

    if out_json is not None:
        atomic_write_json(Path(out_json), payload, indent=2, ensure_ascii=False)
    if summary_csv is not None:
        write_cohort_group_summary_csv(report.group_summaries, Path(summary_csv))
    if subjects_csv is not None:
        write_cohort_subject_rows_csv(report.subject_rows, Path(subjects_csv))
    if transitions_csv is not None:
        write_transition_matrix_csv(report.transitions, Path(transitions_csv))

    if json_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_cohort_report_table(report))

    return 0
