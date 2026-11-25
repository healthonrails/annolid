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
    CohortComparisonRow,
    CohortComputationError,
    CohortGroupSummaryRow,
    CohortRecordingSpec,
    CohortSubjectRow,
    TransitionEdge,
    analyze_behavior_cohort,
    apply_cohort_metadata_csv,
    format_cohort_report_table,
    load_cohort_manifest,
    write_cohort_comparisons_csv,
    write_cohort_group_summary_csv,
    write_cohort_subject_rows_csv,
    write_transition_matrix_csv,
    compute_observation_transitions,
    write_transitions_graphviz_dot,
    write_transitions_svg,
)
from annolid.behavior.ethogram import write_ethogram_svg
from annolid.behavior.external_data import (
    ExternalDataError,
    load_external_data_stream,
    summarize_external_stream_by_behavior,
    write_external_data_svg,
)
from annolid.behavior.boris_io import (
    BorisIOError,
    BorisObservation,
    load_boris_observation_from_csv,
    materialize_boris_project,
    write_boris_aggregated_csv,
    write_boris_project,
    write_boris_tabular_csv,
)
from annolid.behavior.irr import (
    BehaviorIRRRow,
    IRRComputationError,
    IRRDisagreementBout,
    IRRReport,
    compute_irr_from_csvs,
    format_irr_report_table,
    write_irr_disagreements_csv,
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
    compute_behavior_cooccurrence,
    compute_binned_time_budget,
    compute_bout_summary,
    extract_behavior_intervals,
    format_binned_time_budget_table,
    format_bout_summary_table,
    format_category_summary,
    format_cooccurrence_table,
    format_time_budget_table,
    summarize_by_category,
    summarize_intervals,
    write_bout_summary_csv,
    write_cooccurrence_csv,
    write_time_budget_csv,
)
from annolid.core.behavior.spec import (
    ProjectSchema,
    default_behavior_spec,
    load_behavior_spec,
)
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text
from annolid.version import get_version

__all__ = [
    "EthogramIssue",
    "EthogramQCReport",
    "BehaviorIRRRow",
    "IRRDisagreementBout",
    "IRRReport",
    "IRRComputationError",
    "CohortComputationError",
    "CohortRecordingSpec",
    "CohortSubjectRow",
    "CohortGroupSummaryRow",
    "CohortComparisonRow",
    "TransitionEdge",
    "CohortBehaviorReport",
    "run_behavior_qc",
    "run_behavior_qc_cli",
    "run_behavior_irr",
    "run_behavior_irr_cli",
    "run_behavior_cohort",
    "run_behavior_cohort_cli",
    "run_behavior_time_budget_cli",
    "run_boris_import_cli",
    "run_boris_export_cli",
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
    disagreements_csv: Optional[Path | str] = None,
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
    if disagreements_csv is not None:
        write_irr_disagreements_csv(report, Path(disagreements_csv))

    if json_stdout:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_irr_report_table(report))

    if min_kappa is not None:
        if not report.rows:
            print(
                f"[annolid-run] No comparable behavior rows found to satisfy "
                f"--min-kappa {float(min_kappa):.4f}.",
                file=sys.stderr,
            )
            return 2
        if report.macro_cohens_kappa < float(min_kappa):
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
    session_duration: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exclude_behaviors: Optional[str] = None,
    schema_path: Optional[Path | str] = None,
    bout_profile: Optional[str] = None,
    bout_name: str = "custom_bout",
    bout_behaviors: Optional[str] = None,
    bout_initiation_behaviors: Optional[str] = None,
    bout_gap_seconds: Optional[float] = None,
    cooccurrence_csv: Optional[Path | str] = None,
    ethogram_svg: Optional[Path | str] = None,
    transitions_dot: Optional[Path | str] = None,
    transitions_svg: Optional[Path | str] = None,
    external_data_csvs: Optional[Sequence[Path | str]] = None,
    external_offset_s: float = 0.0,
    external_data_svg: Optional[Path | str] = None,
) -> int:
    """Execute ``annolid-run time-budget`` and return a process exit code."""
    csv_path = Path(events_csv)
    if not csv_path.exists():
        print(f"[annolid-run] Events file not found: {csv_path}", file=sys.stderr)
        return 2

    try:
        schema = _load_optional_schema(schema_path)
    except (FileNotFoundError, ValueError, OSError, BorisIOError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    try:
        rows = _read_event_rows(csv_path)
        intervals, warnings = extract_behavior_intervals(rows)
    except (OSError, TimeBudgetComputationError) as exc:
        print(f"[annolid-run] Failed to read events file: {exc}", file=sys.stderr)
        return 2

    excl_list = (
        [item.strip() for item in str(exclude_behaviors).split(",") if item.strip()]
        if exclude_behaviors
        else None
    )

    try:
        summary = summarize_intervals(
            intervals,
            session_duration=session_duration,
            start_time=start_time,
            end_time=end_time,
            exclude_behaviors=excl_list,
        )
        binned_rows = (
            compute_binned_time_budget(
                intervals,
                float(bin_size),
                start_time=start_time,
                end_time=end_time,
            )
            if bin_size is not None
            else None
        )
    except ValueError as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    coocc_rows = compute_behavior_cooccurrence(intervals)
    obs_transitions = compute_observation_transitions(intervals)

    loaded_ext_streams = []
    ext_bout_rows = []
    if external_data_csvs:
        try:
            for ext_item in external_data_csvs:
                for token in str(ext_item).split(","):
                    cleaned = token.strip()
                    if not cleaned:
                        continue
                    st = load_external_data_stream(
                        cleaned,
                        base_dir=csv_path.parent,
                        time_offset_s=float(external_offset_s or 0.0),
                    )
                    loaded_ext_streams.append(st)
                    ext_bout_rows.extend(
                        summarize_external_stream_by_behavior(st, intervals)
                    )
        except (OSError, ValueError, ExternalDataError) as exc:
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
                "std_duration": round(float(r.std_duration), 6),
                "median_duration": round(float(r.median_duration), 6),
                "min_duration": round(float(r.min_duration), 6),
                "max_duration": round(float(r.max_duration), 6),
                "percent_of_session": round(float(r.percent_of_session), 6),
                "rate_per_min": round(float(r.rate_per_min), 6),
                "latency_to_first": round(float(r.latency_to_first), 6),
                "inter_event_interval_mean": (
                    round(float(r.inter_event_interval_mean), 6)
                    if r.inter_event_interval_mean is not None
                    else None
                ),
                "inter_event_interval_std": (
                    round(float(r.inter_event_interval_std), 6)
                    if r.inter_event_interval_std is not None
                    else None
                ),
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
                "occurrences": int(r.occurrences),
                "duration": round(float(r.duration), 6),
            }
            for r in binned_rows
        ]
    if coocc_rows:
        payload["cooccurrences"] = [
            {
                "subject_a": r.subject_a,
                "behavior_a": r.behavior_a,
                "subject_b": r.subject_b,
                "behavior_b": r.behavior_b,
                "overlap_bouts": int(r.overlap_bouts),
                "overlap_duration": round(float(r.overlap_duration), 6),
                "pct_of_a": round(float(r.pct_of_a), 4),
                "pct_of_b": round(float(r.pct_of_b), 4),
                "jaccard_index": round(float(r.jaccard_index), 6),
            }
            for r in coocc_rows
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
    if ext_bout_rows:
        payload["external_data_summary"] = [
            {
                "stream": r.stream_label,
                "subject": r.subject,
                "behavior": r.behavior,
                "bouts": int(r.bouts),
                "samples": int(r.samples),
                "mean_value": round(float(r.mean_value), 6),
                "std_value": round(float(r.std_value), 6),
                "min_value": round(float(r.min_value), 6),
                "max_value": round(float(r.max_value), 6),
            }
            for r in ext_bout_rows
        ]

    if json_out is not None:
        atomic_write_json(Path(json_out), payload, indent=2, ensure_ascii=False)
    if cooccurrence_csv is not None:
        write_cooccurrence_csv(coocc_rows, Path(cooccurrence_csv))
    if ethogram_svg is not None:
        write_ethogram_svg(
            intervals,
            Path(ethogram_svg),
            schema=schema,
            start_time=start_time,
            end_time=end_time,
            exclude_behaviors=excl_list,
        )
    if transitions_dot is not None:
        write_transitions_graphviz_dot(obs_transitions, Path(transitions_dot))
    if transitions_svg is not None:
        write_transitions_svg(obs_transitions, Path(transitions_svg))
    if external_data_svg is not None and loaded_ext_streams:
        write_external_data_svg(
            loaded_ext_streams,
            Path(external_data_svg),
            intervals=intervals,
            start_time=start_time,
            end_time=end_time,
        )

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
        if binned_rows is not None:
            binned_path = out_path.with_name(
                out_path.stem + "_binned" + out_path.suffix
            )
            bin_buf = io.StringIO()
            bin_writer = csv.writer(bin_buf)
            bin_writer.writerow(
                [
                    "Subject",
                    "Behavior",
                    "BinStartSeconds",
                    "BinEndSeconds",
                    "Occurrences",
                    "DurationSeconds",
                ]
            )
            for b_row in binned_rows:
                bin_writer.writerow(
                    [
                        b_row.subject,
                        b_row.behavior,
                        f"{b_row.bin_start:.6f}",
                        f"{b_row.bin_end:.6f}",
                        int(b_row.occurrences),
                        f"{b_row.duration:.6f}",
                    ]
                )
            atomic_write_text(binned_path, bin_buf.getvalue(), encoding="utf-8")

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
        if coocc_rows:
            print()
            print("Behavioral Co-occurrence")
            print(format_cooccurrence_table(coocc_rows))
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
    analysis_unit: str = "subject",
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
        analysis_unit=analysis_unit,
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
    analysis_unit: str = "subject",
    json_stdout: bool = False,
    out_json: Optional[Path | str] = None,
    summary_csv: Optional[Path | str] = None,
    subjects_csv: Optional[Path | str] = None,
    transitions_csv: Optional[Path | str] = None,
    comparisons_csv: Optional[Path | str] = None,
    transitions_dot: Optional[Path | str] = None,
    transitions_svg: Optional[Path | str] = None,
) -> int:
    """Execute ``annolid-run behavior-cohort`` and return a process exit code."""
    excluded: List[Path] = [
        Path(p)
        for p in (
            out_json,
            summary_csv,
            subjects_csv,
            transitions_csv,
            comparisons_csv,
            transitions_dot,
            transitions_svg,
        )
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
            analysis_unit=analysis_unit,
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
            "analysis_unit": str(analysis_unit or "subject"),
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
    if transitions_dot is not None:
        write_transitions_graphviz_dot(report.transitions, Path(transitions_dot))
    if transitions_svg is not None:
        write_transitions_svg(report.transitions, Path(transitions_svg))
    if comparisons_csv is not None:
        write_cohort_comparisons_csv(report.comparisons, Path(comparisons_csv))

    if json_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_cohort_report_table(report))

    return 0


def run_boris_import_cli(
    *,
    boris_path: Path | str,
    output_dir: Path | str,
    observations: Optional[Sequence[str]] = None,
    apply_time_offset: bool = False,
    group_variable: Optional[str] = None,
    json_stdout: bool = False,
) -> int:
    """Execute ``annolid-run boris-import`` and return a process exit code."""
    try:
        artifacts = materialize_boris_project(
            boris_path,
            output_dir,
            observation_filter=observations,
            apply_time_offset=apply_time_offset,
            group_variable=group_variable,
        )
    except (FileNotFoundError, ValueError, OSError, BorisIOError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    for w in artifacts.warnings:
        print(f"[warning] {w}", file=sys.stderr)

    payload = {
        "schema": "boris_import/v1",
        "output_dir": str(artifacts.output_dir),
        "schema_path": str(artifacts.schema_path),
        "observations": {k: str(v) for k, v in artifacts.observation_csvs.items()},
        "cohort_manifest": (
            str(artifacts.cohort_manifest_path)
            if artifacts.cohort_manifest_path is not None
            else None
        ),
        "warnings": list(artifacts.warnings),
    }
    if json_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Imported BORIS project into: {artifacts.output_dir}")
        print(f"  Schema: {artifacts.schema_path}")
        print(f"  Observations exported: {len(artifacts.observation_csvs)}")
        if artifacts.cohort_manifest_path is not None:
            print(f"  Cohort manifest: {artifacts.cohort_manifest_path}")
    return 0


def run_boris_export_cli(
    *,
    schema_path: Optional[Path | str] = None,
    events_csvs: Optional[Sequence[Path | str]] = None,
    output_boris: Optional[Path | str] = None,
    tabular_csv: Optional[Path | str] = None,
    aggregated_csv: Optional[Path | str] = None,
    project_name: str = "Annolid Behavioral Project",
    json_stdout: bool = False,
) -> int:
    """Execute ``annolid-run boris-export`` and return a process exit code."""
    if output_boris is None and tabular_csv is None and aggregated_csv is None:
        print(
            "[annolid-run] Provide at least one output: -o/--output (.boris), --tabular-csv, or --aggregated-csv.",
            file=sys.stderr,
        )
        return 2

    try:
        schema = (
            _load_optional_schema(schema_path)
            if schema_path is not None
            else default_behavior_spec()
        )
        assert schema is not None
        observations: List[BorisObservation] = [
            load_boris_observation_from_csv(p) for p in (events_csvs or ())
        ]
        # Auto-enrich schema behaviors/subjects from events if schema had only placeholder defaults
        known_codes = {b.code for b in schema.behaviors}
        known_subjs = {s.name for s in schema.subjects} | {
            s.id for s in schema.subjects
        }
        from annolid.behavior.project_schema import (
            BehaviorDefinition,
            SubjectDefinition,
        )

        for obs in observations:
            for ev in obs.events:
                if ev.subject and ev.subject not in known_subjs:
                    known_subjs.add(ev.subject)
                    schema.subjects.append(
                        SubjectDefinition(
                            id=ev.subject.lower().replace(" ", "_"), name=ev.subject
                        )
                    )
                if ev.behavior and ev.behavior not in known_codes:
                    known_codes.add(ev.behavior)
                    schema.behaviors.append(
                        BehaviorDefinition(
                            code=ev.behavior,
                            name=ev.behavior,
                            is_state=(ev.event_kind != "point"),
                        )
                    )

        written: Dict[str, str] = {}
        if output_boris is not None:
            p_boris = write_boris_project(
                output_boris,
                schema,
                observations=observations,
                project_name=project_name,
            )
            written["boris_project"] = str(p_boris)
        if tabular_csv is not None:
            all_rows = [ev for obs in observations for ev in obs.events]
            obs_id = observations[0].observation_id if observations else "Observation_1"
            p_tab = write_boris_tabular_csv(
                all_rows, tabular_csv, observation_id=obs_id, schema=schema
            )
            written["tabular_csv"] = str(p_tab)
        if aggregated_csv is not None:
            all_rows = [ev for obs in observations for ev in obs.events]
            obs_id = observations[0].observation_id if observations else "Observation_1"
            p_agg = write_boris_aggregated_csv(
                all_rows, aggregated_csv, observation_id=obs_id, schema=schema
            )
            written["aggregated_csv"] = str(p_agg)
    except (FileNotFoundError, ValueError, OSError, BorisIOError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    if json_stdout:
        print(json.dumps({"ok": True, **written}, indent=2, ensure_ascii=False))
    else:
        for label, dst in written.items():
            print(f"Wrote {label}: {dst}")
    return 0
