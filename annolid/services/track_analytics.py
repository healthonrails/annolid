"""Service-layer orchestration for multi-animal track quality control and identity repair."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Mapping, Optional

from annolid.core.behavior.spec import ProjectSchema, load_behavior_spec
from annolid.postprocessing.identity_governor import (
    GovernorPolicy,
    IdentityCorrection,
    IdentityGovernorResult,
    run_identity_governor,
)
from annolid.postprocessing.temporal_identity_repair import run_temporal_identity_repair
from annolid.postprocessing.tracking_reports import (
    find_tracking_gaps,
    generate_reports as generate_tracking_gap_reports,
    summarize_tracking_gaps,
)
from annolid.postprocessing.tracking_stats_dashboard import (
    TrackingStatsArtifacts,
    analyze_and_visualize_tracking_stats,
    discover_tracking_stats_files,
)
from annolid.postprocessing.track_qc import (
    InstanceTrackSummary,
    ReviewBout,
    TrackQCComputationError,
    TrackQCIssue,
    TrackQCReport,
    evaluate_track_qc,
    format_track_qc_report_table,
    resolve_pose_schema,
    write_review_queue_csv,
    write_track_qc_issues_csv,
    write_track_qc_report_json,
)
from annolid.utils.atomic_io import atomic_write_json
from annolid.version import get_version

__all__ = [
    "TrackQCComputationError",
    "TrackQCIssue",
    "InstanceTrackSummary",
    "ReviewBout",
    "TrackQCReport",
    "TrackingStatsArtifacts",
    "IdentityCorrection",
    "IdentityGovernorResult",
    "run_track_qc",
    "run_track_qc_cli",
    "run_track_repair",
    "run_track_repair_cli",
    "run_gap_report",
    "run_gap_report_cli",
    "run_tracking_stats",
    "run_tracking_stats_cli",
]


def _load_optional_project_schema(
    schema_path: Optional[Path | str],
) -> tuple[Optional[ProjectSchema], Optional[Path]]:
    if schema_path is None:
        return None, None
    resolved = Path(schema_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Schema file not found: {resolved}")
    schema, _ = load_behavior_spec(path=resolved)
    return schema, resolved.parent


def _load_governor_policy(policy_path: Path | str) -> GovernorPolicy:
    resolved = Path(policy_path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Governor policy file not found: {resolved}")
    raw_text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - pyyaml is in base deps
            raise ValueError(
                f"PyYAML is required to load YAML policy file: {resolved}"
            ) from exc
        payload = yaml.safe_load(raw_text)
    else:
        payload = json.loads(raw_text)
    if not isinstance(payload, Mapping):
        raise ValueError(
            f"Governor policy file '{resolved}' must contain a JSON/YAML object."
        )
    return GovernorPolicy.from_dict(payload)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_track_qc(
    input_path: Path | str,
    *,
    schema_path: Optional[Path | str] = None,
    pose_schema_path: Optional[Path | str] = None,
    expected_instances: Optional[int] = None,
    fps: Optional[float] = None,
    max_speed_px_per_frame: Optional[float] = None,
    speed_mad_z: float = 6.0,
    min_jump_px: float = 25.0,
    max_area_ratio: float = 2.5,
    overlap_iou_threshold: float = 0.25,
    close_contact_px: float = 80.0,
    max_edge_stretch_ratio: float = 2.5,
    min_score: float = 0.35,
    bout_merge_gap_frames: int = 5,
    top_k_review: Optional[int] = 25,
) -> TrackQCReport:
    """Evaluate multi-animal track quality and build a prioritized review queue."""
    project_schema, schema_dir = _load_optional_project_schema(schema_path)
    pose_schema = resolve_pose_schema(
        pose_schema_path=pose_schema_path,
        project_schema=project_schema,
        project_schema_dir=schema_dir,
    )
    return evaluate_track_qc(
        input_path,
        pose_schema=pose_schema,
        project_schema=project_schema,
        expected_instances=expected_instances,
        fps=fps,
        max_speed_px_per_frame=max_speed_px_per_frame,
        speed_mad_z=speed_mad_z,
        min_jump_px=min_jump_px,
        max_area_ratio=max_area_ratio,
        overlap_iou_threshold=overlap_iou_threshold,
        close_contact_px=close_contact_px,
        max_edge_stretch_ratio=max_edge_stretch_ratio,
        min_score=min_score,
        bout_merge_gap_frames=bout_merge_gap_frames,
        top_k_review=top_k_review,
    )


def run_track_qc_cli(
    *,
    input_path: Path | str,
    schema_path: Optional[Path | str] = None,
    pose_schema_path: Optional[Path | str] = None,
    expected_instances: Optional[int] = None,
    fps: Optional[float] = None,
    max_speed_px_per_frame: Optional[float] = None,
    speed_mad_z: float = 6.0,
    min_jump_px: float = 25.0,
    max_area_ratio: float = 2.5,
    overlap_iou_threshold: float = 0.25,
    close_contact_px: float = 80.0,
    max_edge_stretch_ratio: float = 2.5,
    min_score: float = 0.35,
    bout_merge_gap_frames: int = 5,
    top_k_review: Optional[int] = 25,
    strict: bool = False,
    json_stdout: bool = False,
    out_json: Optional[Path | str] = None,
    issues_csv: Optional[Path | str] = None,
    review_csv: Optional[Path | str] = None,
) -> int:
    """Execute ``annolid-run track-qc`` and return a process exit code."""
    try:
        report = run_track_qc(
            input_path,
            schema_path=schema_path,
            pose_schema_path=pose_schema_path,
            expected_instances=expected_instances,
            fps=fps,
            max_speed_px_per_frame=max_speed_px_per_frame,
            speed_mad_z=speed_mad_z,
            min_jump_px=min_jump_px,
            max_area_ratio=max_area_ratio,
            overlap_iou_threshold=overlap_iou_threshold,
            close_contact_px=close_contact_px,
            max_edge_stretch_ratio=max_edge_stretch_ratio,
            min_score=min_score,
            bout_merge_gap_frames=bout_merge_gap_frames,
            top_k_review=top_k_review,
        )
    except (FileNotFoundError, ValueError, OSError, TrackQCComputationError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    provenance: Dict[str, Any] = {
        "schema": "track_qc/v1",
        "annolid_version": str(get_version() or "0.0.0"),
        "generated_at_utc": _utc_now_iso(),
        "parameters": {
            "expected_instances": expected_instances,
            "fps": fps,
            "max_speed_px_per_frame": max_speed_px_per_frame,
            "speed_mad_z": float(speed_mad_z),
            "min_jump_px": float(min_jump_px),
            "max_area_ratio": float(max_area_ratio),
            "overlap_iou_threshold": float(overlap_iou_threshold),
            "close_contact_px": float(close_contact_px),
            "max_edge_stretch_ratio": float(max_edge_stretch_ratio),
            "min_score": float(min_score),
            "bout_merge_gap_frames": int(bout_merge_gap_frames),
            "top_k_review": top_k_review,
            "strict": bool(strict),
        },
    }

    if out_json is not None:
        write_track_qc_report_json(report, Path(out_json), provenance=provenance)
    if issues_csv is not None:
        write_track_qc_issues_csv(report.issues, Path(issues_csv))
    if review_csv is not None:
        write_review_queue_csv(report.review_queue, Path(review_csv))

    if json_stdout:
        payload = report.to_dict()
        payload["provenance"] = provenance
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_track_qc_report_table(report))

    if report.has_errors or (strict and report.warning_count > 0):
        return 2
    return 0


def _correction_to_dict(correction: IdentityCorrection) -> Dict[str, Any]:
    return {
        "track_id": correction.track_id,
        "frame_start": int(correction.frame_start),
        "frame_end": int(correction.frame_end),
        "observed_label": correction.observed_label,
        "corrected_label": correction.corrected_label,
        "rule_name": correction.rule_name,
        "rule_frame_start": int(correction.rule_frame_start),
        "rule_frame_end": int(correction.rule_frame_end),
        "observation_count": int(correction.observation_count),
    }


def _enrich_repair_report_provenance(
    report_path: Path, provenance: Mapping[str, Any]
) -> None:
    if not report_path.exists():
        return
    try:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    if not isinstance(existing, dict):
        return
    existing["provenance"] = dict(provenance)
    atomic_write_json(report_path, existing, indent=2, ensure_ascii=False)


def run_track_repair(
    annotation_dir: Path | str,
    *,
    mode: str = "temporal",
    apply_changes: bool = False,
    report_path: Optional[Path | str] = None,
    start_frame: int = 0,
    expected_instance_count: Optional[int] = None,
    max_gap_frames: int = 5,
    max_match_distance: float = 80.0,
    policy_path: Optional[Path | str] = None,
    zone_file: Optional[Path | str] = None,
) -> IdentityGovernorResult:
    """Run temporal continuity or policy-based identity repair over LabelMe JSON frames.

    Defaults to ``apply_changes=False`` (non-mutating dry-run) so frame JSON files
    are never rewritten unless explicitly requested.
    """
    resolved_dir = Path(annotation_dir).expanduser().resolve()
    if not resolved_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {resolved_dir}")
    if not resolved_dir.is_dir():
        raise ValueError(
            f"Track repair requires a frame JSON directory: {resolved_dir}"
        )

    mode_norm = str(mode or "temporal").strip().lower()
    if mode_norm not in {"temporal", "governor"}:
        raise ValueError(
            f"Unsupported repair mode '{mode}'. Expected 'temporal' or 'governor'."
        )

    provenance: Dict[str, Any] = {
        "schema": "track_repair/v1",
        "annolid_version": str(get_version() or "0.0.0"),
        "generated_at_utc": _utc_now_iso(),
        "mode": mode_norm,
        "apply_changes": bool(apply_changes),
        "parameters": {
            "start_frame": int(start_frame),
            "expected_instance_count": expected_instance_count,
            "max_gap_frames": int(max_gap_frames),
            "max_match_distance": float(max_match_distance),
            "policy_path": str(policy_path) if policy_path is not None else None,
            "zone_file": str(zone_file) if zone_file is not None else None,
        },
    }

    if mode_norm == "temporal":
        result = run_temporal_identity_repair(
            resolved_dir,
            start_frame=max(0, int(start_frame)),
            expected_instance_count=expected_instance_count,
            max_gap_frames=max(0, int(max_gap_frames)),
            max_match_distance=max(0.0, float(max_match_distance)),
            apply_changes=bool(apply_changes),
            report_path=report_path,
        )
    else:
        if policy_path is None:
            raise ValueError(
                "--policy is required when running repair-tracks with --mode governor."
            )
        policy = _load_governor_policy(policy_path)
        resolved_zone: Optional[Path] = None
        if zone_file is not None:
            resolved_zone = Path(zone_file).expanduser().resolve()
            if not resolved_zone.exists():
                raise FileNotFoundError(f"Zone file not found: {resolved_zone}")
        result = run_identity_governor(
            resolved_dir,
            policy,
            zone_file=resolved_zone,
            apply_changes=bool(apply_changes),
            report_path=report_path,
        )

    _enrich_repair_report_provenance(result.report_path, provenance)
    return result


def _format_repair_summary_table(result: IdentityGovernorResult, *, mode: str) -> str:
    mode_banner = "DRY-RUN (no frame JSONs modified)" if result.dry_run else "APPLIED"
    lines: List[str] = [
        f"Directory: {result.annotation_dir}",
        f"Mode: {mode} | Status: {mode_banner}",
        (
            f"Scanned: {result.scanned_files} file(s), "
            f"{result.scanned_observations} observation(s) | "
            f"Proposed corrections: {len(result.proposed_corrections)} | "
            f"Updated: {result.updated_files} file(s), {result.updated_shapes} shape(s)"
        ),
        f"Audit report: {result.report_path}",
    ]
    if result.proposed_corrections:
        lines.append("")
        lines.append(
            f"{'Frames':<16} {'Track':<12} {'Observed':<18} {'Corrected':<18} {'Rule'}"
        )
        lines.append("-" * 86)
        for corr in result.proposed_corrections[:30]:
            span = f"{corr.frame_start}..{corr.frame_end}"
            lines.append(
                f"{span:<16} {corr.track_id:<12} {corr.observed_label:<18} "
                f"{corr.corrected_label:<18} {corr.rule_name}"
            )
        if len(result.proposed_corrections) > 30:
            lines.append(
                f"... ({len(result.proposed_corrections) - 30} additional correction(s) in {result.report_path})"
            )
    return "\n".join(lines)


def run_track_repair_cli(
    *,
    annotation_dir: Path | str,
    mode: str = "temporal",
    apply_changes: bool = False,
    report_path: Optional[Path | str] = None,
    json_out: Optional[Path | str] = None,
    json_stdout: bool = False,
    start_frame: int = 0,
    expected_instance_count: Optional[int] = None,
    max_gap_frames: int = 5,
    max_match_distance: float = 80.0,
    policy_path: Optional[Path | str] = None,
    zone_file: Optional[Path | str] = None,
) -> int:
    """Execute ``annolid-run repair-tracks`` and return a process exit code."""
    try:
        result = run_track_repair(
            annotation_dir,
            mode=mode,
            apply_changes=apply_changes,
            report_path=report_path,
            start_frame=start_frame,
            expected_instance_count=expected_instance_count,
            max_gap_frames=max_gap_frames,
            max_match_distance=max_match_distance,
            policy_path=policy_path,
            zone_file=zone_file,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    payload: Dict[str, Any] = {
        "schema": "track_repair_summary/v1",
        "annotation_dir": str(result.annotation_dir),
        "mode": str(mode or "temporal").strip().lower(),
        "dry_run": bool(result.dry_run),
        "scanned_files": int(result.scanned_files),
        "scanned_observations": int(result.scanned_observations),
        "proposed_correction_count": len(result.proposed_corrections),
        "updated_files": int(result.updated_files),
        "updated_shapes": int(result.updated_shapes),
        "report_path": str(result.report_path),
        "proposed_corrections": [
            _correction_to_dict(c) for c in result.proposed_corrections
        ],
        "provenance": {
            "annolid_version": str(get_version() or "0.0.0"),
            "generated_at_utc": _utc_now_iso(),
        },
    }

    if json_out is not None:
        atomic_write_json(Path(json_out), payload, indent=2, ensure_ascii=False)

    if json_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            _format_repair_summary_table(
                result, mode=str(mode or "temporal").strip().lower()
            )
        )
    return 0


def run_gap_report(
    source_path: Path | str,
    *,
    fps: Optional[float] = None,
    write_reports: bool = True,
    out_json: Optional[Path | str] = None,
) -> Dict[str, Any]:
    """Scan a video, tracking CSV, or frame JSON directory for per-instance tracking gaps."""
    resolved = Path(source_path).expanduser().resolve()
    if not resolved.exists():
        # Also check if companion CSV or JSON folder exists when a video path is passed
        stem = resolved.stem
        has_companion = (
            (resolved.parent / stem).is_dir()
            or (resolved.parent / f"{stem}_tracking.csv").is_file()
            or (resolved.parent / f"{stem}_tracked.csv").is_file()
        )
        if not has_companion:
            raise FileNotFoundError(f"Tracking source not found: {resolved}")

    summary = summarize_tracking_gaps(resolved, fps=fps)
    gaps = find_tracking_gaps(str(resolved))
    markdown_path: Optional[Path] = None
    csv_path: Optional[Path] = None
    if write_reports:
        md_result = generate_tracking_gap_reports(gaps, str(resolved))
        if md_result is not None:
            markdown_path = Path(md_result)
            candidate_csv = markdown_path.with_name(
                markdown_path.name.replace(
                    "_tracking_gaps_report.md", "_gaps_report.csv"
                )
            )
            if candidate_csv.exists():
                csv_path = candidate_csv

    payload: Dict[str, Any] = {
        **summary,
        "markdown_report": str(markdown_path) if markdown_path else None,
        "gaps_csv": str(csv_path) if csv_path else None,
        "provenance": {
            "annolid_version": str(get_version() or "0.0.0"),
            "generated_at_utc": _utc_now_iso(),
        },
    }
    if out_json is not None:
        atomic_write_json(Path(out_json), payload, indent=2, ensure_ascii=False)
    return payload


def _format_gap_report_table(payload: Mapping[str, Any]) -> str:
    lines: List[str] = [
        f"Source: {payload.get('video_path', '')}",
        (
            f"Total frames: {payload.get('total_frames', 0)} | "
            f"FPS: {payload.get('fps') or 'n/a'} | "
            f"Gap events: {payload.get('total_gap_events', 0)}"
        ),
    ]
    instances = payload.get("instances") or []
    if instances:
        lines.append("")
        lines.append(
            f"{'Instance':<22} {'Observed':>10} {'Missing':>10} {'Coverage':>10} {'Gaps':>8} {'MaxGap(f)':>10}"
        )
        lines.append("-" * 76)
        for inst in instances:
            cov_pct = float(inst.get("coverage_ratio", 0.0)) * 100.0
            lines.append(
                f"{str(inst.get('instance', '')):<22} "
                f"{int(inst.get('observed_frames', 0)):>10} "
                f"{int(inst.get('missing_frames', 0)):>10} "
                f"{cov_pct:>9.2f}% "
                f"{int(inst.get('gap_count', 0)):>8} "
                f"{int(inst.get('longest_gap_frames', 0)):>10}"
            )
    else:
        lines.append("No tracked instances discovered.")
    if payload.get("markdown_report"):
        lines.append(f"Markdown report: {payload['markdown_report']}")
    if payload.get("gaps_csv"):
        lines.append(f"Gaps CSV: {payload['gaps_csv']}")
    return "\n".join(lines)


def run_gap_report_cli(
    *,
    source_path: Path | str,
    fps: Optional[float] = None,
    write_reports: bool = True,
    out_json: Optional[Path | str] = None,
    json_stdout: bool = False,
) -> int:
    """Execute ``annolid-run gap-report`` and return an exit code."""
    try:
        payload = run_gap_report(
            source_path,
            fps=fps,
            write_reports=write_reports,
            out_json=out_json,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    if json_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(_format_gap_report_table(payload))
    return 0


def run_tracking_stats(
    root_dir: Path | str,
    *,
    output_dir: Optional[Path | str] = None,
    include_plots: bool = True,
) -> TrackingStatsArtifacts:
    """Aggregate ``*_tracking_stats.json`` files recursively into CSVs and optional plots."""
    resolved_root = Path(root_dir).expanduser().resolve()
    if not resolved_root.exists():
        raise FileNotFoundError(f"Root directory not found: {resolved_root}")
    if not resolved_root.is_dir():
        raise ValueError(f"Root path must be a directory: {resolved_root}")
    resolved_out = (
        Path(output_dir).expanduser().resolve() if output_dir is not None else None
    )
    return analyze_and_visualize_tracking_stats(
        root_dir=resolved_root,
        output_dir=resolved_out,
        include_plots=bool(include_plots),
    )


def run_tracking_stats_cli(
    *,
    root_dir: Path | str,
    output_dir: Optional[Path | str] = None,
    include_plots: bool = True,
    out_json: Optional[Path | str] = None,
    json_stdout: bool = False,
) -> int:
    """Execute ``annolid-run tracking-stats`` and return an exit code."""
    try:
        resolved_root = Path(root_dir).expanduser().resolve()
        stats_files = discover_tracking_stats_files(resolved_root)
        artifacts = run_tracking_stats(
            resolved_root,
            output_dir=output_dir,
            include_plots=include_plots,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    import pandas as pd

    try:
        overview_df = pd.read_csv(artifacts.overview_csv)
    except Exception:
        overview_df = pd.DataFrame()
    try:
        abnormal_df = pd.read_csv(artifacts.abnormal_segments_csv)
    except Exception:
        abnormal_df = pd.DataFrame()
    try:
        bad_shape_df = pd.read_csv(artifacts.bad_shape_events_csv)
    except Exception:
        bad_shape_df = pd.DataFrame()

    payload: Dict[str, Any] = {
        "schema": "tracking_stats_summary/v1",
        "root_dir": str(resolved_root),
        "discovered_stats_files": len(stats_files),
        "videos_analyzed": int(len(overview_df)),
        "abnormal_segment_rows": int(len(abnormal_df)),
        "bad_shape_event_rows": int(len(bad_shape_df)),
        "output_dir": str(artifacts.output_dir),
        "overview_csv": str(artifacts.overview_csv),
        "abnormal_segments_csv": str(artifacts.abnormal_segments_csv),
        "bad_shape_events_csv": str(artifacts.bad_shape_events_csv),
        "manual_badshape_plot": (
            str(artifacts.manual_badshape_plot)
            if artifacts.manual_badshape_plot
            else None
        ),
        "abnormal_segments_plot": (
            str(artifacts.abnormal_segments_plot)
            if artifacts.abnormal_segments_plot
            else None
        ),
        "unresolved_bad_shapes_plot": (
            str(artifacts.unresolved_bad_shapes_plot)
            if artifacts.unresolved_bad_shapes_plot
            else None
        ),
        "provenance": {
            "annolid_version": str(get_version() or "0.0.0"),
            "generated_at_utc": _utc_now_iso(),
        },
    }
    if out_json is not None:
        atomic_write_json(Path(out_json), payload, indent=2, ensure_ascii=False)

    if json_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(f"Root directory: {resolved_root}")
        print(
            f"Discovered stats files: {len(stats_files)} | "
            f"Videos: {len(overview_df)} | "
            f"Abnormal segments: {len(abnormal_df)} | "
            f"Bad-shape events: {len(bad_shape_df)}"
        )
        print(f"Output directory: {artifacts.output_dir}")
        print(f"Overview CSV: {artifacts.overview_csv}")
    return 0
