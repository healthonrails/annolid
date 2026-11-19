"""Core FunctionTools for tracking QC, kinematics, zones, behavior statistics, and unified reporting."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from annolid.behavior.cohort import load_cohort_manifest
from annolid.behavior.time_budget import (
    _read_event_rows,
    compute_binned_time_budget,
    extract_behavior_intervals,
    summarize_intervals,
    write_time_budget_csv,
)
from annolid.core.behavior.spec import load_behavior_spec
from annolid.postprocessing.kinematics_stats import (
    KinematicsComputationError,
    write_kinematics_events_csv,
    write_kinematics_report_bundle,
)
from annolid.postprocessing.track_qc import (
    TrackQCComputationError,
    write_review_queue_csv,
    write_track_qc_issues_csv,
    write_track_qc_report_json,
)
from annolid.postprocessing.tracking_stats_dashboard import (
    discover_tracking_stats_files,
)
from annolid.services.behavior_analytics import (
    CohortComputationError,
    IRRComputationError,
    run_behavior_cohort,
    run_behavior_irr,
    run_behavior_qc,
)
from annolid.services.kinematics_analytics import run_kinematics_analysis
from annolid.services.track_analytics import run_track_qc, run_tracking_stats
from annolid.services.unified_reporting import build_unified_analysis_report
from annolid.services.zone_analytics import run_zone_analysis
from annolid.utils.atomic_io import atomic_write_json

from .common import (
    _is_within_root,
    _normalize_allowed_read_roots,
    _resolve_read_path,
    _resolve_write_path,
)
from .function_base import FunctionTool

__all__ = [
    "AnnolidTrackingStatsTool",
    "AnnolidTrackQCTool",
    "AnnolidKinematicsTool",
    "AnnolidZoneAnalysisTool",
    "AnnolidTimeBudgetTool",
    "AnnolidBehaviorQCTool",
    "AnnolidBehaviorIRRTool",
    "AnnolidBehaviorCohortTool",
    "AnnolidUnifiedReportTool",
]


class _SandboxedAnalyticsTool(FunctionTool):
    """Shared path-sandboxing base for Annolid analytics and reporting tools."""

    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir, allowed_read_roots
        )

    def _read_path(self, raw: Any) -> Path:
        text = str(raw or "").strip()
        if not text:
            raise ValueError("A non-empty input path is required.")
        return _resolve_read_path(
            text,
            allowed_dir=self._allowed_dir,
            allowed_read_roots=self._allowed_read_roots,
        )

    def _opt_read_path(self, raw: Any) -> Optional[Path]:
        text = str(raw or "").strip()
        if not text:
            return None
        return self._read_path(text)

    def _opt_write_path(self, raw: Any) -> Optional[Path]:
        text = str(raw or "").strip()
        if not text:
            return None
        return _resolve_write_path(text, allowed_dir=self._allowed_dir)

    def _default_output_dir(self, source: Path, suffix_name: str) -> Path:
        base_parent = source if source.is_dir() else source.parent
        candidate = base_parent / suffix_name
        if self._allowed_dir is not None and not _is_within_root(
            candidate.resolve(), self._allowed_dir
        ):
            return self._allowed_dir / "reports" / suffix_name
        return candidate


class AnnolidTrackingStatsTool(_SandboxedAnalyticsTool):
    """Aggregate *_tracking_stats.json telemetry across a directory of videos."""

    @property
    def name(self) -> str:
        return "annolid_tracking_stats"

    @property
    def description(self) -> str:
        return (
            "Aggregate `*_tracking_stats.json` files recursively under a root directory "
            "into overview, abnormal-segment, and bad-shape CSVs and optional plots."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "root_dir": {
                    "type": "string",
                    "description": "Root directory to scan for *_tracking_stats.json files.",
                },
                "output_dir": {
                    "type": "string",
                    "description": "Optional directory inside the workspace for CSV/plot outputs.",
                },
                "include_plots": {
                    "type": "boolean",
                    "description": "Generate summary PNG plots when matplotlib is available.",
                },
            },
            "required": ["root_dir"],
        }

    async def execute(self, **kwargs: Any) -> str:
        root = self._read_path(kwargs.get("root_dir"))
        out_dir = self._opt_write_path(
            kwargs.get("output_dir")
        ) or self._default_output_dir(root, "tracking_stats_dashboard")
        include_plots = bool(kwargs.get("include_plots", True))
        return await asyncio.to_thread(self._run, root, out_dir, include_plots)

    @staticmethod
    def _run(root: Path, out_dir: Path, include_plots: bool) -> str:
        try:
            stats_files = discover_tracking_stats_files(root)
            artifacts = run_tracking_stats(
                root, output_dir=out_dir, include_plots=include_plots
            )
            ov_df = pd.read_csv(artifacts.overview_csv)
            ab_df = pd.read_csv(artifacts.abnormal_segments_csv)
            bs_df = pd.read_csv(artifacts.bad_shape_events_csv)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        return json.dumps(
            {
                "ok": True,
                "schema": "tracking_stats_summary/v1",
                "root_dir": str(root),
                "discovered_stats_files": len(stats_files),
                "videos_analyzed": int(len(ov_df)),
                "abnormal_segment_rows": int(len(ab_df)),
                "bad_shape_event_rows": int(len(bs_df)),
                "overview_csv": str(artifacts.overview_csv),
                "abnormal_segments_csv": str(artifacts.abnormal_segments_csv),
                "bad_shape_events_csv": str(artifacts.bad_shape_events_csv),
                "top_videos": ov_df.head(10).to_dict(orient="records"),
            },
            ensure_ascii=False,
        )


class AnnolidTrackQCTool(_SandboxedAnalyticsTool):
    """Audit multi-animal tracking CSVs or frame JSON folders for jumps, swaps, and gaps."""

    @property
    def name(self) -> str:
        return "annolid_track_qc"

    @property
    def description(self) -> str:
        return (
            "Audit multi-animal tracks (tracking CSV or LabelMe frame JSON directory) "
            "for velocity spikes, identity swaps, area jumps, pose bone stretch, and "
            "prioritized manual review bouts."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "input_path": {
                    "type": "string",
                    "description": "Path to *_tracking.csv, *_tracked.csv, or frame JSON directory.",
                },
                "schema_path": {"type": "string"},
                "pose_schema_path": {"type": "string"},
                "expected_instances": {"type": "integer", "minimum": 1},
                "fps": {"type": "number", "exclusiveMinimum": 0},
                "top_k_review": {"type": "integer", "minimum": 1, "default": 20},
                "out_json": {"type": "string"},
                "issues_csv": {"type": "string"},
                "review_csv": {"type": "string"},
            },
            "required": ["input_path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        inp = self._read_path(kwargs.get("input_path"))
        schema_p = self._opt_read_path(kwargs.get("schema_path"))
        pose_p = self._opt_read_path(kwargs.get("pose_schema_path"))
        out_json = self._opt_write_path(kwargs.get("out_json"))
        issues_csv = self._opt_write_path(kwargs.get("issues_csv"))
        review_csv = self._opt_write_path(kwargs.get("review_csv"))
        exp_inst = (
            int(kwargs["expected_instances"])
            if kwargs.get("expected_instances") is not None
            else None
        )
        fps = float(kwargs["fps"]) if kwargs.get("fps") is not None else None
        top_k = int(kwargs.get("top_k_review", 20))
        return await asyncio.to_thread(
            self._run,
            inp,
            schema_p,
            pose_p,
            exp_inst,
            fps,
            top_k,
            out_json,
            issues_csv,
            review_csv,
        )

    @staticmethod
    def _run(
        inp: Path,
        schema_p: Optional[Path],
        pose_p: Optional[Path],
        exp_inst: Optional[int],
        fps: Optional[float],
        top_k: int,
        out_json: Optional[Path],
        issues_csv: Optional[Path],
        review_csv: Optional[Path],
    ) -> str:
        try:
            report = run_track_qc(
                inp,
                schema_path=schema_p,
                pose_schema_path=pose_p,
                expected_instances=exp_inst,
                fps=fps,
                top_k_review=top_k,
            )
            if out_json is not None:
                write_track_qc_report_json(report, out_json)
            if issues_csv is not None:
                write_track_qc_issues_csv(report.issues, issues_csv)
            if review_csv is not None:
                write_review_queue_csv(report.review_queue, review_csv)
        except (FileNotFoundError, ValueError, OSError, TrackQCComputationError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        payload = report.to_dict()
        payload["ok"] = True
        return json.dumps(payload, ensure_ascii=False)


class AnnolidKinematicsTool(_SandboxedAnalyticsTool):
    """Compute calibrated kinematics, velocity, tortuosity, freezing bouts, and pairwise distances."""

    @property
    def name(self) -> str:
        return "annolid_kinematics_stats"

    @property
    def description(self) -> str:
        return (
            "Compute calibrated trajectory kinematics (distance in px/cm, speed, acceleration, "
            "angular velocity, straightness, tortuosity), freezing and high-activity bouts, "
            "and pairwise inter-animal proximity from tracking CSVs or frame JSON folders."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "Path to *_tracking.csv, *_tracked.csv, or frame JSON directory.",
                },
                "fps": {"type": "number", "exclusiveMinimum": 0},
                "px_per_cm": {"type": "number", "exclusiveMinimum": 0},
                "calibration_json": {
                    "type": "string",
                    "description": "Optional perspective homography / 4-corner arena calibration JSON file.",
                },
                "freeze_speed_threshold": {"type": "number", "minimum": 0},
                "min_freeze_duration_s": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.5,
                },
                "output_dir": {"type": "string"},
                "events_csv": {"type": "string"},
            },
            "required": ["source_path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        src = self._read_path(kwargs.get("source_path"))
        cal_p = self._opt_read_path(kwargs.get("calibration_json"))
        out_dir = self._opt_write_path(kwargs.get("output_dir"))
        ev_csv = self._opt_write_path(kwargs.get("events_csv"))
        fps = float(kwargs["fps"]) if kwargs.get("fps") is not None else None
        px_cm = (
            float(kwargs["px_per_cm"]) if kwargs.get("px_per_cm") is not None else None
        )
        freeze_thr = (
            float(kwargs["freeze_speed_threshold"])
            if kwargs.get("freeze_speed_threshold") is not None
            else None
        )
        min_freeze = float(kwargs.get("min_freeze_duration_s", 0.5))
        return await asyncio.to_thread(
            self._run, src, fps, px_cm, cal_p, freeze_thr, min_freeze, out_dir, ev_csv
        )

    @staticmethod
    def _run(
        src: Path,
        fps: Optional[float],
        px_cm: Optional[float],
        cal_p: Optional[Path],
        freeze_thr: Optional[float],
        min_freeze: float,
        out_dir: Optional[Path],
        ev_csv: Optional[Path],
    ) -> str:
        try:
            report = run_kinematics_analysis(
                src,
                fps=fps,
                px_per_cm=px_cm,
                calibration=cal_p,
                freeze_speed_threshold=freeze_thr,
                min_freeze_duration_s=min_freeze,
            )
            artifacts: Dict[str, str] = {}
            if out_dir is not None:
                artifacts = write_kinematics_report_bundle(
                    report, out_dir, include_plots=True
                )
            if ev_csv is not None:
                saved = write_kinematics_events_csv(report, ev_csv)
                artifacts["events_csv"] = str(saved)
        except (
            FileNotFoundError,
            ValueError,
            OSError,
            KinematicsComputationError,
        ) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        payload = report.to_dict()
        payload["ok"] = True
        if artifacts:
            payload["artifacts"] = artifacts
        return json.dumps(payload, ensure_ascii=False)


class AnnolidZoneAnalysisTool(_SandboxedAnalyticsTool):
    """Compute arena zone occupancy, entries, latency, place preference, and social proximity."""

    @property
    def name(self) -> str:
        return "annolid_zone_analysis"

    @property
    def description(self) -> str:
        return (
            "Compute spatial ROI zone occupancy %, entries, first-entry latency, visit dwell "
            "times, assay place-preference summaries, and social proximity metrics."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "video_path": {
                    "type": "string",
                    "description": "Path to video file, tracking CSV, or frame JSON directory.",
                },
                "zone_file": {
                    "type": "string",
                    "description": "Optional *_zones.json file defining polygon/rectangle/circle zones.",
                },
                "assay_profile": {"type": "string", "default": "generic"},
                "mode": {
                    "type": "string",
                    "enum": [
                        "legacy_csv",
                        "zone_metrics",
                        "assay_summary",
                        "social_summary",
                        "zone_corrected_tracked_csv",
                    ],
                    "default": "zone_metrics",
                },
                "anchor_part": {"type": "string"},
                "fps": {"type": "number", "exclusiveMinimum": 0},
                "output_dir": {"type": "string"},
                "events_csv": {"type": "string"},
            },
            "required": ["video_path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        vid = self._read_path(kwargs.get("video_path"))
        z_file = self._opt_read_path(kwargs.get("zone_file"))
        out_dir = self._opt_write_path(
            kwargs.get("output_dir")
        ) or self._default_output_dir(vid, f"{vid.stem}_zones")
        ev_csv = self._opt_write_path(kwargs.get("events_csv"))
        profile = str(kwargs.get("assay_profile", "generic") or "generic")
        mode = str(kwargs.get("mode", "zone_metrics") or "zone_metrics")
        anchor = (
            str(kwargs["anchor_part"]).strip() if kwargs.get("anchor_part") else None
        )
        fps = float(kwargs["fps"]) if kwargs.get("fps") is not None else None
        return await asyncio.to_thread(
            self._run, vid, z_file, profile, mode, anchor, fps, out_dir, ev_csv
        )

    @staticmethod
    def _run(
        vid: Path,
        z_file: Optional[Path],
        profile: str,
        mode: str,
        anchor: Optional[str],
        fps: Optional[float],
        out_dir: Path,
        ev_csv: Optional[Path],
    ) -> str:
        try:
            bundle = run_zone_analysis(
                vid,
                zone_file=z_file,
                fps=fps,
                assay_profile=profile,
                mode=mode,
                anchor_part=anchor,
                output_dir=out_dir,
                events_csv=ev_csv,
            )
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        payload = bundle.to_dict()
        payload["ok"] = True
        return json.dumps(payload, ensure_ascii=False)


class AnnolidTimeBudgetTool(_SandboxedAnalyticsTool):
    """Compute behavior time budgets, bout durations, and binned summaries from an event CSV."""

    @property
    def name(self) -> str:
        return "annolid_behavior_time_budget"

    @property
    def description(self) -> str:
        return (
            "Compute BORIS-style behavior time budgets (occurrences, total/mean/median/min/max "
            "bout duration, % of session, and optional time-binned summaries) from a behavior event CSV."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "events_csv": {"type": "string"},
                "session_duration": {"type": "number", "exclusiveMinimum": 0},
                "bin_size": {"type": "number", "exclusiveMinimum": 0},
                "schema_path": {"type": "string"},
                "output_csv": {"type": "string"},
            },
            "required": ["events_csv"],
        }

    async def execute(self, **kwargs: Any) -> str:
        ev = self._read_path(kwargs.get("events_csv"))
        schema_p = self._opt_read_path(kwargs.get("schema_path"))
        out_csv = self._opt_write_path(kwargs.get("output_csv"))
        sess_dur = (
            float(kwargs["session_duration"])
            if kwargs.get("session_duration") is not None
            else None
        )
        bin_sz = (
            float(kwargs["bin_size"]) if kwargs.get("bin_size") is not None else None
        )
        return await asyncio.to_thread(
            self._run, ev, sess_dur, bin_sz, schema_p, out_csv
        )

    @staticmethod
    def _run(
        ev: Path,
        sess_dur: Optional[float],
        bin_sz: Optional[float],
        schema_p: Optional[Path],
        out_csv: Optional[Path],
    ) -> str:
        try:
            schema = load_behavior_spec(path=schema_p)[0] if schema_p else None
            rows = _read_event_rows(ev)
            intervals, warnings = extract_behavior_intervals(rows)
            summary = summarize_intervals(intervals, session_duration=sess_dur)
            binned = (
                compute_binned_time_budget(intervals, float(bin_sz))
                if bin_sz is not None and bin_sz > 0
                else []
            )
            if out_csv is not None:
                write_time_budget_csv(summary, out_csv, schema=schema)
        except (FileNotFoundError, ValueError, OSError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        return json.dumps(
            {
                "ok": True,
                "schema": "behavior_time_budget/v1",
                "events_csv": str(ev),
                "interval_count": len(intervals),
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
                        "percent_of_session": round(float(r.percent_of_session), 4),
                    }
                    for r in summary
                ],
                "binned_rows": [
                    {
                        "subject": b.subject,
                        "behavior": b.behavior,
                        "bin_index": idx,
                        "bin_start": round(float(b.bin_start), 4),
                        "bin_end": round(float(b.bin_end), 4),
                        "duration": round(float(b.duration), 6),
                        "percent_of_bin": round(
                            100.0
                            * float(b.duration)
                            / max(1e-9, float(b.bin_end) - float(b.bin_start)),
                            4,
                        ),
                    }
                    for idx, b in enumerate(binned)
                ],
            },
            ensure_ascii=False,
        )


class AnnolidBehaviorQCTool(_SandboxedAnalyticsTool):
    """Validate behavior event CSVs for unpaired state intervals and mutual-exclusion violations."""

    @property
    def name(self) -> str:
        return "annolid_behavior_qc"

    @property
    def description(self) -> str:
        return (
            "Audit a behavior event CSV for unpaired start/stop events, duplicate starts, "
            "negative/zero intervals, and mutually exclusive behavior overlaps."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "events_csv": {"type": "string"},
                "schema_path": {"type": "string"},
                "strict_schema": {"type": "boolean", "default": False},
            },
            "required": ["events_csv"],
        }

    async def execute(self, **kwargs: Any) -> str:
        ev = self._read_path(kwargs.get("events_csv"))
        schema_p = self._opt_read_path(kwargs.get("schema_path"))
        strict_schema = bool(kwargs.get("strict_schema", False))
        return await asyncio.to_thread(self._run, ev, schema_p, strict_schema)

    @staticmethod
    def _run(ev: Path, schema_p: Optional[Path], strict_schema: bool) -> str:
        try:
            report = run_behavior_qc(
                ev, schema_path=schema_p, strict_schema=strict_schema
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        payload = report.to_dict()
        payload["ok"] = True
        return json.dumps(payload, ensure_ascii=False)


class AnnolidBehaviorIRRTool(_SandboxedAnalyticsTool):
    """Compute Inter-Rater Reliability (Cohen's kappa, Fleiss' kappa, Krippendorff's alpha, CI, disagreements)."""

    @property
    def name(self) -> str:
        return "annolid_behavior_irr"

    @property
    def description(self) -> str:
        return (
            "Compute Inter-Rater Reliability across 2 or more behavior event CSVs: "
            "Cohen's kappa (with bootstrap 95% CI), Fleiss' kappa, Krippendorff's alpha, "
            "PABAK, bout onset/offset F1, and rater disagreement windows."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "events_csvs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 2,
                },
                "rater_names": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "bin_seconds": {
                    "type": "number",
                    "exclusiveMinimum": 0,
                    "default": 0.1,
                },
                "onset_tolerance_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.5,
                },
                "pool_subjects": {"type": "boolean", "default": False},
                "schema_path": {"type": "string"},
            },
            "required": ["events_csvs"],
        }

    async def execute(self, **kwargs: Any) -> str:
        raw_csvs = kwargs.get("events_csvs")
        if not isinstance(raw_csvs, list) or len(raw_csvs) < 2:
            raise ValueError("Provide at least 2 CSV paths in `events_csvs`.")
        csv_paths = [self._read_path(p) for p in raw_csvs]
        schema_p = self._opt_read_path(kwargs.get("schema_path"))
        raters = (
            [str(r).strip() for r in kwargs["rater_names"] if str(r).strip()]
            if isinstance(kwargs.get("rater_names"), list)
            else None
        )
        bin_s = float(kwargs.get("bin_seconds", 0.1))
        tol_s = float(kwargs.get("onset_tolerance_seconds", 0.5))
        pool = bool(kwargs.get("pool_subjects", False))
        return await asyncio.to_thread(
            self._run, csv_paths, raters, schema_p, bin_s, tol_s, pool
        )

    @staticmethod
    def _run(
        csv_paths: List[Path],
        raters: Optional[List[str]],
        schema_p: Optional[Path],
        bin_s: float,
        tol_s: float,
        pool: bool,
    ) -> str:
        try:
            report = run_behavior_irr(
                csv_paths,
                rater_names=raters,
                schema_path=schema_p,
                bin_seconds=bin_s,
                onset_tolerance_seconds=tol_s,
                pool_subjects=pool,
            )
        except (FileNotFoundError, ValueError, OSError, IRRComputationError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        payload = report.to_dict()
        payload["ok"] = True
        return json.dumps(payload, ensure_ascii=False)


class AnnolidBehaviorCohortTool(_SandboxedAnalyticsTool):
    """Aggregate multi-recording cohort behavior endpoints, inferential group tests, and Markov transitions."""

    @property
    def name(self) -> str:
        return "annolid_behavior_cohort"

    @property
    def description(self) -> str:
        return (
            "Analyze multi-recording experimental cohorts: group summaries, inferential "
            "hypothesis tests (Welch t/ANOVA, Mann-Whitney/Kruskal-Wallis, Hedges' g, "
            "Cliff's delta, Kaplan-Meier latency + log-rank, BH-FDR q-values), and Markov transitions."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "events_csvs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "manifest_path": {"type": "string"},
                "metadata_csv": {"type": "string"},
                "schema_path": {"type": "string"},
                "session_duration": {"type": "number", "exclusiveMinimum": 0},
                "analysis_unit": {
                    "type": "string",
                    "enum": ["subject", "recording"],
                    "default": "subject",
                },
                "out_json": {"type": "string"},
            },
        }

    async def execute(self, **kwargs: Any) -> str:
        raw_csvs = kwargs.get("events_csvs") or []
        csv_paths = [self._read_path(p) for p in raw_csvs]
        manifest_p = self._opt_read_path(kwargs.get("manifest_path"))
        meta_p = self._opt_read_path(kwargs.get("metadata_csv"))
        schema_p = self._opt_read_path(kwargs.get("schema_path"))
        out_json = self._opt_write_path(kwargs.get("out_json"))
        if not csv_paths and manifest_p is None:
            raise ValueError("Provide `events_csvs` or `manifest_path`.")
        if manifest_p is not None:
            for spec in load_cohort_manifest(manifest_p):
                self._read_path(str(spec.events_csv))
        sess_dur = (
            float(kwargs["session_duration"])
            if kwargs.get("session_duration") is not None
            else None
        )
        unit = str(kwargs.get("analysis_unit", "subject") or "subject")
        return await asyncio.to_thread(
            self._run, csv_paths, manifest_p, meta_p, schema_p, sess_dur, unit, out_json
        )

    @staticmethod
    def _run(
        csv_paths: List[Path],
        manifest_p: Optional[Path],
        meta_p: Optional[Path],
        schema_p: Optional[Path],
        sess_dur: Optional[float],
        unit: str,
        out_json: Optional[Path],
    ) -> str:
        try:
            report = run_behavior_cohort(
                manifest_path=manifest_p,
                events_csvs=csv_paths,
                metadata_csv=meta_p,
                schema_path=schema_p,
                session_duration=sess_dur,
                analysis_unit=unit,
            )
            payload = report.to_dict()
            if out_json is not None:
                atomic_write_json(out_json, payload, indent=2, ensure_ascii=False)
        except (FileNotFoundError, ValueError, OSError, CohortComputationError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        payload["ok"] = True
        return json.dumps(payload, ensure_ascii=False)


class AnnolidUnifiedReportTool(_SandboxedAnalyticsTool):
    """Compose a unified HTML5 + inline SVG + Markdown + JSON scientific report."""

    @property
    def name(self) -> str:
        return "annolid_generate_report"

    @property
    def description(self) -> str:
        return (
            "Generate a self-contained scientific analysis report (HTML5 with inline SVG "
            "charts, Markdown, and JSON) combining Track QC, Gap Analysis, Calibrated "
            "Kinematics & Freezing, Spatial Zones, Body-Part Ethograms, and Behavior Time Budgets."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source_path": {
                    "type": "string",
                    "description": "Path to video, tracking CSV, frame JSON directory, or workspace folder.",
                },
                "output_dir": {"type": "string"},
                "events_csv": {"type": "string"},
                "zone_file": {"type": "string"},
                "calibration_json": {
                    "type": "string",
                    "description": "Optional perspective homography / 4-corner arena calibration JSON file.",
                },
                "schema_path": {"type": "string"},
                "pose_schema_path": {"type": "string"},
                "fps": {"type": "number", "exclusiveMinimum": 0},
                "px_per_cm": {"type": "number", "exclusiveMinimum": 0},
                "title": {"type": "string"},
                "include_plots": {"type": "boolean", "default": True},
            },
            "required": ["source_path"],
        }

    async def execute(self, **kwargs: Any) -> str:
        src = self._read_path(kwargs.get("source_path"))
        out_dir = self._opt_write_path(
            kwargs.get("output_dir")
        ) or self._default_output_dir(src, f"{src.stem or src.name}_analysis_report")
        ev_csv = self._opt_read_path(kwargs.get("events_csv"))
        z_file = self._opt_read_path(kwargs.get("zone_file"))
        cal_p = self._opt_read_path(kwargs.get("calibration_json"))
        schema_p = self._opt_read_path(kwargs.get("schema_path"))
        pose_p = self._opt_read_path(kwargs.get("pose_schema_path"))
        fps = float(kwargs["fps"]) if kwargs.get("fps") is not None else None
        px_cm = (
            float(kwargs["px_per_cm"]) if kwargs.get("px_per_cm") is not None else None
        )
        title = str(kwargs["title"]).strip() if kwargs.get("title") else None
        inc_plots = bool(kwargs.get("include_plots", True))
        return await asyncio.to_thread(
            self._run,
            src,
            out_dir,
            ev_csv,
            z_file,
            cal_p,
            schema_p,
            pose_p,
            fps,
            px_cm,
            title,
            inc_plots,
        )

    @staticmethod
    def _run(
        src: Path,
        out_dir: Path,
        ev_csv: Optional[Path],
        z_file: Optional[Path],
        cal_p: Optional[Path],
        schema_p: Optional[Path],
        pose_p: Optional[Path],
        fps: Optional[float],
        px_cm: Optional[float],
        title: Optional[str],
        inc_plots: bool,
    ) -> str:
        try:
            artifacts = build_unified_analysis_report(
                src,
                output_dir=out_dir,
                events_csv=ev_csv,
                zone_file=z_file,
                calibration_path=cal_p,
                schema_path=schema_p,
                pose_schema_path=pose_p,
                fps=fps,
                px_per_cm=px_cm,
                title=title,
                include_plots=inc_plots,
            )
        except (FileNotFoundError, ValueError, OSError) as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)
        return json.dumps(
            {
                "ok": True,
                "schema": "unified_analysis_report/v1",
                "overall_status": artifacts.payload.get("overall_status"),
                "status_reasons": artifacts.payload.get("status_reasons"),
                "output_dir": str(artifacts.output_dir),
                "html_report": str(artifacts.html_path),
                "markdown_report": str(artifacts.markdown_path),
                "json_report": str(artifacts.json_path),
                "sections_present": [
                    k
                    for k, v in (artifacts.payload.get("sections") or {}).items()
                    if v is not None
                ],
            },
            ensure_ascii=False,
        )
