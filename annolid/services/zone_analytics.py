"""Service-layer orchestration for generic, assay-profiled, and social zone analysis."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_assay_profiles import available_assay_profiles
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text
from annolid.version import get_version

__all__ = [
    "ZoneAnalysisBundle",
    "list_zone_assay_profiles",
    "export_zone_visits_to_behavior_events_csv",
    "run_zone_analysis",
    "run_zone_analysis_cli",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ZoneAnalysisBundle:
    """Structured output from zone / place-preference / social proximity analysis."""

    video_path: str
    zone_file: Optional[str]
    assay_profile: str
    mode: str
    fps: float
    zone_labels: Tuple[str, ...]
    summary_rows: List[Dict[str, Any]]
    artifacts: Dict[str, str]
    provenance: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": "zone_analysis/v1",
            "video_path": self.video_path,
            "zone_file": self.zone_file,
            "assay_profile": self.assay_profile,
            "mode": self.mode,
            "fps": round(float(self.fps), 6),
            "zone_labels": list(self.zone_labels),
            "summary_rows": self.summary_rows,
            "artifacts": dict(self.artifacts),
            "provenance": dict(self.provenance),
        }


def list_zone_assay_profiles() -> List[Dict[str, str]]:
    """Return available zone assay profile descriptors for UI/CLI/agent tools."""
    return [
        {
            "name": p.name,
            "title": p.title,
            "description": p.description,
        }
        for p in available_assay_profiles()
    ]


def export_zone_visits_to_behavior_events_csv(
    analyzer: TrackingResultsAnalyzer,
    output_path: Path | str,
) -> Path:
    """Convert per-instance contiguous ``ZoneVisit`` segments into a Behavior Event CSV."""
    dst = Path(output_path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Recording time", "Subject", "Behavior", "Event", "Modifiers"])

    df = analyzer._analysis_dataframe()
    engine = analyzer._engine_for_profile(analyzer.assay_profile)
    eff_fps = max(1e-3, float(analyzer.fps or 30.0))
    events: List[Tuple[float, str, str, str, str]] = []

    if engine is not None and not df.empty and "instance_name" in df.columns:
        names = sorted(
            {
                str(n).strip()
                for n in df["instance_name"].dropna().astype(str).unique()
                if str(n).strip() and ":" not in str(n)
            }
        )
        for inst in names:
            res = engine.analyze_instance(df, inst)
            for visit in res.segments:
                s_t = float(visit.start_frame) / eff_fps
                e_t = float(visit.end_frame + 1) / eff_fps
                beh = f"zone_{visit.zone_label}"
                events.append((s_t, inst, beh, "start", f"zone={visit.zone_label}"))
                events.append((e_t, inst, beh, "stop", f"zone={visit.zone_label}"))

    events.sort(key=lambda item: (item[0], 0 if item[3] == "stop" else 1, item[1]))
    for rec_t, subj, beh, ev_kind, mod in events:
        writer.writerow([f"{rec_t:.4f}", subj, beh, ev_kind, mod])
    atomic_write_text(dst, buf.getvalue(), encoding="utf-8")
    return dst


def _clean_nan_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    reset = df.reset_index() if df.index.name else df.copy()
    records: List[Dict[str, Any]] = []
    for raw in reset.to_dict(orient="records"):
        cleaned: Dict[str, Any] = {}
        for k, v in raw.items():
            if isinstance(v, float) and (pd.isna(v) or not pd.notna(v)):
                cleaned[str(k)] = None
            elif pd.isna(v):
                cleaned[str(k)] = None
            else:
                cleaned[str(k)] = v
        records.append(cleaned)
    return records


def run_zone_analysis(
    video_path: Path | str,
    *,
    zone_file: Optional[Path | str] = None,
    zone_policy_file: Optional[Path | str] = None,
    fps: Optional[float] = None,
    assay_profile: str = "generic",
    mode: str = "zone_metrics",
    anchor_part: Optional[str] = None,
    latency_reference_frame: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
    events_csv: Optional[Path | str] = None,
) -> ZoneAnalysisBundle:
    """Execute zone / assay / social proximity analysis and return structured summary + paths."""
    resolved_video = Path(video_path).expanduser().resolve()
    if resolved_video.suffix.lower() == ".csv":
        stem = resolved_video.stem
        for suffix in ("_tracking", "_tracked"):
            if stem.endswith(suffix) and len(stem) > len(suffix):
                stem = stem[: -len(suffix)]
                break
        resolved_video = resolved_video.parent / f"{stem}.mp4"
    elif resolved_video.is_dir():
        resolved_video = resolved_video.parent / f"{resolved_video.name}.mp4"
    analyzer = TrackingResultsAnalyzer(
        resolved_video,
        zone_file=Path(zone_file).expanduser().resolve() if zone_file else None,
        fps=fps,
        assay_profile=assay_profile,
        zone_policy_file=(
            Path(zone_policy_file).expanduser().resolve() if zone_policy_file else None
        ),
        anchor_part=anchor_part,
    )
    analyzer.read_csv_files()

    mode_norm = str(mode or "zone_metrics").strip().lower()
    valid_modes = {
        "legacy_csv",
        "zone_metrics",
        "assay_summary",
        "social_summary",
        "zone_corrected_tracked_csv",
    }
    if mode_norm not in valid_modes:
        raise ValueError(
            f"Unsupported zone analysis mode '{mode}'. Expected one of {sorted(valid_modes)}."
        )

    out_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else resolved_video.parent
    )
    out_root.mkdir(parents=True, exist_ok=True)

    artifacts: Dict[str, str] = {}
    summary_df = pd.DataFrame()

    if mode_norm == "legacy_csv":
        csv_out = out_root / f"{resolved_video.stem}_place_preference.csv"
        saved = analyzer.save_all_instances_zone_time_to_csv(str(csv_out))
        artifacts["primary_output"] = str(saved)
        if Path(saved).is_file():
            summary_df = pd.read_csv(saved)
    elif mode_norm == "zone_metrics":
        csv_out = (
            out_root
            / f"{resolved_video.stem}_{analyzer.assay_profile.name}_zone_metrics.csv"
        )
        saved = analyzer.save_zone_metrics_to_csv(str(csv_out))
        artifacts["primary_output"] = str(saved)
        if Path(saved).is_file():
            summary_df = pd.read_csv(saved)
    elif mode_norm == "assay_summary":
        md_out = analyzer.save_assay_summary_report(str(out_root))
        artifacts["primary_output"] = str(md_out)
        artifacts["markdown_report"] = str(md_out)
        csv_cand = (
            out_root
            / f"{resolved_video.stem}_{analyzer.assay_profile.name}_assay_summary.csv"
        )
        if csv_cand.is_file():
            artifacts["summary_csv"] = str(csv_cand)
            summary_df = pd.read_csv(csv_cand)
    elif mode_norm == "social_summary":
        md_out = analyzer.save_social_summary_report(
            str(out_root),
            assay_profile=analyzer.assay_profile.name,
            latency_reference_frame=latency_reference_frame,
        )
        artifacts["primary_output"] = str(md_out)
        artifacts["markdown_report"] = str(md_out)
        csv_cand = (
            out_root
            / f"{resolved_video.stem}_{analyzer.assay_profile.name}_social_summary.csv"
        )
        if csv_cand.is_file():
            artifacts["summary_csv"] = str(csv_cand)
            summary_df = pd.read_csv(csv_cand)
    elif mode_norm == "zone_corrected_tracked_csv":
        csv_out = out_root / f"{resolved_video.stem}_tracked_zone_corrected.csv"
        saved = analyzer.save_zone_corrected_tracked_csv(str(csv_out))
        artifacts["primary_output"] = str(saved)
        if Path(saved).is_file():
            summary_df = pd.read_csv(saved).head(50)

    if events_csv is not None:
        ev_saved = export_zone_visits_to_behavior_events_csv(analyzer, events_csv)
        artifacts["events_csv"] = str(ev_saved)

    provenance: Dict[str, Any] = {
        "schema": "zone_analysis/v1",
        "annolid_version": str(get_version() or "0.0.0"),
        "generated_at_utc": _utc_now_iso(),
        "parameters": {
            "assay_profile": analyzer.assay_profile.name,
            "mode": mode_norm,
            "anchor_part": anchor_part,
            "latency_reference_frame": latency_reference_frame,
            "zone_policy_file": str(zone_policy_file) if zone_policy_file else None,
        },
    }

    return ZoneAnalysisBundle(
        video_path=str(resolved_video),
        zone_file=str(analyzer.zone_file) if analyzer.zone_file else None,
        assay_profile=analyzer.assay_profile.name,
        mode=mode_norm,
        fps=float(analyzer.fps or 30.0),
        zone_labels=tuple(spec.display_label for spec in analyzer.zone_specs),
        summary_rows=_clean_nan_records(summary_df),
        artifacts=artifacts,
        provenance=provenance,
    )


def run_zone_analysis_cli(
    *,
    video_path: Path | str,
    zone_file: Optional[Path | str] = None,
    zone_policy_file: Optional[Path | str] = None,
    fps: Optional[float] = None,
    assay_profile: str = "generic",
    mode: str = "zone_metrics",
    anchor_part: Optional[str] = None,
    latency_reference_frame: Optional[int] = None,
    output_dir: Optional[Path | str] = None,
    out_json: Optional[Path | str] = None,
    events_csv: Optional[Path | str] = None,
    json_stdout: bool = False,
) -> int:
    """Execute ``annolid-run zone-analysis`` and return a process exit code."""
    try:
        bundle = run_zone_analysis(
            video_path,
            zone_file=zone_file,
            zone_policy_file=zone_policy_file,
            fps=fps,
            assay_profile=assay_profile,
            mode=mode,
            anchor_part=anchor_part,
            latency_reference_frame=latency_reference_frame,
            output_dir=output_dir,
            events_csv=events_csv,
        )
    except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    payload = bundle.to_dict()
    if out_json is not None:
        atomic_write_json(Path(out_json), payload, indent=2, ensure_ascii=False)

    if json_stdout:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(
            f"Video: {bundle.video_path} | Profile: {bundle.assay_profile} | Mode: {bundle.mode}"
        )
        print(
            f"Zones ({len(bundle.zone_labels)}): {', '.join(bundle.zone_labels) or '(none)'}"
        )
        for k, v in bundle.artifacts.items():
            print(f"  {k}: {v}")
    return 0
