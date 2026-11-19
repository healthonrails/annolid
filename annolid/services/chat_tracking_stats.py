"""Service helpers for GUI chat tracking-stats, QC, kinematics, and unified report tools."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from annolid.core.agent.gui_backend.tool_handlers_tracking_stats import (
    analyze_tracking_stats_tool as gui_analyze_tracking_stats_tool,
)
from annolid.postprocessing.kinematics_stats import format_kinematics_report_table
from annolid.postprocessing.track_qc import format_track_qc_report_table
from annolid.behavior.qc import format_qc_report_table
from annolid.services.behavior_analytics import run_behavior_qc
from annolid.services.kinematics_analytics import run_kinematics_analysis
from annolid.services.track_analytics import run_track_qc
from annolid.services.unified_reporting import build_unified_analysis_report


def analyze_chat_tracking_stats_tool(**kwargs: Any) -> dict[str, Any]:
    return gui_analyze_tracking_stats_tool(**kwargs)


def _infer_tracking_source(candidate: Path) -> Optional[Path]:
    if candidate.is_file():
        if candidate.suffix.lower() == ".csv":
            return candidate
        for suf in ("_tracking.csv", "_tracked.csv"):
            csv_cand = candidate.parent / f"{candidate.stem}{suf}"
            if csv_cand.is_file():
                return csv_cand
        json_dir = candidate.with_suffix("")
        if json_dir.is_dir():
            return json_dir
    elif candidate.is_dir():
        frame_jsons = [
            p
            for p in candidate.glob("*.json")
            if not p.name.endswith("_zones.json")
            and not p.name.endswith("_tracking_stats.json")
            and p.name != "project.annolid.json"
        ]
        has_ndjson_store = any(
            p.is_file() for p in candidate.glob("*_annotations.ndjson")
        )
        if frame_jsons or has_ndjson_store:
            return candidate
        for suf in ("*_tracking.csv", "*_tracked.csv"):
            matches = sorted(candidate.glob(suf))
            if len(matches) == 1:
                return matches[0]
    return None


def _is_behavior_events_csv(path: Path) -> bool:
    if not path.is_file() or path.suffix.lower() != ".csv":
        return False
    try:
        cols = {str(c).strip().lower() for c in pd.read_csv(path, nrows=0).columns}
    except Exception:
        return False
    return "behavior" in cols and "frame_number" not in cols


def run_chat_stats_slash_tool(
    *,
    path: str = "",
    fps: Optional[float] = None,
    px_per_cm: Optional[float] = None,
    calibration: str = "",
    default_source: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute `/stats` on a recording (kinematics) or directory (`*_tracking_stats.json`)."""
    target_str = str(path or "").strip()
    target = (
        Path(target_str).expanduser().resolve()
        if target_str
        else (Path(default_source).expanduser().resolve() if default_source else None)
    )
    if target is None or not target.exists():
        return {
            "ok": False,
            "error": "No valid video, tracking CSV, or workspace path provided for `/stats`.",
        }

    cal_arg = calibration.strip() or None
    track_src = _infer_tracking_source(target)
    if track_src is not None:
        try:
            report = run_kinematics_analysis(
                track_src,
                fps=fps,
                px_per_cm=px_per_cm,
                calibration=cal_arg,
            )
            return {
                "ok": True,
                "mode": "kinematics",
                "summary_text": format_kinematics_report_table(report),
                "report": report.to_dict(),
            }
        except Exception as exc:
            if not target.is_dir():
                return {"ok": False, "error": f"Kinematics stats failed: {exc}"}

    root_dir = target if target.is_dir() else target.parent
    stats_res = gui_analyze_tracking_stats_tool(
        root_dir=str(root_dir),
        include_plots=False,
        default_root_dir=root_dir,
    )
    if stats_res.get("ok"):
        return {
            "ok": True,
            "mode": "tracking_stats",
            "summary_text": (
                f"Tracking telemetry summary for `{root_dir}`:\n"
                f"- Discovered stats files: {stats_res.get('stats_files_count', 0)}\n"
                f"- Overview CSV: {stats_res.get('overview_csv', '')}\n"
                f"- Abnormal segments: {stats_res.get('abnormal_segment_rows', 0)}\n"
                f"- Bad-shape events: {stats_res.get('bad_shape_event_rows', 0)}"
            ),
            "report": stats_res,
        }
    return stats_res


def run_chat_qc_slash_tool(
    *,
    path: str = "",
    fps: Optional[float] = None,
    schema_path: str = "",
    default_source: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute `/qc` for either Behavior Event CSVs or Multi-Animal Tracking data."""
    target_str = str(path or "").strip()
    target = (
        Path(target_str).expanduser().resolve()
        if target_str
        else (Path(default_source).expanduser().resolve() if default_source else None)
    )
    if target is None or not target.exists():
        return {
            "ok": False,
            "error": "No valid tracking source or behavior events CSV found for `/qc`.",
        }

    schema_opt = (
        Path(schema_path).expanduser().resolve() if schema_path.strip() else None
    )
    if _is_behavior_events_csv(target):
        try:
            bqc = run_behavior_qc(target, schema_path=schema_opt)
            return {
                "ok": True,
                "mode": "behavior_qc",
                "summary_text": format_qc_report_table(bqc),
                "report": bqc.to_dict(),
            }
        except Exception as exc:
            return {"ok": False, "error": f"Behavior QC failed: {exc}"}

    track_src = _infer_tracking_source(target) or target
    try:
        tqc = run_track_qc(track_src, schema_path=schema_opt, fps=fps, top_k_review=15)
        return {
            "ok": True,
            "mode": "track_qc",
            "summary_text": format_track_qc_report_table(tqc),
            "report": tqc.to_dict(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"Track QC failed: {exc}"}


def run_chat_report_slash_tool(
    *,
    path: str = "",
    output_dir: str = "",
    events_csv: str = "",
    zone_file: str = "",
    calibration: str = "",
    fps: Optional[float] = None,
    px_per_cm: Optional[float] = None,
    default_source: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute `/report` to build a unified HTML5 + SVG + Markdown + JSON report."""
    target_str = str(path or "").strip()
    target = (
        Path(target_str).expanduser().resolve()
        if target_str
        else (Path(default_source).expanduser().resolve() if default_source else None)
    )
    if target is None or not target.exists():
        return {
            "ok": False,
            "error": "No valid video, tracking CSV, or folder found for `/report`.",
        }

    try:
        artifacts = build_unified_analysis_report(
            target,
            output_dir=output_dir.strip() or None,
            events_csv=events_csv.strip() or None,
            zone_file=zone_file.strip() or None,
            calibration_path=calibration.strip() or None,
            fps=fps,
            px_per_cm=px_per_cm,
            include_plots=True,
        )
    except Exception as exc:
        return {"ok": False, "error": f"Unified report generation failed: {exc}"}

    status = artifacts.payload.get("overall_status", "PASS")
    reasons = "; ".join(artifacts.payload.get("status_reasons") or [])
    summary_text = (
        f"Unified Scientific Analysis Report generated (`{status}` — {reasons}):\n"
        f"- HTML Report: `{artifacts.html_path}`\n"
        f"- Markdown Report: `{artifacts.markdown_path}`\n"
        f"- JSON Report: `{artifacts.json_path}`"
    )
    return {
        "ok": True,
        "overall_status": status,
        "html_report": str(artifacts.html_path),
        "markdown_report": str(artifacts.markdown_path),
        "json_report": str(artifacts.json_path),
        "summary_text": summary_text,
    }


__all__ = [
    "analyze_chat_tracking_stats_tool",
    "run_chat_stats_slash_tool",
    "run_chat_qc_slash_tool",
    "run_chat_report_slash_tool",
]
