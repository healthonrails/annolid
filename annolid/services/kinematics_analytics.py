"""Service-layer orchestration for calibrated trajectory kinematics and body-part ethograms."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import io
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

from annolid.postprocessing.body_part_ethogram import (
    BodyPartEthogramReport,
    analyze_body_part_ethogram,
    export_body_part_ethogram_csvs,
)
from annolid.postprocessing.kinematics_stats import (
    CalibrationError,
    InstanceKinematicsRow,
    KinematicsBout,
    KinematicsComputationError,
    KinematicsReport,
    PairwiseKinematicsRow,
    PlanarCalibration,
    analyze_kinematics,
    format_kinematics_report_table,
    write_kinematics_events_csv,
    write_kinematics_report_bundle,
)
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text
from annolid.version import get_version

__all__ = [
    "CalibrationError",
    "PlanarCalibration",
    "KinematicsComputationError",
    "InstanceKinematicsRow",
    "KinematicsBout",
    "PairwiseKinematicsRow",
    "KinematicsReport",
    "BodyPartEthogramReport",
    "run_kinematics_analysis",
    "run_kinematics_cli",
    "export_body_part_ethogram_events_csv",
    "run_standalone_body_part_ethogram",
    "run_body_part_ethogram_cli",
]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_kinematics_analysis(
    source_path: Path | str,
    *,
    fps: Optional[float] = None,
    px_per_cm: Optional[float] = None,
    calibration: PlanarCalibration | Path | str | Dict[str, Any] | None = None,
    smooth_window: int = 5,
    freeze_speed_threshold: Optional[float] = None,
    min_freeze_duration_s: float = 0.5,
    close_contact_distance_px: float = 80.0,
) -> KinematicsReport:
    """Compute calibrated kinematics, freezing bouts, and pairwise inter-animal metrics."""
    return analyze_kinematics(
        source_path,
        fps=fps,
        px_per_cm=px_per_cm,
        calibration=calibration,
        smooth_window=smooth_window,
        freeze_speed_threshold=freeze_speed_threshold,
        min_freeze_duration_s=min_freeze_duration_s,
        close_contact_distance_px=close_contact_distance_px,
    )


def run_kinematics_cli(
    *,
    source_path: Path | str,
    fps: Optional[float] = None,
    px_per_cm: Optional[float] = None,
    calibration_path: Optional[Path | str] = None,
    smooth_window: int = 5,
    freeze_speed_threshold: Optional[float] = None,
    min_freeze_duration_s: float = 0.5,
    close_contact_distance_px: float = 80.0,
    output_dir: Optional[Path | str] = None,
    out_json: Optional[Path | str] = None,
    events_csv: Optional[Path | str] = None,
    include_plots: bool = True,
    json_stdout: bool = False,
) -> int:
    """Execute ``annolid-run kinematics`` and return a process exit code."""
    try:
        report = run_kinematics_analysis(
            source_path,
            fps=fps,
            px_per_cm=px_per_cm,
            calibration=calibration_path,
            smooth_window=smooth_window,
            freeze_speed_threshold=freeze_speed_threshold,
            min_freeze_duration_s=min_freeze_duration_s,
            close_contact_distance_px=close_contact_distance_px,
        )
    except (FileNotFoundError, ValueError, OSError, KinematicsComputationError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    provenance: Dict[str, Any] = {
        "schema": "kinematics_report/v1",
        "annolid_version": str(get_version() or "0.0.0"),
        "generated_at_utc": _utc_now_iso(),
        "parameters": {
            "fps": fps,
            "px_per_cm": px_per_cm,
            "calibration_path": str(calibration_path) if calibration_path else None,
            "smooth_window": int(smooth_window),
            "freeze_speed_threshold": freeze_speed_threshold,
            "min_freeze_duration_s": float(min_freeze_duration_s),
            "close_contact_distance_px": float(close_contact_distance_px),
        },
    }

    artifacts: Dict[str, str] = {}
    if output_dir is not None:
        artifacts = write_kinematics_report_bundle(
            report,
            output_dir,
            include_plots=include_plots,
            export_events_csv=True,
            provenance=provenance,
        )
    if out_json is not None:
        payload = report.to_dict()
        payload["provenance"] = provenance
        if artifacts:
            payload["artifacts"] = artifacts
        atomic_write_json(Path(out_json), payload, indent=2, ensure_ascii=False)
    if events_csv is not None:
        write_kinematics_events_csv(report, Path(events_csv))

    if json_stdout:
        payload = report.to_dict()
        payload["provenance"] = provenance
        if artifacts:
            payload["artifacts"] = artifacts
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print(format_kinematics_report_table(report))
    return 0


def export_body_part_ethogram_events_csv(
    report: BodyPartEthogramReport, output_path: Path | str
) -> Path:
    """Export directed social-contact bouts and posture states to canonical Behavior Event CSV."""
    dst = Path(output_path)
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(["Recording time", "Subject", "Behavior", "Event", "Modifiers"])

    events: List[Tuple[float, str, str, str, str]] = []
    eff_fps = max(1e-3, float(report.fps))

    for bout in report.bouts:
        s_t = float(bout.start_frame) / eff_fps
        e_t = float(bout.end_frame + 1) / eff_fps
        mod = f"recipient={bout.recipient}"
        events.append((s_t, bout.actor, bout.behavior, "start", mod))
        events.append((e_t, bout.actor, bout.behavior, "stop", mod))

    if not report.kinematics_df.empty and {
        "frame_number",
        "instance_name",
        "posture_state",
    }.issubset(report.kinematics_df.columns):
        sorted_kin = report.kinematics_df.sort_values(
            ["instance_name", "frame_number"], kind="mergesort"
        )
        for inst, grp in sorted_kin.groupby("instance_name", sort=True):
            frames = grp["frame_number"].astype(int).tolist()
            states = grp["posture_state"].fillna("").astype(str).tolist()
            if not frames:
                continue
            r_start = frames[0]
            r_prev = frames[0]
            r_state = states[0]
            for f_cur, st_cur in zip(frames[1:], states[1:]):
                if st_cur == r_state and (f_cur - r_prev) <= 2:
                    r_prev = f_cur
                else:
                    if r_state and r_state != "neutral":
                        events.append(
                            (
                                float(r_start) / eff_fps,
                                str(inst),
                                f"posture_{r_state}",
                                "start",
                                "",
                            )
                        )
                        events.append(
                            (
                                float(r_prev + 1) / eff_fps,
                                str(inst),
                                f"posture_{r_state}",
                                "stop",
                                "",
                            )
                        )
                    r_start = f_cur
                    r_prev = f_cur
                    r_state = st_cur
            if r_state and r_state != "neutral":
                events.append(
                    (
                        float(r_start) / eff_fps,
                        str(inst),
                        f"posture_{r_state}",
                        "start",
                        "",
                    )
                )
                events.append(
                    (
                        float(r_prev + 1) / eff_fps,
                        str(inst),
                        f"posture_{r_state}",
                        "stop",
                        "",
                    )
                )

    events.sort(key=lambda item: (item[0], 0 if item[3] == "stop" else 1, item[1]))
    for rec_t, subj, beh, ev_kind, mod in events:
        writer.writerow([f"{rec_t:.4f}", subj, beh, ev_kind, mod])
    atomic_write_text(dst, buf.getvalue(), encoding="utf-8")
    return dst


def run_standalone_body_part_ethogram(
    annotation_dir: Path | str,
    *,
    fps: float = 30.0,
    contact_margin_px: float = 6.0,
    min_bout_frames: int = 1,
    max_uncertain_frames: int = 10,
    camera_view: str = "auto",
) -> BodyPartEthogramReport:
    """Compute anatomical body-part kinematics and social-contact ethogram on existing annotations."""
    resolved = Path(annotation_dir).expanduser().resolve()
    if resolved.is_file():
        maybe_dir = resolved.with_suffix("")
        if maybe_dir.is_dir():
            resolved = maybe_dir
    return analyze_body_part_ethogram(
        resolved,
        fps=fps,
        contact_margin_px=contact_margin_px,
        min_bout_frames=min_bout_frames,
        max_uncertain_frames=max_uncertain_frames,
        camera_view=camera_view,
    )


def run_body_part_ethogram_cli(
    *,
    annotation_dir: Path | str,
    fps: float = 30.0,
    contact_margin_px: float = 6.0,
    min_bout_frames: int = 1,
    max_uncertain_frames: int = 10,
    camera_view: str = "auto",
    output_prefix: Optional[Path | str] = None,
    events_csv: Optional[Path | str] = None,
    out_json: Optional[Path | str] = None,
    json_stdout: bool = False,
) -> int:
    """Execute ``annolid-run body-part-ethogram`` and return a process exit code."""
    try:
        report = run_standalone_body_part_ethogram(
            annotation_dir,
            fps=fps,
            contact_margin_px=contact_margin_px,
            min_bout_frames=min_bout_frames,
            max_uncertain_frames=max_uncertain_frames,
            camera_view=camera_view,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"[annolid-run] {exc}", file=sys.stderr)
        return 2

    summary = report.to_summary_dict()
    summary["schema"] = "body_part_ethogram/v1"
    summary["provenance"] = {
        "annolid_version": str(get_version() or "0.0.0"),
        "generated_at_utc": _utc_now_iso(),
        "camera_view": camera_view,
        "contact_margin_px": float(contact_margin_px),
    }

    exported_files: Dict[str, str] = {}
    if output_prefix is not None:
        paths = export_body_part_ethogram_csvs(report, Path(output_prefix))
        exported_files.update({k: str(v) for k, v in paths.items()})
    if events_csv is not None:
        ev_path = export_body_part_ethogram_events_csv(report, Path(events_csv))
        exported_files["events_csv"] = str(ev_path)
    if exported_files:
        summary["artifacts"] = exported_files
    if out_json is not None:
        atomic_write_json(Path(out_json), summary, indent=2, ensure_ascii=False)

    if json_stdout:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    else:
        print(
            f"Frames analyzed: {summary['frames_analyzed']} | "
            f"Instances: {', '.join(summary['instances']) or '(none)'} | "
            f"Contact bouts: {summary['total_bouts']}"
        )
        if summary["posture_frame_counts"]:
            print(f"Posture frames: {summary['posture_frame_counts']}")
        if summary["bout_counts_by_behavior"]:
            print(f"Contact bouts by behavior: {summary['bout_counts_by_behavior']}")
    return 0
