from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

from annolid.utils.logger import logger

try:
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except Exception:  # pragma: no cover - optional visualization dependency
    MATPLOTLIB_AVAILABLE = False


TRACKING_STATS_SUFFIX = "_tracking_stats.json"
OVERVIEW_COLUMNS = [
    "video_id",
    "stats_path",
    "stats_version",
    "manual_frames",
    "manual_segments",
    "bad_shape_frames",
    "bad_shape_failed_frames",
    "abnormal_segment_events",
    "bad_shape_events_total",
    "bad_shape_events_resolved",
    "bad_shape_events_unresolved",
    "abnormal_segment_status_counts",
    "updated_at",
]
ABNORMAL_SEGMENT_COLUMNS = [
    "video_id",
    "stats_path",
    "start_frame",
    "end_frame",
    "status",
    "timestamp",
]
BAD_SHAPE_EVENT_COLUMNS = [
    "video_id",
    "stats_path",
    "frame",
    "label",
    "reason",
    "resolved",
    "repair_source",
    "timestamp",
]


@dataclass(frozen=True)
class TrackingStatsArtifacts:
    output_dir: Path
    overview_csv: Path
    abnormal_segments_csv: Path
    bad_shape_events_csv: Path
    manual_badshape_plot: Optional[Path] = None
    abnormal_segments_plot: Optional[Path] = None
    unresolved_bad_shapes_plot: Optional[Path] = None


def discover_tracking_stats_files(root_dir: Path) -> List[Path]:
    root = Path(root_dir).expanduser().resolve()
    if not root.exists():
        return []
    return sorted(
        [path for path in root.rglob(f"*{TRACKING_STATS_SUFFIX}") if path.is_file()]
    )


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _load_tracking_stats(path: Path) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
    except Exception as exc:
        logger.warning("Skipping invalid tracking stats %s: %s", path, exc)
        return None
    if not isinstance(payload, dict):
        logger.warning("Skipping non-dict tracking stats payload: %s", path)
        return None
    return payload


def _infer_video_id(path: Path, payload: Dict[str, Any]) -> str:
    configured = str(payload.get("video_name", "") or "").strip()
    if configured:
        return Path(configured).stem or configured
    stem = path.stem
    if stem.endswith("_tracking_stats"):
        stem = stem[: -len("_tracking_stats")]
    return stem


def _extract_overview_row(path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}

    bad_shape_events = payload.get("bad_shape_events", [])
    if not isinstance(bad_shape_events, list):
        bad_shape_events = []

    prediction_segments = payload.get("prediction_segments", [])
    if not isinstance(prediction_segments, list):
        prediction_segments = []

    unresolved_bad_shape_events = sum(
        1
        for event in bad_shape_events
        if isinstance(event, dict) and not bool(event.get("resolved", False))
    )
    resolved_bad_shape_events = max(
        0, len(bad_shape_events) - unresolved_bad_shape_events
    )

    abnormal_status_counts: Dict[str, int] = {}
    for seg in prediction_segments:
        if not isinstance(seg, dict):
            continue
        status = str(seg.get("status", "unknown") or "unknown")
        abnormal_status_counts[status] = abnormal_status_counts.get(status, 0) + 1

    return {
        "video_id": _infer_video_id(path, payload),
        "stats_path": str(path),
        "stats_version": _safe_int(payload.get("version"), default=0),
        "manual_frames": _safe_int(summary.get("manual_frames"), default=0),
        "manual_segments": _safe_int(
            len(summary.get("manual_segments", []))
            if isinstance(summary.get("manual_segments"), list)
            else 0
        ),
        "bad_shape_frames": _safe_int(summary.get("bad_shape_frames"), default=0),
        "bad_shape_failed_frames": _safe_int(
            summary.get("bad_shape_failed_frames"), default=0
        ),
        "abnormal_segment_events": _safe_int(
            summary.get("abnormal_segment_events"), default=0
        ),
        "bad_shape_events_total": _safe_int(len(bad_shape_events), default=0),
        "bad_shape_events_resolved": _safe_int(resolved_bad_shape_events, default=0),
        "bad_shape_events_unresolved": _safe_int(
            unresolved_bad_shape_events, default=0
        ),
        "abnormal_segment_status_counts": json.dumps(
            abnormal_status_counts, sort_keys=True
        ),
        "updated_at": str(payload.get("updated_at", "")),
    }


def _extract_abnormal_segments_rows(
    path: Path, payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    video_id = _infer_video_id(path, payload)
    prediction_segments = payload.get("prediction_segments", [])
    if not isinstance(prediction_segments, list):
        return []
    rows: List[Dict[str, Any]] = []
    for seg in prediction_segments:
        if not isinstance(seg, dict):
            continue
        rows.append(
            {
                "video_id": video_id,
                "stats_path": str(path),
                "start_frame": _safe_int(seg.get("start_frame"), default=-1),
                "end_frame": _safe_int(seg.get("end_frame"), default=-1),
                "status": str(seg.get("status", "")),
                "timestamp": str(seg.get("timestamp", "")),
            }
        )
    return rows


def _extract_bad_shape_event_rows(
    path: Path, payload: Dict[str, Any]
) -> List[Dict[str, Any]]:
    video_id = _infer_video_id(path, payload)
    events = payload.get("bad_shape_events", [])
    if not isinstance(events, list):
        return []
    rows: List[Dict[str, Any]] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        rows.append(
            {
                "video_id": video_id,
                "stats_path": str(path),
                "frame": _safe_int(event.get("frame"), default=-1),
                "label": str(event.get("label", "")),
                "reason": str(event.get("reason", "")),
                "resolved": bool(event.get("resolved", False)),
                "repair_source": str(event.get("repair_source", "")),
                "timestamp": str(event.get("timestamp", "")),
            }
        )
    return rows


def _render_plots(
    overview_df: pd.DataFrame, output_dir: Path
) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if not MATPLOTLIB_AVAILABLE:
        logger.warning("matplotlib is unavailable; skipping tracking stats plots.")
        return None, None, None
    if overview_df.empty:
        return None, None, None

    manual_badshape_plot = output_dir / "tracking_stats_manual_vs_bad_shapes.png"
    abnormal_segments_plot = output_dir / "tracking_stats_abnormal_segments.png"
    unresolved_bad_shapes_plot = output_dir / "tracking_stats_unresolved_bad_shapes.png"

    sorted_df = overview_df.sort_values("video_id")

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(sorted_df["video_id"], sorted_df["manual_frames"], label="manual_frames")
    ax.bar(
        sorted_df["video_id"],
        sorted_df["bad_shape_frames"],
        bottom=sorted_df["manual_frames"],
        label="bad_shape_frames",
    )
    ax.set_title("Manual/Bad-Shape Frames by Video")
    ax.set_ylabel("Frame Count")
    ax.tick_params(axis="x", rotation=60)
    ax.legend()
    fig.tight_layout()
    fig.savefig(manual_badshape_plot, dpi=180)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(11, 5))
    ax2.bar(sorted_df["video_id"], sorted_df["abnormal_segment_events"])
    ax2.set_title("Abnormal Segment Events by Video")
    ax2.set_ylabel("Event Count")
    ax2.tick_params(axis="x", rotation=60)
    fig2.tight_layout()
    fig2.savefig(abnormal_segments_plot, dpi=180)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.scatter(
        sorted_df["manual_frames"],
        sorted_df["bad_shape_events_unresolved"],
        alpha=0.8,
    )
    ax3.set_xlabel("Manual Frames")
    ax3.set_ylabel("Unresolved Bad-Shape Events")
    ax3.set_title("Manual Frames vs Unresolved Bad-Shape Events")
    for _, row in sorted_df.iterrows():
        ax3.annotate(
            str(row["video_id"]),
            (
                float(row["manual_frames"]),
                float(row["bad_shape_events_unresolved"]),
            ),
            fontsize=8,
            alpha=0.8,
        )
    fig3.tight_layout()
    fig3.savefig(unresolved_bad_shapes_plot, dpi=180)
    plt.close(fig3)

    return manual_badshape_plot, abnormal_segments_plot, unresolved_bad_shapes_plot


def analyze_and_visualize_tracking_stats(
    root_dir: Path,
    output_dir: Optional[Path] = None,
    include_plots: bool = True,
) -> TrackingStatsArtifacts:
    root = Path(root_dir).expanduser().resolve()
    stats_files = discover_tracking_stats_files(root)
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else root / "tracking_stats_dashboard"
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    overview_rows: List[Dict[str, Any]] = []
    abnormal_segments_rows: List[Dict[str, Any]] = []
    bad_shape_event_rows: List[Dict[str, Any]] = []

    for stats_file in stats_files:
        payload = _load_tracking_stats(stats_file)
        if payload is None:
            continue
        overview_rows.append(_extract_overview_row(stats_file, payload))
        abnormal_segments_rows.extend(
            _extract_abnormal_segments_rows(stats_file, payload)
        )
        bad_shape_event_rows.extend(_extract_bad_shape_event_rows(stats_file, payload))

    overview_df = pd.DataFrame(overview_rows, columns=OVERVIEW_COLUMNS)
    if not overview_df.empty:
        overview_df = overview_df.sort_values(
            by=["bad_shape_events_unresolved", "abnormal_segment_events", "video_id"],
            ascending=[False, False, True],
        )

    abnormal_df = pd.DataFrame(abnormal_segments_rows, columns=ABNORMAL_SEGMENT_COLUMNS)
    if not abnormal_df.empty:
        abnormal_df = abnormal_df.sort_values(
            by=["video_id", "start_frame", "end_frame"]
        )

    bad_shape_df = pd.DataFrame(bad_shape_event_rows, columns=BAD_SHAPE_EVENT_COLUMNS)
    if not bad_shape_df.empty:
        bad_shape_df = bad_shape_df.sort_values(by=["video_id", "frame", "label"])

    overview_csv = resolved_output_dir / "tracking_stats_overview.csv"
    abnormal_csv = resolved_output_dir / "tracking_stats_abnormal_segments.csv"
    bad_shape_csv = resolved_output_dir / "tracking_stats_bad_shape_events.csv"

    overview_df.to_csv(overview_csv, index=False)
    abnormal_df.to_csv(abnormal_csv, index=False)
    bad_shape_df.to_csv(bad_shape_csv, index=False)

    manual_badshape_plot: Optional[Path] = None
    abnormal_segments_plot: Optional[Path] = None
    unresolved_bad_shapes_plot: Optional[Path] = None
    if include_plots:
        plots = _render_plots(overview_df, resolved_output_dir)
        manual_badshape_plot, abnormal_segments_plot, unresolved_bad_shapes_plot = plots

    logger.info(
        "Tracking stats dashboard generated for %d video(s) at %s.",
        len(overview_df),
        resolved_output_dir,
    )
    return TrackingStatsArtifacts(
        output_dir=resolved_output_dir,
        overview_csv=overview_csv,
        abnormal_segments_csv=abnormal_csv,
        bad_shape_events_csv=bad_shape_csv,
        manual_badshape_plot=manual_badshape_plot,
        abnormal_segments_plot=abnormal_segments_plot,
        unresolved_bad_shapes_plot=unresolved_bad_shapes_plot,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate and visualize Annolid tracking stats across multiple videos."
        )
    )
    parser.add_argument(
        "root_dir",
        type=str,
        help="Root directory to recursively scan for *_tracking_stats.json files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to write CSV/plot artifacts. Defaults to <root_dir>/tracking_stats_dashboard.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation and only write CSV artifacts.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    artifacts = analyze_and_visualize_tracking_stats(
        root_dir=Path(args.root_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        include_plots=not bool(args.no_plots),
    )
    logger.info("Overview CSV: %s", artifacts.overview_csv)
    logger.info("Abnormal segments CSV: %s", artifacts.abnormal_segments_csv)
    logger.info("Bad shape events CSV: %s", artifacts.bad_shape_events_csv)
    if artifacts.manual_badshape_plot:
        logger.info("Manual/bad-shape plot: %s", artifacts.manual_badshape_plot)
    if artifacts.abnormal_segments_plot:
        logger.info("Abnormal segments plot: %s", artifacts.abnormal_segments_plot)
    if artifacts.unresolved_bad_shapes_plot:
        logger.info(
            "Unresolved bad-shape plot: %s", artifacts.unresolved_bad_shapes_plot
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
