from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from annolid.postprocessing.tracking_stats_dashboard import (
    analyze_and_visualize_tracking_stats,
)


def _normalize_root_dir(
    root_dir: str,
    default_root_dir: Optional[Path],
) -> Path:
    root_text = str(root_dir or "").strip()
    if root_text:
        return Path(root_text).expanduser().resolve()
    if default_root_dir is not None:
        return Path(default_root_dir).expanduser().resolve()
    return Path.cwd().resolve()


def _build_artifact_payload(artifacts) -> Dict[str, str]:
    return {
        "overview_csv": str(artifacts.overview_csv),
        "abnormal_segments_csv": str(artifacts.abnormal_segments_csv),
        "bad_shape_events_csv": str(artifacts.bad_shape_events_csv),
        "manual_badshape_plot": str(artifacts.manual_badshape_plot or ""),
        "abnormal_segments_plot": str(artifacts.abnormal_segments_plot or ""),
        "unresolved_bad_shapes_plot": str(artifacts.unresolved_bad_shapes_plot or ""),
    }


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _summarize_videos(dataframe: pd.DataFrame, top_k: int) -> list[dict[str, Any]]:
    if dataframe.empty:
        return []
    ranked = dataframe.sort_values(
        by=["bad_shape_events_unresolved", "abnormal_segment_events", "manual_frames"],
        ascending=[False, False, False],
    ).head(max(1, min(_safe_int(top_k, default=10), 100)))
    videos: list[dict[str, Any]] = []
    for _, row in ranked.iterrows():
        videos.append(
            {
                "video_id": str(row.get("video_id", "")),
                "manual_frames": _safe_int(row.get("manual_frames", 0)),
                "bad_shape_frames": _safe_int(row.get("bad_shape_frames", 0)),
                "bad_shape_failed_frames": _safe_int(
                    row.get("bad_shape_failed_frames", 0)
                ),
                "abnormal_segment_events": _safe_int(
                    row.get("abnormal_segment_events", 0)
                ),
                "bad_shape_events_unresolved": _safe_int(
                    row.get("bad_shape_events_unresolved", 0)
                ),
                "stats_path": str(row.get("stats_path", "")),
                "updated_at": str(row.get("updated_at", "")),
            }
        )
    return videos


def analyze_tracking_stats_tool(
    *,
    root_dir: str = "",
    output_dir: str = "",
    video_id: str = "",
    top_k: int = 10,
    include_plots: bool = True,
    default_root_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    resolved_root = _normalize_root_dir(root_dir, default_root_dir)

    if not resolved_root.exists():
        return {
            "ok": False,
            "error": f"Root directory does not exist: {resolved_root}",
            "root_dir": str(resolved_root),
        }

    output_path: Optional[Path] = None
    output_text = str(output_dir or "").strip()
    if output_text:
        output_path = Path(output_text).expanduser().resolve()

    artifacts = analyze_and_visualize_tracking_stats(
        root_dir=resolved_root,
        output_dir=output_path,
        include_plots=bool(include_plots),
    )

    overview_df = pd.read_csv(artifacts.overview_csv)
    filtered_df = overview_df
    video_filter = str(video_id or "").strip()
    if video_filter and not overview_df.empty and "video_id" in overview_df.columns:
        lowered = video_filter.lower()
        filtered_df = overview_df[
            overview_df["video_id"].astype(str).str.lower().str.contains(lowered)
        ]

    if filtered_df.empty:
        return {
            "ok": True,
            "root_dir": str(resolved_root),
            "output_dir": str(artifacts.output_dir),
            "video_filter": video_filter,
            "video_count": 0,
            "summary": (
                "No tracking stats matched the request."
                if video_filter
                else "No tracking stats files were found."
            ),
            "totals": {
                "manual_frames": 0,
                "bad_shape_frames": 0,
                "bad_shape_failed_frames": 0,
                "abnormal_segment_events": 0,
                "bad_shape_events_unresolved": 0,
            },
            "videos": [],
            "artifacts": _build_artifact_payload(artifacts),
        }

    def _col_sum(name: str) -> int:
        if name not in filtered_df.columns:
            return 0
        return int(pd.to_numeric(filtered_df[name], errors="coerce").fillna(0).sum())

    per_video = _summarize_videos(filtered_df, top_k)
    top_video = per_video[0] if per_video else {}
    summary_text = (
        f"Analyzed {int(len(filtered_df))} video(s) under {resolved_root}. "
        f"Manual frames: {_col_sum('manual_frames')}, "
        f"abnormal segment events: {_col_sum('abnormal_segment_events')}, "
        f"unresolved bad-shape events: {_col_sum('bad_shape_events_unresolved')}."
    )
    if top_video:
        summary_text += (
            f" Top-ranked video: {top_video['video_id']} "
            f"(manual={top_video['manual_frames']}, "
            f"abnormal={top_video['abnormal_segment_events']}, "
            f"unresolved_bad_shapes={top_video['bad_shape_events_unresolved']})."
        )

    return {
        "ok": True,
        "root_dir": str(resolved_root),
        "output_dir": str(artifacts.output_dir),
        "video_filter": video_filter,
        "video_count": int(len(filtered_df)),
        "summary": summary_text,
        "totals": {
            "manual_frames": _col_sum("manual_frames"),
            "bad_shape_frames": _col_sum("bad_shape_frames"),
            "bad_shape_failed_frames": _col_sum("bad_shape_failed_frames"),
            "abnormal_segment_events": _col_sum("abnormal_segment_events"),
            "bad_shape_events_unresolved": _col_sum("bad_shape_events_unresolved"),
        },
        "videos": per_video,
        "artifacts": _build_artifact_payload(artifacts),
    }
