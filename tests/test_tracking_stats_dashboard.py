from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from annolid.postprocessing.tracking_stats_dashboard import (
    analyze_and_visualize_tracking_stats,
    discover_tracking_stats_files,
)


def _write_stats(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_discover_tracking_stats_files_recurses(tmp_path: Path) -> None:
    a = tmp_path / "a_tracking_stats.json"
    b = tmp_path / "nested" / "b_tracking_stats.json"
    _write_stats(a, {"version": 4, "summary": {}})
    _write_stats(b, {"version": 4, "summary": {}})

    found = discover_tracking_stats_files(tmp_path)
    assert found == [a, b]


def test_analyze_tracking_stats_generates_cross_video_csvs(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    out_dir = tmp_path / "dashboard"

    _write_stats(
        root / "video_a" / "video_a_tracking_stats.json",
        {
            "version": 4,
            "video_name": "video_a.mp4",
            "updated_at": "2026-03-25T00:00:00Z",
            "summary": {
                "manual_frames": 5,
                "manual_segments": [[0, 4]],
                "bad_shape_frames": 1,
                "bad_shape_failed_frames": 1,
                "abnormal_segment_events": 2,
            },
            "prediction_segments": [
                {"start_frame": 10, "end_frame": 20, "status": "halted"},
                {
                    "start_frame": 21,
                    "end_frame": 22,
                    "status": "skipped_completed",
                },
            ],
            "bad_shape_events": [
                {
                    "frame": 15,
                    "label": "mouse",
                    "reason": "polygon_conversion_failed",
                    "resolved": False,
                    "repair_source": "",
                    "timestamp": "2026-03-25T00:00:00Z",
                }
            ],
        },
    )
    _write_stats(
        root / "video_b" / "video_b_tracking_stats.json",
        {
            "version": 4,
            "video_name": "video_b.mp4",
            "updated_at": "2026-03-25T00:05:00Z",
            "summary": {
                "manual_frames": 2,
                "manual_segments": [[11, 12]],
                "bad_shape_frames": 1,
                "bad_shape_failed_frames": 0,
                "abnormal_segment_events": 1,
            },
            "prediction_segments": [
                {"start_frame": 30, "end_frame": 31, "status": "halted"}
            ],
            "bad_shape_events": [
                {
                    "frame": 30,
                    "label": "rat",
                    "reason": "frame_sized_artifact_rejected_no_fallback",
                    "resolved": True,
                    "repair_source": "recent_mask",
                    "timestamp": "2026-03-25T00:06:00Z",
                }
            ],
        },
    )

    artifacts = analyze_and_visualize_tracking_stats(root, out_dir)
    assert artifacts.output_dir == out_dir.resolve()
    assert artifacts.overview_csv.exists()
    assert artifacts.abnormal_segments_csv.exists()
    assert artifacts.bad_shape_events_csv.exists()

    overview = pd.read_csv(artifacts.overview_csv)
    assert set(overview["video_id"].tolist()) == {"video_a", "video_b"}
    row_a = overview.loc[overview["video_id"] == "video_a"].iloc[0]
    assert int(row_a["manual_frames"]) == 5
    assert int(row_a["abnormal_segment_events"]) == 2
    assert int(row_a["bad_shape_events_unresolved"]) == 1

    abnormal = pd.read_csv(artifacts.abnormal_segments_csv)
    assert len(abnormal) == 3
    assert set(abnormal["status"].tolist()) == {"halted", "skipped_completed"}

    bad_shape = pd.read_csv(artifacts.bad_shape_events_csv)
    assert len(bad_shape) == 2
    assert set(bad_shape["resolved"].tolist()) == {False, True}


def test_analyze_tracking_stats_handles_missing_sections(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _write_stats(
        root / "x" / "x_tracking_stats.json",
        {
            "version": 4,
            "video_name": "x.mp4",
            "summary": {"manual_frames": 1},
        },
    )

    artifacts = analyze_and_visualize_tracking_stats(root)
    overview = pd.read_csv(artifacts.overview_csv)
    assert len(overview) == 1
    row = overview.iloc[0]
    assert int(row["manual_frames"]) == 1
    assert int(row["bad_shape_events_total"]) == 0
    abnormal = pd.read_csv(artifacts.abnormal_segments_csv)
    bad_shape = pd.read_csv(artifacts.bad_shape_events_csv)
    assert abnormal.empty
    assert bad_shape.empty
