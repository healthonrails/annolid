from __future__ import annotations

import json
from pathlib import Path

from annolid.services.chat_tracking_stats import analyze_chat_tracking_stats_tool


def _write_stats(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_analyze_chat_tracking_stats_tool_returns_totals(tmp_path: Path) -> None:
    _write_stats(
        tmp_path / "video_a" / "video_a_tracking_stats.json",
        {
            "version": 4,
            "video_name": "video_a.mp4",
            "updated_at": "2026-03-25T00:00:00Z",
            "summary": {
                "manual_frames": 4,
                "manual_segments": [[0, 3]],
                "bad_shape_frames": 1,
                "bad_shape_failed_frames": 1,
                "abnormal_segment_events": 2,
            },
            "prediction_segments": [
                {"start_frame": 1, "end_frame": 3, "status": "halted"}
            ],
            "bad_shape_events": [
                {
                    "frame": 3,
                    "label": "mouse",
                    "reason": "polygon_conversion_failed",
                    "resolved": False,
                    "repair_source": "",
                    "timestamp": "2026-03-25T00:00:00Z",
                }
            ],
        },
    )

    payload = analyze_chat_tracking_stats_tool(
        root_dir=str(tmp_path),
        top_k=5,
        include_plots=False,
    )

    assert payload["ok"] is True
    assert int(payload["video_count"]) == 1
    assert int(payload["totals"]["manual_frames"]) == 4
    assert int(payload["totals"]["abnormal_segment_events"]) == 2
    assert int(payload["totals"]["bad_shape_events_unresolved"]) == 1
    assert len(payload["videos"]) == 1


def test_analyze_chat_tracking_stats_tool_applies_video_filter(tmp_path: Path) -> None:
    _write_stats(
        tmp_path / "video_a" / "video_a_tracking_stats.json",
        {"version": 4, "video_name": "video_a.mp4", "summary": {"manual_frames": 1}},
    )
    _write_stats(
        tmp_path / "video_b" / "video_b_tracking_stats.json",
        {"version": 4, "video_name": "video_b.mp4", "summary": {"manual_frames": 2}},
    )

    payload = analyze_chat_tracking_stats_tool(
        root_dir=str(tmp_path),
        video_id="video_b",
        include_plots=False,
    )

    assert payload["ok"] is True
    assert int(payload["video_count"]) == 1
    assert len(payload["videos"]) == 1
    assert payload["videos"][0]["video_id"] == "video_b"
