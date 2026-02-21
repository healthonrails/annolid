from __future__ import annotations

from pathlib import Path

import annolid.core.agent.gui_backend.tool_handlers_video_workflow as workflow


def test_label_behavior_segments_tool_adapts_uniform_short_video(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(workflow, "_video_total_frames", lambda _path: 43)

    payload = workflow.label_behavior_segments_tool(
        path="mouse.mp4",
        behavior_labels=["rearing", "walking"],
        segment_mode="uniform",
        segment_frames=60,
        max_segments=120,
        subject="Agent",
        overwrite_existing=False,
        llm_profile="",
        llm_provider="",
        llm_model="",
        resolve_video_path=lambda _path: Path("/tmp/mouse.mp4"),
        invoke_label_behavior=lambda vpath, labels, mode, frames, max_seg, *_: (
            captured.update(
                {
                    "path": vpath,
                    "labels": labels,
                    "mode": mode,
                    "frames": frames,
                    "max_segments": max_seg,
                }
            )
            or True
        ),
        get_action_result=lambda _name: {},
    )

    assert payload["ok"] is True
    assert payload["queued"] is True
    assert captured["mode"] == "uniform"
    assert int(captured["frames"]) < 60
    assert int(captured["frames"]) == 10


def test_label_behavior_segments_tool_recovers_from_single_segment_cap(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(workflow, "_video_total_frames", lambda _path: 43)

    payload = workflow.label_behavior_segments_tool(
        path="mouse.mp4",
        behavior_labels=["rearing", "walking"],
        segment_mode="uniform",
        segment_frames=10,
        max_segments=1,
        subject="Agent",
        overwrite_existing=False,
        llm_profile="",
        llm_provider="",
        llm_model="",
        resolve_video_path=lambda _path: Path("/tmp/mouse.mp4"),
        invoke_label_behavior=lambda _vpath, _labels, _mode, _frames, max_seg, *_: (
            captured.update({"max_segments": max_seg}) or True
        ),
        get_action_result=lambda _name: {},
    )

    assert payload["ok"] is True
    assert int(captured["max_segments"]) > 1
