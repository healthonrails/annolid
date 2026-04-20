from __future__ import annotations

from pathlib import Path

from annolid.services.chat_video import (
    label_chat_behavior_segments_tool,
    open_chat_video_tool,
    process_chat_video_behaviors_tool,
    resolve_chat_video_path_for_gui_tool,
    score_chat_aggression_bouts_tool,
    segment_track_chat_video_tool,
)


def test_chat_video_wrappers(monkeypatch) -> None:
    import annolid.services.chat_video as video_mod

    monkeypatch.setattr(
        video_mod,
        "gui_open_video_tool",
        lambda path, **kwargs: {"ok": True, "kind": "open_video", "path": path},
    )
    monkeypatch.setattr(
        video_mod,
        "gui_resolve_video_path_for_gui_tool",
        lambda raw_path, **kwargs: Path("/tmp") / raw_path,
    )
    monkeypatch.setattr(
        video_mod,
        "gui_segment_track_video_tool",
        lambda **kwargs: {"ok": True, "kind": "segment_track", **kwargs},
    )
    monkeypatch.setattr(
        video_mod,
        "gui_label_behavior_segments_tool",
        lambda **kwargs: {"ok": True, "kind": "label_behavior", **kwargs},
    )
    monkeypatch.setattr(
        video_mod,
        "gui_process_video_behaviors_tool",
        lambda **kwargs: {"ok": True, "kind": "process_video_behaviors", **kwargs},
    )
    monkeypatch.setattr(
        video_mod,
        "gui_score_aggression_bouts_tool",
        lambda **kwargs: {"ok": True, "kind": "score_aggression_bouts", **kwargs},
    )

    assert open_chat_video_tool("clip.mp4") == {
        "ok": True,
        "kind": "open_video",
        "path": "clip.mp4",
    }
    assert resolve_chat_video_path_for_gui_tool("clip.mp4") == Path("/tmp/clip.mp4")
    assert (
        segment_track_chat_video_tool(path="clip.mp4", text_prompt="mouse")["kind"]
        == "segment_track"
    )
    assert (
        label_chat_behavior_segments_tool(path="clip.mp4", behavior_labels=["run"])[
            "kind"
        ]
        == "label_behavior"
    )
    assert (
        process_chat_video_behaviors_tool(
            path="clip.mp4",
            text_prompt="mouse",
            run_tracking=True,
            run_behavior_labeling=True,
        )["kind"]
        == "process_video_behaviors"
    )
    assert (
        score_chat_aggression_bouts_tool(
            path="clip.mp4",
            artifacts_ndjson="/tmp/artifacts.ndjson",
        )["kind"]
        == "score_aggression_bouts"
    )
