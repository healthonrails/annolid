from __future__ import annotations

from pathlib import Path

import annolid.core.agent.gui_backend.tool_handlers_video_workflow as workflow


def test_label_behavior_segments_tool_adapts_uniform_short_video(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(workflow, "_video_total_frames", lambda _path: 43)

    payload = workflow.label_behavior_segments_tool(
        path="mouse.mp4",
        behavior_labels=["rearing", "walking"],
        use_defined_behavior_list=True,
        segment_mode="uniform",
        segment_frames=60,
        segment_seconds=None,
        sample_frames_per_segment=3,
        max_segments=120,
        subject="Agent",
        overwrite_existing=False,
        llm_profile="",
        llm_provider="",
        llm_model="",
        resolve_video_path=lambda _path: Path("/tmp/mouse.mp4"),
        invoke_label_behavior=lambda vpath,
        labels,
        use_defined,
        mode,
        frames,
        _seconds,
        _samples,
        max_seg,
        *_: (
            captured.update(
                {
                    "path": vpath,
                    "labels": labels,
                    "use_defined": bool(use_defined),
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
    assert captured["use_defined"] is True
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
        use_defined_behavior_list=True,
        segment_mode="uniform",
        segment_frames=10,
        segment_seconds=None,
        sample_frames_per_segment=3,
        max_segments=1,
        subject="Agent",
        overwrite_existing=False,
        llm_profile="",
        llm_provider="",
        llm_model="",
        resolve_video_path=lambda _path: Path("/tmp/mouse.mp4"),
        invoke_label_behavior=lambda _vpath,
        _labels,
        _use_defined,
        _mode,
        _frames,
        _seconds,
        _samples,
        max_seg,
        *_: (captured.update({"max_segments": max_seg}) or True),
        get_action_result=lambda _name: {},
    )

    assert payload["ok"] is True
    assert int(captured["max_segments"]) > 1


def test_label_behavior_segments_tool_uses_seconds_from_fps(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(workflow, "_video_total_frames", lambda _path: 300)
    monkeypatch.setattr(workflow, "_video_fps", lambda _path: 29.97)

    payload = workflow.label_behavior_segments_tool(
        path="mouse.mp4",
        behavior_labels=["rearing", "walking"],
        use_defined_behavior_list=False,
        segment_mode="uniform",
        segment_frames=60,
        segment_seconds=1.0,
        sample_frames_per_segment=4,
        max_segments=120,
        subject="Agent",
        overwrite_existing=False,
        llm_profile="",
        llm_provider="",
        llm_model="",
        resolve_video_path=lambda _path: Path("/tmp/mouse.mp4"),
        invoke_label_behavior=lambda _vpath,
        _labels,
        use_defined,
        _mode,
        frames,
        seconds,
        samples,
        *_: (
            captured.update(
                {
                    "use_defined": bool(use_defined),
                    "segment_frames": frames,
                    "segment_seconds": seconds,
                    "samples": samples,
                }
            )
            or True
        ),
        get_action_result=lambda _name: {},
    )

    assert payload["ok"] is True
    assert captured["use_defined"] is False
    assert int(captured["segment_frames"]) == 30
    assert float(captured["segment_seconds"]) == 1.0
    assert int(captured["samples"]) == 4


def test_behavior_catalog_tool_returns_widget_result(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _invoke(payload_json: str) -> bool:
        captured["payload_json"] = payload_json
        return True

    def _get_result(name: str) -> dict[str, object]:
        assert name == "behavior_catalog"
        return {
            "ok": True,
            "action": "create",
            "message": "Created behavior 'grooming'.",
            "behavior": {"code": "grooming", "name": "Grooming"},
            "saved": True,
            "path": "/tmp/project.annolid.json",
        }

    payload = workflow.behavior_catalog_tool(
        action="create",
        code="grooming",
        name="Grooming",
        description="",
        category_id="",
        modifier_ids=["core", ""],
        key_binding="g",
        is_state=True,
        exclusive_with=["walking"],
        save=True,
        invoke_behavior_catalog=_invoke,
        get_action_result=_get_result,
    )

    assert payload["ok"] is True
    assert payload["action"] == "create"
    assert payload["behavior"]["code"] == "grooming"
    assert payload["saved"] is True
    assert captured["payload_json"]
