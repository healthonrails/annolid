from __future__ import annotations

import json
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


def test_label_behavior_segments_tool_forwards_behavior_context(monkeypatch) -> None:
    captured: dict[str, object] = {}

    payload = workflow.label_behavior_segments_tool(
        path="mouse.mp4",
        behavior_labels=["aggression_bout"],
        use_defined_behavior_list=False,
        segment_mode="uniform",
        segment_frames=30,
        segment_seconds=1.0,
        sample_frames_per_segment=3,
        max_segments=10,
        subject="Mouse-1",
        overwrite_existing=False,
        llm_profile="",
        llm_provider="",
        llm_model="",
        video_description="Two mice in arena.",
        instance_count=2,
        experiment_context="Resident intruder protocol.",
        behavior_definitions="Aggression bout includes slap in face and run away.",
        focus_points="Count bouts and fight initiation.",
        resolve_video_path=lambda _path: Path("/tmp/mouse.mp4"),
        invoke_label_behavior=lambda *_args: (
            captured.update(
                {
                    "video_description": _args[13],
                    "instance_count": _args[14],
                    "experiment_context": _args[15],
                    "behavior_definitions": _args[16],
                    "focus_points": _args[17],
                }
            )
            or True
        ),
        get_action_result=lambda _name: {},
    )

    assert payload["ok"] is True
    assert payload["video_description"] == "Two mice in arena."
    assert payload["instance_count"] == 2
    assert payload["experiment_context"] == "Resident intruder protocol."
    assert "slap in face" in payload["behavior_definitions"]
    assert "fight initiation" in payload["focus_points"]
    assert captured["video_description"] == "Two mice in arena."
    assert captured["instance_count"] == 2


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


def test_process_video_behaviors_tool_runs_tracking_and_labeling(monkeypatch) -> None:
    calls: dict[str, int] = {"track": 0, "label": 0}

    payload = workflow.process_video_behaviors_tool(
        path="mouse.mp4",
        text_prompt="mouse",
        mode="track",
        use_countgd=False,
        model_name="Cutie",
        to_frame=120,
        behavior_labels=["run_away", "fight_initiation"],
        use_defined_behavior_list=False,
        segment_mode="uniform",
        segment_frames=30,
        segment_seconds=None,
        sample_frames_per_segment=3,
        max_segments=10,
        subject="Mouse-1",
        overwrite_existing=False,
        llm_profile="",
        llm_provider="",
        llm_model="",
        run_tracking=True,
        run_behavior_labeling=True,
        resolve_video_path=lambda _path: Path("/tmp/mouse.mp4"),
        invoke_segment_track=lambda *_args: (
            calls.__setitem__("track", calls["track"] + 1) or True
        ),
        invoke_label_behavior=lambda *_args: (
            calls.__setitem__("label", calls["label"] + 1) or True
        ),
        get_action_result=lambda _name: {},
    )

    assert payload["ok"] is True
    assert payload["tracking_executed"] is True
    assert payload["behavior_labeling_executed"] is True
    assert calls["track"] == 1
    assert calls["label"] == 1
    assert payload["stages"]["tracking"]["ok"] is True
    assert payload["stages"]["behavior_labeling"]["ok"] is True


def test_process_video_behaviors_tool_returns_stage_failure(monkeypatch) -> None:
    payload = workflow.process_video_behaviors_tool(
        path="mouse.mp4",
        text_prompt="mouse",
        mode="track",
        use_countgd=False,
        model_name="Cutie",
        to_frame=120,
        behavior_labels=["run_away"],
        use_defined_behavior_list=True,
        segment_mode="timeline",
        segment_frames=60,
        segment_seconds=None,
        sample_frames_per_segment=3,
        max_segments=10,
        subject="Agent",
        overwrite_existing=False,
        llm_profile="",
        llm_provider="",
        llm_model="",
        run_tracking=True,
        run_behavior_labeling=True,
        resolve_video_path=lambda _path: Path("/tmp/mouse.mp4"),
        invoke_segment_track=lambda *_args: False,
        invoke_label_behavior=lambda *_args: True,
        get_action_result=lambda _name: {},
    )

    assert payload["ok"] is False
    assert payload["stage"] == "tracking"
    assert payload["stages"]["tracking"]["ok"] is False


def test_score_aggression_bouts_tool_generates_manifest(tmp_path: Path) -> None:
    artifacts = tmp_path / "agent.ndjson"
    artifacts.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "artifact_id": "a1",
                        "frame_index": 10,
                        "track_id": "mouse_1",
                        "label": "slap in the face",
                        "meta": {"count": 2},
                    }
                ),
                json.dumps(
                    {
                        "artifact_id": "a2",
                        "frame_index": 15,
                        "track_id": "mouse_2",
                        "label": "run away",
                        "meta": {"count": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    payload = workflow.score_aggression_bouts_tool(
        path="mouse.mp4",
        artifacts_ndjson=str(artifacts),
        run_id="run_workflow_001",
        results_dir=str(tmp_path / "results"),
        context_prompt="score aggression bouts",
        assay="aggression",
        bout_frame_gap=20,
        resolve_video_path=lambda _path: tmp_path / "mouse.mp4",
    )

    assert payload["ok"] is True
    assert payload["run_id"] == "run_workflow_001"
    assert payload["task_plan_assay"] == "aggression"
    assert len(payload["bout_counts"]) == 1
    assert Path(str(payload["manifest_path"])).exists()
    assert payload["artifacts_source"] == "explicit"
    assert payload["artifacts_ndjson"] == str(artifacts.resolve())


def test_score_aggression_bouts_tool_resolves_default_sidecar_artifacts(
    tmp_path: Path,
) -> None:
    sidecar_dir = tmp_path / "mouse"
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    artifacts = sidecar_dir / "agent.ndjson"
    artifacts.write_text(
        json.dumps(
            {
                "artifact_id": "a1",
                "frame_index": 10,
                "track_id": "mouse_1",
                "label": "fight initiation",
                "meta": {"count": 1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = workflow.score_aggression_bouts_tool(
        path="mouse.mp4",
        artifacts_ndjson="",
        run_id="run_workflow_default_sidecar",
        results_dir=str(tmp_path / "results"),
        resolve_video_path=lambda _path: tmp_path / "mouse.mp4",
    )

    assert payload["ok"] is True
    assert payload["artifacts_source"] == "default"
    assert payload["artifacts_ndjson"] == str(artifacts.resolve())
    assert len(payload["bout_counts"]) == 1


def test_score_aggression_bouts_tool_errors_on_missing_explicit_artifacts(
    tmp_path: Path,
) -> None:
    payload = workflow.score_aggression_bouts_tool(
        path="mouse.mp4",
        artifacts_ndjson="missing.ndjson",
        run_id="run_workflow_missing_sidecar",
        results_dir=str(tmp_path / "results"),
        resolve_video_path=lambda _path: tmp_path / "mouse.mp4",
    )

    assert payload["ok"] is False
    assert "artifacts_ndjson was provided" in str(payload.get("error"))
    assert payload["artifacts_ndjson_input"] == "missing.ndjson"
    assert payload["searched_paths"]
