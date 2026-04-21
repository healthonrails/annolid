from __future__ import annotations

from types import SimpleNamespace

import pytest

from annolid.services.inference import (
    initialize_behavior_video_agent,
    run_behavior_video_agent,
)
from annolid.services.agent import AgentPipelineRequest, run_agent_pipeline


def test_run_agent_pipeline_delegates_to_core_service(monkeypatch) -> None:
    import annolid.core.agent.service as core_service

    captured = {}

    def _fake_run_agent_to_results(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(results_dir="out", records_written=1)

    monkeypatch.setattr(
        core_service, "run_agent_to_results", _fake_run_agent_to_results
    )

    result = run_agent_pipeline(
        AgentPipelineRequest(
            video_path="video.mp4",
            behavior_spec_path="project.annolid.json",
            results_dir="results",
            stride=2,
        )
    )

    assert result.records_written == 1
    assert captured["video_path"] == "video.mp4"
    assert captured["behavior_spec_path"] == "project.annolid.json"
    assert captured["results_dir"] == "results"
    assert captured["config"].stride == 2
    assert captured["reuse_cache"] is True


def test_run_agent_pipeline_requires_dino_weights() -> None:
    with pytest.raises(ValueError, match="weights path"):
        run_agent_pipeline(
            AgentPipelineRequest(
                video_path="video.mp4",
                vision_adapter="dino_kpseg",
            )
        )


def test_initialize_behavior_video_agent_delegates_to_behavior_agent(
    monkeypatch,
) -> None:
    import annolid.agents.behavior_agent as behavior_agent_module

    expected = object()

    monkeypatch.setattr(
        behavior_agent_module,
        "initialize_agent",
        lambda *args, **kwargs: expected,
    )

    assert initialize_behavior_video_agent() is expected


def test_run_behavior_video_agent_uses_service_initializer_when_agent_missing(
    monkeypatch,
) -> None:
    import annolid.agents.behavior_agent as behavior_agent_module

    expected_agent = object()
    observed: dict[str, object] = {}

    monkeypatch.setattr(
        "annolid.services.inference.initialize_behavior_video_agent",
        lambda: expected_agent,
    )

    def _fake_process_video_with_agent(video_path, user_prompt, agent):
        observed["video_path"] = video_path
        observed["user_prompt"] = user_prompt
        observed["agent"] = agent
        return "ok"

    monkeypatch.setattr(
        behavior_agent_module,
        "process_video_with_agent",
        _fake_process_video_with_agent,
    )

    result = run_behavior_video_agent("video.mp4", "describe it")

    assert result == "ok"
    assert observed == {
        "video_path": "video.mp4",
        "user_prompt": "describe it",
        "agent": expected_agent,
    }
