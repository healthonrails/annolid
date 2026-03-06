from __future__ import annotations

from types import SimpleNamespace

import pytest

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
