"""Shared runtime helpers for behavior-agent execution surfaces.

This module centralizes default pipeline construction so CLI, GUI tools, and
future integrations reuse the same behavior-agent wiring.
"""

from __future__ import annotations

from pathlib import Path

from annolid.services.behavior_agent.defaults import (
    AggressionSubEventSegmenter,
    DeterministicAnalysisRunner,
    InMemoryMemoryStore,
    KeywordTaskInferencer,
    NDJSONPerceptionAdapter,
    PassThroughPerceptionAdapter,
)
from annolid.services.behavior_agent.artifact_store import BehaviorAgentArtifactStore
from annolid.services.behavior_agent.pipeline import (
    BehaviorAgentPipeline,
    BehaviorAgentPipelineResult,
)


def resolve_behavior_results_root(
    *,
    video_path: str | Path,
    results_dir: str | Path | None = None,
) -> Path:
    resolved_video = Path(video_path).expanduser().resolve()
    if results_dir is None or not str(results_dir).strip():
        return resolved_video.with_suffix("")
    return Path(results_dir).expanduser().resolve()


def build_default_behavior_agent_pipeline(
    *,
    results_root: str | Path,
    default_assay: str = "aggression",
    bout_frame_gap: int = 20,
    artifacts_ndjson: str | Path | None = None,
    use_memory: bool = True,
    use_analysis: bool = True,
) -> BehaviorAgentPipeline:
    resolved_artifacts = (
        Path(artifacts_ndjson).expanduser().resolve()
        if artifacts_ndjson is not None and str(artifacts_ndjson).strip()
        else None
    )
    perception_adapter = (
        NDJSONPerceptionAdapter(resolved_artifacts)
        if resolved_artifacts is not None
        else PassThroughPerceptionAdapter()
    )
    return BehaviorAgentPipeline(
        task_inferencer=KeywordTaskInferencer(default_assay=str(default_assay)),
        perception_adapter=perception_adapter,
        behavior_segmenter=AggressionSubEventSegmenter(
            frame_gap_threshold=max(1, int(bout_frame_gap))
        ),
        artifact_store=BehaviorAgentArtifactStore(Path(results_root)),
        memory_store=InMemoryMemoryStore() if bool(use_memory) else None,
        analysis_runner=DeterministicAnalysisRunner() if bool(use_analysis) else None,
    )


def run_default_behavior_agent_pipeline(
    *,
    video_path: str | Path,
    results_dir: str | Path | None = None,
    artifacts_ndjson: str | Path | None = None,
    run_id: str | None = None,
    episode_id: str | None = None,
    context_prompt: str = "",
    assay: str = "",
    default_assay: str = "aggression",
    model_policy: str = "annolid_behavior_agent_v1",
    bout_frame_gap: int = 20,
    use_memory: bool = True,
    use_analysis: bool = True,
) -> BehaviorAgentPipelineResult:
    resolved_video = Path(video_path).expanduser().resolve()
    results_root = resolve_behavior_results_root(
        video_path=resolved_video,
        results_dir=results_dir,
    )
    pipeline = build_default_behavior_agent_pipeline(
        results_root=results_root,
        default_assay=default_assay,
        bout_frame_gap=bout_frame_gap,
        artifacts_ndjson=artifacts_ndjson,
        use_memory=use_memory,
        use_analysis=use_analysis,
    )
    return pipeline.run(
        video_path=resolved_video,
        run_id=(str(run_id) if run_id else None),
        episode_id=(str(episode_id) if episode_id else None),
        context={
            "prompt": str(context_prompt or ""),
            "assay": str(assay or ""),
        },
        model_policy=str(model_policy or "annolid_behavior_agent_v1"),
    )


__all__ = [
    "build_default_behavior_agent_pipeline",
    "resolve_behavior_results_root",
    "run_default_behavior_agent_pipeline",
]
