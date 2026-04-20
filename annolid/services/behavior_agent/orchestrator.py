"""Pure orchestration service for the typed behavior-agent core.

This layer coordinates interface implementations and keeps domain/business
logic outside the orchestrator.
"""

from __future__ import annotations

from dataclasses import dataclass

from annolid.domain.behavior_agent import (
    BehaviorSegment,
    TaskPlan,
    TrackArtifact,
    VideoRef,
)
from annolid.services.behavior_agent.interfaces import (
    AnalysisRunner,
    BehaviorSegmenter,
    MemoryStore,
    PerceptionAdapter,
    TaskInferencer,
)


@dataclass(frozen=True)
class OrchestrationResult:
    video: VideoRef
    task_plan: TaskPlan
    artifacts: list[TrackArtifact]
    segments: list[BehaviorSegment]
    analysis_code: str | None = None
    analysis_result: dict | None = None


class BehaviorAgentOrchestrator:
    """Coordinate task inference, perception, segmentation, memory, analysis."""

    def __init__(
        self,
        *,
        task_inferencer: TaskInferencer,
        perception_adapter: PerceptionAdapter,
        behavior_segmenter: BehaviorSegmenter,
        memory_store: MemoryStore | None = None,
        analysis_runner: AnalysisRunner | None = None,
    ) -> None:
        self._task_inferencer = task_inferencer
        self._perception_adapter = perception_adapter
        self._behavior_segmenter = behavior_segmenter
        self._memory_store = memory_store
        self._analysis_runner = analysis_runner

    def run(
        self,
        *,
        video: VideoRef,
        context: dict | None = None,
        memory_namespace: str | None = None,
        memory_records: list[dict] | None = None,
        analysis_inputs: dict | None = None,
    ) -> OrchestrationResult:
        task_plan = self._task_inferencer.infer(video, context=context)
        artifacts = self._perception_adapter.run(video, task_plan)
        segments = self._behavior_segmenter.segment(task_plan, artifacts)

        if (
            self._memory_store is not None
            and memory_namespace
            and isinstance(memory_records, list)
            and memory_records
        ):
            self._memory_store.upsert(memory_namespace, memory_records)

        analysis_code: str | None = None
        analysis_result: dict | None = None
        if self._analysis_runner is not None:
            inputs = dict(analysis_inputs or {})
            analysis_code = self._analysis_runner.generate_code(task_plan, inputs)
            analysis_result = self._analysis_runner.execute(analysis_code, inputs)

        return OrchestrationResult(
            video=video,
            task_plan=task_plan,
            artifacts=artifacts,
            segments=segments,
            analysis_code=analysis_code,
            analysis_result=analysis_result,
        )


__all__ = [
    "BehaviorAgentOrchestrator",
    "OrchestrationResult",
]
