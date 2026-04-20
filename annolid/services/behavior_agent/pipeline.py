"""Typed behavior-agent orchestration pipeline.

Phase 2 orchestration: task inference -> perception -> segmentation -> memory ->
report/artifacts. This module is additive and does not replace existing GUI/CLI
flows until integrated explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from annolid import version as annolid_version
from annolid.domain.behavior_agent import (
    AnalysisRun,
    Episode,
    MemoryRecord,
)
from annolid.services.behavior_agent.artifact_store import BehaviorAgentArtifactStore
from annolid.services.behavior_agent.bout_scoring import (
    AggressionBoutCount,
    aggregate_aggression_bout_counts,
    validate_aggression_bout_counts,
)
from annolid.services.behavior_agent.interfaces import (
    AnalysisRunner,
    BehaviorSegmenter,
    MemoryStore,
    PerceptionAdapter,
    TaskInferencer,
)
from annolid.services.behavior_agent.orchestrator import BehaviorAgentOrchestrator


@dataclass(frozen=True)
class BehaviorAgentPipelineResult:
    run_id: str
    manifest_path: Path
    episode: Episode
    task_plan_assay: str
    artifact_count: int
    segment_count: int
    bout_counts: list[AggressionBoutCount]
    validation_errors: list[str]


class BehaviorAgentPipeline:
    def __init__(
        self,
        *,
        task_inferencer: TaskInferencer,
        perception_adapter: PerceptionAdapter,
        behavior_segmenter: BehaviorSegmenter,
        artifact_store: BehaviorAgentArtifactStore,
        memory_store: MemoryStore | None = None,
        analysis_runner: AnalysisRunner | None = None,
    ) -> None:
        self._task_inferencer = task_inferencer
        self._perception_adapter = perception_adapter
        self._behavior_segmenter = behavior_segmenter
        self._artifact_store = artifact_store
        self._memory_store = memory_store
        self._analysis_runner = analysis_runner
        self._orchestrator = BehaviorAgentOrchestrator(
            task_inferencer=task_inferencer,
            perception_adapter=perception_adapter,
            behavior_segmenter=behavior_segmenter,
            memory_store=memory_store,
            analysis_runner=analysis_runner,
        )

    def run(
        self,
        *,
        video_path: str | Path,
        run_id: str | None = None,
        context: dict | None = None,
        episode_id: str | None = None,
        model_policy: str = "annolid_behavior_agent_v1",
    ) -> BehaviorAgentPipelineResult:
        resolved_video = Path(video_path).expanduser().resolve()
        generated_run_id = str(run_id or f"run_{uuid4().hex[:12]}")
        generated_episode_id = str(
            episode_id or f"episode_{resolved_video.stem or uuid4().hex[:8]}"
        )

        orchestration = self._orchestrator.run(
            video=resolved_video,
            context=context,
        )
        task_plan = orchestration.task_plan
        artifacts = orchestration.artifacts
        segments = orchestration.segments

        bout_counts = aggregate_aggression_bout_counts(segments)
        validation_errors = validate_aggression_bout_counts(bout_counts)

        now_iso = datetime.now(UTC).isoformat()
        episode = Episode(
            episode_id=generated_episode_id,
            video_path=str(resolved_video),
            metadata={"context": dict(context or {})},
        )
        analysis_run = AnalysisRun(
            run_id=generated_run_id,
            episode_id=generated_episode_id,
            status="completed" if not validation_errors else "failed",
            model_policy=str(model_policy),
            annolid_version=getattr(annolid_version, "__version__", None),
            created_at=now_iso,
            metadata={"validation_error_count": len(validation_errors)},
        )

        memory_records = [
            MemoryRecord(
                key="assay_type",
                value=str(task_plan.assay_type),
                namespace="behavior_agent",
                source="task_inferencer",
                created_at=now_iso,
                evidence_refs=[generated_run_id],
            )
        ]
        if self._memory_store is not None:
            self._memory_store.upsert(
                "behavior_agent",
                [record.to_dict() for record in memory_records],
            )

        analysis_code = None
        report_text = None
        if self._analysis_runner is not None:
            inputs = {
                "artifact_count": len(artifacts),
                "segment_count": len(segments),
                "bout_count": len(bout_counts),
            }
            analysis_code = self._analysis_runner.generate_code(task_plan, inputs)
            result = self._analysis_runner.execute(analysis_code, inputs)
            report_text = (
                "# Behavior Agent Report\n"
                f"run_id: {generated_run_id}\n"
                f"assay_type: {task_plan.assay_type}\n"
                f"result: {result}\n"
            )

        metrics_rows = [row.to_dict() for row in bout_counts]
        if validation_errors:
            metrics_rows.append(
                {
                    "metric": "validation_errors",
                    "count": len(validation_errors),
                    "errors": list(validation_errors),
                }
            )

        manifest_path = self._artifact_store.write_run(
            analysis_run=analysis_run,
            episode=episode,
            task_plan=task_plan,
            track_artifacts=artifacts,
            behavior_segments=segments,
            memory_records=memory_records,
            metrics_rows=metrics_rows,
            analysis_code=analysis_code,
            report_text=report_text,
        )

        return BehaviorAgentPipelineResult(
            run_id=generated_run_id,
            manifest_path=manifest_path,
            episode=episode,
            task_plan_assay=task_plan.assay_type,
            artifact_count=len(artifacts),
            segment_count=len(segments),
            bout_counts=bout_counts,
            validation_errors=validation_errors,
        )


__all__ = [
    "BehaviorAgentPipeline",
    "BehaviorAgentPipelineResult",
]
