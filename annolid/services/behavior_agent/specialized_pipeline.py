"""Phase 3 specialized multi-agent behavior analysis pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from annolid import version as annolid_version
from annolid.agents.assay_agent import AssayInferenceAgent
from annolid.agents.coding_agent import AnalysisCodingAgent
from annolid.agents.feature_agent import FeaturePlanningAgent
from annolid.agents.report_agent import ReportAgent
from annolid.agents.routing_agent import PerceptionRoutingAgent
from annolid.agents.segmentation_agent import BehaviorSegmentationAgent
from annolid.domain.behavior_agent import AnalysisRun, Episode, MemoryRecord, TaskPlan
from annolid.services.behavior_agent.artifact_store import BehaviorAgentArtifactStore
from annolid.services.behavior_agent.defaults import (
    NDJSONPerceptionAdapter,
    PassThroughPerceptionAdapter,
)
from annolid.services.behavior_agent.model_policy import (
    resolve_behavior_model_policy,
)
from annolid.services.behavior_agent.bout_scoring import validate_aggression_bout_counts


@dataclass(frozen=True)
class SpecializedBehaviorPipelineResult:
    run_id: str
    manifest_path: Path
    episode: Episode
    task_plan: TaskPlan
    backend: str
    artifact_count: int
    segment_count: int
    validation_errors: list[str]


class SpecializedBehaviorAgentPipeline:
    """Coordinate specialized agents with replayable evidence output."""

    def __init__(
        self,
        *,
        artifact_store: BehaviorAgentArtifactStore,
        perception_adapters: dict[str, Any] | None = None,
        assay_agent: AssayInferenceAgent | None = None,
        feature_agent: FeaturePlanningAgent | None = None,
        routing_agent: PerceptionRoutingAgent | None = None,
        segmentation_agent: BehaviorSegmentationAgent | None = None,
        coding_agent: AnalysisCodingAgent | None = None,
        report_agent: ReportAgent | None = None,
    ) -> None:
        self._artifact_store = artifact_store
        self._perception_adapters = dict(perception_adapters or {})
        self._assay_agent = assay_agent or AssayInferenceAgent()
        self._feature_agent = feature_agent or FeaturePlanningAgent()
        self._routing_agent = routing_agent or PerceptionRoutingAgent()
        self._segmentation_agent = segmentation_agent or BehaviorSegmentationAgent()
        self._coding_agent = coding_agent or AnalysisCodingAgent()
        self._report_agent = report_agent or ReportAgent()

    def run(
        self,
        *,
        video_path: str | Path,
        context: dict | None = None,
        run_id: str | None = None,
        episode_id: str | None = None,
        model_policy_name: str | None = None,
        artifacts_ndjson: str | None = None,
    ) -> SpecializedBehaviorPipelineResult:
        resolved_video = Path(video_path).expanduser().resolve()
        generated_run_id = str(run_id or f"run_{uuid4().hex[:12]}")
        generated_episode_id = str(
            episode_id or f"episode_{resolved_video.stem or uuid4().hex[:8]}"
        )
        now_iso = datetime.now(UTC).isoformat()

        policy = resolve_behavior_model_policy(model_policy_name)

        assay = self._assay_agent.infer(resolved_video, context=context)
        feature = self._feature_agent.plan(assay.assay_type, context=context)
        task_plan = TaskPlan(
            assay_type=assay.assay_type,
            confidence=assay.confidence,
            objectives=list(feature.objectives),
            target_features=list(feature.target_features),
            context=dict(context or {}),
        )

        route = self._routing_agent.route(
            assay_type=task_plan.assay_type, policy=policy
        )
        perception = self._resolve_perception_adapter(
            backend=route.backend,
            artifacts_ndjson=artifacts_ndjson,
        )
        artifacts = perception.run(resolved_video, task_plan)

        segmentation = self._segmentation_agent.segment(
            plan=task_plan, artifacts=artifacts
        )
        segments = list(segmentation.segments)

        coding = self._coding_agent.run(
            plan=task_plan, artifacts=artifacts, segments=segments
        )
        metrics = list(coding.derived_metrics)

        # Focus validation on aggression bout rows when present.
        bout_rows = [
            row for row in metrics if str(row.get("metric") or "") == "aggression_bout"
        ]
        validation_errors = []
        if bout_rows:
            from annolid.services.behavior_agent.bout_scoring import AggressionBoutCount

            counts = [
                AggressionBoutCount(
                    bout_id=str(row.get("bout_id") or ""),
                    segment_id=str(row.get("segment_id") or ""),
                    start_frame=int(row.get("start_frame") or 0),
                    end_frame=int(row.get("end_frame") or 0),
                    duration_frames=int(row.get("duration_frames") or 0),
                    total_sub_events=int(row.get("total_sub_events") or 0),
                    slap_face_count=int(row.get("slap_face_count") or 0),
                    run_away_count=int(row.get("run_away_count") or 0),
                    fight_initiation_count=int(row.get("fight_initiation_count") or 0),
                )
                for row in bout_rows
            ]
            validation_errors = validate_aggression_bout_counts(counts)

        provenance = {
            "annolid_version": getattr(annolid_version, "__version__", None),
            "model_policy": policy.name,
            "task_inference_mode": policy.task_inference_mode,
            "perception_backend": route.backend,
        }
        reporting = self._report_agent.build(
            task_plan=task_plan,
            artifacts=artifacts,
            segments=segments,
            metrics=metrics,
            provenance=provenance,
        )

        evidence_rows: list[dict[str, Any]] = []
        evidence_rows.extend(assay.evidence)
        evidence_rows.extend(feature.evidence)
        evidence_rows.extend(route.evidence)
        evidence_rows.extend(segmentation.evidence)
        evidence_rows.extend(coding.evidence)
        evidence_rows.extend(reporting.evidence)
        evidence_rows.extend(
            [
                {
                    "stage": "analysis_outputs",
                    "generated_code": coding.code,
                    "execution_output": dict(coding.execution_output),
                }
            ]
        )

        memory_records = [
            MemoryRecord(
                key="assay_type",
                value=str(task_plan.assay_type),
                namespace="behavior_agent",
                source="specialized_pipeline",
                created_at=now_iso,
                evidence_refs=[generated_run_id],
                meta={"backend": route.backend},
            )
        ]

        analysis_run = AnalysisRun(
            run_id=generated_run_id,
            episode_id=generated_episode_id,
            status="completed" if not validation_errors else "failed",
            model_policy=policy.name,
            annolid_version=getattr(annolid_version, "__version__", None),
            created_at=now_iso,
            metadata={"validation_error_count": len(validation_errors)},
        )
        episode = Episode(
            episode_id=generated_episode_id,
            video_path=str(resolved_video),
            metadata={"context": dict(context or {})},
        )

        manifest_path = self._artifact_store.write_run(
            analysis_run=analysis_run,
            episode=episode,
            task_plan=task_plan,
            track_artifacts=artifacts,
            behavior_segments=segments,
            memory_records=memory_records,
            metrics_rows=metrics,
            analysis_code=coding.code,
            report_text=reporting.markdown,
            report_html=reporting.html,
            evidence_rows=evidence_rows,
            provenance=provenance,
        )

        return SpecializedBehaviorPipelineResult(
            run_id=generated_run_id,
            manifest_path=manifest_path,
            episode=episode,
            task_plan=task_plan,
            backend=route.backend,
            artifact_count=len(artifacts),
            segment_count=len(segments),
            validation_errors=list(validation_errors),
        )

    def _resolve_perception_adapter(
        self, *, backend: str, artifacts_ndjson: str | None
    ):
        if artifacts_ndjson:
            return NDJSONPerceptionAdapter(artifacts_ndjson)
        adapter = self._perception_adapters.get(str(backend))
        if adapter is not None:
            return adapter
        return PassThroughPerceptionAdapter()


__all__ = ["SpecializedBehaviorAgentPipeline", "SpecializedBehaviorPipelineResult"]
