from __future__ import annotations

import json

from annolid.domain.behavior_agent import TaskPlan, TrackArtifact
from annolid.services.behavior_agent import (
    AggressionSubEventSegmenter,
    BehaviorAgentArtifactStore,
    BehaviorAgentPipeline,
    DeterministicAnalysisRunner,
    InMemoryMemoryStore,
    KeywordTaskInferencer,
)
from annolid.services.behavior_agent.interfaces import PerceptionAdapter


class FixturePerceptionAdapter(PerceptionAdapter):
    def __init__(self, artifacts: list[TrackArtifact]) -> None:
        self._artifacts = artifacts

    def run(self, video, plan: TaskPlan) -> list[TrackArtifact]:
        _ = (video, plan)
        return list(self._artifacts)


def test_behavior_agent_pipeline_end_to_end_writes_manifest(tmp_path) -> None:
    artifacts = [
        TrackArtifact(
            artifact_id="a1",
            frame_index=10,
            label="slap in the face",
            track_id="mouse_1",
            meta={"count": 2},
        ),
        TrackArtifact(
            artifact_id="a2",
            frame_index=15,
            label="run away",
            track_id="mouse_2",
            meta={"count": 1},
        ),
        TrackArtifact(
            artifact_id="a3",
            frame_index=55,
            label="fight_initiation",
            track_id="mouse_1",
            meta={"count": 1},
        ),
    ]

    pipeline = BehaviorAgentPipeline(
        task_inferencer=KeywordTaskInferencer(default_assay="aggression"),
        perception_adapter=FixturePerceptionAdapter(artifacts),
        behavior_segmenter=AggressionSubEventSegmenter(frame_gap_threshold=20),
        artifact_store=BehaviorAgentArtifactStore(tmp_path),
        memory_store=InMemoryMemoryStore(),
        analysis_runner=DeterministicAnalysisRunner(),
    )

    result = pipeline.run(
        video_path="/tmp/aggression_video.mp4",
        run_id="run_phase2_001",
        context={"prompt": "analyze aggression bouts"},
    )

    assert result.run_id == "run_phase2_001"
    assert result.task_plan_assay == "aggression"
    assert result.artifact_count == 3
    assert result.segment_count == 2
    assert result.validation_errors == []

    assert len(result.bout_counts) == 2
    assert result.bout_counts[0].slap_face_count == 2
    assert result.bout_counts[0].run_away_count == 1
    assert result.bout_counts[0].fight_initiation_count == 0
    assert result.bout_counts[1].fight_initiation_count == 1

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["analysis_run"]["run_id"] == "run_phase2_001"
    assert manifest["task_plan"]["assay_type"] == "aggression"
    assert manifest["artifacts"]["tracks"] == "artifacts/tracks.ndjson"
    assert manifest["artifacts"]["segments"] == "artifacts/behaviors.ndjson"
    assert manifest["artifacts"]["report"] == "artifacts/report.md"


def test_keyword_task_inferencer_matches_expected_assay() -> None:
    inferencer = KeywordTaskInferencer(default_assay="unknown")
    plan = inferencer.infer(
        "/tmp/mouse_nor.mp4",
        context={"prompt": "run novel object recognition summary"},
    )
    assert plan.assay_type == "novel_object_recognition"
    assert plan.confidence == 0.8
    assert "nose_tip" in plan.target_features
