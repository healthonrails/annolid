from __future__ import annotations

import json

from annolid.domain.behavior_agent import (
    AnalysisRun,
    BehaviorSegment,
    BehaviorSubEvent,
    Episode,
    MemoryRecord,
    TaskPlan,
    TrackArtifact,
)
from annolid.services.behavior_agent import (
    BehaviorAgentArtifactStore,
    aggregate_aggression_bout_counts,
    normalize_aggression_sub_event_type,
    validate_aggression_bout_counts,
)


def test_aggregate_aggression_bout_counts_is_stable_and_deterministic() -> None:
    segments = [
        BehaviorSegment(
            segment_id="seg-b",
            label="aggression",
            start_frame=20,
            end_frame=24,
            sub_events=[
                BehaviorSubEvent(
                    event_type="run_away",
                    frame_index=22,
                    count=2,
                ),
                BehaviorSubEvent(
                    event_type="slap_face",
                    frame_index=21,
                    count=1,
                ),
            ],
        ),
        BehaviorSegment(
            segment_id="seg-a",
            label="aggression",
            start_frame=10,
            end_frame=12,
            sub_events=[
                BehaviorSubEvent(
                    event_type="fight_initiation",
                    frame_index=11,
                    count=1,
                ),
                BehaviorSubEvent(
                    event_type="slap in the face",
                    frame_index=10,
                    count=3,
                ),
            ],
        ),
    ]

    rows = aggregate_aggression_bout_counts(segments)

    assert [row.bout_id for row in rows] == ["bout_0001", "bout_0002"]
    assert [row.segment_id for row in rows] == ["seg-a", "seg-b"]
    assert rows[0].slap_face_count == 3
    assert rows[0].run_away_count == 0
    assert rows[0].fight_initiation_count == 1
    assert rows[0].total_sub_events == 4
    assert rows[1].slap_face_count == 1
    assert rows[1].run_away_count == 2
    assert rows[1].fight_initiation_count == 0
    assert rows[1].total_sub_events == 3
    assert validate_aggression_bout_counts(rows) == []


def test_normalize_aggression_sub_event_type_aliases() -> None:
    assert normalize_aggression_sub_event_type("slap in the face") == "slap_face"
    assert normalize_aggression_sub_event_type("run away") == "run_away"
    assert (
        normalize_aggression_sub_event_type("initiation_of_bigger_fights")
        == "fight_initiation"
    )
    assert normalize_aggression_sub_event_type("unknown") is None


def test_behavior_agent_artifact_store_writes_manifest_and_is_immutable(
    tmp_path,
) -> None:
    store = BehaviorAgentArtifactStore(tmp_path)
    run = AnalysisRun(
        run_id="run_001",
        episode_id="episode_001",
        status="completed",
        model_policy="hosted_reasoning_local_tracking_v1",
    )

    manifest_path = store.write_run(
        analysis_run=run,
        episode=Episode(
            episode_id="episode_001",
            video_path="/tmp/video.mp4",
            fps=30.0,
            total_frames=90,
        ),
        task_plan=TaskPlan(
            assay_type="novel_object_recognition",
            confidence=0.9,
            objectives=["measure investigation time"],
            target_features=["nose_tip", "object_centers"],
        ),
        track_artifacts=[
            TrackArtifact(
                artifact_id="track_001",
                frame_index=1,
                track_id="mouse_1",
                label="mouse",
                bbox_xyxy=[1.0, 2.0, 3.0, 4.0],
                source="tracker",
            )
        ],
        behavior_segments=[
            BehaviorSegment(
                segment_id="seg_001",
                label="aggression",
                start_frame=10,
                end_frame=15,
                sub_events=[
                    BehaviorSubEvent(
                        event_type="slap_face",
                        frame_index=12,
                        count=1,
                    )
                ],
            )
        ],
        memory_records=[
            MemoryRecord(
                key="assay_type",
                value="nor",
                namespace="analysis",
                source="task_inferencer",
            )
        ],
        metrics_rows=[{"metric": "distance", "value": 1.2}],
        analysis_code="print('ok')\n",
        report_text="# Report\n",
    )

    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert manifest["analysis_run"]["run_id"] == "run_001"
    assert manifest["task_plan"]["assay_type"] == "novel_object_recognition"
    assert manifest["artifacts"]["tracks"] == "artifacts/tracks.ndjson"
    assert manifest["artifacts"]["segments"] == "artifacts/behaviors.ndjson"
    assert manifest["artifacts"]["memory"] == "artifacts/memory.ndjson"
    assert manifest["artifacts"]["metrics"]["ndjson"] == "metrics.ndjson"
    assert "parquet" in manifest["artifacts"]["metrics"]
    assert manifest["artifacts"]["analysis_code"] == "artifacts/analysis.py"
    assert manifest["artifacts"]["report"] == "artifacts/report.md"

    try:
        store.write_run(
            analysis_run=run,
            episode=Episode(episode_id="episode_001", video_path="/tmp/video.mp4"),
            task_plan=TaskPlan(assay_type="novel_object_recognition"),
            track_artifacts=[],
            behavior_segments=[],
            memory_records=[],
        )
        assert False, "Expected immutable run directory guard to raise FileExistsError"
    except FileExistsError:
        pass
