from __future__ import annotations

import json

import pytest

from annolid.services.behavior_agent import (
    BehaviorAgentArtifactStore,
    SpecializedBehaviorAgentPipeline,
)


@pytest.mark.active_provider
def test_specialized_pipeline_writes_evidence_and_provenance(tmp_path) -> None:
    artifacts = tmp_path / "tracks.ndjson"
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

    pipeline = SpecializedBehaviorAgentPipeline(
        artifact_store=BehaviorAgentArtifactStore(tmp_path),
    )
    result = pipeline.run(
        video_path="/tmp/mouse.mp4",
        run_id="run_phase3_001",
        context={
            "prompt": "score aggression bouts",
            "total_frames": 120,
        },
        model_policy_name="hosted_reasoning_local_tracking_v1",
        artifacts_ndjson=str(artifacts),
    )

    assert result.run_id == "run_phase3_001"
    assert result.backend in {"annolid_tracking", "grounding_dino", "sam2_server"}
    assert result.artifact_count == 2
    assert result.segment_count >= 1
    assert result.manifest_path.exists()

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["analysis_run"]["run_id"] == "run_phase3_001"
    assert manifest["task_plan"]["assay_type"] == "aggression"
    assert manifest["artifacts"]["evidence"] == "artifacts/evidence.ndjson"
    assert manifest["artifacts"]["report_html"] == "artifacts/report.html"
    assert (
        manifest["provenance"]["model_policy"] == "hosted_reasoning_local_tracking_v1"
    )


@pytest.mark.active_provider
def test_specialized_pipeline_is_immutable_by_run_id(tmp_path) -> None:
    pipeline = SpecializedBehaviorAgentPipeline(
        artifact_store=BehaviorAgentArtifactStore(tmp_path),
    )

    first = pipeline.run(
        video_path="/tmp/mouse.mp4",
        run_id="run_phase3_immut",
        context={"prompt": "open field"},
    )
    assert first.manifest_path.exists()

    try:
        pipeline.run(
            video_path="/tmp/mouse.mp4",
            run_id="run_phase3_immut",
            context={"prompt": "open field"},
        )
        assert False, (
            "Expected immutable run directory collision to raise FileExistsError"
        )
    except FileExistsError:
        pass
