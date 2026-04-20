from __future__ import annotations

from annolid.agents.assay_agent import AssayInferenceAgent
from annolid.agents.feature_agent import FeaturePlanningAgent
from annolid.agents.routing_agent import PerceptionRoutingAgent
from annolid.agents.segmentation_agent import BehaviorSegmentationAgent
from annolid.agents.coding_agent import AnalysisCodingAgent
from annolid.agents.report_agent import ReportAgent
from annolid.domain.behavior_agent import TaskPlan, TrackArtifact
from annolid.services.behavior_agent.model_policy import resolve_behavior_model_policy


def test_assay_feature_routing_agents_work_together() -> None:
    assay_agent = AssayInferenceAgent()
    feature_agent = FeaturePlanningAgent()
    routing_agent = PerceptionRoutingAgent()

    assay = assay_agent.infer(
        "/tmp/mouse_nor.mp4",
        context={"prompt": "novel object recognition behavior"},
    )
    assert assay.assay_type == "novel_object_recognition"
    assert assay.confidence >= 0.8

    feature = feature_agent.plan(assay.assay_type)
    assert "nose_tip" in feature.target_features

    route = routing_agent.route(
        assay_type=assay.assay_type,
        policy=resolve_behavior_model_policy("hosted_reasoning_local_tracking_v1"),
    )
    assert route.backend in {"annolid_tracking", "grounding_dino", "sam2_server"}


def test_segmentation_coding_reporting_agents() -> None:
    plan = TaskPlan(assay_type="aggression")
    artifacts = [
        TrackArtifact(
            artifact_id="a1",
            frame_index=10,
            track_id="mouse_1",
            label="slap in the face",
            meta={"count": 2},
        ),
        TrackArtifact(
            artifact_id="a2",
            frame_index=15,
            track_id="mouse_2",
            label="run away",
            meta={"count": 1},
        ),
    ]

    segmentation_agent = BehaviorSegmentationAgent()
    segmentation = segmentation_agent.segment(plan=plan, artifacts=artifacts)
    assert len(segmentation.segments) == 1
    assert segmentation.segments[0].label == "aggression_bout"
    assert "rationale" in segmentation.segments[0].to_dict()

    coding_agent = AnalysisCodingAgent()
    coding = coding_agent.run(
        plan=plan,
        artifacts=artifacts,
        segments=segmentation.segments,
    )
    assert "def run(inputs):" in coding.code
    assert coding.execution_output["status"] == "ok"
    assert any(row.get("metric") == "aggression_bout" for row in coding.derived_metrics)

    report_agent = ReportAgent()
    report = report_agent.build(
        task_plan=plan,
        artifacts=artifacts,
        segments=segmentation.segments,
        metrics=coding.derived_metrics,
        provenance={"model_policy": "hosted_reasoning_local_tracking_v1"},
    )
    assert "# Behavior Agent Report" in report.markdown
    assert "model_policy" in report.markdown
    assert report.html.startswith("<html>")
