from __future__ import annotations


def test_behavior_agent_modules_import_without_circular_dependency() -> None:
    from annolid.agents.routing_agent import PerceptionRoutingAgent
    from annolid.agents.segmentation_agent import BehaviorSegmentationAgent
    from annolid.services.behavior_agent.model_policy import (
        resolve_behavior_model_policy,
    )

    route_agent = PerceptionRoutingAgent()
    segmentation_agent = BehaviorSegmentationAgent()
    policy = resolve_behavior_model_policy("hosted_reasoning_local_tracking_v1")

    assert route_agent is not None
    assert segmentation_agent is not None
    assert policy.name == "hosted_reasoning_local_tracking_v1"
