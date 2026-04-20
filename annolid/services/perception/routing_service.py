"""Perception routing service selecting adapter backends."""

from __future__ import annotations

from pathlib import Path

from annolid.agents.routing_agent import PerceptionRoutingAgent
from annolid.domain.behavior_agent import TaskPlan, TrackArtifact, VideoRef
from annolid.services.behavior_agent.interfaces import PerceptionAdapter
from annolid.services.behavior_agent.model_policy import (
    BehaviorModelPolicy,
    resolve_behavior_model_policy,
)


class RoutedPerceptionService(PerceptionAdapter):
    def __init__(
        self,
        *,
        adapters: dict[str, PerceptionAdapter] | None = None,
        routing_agent: PerceptionRoutingAgent | None = None,
        model_policy: BehaviorModelPolicy | None = None,
    ) -> None:
        self._adapters = dict(adapters or {})
        self._routing_agent = routing_agent or PerceptionRoutingAgent()
        self._model_policy = model_policy or resolve_behavior_model_policy(None)

    def run(self, video: VideoRef, plan: TaskPlan) -> list[TrackArtifact]:
        resolved = Path(video).expanduser() if isinstance(video, (str, Path)) else video
        route = self._routing_agent.route(
            assay_type=plan.assay_type,
            policy=self._model_policy,
        )
        adapter = self._adapters.get(route.backend)
        if adapter is None:
            return []
        return adapter.run(resolved, plan)


__all__ = ["RoutedPerceptionService"]
