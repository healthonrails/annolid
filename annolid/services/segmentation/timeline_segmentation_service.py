"""Segmentation service built on behavior segmentation agent."""

from __future__ import annotations

from annolid.agents.segmentation_agent import BehaviorSegmentationAgent
from annolid.domain.behavior_agent import BehaviorSegment, TaskPlan, TrackArtifact
from annolid.services.behavior_agent.interfaces import BehaviorSegmenter


class TimelineSegmentationService(BehaviorSegmenter):
    def __init__(
        self, *, segmentation_agent: BehaviorSegmentationAgent | None = None
    ) -> None:
        self._segmentation_agent = segmentation_agent or BehaviorSegmentationAgent()

    def segment(
        self,
        plan: TaskPlan,
        artifacts: list[TrackArtifact],
    ) -> list[BehaviorSegment]:
        return self._segmentation_agent.segment(plan=plan, artifacts=artifacts).segments


__all__ = ["TimelineSegmentationService"]
