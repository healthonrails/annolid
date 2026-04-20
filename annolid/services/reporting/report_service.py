"""Reporting service built on report agent."""

from __future__ import annotations

from annolid.agents.report_agent import ReportAgent, ReportResult
from annolid.domain.behavior_agent import BehaviorSegment, TaskPlan, TrackArtifact


class ReportService:
    def __init__(self, *, report_agent: ReportAgent | None = None) -> None:
        self._report_agent = report_agent or ReportAgent()

    def build(
        self,
        *,
        task_plan: TaskPlan,
        artifacts: list[TrackArtifact],
        segments: list[BehaviorSegment],
        metrics: list[dict],
        provenance: dict,
    ) -> ReportResult:
        return self._report_agent.build(
            task_plan=task_plan,
            artifacts=artifacts,
            segments=segments,
            metrics=metrics,
            provenance=provenance,
        )


__all__ = ["ReportService"]
