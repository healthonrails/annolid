"""Task inference service built on specialized assay/feature agents."""

from __future__ import annotations

from pathlib import Path

from annolid.agents.assay_agent import AssayInferenceAgent
from annolid.agents.feature_agent import FeaturePlanningAgent
from annolid.domain.behavior_agent import TaskPlan, VideoRef
from annolid.services.behavior_agent.interfaces import TaskInferencer


class AssayInferenceService(TaskInferencer):
    def __init__(
        self,
        *,
        assay_agent: AssayInferenceAgent | None = None,
        feature_agent: FeaturePlanningAgent | None = None,
    ) -> None:
        self._assay_agent = assay_agent or AssayInferenceAgent()
        self._feature_agent = feature_agent or FeaturePlanningAgent()

    def infer(self, video: VideoRef, context: dict | None = None) -> TaskPlan:
        resolved = Path(video).expanduser() if isinstance(video, (str, Path)) else video
        assay = self._assay_agent.infer(resolved, context=context)
        feature_plan = self._feature_agent.plan(assay.assay_type, context=context)
        merged_context = dict(context or {})
        merged_context.setdefault("sampled_frame_indices", assay.sampled_frame_indices)
        return TaskPlan(
            assay_type=assay.assay_type,
            confidence=assay.confidence,
            objectives=list(feature_plan.objectives),
            target_features=list(feature_plan.target_features),
            context=merged_context,
        )


__all__ = ["AssayInferenceService"]
