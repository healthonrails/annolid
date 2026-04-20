"""Behavior segmentation agent for timeline generation with rationales."""

from __future__ import annotations

from dataclasses import dataclass, field

from annolid.domain.behavior_agent import (
    BehaviorSegment,
    BehaviorSubEvent,
    TaskPlan,
    TrackArtifact,
)
from annolid.services.behavior_agent.bout_scoring import (
    normalize_aggression_sub_event_type,
)


@dataclass(frozen=True)
class SegmentationResult:
    segments: list[BehaviorSegment]
    evidence: list[dict[str, object]] = field(default_factory=list)


class BehaviorSegmentationAgent:
    """Generate timeline segments from artifacts and assay context."""

    def segment(
        self, *, plan: TaskPlan, artifacts: list[TrackArtifact]
    ) -> SegmentationResult:
        assay = str(plan.assay_type or "unknown").strip().lower()
        if assay == "aggression":
            segments = self._segment_aggression(artifacts)
        else:
            segments = self._segment_generic(plan, artifacts)
        evidence = [
            {
                "stage": "behavior_segmentation",
                "assay_type": assay,
                "segment_count": len(segments),
                "artifact_count": len(artifacts),
            }
        ]
        return SegmentationResult(segments=segments, evidence=evidence)

    def _segment_aggression(
        self, artifacts: list[TrackArtifact]
    ) -> list[BehaviorSegment]:
        events: list[BehaviorSubEvent] = []
        for artifact in artifacts:
            labels = [str(artifact.label or "")]
            meta_event = (
                artifact.meta.get("event_type")
                if isinstance(artifact.meta, dict)
                else None
            )
            if isinstance(meta_event, str):
                labels.append(meta_event)
            for raw in labels:
                normalized = normalize_aggression_sub_event_type(raw)
                if normalized is None:
                    continue
                count = (
                    int(artifact.meta.get("count", 1))
                    if isinstance(artifact.meta, dict)
                    else 1
                )
                if count < 0:
                    continue
                events.append(
                    BehaviorSubEvent(
                        event_type=normalized,
                        frame_index=int(artifact.frame_index),
                        count=count,
                        actor_id=str(artifact.track_id) if artifact.track_id else None,
                        meta={"artifact_id": artifact.artifact_id},
                    )
                )

        if not events:
            return []
        events.sort(key=lambda item: (item.frame_index, item.event_type))

        gap = 20
        groups: list[list[BehaviorSubEvent]] = []
        current: list[BehaviorSubEvent] = [events[0]]
        for event in events[1:]:
            if event.frame_index - current[-1].frame_index <= gap:
                current.append(event)
            else:
                groups.append(current)
                current = [event]
        groups.append(current)

        segments: list[BehaviorSegment] = []
        for idx, group in enumerate(groups, start=1):
            segments.append(
                BehaviorSegment(
                    segment_id=f"aggr_seg_{idx:04d}",
                    label="aggression_bout",
                    start_frame=min(item.frame_index for item in group),
                    end_frame=max(item.frame_index for item in group),
                    confidence=1.0,
                    rationale="Multistage temporal grouping from aggression sub-events",
                    sub_events=list(group),
                    meta={"method": "event_gap_grouping"},
                )
            )
        return segments

    def _segment_generic(
        self, plan: TaskPlan, artifacts: list[TrackArtifact]
    ) -> list[BehaviorSegment]:
        if not artifacts:
            return []
        ordered = sorted(artifacts, key=lambda item: int(item.frame_index))
        start = int(ordered[0].frame_index)
        end = int(ordered[-1].frame_index)
        return [
            BehaviorSegment(
                segment_id="generic_seg_0001",
                label=f"{plan.assay_type}_timeline",
                start_frame=start,
                end_frame=end,
                confidence=0.7,
                rationale="Single-pass timeline from available track artifacts",
                sub_events=[],
                meta={"method": "range_cover"},
            )
        ]


__all__ = ["BehaviorSegmentationAgent", "SegmentationResult"]
