"""Typed contracts for the behavior-agent pipeline.

These contracts are additive and do not change existing GUI/CLI annotation behavior.
They provide a stable schema for replayable behavior-agent analysis runs.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

SCHEMA_VERSION = "1.0"
VideoRef = str | Path

AggressionSubEventType = Literal[
    "slap_face",
    "run_away",
    "fight_initiation",
]


@dataclass(frozen=True)
class Episode:
    episode_id: str
    video_path: str
    fps: float | None = None
    total_frames: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TaskPlan:
    assay_type: str
    confidence: float | None = None
    objectives: list[str] = field(default_factory=list)
    target_features: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrackArtifact:
    artifact_id: str
    frame_index: int
    track_id: str | None = None
    label: str | None = None
    bbox_xyxy: list[float] | None = None
    keypoints: dict[str, list[float]] = field(default_factory=dict)
    score: float | None = None
    source: str = "unknown"
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorSubEvent:
    event_type: AggressionSubEventType
    frame_index: int
    count: int = 1
    actor_id: str | None = None
    target_id: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BehaviorSegment:
    segment_id: str
    label: str
    start_frame: int
    end_frame: int
    confidence: float | None = None
    rationale: str | None = None
    sub_events: list[BehaviorSubEvent] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "label": self.label,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "sub_events": [event.to_dict() for event in self.sub_events],
            "meta": dict(self.meta),
        }


@dataclass(frozen=True)
class AnalysisRun:
    run_id: str
    episode_id: str
    status: Literal["pending", "running", "completed", "failed"]
    model_policy: str
    annolid_version: str | None = None
    created_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MemoryRecord:
    key: str
    value: str
    namespace: str
    source: str | None = None
    created_at: str | None = None
    evidence_refs: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = [
    "SCHEMA_VERSION",
    "VideoRef",
    "AggressionSubEventType",
    "Episode",
    "TaskPlan",
    "TrackArtifact",
    "BehaviorSubEvent",
    "BehaviorSegment",
    "AnalysisRun",
    "MemoryRecord",
]
