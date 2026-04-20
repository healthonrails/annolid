"""Service interfaces for the behavior-agent core orchestration."""

from __future__ import annotations

from typing import Protocol

from annolid.domain.behavior_agent import (
    BehaviorSegment,
    TaskPlan,
    TrackArtifact,
    VideoRef,
)


class TaskInferencer(Protocol):
    def infer(self, video: VideoRef, context: dict | None = None) -> TaskPlan: ...


class PerceptionAdapter(Protocol):
    def run(self, video: VideoRef, plan: TaskPlan) -> list[TrackArtifact]: ...


class BehaviorSegmenter(Protocol):
    def segment(
        self,
        plan: TaskPlan,
        artifacts: list[TrackArtifact],
    ) -> list[BehaviorSegment]: ...


class MemoryStore(Protocol):
    def upsert(self, namespace: str, records: list[dict]) -> None: ...

    def search(self, namespace: str, query: str, top_k: int = 5) -> list[dict]: ...


class AnalysisRunner(Protocol):
    def generate_code(self, plan: TaskPlan, inputs: dict) -> str: ...

    def execute(self, code: str, inputs: dict) -> dict: ...


__all__ = [
    "TaskInferencer",
    "PerceptionAdapter",
    "BehaviorSegmenter",
    "MemoryStore",
    "AnalysisRunner",
]
