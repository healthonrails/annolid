"""Evaluation helpers for Phase 3 benchmark reporting."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Iterable


@dataclass(frozen=True)
class BehaviorAgentBenchmarkRow:
    episode_id: str
    tracking_continuity: float | None = None
    assay_classification_correct: bool | None = None
    segment_f1: float | None = None
    correction_load_per_10min: float | None = None
    reproducibility_score: float | None = None
    cost_per_video_hour: float | None = None


@dataclass(frozen=True)
class BehaviorAgentBenchmarkSummary:
    episode_count: int
    tracking_continuity_mean: float | None
    assay_accuracy: float | None
    segment_f1_mean: float | None
    correction_load_mean: float | None
    reproducibility_mean: float | None
    cost_per_video_hour_mean: float | None


def summarize_behavior_agent_benchmarks(
    rows: Iterable[BehaviorAgentBenchmarkRow],
) -> BehaviorAgentBenchmarkSummary:
    materialized = list(rows)
    return BehaviorAgentBenchmarkSummary(
        episode_count=len(materialized),
        tracking_continuity_mean=_mean(
            [
                row.tracking_continuity
                for row in materialized
                if row.tracking_continuity is not None
            ]
        ),
        assay_accuracy=_mean(
            [
                1.0 if bool(row.assay_classification_correct) else 0.0
                for row in materialized
                if row.assay_classification_correct is not None
            ]
        ),
        segment_f1_mean=_mean(
            [row.segment_f1 for row in materialized if row.segment_f1 is not None]
        ),
        correction_load_mean=_mean(
            [
                row.correction_load_per_10min
                for row in materialized
                if row.correction_load_per_10min is not None
            ]
        ),
        reproducibility_mean=_mean(
            [
                row.reproducibility_score
                for row in materialized
                if row.reproducibility_score is not None
            ]
        ),
        cost_per_video_hour_mean=_mean(
            [
                row.cost_per_video_hour
                for row in materialized
                if row.cost_per_video_hour is not None
            ]
        ),
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    # Normalize floating-point representation so benchmark summaries are stable.
    return round(float(mean(values)), 12)


__all__ = [
    "BehaviorAgentBenchmarkRow",
    "BehaviorAgentBenchmarkSummary",
    "summarize_behavior_agent_benchmarks",
]
