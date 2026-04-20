from __future__ import annotations

from annolid.services.behavior_agent import (
    BehaviorAgentBenchmarkRow,
    summarize_behavior_agent_benchmarks,
)


def test_behavior_agent_benchmark_summary() -> None:
    summary = summarize_behavior_agent_benchmarks(
        [
            BehaviorAgentBenchmarkRow(
                episode_id="e1",
                tracking_continuity=0.9,
                assay_classification_correct=True,
                segment_f1=0.8,
                correction_load_per_10min=2.0,
                reproducibility_score=0.95,
                cost_per_video_hour=1.2,
            ),
            BehaviorAgentBenchmarkRow(
                episode_id="e2",
                tracking_continuity=0.7,
                assay_classification_correct=False,
                segment_f1=0.6,
                correction_load_per_10min=4.0,
                reproducibility_score=0.85,
                cost_per_video_hour=1.8,
            ),
        ]
    )
    assert summary.episode_count == 2
    assert summary.tracking_continuity_mean == 0.8
    assert summary.assay_accuracy == 0.5
    assert summary.segment_f1_mean == 0.7
    assert summary.correction_load_mean == 3.0
    assert summary.reproducibility_mean == 0.9
    assert summary.cost_per_video_hour_mean == 1.5
