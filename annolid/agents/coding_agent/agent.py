"""Analysis coding agent producing tiny deterministic sandboxed analyses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from annolid.domain.behavior_agent import BehaviorSegment, TaskPlan, TrackArtifact
from annolid.infrastructure.sandbox.execution import execute_generated_analysis
from annolid.services.behavior_agent.bout_scoring import (
    aggregate_aggression_bout_counts,
)


@dataclass(frozen=True)
class CodingResult:
    code: str
    execution_output: dict[str, Any]
    derived_metrics: list[dict[str, Any]]
    evidence: list[dict[str, object]] = field(default_factory=list)


class AnalysisCodingAgent:
    """Generate and run compact deterministic analysis code."""

    def run(
        self,
        *,
        plan: TaskPlan,
        artifacts: list[TrackArtifact],
        segments: list[BehaviorSegment],
    ) -> CodingResult:
        code = self._generate_code(plan)
        inputs = {
            "assay_type": str(plan.assay_type),
            "artifact_count": int(len(artifacts)),
            "segment_count": int(len(segments)),
        }
        exec_output = execute_generated_analysis(code, inputs)
        metrics = self._derive_metrics(
            plan=plan, artifacts=artifacts, segments=segments
        )
        evidence = [
            {
                "stage": "analysis_coding",
                "assay_type": str(plan.assay_type),
                "generated_code_chars": len(code),
                "execution_status": str(exec_output.get("status") or "ok"),
                "metric_count": len(metrics),
            }
        ]
        return CodingResult(
            code=code,
            execution_output=exec_output,
            derived_metrics=metrics,
            evidence=evidence,
        )

    def generate_code(self, plan: TaskPlan) -> str:
        return self._generate_code(plan)

    def execute_code(self, code: str, inputs: dict[str, Any]) -> dict[str, Any]:
        return execute_generated_analysis(code, inputs)

    @staticmethod
    def _generate_code(plan: TaskPlan) -> str:
        return (
            "def run(inputs):\n"
            "    assay = str(inputs.get('assay_type', 'unknown'))\n"
            "    artifacts = int(inputs.get('artifact_count', 0))\n"
            "    segments = int(inputs.get('segment_count', 0))\n"
            "    return {\n"
            "        'status': 'ok',\n"
            "        'assay_type': assay,\n"
            "        'artifact_count': artifacts,\n"
            "        'segment_count': segments,\n"
            "    }\n"
        )

    @staticmethod
    def _derive_metrics(
        *,
        plan: TaskPlan,
        artifacts: list[TrackArtifact],
        segments: list[BehaviorSegment],
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [
            {
                "metric": "artifact_count",
                "value": int(len(artifacts)),
            },
            {
                "metric": "segment_count",
                "value": int(len(segments)),
            },
        ]
        if str(plan.assay_type).strip().lower() == "aggression":
            for row in aggregate_aggression_bout_counts(segments):
                rows.append(
                    {
                        "metric": "aggression_bout",
                        **row.to_dict(),
                    }
                )
        return rows


__all__ = ["AnalysisCodingAgent", "CodingResult"]
