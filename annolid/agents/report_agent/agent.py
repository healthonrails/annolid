"""Report agent for markdown/html summaries with provenance links."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from annolid.domain.behavior_agent import BehaviorSegment, TaskPlan, TrackArtifact


@dataclass(frozen=True)
class ReportResult:
    markdown: str
    html: str
    evidence: list[dict[str, object]] = field(default_factory=list)


class ReportAgent:
    def build(
        self,
        *,
        task_plan: TaskPlan,
        artifacts: list[TrackArtifact],
        segments: list[BehaviorSegment],
        metrics: list[dict[str, Any]],
        provenance: dict[str, Any],
    ) -> ReportResult:
        md = self._build_markdown(
            task_plan=task_plan,
            artifacts=artifacts,
            segments=segments,
            metrics=metrics,
            provenance=provenance,
        )
        html = self._markdown_to_html(md)
        evidence = [
            {
                "stage": "reporting",
                "artifact_count": len(artifacts),
                "segment_count": len(segments),
                "metric_count": len(metrics),
                "provenance_keys": sorted(list(provenance.keys())),
            }
        ]
        return ReportResult(markdown=md, html=html, evidence=evidence)

    @staticmethod
    def _build_markdown(
        *,
        task_plan: TaskPlan,
        artifacts: list[TrackArtifact],
        segments: list[BehaviorSegment],
        metrics: list[dict[str, Any]],
        provenance: dict[str, Any],
    ) -> str:
        lines = [
            "# Behavior Agent Report",
            "",
            f"- Assay: {task_plan.assay_type}",
            f"- Artifacts: {len(artifacts)}",
            f"- Segments: {len(segments)}",
            f"- Metrics rows: {len(metrics)}",
            "",
            "## Objectives",
        ]
        for obj in task_plan.objectives:
            lines.append(f"- {obj}")
        lines.extend(["", "## Target Features"])
        for feat in task_plan.target_features:
            lines.append(f"- {feat}")
        lines.extend(["", "## Provenance"])
        for key in sorted(provenance.keys()):
            lines.append(f"- {key}: {provenance[key]}")
        return "\n".join(lines).strip() + "\n"

    @staticmethod
    def _markdown_to_html(markdown: str) -> str:
        escaped = (
            str(markdown)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
        )
        return (
            "<html><head><meta charset='utf-8'><title>Behavior Agent Report</title></head>"
            "<body><pre>" + escaped + "</pre></body></html>"
        )


__all__ = ["ReportAgent", "ReportResult"]
