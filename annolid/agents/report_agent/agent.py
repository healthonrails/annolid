"""Report agent for markdown/html summaries with provenance links."""

from __future__ import annotations

from dataclasses import dataclass, field
import html as _html
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
        html_doc = self._render_html(
            task_plan=task_plan,
            artifacts=artifacts,
            segments=segments,
            metrics=metrics,
            provenance=provenance,
            markdown=md,
        )
        evidence = [
            {
                "stage": "reporting",
                "artifact_count": len(artifacts),
                "segment_count": len(segments),
                "metric_count": len(metrics),
                "provenance_keys": sorted(list(provenance.keys())),
            }
        ]
        return ReportResult(markdown=md, html=html_doc, evidence=evidence)

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
        if task_plan.objectives:
            for obj in task_plan.objectives:
                lines.append(f"- {obj}")
        else:
            lines.append("- None specified")
        lines.extend(["", "## Target Features"])
        if task_plan.target_features:
            for feat in task_plan.target_features:
                lines.append(f"- {feat}")
        else:
            lines.append("- None specified")

        if metrics:
            lines.extend(
                [
                    "",
                    "## Derived Metrics",
                    "",
                    "| Metric | Value / Summary |",
                    "|---|---|",
                ]
            )
            for row in metrics:
                if not isinstance(row, dict):
                    continue
                m_name = str(row.get("metric") or row.get("name") or "metric")
                if "value" in row and len(row) <= 2:
                    m_val = str(row.get("value"))
                else:
                    extras = [
                        f"{k}={v}"
                        for k, v in sorted(row.items())
                        if k not in {"metric", "name"}
                    ]
                    m_val = ", ".join(extras) if extras else str(row.get("value", ""))
                lines.append(f"| `{m_name}` | {m_val} |")

        if segments:
            lines.extend(
                [
                    "",
                    "## Behavior Segments",
                    "",
                    "| Segment ID | Label | Start Frame | End Frame | Confidence |",
                    "|---|---|---:|---:|---:|",
                ]
            )
            for seg in segments[:100]:
                seg_id = str(getattr(seg, "segment_id", ""))
                seg_lbl = str(getattr(seg, "label", ""))
                s_frame = int(getattr(seg, "start_frame", 0))
                e_frame = int(getattr(seg, "end_frame", 0))
                conf = getattr(seg, "confidence", None)
                conf_txt = (
                    f"{float(conf):.3f}" if isinstance(conf, (int, float)) else "NA"
                )
                lines.append(
                    f"| `{seg_id}` | `{seg_lbl}` | {s_frame} | {e_frame} | {conf_txt} |"
                )

        lines.extend(["", "## Provenance"])
        for key in sorted(provenance.keys()):
            lines.append(f"- {key}: {provenance[key]}")
        return "\n".join(lines).strip() + "\n"

    @classmethod
    def _render_html(
        cls,
        *,
        task_plan: TaskPlan,
        artifacts: list[TrackArtifact],
        segments: list[BehaviorSegment],
        metrics: list[dict[str, Any]],
        provenance: dict[str, Any],
        markdown: str,
    ) -> str:
        esc = _html.escape
        metric_rows_html = []
        for row in metrics:
            if not isinstance(row, dict):
                continue
            m_name = str(row.get("metric") or row.get("name") or "metric")
            if "value" in row and len(row) <= 2:
                m_val = str(row.get("value"))
            else:
                extras = [
                    f"{k}={v}"
                    for k, v in sorted(row.items())
                    if k not in {"metric", "name"}
                ]
                m_val = ", ".join(extras) if extras else str(row.get("value", ""))
            metric_rows_html.append(
                f"<tr><td><code>{esc(m_name)}</code></td><td>{esc(m_val)}</td></tr>"
            )

        segment_rows_html = []
        for seg in segments[:100]:
            seg_id = str(getattr(seg, "segment_id", ""))
            seg_lbl = str(getattr(seg, "label", ""))
            s_frame = int(getattr(seg, "start_frame", 0))
            e_frame = int(getattr(seg, "end_frame", 0))
            conf = getattr(seg, "confidence", None)
            conf_txt = f"{float(conf):.3f}" if isinstance(conf, (int, float)) else "NA"
            segment_rows_html.append(
                "<tr>"
                f"<td><code>{esc(seg_id)}</code></td>"
                f"<td>{esc(seg_lbl)}</td>"
                f"<td>{s_frame}</td>"
                f"<td>{e_frame}</td>"
                f"<td>{esc(conf_txt)}</td>"
                "</tr>"
            )

        prov_items = "".join(
            f"<li><strong>{esc(str(k))}:</strong> {esc(str(provenance[k]))}</li>"
            for k in sorted(provenance.keys())
        )
        obj_items = "".join(
            f"<li>{esc(str(o))}</li>"
            for o in (task_plan.objectives or ["None specified"])
        )
        feat_items = "".join(
            f"<li>{esc(str(f))}</li>"
            for f in (task_plan.target_features or ["None specified"])
        )
        metrics_table = (
            "<table><thead><tr><th>Metric</th><th>Value / Summary</th></tr></thead>"
            f"<tbody>{''.join(metric_rows_html)}</tbody></table>"
            if metric_rows_html
            else "<p>No derived metrics recorded.</p>"
        )
        segments_table = (
            "<table><thead><tr><th>Segment ID</th><th>Label</th><th>Start Frame</th>"
            "<th>End Frame</th><th>Confidence</th></tr></thead>"
            f"<tbody>{''.join(segment_rows_html)}</tbody></table>"
            if segment_rows_html
            else "<p>No behavior segments recorded.</p>"
        )
        style = (
            "body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;"
            "margin:24px;color:#102a43;background:#f8fbff;line-height:1.5;}"
            ".cards{display:flex;gap:12px;flex-wrap:wrap;margin:16px 0;}"
            ".card{background:#fff;border:1px solid #d9e2ec;border-radius:10px;padding:12px 16px;min-width:140px;}"
            ".card .k{font-size:12px;color:#627d98;text-transform:uppercase;}"
            ".card .v{font-size:22px;font-weight:700;color:#102a43;}"
            "table{border-collapse:collapse;width:100%;background:#fff;margin:12px 0;border:1px solid #d9e2ec;}"
            "th,td{padding:8px 12px;border-bottom:1px solid #e6eff8;text-align:left;font-size:13px;}"
            "th{background:#f0f4f8;font-weight:600;}"
            "code{background:#f0f4f8;padding:2px 5px;border-radius:4px;}"
            "details{margin-top:16px;}"
        )
        return (
            "<html><head><meta charset='utf-8'><title>Behavior Agent Report</title>"
            f"<style>{style}</style></head><body>"
            "<h1>Behavior Agent Report</h1>"
            "<div class='cards'>"
            f"<div class='card'><div class='k'>Assay</div><div class='v'>{esc(str(task_plan.assay_type))}</div></div>"
            f"<div class='card'><div class='k'>Artifacts</div><div class='v'>{len(artifacts)}</div></div>"
            f"<div class='card'><div class='k'>Segments</div><div class='v'>{len(segments)}</div></div>"
            f"<div class='card'><div class='k'>Metrics</div><div class='v'>{len(metrics)}</div></div>"
            "</div>"
            f"<h2>Objectives</h2><ul>{obj_items}</ul>"
            f"<h2>Target Features</h2><ul>{feat_items}</ul>"
            f"<h2>Derived Metrics</h2>{metrics_table}"
            f"<h2>Behavior Segments</h2>{segments_table}"
            f"<h2>Provenance</h2><ul>{prov_items}</ul>"
            f"<details><summary>Raw Markdown</summary><pre>{esc(markdown)}</pre></details>"
            "</body></html>"
        )

    @staticmethod
    def _markdown_to_html(markdown: str) -> str:
        escaped = _html.escape(str(markdown))
        return (
            "<html><head><meta charset='utf-8'><title>Behavior Agent Report</title></head>"
            "<body><pre>" + escaped + "</pre></body></html>"
        )


__all__ = ["ReportAgent", "ReportResult"]
