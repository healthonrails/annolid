"""Experiment report generation and optional external tracker interop for Annolid.

Provides:
- Zero-dependency self-contained HTML5 + inline SVG + Markdown + JSON + CSV
  experiment comparison reports (works completely offline in any browser).
- Optional, lazily-imported export adapters for MLflow, Weights & Biases, and
  Rerun so labs using external trackers can push or archive Annolid runs with
  one click or Python call without adding mandatory dependencies.
"""

from __future__ import annotations

import csv
import html
import importlib.metadata
import importlib.util
import io
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from annolid.services.training_monitor import (
    DiscoveredRun,
    MetricSeries,
    compute_hparam_diff,
    extract_effective_hparams,
)
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text


_SVG_COLORS: Tuple[str, ...] = (
    "#2563EB",
    "#EA580C",
    "#16A34A",
    "#9333EA",
    "#0D9488",
    "#DC2626",
    "#CA8A04",
    "#DB2777",
)


@dataclass(frozen=True)
class TrackerAvailability:
    """Installation status of an optional experiment tracking backend."""

    name: str
    display_name: str
    installed: bool
    version: Optional[str]
    install_hint: str
    description: str


@dataclass(frozen=True)
class ExperimentReportArtifacts:
    """Paths written by ``export_training_experiment_report``."""

    output_dir: Path
    html_path: Path
    markdown_path: Path
    json_path: Path
    csv_path: Path


def _pkg_version(dist_name: str) -> Optional[str]:
    try:
        return importlib.metadata.version(dist_name)
    except Exception:
        return None


def detect_available_trackers() -> List[TrackerAvailability]:
    """Probe optional experiment tracker backends via ``find_spec`` (no imports)."""
    specs = (
        (
            "tensorboard",
            "TensorBoard",
            "tensorboard",
            'pip install "annolid[training]"',
            "Embedded & browser scalar/image/projector viewer",
        ),
        (
            "mlflow",
            "MLflow",
            "mlflow",
            "pip install mlflow",
            "Offline/self-hosted experiment tracking & model registry",
        ),
        (
            "wandb",
            "Weights & Biases",
            "wandb",
            "pip install wandb",
            "Cloud/offline experiment tracking, sweeps & artifact lineage",
        ),
        (
            "rerun",
            "Rerun (.rrd)",
            "rerun-sdk",
            "pip install rerun-sdk",
            "Interactive multimodal timeline viewer for vision, pose & 3D",
        ),
    )
    out: List[TrackerAvailability] = []
    for mod_name, title, dist_name, hint, desc in specs:
        try:
            installed = importlib.util.find_spec(mod_name) is not None
        except Exception:
            installed = False
        out.append(
            TrackerAvailability(
                name=mod_name,
                display_name=title,
                installed=installed,
                version=_pkg_version(dist_name) if installed else None,
                install_hint=hint,
                description=desc,
            )
        )
    return out


def _run_label(run: DiscoveredRun) -> str:
    if run.metadata.alias:
        return run.metadata.alias
    name = run.run_dir.name
    if run.model and run.model != name:
        return f"{run.model}/{name}"
    return name


def _fmt_opt(val: Optional[float], digits: int = 4) -> str:
    if val is None:
        return "—"
    if abs(val) > 0 and abs(val) < 1e-3:
        return f"{val:.2e}"
    return f"{val:.{digits}f}"


def _md_cell(value: object) -> str:
    """Sanitize a string for safe inclusion inside a GitHub-Flavored Markdown table cell."""
    text = str(value or "").replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    return text.replace("|", "\\|").strip()


def _sanitize_mlflow_metric_key(tag: str) -> str:
    """Restrict metric keys to MLflow's permitted character set."""
    cleaned = re.sub(r"[^A-Za-z0-9_.\-/ ]+", "_", str(tag or "metric")).strip("_ ")
    return (cleaned or "metric")[:240]


def export_runs_comparison_csv(
    runs: Sequence[DiscoveredRun], output_path: Path
) -> Path:
    """Write a flat run summary & best-metrics table atomically as CSV."""
    target = Path(output_path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label",
        "starred",
        "task",
        "model",
        "status",
        "step_unit",
        "last_step",
        "total_steps",
        "train_loss",
        "min_train_loss",
        "val_loss",
        "min_val_loss",
        "primary_metric_name",
        "primary_metric_latest",
        "primary_metric_best",
        "primary_metric_best_step",
        "tags",
        "notes",
        "health_warnings",
        "checkpoints",
        "run_dir",
    ]
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=fieldnames)
    writer.writeheader()
    for run in runs:
        writer.writerow(
            {
                "label": _run_label(run),
                "starred": "yes" if run.metadata.starred else "",
                "task": run.task,
                "model": run.model,
                "status": run.status,
                "step_unit": run.step_unit,
                "last_step": "" if run.last_step is None else run.last_step,
                "total_steps": "" if run.total_steps is None else run.total_steps,
                "train_loss": "" if run.train_loss is None else f"{run.train_loss:.6f}",
                "min_train_loss": (
                    "" if run.min_train_loss is None else f"{run.min_train_loss:.6f}"
                ),
                "val_loss": "" if run.val_loss is None else f"{run.val_loss:.6f}",
                "min_val_loss": (
                    "" if run.min_val_loss is None else f"{run.min_val_loss:.6f}"
                ),
                "primary_metric_name": run.primary_metric_name or "",
                "primary_metric_latest": (
                    ""
                    if run.primary_metric_value is None
                    else f"{run.primary_metric_value:.6f}"
                ),
                "primary_metric_best": (
                    ""
                    if run.best_metric_value is None
                    else f"{run.best_metric_value:.6f}"
                ),
                "primary_metric_best_step": (
                    "" if run.best_metric_step is None else run.best_metric_step
                ),
                "tags": ", ".join(run.metadata.tags),
                "notes": run.metadata.notes.replace("\n", " ").strip(),
                "health_warnings": "; ".join(f.message for f in run.health_findings),
                "checkpoints": "; ".join(str(p) for p in run.checkpoints),
                "run_dir": str(run.run_dir),
            }
        )
    atomic_write_text(target, buf.getvalue())
    return target


def _pick_series(
    run: DiscoveredRun, candidates: Sequence[str]
) -> Optional[MetricSeries]:
    lowered = {k.lower(): v for k, v in run.metrics.items()}
    for cand in candidates:
        hit = lowered.get(cand.lower())
        if hit is not None and hit.steps:
            return hit
    return None


def _render_svg_chart(
    *,
    title: str,
    series_list: Sequence[Tuple[str, str, Sequence[int], Sequence[float], bool]],
    width: int = 560,
    height: int = 260,
    fixed_unit_range: bool = False,
) -> str:
    """Render a crisp standalone inline SVG multi-series line chart."""
    pad_l, pad_r, pad_t, pad_b = 56, 18, 42, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    valid: List[Tuple[str, str, List[int], List[float], bool]] = []
    for lbl, col, xs, ys, dashed in series_list:
        clean_xs: List[int] = []
        clean_ys: List[float] = []
        for x, y in zip(xs, ys):
            yf = float(y)
            if math.isfinite(yf):
                clean_xs.append(int(x))
                clean_ys.append(yf)
        if clean_xs:
            valid.append((lbl, col, clean_xs, clean_ys, dashed))
    esc_title = html.escape(title)
    if not valid:
        return (
            f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
            f'role="img" aria-label="{esc_title}" style="max-width:100%;height:auto;'
            f'background:#0f172a;border:1px solid #1e293b;border-radius:10px;">'
            f'<text x="16" y="24" fill="#e2e8f0" font-size="13" font-weight="700">{esc_title}</text>'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" fill="#94a3b8" font-size="12">'
            f"No scalar metrics available</text></svg>"
        )

    all_x = [x for _l, _c, xs, _ys, _d in valid for x in xs]
    all_y = [y for _l, _c, _xs, ys, _d in valid for y in ys]
    min_x, max_x = min(all_x), max(all_x)
    if max_x == min_x:
        max_x = min_x + 1

    if fixed_unit_range:
        min_y = 0.0
        max_y = max(1.0, max(all_y) * 1.05)
    else:
        min_y, max_y = min(all_y), max(all_y)
        span = max_y - min_y
        if span <= 1e-12:
            pad = max(0.1, abs(max_y) * 0.1)
            min_y = max(0.0, min_y - pad)
            max_y = max_y + pad
        else:
            min_y = max(0.0, min_y - span * 0.08) if min_y >= 0 else min_y - span * 0.08
            max_y = max_y + span * 0.08

    def sx(x: int) -> float:
        return pad_l + ((x - min_x) / float(max_x - min_x)) * plot_w

    def sy(y: float) -> float:
        ny = max(0.0, min(1.0, (y - min_y) / float(max_y - min_y)))
        return pad_t + (1.0 - ny) * plot_h

    parts: List[str] = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'role="img" aria-label="{esc_title}" style="max-width:100%;height:auto;'
        f'background:#0f172a;border:1px solid #1e293b;border-radius:10px;">',
        f'<text x="16" y="24" fill="#f8fafc" font-size="13" font-weight="700">{esc_title}</text>',
    ]

    # Grid & Y ticks
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y_px = pad_t + (1.0 - frac) * plot_h
        y_val = min_y + frac * (max_y - min_y)
        parts.append(
            f'<line x1="{pad_l}" y1="{y_px:.1f}" x2="{width - pad_r}" y2="{y_px:.1f}" '
            f'stroke="#1e293b" stroke-dasharray="3,3" />'
        )
        parts.append(
            f'<text x="{pad_l - 8}" y="{y_px + 4:.1f}" text-anchor="end" '
            f'fill="#94a3b8" font-size="10">{y_val:.3g}</text>'
        )

    # X axis ticks
    parts.append(
        f'<text x="{pad_l}" y="{height - 10}" fill="#94a3b8" font-size="10">{min_x}</text>'
    )
    parts.append(
        f'<text x="{width - pad_r}" y="{height - 10}" text-anchor="end" '
        f'fill="#94a3b8" font-size="10">{max(all_x)}</text>'
    )

    # Curves
    legend_items: List[str] = []
    for idx, (lbl, col, xs, ys, dashed) in enumerate(valid):
        pts_attr = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
        dash_attr = ' stroke-dasharray="5,4"' if dashed else ""
        parts.append(
            f'<polyline fill="none" stroke="{col}" stroke-width="2"{dash_attr} '
            f'points="{pts_attr}" />'
        )
        if len(xs) <= 16:
            for x, y in zip(xs, ys):
                parts.append(
                    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="2.6" fill="{col}" />'
                )
        if idx < 6:
            legend_items.append(
                f'<span style="display:inline-flex;align-items:center;gap:5px;margin-right:12px;">'
                f'<span style="width:10px;height:10px;border-radius:2px;background:{col};display:inline-block;"></span>'
                f"{html.escape(lbl)}: {_fmt_opt(ys[-1], 4)}</span>"
            )

    parts.append("</svg>")
    legend_html = (
        '<div style="font-size:11px;color:#cbd5e1;margin-top:6px;">'
        + "".join(legend_items)
        + "</div>"
    )
    return "".join(parts) + legend_html


def export_training_experiment_report(
    runs: Sequence[DiscoveredRun],
    output_dir: Path,
    *,
    title: str = "Annolid Training Experiment Report",
) -> ExperimentReportArtifacts:
    """Export a self-contained HTML5 + inline SVG + Markdown + JSON + CSV report."""
    out = Path(output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    run_list = list(runs)

    csv_path = export_runs_comparison_csv(run_list, out / "runs_comparison.csv")
    hparam_rows = compute_hparam_diff(run_list, only_diffs=False)
    generated_at = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    # Build inline SVG series
    loss_series: List[Tuple[str, str, Sequence[int], Sequence[float], bool]] = []
    metric_series: List[Tuple[str, str, Sequence[int], Sequence[float], bool]] = []
    for idx, run in enumerate(run_list):
        col = _SVG_COLORS[idx % len(_SVG_COLORS)]
        rlabel = _run_label(run)
        tr = _pick_series(
            run,
            (
                "train_loss",
                "loss/total",
                "train_loss_total",
                "total_loss",
                "loss",
                "train/box_loss",
            ),
        )
        va = _pick_series(
            run, ("val_loss", "val/loss", "val_loss_total", "val/box_loss")
        )
        if tr is not None:
            loss_series.append((f"{rlabel} (train)", col, tr.steps, tr.values, False))
        if va is not None:
            loss_series.append((f"{rlabel} (val)", col, va.steps, va.values, True))
        if run.primary_metric_name and run.primary_metric_name in run.metrics:
            ms = run.metrics[run.primary_metric_name]
            metric_series.append(
                (f"{rlabel} ({ms.tag.split('/')[-1]})", col, ms.steps, ms.values, False)
            )

    loss_svg = _render_svg_chart(
        title="Training & Validation Loss",
        series_list=loss_series,
        fixed_unit_range=False,
    )
    fixed_unit = bool(
        metric_series
        and all(0.0 <= y <= 1.05 for _l, _c, _xs, ys, _d in metric_series for y in ys)
    )
    metric_svg = _render_svg_chart(
        title="Primary Validation Metric",
        series_list=metric_series,
        fixed_unit_range=fixed_unit,
    )

    # Structured JSON summary
    summary_payload: Dict[str, Any] = {
        "title": title,
        "generated_at": generated_at,
        "run_count": len(run_list),
        "runs": [
            {
                "label": _run_label(r),
                "run_dir": str(r.run_dir),
                "task": r.task,
                "model": r.model,
                "status": r.status,
                "starred": r.metadata.starred,
                "tags": list(r.metadata.tags),
                "notes": r.metadata.notes,
                "step_unit": r.step_unit,
                "last_step": r.last_step,
                "total_steps": r.total_steps,
                "train_loss": r.train_loss,
                "min_train_loss": r.min_train_loss,
                "val_loss": r.val_loss,
                "min_val_loss": r.min_val_loss,
                "primary_metric_name": r.primary_metric_name,
                "primary_metric_latest": r.primary_metric_value,
                "primary_metric_best": r.best_metric_value,
                "primary_metric_best_step": r.best_metric_step,
                "checkpoints": [str(p) for p in r.checkpoints],
                "health_findings": [
                    {"severity": f.severity, "code": f.code, "message": f.message}
                    for f in r.health_findings
                ],
                "hparams": extract_effective_hparams(r),
            }
            for r in run_list
        ],
        "hparam_diff": [
            {"key": row.key, "differs": row.differs, "values": list(row.values)}
            for row in hparam_rows
        ],
    }
    json_path = out / "experiment_summary.json"
    atomic_write_json(json_path, summary_payload, indent=2)

    # Markdown report
    md_lines: List[str] = [
        f"# {title}",
        "",
        f"Generated: `{generated_at}` · Runs compared: **{len(run_list)}**",
        "",
        "## Run Summary",
        "",
        "| Run | Task | Model | Status | Step | Train Loss (min) | Val Loss (min) | Best Metric | Tags | Health |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in run_list:
        step_str = (
            f"{r.last_step}/{r.total_steps}"
            if r.last_step is not None and r.total_steps
            else (str(r.last_step) if r.last_step is not None else "—")
        )
        best_m = (
            f"{(r.primary_metric_name or '').split('/')[-1]}={_fmt_opt(r.best_metric_value, 4)} (@{r.best_metric_step})"
            if r.best_metric_value is not None
            else "—"
        )
        health_str = (
            "; ".join(f"[{f.severity.upper()}] {f.message}" for f in r.health_findings)
            if r.health_findings
            else "Healthy"
        )
        star = "★ " if r.metadata.starred else ""
        md_lines.append(
            f"| {_md_cell(star + _run_label(r))} | {_md_cell(r.task)} | {_md_cell(r.model)} | "
            f"{_md_cell(r.status)} | {_md_cell(step_str)} | "
            f"{_fmt_opt(r.train_loss)} ({_fmt_opt(r.min_train_loss)}) | "
            f"{_fmt_opt(r.val_loss)} ({_fmt_opt(r.min_val_loss)}) | "
            f"{_md_cell(best_m)} | {_md_cell(', '.join(r.metadata.tags) or '—')} | "
            f"{_md_cell(health_str)} |"
        )

    if any(r.metadata.notes for r in run_list):
        md_lines += ["", "## Lab Notes", ""]
        for r in run_list:
            if r.metadata.notes:
                md_lines.append(f"### {_run_label(r)}")
                md_lines.append("")
                md_lines.append(r.metadata.notes.strip())
                md_lines.append("")

    diff_only = [row for row in hparam_rows if row.differs]
    if diff_only:
        md_lines += ["", "## Hyperparameter Differences", ""]
        headers = (
            "| Parameter | "
            + " | ".join(_md_cell(_run_label(r)) for r in run_list)
            + " |"
        )
        sep = "| --- | " + " | ".join("---" for _ in run_list) + " |"
        md_lines += [headers, sep]
        for row in diff_only:
            md_lines.append(
                "| `"
                + _md_cell(row.key)
                + "` | "
                + " | ".join(_md_cell(v) for v in row.values)
                + " |"
            )

    md_path = out / "experiment_report.md"
    atomic_write_text(md_path, "\n".join(md_lines) + "\n")

    # HTML5 self-contained report
    run_rows_html: List[str] = []
    for r in run_list:
        step_str = (
            f"{r.last_step}/{r.total_steps}"
            if r.last_step is not None and r.total_steps
            else (str(r.last_step) if r.last_step is not None else "—")
        )
        best_m = (
            f"{html.escape((r.primary_metric_name or '').split('/')[-1])} = "
            f"<b>{_fmt_opt(r.best_metric_value, 4)}</b> ({r.step_unit} {r.best_metric_step})"
            if r.best_metric_value is not None
            else "—"
        )
        tags_html = (
            " ".join(
                f'<span class="tag">{html.escape(t)}</span>' for t in r.metadata.tags
            )
            or "—"
        )
        if r.health_findings:
            badges = " ".join(
                f'<span class="badge {html.escape(f.severity)}">{html.escape(f.message)}</span>'
                for f in r.health_findings
            )
        else:
            badges = '<span class="badge ok">Healthy</span>'
        star = "★ " if r.metadata.starred else ""
        notes_sub = (
            f'<div class="notes">{html.escape(r.metadata.notes)}</div>'
            if r.metadata.notes
            else ""
        )
        run_rows_html.append(
            "<tr>"
            f"<td><b>{star}{html.escape(_run_label(r))}</b><div class='sub'>{html.escape(str(r.run_dir))}</div>{notes_sub}</td>"
            f"<td>{html.escape(r.task)}</td>"
            f"<td>{html.escape(r.model)}</td>"
            f"<td>{html.escape(r.status)}</td>"
            f"<td>{html.escape(step_str)}</td>"
            f"<td>{_fmt_opt(r.train_loss)} <span class='sub'>(min {_fmt_opt(r.min_train_loss)})</span></td>"
            f"<td>{_fmt_opt(r.val_loss)} <span class='sub'>(min {_fmt_opt(r.min_val_loss)})</span></td>"
            f"<td>{best_m}</td>"
            f"<td>{tags_html}</td>"
            f"<td>{badges}</td>"
            "</tr>"
        )

    hp_headers = "".join(f"<th>{html.escape(_run_label(r))}</th>" for r in run_list)
    hp_rows_html: List[str] = []
    for row in hparam_rows:
        cls = ' class="diff"' if row.differs else ""
        cells = "".join(f"<td>{html.escape(v)}</td>" for v in row.values)
        hp_rows_html.append(
            f"<tr{cls}><td><code>{html.escape(row.key)}</code></td>{cells}</tr>"
        )

    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
          background: #0b1120; color: #e2e8f0; margin: 0; padding: 24px; line-height: 1.45; }}
  h1 {{ margin: 0 0 4px 0; font-size: 22px; }}
  h2 {{ margin: 24px 0 10px 0; font-size: 16px; color: #93c5fd; }}
  .meta {{ color: #94a3b8; font-size: 12px; margin-bottom: 18px; }}
  .charts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
  .card {{ background: #111827; border: 1px solid #1e293b; border-radius: 12px; padding: 14px; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
  th, td {{ text-align: left; padding: 8px 10px; border-bottom: 1px solid #1e293b; vertical-align: top; }}
  th {{ color: #93c5fd; background: #0f172a; }}
  tr.diff td {{ background: rgba(37, 99, 235, 0.14); }}
  .sub {{ color: #94a3b8; font-size: 11px; }}
  .notes {{ color: #cbd5e1; font-size: 11px; margin-top: 4px; white-space: pre-wrap; }}
  .tag {{ display: inline-block; padding: 2px 7px; border-radius: 999px; background: #1e293b; color: #93c5fd; font-size: 11px; margin: 1px; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 6px; font-size: 11px; margin: 1px 0; }}
  .badge.ok {{ background: rgba(22, 163, 74, 0.18); color: #86efac; }}
  .badge.info {{ background: rgba(37, 99, 235, 0.2); color: #93c5fd; }}
  .badge.warning {{ background: rgba(234, 88, 12, 0.22); color: #fdba74; }}
  .badge.critical {{ background: rgba(220, 38, 38, 0.24); color: #fca5a5; }}
</style>
</head>
<body>
  <h1>{html.escape(title)}</h1>
  <div class="meta">Generated {html.escape(generated_at)} · {len(run_list)} run(s) · Companion files: runs_comparison.csv, experiment_report.md, experiment_summary.json</div>
  <div class="charts">
    <div class="card">{loss_svg}</div>
    <div class="card">{metric_svg}</div>
  </div>
  <h2>Run Comparison Matrix</h2>
  <div class="card" style="overflow-x:auto;">
    <table>
      <thead>
        <tr>
          <th>Run</th><th>Task</th><th>Model</th><th>Status</th><th>Step</th>
          <th>Train Loss</th><th>Val Loss</th><th>Best Metric</th><th>Tags</th><th>Diagnostics</th>
        </tr>
      </thead>
      <tbody>{"".join(run_rows_html)}</tbody>
    </table>
  </div>
  <h2>Hyperparameter Diff (highlighted rows differ across runs)</h2>
  <div class="card" style="overflow-x:auto;">
    <table>
      <thead><tr><th>Parameter</th>{hp_headers}</tr></thead>
      <tbody>{"".join(hp_rows_html)}</tbody>
    </table>
  </div>
</body>
</html>
"""
    html_path = out / "experiment_report.html"
    atomic_write_text(html_path, html_doc)

    return ExperimentReportArtifacts(
        output_dir=out,
        html_path=html_path,
        markdown_path=md_path,
        json_path=json_path,
        csv_path=csv_path,
    )


def _log_mlflow_metrics_batched(
    mlflow_mod: Any, run_id: str, run: DiscoveredRun
) -> None:
    """Log scalar series using ``MlflowClient.log_batch`` when available, else ``log_metric``."""
    metric_tuples: List[Tuple[str, float, int]] = []
    for tag, metric_series in run.metrics.items():
        clean_key = _sanitize_mlflow_metric_key(tag)
        for step, val in zip(metric_series.steps, metric_series.values):
            vf = float(val)
            if math.isfinite(vf):
                metric_tuples.append((clean_key, vf, int(step)))

    if not metric_tuples:
        return

    client_cls = getattr(getattr(mlflow_mod, "tracking", None), "MlflowClient", None)
    metric_cls = getattr(getattr(mlflow_mod, "entities", None), "Metric", None)
    if callable(client_cls) and callable(metric_cls):
        client = client_cls()
        ts_ms = int(time.time() * 1000)
        for start in range(0, len(metric_tuples), 1000):
            batch = [
                metric_cls(key=k, value=v, timestamp=ts_ms, step=st)
                for k, v, st in metric_tuples[start : start + 1000]
            ]
            client.log_batch(run_id=run_id, metrics=batch)
        return

    for clean_key, vf, step in metric_tuples:
        mlflow_mod.log_metric(clean_key, vf, step=step)


def export_runs_to_mlflow(
    runs: Sequence[DiscoveredRun],
    *,
    tracking_uri: Optional[str] = None,
    experiment_name: str = "Annolid",
) -> Dict[str, Any]:
    """Export one or more Annolid runs into an MLflow tracking store (lazy import)."""
    try:
        import mlflow  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "MLflow export requires the optional 'mlflow' package. "
            "Install it with: pip install mlflow"
        ) from exc

    prev_uri: Optional[str] = None
    if tracking_uri:
        try:
            prev_uri = mlflow.get_tracking_uri()
        except Exception:
            prev_uri = None
        mlflow.set_tracking_uri(str(tracking_uri))

    try:
        mlflow.set_experiment(str(experiment_name or "Annolid"))
        resolved_uri = mlflow.get_tracking_uri()
        exported_ids: List[str] = []
        for run in runs:
            with mlflow.start_run(run_name=_run_label(run)) as active:
                run_id = str(active.info.run_id)
                mlflow.set_tags(
                    {
                        "annolid.task": run.task,
                        "annolid.model": run.model,
                        "annolid.status": run.status,
                        "annolid.run_dir": str(run.run_dir),
                        "annolid.starred": str(run.metadata.starred).lower(),
                        "annolid.tags": ",".join(run.metadata.tags),
                    }
                )
                if run.metadata.notes:
                    mlflow.set_tag("mlflow.note.content", run.metadata.notes)

                # MLflow's batch limit is 100 params per log_params() call (Ultralytics args.yaml often exceeds 100).
                param_items = [
                    (_sanitize_mlflow_metric_key(k), str(v)[:500])
                    for k, v in extract_effective_hparams(run).items()
                    if k
                ]
                for start in range(0, len(param_items), 100):
                    chunk = dict(param_items[start : start + 100])
                    if chunk:
                        mlflow.log_params(chunk)

                _log_mlflow_metrics_batched(mlflow, run_id, run)

                for img_path in run.plot_images[:20]:
                    if img_path.is_file():
                        mlflow.log_artifact(str(img_path), artifact_path="plots")
                exported_ids.append(run_id)

        return {
            "backend": "mlflow",
            "tracking_uri": resolved_uri,
            "experiment_name": experiment_name,
            "exported_run_ids": exported_ids,
        }
    finally:
        if tracking_uri and prev_uri is not None:
            try:
                mlflow.set_tracking_uri(prev_uri)
            except Exception:
                pass


def export_run_to_wandb(
    run: DiscoveredRun,
    *,
    project: str = "annolid",
    mode: str = "offline",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Replay a completed Annolid run into Weights & Biases (offline by default)."""
    try:
        import wandb  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Weights & Biases export requires the optional 'wandb' package. "
            "Install it with: pip install wandb"
        ) from exc

    init_kwargs: Dict[str, Any] = {
        "project": str(project or "annolid"),
        "name": _run_label(run),
        "tags": list(run.metadata.tags) or None,
        "notes": run.metadata.notes or None,
        "config": extract_effective_hparams(run),
        "mode": str(mode or "offline"),
        "reinit": True,
    }
    target_dir = output_dir or (run.run_dir if run.run_dir.is_dir() else None)
    if target_dir is not None:
        init_kwargs["dir"] = str(Path(target_dir).expanduser().resolve())

    wb_run = wandb.init(**init_kwargs)
    try:
        steps_payload: Dict[int, Dict[str, float]] = {}
        for tag, metric_series in run.metrics.items():
            for step, val in zip(metric_series.steps, metric_series.values):
                vf = float(val)
                if math.isfinite(vf):
                    steps_payload.setdefault(int(step), {})[tag] = vf
        for step in sorted(steps_payload.keys()):
            wb_run.log(steps_payload[step], step=step)

        for img_path in run.plot_images[:12]:
            if img_path.is_file():
                wb_run.log({f"plots/{img_path.stem}": wandb.Image(str(img_path))})
        run_id = str(getattr(wb_run, "id", ""))
        run_dir_out = str(getattr(wb_run, "dir", ""))
    finally:
        wb_run.finish()

    return {
        "backend": "wandb",
        "mode": mode,
        "project": project,
        "run_id": run_id,
        "wandb_dir": run_dir_out,
    }


def _rerun_set_step(rr_mod: Any, timeline: str, step: int) -> None:
    """Set sequence timeline compatibly across rerun-sdk <=0.21 and >=0.22."""
    set_time_fn = getattr(rr_mod, "set_time", None)
    if callable(set_time_fn):
        set_time_fn(timeline, sequence=int(step))
    else:
        rr_mod.set_time_sequence(timeline, int(step))


def export_run_to_rerun(run: DiscoveredRun, output_rrd: Path) -> Path:
    """Export scalar curves and plot images into a Rerun ``.rrd`` recording."""
    try:
        import rerun as rr  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Rerun export requires the optional 'rerun-sdk' package. "
            "Install it with: pip install rerun-sdk"
        ) from exc

    target = Path(output_rrd).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    rr.init(f"annolid_{_run_label(run)}", spawn=False)
    rr.save(str(target))

    for tag, metric_series in run.metrics.items():
        entity = f"metrics/{tag}"
        for step, val in zip(metric_series.steps, metric_series.values):
            vf = float(val)
            if not math.isfinite(vf):
                continue
            _rerun_set_step(rr, run.step_unit, int(step))
            if hasattr(rr, "Scalars"):
                rr.log(entity, rr.Scalars(vf))
            else:
                rr.log(entity, rr.Scalar(vf))

    if run.plot_images:
        try:
            import numpy as np
            from PIL import Image

            _rerun_set_step(rr, run.step_unit, int(run.last_step or 0))
            for img_path in run.plot_images[:12]:
                if not img_path.is_file():
                    continue
                with Image.open(img_path) as im:
                    arr = np.asarray(im.convert("RGB"))
                rr.log(f"plots/{img_path.stem}", rr.Image(arr))
        except Exception:
            pass

    disconnect_fn = getattr(rr, "disconnect", None)
    if callable(disconnect_fn):
        try:
            disconnect_fn()
        except Exception:
            pass

    return target


__all__ = [
    "ExperimentReportArtifacts",
    "TrackerAvailability",
    "detect_available_trackers",
    "export_run_to_rerun",
    "export_run_to_wandb",
    "export_runs_comparison_csv",
    "export_runs_to_mlflow",
    "export_training_experiment_report",
]
