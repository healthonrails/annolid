from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np

from .models import (
    Sim2RealIterationRecord,
    Sim2RealProjectState,
    VideoEnvironmentProfile,
    utc_now_iso,
)


def _fmt_float(val: Any, digits: int = 3, default: str = "—") -> str:
    if val is None:
        return default
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return default


def _fmt_pct(val: Any, default: str = "—") -> str:
    if val is None:
        return default
    try:
        return f"{float(val) * 100.0:.1f}%"
    except (TypeError, ValueError):
        return default


def _image_to_data_uri(path_str: str) -> str:
    if not path_str:
        return ""
    p = Path(path_str)
    if not p.is_file():
        return ""
    raw = p.read_bytes()
    mime = "image/png" if p.suffix.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _build_env_vs_sim_contact_sheet(
    profile: Optional[VideoEnvironmentProfile],
    iterations: Sequence[Sim2RealIterationRecord],
    output_path: Path,
) -> str:
    """Create side-by-side comparison sheet: real example samples vs matched 3D sim frames."""
    left_img = None
    if (
        profile
        and profile.sample_montage_path
        and Path(profile.sample_montage_path).is_file()
    ):
        left_img = cv2.imread(profile.sample_montage_path, cv2.IMREAD_COLOR)
    elif (
        profile
        and profile.reference_background_path
        and Path(profile.reference_background_path).is_file()
    ):
        left_img = cv2.imread(profile.reference_background_path, cv2.IMREAD_COLOR)

    sim_previews: list[str] = []
    for it in reversed(iterations):
        ds = it.dataset or {}
        imgs = list(ds.get("preview_images") or [])
        if imgs:
            sim_previews = imgs
            break

    right_img = None
    if sim_previews:
        tiles: list[np.ndarray] = []
        for p in sim_previews[:4]:
            im = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if im is not None:
                tiles.append(cv2.resize(im, (320, 240), interpolation=cv2.INTER_AREA))
        if len(tiles) == 1:
            right_img = tiles[0]
        elif len(tiles) == 2:
            right_img = np.concatenate(tiles, axis=1)
        elif tiles:
            while len(tiles) < 4:
                tiles.append(np.zeros_like(tiles[0]))
            right_img = np.concatenate(
                [np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:4], axis=1)],
                axis=0,
            )

    if left_img is None and right_img is None:
        return ""
    target_w, target_h = 560, 420
    panels: list[np.ndarray] = []
    for img, title in (
        (left_img, "Real Example Video Samples"),
        (right_img, "Matched 3D Simulation Dataset"),
    ):
        if img is None:
            continue
        panel = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        bar = np.full((34, target_w, 3), (24, 28, 38), dtype=np.uint8)
        cv2.putText(
            bar,
            title,
            (12, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (238, 244, 255),
            1,
            cv2.LINE_AA,
        )
        panels.append(np.concatenate([bar, panel], axis=0))

    if not panels:
        return ""
    combined = np.concatenate(panels, axis=1) if len(panels) > 1 else panels[0]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), combined)
    return str(output_path)


def _build_inline_svg_chart(iterations: Sequence[Sim2RealIterationRecord]) -> str:
    """Render a compact inline SVG line chart comparing Sim-to-Real score & coverage across iterations."""
    points: list[tuple[int, float, float, float]] = []
    for it in iterations:
        ev = it.evaluation or {}
        score = float(ev.get("sim2real_score") or 0.0)
        cov = float(ev.get("detection_coverage") or 0.0)
        val_m = ev.get("val_mask_map50")
        if val_m is None:
            val_m = ev.get("val_box_map50")
        val_f = float(val_m) if val_m is not None else 0.0
        points.append((int(it.iteration), score, cov, val_f))

    if not points:
        return ""

    w, h = 640, 210
    pad_l, pad_r, pad_t, pad_b = 46, 20, 24, 34
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b

    def _xy(idx: int, val: float) -> tuple[float, float]:
        x = (
            pad_l + plot_w * 0.5
            if len(points) == 1
            else pad_l + (idx / float(len(points) - 1)) * plot_w
        )
        y = pad_t + (1.0 - max(0.0, min(1.0, val))) * plot_h
        return x, y

    grid_lines: list[str] = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        gy = pad_t + (1.0 - tick) * plot_h
        grid_lines.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{w - pad_r}" y2="{gy:.1f}" '
            f'stroke="#334155" stroke-dasharray="3,3" stroke-width="1"/>'
            f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" text-anchor="end" fill="#94a3b8" font-size="10">{tick:.2f}</text>'
        )

    series_spec = [
        ("Sim2Real Score", 1, "#38bdf8"),
        ("Real Coverage", 2, "#34d399"),
        ("Val mAP@50", 3, "#f59e0b"),
    ]
    paths: list[str] = []
    for label, col_idx, color in series_spec:
        coords = [_xy(i, pt[col_idx]) for i, pt in enumerate(points)]
        poly = " ".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        if len(coords) >= 2:
            paths.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{poly}"/>'
            )
        for x, y in coords:
            paths.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')

    x_labels: list[str] = []
    for i, pt in enumerate(points):
        x, _ = _xy(i, 0.0)
        x_labels.append(
            f'<text x="{x:.1f}" y="{h - 10}" text-anchor="middle" fill="#cbd5e1" font-size="11">Iter {pt[0]}</text>'
        )

    legend = (
        '<g font-size="11" fill="#e2e8f0">'
        '<circle cx="60" cy="12" r="4" fill="#38bdf8"/><text x="70" y="16">Sim2Real Score</text>'
        '<circle cx="190" cy="12" r="4" fill="#34d399"/><text x="200" y="16">Real Video Coverage</text>'
        '<circle cx="350" cy="12" r="4" fill="#f59e0b"/><text x="360" y="16">Val mAP@50</text>'
        "</g>"
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:{w}px;background:#0f172a;border-radius:8px;">'
        + legend
        + "".join(grid_lines)
        + "".join(paths)
        + "".join(x_labels)
        + "</svg>"
    )


def generate_sim2real_reports(
    state: Sim2RealProjectState,
    *,
    output_dir: Optional[str | Path] = None,
) -> dict[str, str]:
    """Generate Markdown, JSON, and self-contained HTML Sim-to-Real workflow reports."""
    proj_dir = Path(state.project_dir).expanduser().resolve()
    rep_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (proj_dir / "reports")
    )
    rep_dir.mkdir(parents=True, exist_ok=True)

    profile = (
        VideoEnvironmentProfile.from_dict(state.profile)
        if isinstance(state.profile, Mapping)
        else None
    )
    iterations = [
        Sim2RealIterationRecord.from_dict(it)
        for it in (state.iterations or [])
        if isinstance(it, Mapping)
    ]

    env_vs_sim_path = _build_env_vs_sim_contact_sheet(
        profile, iterations, rep_dir / "contact_sheet_env_vs_sim.png"
    )

    latest_overlay = ""
    for it in reversed(iterations):
        cand = str((it.evaluation or {}).get("overlay_contact_sheet_path") or "")
        if cand and Path(cand).is_file():
            latest_overlay = cand
            break

    generated_at = utc_now_iso()

    # ------------------------------------------------------------------ #
    # 1. Structured JSON report
    # ------------------------------------------------------------------ #
    json_payload: dict[str, Any] = {
        "schema_version": state.schema_version,
        "generated_at": generated_at,
        "project_id": state.project_id,
        "project_dir": str(proj_dir),
        "example_videos": list(state.example_videos),
        "best_iteration": state.best_iteration,
        "best_score": state.best_score,
        "best_weights_path": state.best_weights_path,
        "environment_profile": profile.to_dict() if profile else None,
        "iterations": [it.to_dict() for it in iterations],
        "artifacts": {
            "env_vs_sim_contact_sheet": env_vs_sim_path,
            "latest_prediction_overlay": latest_overlay,
        },
    }
    json_path = rep_dir / "sim2real_report.json"
    json_path.write_text(
        json.dumps(json_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # ------------------------------------------------------------------ #
    # 2. Markdown report
    # ------------------------------------------------------------------ #
    md_lines: list[str] = [
        f"# Annolid Sim-to-Real Training & Evaluation Report (`{state.project_id}`)",
        "",
        f"- **Generated (UTC)**: `{generated_at}`",
        f"- **Project directory**: `{proj_dir}`",
        f"- **Example video(s)**: {', '.join(f'`{Path(v).name}`' for v in state.example_videos) or '*(none)*'}",
        f"- **Total iterations completed**: `{len(iterations)}`",
        f"- **Best iteration**: `{state.best_iteration if state.best_iteration is not None else '—'}` "
        f"(Sim-to-Real score: `{_fmt_float(state.best_score)}`)",
        f"- **Best checkpoint**: `{state.best_weights_path or '—'}`",
        "",
    ]

    if profile is not None:
        md_lines.extend(
            [
                "## 1. Real Example Video Environment Profile",
                "",
                "| Property | Measured / Inferred Value | Rationale |",
                "| --- | --- | --- |",
                f"| Species | `{profile.species}` (conf `{profile.species_confidence:.2f}`) | {profile.match_rationale.get('species', '')} |",
                f"| Resolution & FPS | `{profile.width}x{profile.height}` @ `{profile.fps:.1f}` fps (`{profile.total_frames}` frames) | Sampled `{profile.sampled_frames}` frames |",
                f"| Illumination | `{profile.recommended_lighting}` (luma `{profile.mean_luminance:.1f}`, sat `{profile.mean_saturation:.1f}`, grayscale=`{profile.is_grayscale}`) | {profile.match_rationale.get('lighting', '')} |",
                f"| Arena Geometry | `{profile.recommended_arena}` | {profile.match_rationale.get('arena', '')} |",
                f"| Camera View | `{profile.recommended_camera_view}` | {profile.match_rationale.get('camera_view', '')} |",
                f"| Subject Appearance | `{profile.recommended_coat_pair}` (est. `{profile.estimated_subject_count}` subject(s)) | {profile.match_rationale.get('coat_pair', '')} |",
                f"| Occluder / Headstage | `{profile.recommended_occluder}` / `{profile.recommended_headstage}` | {profile.match_rationale.get('occluder', '')} |",
                f"| Optics (Blur / Noise / Vignette) | blur `{profile.estimated_blur_sigma:.2f}`, noise `{profile.estimated_noise_sigma:.2f}`, vignette `{profile.vignette_strength:.2f}` | Used for photometric matching |",
                "",
            ]
        )

    if env_vs_sim_path:
        rel_env = Path(env_vs_sim_path).name
        md_lines.extend(
            [
                "### Environment Match Visual Comparison",
                "",
                f"![Real vs Matched 3D Simulation]({rel_env})",
                "",
            ]
        )

    md_lines.extend(
        [
            "## 2. Iteration Progression Summary",
            "",
        ]
    )
    if iterations:
        md_lines.extend(
            [
                "| Iter | Status | Renderer | Frames | Backend / GPU | Val mAP@50 | Real Coverage | Mean Conf | Count Match | Temp IoU | Sim2Real Score | Delta |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for it in iterations:
            ds = it.dataset or {}
            ev = it.evaluation or {}
            val_map = (
                ev.get("val_mask_map50")
                if ev.get("val_mask_map50") is not None
                else ev.get("val_box_map50")
            )
            delta = (
                f"{float(it.score_delta_vs_prev):+.3f}"
                if it.score_delta_vs_prev is not None
                else "—"
            )
            md_lines.append(
                f"| `{it.iteration}` | `{it.status}` | `{ds.get('renderer', '—')}` | "
                f"`{ds.get('total_frames', '—')}` | `{it.train_backend}/{it.accelerator}` | "
                f"`{_fmt_float(val_map)}` | `{_fmt_pct(ev.get('detection_coverage'))}` | "
                f"`{_fmt_float(ev.get('mean_confidence'))}` | `{_fmt_pct(ev.get('subject_count_match_rate'))}` | "
                f"`{_fmt_float(ev.get('temporal_iou_stability'))}` | **`{_fmt_float(ev.get('sim2real_score'))}`** | `{delta}` |"
            )
        md_lines.append("")

        for it in iterations:
            ev = it.evaluation or {}
            adj = it.adjustment or {}
            ds = it.dataset or {}
            md_lines.extend(
                [
                    f"### Iteration {it.iteration} (`{it.status}`)",
                    "",
                    f"- **Dataset**: `{ds.get('data_yaml_path', '—')}` "
                    f"(renderer=`{ds.get('renderer', '—')}`, classes=`{ds.get('class_names', [])}`)",
                    f"- **Training**: backend=`{it.train_backend}`, model=`{it.train_model}`, "
                    f"weights_in=`{it.train_weights_in or 'default'}`, epochs=`{it.train_epochs}`, "
                    f"imgsz=`{it.train_imgsz}`, job=`{it.colab_train_job_id or 'local'}`",
                    f"- **Checkpoint**: `{it.best_weights_path or '—'}`",
                ]
            )
            if ev.get("supervised_f1") is not None:
                md_lines.append(
                    f"- **Supervised validation**: mIoU=`{_fmt_float(ev.get('supervised_miou'))}`, "
                    f"Precision=`{_fmt_float(ev.get('supervised_precision'))}`, "
                    f"Recall=`{_fmt_float(ev.get('supervised_recall'))}`, "
                    f"F1=`{_fmt_float(ev.get('supervised_f1'))}`"
                )
            if adj.get("diagnosis"):
                md_lines.append("- **Diagnosis**:")
                for line in adj["diagnosis"]:
                    md_lines.append(f"  - {line}")
            if adj.get("recommendations"):
                md_lines.append("- **Next-iteration adjustments**:")
                for line in adj["recommendations"]:
                    md_lines.append(f"  - {line}")
            if it.error:
                md_lines.append(f"- **Error**: `{it.error}`")
            md_lines.append("")
    else:
        md_lines.append("_No training iterations have been recorded yet._\n")

    if latest_overlay:
        overlay_rel = Path(latest_overlay).name
        dst_overlay = rep_dir / overlay_rel
        if Path(latest_overlay).resolve() != dst_overlay.resolve():
            try:
                dst_overlay.write_bytes(Path(latest_overlay).read_bytes())
            except Exception:
                pass
        md_lines.extend(
            [
                "## 3. Real-Video Prediction Overlay Preview",
                "",
                f"![Latest Real Video Predictions]({overlay_rel})",
                "",
            ]
        )

    md_lines.extend(
        [
            "## 4. Reproducibility & Next Steps",
            "",
            "```bash",
            f"annolid-run sim2real status --project {proj_dir}",
            f"annolid-run sim2real iterate --project {proj_dir} --iterations 1 --gpu T4 --epochs 20",
            f"annolid-run sim2real report --project {proj_dir}",
            "```",
            "",
        ]
    )

    md_path = rep_dir / "sim2real_report.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    # ------------------------------------------------------------------ #
    # 3. Self-contained HTML report with embedded visual sheets & SVG
    # ------------------------------------------------------------------ #
    svg_chart = _build_inline_svg_chart(iterations)
    env_uri = _image_to_data_uri(env_vs_sim_path)
    ov_uri = _image_to_data_uri(latest_overlay)

    rows_html: list[str] = []
    for it in iterations:
        ds = it.dataset or {}
        ev = it.evaluation or {}
        val_map = (
            ev.get("val_mask_map50")
            if ev.get("val_mask_map50") is not None
            else ev.get("val_box_map50")
        )
        rows_html.append(
            "<tr>"
            f"<td>{it.iteration}</td>"
            f"<td>{html.escape(str(it.status))}</td>"
            f"<td>{html.escape(str(ds.get('renderer', '—')))}</td>"
            f"<td>{html.escape(str(ds.get('total_frames', '—')))}</td>"
            f"<td>{html.escape(_fmt_float(val_map))}</td>"
            f"<td>{html.escape(_fmt_pct(ev.get('detection_coverage')))}</td>"
            f"<td>{html.escape(_fmt_float(ev.get('mean_confidence')))}</td>"
            f"<td>{html.escape(_fmt_float(ev.get('temporal_iou_stability')))}</td>"
            f"<td><strong>{html.escape(_fmt_float(ev.get('sim2real_score')))}</strong></td>"
            "</tr>"
        )

    latest_recs = ""
    if iterations and iterations[-1].adjustment:
        adj = iterations[-1].adjustment or {}
        diag_items = "".join(
            f"<li>{html.escape(str(x))}</li>" for x in (adj.get("diagnosis") or [])
        )
        rec_items = "".join(
            f"<li>{html.escape(str(x))}</li>"
            for x in (adj.get("recommendations") or [])
        )
        latest_recs = (
            f"<h2>Latest Diagnosis &amp; Recommendations (Iter {iterations[-1].iteration})</h2>"
            f"<ul>{diag_items}{rec_items}</ul>"
        )

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>Annolid Sim-to-Real Report - {html.escape(state.project_id)}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #0b1120; color: #e2e8f0; max-width: 1080px; margin: 24px auto; padding: 0 20px; line-height: 1.5; }}
  h1, h2, h3 {{ color: #f8fafc; }}
  .cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin: 16px 0; }}
  .card {{ background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 14px; }}
  .card .k {{ color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
  .card .v {{ color: #38bdf8; font-size: 22px; font-weight: 700; margin-top: 4px; }}
  table {{ border-collapse: collapse; width: 100%; margin: 16px 0; background: #111827; border-radius: 8px; overflow: hidden; }}
  th, td {{ border: 1px solid #334155; padding: 8px 10px; text-align: left; font-size: 13px; }}
  th {{ background: #1e293b; color: #cbd5e1; }}
  img.sheet {{ width: 100%; border-radius: 8px; border: 1px solid #334155; margin: 12px 0; }}
  code {{ background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 12px; }}
</style>
</head>
<body>
  <h1>Annolid Sim-to-Real Report</h1>
  <p>Project <code>{html.escape(state.project_id)}</code> &middot; Generated <code>{html.escape(generated_at)}</code></p>
  <div class="cards">
    <div class="card"><div class="k">Species / Arena</div><div class="v">{html.escape(profile.species if profile else "—")} / {html.escape(profile.recommended_arena if profile else "—")}</div></div>
    <div class="card"><div class="k">Iterations</div><div class="v">{len(iterations)}</div></div>
    <div class="card"><div class="k">Best Sim2Real Score</div><div class="v">{html.escape(_fmt_float(state.best_score))}</div></div>
    <div class="card"><div class="k">Best Iteration</div><div class="v">{html.escape(str(state.best_iteration if state.best_iteration is not None else "—"))}</div></div>
  </div>
  {f"<h2>Iteration Metric Trend</h2>{svg_chart}" if svg_chart else ""}
  <h2>Iteration Comparison</h2>
  <table>
    <thead><tr><th>Iter</th><th>Status</th><th>Renderer</th><th>Frames</th><th>Val mAP@50</th><th>Real Coverage</th><th>Mean Conf</th><th>Temporal IoU</th><th>Sim2Real Score</th></tr></thead>
    <tbody>{"".join(rows_html) if rows_html else '<tr><td colspan="9">No iterations recorded yet.</td></tr>'}</tbody>
  </table>
  {latest_recs}
  {f'<h2>Real Example vs Matched 3D Simulation</h2><img class="sheet" src="{env_uri}" alt="Environment comparison"/>' if env_uri else ""}
  {f'<h2>Latest Real-Video Prediction Overlays</h2><img class="sheet" src="{ov_uri}" alt="Prediction overlays"/>' if ov_uri else ""}
</body>
</html>
"""
    html_path = rep_dir / "sim2real_report.html"
    html_path.write_text(html_doc, encoding="utf-8")

    return {
        "markdown": str(md_path),
        "json": str(json_path),
        "html": str(html_path),
        "env_vs_sim_contact_sheet": env_vs_sim_path,
        "latest_prediction_overlay": latest_overlay,
    }
