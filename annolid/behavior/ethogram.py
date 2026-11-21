"""BORIS-style multi-subject ethogram plotting (Matplotlib + deterministic SVG).

Supports both State intervals (bars) and Point events (diamond markers),
schema/category colors, time-window clipping, and standalone SVG export.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

from annolid.behavior.project_schema import ProjectSchema
from annolid.behavior.time_budget import BehaviorInterval, extract_behavior_intervals
from annolid.utils.atomic_io import atomic_write_text

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "create_ethogram",
    "render_ethogram_svg",
    "write_ethogram_svg",
]

_DEFAULT_SVG_COLORS = (
    "#2563eb",
    "#16a34a",
    "#ea580c",
    "#9333ea",
    "#0891b2",
    "#dc2626",
    "#ca8a04",
    "#4f46e5",
)


def _xml_escape(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _resolve_behavior_color_map(
    behaviors: Iterable[str],
    *,
    schema: Optional[ProjectSchema] = None,
    behavior_colors: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    if schema is not None:
        cat_map = schema.category_map()
        for beh in schema.behaviors:
            if beh.color:
                resolved[beh.code] = beh.color
            elif beh.category_id and beh.category_id in cat_map:
                cat = cat_map[beh.category_id]
                if cat.color:
                    resolved[beh.code] = cat.color
    if behavior_colors:
        resolved.update({str(k): str(v) for k, v in behavior_colors.items() if v})
    for idx, beh_name in enumerate(sorted(set(behaviors))):
        if beh_name not in resolved:
            resolved[beh_name] = _DEFAULT_SVG_COLORS[idx % len(_DEFAULT_SVG_COLORS)]
    return resolved


def create_ethogram(
    data: pd.DataFrame,
    subject_column: str = "Subject",
    behavior_column: str = "Behavior",
    start_column: str = "Recording time",
    end_column: str = "Recording time",
    duration_column: Optional[str] = None,
    subject_colors: Optional[Dict[str, str]] = None,
    event_column: str = "Event",
    behavior_colors: Optional[Dict[str, str]] = None,
    schema: Optional[ProjectSchema] = None,
):
    """Create a Matplotlib ethogram figure supporting State bars and Point diamonds."""
    import matplotlib
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    required_cols = [subject_column, behavior_column, start_column]
    if not all(col in data.columns for col in required_cols):
        raise ValueError(f"Data must contain columns: {', '.join(required_cols)}")

    if duration_column is not None and duration_column not in data.columns:
        raise ValueError(f"Data must contain duration column: {duration_column}")

    cmap = matplotlib.colormaps["Set1"]
    fig, ax = plt.subplots(figsize=(15, 8))

    # Automatically pair start/stop boundary rows when given an event log DataFrame
    if (
        duration_column is None
        and start_column == end_column
        and event_column in data.columns
    ):
        row_dicts = [
            {
                "subject": row[subject_column],
                "behavior": row[behavior_column],
                "recording time": row[start_column],
                "event": row[event_column],
            }
            for _, row in data.iterrows()
        ]
        intervals, _ = extract_behavior_intervals(row_dicts)
        grouped: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for iv in intervals:
            grouped[iv.subject][iv.behavior].append((iv.start, iv.end))
    else:
        grouped = defaultdict(lambda: defaultdict(list))
        for _, row in data.iterrows():
            subj = str(row[subject_column])
            beh = str(row[behavior_column])
            s_val = float(row[start_column])
            e_val = (
                s_val + float(row[duration_column])
                if duration_column is not None
                else float(row[end_column])
            )
            grouped[subj][beh].append((s_val, e_val))

    all_behaviors = [b for bmap in grouped.values() for b in bmap.keys()]
    resolved_beh_colors = _resolve_behavior_color_map(
        all_behaviors, schema=schema, behavior_colors=behavior_colors
    )

    y_ticks: List[int] = []
    y_labels: List[str] = []
    max_time = 1.0
    row_counter = 0

    for subj_idx, (subject, beh_dict) in enumerate(grouped.items()):
        for behavior, spans in beh_dict.items():
            y_pos = row_counter
            row_counter += 1
            y_ticks.append(y_pos)
            y_labels.append(f"{subject} - {behavior}")

            if behavior_colors is not None or schema is not None:
                color = resolved_beh_colors.get(behavior, "steelblue")
            elif isinstance(subject_colors, dict):
                color = subject_colors.get(subject, "gray")
            else:
                color = cmap(subj_idx % 9)

            for start, end in spans:
                max_time = max(max_time, end)
                if abs(end - start) <= 1e-9:
                    # Point event: plot visible diamond marker
                    ax.plot(
                        start,
                        y_pos,
                        marker="D",
                        color=color,
                        markeredgecolor="black",
                        markersize=7,
                    )
                else:
                    rect = patches.Rectangle(
                        (start, y_pos - 0.28),
                        end - start,
                        0.56,
                        linewidth=0.8,
                        edgecolor="black",
                        facecolor=color,
                    )
                    ax.add_patch(rect)

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylim(-0.6, max(0.6, len(y_ticks) - 0.4))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Subjects and Behaviors")
    ax.set_title("Behavioral Ethogram")
    ax.set_xlim(0, max_time * 1.05 + 0.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def render_ethogram_svg(
    intervals: Iterable[BehaviorInterval],
    *,
    schema: Optional[ProjectSchema] = None,
    behavior_colors: Optional[Dict[str, str]] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exclude_behaviors: Optional[Iterable[str]] = None,
    title: str = "Behavioral Ethogram",
    width: int = 920,
) -> str:
    """Render a publication-clean, deterministic SVG ethogram for State & Point events."""
    iv_list = list(intervals)
    if not iv_list:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} 140" width="{width}" height="140">'
            f'<rect width="{width}" height="140" fill="#ffffff"/>'
            f'<text x="{width / 2:.1f}" y="75" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#64748b">No behavior events to plot.</text>'
            "</svg>"
        )

    t0 = float(start_time) if start_time is not None else min(i.start for i in iv_list)
    t1 = float(end_time) if end_time is not None else max(i.end for i in iv_list)
    if t1 <= t0 + 1e-9:
        t1 = t0 + 1.0
    span = t1 - t0

    tracks: List[Tuple[str, str]] = sorted({(i.subject, i.behavior) for i in iv_list})
    color_map = _resolve_behavior_color_map(
        (b for _, b in tracks), schema=schema, behavior_colors=behavior_colors
    )
    excl_set = {str(x).strip() for x in (exclude_behaviors or ()) if str(x).strip()}

    left_pad = 190
    right_pad = 30
    top_pad = 48
    bottom_pad = 42
    row_h = 26
    plot_w = max(200, width - left_pad - right_pad)
    height = top_pad + len(tracks) * row_h + bottom_pad

    def _x_of(sec: float) -> float:
        clamped = min(max(sec, t0), t1)
        return left_pad + ((clamped - t0) / span) * plot_w

    svg: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{width / 2:.1f}" y="26" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="15" font-weight="bold" fill="#0f172a">{_xml_escape(title)}</text>',
    ]

    # Rows + gridlines
    track_idx = {pair: idx for idx, pair in enumerate(tracks)}
    for idx, (subj, beh) in enumerate(tracks):
        ry = top_pad + idx * row_h
        bg = "#f8fafc" if idx % 2 == 0 else "#ffffff"
        svg.append(
            f'<rect x="{left_pad}" y="{ry}" width="{plot_w}" height="{row_h}" fill="{bg}" stroke="#e2e8f0" stroke-width="0.6"/>'
        )
        lbl = f"{subj} · {beh}"
        if len(lbl) > 26:
            lbl = lbl[:24] + "…"
        svg.append(
            f'<text x="{left_pad - 8}" y="{ry + row_h * 0.65:.1f}" text-anchor="end" font-family="Helvetica, Arial, sans-serif" font-size="11" fill="#1e293b">{_xml_escape(lbl)}</text>'
        )

    # Intervals & point diamonds
    for iv in iv_list:
        idx = track_idx.get((iv.subject, iv.behavior))
        if idx is None:
            continue
        if iv.end < t0 - 1e-9 or iv.start > t1 + 1e-9:
            continue
        ry = top_pad + idx * row_h
        cy = ry + row_h / 2.0
        col = (
            "#94a3b8"
            if iv.behavior in excl_set
            else color_map.get(iv.behavior, "#2563eb")
        )
        if abs(iv.end - iv.start) <= 1e-9:
            px = _x_of(iv.start)
            d = 5.0
            svg.append(
                f'<polygon points="{px:.1f},{cy - d:.1f} {px + d:.1f},{cy:.1f} {px:.1f},{cy + d:.1f} {px - d:.1f},{cy:.1f}" '
                f'fill="{col}" stroke="#0f172a" stroke-width="0.9">'
                f"<title>{_xml_escape(iv.subject)} - {_xml_escape(iv.behavior)} @ {iv.start:.3f}s (POINT)</title></polygon>"
            )
        else:
            x0 = _x_of(max(t0, iv.start))
            x1 = _x_of(min(t1, iv.end))
            bw = max(1.8, x1 - x0)
            bh = row_h * 0.64
            opacity = "0.45" if iv.behavior in excl_set else "0.88"
            svg.append(
                f'<rect x="{x0:.1f}" y="{cy - bh / 2.0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="2.5" '
                f'fill="{col}" fill-opacity="{opacity}" stroke="#0f172a" stroke-opacity="0.35" stroke-width="0.6">'
                f"<title>{_xml_escape(iv.subject)} - {_xml_escape(iv.behavior)}: {iv.start:.3f}s–{iv.end:.3f}s ({iv.duration:.3f}s)</title></rect>"
            )

    # Time axis ticks
    axis_y = top_pad + len(tracks) * row_h
    svg.append(
        f'<line x1="{left_pad}" y1="{axis_y}" x2="{left_pad + plot_w}" y2="{axis_y}" stroke="#475569" stroke-width="1"/>'
    )
    n_ticks = 6
    for ti in range(n_ticks + 1):
        frac = ti / float(n_ticks)
        tx = left_pad + frac * plot_w
        t_val = t0 + frac * span
        svg.append(
            f'<line x1="{tx:.1f}" y1="{axis_y}" x2="{tx:.1f}" y2="{axis_y + 5}" stroke="#475569" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{tx:.1f}" y="{axis_y + 18}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#334155">{t_val:.1f}s</text>'
        )
    svg.append(
        f'<text x="{left_pad + plot_w / 2.0:.1f}" y="{height - 6}" text-anchor="middle" font-family="Helvetica, Arial, sans-serif" font-size="11" font-weight="600" fill="#334155">Time (s)</text>'
    )
    svg.append("</svg>")
    return "\n".join(svg)


def write_ethogram_svg(
    intervals: Iterable[BehaviorInterval],
    output_path: Path | str,
    **kwargs,
) -> Path:
    dst = Path(output_path)
    atomic_write_text(dst, render_ethogram_svg(intervals, **kwargs), encoding="utf-8")
    return dst
