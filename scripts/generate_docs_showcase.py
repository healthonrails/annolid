#!/usr/bin/env python3
"""Generate reproducible documentation & website showcase visuals for Annolid.

Produces:
  1. Real video + tracking visuals, animated tracking GIF, behavior storyboard,
     ethogram/kinematics dashboard, spatial zone analysis figure, and a live
     sanitized HTML5/SVG unified report from an Annolid-tracked video (for
     example, `mouse083_task1_annotator1.mp4` and its sibling JSON/CSV outputs).
  2. Offline Three.js 3D simulation captures via Playwright Chromium:
     - Two-Mouse 3D realistic open-field simulation
     - Two-Mouse 3D overhead 13-keypoint pose skeleton view
     - Fly-Brain LIF spiking connectome mode (`?fly_brain=1`)
     - FlyBody 3D simulation viewer
  3. Optional headless Qt screenshots of the Annolid desktop GUI and
     Analysis & Reporting Studio.
  4. Synced, web-optimized copies under `website/assets/images/showcase/`.

All external input paths are passed via CLI flags or environment variables so
no machine-local absolute paths are committed to the repository.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_SHOWCASE_DIR = REPO_ROOT / "docs" / "imgs" / "showcase"
DOCS_EXAMPLES_DIR = REPO_ROOT / "docs" / "assets" / "reports"
WEBSITE_SHOWCASE_DIR = REPO_ROOT / "website" / "assets" / "images" / "showcase"

# Canonical 7-keypoint mouse skeleton edges (suffixes after `<instance>_`)
MOUSE_SKELETON_EDGES: list[tuple[str, str]] = [
    ("nose", "left_ear"),
    ("nose", "right_ear"),
    ("left_ear", "neck"),
    ("right_ear", "neck"),
    ("nose", "neck"),
    ("neck", "left_hip"),
    ("neck", "right_hip"),
    ("left_hip", "tail_base"),
    ("right_hip", "tail_base"),
    ("neck", "tail_base"),
]

# Colors in BGR for OpenCV drawing, and Hex for Matplotlib
INSTANCE_BGR = {
    "resident": (200, 212, 0),  # crisp cyan/teal
    "intruder": (72, 146, 255),  # crisp warm amber/coral
}
INSTANCE_HEX = {
    "resident": "#0f9d9a",
    "intruder": "#ea6b2c",
}
BEHAVIOR_HEX = {
    "other": "#cbd5e1",
    "investigation": "#0ea5e9",
    "mount": "#f97316",
}
BEHAVIOR_BGR = {
    "other": (180, 160, 140),
    "investigation": (233, 165, 14),
    "mount": (22, 115, 249),
}

# Website showcase subset synced from docs/imgs/showcase/
WEBSITE_SYNC_FILES = [
    "mouse083_tracking_hero.png",
    "mouse083_tracking_demo.gif",
    "mouse083_behavior_storyboard.png",
    "mouse083_ethogram_kinematics.png",
    "mouse083_zone_analysis.png",
    "mouse083_report_preview.png",
    "mouse083_gui_workspace.png",
    "sim_two_mice_arena.png",
    "sim_two_mice_pose.png",
    "sim_fly_brain_connectome.png",
    "sim_flybody_3d.png",
    "sim_body_part_segmentation.png",
]


def _find_annolid_run(repo_root: Path = REPO_ROOT) -> list[str]:
    venv_cli = repo_root / ".venv" / "bin" / "annolid-run"
    if venv_cli.is_file():
        return [str(venv_cli)]
    which_cli = shutil.which("annolid-run")
    if which_cli:
        return [which_cli]
    return [sys.executable, "-m", "annolid.engine.cli"]


def sanitize_html_paths(html_text: str, prefixes: Sequence[str | Path]) -> str:
    """Scrub machine-local directory prefixes from generated HTML reports."""
    cleaned = html_text
    for prefix in prefixes:
        if not prefix:
            continue
        p_str = str(Path(prefix).resolve())
        for variant in (p_str + "/", p_str, str(prefix).rstrip("/") + "/", str(prefix)):
            cleaned = cleaned.replace(variant, "")
    # Catch any remaining `/Users/<name>/.../<filename>` paths in metadata labels
    cleaned = re.sub(
        r"(?:/Users/[A-Za-z0-9._-]+/[^\s\"'<>]*/|/home/[A-Za-z0-9._-]+/[^\s\"'<>]*/)",
        "",
        cleaned,
    )
    return cleaned


def _save_optimized_png(path: Path, max_width: int | None = None) -> None:
    """Re-save a PNG with PIL optimization (and optional downscale) for lean repo size."""
    img = Image.open(path)
    if max_width and img.width > max_width:
        ratio = max_width / float(img.width)
        img = img.resize(
            (max_width, int(round(img.height * ratio))), Image.Resampling.LANCZOS
        )
    img.save(path, format="PNG", optimize=True, compress_level=9)


def _read_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Could not read frame {frame_idx} from {video_path}")
    return frame


def _load_frame_json(
    annotations_dir: Path, stem: str, frame_idx: int
) -> dict[str, Any]:
    path = annotations_dir / f"{stem}_{frame_idx:09d}.json"
    if not path.is_file():
        return {"shapes": [], "flags": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _extract_behavior_flag(frame_data: dict[str, Any]) -> str:
    flags = frame_data.get("flags") or {}
    for preferred in ("mount", "investigation", "other"):
        if flags.get(preferred):
            return preferred
    for key, val in flags.items():
        if val:
            return str(key)
    return "unlabeled"


def draw_annotated_frame(
    frame_bgr: np.ndarray,
    frame_data: dict[str, Any],
    *,
    frame_idx: int,
    fps: float = 30.0,
    trails: dict[str, list[tuple[float, float]]] | None = None,
    pair_distance_px: float | None = None,
    banner_title: str | None = None,
    alert_text: str | None = None,
    show_hud: bool = True,
) -> np.ndarray:
    """Render clean publication-style polygons, keypoints, skeletons, trails, and HUD."""
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]
    overlay = canvas.copy()

    shapes = frame_data.get("shapes") or []
    keypoints: dict[str, dict[str, tuple[float, float]]] = {
        "resident": {},
        "intruder": {},
    }
    centroids: dict[str, tuple[float, float]] = {}

    # 1. Filled semi-transparent instance polygons
    for shape in shapes:
        label = str(shape.get("label") or "")
        stype = str(shape.get("shape_type") or "")
        pts = shape.get("points") or []
        if stype == "polygon" and len(pts) >= 3:
            color = INSTANCE_BGR.get(label, (200, 200, 200))
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(overlay, [arr], color)
            centroids[label] = (
                float(np.mean(arr[:, 0, 0])),
                float(np.mean(arr[:, 0, 1])),
            )
        elif stype == "point" and len(pts) >= 1:
            px, py = float(pts[0][0]), float(pts[0][1])
            for inst in ("resident", "intruder"):
                prefix = f"{inst}_"
                if label.startswith(prefix):
                    keypoints[inst][label[len(prefix) :]] = (px, py)

    cv2.addWeighted(overlay, 0.28, canvas, 0.72, 0.0, canvas)

    # 2. Recent motion trails
    if trails:
        for inst, pts in trails.items():
            if len(pts) < 2:
                continue
            color = INSTANCE_BGR.get(inst, (220, 220, 220))
            for i in range(1, len(pts)):
                alpha = 0.25 + 0.75 * (i / len(pts))
                c = tuple(int(v * alpha) for v in color)
                p0 = (int(round(pts[i - 1][0])), int(round(pts[i - 1][1])))
                p1 = (int(round(pts[i][0])), int(round(pts[i][1])))
                cv2.line(canvas, p0, p1, c, 2, cv2.LINE_AA)

    # 3. Crisp polygon contours + instance pills
    for shape in shapes:
        label = str(shape.get("label") or "")
        stype = str(shape.get("shape_type") or "")
        pts = shape.get("points") or []
        if stype == "polygon" and len(pts) >= 3:
            color = INSTANCE_BGR.get(label, (220, 220, 220))
            arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(canvas, [arr], True, color, 2, cv2.LINE_AA)
            min_x = int(np.min(arr[:, 0, 0]))
            min_y = max(26, int(np.min(arr[:, 0, 1])) - 10)
            text = label
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.56, 2)
            box_x1 = max(8, min(w - tw - 16, min_x))
            box_y1 = max(44 if show_hud else 8, min_y - th - 6)
            cv2.rectangle(
                canvas,
                (box_x1, box_y1),
                (box_x1 + tw + 12, box_y1 + th + 8),
                (16, 24, 38),
                -1,
            )
            cv2.rectangle(
                canvas,
                (box_x1, box_y1),
                (box_x1 + tw + 12, box_y1 + th + 8),
                color,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                text,
                (box_x1 + 6, box_y1 + th + 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.56,
                (245, 250, 255),
                1,
                cv2.LINE_AA,
            )

    # 4. Anatomical keypoint skeletons + bright landmark dots
    for inst, kp_map in keypoints.items():
        if not kp_map:
            continue
        color = INSTANCE_BGR.get(inst, (255, 255, 255))
        for a, b in MOUSE_SKELETON_EDGES:
            if a in kp_map and b in kp_map:
                p0 = (int(round(kp_map[a][0])), int(round(kp_map[a][1])))
                p1 = (int(round(kp_map[b][0])), int(round(kp_map[b][1])))
                cv2.line(canvas, p0, p1, (240, 248, 255), 1, cv2.LINE_AA)
        for kp_name, (kx, ky) in kp_map.items():
            pt = (int(round(kx)), int(round(ky)))
            r = 5 if kp_name == "nose" else 4
            cv2.circle(canvas, pt, r + 1, (15, 22, 34), -1, cv2.LINE_AA)
            dot_color = (90, 255, 255) if kp_name == "nose" else color
            cv2.circle(canvas, pt, r, dot_color, -1, cv2.LINE_AA)

    # 5. Optional QC alert badge
    if alert_text:
        (tw, th), _ = cv2.getTextSize(alert_text, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
        ax1, ay1 = 16, h - th - 26
        cv2.rectangle(
            canvas, (ax1, ay1), (ax1 + tw + 20, ay1 + th + 16), (30, 34, 150), -1
        )
        cv2.rectangle(
            canvas,
            (ax1, ay1),
            (ax1 + tw + 20, ay1 + th + 16),
            (95, 130, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            alert_text,
            (ax1 + 10, ay1 + th + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 6. Sleek top HUD bar
    if show_hud:
        hud = canvas.copy()
        cv2.rectangle(hud, (0, 0), (w, 38), (10, 18, 30), -1)
        cv2.addWeighted(hud, 0.80, canvas, 0.20, 0.0, canvas)
        behavior = _extract_behavior_flag(frame_data)
        sec = frame_idx / fps
        left_label = (
            banner_title
            or f"mouse083_task1_annotator1  |  Frame {frame_idx:04d} ({sec:05.1f}s)"
        )
        cv2.putText(
            canvas,
            left_label,
            (14, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (238, 246, 255),
            1,
            cv2.LINE_AA,
        )

        # Behavior pill on the right
        pill_text = f"Behavior: {behavior}"
        if pair_distance_px is not None and not np.isnan(pair_distance_px):
            pill_text += f"  |  Dist: {pair_distance_px:.0f} px"
        (pw, ph), _ = cv2.getTextSize(pill_text, cv2.FONT_HERSHEY_SIMPLEX, 0.54, 1)
        px1 = w - pw - 26
        pill_col = BEHAVIOR_BGR.get(behavior, (120, 140, 160))
        cv2.rectangle(canvas, (px1, 7), (w - 10, 31), pill_col, -1)
        text_col = (15, 23, 42) if behavior == "other" else (255, 255, 255)
        cv2.putText(
            canvas,
            pill_text,
            (px1 + 8, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            text_col,
            1,
            cv2.LINE_AA,
        )

    return canvas


def extract_session_metadata(
    annotations_dir: Path,
    stem: str,
    tracked_csv: Path,
    fps: float = 30.0,
) -> tuple[
    list[str], list[tuple[str, int, int, int]], pd.DataFrame, pd.Series, list[int]
]:
    """Read frame-level behavior flags, bouts, centroid tracks, pair distance, and missing frames."""
    df_tracked = pd.read_csv(tracked_csv)
    n_frames = int(df_tracked["frame_number"].max()) + 1

    behaviors: list[str] = []
    for i in range(n_frames):
        d = _load_frame_json(annotations_dir, stem, i)
        behaviors.append(_extract_behavior_flag(d))

    bouts: list[tuple[str, int, int, int]] = []
    if behaviors:
        cur_b, start = behaviors[0], 0
        for i, b in enumerate(behaviors[1:], 1):
            if b != cur_b:
                bouts.append((cur_b, start, i - 1, i - start))
                cur_b, start = b, i
        bouts.append((cur_b, start, len(behaviors) - 1, len(behaviors) - start))

    piv = df_tracked.pivot(
        index="frame_number", columns="instance_name", values=["cx", "cy"]
    )
    dist_series = pd.Series(dtype=float)
    if "resident" in piv["cx"].columns and "intruder" in piv["cx"].columns:
        dist_series = np.hypot(
            piv["cx"]["resident"] - piv["cx"]["intruder"],
            piv["cy"]["resident"] - piv["cy"]["intruder"],
        )

    res_frames = set(
        df_tracked[df_tracked["instance_name"] == "resident"]["frame_number"]
    )
    int_frames = set(
        df_tracked[df_tracked["instance_name"] == "intruder"]["frame_number"]
    )
    missing_int = sorted(res_frames - int_frames)
    return behaviors, bouts, df_tracked, dist_series, missing_int


def write_canonical_events_csv(
    bouts: Sequence[tuple[str, int, int, int]],
    out_csv: Path,
    *,
    fps: float = 30.0,
    subject: str = "resident",
) -> Path:
    """Write BORIS/Annolid-compatible behavior event CSV (`Recording time,Subject,Behavior,Event,Modifiers`)."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Recording time", "Subject", "Behavior", "Event", "Modifiers"])
        for behavior, start_f, end_f, _dur in bouts:
            if behavior not in {"investigation", "mount", "other"}:
                continue
            writer.writerow([f"{start_f / fps:.4f}", subject, behavior, "start", ""])
            writer.writerow([f"{(end_f + 1) / fps:.4f}", subject, behavior, "stop", ""])
    return out_csv


def write_demo_zones_json(
    out_json: Path, *, width: int = 1024, height: int = 570
) -> Path:
    """Create canonical Annolid zone definitions covering the mouse083 home-cage layout."""
    zones_payload = {
        "version": "5.1.1",
        "flags": {},
        "imagePath": "mouse083_task1_annotator1_000000000.png",
        "imageData": None,
        "imageHeight": height,
        "imageWidth": width,
        "shapes": [
            {
                "label": "left_nest_zone",
                "points": [[95.0, 95.0], [365.0, 95.0], [365.0, 530.0], [95.0, 530.0]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {
                    "semantic_type": "zone",
                    "zone_kind": "nest",
                    "zone_group": "arena_region",
                    "phase": "generic",
                    "occupant_role": "subject",
                    "access_state": "open",
                },
            },
            {
                "label": "center_arena",
                "points": [
                    [365.0, 95.0],
                    [645.0, 95.0],
                    [645.0, 530.0],
                    [365.0, 530.0],
                ],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {
                    "semantic_type": "zone",
                    "zone_kind": "arena",
                    "zone_group": "arena_region",
                    "phase": "generic",
                    "occupant_role": "subject",
                    "access_state": "open",
                },
            },
            {
                "label": "right_port_zone",
                "points": [
                    [645.0, 95.0],
                    [905.0, 95.0],
                    [905.0, 530.0],
                    [645.0, 530.0],
                ],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {
                    "semantic_type": "zone",
                    "zone_kind": "interaction",
                    "zone_group": "arena_region",
                    "phase": "generic",
                    "occupant_role": "subject",
                    "access_state": "open",
                },
            },
        ],
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(zones_payload, indent=2), encoding="utf-8")
    return out_json


def generate_video_showcase_assets(
    video_path: Path,
    annotations_dir: Path,
    tracked_csv: Path,
    *,
    docs_showcase_dir: Path = DOCS_SHOWCASE_DIR,
    docs_examples_dir: Path = DOCS_EXAMPLES_DIR,
) -> dict[str, Any]:
    """Build all screenshots, animated GIF, charts, and live HTML5 report from a tracked video."""
    docs_showcase_dir.mkdir(parents=True, exist_ok=True)
    docs_examples_dir.mkdir(parents=True, exist_ok=True)
    stem = video_path.stem

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 1024)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 570)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    behaviors, bouts, df_tracked, dist_series, missing_int = extract_session_metadata(
        annotations_dir, stem, tracked_csv, fps=fps
    )

    # Pre-index centroids by instance & frame for trail overlays
    inst_tracks: dict[str, dict[int, tuple[float, float]]] = {}
    for inst_name, grp in df_tracked.groupby("instance_name"):
        inst_tracks[str(inst_name)] = {
            int(r.frame_number): (float(r.cx), float(r.cy))
            for r in grp.itertuples(index=False)
        }

    def get_trails(
        frame_idx: int, window: int = 35
    ) -> dict[str, list[tuple[float, float]]]:
        out: dict[str, list[tuple[float, float]]] = {}
        for inst, mapping in inst_tracks.items():
            pts = [
                mapping[f]
                for f in range(max(0, frame_idx - window), frame_idx + 1)
                if f in mapping
            ]
            out[inst] = pts
        return out

    # 1. Hero annotated frame (frame 260 = crisp investigation interaction, or 45 if shorter)
    hero_idx = min(260, max(0, total_frames - 1))
    hero_frame = _read_video_frame(video_path, hero_idx)
    hero_json = _load_frame_json(annotations_dir, stem, hero_idx)
    hero_dist = (
        float(dist_series.get(hero_idx, np.nan))
        if hero_idx in dist_series.index
        else None
    )
    hero_vis = draw_annotated_frame(
        hero_frame,
        hero_json,
        frame_idx=hero_idx,
        fps=fps,
        trails=get_trails(hero_idx, window=45),
        pair_distance_px=hero_dist,
    )
    hero_path = docs_showcase_dir / "mouse083_tracking_hero.png"
    cv2.imwrite(str(hero_path), hero_vis)
    _save_optimized_png(hero_path)
    print(f"[showcase] Wrote {hero_path.relative_to(REPO_ROOT)}")

    # 2. Animated tracking loop GIF (frames 20..195 step 6 -> 30 frames @ 8fps = 3.8s compact loop)
    gif_frames_rgb: list[Image.Image] = []
    cap = cv2.VideoCapture(str(video_path))
    start_gif, end_gif, step_gif = 20, min(195, total_frames - 1), 6
    for fidx in range(start_gif, end_gif + 1, step_gif):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        fdata = _load_frame_json(annotations_dir, stem, fidx)
        fdist = (
            float(dist_series.get(fidx, np.nan)) if fidx in dist_series.index else None
        )
        drawn = draw_annotated_frame(
            frame,
            fdata,
            frame_idx=fidx,
            fps=fps,
            trails=get_trails(fidx, window=28),
            pair_distance_px=fdist,
        )
        resized = cv2.resize(
            drawn, (480, int(round(480 * height / width))), interpolation=cv2.INTER_AREA
        )
        gif_frames_rgb.append(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
    cap.release()

    gif_path = docs_showcase_dir / "mouse083_tracking_demo.gif"
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin and gif_frames_rgb:
        with tempfile.TemporaryDirectory(prefix="annolid-gif-") as tmp_dir:
            tmp = Path(tmp_dir)
            for idx, im in enumerate(gif_frames_rgb):
                im.save(tmp / f"f_{idx:04d}.png")
            palette = tmp / "palette.png"
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-framerate",
                    "8",
                    "-i",
                    str(tmp / "f_%04d.png"),
                    "-vf",
                    "hqdn3d=1.5:1.5:3:3,palettegen=max_colors=64:stats_mode=diff",
                    str(palette),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-framerate",
                    "8",
                    "-i",
                    str(tmp / "f_%04d.png"),
                    "-i",
                    str(palette),
                    "-lavfi",
                    "hqdn3d=1.5:1.5:3:3 [x]; [x][1:v] paletteuse=dither=none:diff_mode=rectangle",
                    "-loop",
                    "0",
                    str(gif_path),
                ],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
    elif gif_frames_rgb:
        gif_frames_rgb[0].save(
            gif_path,
            save_all=True,
            append_images=gif_frames_rgb[1:],
            duration=125,
            loop=0,
            optimize=True,
        )
    print(
        f"[showcase] Wrote {gif_path.relative_to(REPO_ROOT)} ({gif_path.stat().st_size / 1024:.0f} KB)"
    )

    # 3. 2x2 Behavior & Track QC Storyboard
    qc_frame = missing_int[1] if len(missing_int) > 1 else min(3362, total_frames - 1)
    storyboard_specs = [
        (0, "A. Seed Frame — Polygons + 14 Keypoints", None),
        (
            min(260, total_frames - 1),
            "B. Social Investigation — Nose-to-Body Tracking",
            None,
        ),
        (
            min(2720, total_frames - 1),
            "C. Mounting Bout — Close-Contact Identity Continuity",
            None,
        ),
        (
            qc_frame,
            "D. Track QC Alert — Occlusion Gap Detection",
            f"QC FLAG: intruder polygon missing (frames {missing_int[0]}-{missing_int[-1]})"
            if missing_int
            else None,
        ),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.4), dpi=130, facecolor="#0b1526")
    fig.subplots_adjust(
        left=0.015, right=0.985, top=0.89, bottom=0.02, wspace=0.04, hspace=0.15
    )
    fig.suptitle(
        "mouse083_task1_annotator1 — End-to-End Multi-Animal Tracking, Behavior & Track QC Storyboard",
        color="#f1f5f9",
        fontsize=13.5,
        fontweight="bold",
        y=0.96,
    )
    for ax, (fidx, title, alert) in zip(axes.flat, storyboard_specs):
        raw = _read_video_frame(video_path, fidx)
        fdata = _load_frame_json(annotations_dir, stem, fidx)
        fdist = (
            float(dist_series.get(fidx, np.nan)) if fidx in dist_series.index else None
        )
        vis = draw_annotated_frame(
            raw,
            fdata,
            frame_idx=fidx,
            fps=fps,
            trails=get_trails(fidx, window=24),
            pair_distance_px=fdist,
            alert_text=alert,
            show_hud=True,
        )
        ax.imshow(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB))
        ax.set_title(title, color="#e2e8f0", fontsize=10.5, fontweight="bold", pad=6)
        ax.axis("off")
    storyboard_path = docs_showcase_dir / "mouse083_behavior_storyboard.png"
    fig.savefig(storyboard_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    _save_optimized_png(storyboard_path, max_width=1440)
    print(f"[showcase] Wrote {storyboard_path.relative_to(REPO_ROOT)}")

    # 4. Ethogram + Inter-Animal Distance + Arena Trajectories Dashboard
    fig = plt.figure(figsize=(14.2, 8.6), dpi=135, facecolor="#ffffff")
    gs = fig.add_gridspec(
        3,
        2,
        height_ratios=[1.05, 1.15, 1.85],
        width_ratios=[1.3, 1.0],
        hspace=0.40,
        wspace=0.22,
        left=0.105,
        right=0.97,
        top=0.905,
        bottom=0.065,
    )
    fig.suptitle(
        "mouse083_task1_annotator1 — Ethogram, Social Proximity & Arena Kinematics (4,881 frames · 162.7 s @ 30 FPS)",
        fontsize=13.0,
        fontweight="bold",
        color="#0f2937",
        y=0.965,
    )

    # Row 1: Ethogram Raster
    ax_eth = fig.add_subplot(gs[0, :])
    behavior_rows = [("other", 0), ("investigation", 1), ("mount", 2)]
    for bname, row_y in behavior_rows:
        segs = [(s / fps, (e - s + 1) / fps) for b, s, e, _ in bouts if b == bname]
        if segs:
            ax_eth.broken_barh(
                segs,
                (row_y - 0.34, 0.68),
                facecolors=BEHAVIOR_HEX[bname],
                edgecolors="none",
            )
    if missing_int:
        qc_sec = missing_int[0] / fps
        ax_eth.axvline(qc_sec, color="#dc2626", linestyle="--", linewidth=1.6, zorder=5)
        ax_eth.text(
            qc_sec + 1.5,
            2.65,
            f"Track QC gap ({missing_int[0]}-{missing_int[-1]})",
            color="#b91c1c",
            fontsize=8.6,
            fontweight="bold",
            va="top",
        )
    ax_eth.set_yticks([0, 1, 2])
    ax_eth.set_yticklabels(
        ["other", "investigation", "mount"], fontsize=9.5, fontweight="bold"
    )
    ax_eth.set_xlim(0, total_frames / fps)
    ax_eth.set_ylim(-0.55, 2.80)
    ax_eth.set_title(
        "Scored Behavior Ethogram Timeline (from LabelMe frame flags)",
        fontsize=10.5,
        loc="left",
        fontweight="bold",
        color="#164757",
    )
    ax_eth.grid(axis="x", alpha=0.25, linestyle=":")
    ax_eth.spines[["top", "right"]].set_visible(False)

    # Row 2: Inter-animal distance over time
    ax_dist = fig.add_subplot(gs[1, :])
    t_sec = dist_series.index.to_numpy(dtype=float) / fps
    d_vals = dist_series.to_numpy(dtype=float)
    for b, s, e, _ in bouts:
        if b in ("investigation", "mount"):
            ax_dist.axvspan(
                s / fps, (e + 1) / fps, color=BEHAVIOR_HEX[b], alpha=0.14, lw=0
            )
    ax_dist.plot(
        t_sec,
        d_vals,
        color="#1e3a5f",
        linewidth=1.15,
        label="Resident ↔ Intruder centroid distance",
    )
    ax_dist.axhline(
        100.0,
        color="#ef4444",
        linestyle=":",
        linewidth=1.2,
        label="Close-contact threshold (100 px)",
    )
    ax_dist.set_xlim(0, total_frames / fps)
    ax_dist.set_ylabel("Distance (px)", fontsize=9.5)
    ax_dist.set_xlabel("Session Time (s)", fontsize=9.5)
    ax_dist.set_title(
        "Inter-Animal Centroid Proximity Aligned with Social Bouts",
        fontsize=10.5,
        loc="left",
        fontweight="bold",
        color="#164757",
    )
    ax_dist.legend(loc="upper right", frameon=True, fontsize=8.5)
    ax_dist.grid(alpha=0.25, linestyle=":")
    ax_dist.spines[["top", "right"]].set_visible(False)

    # Row 3 Left: Trajectories over actual background frame
    ax_traj = fig.add_subplot(gs[2, 0])
    bg = _read_video_frame(video_path, 0)
    bg_dim = (cv2.cvtColor(bg, cv2.COLOR_BGR2RGB).astype(np.float32) * 0.48).astype(
        np.uint8
    )
    ax_traj.imshow(bg_dim)
    for inst_name in ("resident", "intruder"):
        sub = df_tracked[df_tracked["instance_name"] == inst_name].sort_values(
            "frame_number"
        )
        ax_traj.plot(
            sub["cx"],
            sub["cy"],
            color=INSTANCE_HEX[inst_name],
            linewidth=1.15,
            alpha=0.88,
            label=f"{inst_name} ({len(sub)} frames)",
        )
        ax_traj.scatter(
            [sub["cx"].iloc[0]],
            [sub["cy"].iloc[0]],
            s=42,
            color=INSTANCE_HEX[inst_name],
            edgecolors="#ffffff",
            linewidths=1.2,
            zorder=5,
        )
    ax_traj.set_title(
        "Arena Trajectories (Thigmotaxis & Wall-Following)",
        fontsize=10.5,
        loc="left",
        fontweight="bold",
        color="#164757",
    )
    ax_traj.legend(
        loc="lower right",
        frameon=True,
        facecolor="#0f172a",
        labelcolor="#f8fafc",
        fontsize=8.5,
    )
    ax_traj.set_xticks([])
    ax_traj.set_yticks([])

    # Row 3 Right: Behavior Time Budget & Bout Counts
    ax_bar = fig.add_subplot(gs[2, 1])
    b_labels = ["other", "investigation", "mount"]
    b_secs = [sum(d for b, _, _, d in bouts if b == lbl) / fps for lbl in b_labels]
    b_pcts = [100.0 * s / (total_frames / fps) for s in b_secs]
    b_counts = [sum(1 for b, _, _, _ in bouts if b == lbl) for lbl in b_labels]
    y_pos = np.arange(len(b_labels))
    bars = ax_bar.barh(
        y_pos,
        b_secs,
        height=0.52,
        color=[BEHAVIOR_HEX[lbl] for lbl in b_labels],
        edgecolor="#334155",
        linewidth=0.7,
    )
    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(
        [f"{lbl}\n({cnt} bouts)" for lbl, cnt in zip(b_labels, b_counts)],
        fontsize=9.5,
        fontweight="bold",
    )
    ax_bar.invert_yaxis()
    ax_bar.set_xlabel("Total Duration (s)", fontsize=9.5)
    ax_bar.set_xlim(0, max(b_secs) * 1.32)
    ax_bar.set_title(
        "Behavior Time Budget & Bout Summary",
        fontsize=10.5,
        loc="left",
        fontweight="bold",
        color="#164757",
    )
    for bar, sec_val, pct_val in zip(bars, b_secs, b_pcts):
        ax_bar.text(
            sec_val + 1.8,
            bar.get_y() + bar.get_height() / 2,
            f"{sec_val:.1f} s ({pct_val:.1f}%)",
            va="center",
            fontsize=9.2,
            fontweight="bold",
            color="#0f2937",
        )
    ax_bar.grid(axis="x", alpha=0.25, linestyle=":")
    ax_bar.spines[["top", "right"]].set_visible(False)

    eth_kin_path = docs_showcase_dir / "mouse083_ethogram_kinematics.png"
    fig.savefig(eth_kin_path, facecolor=fig.get_facecolor())
    plt.close(fig)
    _save_optimized_png(eth_kin_path, max_width=1440)
    print(f"[showcase] Wrote {eth_kin_path.relative_to(REPO_ROOT)}")

    # 5. Real `annolid-run zone-analysis` & `annolid-run report`
    summary_metrics: dict[str, Any] = {}
    with tempfile.TemporaryDirectory(prefix="annolid-mouse083-cli-") as tmp_dir:
        tmp = Path(tmp_dir)
        events_csv = write_canonical_events_csv(
            bouts, tmp / "mouse083_events.csv", fps=fps
        )
        zones_json = write_demo_zones_json(
            tmp / "mouse083_zones.json", width=width, height=height
        )
        schema_json = annotations_dir / "project.annolid.json"
        report_dir = tmp / "report_out"

        env = dict(os.environ)
        env["MPLCONFIGDIR"] = str(tmp / "mpl")

        # Run real zone-analysis CLI
        zone_out_dir = tmp / "zone_out"
        zone_out_dir.mkdir(parents=True, exist_ok=True)
        zone_json_summary = tmp / "zone_summary.json"
        zone_cmd = [
            *_find_annolid_run(),
            "zone-analysis",
            str(tracked_csv),
            "--zone-file",
            str(zones_json),
            "--fps",
            str(fps),
            "--mode",
            "zone_metrics",
            "--output-dir",
            str(zone_out_dir),
            "--out-json",
            str(zone_json_summary),
        ]
        subprocess.run(zone_cmd, check=True, env=env, capture_output=True, text=True)
        zone_csv_files = sorted(zone_out_dir.glob("*.csv"))
        if not zone_csv_files:
            raise RuntimeError(
                f"Expected zone metrics CSV in {zone_out_dir}, found none."
            )
        df_zone = pd.read_csv(zone_csv_files[0])
        summary_metrics["zone_metrics"] = df_zone.to_dict(orient="records")

        # Build Zone Analysis Visual Figure directly from Annolid's computed zone metrics
        fig, (ax_zmap, ax_zbar) = plt.subplots(
            1,
            2,
            figsize=(13.8, 5.4),
            dpi=135,
            gridspec_kw={"width_ratios": [1.35, 1.0]},
            facecolor="#ffffff",
        )
        fig.subplots_adjust(left=0.03, right=0.97, top=0.86, bottom=0.12, wspace=0.18)
        fig.suptitle(
            "mouse083_task1_annotator1 — Spatial Zone Layout & Occupancy Analysis (annolid-run zone-analysis)",
            fontsize=12.5,
            fontweight="bold",
            color="#0f2937",
            y=0.96,
        )

        bg_z = (
            cv2.cvtColor(_read_video_frame(video_path, 0), cv2.COLOR_BGR2RGB).astype(
                np.float32
            )
            * 0.62
        ).astype(np.uint8)
        ax_zmap.imshow(bg_z)
        zone_colors = {
            "left_nest_zone": "#0ea5e9",
            "center_arena": "#10b981",
            "right_port_zone": "#f59e0b",
        }
        zone_data = json.loads(zones_json.read_text(encoding="utf-8"))
        for zshape in zone_data["shapes"]:
            zname = zshape["label"]
            pts = np.array(zshape["points"])
            poly = mpatches.Polygon(
                pts,
                closed=True,
                facecolor=zone_colors.get(zname, "#94a3b8"),
                edgecolor=zone_colors.get(zname, "#ffffff"),
                alpha=0.22,
                linewidth=2.2,
            )
            ax_zmap.add_patch(poly)
            poly_edge = mpatches.Polygon(
                pts,
                closed=True,
                facecolor="none",
                edgecolor=zone_colors.get(zname, "#ffffff"),
                linewidth=2.2,
                linestyle="--",
            )
            ax_zmap.add_patch(poly_edge)
            cx_z = float(np.mean(pts[:, 0]))
            ax_zmap.text(
                cx_z,
                122,
                zname,
                ha="center",
                va="top",
                fontsize=9.2,
                fontweight="bold",
                color="#ffffff",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "#0f172a",
                    "edgecolor": zone_colors[zname],
                    "alpha": 0.88,
                },
            )
        for inst_name in ("resident", "intruder"):
            sub = df_tracked[df_tracked["instance_name"] == inst_name].iloc[::6]
            ax_zmap.scatter(
                sub["cx"],
                sub["cy"],
                s=9,
                alpha=0.58,
                color=INSTANCE_HEX[inst_name],
                label=inst_name,
            )
        ax_zmap.set_title(
            "Arena Zone Polygons & Tracked Centroids",
            fontsize=10.5,
            loc="left",
            fontweight="bold",
            color="#164757",
        )
        ax_zmap.legend(
            loc="lower right",
            frameon=True,
            facecolor="#0f172a",
            labelcolor="#f8fafc",
            fontsize=8.8,
        )
        ax_zmap.set_xticks([])
        ax_zmap.set_yticks([])

        # Right panel: Zone occupancy seconds & entries from `df_zone`
        zone_names = ["left_nest_zone", "center_arena", "right_port_zone"]
        x_idx = np.arange(len(zone_names))
        bar_w = 0.34
        for offset, inst_name in [(-bar_w / 2, "resident"), (bar_w / 2, "intruder")]:
            row_df = df_zone[df_zone["instance_name"] == inst_name]
            occ_vals = []
            entry_vals = []
            for zn in zone_names:
                sec_col = f"occupancy_seconds__{zn}"
                ent_col = f"entry_count__{zn}"
                occ_vals.append(
                    float(row_df[sec_col].iloc[0])
                    if not row_df.empty and sec_col in row_df.columns
                    else 0.0
                )
                entry_vals.append(
                    int(row_df[ent_col].iloc[0])
                    if not row_df.empty and ent_col in row_df.columns
                    else 0
                )
            bars = ax_zbar.bar(
                x_idx + offset,
                occ_vals,
                width=bar_w,
                color=INSTANCE_HEX[inst_name],
                alpha=0.92,
                label=inst_name,
            )
            for b_rect, sec_v, ent_v in zip(bars, occ_vals, entry_vals):
                ax_zbar.text(
                    b_rect.get_x() + b_rect.get_width() / 2,
                    sec_v + 1.8,
                    f"{sec_v:.1f}s\n({ent_v}×)",
                    ha="center",
                    va="bottom",
                    fontsize=8.2,
                    fontweight="bold",
                    color="#0f2937",
                )
        ax_zbar.set_xticks(x_idx)
        ax_zbar.set_xticklabels(zone_names, fontsize=9.2, fontweight="bold")
        ax_zbar.set_ylabel("Occupancy Duration (s)", fontsize=9.5)
        ax_zbar.set_ylim(0, max(100.0, float(ax_zbar.get_ylim()[1]) * 1.24))
        ax_zbar.set_title(
            "Zone Occupancy Time & Entry Counts",
            fontsize=10.5,
            loc="left",
            fontweight="bold",
            color="#164757",
        )
        ax_zbar.legend(loc="upper right", frameon=True, fontsize=9.0)
        ax_zbar.grid(axis="y", alpha=0.25, linestyle=":")
        ax_zbar.spines[["top", "right"]].set_visible(False)

        zone_fig_path = docs_showcase_dir / "mouse083_zone_analysis.png"
        fig.savefig(zone_fig_path, facecolor=fig.get_facecolor())
        plt.close(fig)
        _save_optimized_png(zone_fig_path, max_width=1440)
        print(f"[showcase] Wrote {zone_fig_path.relative_to(REPO_ROOT)}")

        # Run real `annolid-run report` CLI
        report_cmd = [
            *_find_annolid_run(),
            "report",
            str(tracked_csv),
            "--events-csv",
            str(events_csv),
            "--zone-file",
            str(zones_json),
            "--fps",
            str(fps),
            "--title",
            "mouse083_task1_annotator1 — Resident-Intruder Social Assay",
            "--output-dir",
            str(report_dir),
            "--json",
        ]
        if schema_json.is_file():
            report_cmd.extend(["--schema", str(schema_json)])
        proc = subprocess.run(
            report_cmd, check=True, env=env, capture_output=True, text=True
        )
        # Extract JSON manifest from stdout (skip log lines if any)
        stdout_lines = proc.stdout.strip()
        json_start = stdout_lines.find("{")
        if json_start != -1:
            summary_metrics["report_cli"] = json.loads(stdout_lines[json_start:])

        raw_html = (report_dir / "analysis_report.html").read_text(encoding="utf-8")
        clean_html = sanitize_html_paths(
            raw_html,
            [video_path.parent, annotations_dir, tmp, REPO_ROOT],
        )
        example_html_path = docs_examples_dir / "mouse083_report.html"
        example_html_path.write_text(clean_html, encoding="utf-8")
        print(f"[showcase] Wrote {example_html_path.relative_to(REPO_ROOT)}")

        # Capture full-width rendered screenshot of the live HTML5/SVG report
        report_preview_path = docs_showcase_dir / "mouse083_report_preview.png"
        _capture_html_file_screenshot(
            example_html_path, report_preview_path, width=1440, height=900
        )
        _save_optimized_png(report_preview_path, max_width=1440)
        print(f"[showcase] Wrote {report_preview_path.relative_to(REPO_ROOT)}")

        # Save structured report summary JSON (sanitized) for reference
        raw_md = sanitize_html_paths(
            (report_dir / "analysis_report.md").read_text(encoding="utf-8"),
            [video_path.parent, annotations_dir, tmp, REPO_ROOT],
        )
        summary_metrics["report_markdown_preview"] = raw_md

    return summary_metrics


def _capture_html_file_screenshot(
    html_path: Path,
    out_png: Path,
    *,
    width: int = 1280,
    height: int = 840,
) -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, channel="chromium")
        except Exception:
            browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            viewport={"width": width, "height": height}, device_scale_factor=1.5
        )
        page.goto(html_path.resolve().as_uri(), timeout=60_000)
        page.wait_for_timeout(400)
        page.screenshot(path=str(out_png), full_page=False)
        browser.close()


def generate_simulation_screenshots(
    *,
    docs_showcase_dir: Path = DOCS_SHOWCASE_DIR,
    body_part_preview: Path | None = None,
) -> None:
    """Capture 100% offline Playwright screenshots of Annolid's bundled Three.js simulations."""
    from playwright.sync_api import sync_playwright

    from annolid.gui.threejs_examples import generate_threejs_example
    from annolid.gui.widgets.threejs_viewer_server import (
        _ensure_threejs_http_server,
        _register_threejs_http_model,
    )

    docs_showcase_dir.mkdir(parents=True, exist_ok=True)
    base = _ensure_threejs_http_server()

    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=True, channel="chromium")
        except Exception:
            browser = p.chromium.launch(headless=True)

        # 1. Two-Mouse Realistic Open Field (overview)
        page = browser.new_page(viewport={"width": 1280, "height": 740})
        page.goto(
            f"{base}/threejs/two_mice.html?dataset=1&seed=20260710", timeout=120_000
        )
        page.wait_for_selector("#c[data-scene-ready='true']", timeout=120_000)
        page.evaluate(
            "() => window.annolidPoseDataset && window.annolidPoseDataset.setFrame(2)"
        )
        page.wait_for_timeout(300)
        sim_arena = docs_showcase_dir / "sim_two_mice_arena.png"
        page.screenshot(path=str(sim_arena))
        _save_optimized_png(sim_arena)
        print(f"[showcase] Wrote {sim_arena.relative_to(REPO_ROOT)}")
        page.close()

        # 2. Two-Mouse Top-Down 13-Keypoint Pose Skeleton (live mode so lil-gui controls exist)
        page = browser.new_page(viewport={"width": 1280, "height": 740})
        page.goto(f"{base}/threejs/two_mice.html?seed=20260710", timeout=120_000)
        page.wait_for_selector("#c[data-scene-ready='true']", timeout=120_000)
        page.wait_for_function(
            "window.annolidTwoMice && window.annolidTwoMice.getDiagnostics().time > 0.6"
        )
        page.locator(".lil-gui.root > .title").click()
        page.wait_for_timeout(150)
        page.locator(".controller").filter(has_text="Pause Simulation").locator(
            "input"
        ).check()
        page.locator(".controller").filter(has_text="Show Pose Skeleton").locator(
            "input"
        ).check()
        page.locator(".controller").filter(has_text="Camera").locator(
            "select"
        ).select_option(label="Top down")
        # Collapse lil-gui so the pose skeleton stays unobstructed
        page.locator(".lil-gui.root > .title").click()
        page.wait_for_timeout(350)
        sim_pose = docs_showcase_dir / "sim_two_mice_pose.png"
        page.screenshot(path=str(sim_pose))
        _save_optimized_png(sim_pose)
        print(f"[showcase] Wrote {sim_pose.relative_to(REPO_ROOT)}")
        page.close()

        # 3. Fly-Brain LIF Spiking Connectome Mode (`?fly_brain=1`)
        page = browser.new_page(viewport={"width": 1280, "height": 760})
        page.goto(
            f"{base}/threejs/two_mice.html?fly_brain=1&seed=20260710", timeout=120_000
        )
        page.wait_for_selector("#c[data-scene-ready='true']", timeout=120_000)
        # Step through Walk -> Turn -> Moonwalk so the spike raster fills colorfully
        page.wait_for_timeout(1200)
        page.evaluate(
            """() => {
              if (window.annolidTwoMice && window.annolidTwoMice.applyFlyOptoPreset) {
                window.annolidTwoMice.applyFlyOptoPreset(1, 'walk');
              }
            }"""
        )
        page.wait_for_timeout(1400)
        page.evaluate(
            """() => {
              if (window.annolidTwoMice && window.annolidTwoMice.applyFlyOptoPreset) {
                window.annolidTwoMice.applyFlyOptoPreset(1, 'moonwalk');
              }
            }"""
        )
        page.wait_for_timeout(2000)
        sim_fb = docs_showcase_dir / "sim_fly_brain_connectome.png"
        page.screenshot(path=str(sim_fb))
        _save_optimized_png(sim_fb)
        print(f"[showcase] Wrote {sim_fb.relative_to(REPO_ROOT)}")
        page.close()

        # 4. FlyBody 3D Simulation Viewer (real Annolid Three.js viewer + assembled FlyBody OBJ mesh)
        with tempfile.TemporaryDirectory(prefix="annolid-flybody-sim-") as fb_tmp:
            fb_dir = Path(fb_tmp)
            fb_json = generate_threejs_example("flybody_simulation_json", fb_dir)
            model_url = _register_threejs_http_model(fb_json)
            viewer_html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <link rel="stylesheet" href="threejs/annolid_threejs_viewer.css" />
  <script type="importmap">
    {{
      "imports": {{
        "three": "https://esm.sh/three@0.182.0",
        "three/": "https://esm.sh/three@0.182.0/"
      }}
    }}
  </script>
  <script>
    window.__annolidThreeTitle = "FlyBody 3D Simulation";
    window.__annolidThreeModelUrl = {json.dumps(model_url)};
    window.__annolidThreeModelExt = ".json";
    window.__annolidEnableEyeControl = false;
    window.__annolidEnableHandControl = false;
    window.__annolidThreePickMode = "";
    window.__annolidThreeObjectRegionMap = {{}};
  </script>
</head>
<body>
  <div id="annolidThreeRoot"><canvas id="annolidThreeCanvas"></canvas></div>
  <div id="annolidThreeToolbar">
    <div class="tool-group">
      <button class="tool-btn" id="btnHome">Home</button>
      <button class="tool-btn" id="btnViewX">X</button>
      <button class="tool-btn" id="btnViewY">Y</button>
      <button class="tool-btn" id="btnViewZ">Z</button>
    </div>
  </div>
  <div id="annolidThreeIndicators">
    <div class="indicator" id="indRotate">Rotating</div>
    <div class="indicator" id="indPan">Panning</div>
    <div class="indicator" id="indZoom">Zooming</div>
  </div>
  <div id="annolidThreeTimeline" hidden>
    <button class="timeline-btn" id="btnSimulationPlay">Play</button>
    <input id="simulationFrameSlider" type="range" min="0" max="0" step="1" value="0" />
    <span id="simulationFrameLabel">Frame 0 / 0</span>
  </div>
  <div id="annolidThreeMeta" hidden></div>
  <div id="annolidThreeFlybodyControls" hidden></div>
  <div id="annolidThreeLegend" hidden></div>
  <div id="annolidThreeCategoryPanel" hidden></div>
  <div id="annolidThreeVolumePanel" hidden></div>
  <div id="annolidThreeStatus">Loading FlyBody simulation…</div>
  <div id="annolidThreeHints">Drag: rotate | Wheel: zoom | Play: animate frames</div>
  <script type="module" src="threejs/annolid_threejs_viewer.js"></script>
</body>
</html>"""
            page = browser.new_page(viewport={"width": 1280, "height": 720})
            preview_url = f"{base}/flybody_preview.html"
            page.route(
                preview_url,
                lambda route: route.fulfill(
                    status=200,
                    content_type="text/html; charset=utf-8",
                    body=viewer_html,
                ),
            )
            page.goto(preview_url, timeout=120_000)
            try:
                page.wait_for_function(
                    """() => {
                      const el = document.getElementById('annolidThreeStatus');
                      const txt = el ? (el.textContent || '') : '';
                      return txt && !txt.includes('Loading');
                    }""",
                    timeout=35_000,
                )
            except Exception:
                pass
            page.wait_for_timeout(1500)
            sim_flybody = docs_showcase_dir / "sim_flybody_3d.png"
            page.screenshot(path=str(sim_flybody))
            _save_optimized_png(sim_flybody)
            print(f"[showcase] Wrote {sim_flybody.relative_to(REPO_ROOT)}")
            page.close()

        browser.close()

    # 5. Optional body-part segmentation ground-truth vs predicted benchmark image
    default_bp = (
        REPO_ROOT
        / "output"
        / "two_mice_body_part_eval"
        / "calms_resident_intruder"
        / "calms_resident_intruder_midframe_preview.png"
    )
    bp_src = (
        body_part_preview
        if (body_part_preview and body_part_preview.is_file())
        else default_bp
    )
    if bp_src.is_file():
        bp_dst = docs_showcase_dir / "sim_body_part_segmentation.png"
        shutil.copy2(bp_src, bp_dst)
        _save_optimized_png(bp_dst, max_width=1440)
        print(f"[showcase] Wrote {bp_dst.relative_to(REPO_ROOT)}")


def generate_gui_screenshot(
    video_path: Path,
    annotations_dir: Path,
    *,
    frame_idx: int = 260,
    docs_showcase_dir: Path = DOCS_SHOWCASE_DIR,
) -> Path | None:
    """Capture a real offscreen Qt screenshot of Annolid's desktop GUI on an annotated frame."""
    docs_showcase_dir.mkdir(parents=True, exist_ok=True)
    out_png = docs_showcase_dir / "mouse083_gui_workspace.png"
    stem = video_path.stem

    with tempfile.TemporaryDirectory(prefix="annolid-gui-capture-") as tmp_dir:
        tmp = Path(tmp_dir)
        frame_img_path = tmp / f"{stem}_{frame_idx:09d}.png"
        frame_json_path = tmp / f"{stem}_{frame_idx:09d}.json"
        raw = _read_video_frame(video_path, frame_idx)
        cv2.imwrite(str(frame_img_path), raw)

        fdata = _load_frame_json(annotations_dir, stem, frame_idx)
        fdata["imagePath"] = frame_img_path.name
        fdata["imageData"] = None
        frame_json_path.write_text(json.dumps(fdata, indent=2), encoding="utf-8")

        helper_code = f"""
import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from qtpy import QtCore, QtWidgets
from annolid.gui.app import AnnolidWindow
from annolid.gui.theme import apply_dark_theme

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
try:
    apply_dark_theme(app)
except Exception:
    pass
win = AnnolidWindow()
win.setFixedSize(1440, 840)
win.setGeometry(0, 0, 1440, 840)
win.show()
app.processEvents()
win.loadFile({json.dumps(str(frame_img_path))})
for _ in range(12):
    app.processEvents()
# Constrain right docks so the video canvas occupies the main widescreen area
right_docks = [
    d for d in win.findChildren(QtWidgets.QDockWidget)
    if d.isVisible() and win.dockWidgetArea(d) == QtCore.Qt.RightDockWidgetArea
]
if right_docks:
    try:
        win.resizeDocks(right_docks, [330] * len(right_docks), QtCore.Qt.Horizontal)
    except Exception:
        pass
try:
    win.canvas.deSelectShape()
except Exception:
    pass
try:
    win.adjustScale(initial=True)
except Exception:
    pass
for _ in range(10):
    app.processEvents()
pm = win.grab()
pm.save({json.dumps(str(out_png))})
win.close()
app.quit()
"""
        env = dict(os.environ)
        env["QT_QPA_PLATFORM"] = "offscreen"
        proc = subprocess.run(
            [sys.executable, "-c", helper_code],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=90,
        )
        if proc.returncode != 0 or not out_png.is_file():
            print(
                f"[showcase] Warning: offscreen GUI capture skipped (exit={proc.returncode}): "
                f"{(proc.stderr or proc.stdout).strip()[:300]}",
                file=sys.stderr,
            )
            return None

    _save_optimized_png(out_png, max_width=1440)
    print(f"[showcase] Wrote {out_png.relative_to(REPO_ROOT)}")
    return out_png


def sync_website_showcase(
    docs_showcase_dir: Path = DOCS_SHOWCASE_DIR,
    website_showcase_dir: Path = WEBSITE_SHOWCASE_DIR,
) -> list[Path]:
    """Copy curated showcase media from `docs/imgs/showcase/` into `website/assets/images/showcase/`."""
    website_showcase_dir.mkdir(parents=True, exist_ok=True)
    synced: list[Path] = []
    for name in WEBSITE_SYNC_FILES:
        src = docs_showcase_dir / name
        if not src.is_file():
            continue
        dst = website_showcase_dir / name
        shutil.copy2(src, dst)
        synced.append(dst)
    print(
        f"[showcase] Synced {len(synced)} showcase asset(s) into {website_showcase_dir.relative_to(REPO_ROOT)}"
    )
    return synced


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate reproducible docs & website showcase media for Annolid."
    )
    parser.add_argument(
        "--video",
        type=Path,
        help="Path to an Annolid-tracked MP4 video (e.g. mouse083_task1_annotator1.mp4).",
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Path to frame LabelMe JSON folder (defaults to <video_dir>/<video_stem>).",
    )
    parser.add_argument(
        "--tracked-csv",
        type=Path,
        help="Path to *_tracked.csv file (defaults to <video_dir>/<video_stem>_tracked.csv).",
    )
    parser.add_argument(
        "--body-part-preview",
        type=Path,
        help="Optional ground-truth vs predicted body-part segmentation PNG.",
    )
    parser.add_argument(
        "--skip-video",
        action="store_true",
        help="Skip video/tracking asset generation.",
    )
    parser.add_argument(
        "--skip-simulations",
        action="store_true",
        help="Skip Playwright 3D simulation captures.",
    )
    parser.add_argument(
        "--skip-gui",
        action="store_true",
        help="Skip offscreen Qt GUI screenshot capture.",
    )
    parser.add_argument(
        "--skip-website-sync",
        action="store_true",
        help="Do not sync assets into website/assets/.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.skip_video and args.video:
        video_path = args.video.expanduser().resolve()
        if not video_path.is_file():
            raise FileNotFoundError(f"Video file not found: {video_path}")
        annotations_dir = (
            args.annotations.expanduser().resolve()
            if args.annotations
            else video_path.parent / video_path.stem
        )
        tracked_csv = (
            args.tracked_csv.expanduser().resolve()
            if args.tracked_csv
            else video_path.parent / f"{video_path.stem}_tracked.csv"
        )
        summary = generate_video_showcase_assets(
            video_path=video_path,
            annotations_dir=annotations_dir,
            tracked_csv=tracked_csv,
        )
        if not args.skip_gui:
            generate_gui_screenshot(
                video_path=video_path, annotations_dir=annotations_dir
            )
        print("[showcase] Report Markdown preview:")
        print(summary.get("report_markdown_preview", "").strip())

    if not args.skip_simulations:
        generate_simulation_screenshots(body_part_preview=args.body_part_preview)

    if not args.skip_website_sync:
        sync_website_showcase()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
