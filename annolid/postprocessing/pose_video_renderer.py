"""Deterministic offline labeled skeleton video renderer (DeepLabCut / SLEAP equivalent).

Renders pose predictions or ground-truth annotations from any Annolid-supported
format (LabelMe dir / AnnotationStore, SLEAP ``.slp``/``.analysis.h5``,
DeepLabCut CSV/H5, COCO JSON, or Annolid tracking CSV) onto a source video with:

- ``PoseSchema`` skeleton bones (solid for visible, dashed for occluded),
- Color modes: ``instance`` (per-animal), ``bodypart`` (DLC-style per-keypoint),
  and ``confidence`` (continuous score heatmap),
- ``p-cutoff`` confidence filtering,
- Fading temporal keypoint motion trails,
- Optional comparison overlay (e.g., raw vs. filtered pose),
- Instance labels, keypoint names, and frame/timestamp HUD.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.annotation.pose_interop.converters import read_pose_dataset
from annolid.annotation.pose_interop.model import PoseDataset, PoseFrame
from annolid.annotation.pose_schema import PoseSchema, find_pose_schema_in_dir
from annolid.utils.draw import (
    SKELETON_INSTANCE_COLORS_BGR,
    _draw_dashed_line_cv2,
)

__all__ = [
    "PoseVideoRenderConfig",
    "PoseVideoRenderResult",
    "color_for_bodypart",
    "color_for_confidence",
    "draw_pose_frame_overlay",
    "render_pose_video",
]

# Vivid distinct BGR palette for DLC-style per-bodypart coloring.
_BODYPART_PALETTE_BGR: Tuple[Tuple[int, int, int], ...] = (
    (66, 133, 244),
    (52, 168, 83),
    (251, 188, 5),
    (234, 67, 53),
    (171, 71, 188),
    (0, 172, 193),
    (255, 112, 67),
    (156, 204, 101),
    (66, 165, 245),
    (236, 64, 122),
    (38, 166, 154),
    (255, 202, 40),
    (126, 87, 194),
    (102, 187, 106),
    (239, 83, 80),
    (41, 182, 246),
    (212, 225, 87),
)


@dataclass(frozen=True)
class PoseVideoRenderConfig:
    """Configuration for offline labeled skeleton video rendering."""

    color_mode: str = "instance"  # "instance" | "bodypart" | "confidence"
    min_confidence: Optional[float] = None
    thickness: int = 2
    point_radius: int = 4
    trail_length: int = 0
    draw_labels: bool = False
    draw_instance_names: bool = True
    draw_frame_index: bool = True
    max_bone_dist: Optional[float] = None
    start_frame: int = 0
    end_frame: Optional[int] = None
    fps: Optional[float] = None
    codec: str = "mp4v"


@dataclass(frozen=True)
class PoseVideoRenderResult:
    """Summary returned after rendering a labeled skeleton video."""

    video_path: str
    pose_source: str
    output_video_path: str
    frames_rendered: int
    frames_with_pose: int
    width: int
    height: int
    fps: float
    color_mode: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_path": self.video_path,
            "pose_source": self.pose_source,
            "output_video_path": self.output_video_path,
            "frames_rendered": int(self.frames_rendered),
            "frames_with_pose": int(self.frames_with_pose),
            "width": int(self.width),
            "height": int(self.height),
            "fps": round(float(self.fps), 4),
            "color_mode": self.color_mode,
        }


def color_for_bodypart(index: int) -> Tuple[int, int, int]:
    return _BODYPART_PALETTE_BGR[int(index) % len(_BODYPART_PALETTE_BGR)]


def color_for_confidence(score: Optional[float]) -> Tuple[int, int, int]:
    """Map confidence in [0, 1] to BGR red -> amber -> emerald green."""
    if score is None or not math.isfinite(float(score)):
        return (235, 200, 50)  # cyan-blue for ground-truth / unscored points
    s = max(0.0, min(1.0, float(score)))
    if s < 0.5:
        t = s / 0.5
        # Red (40, 55, 235) -> Amber (30, 195, 245)
        b = int(round(40 + (30 - 40) * t))
        g = int(round(55 + (195 - 55) * t))
        r = int(round(235 + (245 - 235) * t))
        return (b, g, r)
    t = (s - 0.5) / 0.5
    # Amber (30, 195, 245) -> Emerald (90, 215, 65)
    b = int(round(30 + (90 - 30) * t))
    g = int(round(195 + (215 - 195) * t))
    r = int(round(245 + (65 - 245) * t))
    return (b, g, r)


def draw_pose_frame_overlay(
    frame_bgr: np.ndarray,
    frame_pose: Optional[PoseFrame],
    *,
    edges: Sequence[Tuple[str, str]],
    keypoint_index: Mapping[str, int],
    instance_index: Mapping[str, int],
    config: PoseVideoRenderConfig,
    trails: Optional[Dict[Tuple[str, str], Deque[Tuple[int, int]]]] = None,
    comparison_pose: Optional[PoseFrame] = None,
    frame_number: Optional[int] = None,
    fps: float = 30.0,
) -> np.ndarray:
    """Render skeleton bones, keypoints, trails, and HUD labels onto ``frame_bgr``."""
    canvas = frame_bgr.copy()
    mode = str(config.color_mode or "instance").strip().lower()
    if mode not in {"instance", "bodypart", "confidence"}:
        mode = "instance"

    # Optional comparison overlay (e.g., unfiltered raw predictions as subtle hollow markers).
    if comparison_pose is not None:
        for inst in comparison_pose.instances:
            for pt in inst.keypoints.values():
                if not pt.is_valid or int(pt.visibility) <= 0:
                    continue
                if (
                    config.min_confidence is not None
                    and pt.score is not None
                    and float(pt.score) < float(config.min_confidence)
                ):
                    continue
                cx, cy = int(round(pt.x)), int(round(pt.y))
                cv2.circle(
                    canvas,
                    (cx, cy),
                    max(2, config.point_radius + 1),
                    (180, 180, 180),
                    1,
                    lineType=cv2.LINE_AA,
                )

    if frame_pose is not None:
        for i_fallback, inst in enumerate(frame_pose.instances):
            inst_name = str(inst.instance_name or f"instance_{i_fallback}")
            inst_idx = instance_index.get(inst_name, i_fallback)
            inst_color = SKELETON_INSTANCE_COLORS_BGR[
                inst_idx % len(SKELETON_INSTANCE_COLORS_BGR)
            ]

            parsed: Dict[str, Tuple[int, int, int, Optional[float]]] = {}
            for kp_name, pt in inst.keypoints.items():
                if not pt.is_valid or int(pt.visibility) <= 0:
                    continue
                if (
                    config.min_confidence is not None
                    and pt.score is not None
                    and math.isfinite(float(pt.score))
                    and float(pt.score) < float(config.min_confidence)
                ):
                    continue
                px, py = int(round(pt.x)), int(round(pt.y))
                vis = (
                    int(KeypointVisibility.OCCLUDED)
                    if int(pt.visibility) == int(KeypointVisibility.OCCLUDED)
                    else int(KeypointVisibility.VISIBLE)
                )
                parsed[kp_name] = (px, py, vis, pt.score)

                if trails is not None and config.trail_length > 1:
                    buf = trails.setdefault(
                        (inst_name, kp_name),
                        deque(maxlen=int(config.trail_length)),
                    )
                    buf.append((px, py))

            # 1. Draw fading motion trails underneath skeleton.
            if trails is not None and config.trail_length > 1:
                for kp_name in parsed:
                    buf = trails.get((inst_name, kp_name))
                    if not buf or len(buf) < 2:
                        continue
                    base_col = (
                        color_for_bodypart(keypoint_index.get(kp_name, 0))
                        if mode == "bodypart"
                        else inst_color
                    )
                    pts_list = list(buf)
                    n_seg = len(pts_list) - 1
                    for s_i in range(n_seg):
                        frac = float(s_i + 1) / float(n_seg)
                        seg_col = tuple(int(c * (0.25 + 0.75 * frac)) for c in base_col)
                        cv2.line(
                            canvas,
                            pts_list[s_i],
                            pts_list[s_i + 1],
                            seg_col,
                            max(1, int(round(config.thickness * frac))),
                            lineType=cv2.LINE_AA,
                        )

            # 2. Draw skeleton bones.
            for a, b in edges:
                pa = parsed.get(a)
                pb = parsed.get(b)
                if pa is None or pb is None:
                    continue
                if config.max_bone_dist is not None and math.hypot(
                    pb[0] - pa[0], pb[1] - pa[1]
                ) > float(config.max_bone_dist):
                    continue
                if mode == "bodypart":
                    bone_color = color_for_bodypart(keypoint_index.get(a, 0))
                elif mode == "confidence":
                    scores = [s for s in (pa[3], pb[3]) if s is not None]
                    bone_color = color_for_confidence(min(scores) if scores else None)
                else:
                    bone_color = inst_color

                occluded = (pa[2] == int(KeypointVisibility.OCCLUDED)) or (
                    pb[2] == int(KeypointVisibility.OCCLUDED)
                )
                p0 = (pa[0], pa[1])
                p1 = (pb[0], pb[1])
                if occluded:
                    _draw_dashed_line_cv2(
                        canvas, p0, p1, (20, 20, 20), thickness=config.thickness + 2
                    )
                    _draw_dashed_line_cv2(
                        canvas, p0, p1, bone_color, thickness=config.thickness
                    )
                else:
                    cv2.line(
                        canvas,
                        p0,
                        p1,
                        (20, 20, 20),
                        config.thickness + 2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.line(
                        canvas,
                        p0,
                        p1,
                        bone_color,
                        config.thickness,
                        lineType=cv2.LINE_AA,
                    )

            # 3. Draw keypoints on top of bones.
            for kp_name, (px, py, vis, score) in parsed.items():
                if mode == "bodypart":
                    pt_color = color_for_bodypart(keypoint_index.get(kp_name, 0))
                elif mode == "confidence":
                    pt_color = color_for_confidence(score)
                else:
                    pt_color = inst_color

                if vis == int(KeypointVisibility.OCCLUDED):
                    cv2.circle(
                        canvas,
                        (px, py),
                        config.point_radius + 1,
                        (20, 20, 20),
                        2,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.circle(
                        canvas,
                        (px, py),
                        config.point_radius,
                        pt_color,
                        2,
                        lineType=cv2.LINE_AA,
                    )
                else:
                    cv2.circle(
                        canvas,
                        (px, py),
                        config.point_radius + 1,
                        (20, 20, 20),
                        -1,
                        lineType=cv2.LINE_AA,
                    )
                    cv2.circle(
                        canvas,
                        (px, py),
                        config.point_radius,
                        pt_color,
                        -1,
                        lineType=cv2.LINE_AA,
                    )

                if config.draw_labels:
                    txt = (
                        f"{kp_name} ({score:.2f})"
                        if (mode == "confidence" and score is not None)
                        else kp_name
                    )
                    origin = (px + config.point_radius + 4, py - 2)
                    cv2.putText(
                        canvas,
                        txt,
                        origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (18, 18, 18),
                        3,
                        cv2.LINE_AA,
                    )
                    cv2.putText(
                        canvas,
                        txt,
                        origin,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        pt_color,
                        1,
                        cv2.LINE_AA,
                    )

            # 4. Instance name badge above skeleton bounding box.
            if config.draw_instance_names and parsed:
                min_x = min(p[0] for p in parsed.values())
                min_y = min(p[1] for p in parsed.values())
                badge_origin = (max(4, min_x), max(16, min_y - 10))
                cv2.putText(
                    canvas,
                    inst_name,
                    badge_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (15, 15, 15),
                    3,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    canvas,
                    inst_name,
                    badge_origin,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    inst_color,
                    1,
                    cv2.LINE_AA,
                )

    if config.draw_frame_index and frame_number is not None:
        sec = float(frame_number) / max(1e-3, float(fps))
        hud = f"Frame {int(frame_number):06d}  ({sec:.2f}s)"
        cv2.putText(
            canvas,
            hud,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (15, 15, 15),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            hud,
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (245, 245, 245),
            1,
            cv2.LINE_AA,
        )

    return canvas


def render_pose_video(
    video_path: str | Path,
    pose_source: str | Path | PoseDataset,
    output_video_path: str | Path,
    *,
    source_format: str = "auto",
    pose_schema: Optional[PoseSchema | str | Path] = None,
    comparison_source: Optional[str | Path | PoseDataset] = None,
    config: Optional[PoseVideoRenderConfig] = None,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> PoseVideoRenderResult:
    """Render an offline labeled skeleton overlay video from any Annolid pose source."""
    vpath = Path(video_path).expanduser().resolve()
    if not vpath.is_file():
        raise FileNotFoundError(f"Video file not found: {vpath}")
    out_path = Path(output_video_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = config or PoseVideoRenderConfig()

    if isinstance(pose_source, PoseDataset):
        dataset = pose_source
        pose_label = "<PoseDataset>"
        pose_path_obj: Optional[Path] = None
    else:
        pose_path_obj = Path(pose_source).expanduser().resolve()
        pose_label = str(pose_path_obj)
        dataset = read_pose_dataset(
            pose_path_obj,
            source_format=source_format,
            video_path=vpath,
            pose_schema=pose_schema,
        )

    comp_by_frame: Dict[int, PoseFrame] = {}
    if comparison_source is not None:
        comp_ds = (
            comparison_source
            if isinstance(comparison_source, PoseDataset)
            else read_pose_dataset(
                comparison_source, video_path=vpath, pose_schema=pose_schema
            )
        )
        for cf in comp_ds.frames:
            comp_by_frame[int(cf.frame_index)] = cf

    if isinstance(pose_schema, PoseSchema):
        schema = pose_schema
    elif pose_schema is not None:
        schema = PoseSchema.load(pose_schema)
    elif pose_path_obj is not None:
        found = find_pose_schema_in_dir(pose_path_obj, search_parent=True)
        schema = (
            PoseSchema.load(found)
            if found is not None
            else dataset.to_pose_schema(preset="auto")
        )
    else:
        schema = dataset.to_pose_schema(preset="auto")

    edges = list(schema.base_edges())
    kp_names = list(schema.keypoints or dataset.resolved_keypoints())
    kp_index = {name: i for i, name in enumerate(kp_names)}
    inst_names = list(dataset.resolved_instances())
    inst_index = {name: i for i, name in enumerate(inst_names)}

    pose_by_frame: Dict[int, PoseFrame] = {
        int(f.frame_index): f for f in dataset.frames
    }

    cap = cv2.VideoCapture(str(vpath))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {vpath}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if not math.isfinite(src_fps) or src_fps <= 0.0:
        src_fps = 30.0
    out_fps = float(cfg.fps) if cfg.fps is not None and cfg.fps > 0 else src_fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    start_f = max(0, int(cfg.start_frame))
    end_f = (
        min(total_frames - 1, int(cfg.end_frame))
        if (cfg.end_frame is not None and total_frames > 0)
        else (total_frames - 1 if total_frames > 0 else None)
    )

    fourcc = cv2.VideoWriter_fourcc(*str(cfg.codec or "mp4v")[:4])
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Unable to open video writer for: {out_path}")

    trails: Dict[Tuple[str, str], Deque[Tuple[int, int]]] = {}
    frames_rendered = 0
    frames_with_pose = 0
    try:
        if start_f > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
        cur_f = start_f
        span = (
            max(1, (end_f - start_f + 1)) if end_f is not None else max(1, total_frames)
        )
        while True:
            if end_f is not None and cur_f > end_f:
                break
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            fp = pose_by_frame.get(cur_f)
            if fp is not None and fp.instances:
                frames_with_pose += 1
            rendered = draw_pose_frame_overlay(
                frame,
                fp,
                edges=edges,
                keypoint_index=kp_index,
                instance_index=inst_index,
                config=cfg,
                trails=trails if cfg.trail_length > 1 else None,
                comparison_pose=comp_by_frame.get(cur_f),
                frame_number=cur_f,
                fps=out_fps,
            )
            writer.write(rendered)
            frames_rendered += 1
            if progress_callback is not None and (
                frames_rendered % 15 == 0 or (end_f is not None and cur_f == end_f)
            ):
                pct = min(100.0, 100.0 * frames_rendered / float(span))
                progress_callback(
                    pct, f"Rendered frame {cur_f} ({frames_rendered}/{span})"
                )
            cur_f += 1
    finally:
        writer.release()
        cap.release()

    return PoseVideoRenderResult(
        video_path=str(vpath),
        pose_source=pose_label,
        output_video_path=str(out_path),
        frames_rendered=frames_rendered,
        frames_with_pose=frames_with_pose,
        width=width,
        height=height,
        fps=out_fps,
        color_mode=cfg.color_mode,
    )
