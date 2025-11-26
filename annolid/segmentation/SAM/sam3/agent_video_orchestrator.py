"""
Utilities to drive SAM3 video tracking with SAM3 Agent-corrected seeds on
window key frames. The agent refines the first frame of each window, then
`Sam3SessionManager` propagates within that window using the configured
tracking thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile

import cv2
from PIL import Image

from annolid.segmentation.SAM.sam3.session import Sam3SessionConfig, Sam3SessionManager
from .video_window_inference import _iter_video_windows
from .sam3.agent.agent_core import agent_inference


@dataclass
class AgentConfig:
    """Configuration for running the SAM3 Agent on key frames."""

    prompt: str
    det_thresh: float = 0.3
    window_size: int = 5
    stride: Optional[int] = None
    output_dir: str = "sam3_agent_out"
    debug: bool = False
    max_generations: int = 100


@dataclass
class TrackingConfig:
    """Configuration for SAM3 video propagation."""

    checkpoint_path: Optional[str] = None
    propagation_direction: str = "forward"
    device: Optional[str] = None
    score_threshold_detection: Optional[float] = None
    new_det_thresh: Optional[float] = None
    epsilon_for_polygon: float = 2.0
    max_frame_num_to_track: Optional[int] = None


def _save_frame_to_tmp(frame, dst_path: Path) -> None:
    """Persist a single BGR frame to disk for agent consumption."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(dst_path)


def _boxes_from_agent_output(outputs: Dict[str, object], det_thresh: float) -> Tuple[List[List[float]], List[float]]:
    """
    Convert agent JSON outputs (normalized xywh) to absolute xyxy boxes and
    filter by detection score.
    """
    boxes_norm = outputs.get("pred_boxes", []) or []
    scores = outputs.get("pred_scores", []) or []
    width = outputs.get("orig_img_w")
    height = outputs.get("orig_img_h")
    if not width or not height:
        return [], []

    boxes_abs: List[List[float]] = []
    kept_scores: List[float] = []
    for box, score in zip(boxes_norm, scores):
        try:
            if score < det_thresh:
                continue
            x, y, w, h = box
            x1 = float(x * width)
            y1 = float(y * height)
            x2 = float((x + w) * width)
            y2 = float((y + h) * height)
            boxes_abs.append([x1, y1, x2, y2])
            kept_scores.append(float(score))
        except Exception:
            continue
    return boxes_abs, kept_scores


def _run_agent_on_frame(
    frame_path: Path,
    agent_cfg: AgentConfig,
) -> Tuple[List[List[float]], List[float]]:
    """
    Run the SAM3 Agent on a single frame image and return filtered boxes/scores.
    """
    try:
        _, outputs, _ = agent_inference(
            img_path=str(frame_path),
            initial_text_prompt=agent_cfg.prompt,
            debug=agent_cfg.debug,
            max_generations=agent_cfg.max_generations,
            output_dir=agent_cfg.output_dir,
        )
    except Exception as exc:
        # Normalize assertion/missing tool responses into a clean RuntimeError for caller fallback.
        raise RuntimeError(
            f"Agent inference failed on {frame_path}: {exc} "
            "Ensure the configured VLM supports tool calls (e.g., qwen3-vl/vision-capable models)."
        ) from exc
    return _boxes_from_agent_output(outputs or {}, agent_cfg.det_thresh)


def run_agent_seeded_sam3_video(
    video_path: str,
    agent_cfg: AgentConfig,
    tracking_cfg: TrackingConfig,
    *,
    id_to_labels: Optional[Dict[int, str]] = None,
) -> Tuple[int, int]:
    """
    Iterate over a video in windows. For each window, run the SAM3 Agent on the
    first frame to obtain refined boxes, then propagate within that window using
    SAM3 video tracking.

    Returns:
        (total_frames_processed, total_masks_written)
    """
    id_to_labels = id_to_labels or {}
    session_cfg = Sam3SessionConfig(
        checkpoint_path=tracking_cfg.checkpoint_path,
        text_prompt=agent_cfg.prompt,
        epsilon_for_polygon=tracking_cfg.epsilon_for_polygon,
        ndjson_filename=None,
        max_frame_num_to_track=tracking_cfg.max_frame_num_to_track,
        propagation_direction=tracking_cfg.propagation_direction,
        device=tracking_cfg.device,
        score_threshold_detection=tracking_cfg.score_threshold_detection,
        new_det_thresh=tracking_cfg.new_det_thresh,
        async_loading_frames=False,
        sliding_window_size=agent_cfg.window_size,
        sliding_window_stride=agent_cfg.stride,
        use_sliding_window_for_text_prompt=False,
    )
    session = Sam3SessionManager(
        video_dir=video_path,
        id_to_labels=id_to_labels,
        config=session_cfg,
    )

    total_frames = 0
    total_masks = 0
    stride = agent_cfg.stride or agent_cfg.window_size

    with tempfile.TemporaryDirectory(prefix="sam3_agent_windows_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        for start_idx, end_idx, frames in _iter_video_windows(
            video_path=video_path,
            window_size=agent_cfg.window_size,
            stride=stride,
        ):
            if not frames:
                continue
            first_frame = frames[0]
            frame_path = tmpdir_path / f"frame_{start_idx:09}.png"
            _save_frame_to_tmp(first_frame, frame_path)
            boxes_abs, scores = _run_agent_on_frame(frame_path, agent_cfg)
            if not boxes_abs:
                continue

            # Build label hints and label ids for the prompts.
            label_hints = [
                f"{agent_cfg.prompt}_{i+1}" for i in range(len(boxes_abs))
            ]
            box_labels = list(range(1, len(boxes_abs) + 1))
            for lid, hint in zip(box_labels, label_hints):
                session.id_to_labels.setdefault(lid, hint)

            # Clear per-run tracking state to avoid leaking across windows.
            session._frames_processed.clear()
            session._frames_with_masks.clear()
            session._frame_masks.clear()
            session.obj_id_to_label.clear()

            max_frames = tracking_cfg.max_frame_num_to_track or (
                end_idx - start_idx)
            with session._session_scope(
                target_device=tracking_cfg.device, auto_close=True
            ):
                session._reset_action_history_if_supported()
                session.add_prompt_boxes_abs(
                    frame_idx=start_idx,
                    boxes_abs=boxes_abs,
                    box_labels=box_labels,
                    record_outputs=True,
                    label_hints=label_hints,
                )
                frames_processed, masks_written = session.propagate(
                    start_frame_idx=start_idx,
                    propagation_direction=tracking_cfg.propagation_direction,
                    max_frame_num_to_track=max_frames,
                )
                total_frames += frames_processed
                total_masks += masks_written
    if total_masks == 0:
        raise RuntimeError(
            "SAM3 agent-seeded run produced no masks; consider providing an API key "
            "for the LLM or falling back to standard tracking."
        )
    return total_frames, total_masks
