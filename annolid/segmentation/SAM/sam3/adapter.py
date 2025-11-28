"""
Thin SAM3 wrapper used by the GUI to run video propagation from existing
LabelMe annotations, now backed by a reusable session manager so we can
later add interactive prompts without changing call sites.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from qtpy import QtCore

from annolid.segmentation.SAM.sam_v2 import load_annotations_from_video
from annolid.utils.logger import logger
from annolid.gui.shape import Shape

from .session import Sam3SessionConfig, Sam3SessionManager
from .agent_video_orchestrator import (
    AgentConfig,
    TrackingConfig,
    run_agent_seeded_sam3_video,
)


class SAM3VideoProcessor(Sam3SessionManager):
    """Run SAM3 video propagation while reusing SAM2 video utilities."""

    def __init__(
        self,
        video_dir,
        id_to_labels,
        annotations,
        checkpoint_path: Optional[str] = None,
        text_prompt: Optional[str] = None,
        epsilon_for_polygon: float = 2.0,
        propagation_direction: str = "both",
        max_frame_num_to_track: Optional[int] = None,
        device: Optional[str] = None,
        score_threshold_detection: Optional[float] = None,
        new_det_thresh: Optional[float] = None,
        sliding_window_size: int = 5,
        sliding_window_stride: Optional[int] = None,
        use_sliding_window_for_text_prompt: bool = True,
    ):
        self.annotations = annotations or []
        self.target_device = device
        config = Sam3SessionConfig(
            checkpoint_path=checkpoint_path,
            text_prompt=text_prompt,
            epsilon_for_polygon=epsilon_for_polygon,
            ndjson_filename=None,
            max_frame_num_to_track=max_frame_num_to_track,
            propagation_direction=propagation_direction,
            device=device,
            score_threshold_detection=score_threshold_detection,
            new_det_thresh=new_det_thresh,
            sliding_window_size=sliding_window_size,
            sliding_window_stride=sliding_window_stride,
            use_sliding_window_for_text_prompt=use_sliding_window_for_text_prompt,
        )
        super().__init__(
            video_dir=video_dir,
            id_to_labels=id_to_labels,
            config=config,
        )
        self.propagation_direction = propagation_direction

    def _run_once(self, target_device: Optional[str]):
        # Run propagation with optional device override
        frames, masks = self.run_offline(
            self.annotations,
            target_device or self.target_device,
            propagation_direction=self.propagation_direction,
            max_frame_num_to_track=self.max_frame_num_to_track,
        )
        return frames, masks

    def run(self):
        """
        Execute a SAM3 run, aggregating masks after frame-by-frame propagation.
        Includes MPS->CPU retry behavior.
        """
        try:
            frames, masks = self._run_once(None)
        except RuntimeError as exc:
            msg = str(exc)
            if "MPS backend out of memory" in msg or "MPS does not support" in msg:
                logger.warning(
                    "SAM3 hit MPS OOM; retrying on CPU. Original error: %s", msg
                )
                frames, masks = self._run_once("cpu")
            else:
                raise

        if frames == 0:
            raise RuntimeError("SAM3 propagate_in_video yielded no frames")
        if masks == 0:
            logger.warning("SAM3 completed but produced no masks on any frame")
            # Fallback: if we have seed annotations but SAM3 produced no masks,
            # persist the seed boxes to NDJSON so downstream tools can still
            # consume a consistent detection stream.
            try:
                self._fallback_write_seed_annotations_to_ndjson()
            except Exception as exc:
                logger.warning(
                    "SAM3 fallback NDJSON emission failed: %s", exc, exc_info=True
                )

        logger.info("SAM3 finished: frames=%d, masks=%d", frames, masks)
        return f"SAM3 done#{masks}"

    def _fallback_write_seed_annotations_to_ndjson(self) -> None:
        if not self.annotations:
            return
        _write_seed_annotations_to_ndjson(self, self.annotations)


def process_video(
    video_path: str | Path,
    *,
    checkpoint_path: Optional[str] = None,
    text_prompt: Optional[str] = None,
    epsilon_for_polygon: float = 2.0,
    propagation_direction: str = "both",
    max_frame_num_to_track: Optional[int] = None,
    device: Optional[str] = None,
    sliding_window_size: int = 5,
    sliding_window_stride: Optional[int] = None,
    use_sliding_window_for_text_prompt: bool = True,
) -> str:
    """
    Run SAM3 video propagation seeded by existing LabelMe annotations.

    Args:
        video_path: MP4 file or directory of frames.
        checkpoint_path: Optional explicit checkpoint; if None, SAM3 will
            auto-download (requires HF auth and GPU).
        text_prompt: Optional text prompt; if None, boxes-only prompting is used.
    Returns:
        A short status string for the GUI.
    """
    annotations, id_to_labels = load_annotations_from_video(video_path)
    processor = SAM3VideoProcessor(
        video_dir=video_path,
        id_to_labels=id_to_labels,
        annotations=annotations,
        checkpoint_path=checkpoint_path,
        text_prompt=text_prompt,
        epsilon_for_polygon=epsilon_for_polygon,
        propagation_direction=propagation_direction,
        max_frame_num_to_track=max_frame_num_to_track,
        device=device,
        sliding_window_size=sliding_window_size,
        sliding_window_stride=sliding_window_stride,
        use_sliding_window_for_text_prompt=use_sliding_window_for_text_prompt,
    )
    return processor.run()


def process_video_with_agent(
    video_path: str | Path,
    *,
    agent_prompt: str,
    agent_det_thresh: float = 0.3,
    window_size: int = 5,
    stride: Optional[int] = None,
    output_dir: str = "sam3_agent_out",
    checkpoint_path: Optional[str] = None,
    propagation_direction: str = "forward",
    device: Optional[str] = None,
    score_threshold_detection: Optional[float] = None,
    new_det_thresh: Optional[float] = None,
) -> Tuple[int, int]:
    """
    Run SAM3 video propagation using SAM3 Agent to refine the first frame of
    each window, then track within that window.

    Returns a short status string for the GUI/CLI.
    """
    os.makedirs(output_dir, exist_ok=True)

    agent_cfg = AgentConfig(
        prompt=agent_prompt,
        det_thresh=agent_det_thresh,
        window_size=window_size,
        stride=stride,
        output_dir=output_dir,
    )
    tracking_cfg = TrackingConfig(
        checkpoint_path=checkpoint_path,
        propagation_direction=propagation_direction,
        device=device,
        score_threshold_detection=score_threshold_detection,
        new_det_thresh=new_det_thresh,
    )
    frames, masks = run_agent_seeded_sam3_video(
        video_path=str(video_path),
        agent_cfg=agent_cfg,
        tracking_cfg=tracking_cfg,
    )
    logger.info(
        "SAM3 agent-seeded propagation finished: frames=%d, masks=%d",
        frames,
        masks,
    )
    return frames, masks


def _shape_from_box(label: str, box: List[float]) -> Shape:
    """
    Build a rectangle Shape from a [x1, y1, x2, y2] box.
    """
    x1, y1, x2, y2 = box
    shape = Shape(
        label=label,
        shape_type="rectangle",
        flags={},
        description="sam3_seed",
    )
    shape.addPoint(QtCore.QPointF(float(x1), float(y1)))
    shape.addPoint(QtCore.QPointF(float(x2), float(y2)))
    return shape


def _group_annotations_by_frame(annotations: List[dict]) -> Dict[int, List[dict]]:
    grouped: Dict[int, List[dict]] = {}
    for ann in annotations:
        try:
            frame_idx = int(ann.get("ann_frame_idx", 0))
        except (TypeError, ValueError):
            frame_idx = 0
        grouped.setdefault(frame_idx, []).append(ann)
    return grouped


def _label_for_annotation(id_to_labels: Dict[int, str], ann: dict) -> str:
    labels = ann.get("labels") or []
    if labels:
        try:
            return id_to_labels.get(int(labels[0]), str(labels[0]))
        except Exception:
            return str(labels[0])
    return "object"


def _extract_box_from_annotation(ann: dict) -> Optional[List[float]]:
    ann_type = ann.get("type")
    if ann_type == "box":
        box = ann.get("box")
        if not box or len(box) != 4:
            return None
        try:
            x1, y1, x2, y2 = map(float, box)
        except Exception:
            return None
        return [x1, y1, x2, y2]
    if ann_type == "mask":
        mask = ann.get("mask")
        if mask is None:
            return None
        try:
            arr = np.asarray(mask, dtype=np.uint8)
            ys, xs = np.nonzero(arr)
            if len(xs) == 0 or len(ys) == 0:
                return None
            x1, x2 = float(xs.min()), float(xs.max())
            y1, y2 = float(ys.min()), float(ys.max())
            return [x1, y1, x2, y2]
        except Exception:
            return None
    return None


def _frame_shape_from_video(processor: Sam3SessionManager) -> Optional[tuple]:
    """
    Best-effort frame shape retrieval for fallback NDJSON writing.
    """
    if processor.frame_shape is not None:
        return processor.frame_shape
    try:
        return processor.get_frame_shape()
    except Exception:
        return None


def _write_seed_annotations_to_ndjson(
    processor: Sam3SessionManager,
    annotations: List[dict],
) -> None:
    """
    Write seed box annotations directly to the SAM3 NDJSON stream when the
    model produced no masks. This keeps downstream consumers working even
    in degenerate cases.
    """
    if processor.ndjson_writer is None or not annotations:
        return

    frame_shape = _frame_shape_from_video(processor)
    if frame_shape is None:
        logger.warning(
            "SAM3 fallback NDJSON emission skipped: unable to determine frame shape."
        )
        return

    grouped = _group_annotations_by_frame(annotations)
    for frame_idx, anns in grouped.items():
        shapes: List[Shape] = []
        for ann in anns:
            box = _extract_box_from_annotation(ann)
            if box is None:
                continue
            label = _label_for_annotation(processor.id_to_labels, ann)
            shapes.append(_shape_from_box(label, box))

        if not shapes:
            continue

        try:
            processor.ndjson_writer.write(
                frame_idx,
                frame_shape,
                shapes,
                frame_other_data={"sam3_fallback": True},
            )
        except Exception as exc:
            logger.warning(
                "SAM3 fallback NDJSON emission failed for frame %s: %s",
                frame_idx,
                exc,
            )
