from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from annolid.utils.logger import logger


@dataclass
class PromptBuildResult:
    frame_idx: Optional[int]
    boxes: List[List[float]]
    box_labels: List[int]
    mask_inputs: List[np.ndarray]
    mask_labels: List[int]
    points: List[List[float]]
    point_labels: List[int]
    obj_ids: List[int]
    point_obj_ids: List[int]

    def as_tuple(
        self,
    ) -> Tuple[
        Optional[int],
        List[List[float]],
        List[int],
        List[np.ndarray],
        List[int],
        List[List[float]],
        List[int],
        List[int],
        List[int],
    ]:
        return (
            self.frame_idx,
            self.boxes,
            self.box_labels,
            self.mask_inputs,
            self.mask_labels,
            self.points,
            self.point_labels,
            self.obj_ids,
            self.point_obj_ids,
        )


def label_hints_from_ids(labels: List[int], id_to_labels: Dict[int, str]) -> List[str]:
    hints: List[str] = []
    for lid in labels:
        try:
            hints.append(id_to_labels.get(int(lid), str(lid)))
        except Exception:
            hints.append(str(lid))
    return hints


def build_prompts_from_annotations(
    *,
    annotations: Iterable[dict],
    text_prompt: Optional[str],
    frame_shape: Tuple[int, int, int],
    video_dir: str,
    first_frame_index: Callable[[], int],
    shape_points_to_mask: Callable[[List[List[float]], Tuple[int, int, int], str], np.ndarray],
) -> PromptBuildResult:
    if not annotations:
        if text_prompt:
            logger.info(
                "SAM3 using text-only prompt; no per-frame annotations found under %s",
                video_dir,
            )
            return PromptBuildResult(
                frame_idx=int(first_frame_index()),
                boxes=[],
                box_labels=[],
                mask_inputs=[],
                mask_labels=[],
                points=[],
                point_labels=[],
                obj_ids=[],
                point_obj_ids=[],
            )
        raise FileNotFoundError(f"No per-frame JSON annotations found under {video_dir}")

    height, width = frame_shape[:2]
    annotations_by_frame: Dict[int, List[dict]] = {}
    for ann in annotations:
        try:
            frame_idx = int(ann.get("ann_frame_idx", 0))
        except (TypeError, ValueError):
            frame_idx = 0
        annotations_by_frame.setdefault(frame_idx, []).append(ann)

    for frame_idx in sorted(annotations_by_frame):
        boxes: List[List[float]] = []
        box_labels: List[int] = []
        mask_inputs: List[np.ndarray] = []
        mask_labels: List[int] = []
        points: List[List[float]] = []
        point_labels: List[int] = []
        obj_ids: List[int] = []
        point_obj_ids: List[int] = []
        for ann in annotations_by_frame[frame_idx]:
            label_val = int(ann["labels"][0]) if ann.get("labels") else 1
            if ann["type"] == "box":
                x1, y1, x2, y2 = ann["box"]
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
            elif ann["type"] == "mask":
                mask = np.asarray(ann["mask"], dtype=np.uint8)
                if mask.ndim != 2:
                    continue
                if mask.shape[:2] != (height, width):
                    try:
                        mask = cv2.resize(
                            mask,
                            (width, height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                    except Exception:
                        continue
                if not np.any(mask):
                    continue
                mask_inputs.append(mask.astype(np.uint8))
                mask_labels.append(1)
                try:
                    obj_ids.append(int(ann.get("obj_id", label_val)))
                except Exception:
                    obj_ids.append(int(label_val))
                continue
            elif ann["type"] in {"polygon", "polyline"}:
                poly_pts = ann.get("polygon") or ann.get("polyline") or []
                if not poly_pts:
                    continue
                try:
                    mask = shape_points_to_mask(
                        poly_pts,
                        frame_shape,
                        "polygon",
                    )
                    if mask is None or not np.any(mask):
                        continue
                except Exception:
                    continue
                mask_inputs.append(mask)
                mask_labels.append(1)
                try:
                    obj_ids.append(int(ann.get("obj_id", label_val)))
                except Exception:
                    obj_ids.append(int(label_val))
                continue
            elif ann["type"] in {"points", "point"}:
                ann_points = ann.get("points") or []
                if not ann_points and ann.get("point"):
                    ann_points = [ann.get("point")]
                if not ann_points:
                    continue
                ann_point_labels = ann.get("labels") or []
                added_any = False
                for idx, pt in enumerate(ann_points):
                    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                        continue
                    try:
                        x = float(pt[0])
                        y = float(pt[1])
                    except Exception:
                        continue
                    if x < 0.0 or y < 0.0:
                        continue
                    points.append([x / width, y / height])
                    raw_label = ann_point_labels[idx] if idx < len(ann_point_labels) else 1
                    try:
                        point_labels.append(1 if int(raw_label) > 0 else 0)
                    except Exception:
                        point_labels.append(1)
                    try:
                        point_obj_ids.append(int(ann.get("obj_id", label_val)))
                    except Exception:
                        point_obj_ids.append(int(label_val))
                    added_any = True
                if added_any:
                    try:
                        obj_ids.append(int(ann.get("obj_id", label_val)))
                    except Exception:
                        obj_ids.append(int(label_val))
                continue
            else:
                continue

            if w <= 0 or h <= 0:
                continue

            boxes.append([x1 / width, y1 / height, w / width, h / height])
            box_labels.append(label_val)
            obj_ids.append(int(ann.get("obj_id", label_val)))

        if boxes or mask_inputs or points:
            return PromptBuildResult(
                frame_idx=int(frame_idx),
                boxes=boxes,
                box_labels=box_labels,
                mask_inputs=mask_inputs,
                mask_labels=mask_labels,
                points=points,
                point_labels=point_labels,
                obj_ids=obj_ids,
                point_obj_ids=point_obj_ids,
            )

    if text_prompt:
        logger.info(
            "SAM3 using text-only prompt; no usable per-frame annotations were found under %s",
            video_dir,
        )
        return PromptBuildResult(
            frame_idx=int(first_frame_index()),
            boxes=[],
            box_labels=[],
            mask_inputs=[],
            mask_labels=[],
            points=[],
            point_labels=[],
            obj_ids=[],
            point_obj_ids=[],
        )

    return PromptBuildResult(
        frame_idx=None,
        boxes=[],
        box_labels=[],
        mask_inputs=[],
        mask_labels=[],
        points=[],
        point_labels=[],
        obj_ids=[],
        point_obj_ids=[],
    )
