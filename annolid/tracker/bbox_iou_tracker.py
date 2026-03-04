"""
Standalone IOU-based bounding-box tracker for Annolid.

This is a self-contained reimplementation that does **not** depend on
detectron2.  It uses :mod:`torchvision.ops.box_iou` for pairwise IoU
computation and :class:`~annolid.tracker.simple_instances.SimpleInstances`
instead of ``detectron2.structures.Instances``.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Set

import numpy as np
import torch
from torchvision.ops import box_iou

from annolid.tracker.simple_instances import SimpleInstances


class BBoxIOUTracker:
    """Greedy IOU tracker that assigns persistent IDs to detections.

    For each pair of current / previous-frame bounding boxes the tracker
    computes pairwise IoU.  Pairs above ``track_iou_threshold`` are matched
    greedily in descending IoU order, giving the new detection the same ID as
    the matched previous detection.  Unmatched detections get fresh IDs.
    """

    def __init__(
        self,
        *,
        video_height: int,
        video_width: int,
        max_num_instances: int = 200,
        max_lost_frame_count: int = 0,
        min_box_rel_dim: float = 0.02,
        min_instance_period: int = 1,
        track_iou_threshold: float = 0.5,
    ) -> None:
        self._video_height = video_height
        self._video_width = video_width
        self._max_num_instances = max_num_instances
        self._max_lost_frame_count = max_lost_frame_count
        self._min_box_rel_dim = min_box_rel_dim
        self._min_instance_period = min_instance_period
        self._track_iou_threshold = track_iou_threshold

        self._prev_instances: Optional[SimpleInstances] = None
        self._matched_idx: Set[int] = set()
        self._matched_ID: Set[int] = set()
        self._untracked_prev_idx: Set[int] = set()
        self._id_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, instances: SimpleInstances) -> SimpleInstances:
        """Assign tracking IDs to *instances* for the current frame.

        Args:
            instances: Current-frame detections.  Must have ``pred_boxes``
                (``Tensor[N, 4]``), ``scores``, and ``pred_classes``.

        Returns:
            The same ``SimpleInstances`` object with ``ID``,
            ``ID_period``, and ``lost_frame_count`` fields populated.
        """
        instances = self._initialize_extra_fields(instances)

        if self._prev_instances is not None:
            iou_all = box_iou(
                instances.pred_boxes,
                self._prev_instances.pred_boxes,
            )
            bbox_pairs = self._create_prediction_pairs(instances, iou_all)
            self._reset_fields()
            for pair in bbox_pairs:
                idx = pair["idx"]
                prev_id = pair["prev_id"]
                if (
                    idx in self._matched_idx
                    or prev_id in self._matched_ID
                    or pair["IoU"] < self._track_iou_threshold
                ):
                    continue
                instances.ID[idx] = prev_id
                instances.ID_period[idx] = pair["prev_period"] + 1
                instances.lost_frame_count[idx] = 0
                self._matched_idx.add(idx)
                self._matched_ID.add(prev_id)
                self._untracked_prev_idx.discard(pair["prev_idx"])

            instances = self._assign_new_id(instances)
            instances = self._merge_untracked_instances(instances)

        self._prev_instances = copy.deepcopy(instances)
        return instances

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_prediction_pairs(
        self, instances: SimpleInstances, iou_all: torch.Tensor
    ) -> List[dict]:
        pairs = []
        for i in range(len(instances)):
            for j in range(len(self._prev_instances)):
                pairs.append(
                    {
                        "idx": i,
                        "prev_idx": j,
                        "prev_id": self._prev_instances.ID[j],
                        "IoU": float(iou_all[i, j]),
                        "prev_period": self._prev_instances.ID_period[j],
                    }
                )
        pairs.sort(key=lambda p: p["IoU"], reverse=True)
        return pairs

    def _initialize_extra_fields(self, instances: SimpleInstances) -> SimpleInstances:
        if not instances.has("ID"):
            instances.set("ID", [None] * len(instances))
        if not instances.has("ID_period"):
            instances.set("ID_period", [None] * len(instances))
        if not instances.has("lost_frame_count"):
            instances.set("lost_frame_count", [None] * len(instances))
        if self._prev_instances is None:
            instances.ID = list(range(len(instances)))
            self._id_count += len(instances)
            instances.ID_period = [1] * len(instances)
            instances.lost_frame_count = [0] * len(instances)
        return instances

    def _reset_fields(self) -> None:
        self._matched_idx = set()
        self._matched_ID = set()
        self._untracked_prev_idx = set(range(len(self._prev_instances)))

    def _assign_new_id(self, instances: SimpleInstances) -> SimpleInstances:
        untracked = set(range(len(instances))).difference(self._matched_idx)
        for idx in untracked:
            instances.ID[idx] = self._id_count
            self._id_count += 1
            instances.ID_period[idx] = 1
            instances.lost_frame_count[idx] = 0
        return instances

    def _merge_untracked_instances(self, instances: SimpleInstances) -> SimpleInstances:
        """Keep previously-tracked instances that were not matched this frame."""
        untracked = SimpleInstances(
            image_size=instances.image_size,
        )
        untracked.set("pred_boxes", [])
        untracked.set("pred_classes", [])
        untracked.set("scores", [])
        untracked.set("ID", [])
        untracked.set("ID_period", [])
        untracked.set("lost_frame_count", [])

        prev_boxes = self._prev_instances.pred_boxes
        prev_classes = self._prev_instances.pred_classes
        prev_scores = self._prev_instances.scores
        prev_ID_period = self._prev_instances.ID_period

        has_masks = instances.has("pred_masks")
        if has_masks:
            untracked.set("pred_masks", [])
            prev_masks = self._prev_instances.pred_masks

        for idx in self._untracked_prev_idx:
            box = prev_boxes[idx]
            if isinstance(box, torch.Tensor):
                x_left, y_top, x_right, y_bot = box.tolist()
            else:
                x_left, y_top, x_right, y_bot = box

            if (
                (x_right - x_left) / self._video_width < self._min_box_rel_dim
                or (y_bot - y_top) / self._video_height < self._min_box_rel_dim
                or self._prev_instances.lost_frame_count[idx]
                >= self._max_lost_frame_count
                or prev_ID_period[idx] <= self._min_instance_period
            ):
                continue

            untracked.pred_boxes.append(
                prev_boxes[idx].tolist()
                if isinstance(prev_boxes[idx], torch.Tensor)
                else list(prev_boxes[idx])
            )
            untracked.pred_classes.append(int(prev_classes[idx]))
            untracked.scores.append(float(prev_scores[idx]))
            untracked.ID.append(self._prev_instances.ID[idx])
            untracked.ID_period.append(self._prev_instances.ID_period[idx])
            untracked.lost_frame_count.append(
                self._prev_instances.lost_frame_count[idx] + 1
            )
            if has_masks:
                mask = prev_masks[idx]
                if isinstance(mask, torch.Tensor):
                    mask = mask.numpy().astype(np.uint8)
                untracked.pred_masks.append(mask)

        # Convert lists back to tensors
        if len(untracked.pred_boxes) > 0:
            untracked.pred_boxes = torch.FloatTensor(untracked.pred_boxes)
        else:
            untracked.pred_boxes = torch.zeros((0, 4), dtype=torch.float32)
        untracked.pred_classes = torch.IntTensor(untracked.pred_classes)
        untracked.scores = torch.FloatTensor(untracked.scores)
        if has_masks and len(untracked.pred_masks) > 0:
            untracked.pred_masks = torch.IntTensor(np.stack(untracked.pred_masks))
        elif has_masks:
            untracked.remove("pred_masks")

        return SimpleInstances.cat([instances, untracked])
