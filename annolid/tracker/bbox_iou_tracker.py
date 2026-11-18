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
        self._video_height = max(1, int(video_height))
        self._video_width = max(1, int(video_width))
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

    @staticmethod
    def _as_box_tensor(boxes: object) -> torch.Tensor:
        if isinstance(boxes, torch.Tensor):
            if boxes.numel() == 0:
                return boxes.reshape(0, 4).to(dtype=torch.float32)
            return boxes.reshape(-1, 4).to(dtype=torch.float32)
        arr = np.asarray(boxes, dtype=np.float32)
        if arr.size == 0:
            return torch.zeros((0, 4), dtype=torch.float32)
        return torch.from_numpy(arr.reshape(-1, 4))

    @staticmethod
    def _snapshot_instances(instances: SimpleInstances) -> SimpleInstances:
        snap = SimpleInstances(image_size=instances.image_size)
        boxes = instances.pred_boxes
        snap.set(
            "pred_boxes",
            boxes.detach().clone()
            if isinstance(boxes, torch.Tensor)
            else copy.deepcopy(boxes),
        )
        for field_name in ("pred_classes", "scores"):
            if instances.has(field_name):
                val = getattr(instances, field_name)
                snap.set(
                    field_name,
                    val.detach().clone()
                    if isinstance(val, torch.Tensor)
                    else list(val),
                )
        for list_field in ("ID", "ID_period", "lost_frame_count"):
            if instances.has(list_field):
                snap.set(list_field, list(getattr(instances, list_field)))
        if instances.has("pred_masks"):
            masks = instances.pred_masks
            snap.set(
                "pred_masks",
                masks.detach() if isinstance(masks, torch.Tensor) else masks,
            )
        return snap

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
            self._reset_fields()
            cur_boxes = self._as_box_tensor(instances.pred_boxes)
            prev_boxes = self._as_box_tensor(self._prev_instances.pred_boxes)
            if cur_boxes.shape[0] > 0 and prev_boxes.shape[0] > 0:
                if prev_boxes.device != cur_boxes.device:
                    prev_boxes = prev_boxes.to(cur_boxes.device)
                iou_all = box_iou(cur_boxes, prev_boxes)
                bbox_pairs = self._create_prediction_pairs(instances, iou_all)
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

        self._prev_instances = self._snapshot_instances(instances)
        return instances

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _create_prediction_pairs(
        self, instances: SimpleInstances, iou_all: torch.Tensor
    ) -> List[dict]:
        _ = instances
        iou_cpu = iou_all.detach().cpu()
        valid = torch.nonzero(
            iou_cpu >= float(self._track_iou_threshold), as_tuple=False
        )
        if valid.numel() == 0:
            return []
        prev_ids = self._prev_instances.ID
        prev_periods = self._prev_instances.ID_period
        pairs = [
            {
                "idx": int(i),
                "prev_idx": int(j),
                "prev_id": prev_ids[int(j)],
                "IoU": float(iou_cpu[int(i), int(j)]),
                "prev_period": prev_periods[int(j)],
            }
            for i, j in valid.tolist()
        ]
        pairs.sort(key=lambda p: (-p["IoU"], p["idx"], p["prev_idx"]))
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
        for idx in range(len(instances)):
            if idx in self._matched_idx:
                continue
            instances.ID[idx] = self._id_count
            self._id_count += 1
            instances.ID_period[idx] = 1
            instances.lost_frame_count[idx] = 0
        return instances

    def _merge_untracked_instances(self, instances: SimpleInstances) -> SimpleInstances:
        """Keep previously-tracked instances that were not matched this frame."""
        if not self._untracked_prev_idx or self._max_lost_frame_count <= 0:
            return instances
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

        has_masks = instances.has("pred_masks") and self._prev_instances.has(
            "pred_masks"
        )
        if has_masks:
            untracked.set("pred_masks", [])
            prev_masks = self._prev_instances.pred_masks

        for idx in sorted(self._untracked_prev_idx):
            box = prev_boxes[idx]
            if isinstance(box, torch.Tensor):
                x_left, y_top, x_right, y_bot = box.detach().cpu().tolist()
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

            untracked.pred_boxes.append([x_left, y_top, x_right, y_bot])
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
                    mask = mask.detach().cpu().numpy().astype(np.uint8)
                untracked.pred_masks.append(mask)

        if len(untracked.pred_boxes) == 0:
            return instances

        target_device = (
            instances.pred_boxes.device
            if isinstance(instances.pred_boxes, torch.Tensor)
            else torch.device("cpu")
        )
        classes_dtype = (
            instances.pred_classes.dtype
            if isinstance(getattr(instances, "pred_classes", None), torch.Tensor)
            else torch.int32
        )
        # Convert lists back to tensors
        untracked.pred_boxes = torch.tensor(
            untracked.pred_boxes, dtype=torch.float32, device=target_device
        )
        untracked.pred_classes = torch.tensor(
            untracked.pred_classes, dtype=classes_dtype, device=target_device
        )
        untracked.scores = torch.tensor(
            untracked.scores, dtype=torch.float32, device=target_device
        )
        if has_masks and len(untracked.pred_masks) > 0:
            mask_dtype = (
                instances.pred_masks.dtype
                if isinstance(instances.pred_masks, torch.Tensor)
                else torch.uint8
            )
            untracked.pred_masks = torch.as_tensor(
                np.stack(untracked.pred_masks), device=target_device
            ).to(dtype=mask_dtype)
        elif instances.has("pred_masks"):
            inst_masks = instances.pred_masks
            if isinstance(inst_masks, torch.Tensor) and inst_masks.ndim >= 3:
                untracked.set(
                    "pred_masks",
                    torch.zeros(
                        (len(untracked.pred_boxes), *inst_masks.shape[1:]),
                        dtype=inst_masks.dtype,
                        device=target_device,
                    ),
                )

        if len(instances) == 0:
            return untracked
        return SimpleInstances.cat([instances, untracked])
