# This code defines a class `D2BBoxIOUTracker` that is responsible for tracking objects
# in a video using the IOUWeightedHungarianBBoxIOUTracker algorithm.

import copy
import numpy as np
from typing import Dict
import torch

# Importing required modules from the detectron2 library
from detectron2.config import CfgNode as CfgNode_
from detectron2.config import instantiate
from detectron2.structures import Boxes, Instances
from detectron2.tracking.base_tracker import build_tracker_head
from detectron2.tracking.iou_weighted_hungarian_bbox_iou_tracker import (
    IOUWeightedHungarianBBoxIOUTracker,
)


class D2BBoxIOUTracker:
    """
    This class is responsible for tracking objects in a video using 
    the IOUWeightedHungarianBBoxIOUTracker algorithm.
    """

    def __init__(self,
                 prev_instances,
                 curr_instances,
                 max_num_instances=10,
                 max_lost_frame_count=3,
                 min_box_rel_dim=0.02,
                 min_instance_period=1,
                 track_iou_threshold=0.5
                 ) -> None:
        """
        Instantiates the class with instances from the previous and current frame.

        Args:
            prev_instances (dict): The prediction in Dict format from the previous frame.
            curr_instances (dict): The prediction in Dict format from the current frame.
            max_num_instances (int, optional): Maximum number of instances to track. Defaults to 10.
            max_lost_frame_count (int, optional): Maximum number of frames that the tracker can lose a track. Defaults to 3.
            min_box_rel_dim (float, optional): Minimum relative dimension of the bounding box. Defaults to 0.02.
            min_instance_period (int, optional): Minimum period of instances. Defaults to 1.
            track_iou_threshold (float, optional): Threshold of Intersection over Union (IOU) for tracking. Defaults to 0.5.

        Example:
            Instances(
                image_size=torch.IntTensor(prediction["image_size"]),
                pred_boxes=Boxes(torch.FloatTensor(prediction["pred_boxes"])),
                pred_masks=torch.IntTensor(prediction["pred_masks"]),
                pred_classes=torch.IntTensor(prediction["pred_classes"]),
                scores=torch.FloatTensor(prediction["scores"]),
            )
        """

        self._prev_instances = prev_instances
        self._curr_instances = curr_instances

        self._max_num_instances = max_num_instances
        self._max_lost_frame_count = max_lost_frame_count
        self._min_box_rel_dim = min_box_rel_dim
        self._min_instance_period = min_instance_period
        self._track_iou_threshold = track_iou_threshold

    def get_tracker(self):
        cfg = {
            "_target_": "detectron2.tracking.iou_weighted_hungarian_bbox_iou_tracker.IOUWeightedHungarianBBoxIOUTracker",  # noqa
            "video_height": self._img_size[0],
            "video_width": self._img_size[1],
            "max_num_instances": self._max_num_instances,
            "max_lost_frame_count": self._max_lost_frame_count,
            "min_box_rel_dim": self._min_box_rel_dim,
            "min_instance_period": self._min_instance_period,
            "track_iou_threshold": self._track_iou_threshold,
        }
        tracker = instantiate(cfg)
        return tracker

    def tracker_from_config(self):
        cfg = CfgNode_()
        cfg.TRACKER_HEADS = CfgNode_()
        cfg.TRACKER_HEADS.TRACKER_NAME = "IOUWeightedHungarianBBoxIOUTracker"
        cfg.TRACKER_HEADS.VIDEO_HEIGHT = int(self._img_size[0])
        cfg.TRACKER_HEADS.VIDEO_WIDTH = int(self._img_size[1])
        cfg.TRACKER_HEADS.MAX_NUM_INSTANCES = self._max_num_instances
        cfg.TRACKER_HEADS.MAX_LOST_FRAME_COUNT = self._max_lost_frame_count
        cfg.TRACKER_HEADS.MIN_BOX_REL_DIM = self._min_box_rel_dim
        cfg.TRACKER_HEADS.MIN_INSTANCE_PERIOD = self._min_instance_period
        cfg.TRACKER_HEADS.TRACK_IOU_THRESHOLD = self._track_iou_threshold
        tracker = build_tracker_head(cfg)
        return tracker

    def extra_fields(self):

        tracker = self.get_tracker()
        instances = tracker._initialize_extra_fields(self._curr_instances)
        # self.assertTrue(instances.has("ID"))
        # self.assertTrue(instances.has("ID_period"))
        # self.assertTrue(instances.has("lost_frame_count"))
        return instances

    def _update(self):

        tracker = self.get_tracker()
        _ = tracker.update(self._prev_instances)
        curr_instances = tracker.update(self._curr_instances)
        return curr_instances
