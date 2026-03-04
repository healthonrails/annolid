"""
Detectron2-based bounding-box IOU tracker for Annolid.

The ``detectron2`` import is optional at module level so that the rest of
annolid can be imported without detectron2 installed.  An informative
``ImportError`` is raised when the tracker is actually constructed.
"""

from __future__ import annotations

import warnings

try:
    from detectron2.config import CfgNode as CfgNode_
    from detectron2.config import instantiate
    from detectron2.tracking.base_tracker import build_tracker_head

    _D2_AVAILABLE = True
except ImportError:
    _D2_AVAILABLE = False


class D2BBoxIOUTracker:
    """Track objects in a video using IOUWeightedHungarianBBoxIOUTracker.

    Parameters
    ----------
    height, width:
        Frame dimensions (pixels).  These are required because the underlying
        D2 tracker uses them to normalise bounding-box coordinates.
    prev_instances, curr_instances:
        Detectron2 ``Instances`` objects from consecutive frames.
    max_num_instances:
        Maximum number of object tracks to maintain.
    max_lost_frame_count:
        Frames a track can be invisible before it is deleted.
    min_box_rel_dim:
        Minimum relative size (fraction of frame) for a box to be tracked.
    min_instance_period:
        Minimum frames an instance must appear before it gets a stable ID.
    track_iou_threshold:
        IOU threshold used to associate detections to tracks.
    """

    def __init__(
        self,
        height: int,
        width: int,
        prev_instances=None,
        curr_instances=None,
        max_num_instances: int = 10,
        max_lost_frame_count: int = 3,
        min_box_rel_dim: float = 0.02,
        min_instance_period: int = 1,
        track_iou_threshold: float = 0.5,
    ) -> None:
        if not _D2_AVAILABLE:
            raise ImportError(
                "detectron2 is required for D2BBoxIOUTracker. "
                "See https://detectron2.readthedocs.io/tutorials/install.html"
            )

        self._img_size = (int(height), int(width))
        self._prev_instances = prev_instances
        self._curr_instances = curr_instances

        self._max_num_instances = max_num_instances
        self._max_lost_frame_count = max_lost_frame_count
        self._min_box_rel_dim = min_box_rel_dim
        self._min_instance_period = min_instance_period
        self._track_iou_threshold = track_iou_threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_tracker(self):
        """Build and return a tracker using the modern ``instantiate()`` API."""
        cfg = {
            "_target_": (
                "detectron2.tracking.iou_weighted_hungarian_bbox_iou_tracker"
                ".IOUWeightedHungarianBBoxIOUTracker"
            ),
            "video_height": self._img_size[0],
            "video_width": self._img_size[1],
            "max_num_instances": self._max_num_instances,
            "max_lost_frame_count": self._max_lost_frame_count,
            "min_box_rel_dim": self._min_box_rel_dim,
            "min_instance_period": self._min_instance_period,
            "track_iou_threshold": self._track_iou_threshold,
        }
        return instantiate(cfg)

    def tracker_from_config(self):
        """Build a tracker using the legacy ``CfgNode`` API.

        .. deprecated::
            Use :meth:`get_tracker` instead.  This method will be removed in a
            future Annolid release.
        """
        warnings.warn(
            "tracker_from_config() is deprecated and will be removed in a "
            "future release. Use get_tracker() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        cfg = CfgNode_()
        cfg.TRACKER_HEADS = CfgNode_()
        cfg.TRACKER_HEADS.TRACKER_NAME = "IOUWeightedHungarianBBoxIOUTracker"
        cfg.TRACKER_HEADS.VIDEO_HEIGHT = self._img_size[0]
        cfg.TRACKER_HEADS.VIDEO_WIDTH = self._img_size[1]
        cfg.TRACKER_HEADS.MAX_NUM_INSTANCES = self._max_num_instances
        cfg.TRACKER_HEADS.MAX_LOST_FRAME_COUNT = self._max_lost_frame_count
        cfg.TRACKER_HEADS.MIN_BOX_REL_DIM = self._min_box_rel_dim
        cfg.TRACKER_HEADS.MIN_INSTANCE_PERIOD = self._min_instance_period
        cfg.TRACKER_HEADS.TRACK_IOU_THRESHOLD = self._track_iou_threshold
        return build_tracker_head(cfg)

    def extra_fields(self):
        tracker = self.get_tracker()
        return tracker._initialize_extra_fields(self._curr_instances)

    def _update(self):
        tracker = self.get_tracker()
        _ = tracker.update(self._prev_instances)
        return tracker.update(self._curr_instances)
