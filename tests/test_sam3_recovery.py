from __future__ import annotations

import numpy as np

from annolid.segmentation.SAM.sam3.recovery import (
    expected_track_ids_for_frame,
    missing_track_ids_for_frame,
    recent_track_mask,
    should_accept_sam3_mask,
)


class _RecoveryManager:
    def __init__(self) -> None:
        self.sliding_window_size = 5
        self._frame_track_ids = {
            0: {1},
            1: {1, 2},
            2: {2},
        }
        mask = np.zeros((8, 8), dtype=np.uint8)
        mask[2:5, 2:5] = 1
        self._frame_masks = {2: {"2": mask}}
        self._track_last_seen_frame = {2: 2}
        self.frame_shape = (8, 8, 3)

    def get_frame_shape(self):
        return self.frame_shape

    @staticmethod
    def _mask_to_bbox_xywh(mask):
        ys, xs = np.nonzero(np.asarray(mask, dtype=np.uint8))
        if len(xs) == 0 or len(ys) == 0:
            return None
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        return np.asarray([x1, y1, x2 - x1, y2 - y1], dtype=float)

    @staticmethod
    def _bbox_center_xywh(box):
        return float(box[0] + box[2] / 2.0), float(box[1] + box[3] / 2.0)

    @staticmethod
    def _bbox_iou_xywh(a, b):
        ax1, ay1, aw, ah = [float(v) for v in a]
        bx1, by1, bw, bh = [float(v) for v in b]
        ax2, ay2 = ax1 + aw, ay1 + ah
        bx2, by2 = bx1 + bw, by1 + bh
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        denom = max(1e-9, aw * ah + bw * bh - inter)
        return float(inter / denom)


def test_expected_and_missing_track_ids_for_frame() -> None:
    manager = _RecoveryManager()
    assert expected_track_ids_for_frame(manager, 3) == {1, 2}
    assert missing_track_ids_for_frame(manager, 3) == [1, 2]


def test_expected_track_ids_prefers_last_seen_fast_path() -> None:
    manager = _RecoveryManager()
    manager._track_last_seen_frame = {1: 1, 2: 2}
    manager._frame_track_ids = {}

    assert expected_track_ids_for_frame(manager, 3) == {1, 2}


def test_recent_track_mask_returns_latest_valid_mask() -> None:
    manager = _RecoveryManager()
    mask = recent_track_mask(manager, 2, frame_idx=3)
    assert mask is not None
    assert mask.shape == (8, 8)
    assert int(mask.sum()) > 0


def test_recent_track_mask_uses_last_seen_fast_path() -> None:
    manager = _RecoveryManager()
    manager._frame_masks = {
        2: {"2": manager._frame_masks[2]["2"]},
        1: {"2": np.zeros((8, 8), dtype=np.uint8)},
    }
    manager._track_last_seen_frame = {2: 2}

    mask = recent_track_mask(manager, 2, frame_idx=3)

    assert mask is not None
    assert int(mask.sum()) > 0


def test_should_accept_sam3_mask_rejects_near_full_frame() -> None:
    manager = _RecoveryManager()
    full = np.ones((8, 8), dtype=np.uint8)
    assert not should_accept_sam3_mask(
        manager,
        frame_idx=3,
        obj_id=2,
        mask=full,
        box_xywh=np.asarray([0.0, 0.0, 8.0, 8.0], dtype=float),
    )
