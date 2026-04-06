from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch

from annolid.utils.logger import logger


def expected_track_ids_for_frame(
    manager: object,
    frame_idx: int,
    *,
    max_gap: Optional[int] = None,
) -> set[int]:
    """Return recently active tracks that should still be considered live."""
    if max_gap is None:
        max_gap = max(3, min(int(getattr(manager, "sliding_window_size", 5) or 5), 10))
    frame_idx = int(frame_idx)
    expected: set[int] = set()
    last_seen_by_track = getattr(manager, "_track_last_seen_frame", {}) or {}
    if last_seen_by_track:
        expected.update(
            int(track_id)
            for track_id, last_seen_frame in last_seen_by_track.items()
            if 0 <= frame_idx - int(last_seen_frame) <= int(max_gap)
            and int(last_seen_frame) < frame_idx
        )
    start_frame = max(0, frame_idx - int(max_gap))
    frame_track_ids = getattr(manager, "_frame_track_ids", {})
    for prev_frame in range(start_frame, frame_idx):
        expected.update(int(v) for v in frame_track_ids.get(int(prev_frame), set()))
    return expected


def missing_track_ids_for_frame(manager: object, frame_idx: int) -> List[int]:
    current = {
        int(v) for v in getattr(manager, "_frame_track_ids", {}).get(int(frame_idx), set())
    }
    expected = expected_track_ids_for_frame(manager, int(frame_idx))
    missing = sorted(expected - current)
    return missing


def recent_track_mask(
    manager: object,
    obj_id: int,
    *,
    frame_idx: Optional[int] = None,
) -> Optional[np.ndarray]:
    track_id = int(obj_id)
    frame_masks = getattr(manager, "_frame_masks", {})
    candidate_frames: List[int] = []
    last_seen_frame = (getattr(manager, "_track_last_seen_frame", {}) or {}).get(track_id)
    if last_seen_frame is not None:
        last_seen_frame = int(last_seen_frame)
        if frame_idx is None or last_seen_frame <= int(frame_idx):
            candidate_frames.append(int(last_seen_frame))
    if not candidate_frames:
        candidate_frames = [
            int(candidate_frame)
            for candidate_frame, masks in frame_masks.items()
            if masks and str(track_id) in masks
        ]
        if frame_idx is not None:
            candidate_frames = [
                int(candidate_frame)
                for candidate_frame in candidate_frames
                if int(candidate_frame) <= int(frame_idx)
            ]
    if not candidate_frames:
        return None

    seen_frames: set[int] = set()
    for candidate_frame in sorted(candidate_frames, reverse=True):
        if int(candidate_frame) in seen_frames:
            continue
        seen_frames.add(int(candidate_frame))
        masks_for_frame = frame_masks.get(int(candidate_frame)) or {}
        mask = masks_for_frame.get(str(track_id))
        if mask is None:
            continue
        arr = np.asarray(mask, dtype=np.uint8)
        if arr.ndim != 2 or not arr.any():
            continue
        return arr
    return None


def should_accept_sam3_mask(
    manager: object,
    *,
    frame_idx: int,
    obj_id: int,
    mask: np.ndarray,
    box_xywh: Optional[np.ndarray] = None,
) -> bool:
    """Reject drifted or full-frame masks before persisting them."""
    arr = np.asarray(mask, dtype=np.uint8)
    if arr.ndim != 2 or not arr.any():
        return False
    frame_shape = getattr(manager, "frame_shape", None) or manager.get_frame_shape()
    frame_h, frame_w = frame_shape[:2]
    frame_area = float(max(1, frame_h * frame_w))
    mask_area = float(np.count_nonzero(arr))
    if mask_area <= 0.0:
        return False
    mask_ratio = mask_area / frame_area
    if mask_ratio >= 0.98:
        return False

    mask_box = manager._mask_to_bbox_xywh(arr)
    if mask_box is None:
        return False

    track_last_seen_frame = getattr(manager, "_track_last_seen_frame", {})
    frame_masks = getattr(manager, "_frame_masks", {})
    last_seen_frame = track_last_seen_frame.get(int(obj_id))
    previous_masks = {}
    if last_seen_frame is not None and int(last_seen_frame) != int(frame_idx):
        previous_masks = frame_masks.get(int(last_seen_frame)) or {}
        prev_mask = previous_masks.get(str(int(obj_id)))
        if prev_mask is not None:
            prev_arr = np.asarray(prev_mask, dtype=np.uint8)
            prev_box = manager._mask_to_bbox_xywh(prev_arr)
            if prev_box is not None:
                iou = manager._bbox_iou_xywh(mask_box, prev_box)
                if iou < 0.02:
                    prev_area = float(np.count_nonzero(prev_arr))
                    if prev_area > 0.0:
                        area_ratio = mask_area / prev_area
                        if area_ratio >= 3.5 or area_ratio <= 0.28:
                            return False
                    center_dist = float(
                        np.hypot(
                            *(
                                np.subtract(
                                    manager._bbox_center_xywh(mask_box),
                                    manager._bbox_center_xywh(prev_box),
                                )
                            )
                        )
                    )
                    prev_diag = float(
                        np.hypot(max(prev_box[2], mask_box[2]), max(prev_box[3], mask_box[3]))
                    )
                    if prev_diag > 0.0 and center_dist / prev_diag > 1.75:
                        return False

    if previous_masks and box_xywh is not None:
        try:
            box_arr = np.asarray(box_xywh, dtype=float)
            if box_arr.shape == (4,):
                center_dist = float(
                    np.hypot(
                        *(
                            np.subtract(
                                manager._bbox_center_xywh(mask_box),
                                manager._bbox_center_xywh(box_arr),
                            )
                        )
                    )
                )
                diag = float(np.hypot(max(mask_box[2], box_arr[2]), max(mask_box[3], box_arr[3])))
                if diag > 0.0 and center_dist / diag > 1.4:
                    return False
                if manager._bbox_iou_xywh(mask_box, box_arr) < 0.01:
                    return False
        except Exception:
            pass

    return True


def reacquire_frame_with_visual_and_text(
    manager: object,
    frame_idx: int,
    target_device: Optional[torch.device | str] = None,
) -> None:
    """
    Run a lightweight SAM3 pass on a single frame using only the text
    prompt to recover tracking when the main tracker path produced none.
    """
    with torch.inference_mode():
        if not getattr(manager, "text_prompt", None):
            return

        missing_track_ids = missing_track_ids_for_frame(manager, int(frame_idx))
        if not missing_track_ids:
            return
        target_track_ids = set(int(track_id) for track_id in missing_track_ids)

        with manager._session_scope(target_device, auto_close=True):
            manager._reset_session_state()
            manager._reset_action_history_if_supported()
            result = manager.add_prompt(
                frame_idx=frame_idx,
                text=manager.text_prompt,
                record_outputs=False,
                label_hints=manager._label_hints_from_ids(target_track_ids, manager.id_to_labels)
                if target_track_ids
                else None,
            )
            outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
            outputs = manager._map_outputs_to_global_ids_at_frame(
                outputs or {},
                frame_idx=int(frame_idx),
                allowed_gids=target_track_ids,
                allow_new_ids=False,
            )
            if manager._output_candidate_mask_count(outputs) <= 0:
                return
            manager._handle_frame_outputs(
                frame_idx=int(frame_idx),
                outputs=outputs,
                total_frames=manager.total_frames_estimate(),
                yielded_frames=len(manager._frames_processed) + 1,
                apply_score_threshold=False,
                merge_existing=True,
            )


def reacquire_frames_with_visual_and_text(
    manager: object,
    frame_indices: List[int],
    target_device: Optional[torch.device | str] = None,
) -> None:
    """
    Re-acquire multiple frames using a single temporary session to avoid
    reloading video data repeatedly, improving speed on CPU fallback.
    """
    if (
        not frame_indices
        or not getattr(manager, "text_prompt", None)
        or not getattr(manager, "_frame_masks", None)
    ):
        return

    with torch.inference_mode():
        with manager._session_scope(target_device, auto_close=True):
            for frame_idx in frame_indices:
                missing_track_ids = missing_track_ids_for_frame(manager, int(frame_idx))
                if not missing_track_ids:
                    continue
                target_track_ids = set(int(track_id) for track_id in missing_track_ids)
                manager._reset_session_state()
                manager._reset_action_history_if_supported()
                try:
                    result = manager.add_prompt(
                        frame_idx=frame_idx,
                        text=manager.text_prompt,
                        record_outputs=False,
                        label_hints=manager._label_hints_from_ids(
                            target_track_ids, manager.id_to_labels
                        )
                        if target_track_ids
                        else None,
                    )
                    outputs = result.get("outputs", {}) if isinstance(result, dict) else {}
                    outputs = manager._map_outputs_to_global_ids_at_frame(
                        outputs or {},
                        frame_idx=int(frame_idx),
                        allowed_gids=target_track_ids,
                        allow_new_ids=False,
                    )
                    if manager._output_candidate_mask_count(outputs) <= 0:
                        continue
                    manager._handle_frame_outputs(
                        frame_idx=int(frame_idx),
                        outputs=outputs,
                        total_frames=manager.total_frames_estimate(),
                        yielded_frames=len(manager._frames_processed) + 1,
                        apply_score_threshold=False,
                        merge_existing=True,
                    )
                except Exception as exc:
                    logger.warning(
                        "SAM3 per-frame reacquisition failed for frame %s: %s",
                        frame_idx,
                        exc,
                    )
