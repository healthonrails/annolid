from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment


def coerce_numpy_float_array(value: object) -> Optional[np.ndarray]:
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value, dtype=float)
    except Exception:
        return None


def normalize_obj_ptr(value: object) -> Optional[np.ndarray]:
    arr = coerce_numpy_float_array(value)
    if arr is None:
        return None
    arr = np.asarray(arr, dtype=float).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0:
        return None
    return arr / norm


def extract_obj_ptr_vectors(
    obj_ptrs: object, expected_count: int
) -> List[Optional[np.ndarray]]:
    if obj_ptrs is None or expected_count <= 0:
        return [None] * max(0, int(expected_count))
    arr = coerce_numpy_float_array(obj_ptrs)
    if arr is None:
        return [None] * expected_count
    if arr.ndim == 1:
        if expected_count != 1:
            return [None] * expected_count
        return [normalize_obj_ptr(arr)]
    out: List[Optional[np.ndarray]] = []
    for idx in range(expected_count):
        if idx >= arr.shape[0]:
            out.append(None)
            continue
        out.append(normalize_obj_ptr(arr[idx]))
    return out


def obj_ptr_similarity(
    track_ptr: Optional[np.ndarray],
    candidate_ptr: Optional[np.ndarray],
) -> Optional[float]:
    if track_ptr is None or candidate_ptr is None:
        return None
    track = np.asarray(track_ptr, dtype=float).reshape(-1)
    cand = np.asarray(candidate_ptr, dtype=float).reshape(-1)
    if track.size == 0 or cand.size == 0:
        return None
    if track.shape != cand.shape:
        return None
    if not np.all(np.isfinite(track)) or not np.all(np.isfinite(cand)):
        return None
    denom = float(np.linalg.norm(track) * np.linalg.norm(cand))
    if denom <= 0.0:
        return None
    similarity = float(np.dot(track, cand) / denom)
    return float(max(0.0, min(1.0, (similarity + 1.0) * 0.5)))


def map_outputs_to_global_ids_at_frame(
    manager: object,
    outputs: Dict[str, object],
    *,
    frame_idx: Optional[int],
    allowed_gids: Optional[set[int]] = None,
    allow_new_ids: bool = True,
    max_new_ids: Optional[int] = None,
    session_id: Optional[str] = None,
) -> Dict[str, object]:
    manager._activate_global_match_session(session_id or getattr(manager, "_session_id", None))
    obj_ids = outputs.get("out_obj_ids", [])
    boxes = outputs.get("out_boxes_xywh", [])
    if boxes is None:
        return outputs
    try:
        boxes_arr = np.asarray(boxes, dtype=float)
    except Exception:
        return outputs
    if len(boxes_arr) != len(obj_ids):
        return outputs
    obj_ptr_vectors = extract_obj_ptr_vectors(outputs.get("obj_ptr"), len(obj_ids))
    local_to_global = getattr(manager, "_session_local_to_global_ids", {})
    manager._session_local_to_global_ids = local_to_global

    mapped: List[Optional[int]] = [None] * len(obj_ids)
    assignment_scores: Dict[int, Optional[float]] = {}
    used: set[int] = set()
    preferred_ids: set[int] = set()
    if frame_idx is not None:
        preferred_ids = manager._nearest_manual_seed_track_ids(int(frame_idx))
        if allowed_gids is not None:
            preferred_ids &= {int(gid) for gid in allowed_gids}

    for det_idx, local_obj_id in enumerate(obj_ids):
        try:
            local_id = int(local_obj_id)
        except Exception:
            continue
        existing_gid = local_to_global.get(local_id)
        if existing_gid is None:
            continue
        if allowed_gids is not None and int(existing_gid) not in allowed_gids:
            continue
        mapped[det_idx] = int(existing_gid)
        used.add(int(existing_gid))
        assignment_scores[int(det_idx)] = 1.0
        manager._record_global_track_observation(
            int(existing_gid),
            np.asarray(boxes_arr[det_idx], dtype=float),
            frame_idx=frame_idx,
            obj_ptr=obj_ptr_vectors[det_idx],
        )

    track_items = [
        (int(gid), np.asarray(prev_ptr, dtype=float))
        for gid, prev_ptr in sorted(getattr(manager, "_global_track_obj_ptr", {}).items())
        if allowed_gids is None or int(gid) in allowed_gids
    ]
    if track_items:
        max_gap = manager._track_match_max_gap()
        valid_cost = 1.0
        invalid_cost = 10.0
        cost_matrix = np.full(
            (len(boxes_arr), len(track_items)),
            fill_value=invalid_cost,
            dtype=float,
        )
        eligible = np.zeros_like(cost_matrix, dtype=bool)

        for det_idx, _det_box in enumerate(boxes_arr):
            if mapped[det_idx] is not None:
                continue
            candidate_ptr = obj_ptr_vectors[det_idx]
            if candidate_ptr is None:
                continue
            for track_idx, (gid, _) in enumerate(track_items):
                if gid in used:
                    continue
                last_seen = getattr(manager, "_global_track_last_seen_frame", {}).get(int(gid))
                age = 0
                if frame_idx is not None and last_seen is not None:
                    age = max(0, int(frame_idx) - int(last_seen))
                    if age > max_gap:
                        continue
                appearance_score = obj_ptr_similarity(
                    getattr(manager, "_global_track_obj_ptr", {}).get(int(gid)),
                    candidate_ptr,
                )
                if appearance_score is None:
                    continue
                score = appearance_score - min(
                    0.25, float(age) / max(1.0, float(max_gap)) * 0.12
                )
                if preferred_ids and int(gid) in preferred_ids:
                    score += 0.08
                if score < 0.35:
                    continue
                eligible[det_idx, track_idx] = True
                cost_matrix[det_idx, track_idx] = max(0.0, valid_cost - float(score))

        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for det_idx, track_idx in zip(row_ind, col_ind):
            if not eligible[det_idx, track_idx]:
                continue
            gid = int(track_items[track_idx][0])
            if gid in used:
                continue
            mapped[det_idx] = gid
            used.add(gid)
            assignment_scores[int(det_idx)] = float(max(0.0, valid_cost - cost_matrix[det_idx, track_idx]))
            try:
                local_to_global[int(obj_ids[det_idx])] = gid
            except Exception:
                pass
            manager._record_global_track_observation(
                gid,
                np.asarray(boxes_arr[det_idx], dtype=float),
                frame_idx=frame_idx,
                obj_ptr=obj_ptr_vectors[det_idx],
            )

    minted_new_ids = 0
    for idx, mapped_gid in enumerate(mapped):
        if mapped_gid is not None:
            continue
        if not allow_new_ids:
            continue
        if max_new_ids is not None and minted_new_ids >= int(max_new_ids):
            continue
        gid = manager._assign_global_track_id(
            box_xywh=np.asarray(boxes_arr[idx], dtype=float),
            candidate_obj_ptr=obj_ptr_vectors[idx],
            used_ids=used,
            frame_idx=frame_idx,
            preferred_ids=preferred_ids,
            obj_ptr=obj_ptr_vectors[idx],
        )
        if gid is None:
            continue
        used.add(gid)
        mapped[idx] = int(gid)
        assignment_scores[int(idx)] = None
        minted_new_ids += 1
        try:
            local_to_global[int(obj_ids[idx])] = int(gid)
        except Exception:
            pass
    keep_indices = [idx for idx, mapped_gid in enumerate(mapped) if mapped_gid is not None]
    filtered_outputs = manager._filter_outputs_by_indices(outputs, keep_indices)
    filtered_outputs["out_obj_ids"] = np.asarray(
        [int(v) for v in mapped if v is not None],
        dtype=np.int64,
    )
    assignments: List[Dict[str, object]] = []
    for idx in keep_indices:
        local_id: Optional[int]
        try:
            local_id = int(obj_ids[idx])
        except Exception:
            local_id = None
        assignments.append(
            {
                "local_id": local_id,
                "global_id": int(mapped[idx]) if mapped[idx] is not None else None,
                "score": assignment_scores.get(int(idx)),
            }
        )
    filtered_outputs["global_id_assignments"] = assignments
    return filtered_outputs
