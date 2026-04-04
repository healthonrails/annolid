from __future__ import annotations

from collections import deque

import numpy as np

from annolid.segmentation.SAM.sam3.track_identity import (
    extract_obj_ptr_vectors,
    map_outputs_to_global_ids_at_frame,
    normalize_obj_ptr,
    obj_ptr_similarity,
)


class _FakeManager:
    def __init__(self) -> None:
        self._session_id = "sess-1"
        self._active_global_match_session_id = None
        self._session_local_to_global_ids = {}
        self._global_track_obj_ptr = {1: np.asarray([1.0, 0.0], dtype=float)}
        self._global_track_last_seen_frame = {1: 3}
        self._global_track_last_box = {
            1: np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float)
        }
        self._global_track_history = {
            1: deque([np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float)], maxlen=4)
        }
        self._global_track_next_id = 2

    def _activate_global_match_session(self, session_id):
        normalized = str(session_id) if session_id else None
        if normalized == self._active_global_match_session_id:
            return
        self._active_global_match_session_id = normalized
        self._session_local_to_global_ids = {}

    def _nearest_manual_seed_track_ids(self, _frame_idx: int) -> set[int]:
        return set()

    def _track_match_max_gap(self) -> int:
        return 8

    def _record_global_track_observation(
        self, gid, box_xywh, *, frame_idx=None, obj_ptr=None
    ):
        self._global_track_last_box[int(gid)] = np.asarray(box_xywh, dtype=float)
        if frame_idx is not None:
            self._global_track_last_seen_frame[int(gid)] = int(frame_idx)
        self._global_track_history.setdefault(int(gid), deque(maxlen=4)).append(
            np.asarray(box_xywh, dtype=float)
        )
        if obj_ptr is not None:
            self._global_track_obj_ptr[int(gid)] = np.asarray(obj_ptr, dtype=float)

    def _assign_global_track_id(
        self,
        *,
        box_xywh,
        candidate_obj_ptr=None,
        used_ids=None,
        frame_idx=None,
        preferred_ids=None,
        obj_ptr=None,
    ):
        gid = int(self._global_track_next_id)
        self._global_track_next_id += 1
        self._record_global_track_observation(
            gid,
            box_xywh,
            frame_idx=frame_idx,
            obj_ptr=obj_ptr if obj_ptr is not None else candidate_obj_ptr,
        )
        return gid

    def _filter_outputs_by_indices(self, outputs, keep_indices):
        idx = np.asarray(keep_indices, dtype=np.int64)
        out = dict(outputs)
        for key in ("out_obj_ids", "out_boxes_xywh", "obj_ptr"):
            if key in out:
                out[key] = np.asarray(out[key])[idx]
        return out


def test_normalize_obj_ptr_returns_unit_vector() -> None:
    vec = normalize_obj_ptr([3.0, 4.0])
    assert vec is not None
    assert np.allclose(vec, [0.6, 0.8])


def test_extract_obj_ptr_vectors_rejects_rank_mismatch() -> None:
    vectors = extract_obj_ptr_vectors(np.asarray([1.0, 0.0]), expected_count=2)
    assert vectors == [None, None]


def test_map_outputs_to_global_ids_reuses_existing_track() -> None:
    manager = _FakeManager()
    outputs = {
        "out_obj_ids": np.asarray([10], dtype=np.int64),
        "out_boxes_xywh": np.asarray([[1.0, 1.0, 10.0, 10.0]], dtype=np.float32),
        "obj_ptr": np.asarray([[1.0, 0.0]], dtype=np.float32),
    }
    mapped = map_outputs_to_global_ids_at_frame(
        manager,
        outputs,
        frame_idx=4,
        session_id="sess-1",
    )
    assert mapped["out_obj_ids"].tolist() == [1]
    assert mapped["global_id_assignments"][0]["local_id"] == 10
    assert mapped["global_id_assignments"][0]["global_id"] == 1
    assert float(mapped["global_id_assignments"][0]["score"]) > 0.9
    assert manager._global_track_next_id == 2
    assert (
        obj_ptr_similarity(manager._global_track_obj_ptr[1], np.asarray([1.0, 0.0]))
        == 1.0
    )
