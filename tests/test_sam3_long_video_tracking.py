from __future__ import annotations

from collections import deque
from contextlib import contextmanager
import json
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import importlib
from PIL import Image
import pycocotools.mask as mask_utils

from annolid.segmentation.SAM.sam3 import agent_video_orchestrator
from annolid.segmentation.SAM.sam3 import adapter as sam3_adapter
from annolid.segmentation.SAM.sam3.session import Sam3SessionManager
from annolid.segmentation.SAM.sam3.video_window_inference import _iter_video_windows
from annolid.segmentation.SAM.sam3.window_refresh import (
    compute_mid_window_refresh_index,
    run_mid_window_refresh,
)


class _FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self._frames = frames
        self._index = 0
        self.set_calls: list[tuple[int, int]] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, prop: int) -> int:
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return len(self._frames)
        return 0

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def set(self, prop: int, value: int) -> bool:
        self.set_calls.append((prop, value))
        return True

    def release(self) -> None:
        self.released = True


def test_iter_video_windows_streams_sequentially_without_seeking(monkeypatch) -> None:
    frames = [np.full((2, 2, 3), fill_value=i, dtype=np.uint8) for i in range(5)]
    fake_cap = _FakeCapture(frames)
    monkeypatch.setattr(
        "annolid.segmentation.SAM.sam3.video_window_inference.cv2.VideoCapture",
        lambda _path: fake_cap,
    )

    windows = list(
        _iter_video_windows(
            video_path="/tmp/fake.mp4",
            window_size=3,
            stride=2,
        )
    )

    assert fake_cap.set_calls == []
    assert fake_cap.released is True
    assert [start for start, _, _ in windows] == [0, 2]
    assert [end for _, end, _ in windows] == [3, 5]
    assert [int(window[0][0, 0, 0]) for _, _, window in windows] == [0, 2]


def test_process_video_with_agent_disables_output_artifacts_by_default(
    monkeypatch, tmp_path
) -> None:
    captured = {}

    def _fake_run_agent_seeded_sam3_video(*, video_path, agent_cfg, tracking_cfg):
        captured["video_path"] = video_path
        captured["output_dir"] = agent_cfg.output_dir
        captured["prompt"] = agent_cfg.prompt
        return 3, 7

    monkeypatch.setattr(
        sam3_adapter,
        "run_agent_seeded_sam3_video",
        _fake_run_agent_seeded_sam3_video,
    )

    frames, masks = sam3_adapter.process_video_with_agent(
        video_path=tmp_path / "video.mp4",
        agent_prompt="mouse",
    )

    assert (frames, masks) == (3, 7)
    assert captured["video_path"] == str(tmp_path / "video.mp4")
    assert captured["prompt"] == "mouse"
    assert captured["output_dir"] is None


def test_global_track_assignment_rejects_stale_tracks() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 5
    session.sliding_window_stride = 4
    session._global_track_next_id = 3
    session._global_track_last_box = {
        1: np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float),
        2: np.asarray([20.0, 20.0, 10.0, 10.0], dtype=float),
    }
    session._global_track_last_seen_frame = {1: 0, 2: 9}
    session._global_track_history = {
        1: deque([np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float)], maxlen=4),
        2: deque([np.asarray([20.0, 20.0, 10.0, 10.0], dtype=float)], maxlen=4),
    }
    session._global_track_obj_ptr = {
        1: np.asarray([1.0, 0.0], dtype=float),
        2: np.asarray([0.0, 1.0], dtype=float),
    }

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([11, 12], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [
                    [0.5, 0.5, 10.0, 10.0],
                    [20.5, 20.5, 10.0, 10.0],
                ],
                dtype=np.float32,
            ),
            "obj_ptr": np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
        },
        frame_idx=10,
    )

    assert outputs["out_obj_ids"].tolist() == [3, 2]
    assert session._global_track_next_id == 4


def test_global_track_assignment_refreshes_match_history() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 15
    session.sliding_window_stride = 14
    session._global_track_next_id = 2
    session._global_track_last_box = {
        1: np.asarray([10.0, 0.0, 10.0, 10.0], dtype=float),
    }
    session._global_track_last_seen_frame = {1: 1}
    session._global_track_history = {
        1: deque(
            [
                np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float),
                np.asarray([10.0, 0.0, 10.0, 10.0], dtype=float),
            ],
            maxlen=4,
        )
    }
    session._global_track_obj_ptr = {1: np.asarray([1.0, 0.0], dtype=float)}

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([42], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [[30.0, 0.0, 10.0, 10.0]],
                dtype=np.float32,
            ),
            "obj_ptr": np.asarray([[1.0, 0.0]], dtype=np.float32),
        },
        frame_idx=3,
    )

    assert outputs["out_obj_ids"].tolist() == [1]
    assert np.allclose(session._global_track_last_box[1], [30.0, 0.0, 10.0, 10.0])
    assert session._global_track_last_seen_frame[1] == 3
    assert len(session._global_track_history[1]) == 3
    assert np.allclose(session._global_track_history[1][-1], [30.0, 0.0, 10.0, 10.0])


def test_global_track_assignment_survives_window_stride_gap() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 15
    session.sliding_window_stride = 14
    session._global_track_next_id = 2
    session._global_track_last_box = {
        1: np.asarray([50.0, 30.0, 12.0, 12.0], dtype=float),
    }
    session._global_track_last_seen_frame = {1: 0}
    session._global_track_history = {
        1: deque(
            [
                np.asarray([48.0, 30.0, 12.0, 12.0], dtype=float),
                np.asarray([50.0, 30.0, 12.0, 12.0], dtype=float),
            ],
            maxlen=4,
        )
    }
    session._global_track_obj_ptr = {1: np.asarray([1.0, 0.0], dtype=float)}

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([7], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [[52.0, 30.0, 12.0, 12.0]],
                dtype=np.float32,
            ),
            "obj_ptr": np.asarray([[1.0, 0.0]], dtype=np.float32),
        },
        frame_idx=14,
    )

    assert outputs["out_obj_ids"].tolist() == [1]
    assert session._global_track_next_id == 2


def test_global_track_assignment_uses_one_to_one_matching_for_ambiguous_pairs() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 15
    session.sliding_window_stride = 14
    session._global_track_next_id = 3
    session._global_track_last_box = {
        1: np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float),
        2: np.asarray([20.0, 0.0, 10.0, 10.0], dtype=float),
    }
    session._global_track_last_seen_frame = {1: 3, 2: 3}
    session._global_track_history = {
        1: deque(
            [
                np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float),
                np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float),
            ],
            maxlen=4,
        ),
        2: deque(
            [
                np.asarray([20.0, 0.0, 10.0, 10.0], dtype=float),
                np.asarray([20.0, 0.0, 10.0, 10.0], dtype=float),
            ],
            maxlen=4,
        ),
    }
    session._global_track_obj_ptr = {
        1: np.asarray([1.0, 0.0], dtype=float),
        2: np.asarray([0.0, 1.0], dtype=float),
    }

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([101, 102], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [10.0, 0.0, 10.0, 10.0],
                ],
                dtype=np.float32,
            ),
            "obj_ptr": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
        },
        frame_idx=4,
    )

    assert sorted(outputs["out_obj_ids"].tolist()) == [1, 2]
    assert outputs["out_obj_ids"].tolist() == [2, 1]
    assert session._global_track_next_id == 3


def test_text_recovery_mapping_does_not_mint_new_ids() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 15
    session.sliding_window_stride = 14
    session._global_track_next_id = 3
    session._global_track_last_box = {
        1: np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float),
        2: np.asarray([20.0, 0.0, 10.0, 10.0], dtype=float),
    }
    session._global_track_last_seen_frame = {1: 3, 2: 3}
    session._global_track_history = {
        1: deque([np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float)], maxlen=4),
        2: deque([np.asarray([20.0, 0.0, 10.0, 10.0], dtype=float)], maxlen=4),
    }
    session._global_track_obj_ptr = {
        1: np.asarray([1.0, 0.0], dtype=float),
        2: np.asarray([0.0, 1.0], dtype=float),
    }

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([99], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [[80.0, 80.0, 10.0, 10.0]],
                dtype=np.float32,
            ),
            "out_binary_masks": np.asarray(
                [np.ones((2, 2), dtype=np.uint8)],
                dtype=object,
            ),
            "obj_ptr": np.asarray([[0.5, 0.5]], dtype=np.float32),
        },
        frame_idx=4,
        allowed_gids={1, 2},
        allow_new_ids=False,
    )

    assert outputs["out_obj_ids"].size == 1
    assert outputs["out_obj_ids"].tolist()[0] in {1, 2}
    assert session._global_track_next_id == 3


def test_global_track_assignment_caps_new_ids_per_frame() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 15
    session.sliding_window_stride = 14
    session._global_track_next_id = 1
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session._global_track_obj_ptr = {}

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([10, 11, 12], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [
                    [0.0, 0.0, 10.0, 10.0],
                    [20.0, 0.0, 10.0, 10.0],
                    [40.0, 0.0, 10.0, 10.0],
                ],
                dtype=np.float32,
            ),
            "obj_ptr": np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]],
                dtype=np.float32,
            ),
            "out_binary_masks": np.asarray(
                [np.ones((2, 2), dtype=np.uint8) for _ in range(3)],
                dtype=object,
            ),
        },
        frame_idx=0,
        max_new_ids=1,
    )

    assert outputs["out_obj_ids"].tolist() == [1]
    assert session._global_track_next_id == 2


def test_global_track_assignment_prefers_nearest_manual_seed_ids() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 15
    session.sliding_window_stride = 14
    session._global_track_next_id = 3
    session._global_track_last_box = {
        1: np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float),
        2: np.asarray([20.0, 0.0, 10.0, 10.0], dtype=float),
    }
    session._global_track_last_seen_frame = {1: 3, 2: 3}
    session._global_track_history = {
        1: deque([np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float)], maxlen=4),
        2: deque([np.asarray([20.0, 0.0, 10.0, 10.0], dtype=float)], maxlen=4),
    }
    session._global_track_obj_ptr = {
        1: np.asarray([1.0, 0.0], dtype=float),
        2: np.asarray([0.0, 1.0], dtype=float),
    }
    session._manual_seed_frames = {3}
    session._frame_track_ids = {3: {2}}

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([99], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [[10.0, 0.0, 10.0, 10.0]],
                dtype=np.float32,
            ),
            "out_binary_masks": np.asarray(
                [np.ones((2, 2), dtype=np.uint8)],
                dtype=object,
            ),
            "obj_ptr": np.asarray([[0.0, 1.0]], dtype=np.float32),
        },
        frame_idx=4,
    )

    assert outputs["out_obj_ids"].tolist() == [2]
    assert session._global_track_next_id == 3


def test_global_track_assignment_prefers_obj_ptr_embedding_over_box_noise() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 15
    session.sliding_window_stride = 14
    session._global_track_next_id = 3
    session._global_track_last_box = {
        1: np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float),
        2: np.asarray([30.0, 0.0, 10.0, 10.0], dtype=float),
    }
    session._global_track_last_seen_frame = {1: 3, 2: 3}
    session._global_track_history = {
        1: deque([np.asarray([0.0, 0.0, 10.0, 10.0], dtype=float)], maxlen=4),
        2: deque([np.asarray([30.0, 0.0, 10.0, 10.0], dtype=float)], maxlen=4),
    }
    session._global_track_obj_ptr = {
        1: np.asarray([1.0, 0.0], dtype=float),
        2: np.asarray([0.0, 1.0], dtype=float),
    }

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([99], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [[27.0, 0.0, 10.0, 10.0]],
                dtype=np.float32,
            ),
            "obj_ptr": np.asarray([[1.0, 0.0]], dtype=np.float32),
        },
        frame_idx=4,
    )

    assert outputs["out_obj_ids"].tolist() == [1]
    assert np.allclose(session._global_track_obj_ptr[1], [1.0, 0.0])


def test_global_track_assignment_reuses_session_local_mapping_without_obj_ptr() -> None:
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 15
    session.sliding_window_stride = 14
    session._global_track_next_id = 1
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session._global_track_obj_ptr = {}
    session._manual_seed_frames = set()
    session._frame_track_ids = {}

    first_outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([7], dtype=np.int64),
            "out_boxes_xywh": np.asarray([[10.0, 10.0, 8.0, 8.0]], dtype=np.float32),
            "obj_ptr": np.asarray([[1.0, 0.0]], dtype=np.float32),
        },
        frame_idx=42,
        session_id="window-1",
    )
    second_outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([7], dtype=np.int64),
            "out_boxes_xywh": np.asarray([[12.0, 10.0, 8.0, 8.0]], dtype=np.float32),
        },
        frame_idx=43,
        session_id="window-1",
    )

    assert first_outputs["out_obj_ids"].tolist() == [1]
    assert second_outputs["out_obj_ids"].tolist() == [1]
    assert session._global_track_next_id == 2


def test_global_track_assignment_does_not_mint_new_id_without_obj_ptr_off_seed_frame() -> (
    None
):
    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.sliding_window_size = 15
    session.sliding_window_stride = 14
    session._global_track_next_id = 1
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session._global_track_obj_ptr = {}
    session._manual_seed_frames = set()
    session._frame_track_ids = {}

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([8], dtype=np.int64),
            "out_boxes_xywh": np.asarray([[50.0, 50.0, 8.0, 8.0]], dtype=np.float32),
            "out_binary_masks": np.asarray(
                [np.ones((2, 2), dtype=np.uint8)], dtype=object
            ),
        },
        frame_idx=43,
        session_id="window-2",
    )

    assert outputs["out_obj_ids"].size == 0
    assert session._global_track_next_id == 1


def test_mid_window_refresh_policy_is_forward_only() -> None:
    assert compute_mid_window_refresh_index(4, "forward") == 2
    assert compute_mid_window_refresh_index(4, "both") is None
    assert compute_mid_window_refresh_index(3, "forward") is None


def test_mid_window_refresh_runner_calls_callbacks_in_order() -> None:
    calls: list[tuple[str, int, int | None]] = []

    def seed_first_frame() -> None:
        calls.append(("seed", 0, None))

    def propagate_segment(start_idx: int, segment_len: int) -> tuple[int, int]:
        calls.append(("propagate", int(start_idx), int(segment_len)))
        return segment_len, segment_len + 1

    def refresh_mid_frame(refresh_idx: int) -> tuple[int, int]:
        calls.append(("refresh", int(refresh_idx), None))
        return 1, 2

    frames_processed, masks_written, refresh_idx = run_mid_window_refresh(
        6,
        "forward",
        seed_first_frame=seed_first_frame,
        propagate_segment=propagate_segment,
        refresh_mid_frame=refresh_mid_frame,
    )

    assert refresh_idx == 3
    assert frames_processed == 6
    assert masks_written == 9
    assert calls == [
        ("seed", 0, None),
        ("propagate", 0, 3),
        ("refresh", 3, None),
        ("propagate", 4, 2),
    ]


def test_window_seed_segments_are_ordered_and_text_only_falls_back_without_seeds() -> (
    None
):
    assert Sam3SessionManager._build_window_seed_segments(
        [5, 2, 5], 8, has_text_prompt=False
    ) == [(2, 5), (5, 8)]
    assert Sam3SessionManager._build_window_seed_segments(
        [], 8, has_text_prompt=True
    ) == [(0, 8)]
    assert Sam3SessionManager._build_window_seed_segments(
        [3, 6], 8, has_text_prompt=True
    ) == [(3, 6), (6, 8)]


def test_first_manual_seed_frame_is_detected_and_skips_earlier_windows(
    monkeypatch,
) -> None:
    frames_a = [np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(4)]
    frames_b = [np.full((4, 4, 3), idx + 10, dtype=np.uint8) for idx in range(4)]
    windows = [
        (0, 4, frames_a),
        (4, 8, frames_b),
    ]
    annotations = [
        {"ann_frame_idx": 6, "type": "polygon", "points": [[0, 0], [1, 0], [1, 1]]},
    ]
    session_starts: list[int] = []
    seed_calls: list[int] = []
    handle_calls: list[int] = []

    class _FakePredictor:
        def __init__(self) -> None:
            self.sessions: list[str] = []

        def start_session(self, *, resource_path: str, offload_video_to_cpu: bool):
            self.sessions.append(resource_path)
            session_starts.append(len(self.sessions))
            return {"session_id": "fake-session"}

        def close_session(self, session_id: str):
            return {"session_id": session_id}

        def propagate_in_video(
            self,
            *,
            session_id: str,
            propagation_direction: str,
            start_frame_idx: int,
            max_frame_num_to_track: int,
        ):
            yield {
                "frame_index": int(start_frame_idx),
                "outputs": {
                    "out_obj_ids": np.asarray([1], dtype=np.int64),
                    "out_boxes_xywh": np.asarray(
                        [[0.0, 0.0, 1.0, 1.0]], dtype=np.float32
                    ),
                    "out_binary_masks": np.asarray(
                        [np.ones((2, 2), dtype=np.uint8)],
                        dtype=object,
                    ),
                },
            }

    monkeypatch.setattr(
        Sam3SessionManager,
        "_resolve_runtime_device",
        lambda self, target_device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "total_frames_estimate",
        lambda self: 8,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_initialize_predictor",
        lambda self, device: _FakePredictor(),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_iter_video_windows",
        lambda self, *, window_size, stride: iter(windows),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_write_window_frames",
        lambda self, window_dir, frames, previous_count=0, shift=0: len(frames),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_reset_global_tracks",
        lambda self: None,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_carry_forward_window_state",
        lambda self, previous_state, shift: None,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_get_active_session_state",
        lambda self: None,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_ensure_prediction_json_coverage",
        lambda self, *, expected_frames: (0, 0),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_prepare_prompts",
        lambda self, frame_annotations, text_prompt: (
            int(frame_annotations[0]["ann_frame_idx"]),
            [],
            [],
            [],
            [],
            [],
            [],
            [1],
            [],
        ),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_apply_seed_prompts",
        lambda self, **kwargs: seed_calls.append(int(kwargs["frame_idx"])) or 1,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_map_outputs_to_global_ids_at_frame",
        lambda self, outputs, *, frame_idx: outputs,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_handle_frame_outputs",
        lambda self, **kwargs: handle_calls.append(int(kwargs["frame_idx"])) or (1, 1),
    )

    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.video_path = "/tmp/video.mp4"
    session.video_dir = "/tmp/video"
    session.text_prompt = None
    session.sliding_window_size = 4
    session.sliding_window_stride = 2
    session.use_sliding_window_for_text_prompt = True
    session.offload_video_to_cpu = True
    session.frame_shape = (4, 4, 3)
    session.frame_names = []
    session._predictor = _FakePredictor()
    session._predictor_device = torch.device("cpu")
    session._session_id = None
    session._stop_event = None
    session._stop_requested = False
    session.id_to_labels = {}
    session.obj_id_to_label = {}
    session._frames_processed = set()
    session._frames_with_masks = set()
    session._frame_masks = {}
    session._frame_track_ids = {}
    session._track_last_seen_frame = {}
    session._global_track_next_id = 1
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session.max_num_objects = 4
    session.multiplex_count = 4
    session.compile_model = False
    session.checkpoint_path = None
    session.default_device = "cpu"
    session.score_threshold_detection = None
    session.new_det_thresh = None
    session.max_frame_num_to_track = 4
    session.propagation_direction = "forward"

    frames_written, masks_written = session._propagate_annotations_windowed(
        annotations=annotations,
        target_device="cpu",
        propagation_direction="forward",
        max_frame_num_to_track=None,
    )

    assert Sam3SessionManager._first_manual_seed_frame(annotations) == 6
    assert len(session_starts) == 1
    assert seed_calls == [2]
    assert handle_calls == [6]
    assert frames_written == 1
    assert masks_written == 1


def test_optional_sam3_agent_modules_import_without_iopath() -> None:
    agent_core = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.agent_core"
    )
    client_sam3 = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.client_sam3"
    )
    inference = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.inference"
    )

    assert hasattr(agent_core, "agent_inference")
    assert hasattr(client_sam3, "call_sam_service")
    assert hasattr(inference, "run_single_image_inference")


def test_agent_core_default_sam_caller_binds_processor(monkeypatch) -> None:
    agent_core = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.agent_core"
    )
    client_sam3 = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.client_sam3"
    )
    captured: dict[str, object] = {}

    monkeypatch.setattr(agent_core, "_get_default_sam3_processor", lambda: "processor")

    def _fake_call_sam_service(
        sam3_processor,
        image_path: str,
        text_prompt: str,
        output_folder_path: str = "sam3_output",
    ) -> str:
        captured["sam3_processor"] = sam3_processor
        captured["image_path"] = image_path
        captured["text_prompt"] = text_prompt
        captured["output_folder_path"] = output_folder_path
        return "/tmp/fake.json"

    monkeypatch.setattr(client_sam3, "call_sam_service", _fake_call_sam_service)

    bound = agent_core._build_default_call_sam_service()
    output_path = bound(
        image_path="/tmp/frame.png",
        text_prompt="fish",
        output_folder_path="/tmp/sam-out",
    )

    assert output_path == "/tmp/fake.json"
    assert captured["sam3_processor"] == "processor"
    assert captured["image_path"] == "/tmp/frame.png"
    assert captured["text_prompt"] == "fish"
    assert captured["output_folder_path"] == "/tmp/sam-out"


def test_agent_core_prefers_env_checkpoint_for_default_processor(
    monkeypatch, tmp_path
) -> None:
    agent_core = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.agent_core"
    )
    ckpt = tmp_path / "sam3.1_multiplex.pt"
    ckpt.write_bytes(b"stub")

    monkeypatch.setenv("SAM3_CKPT_PATH", str(ckpt))
    resolved = agent_core._resolve_default_sam3_checkpoint_path()
    assert resolved == str(ckpt)


def test_agent_core_finds_cached_sam31_checkpoint(monkeypatch, tmp_path) -> None:
    agent_core = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.agent_core"
    )
    fake_home = tmp_path / "home"
    cached_ckpt = (
        fake_home
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--facebook--sam3.1"
        / "snapshots"
        / "abc123"
        / "sam3.1_multiplex.pt"
    )
    cached_ckpt.parent.mkdir(parents=True, exist_ok=True)
    cached_ckpt.write_bytes(b"stub")

    monkeypatch.delenv("SAM3_CKPT_PATH", raising=False)
    monkeypatch.setattr(agent_core.Path, "home", staticmethod(lambda: Path(fake_home)))
    resolved = agent_core._resolve_default_sam3_checkpoint_path()
    assert resolved == str(cached_ckpt)


def test_agent_core_default_processor_uses_model_device(monkeypatch) -> None:
    agent_core = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.agent_core"
    )
    captured: dict[str, object] = {}

    class _FakeModel:
        def __init__(self) -> None:
            self._param = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def parameters(self):
            yield self._param

    class _FakeProcessor:
        def __init__(
            self, model, resolution=1008, device=None, confidence_threshold=0.5
        ):
            captured["model"] = model
            captured["device"] = device

    fake_model_builder = types.ModuleType("sam3.model_builder")
    fake_model_builder.build_sam3_image_model = lambda *args, **kwargs: _FakeModel()
    fake_image_processor = types.ModuleType("sam3.model.sam3_image_processor")
    fake_image_processor.Sam3Processor = _FakeProcessor

    monkeypatch.setitem(sys.modules, "sam3.model_builder", fake_model_builder)
    monkeypatch.setitem(
        sys.modules, "sam3.model.sam3_image_processor", fake_image_processor
    )

    monkeypatch.setattr(agent_core, "_DEFAULT_SAM3_PROCESSOR", None)
    monkeypatch.setattr(
        agent_core, "_resolve_default_sam3_checkpoint_path", lambda: None
    )

    processor = agent_core._get_default_sam3_processor()

    assert isinstance(processor, _FakeProcessor)
    assert captured["device"] == "cpu"


def test_client_sam_service_raises_when_inference_fails(monkeypatch, tmp_path) -> None:
    client_sam3 = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.client_sam3"
    )

    def _fail_inference(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(client_sam3, "sam3_inference", _fail_inference)

    with pytest.raises(RuntimeError, match="SAM3 image grounding failed"):
        client_sam3.call_sam_service(
            sam3_processor=object(),
            image_path=str(tmp_path / "frame_000.png"),
            text_prompt="fish",
            output_folder_path=str(tmp_path / "sam_out"),
        )


def test_send_generate_request_serializes_native_tool_calls(monkeypatch) -> None:
    client_llm = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.client_llm"
    )
    captured: dict[str, object] = {}

    class _FakeFunction:
        name = "report_no_mask"
        arguments = "{}"

    class _FakeToolCall:
        function = _FakeFunction()

    class _FakeMessage:
        content = None
        tool_calls = [_FakeToolCall()]

    class _FakeChoice:
        message = _FakeMessage()

    class _FakeResponse:
        choices = [_FakeChoice()]

    class _FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return _FakeResponse()

    class _FakeChat:
        completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key=None, base_url=None):
            captured["api_key"] = api_key
            captured["base_url"] = base_url
            self.chat = _FakeChat()

    monkeypatch.setattr(client_llm, "OpenAI", _FakeClient)
    result = client_llm.send_generate_request(
        messages=[{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        server_url="http://localhost",
        model="moonshotai/kimi-k2.5",
        api_key="test-key",
    )

    assert result == '<tool>{"name": "report_no_mask", "parameters": {}}</tool>'
    assert captured["base_url"] == "http://localhost"
    assert captured["api_key"] == "test-key"
    assert isinstance(captured["tools"], list)
    assert len(captured["tools"]) == 4
    assert captured["tool_choice"] == "auto"


def test_agent_inference_processes_round_two_select_without_false_none_error(
    monkeypatch, tmp_path
) -> None:
    agent_core = importlib.import_module(
        "annolid.segmentation.SAM.sam3.sam3.agent.agent_core"
    )
    raw_image = tmp_path / "frame_000000000.png"
    masked_image = tmp_path / "fish.png"
    Image.new("RGB", (8, 8), color="black").save(raw_image)
    Image.new("RGB", (8, 8), color="white").save(masked_image)

    sam_json = tmp_path / "sam_out" / "fish.json"
    sam_json.parent.mkdir(parents=True, exist_ok=True)
    sam_json.write_text(
        json.dumps(
            {
                "original_image_path": str(raw_image),
                "orig_img_h": 8,
                "orig_img_w": 8,
                "pred_boxes": [[1.0, 1.0, 3.0, 3.0]],
                "pred_scores": [0.9],
                "pred_masks": ["encoded-mask"],
                "output_image_path": str(masked_image),
            }
        )
    )

    responses = iter(
        [
            '<tool>{"name":"segment_phrase","parameters":{"text_prompt":"fish"}}</tool>',
            (
                "All fish are covered.\n"
                '<tool>{"name":"select_masks_and_return","parameters":{"final_answer_masks":[1]}}</tool>'
            ),
        ]
    )

    def _fake_send_generate_request(_messages):
        return next(responses, None)

    def _fake_call_sam_service(
        *, image_path: str, text_prompt: str, output_folder_path: str
    ):
        assert image_path == str(raw_image)
        assert text_prompt == "fish"
        assert output_folder_path.endswith("sam_out")
        return str(sam_json)

    monkeypatch.setattr(
        agent_core,
        "visualize",
        lambda _outputs: Image.new("RGB", (8, 8), color="blue"),
    )

    messages, final_outputs, rendered = agent_core.agent_inference(
        img_path=str(raw_image),
        initial_text_prompt="fish",
        send_generate_request=_fake_send_generate_request,
        call_sam_service=_fake_call_sam_service,
        output_dir=str(tmp_path / "agent_out"),
    )

    assert len(final_outputs["pred_masks"]) == 1
    assert final_outputs["pred_boxes"] == [[1.0, 1.0, 3.0, 3.0]]
    assert isinstance(rendered, Image.Image)
    assert any(
        msg.get("role") == "assistant"
        and any(
            isinstance(c, dict)
            and c.get("type") == "text"
            and "select_masks_and_return" in c.get("text", "")
            for c in msg.get("content", [])
        )
        for msg in messages
    )


def test_sam3_renderer_falls_back_without_iopath(tmp_path) -> None:
    image_path = tmp_path / "image.png"
    Image.new("RGB", (10, 10), color="white").save(image_path)

    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:7, 3:8] = 1
    rle = mask_utils.encode(np.asfortranarray(mask))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")

    render = importlib.import_module("annolid.segmentation.SAM.sam3.sam3.agent.render")
    full = render.visualize(
        {
            "original_image_path": str(image_path),
            "orig_img_h": 10,
            "orig_img_w": 10,
            "pred_boxes": [[3.0, 2.0, 5.0, 5.0]],
            "pred_masks": [counts],
            "pred_scores": [0.9],
        }
    )
    zoom, zoomed = render.visualize(
        {
            "original_image_path": str(image_path),
            "orig_img_h": 10,
            "orig_img_w": 10,
            "pred_boxes": [[3.0, 2.0, 5.0, 5.0]],
            "pred_masks": [counts],
            "pred_scores": [0.9],
        },
        zoom_in_index=0,
    )

    assert full.size == (10, 10)
    assert zoom.size[0] > 0 and zoom.size[1] > 0
    assert zoomed.size[0] > 0 and zoomed.size[1] > 0


def test_agent_seeded_video_keeps_cross_window_state(monkeypatch, tmp_path) -> None:
    frames_a = [
        np.full((4, 4, 3), 1, dtype=np.uint8),
        np.full((4, 4, 3), 2, dtype=np.uint8),
        np.full((4, 4, 3), 3, dtype=np.uint8),
        np.full((4, 4, 3), 4, dtype=np.uint8),
    ]
    frames_b = [
        np.full((4, 4, 3), 5, dtype=np.uint8),
        np.full((4, 4, 3), 6, dtype=np.uint8),
        np.full((4, 4, 3), 7, dtype=np.uint8),
        np.full((4, 4, 3), 8, dtype=np.uint8),
    ]
    windows = [
        (0, 4, frames_a),
        (10, 14, frames_b),
    ]
    created_sessions: list["_FakeSession"] = []
    agent_calls: list[str] = []

    class _FakeSession:
        def __init__(self, *args, **kwargs) -> None:
            self._frames_processed: set[int] = set()
            self._frames_with_masks: set[int] = set()
            self._frame_masks: dict[int, dict[str, np.ndarray]] = {}
            self.obj_id_to_label: dict[str, str] = {}
            self.id_to_labels: dict[int, str] = kwargs.get("id_to_labels", {})
            self._session_id = None
            self.prompt_calls: list[tuple[int, int]] = []
            created_sessions.append(self)

        @contextmanager
        def _session_scope(self, target_device=None, *, auto_close=True):
            self._session_id = "fake-session"
            yield self._session_id
            if auto_close:
                self._session_id = None

        def _reset_action_history_if_supported(self) -> None:
            return None

        def add_prompt_boxes_abs(self, frame_idx, boxes_abs, box_labels, **kwargs):
            self.prompt_calls.append((int(frame_idx), len(boxes_abs)))
            self._frames_processed.add(int(frame_idx))
            self._frames_with_masks.add(int(frame_idx))
            self._frame_masks.setdefault(int(frame_idx), {})
            return {"outputs": {"out_obj_ids": np.asarray(box_labels, dtype=np.int64)}}

        def _map_outputs_to_global_ids_at_frame(self, outputs, *, frame_idx):
            return outputs

        def propagate(
            self, *, start_frame_idx, propagation_direction, max_frame_num_to_track
        ):
            start = int(start_frame_idx)
            end = min(start + 2, start + int(max_frame_num_to_track))
            for frame_idx in range(start, end):
                self._frames_processed.add(frame_idx)
                self._frames_with_masks.add(frame_idx)
                self._frame_masks.setdefault(frame_idx, {})["1"] = np.ones(
                    (2, 2), dtype=np.uint8
                )
            return max(0, end - start), max(0, end - start)

    monkeypatch.setattr(
        agent_video_orchestrator,
        "Sam3SessionManager",
        _FakeSession,
    )
    monkeypatch.setattr(
        agent_video_orchestrator,
        "_iter_video_windows",
        lambda **kwargs: iter(windows),
    )
    monkeypatch.setattr(
        agent_video_orchestrator,
        "_run_agent_on_frame",
        lambda frame_path, agent_cfg: (
            agent_calls.append(str(frame_path))
            or (
                [[0.0, 0.0, 1.0, 1.0], [2.0, 2.0, 3.0, 3.0]]
                if frame_path.name.endswith("000000002.png")
                or frame_path.name.endswith("000000012.png")
                else [[0.0, 0.0, 1.0, 1.0]]
            ),
            [0.99, 0.95]
            if frame_path.name.endswith("000000002.png")
            or frame_path.name.endswith("000000012.png")
            else [0.99],
        ),
    )

    agent_cfg = agent_video_orchestrator.AgentConfig(
        prompt="mouse",
        window_size=4,
        stride=1,
        output_dir=str(tmp_path / "agent"),
    )
    tracking_cfg = agent_video_orchestrator.TrackingConfig(
        checkpoint_path="checkpoint.pt",
        device="cpu",
        max_frame_num_to_track=2,
    )

    total_frames, total_masks = agent_video_orchestrator.run_agent_seeded_sam3_video(
        video_path=str(tmp_path / "video.mp4"),
        agent_cfg=agent_cfg,
        tracking_cfg=tracking_cfg,
    )

    assert total_frames == 8
    assert total_masks == 8
    assert len(agent_calls) == 4
    assert any(path.endswith("000000002.png") for path in agent_calls)
    assert any(path.endswith("000000012.png") for path in agent_calls)
    assert len(created_sessions) == 1
    session = created_sessions[0]
    assert session._frames_processed == {0, 1, 2, 3, 10, 11, 12, 13}
    assert session._frames_with_masks == {0, 1, 2, 3, 10, 11, 12, 13}
    assert 0 in session._frame_masks and 2 in session._frame_masks
    assert 10 in session._frame_masks and 12 in session._frame_masks
    assert session.prompt_calls == [(0, 1), (2, 1), (10, 1), (12, 1)]


def test_text_windowed_refresh_seeds_mid_window(monkeypatch, tmp_path) -> None:
    frames = [np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(4)]
    prompt_calls: list[tuple[int, bool]] = []
    propagate_calls: list[tuple[int, int]] = []
    handle_calls: list[int] = []
    created_predictors: list[object] = []

    class _FakePredictor:
        def __init__(self) -> None:
            self.sessions: list[str] = []

        def start_session(self, *, resource_path: str, offload_video_to_cpu: bool):
            self.sessions.append(resource_path)
            return {"session_id": "fake-session"}

        def close_session(self, session_id: str):
            return {"session_id": session_id}

        def propagate_in_video(
            self,
            *,
            session_id: str,
            propagation_direction: str,
            start_frame_idx: int,
            max_frame_num_to_track: int,
        ):
            propagate_calls.append((int(start_frame_idx), int(max_frame_num_to_track)))
            if int(start_frame_idx) == 0:
                yield {
                    "frame_index": 1,
                    "outputs": {
                        "out_obj_ids": np.asarray([1], dtype=np.int64),
                        "out_boxes_xywh": np.asarray(
                            [[0.0, 0.0, 1.0, 1.0]], dtype=np.float32
                        ),
                        "out_binary_masks": np.asarray(
                            [np.ones((2, 2), dtype=np.uint8)],
                            dtype=object,
                        ),
                    },
                }
            else:
                yield {
                    "frame_index": 3,
                    "outputs": {
                        "out_obj_ids": np.asarray([1], dtype=np.int64),
                        "out_boxes_xywh": np.asarray(
                            [[0.0, 0.0, 1.0, 1.0]], dtype=np.float32
                        ),
                        "out_binary_masks": np.asarray(
                            [np.ones((2, 2), dtype=np.uint8)],
                            dtype=object,
                        ),
                    },
                }

    def _fake_session_scope(self, target_device=None, *, auto_close=True):
        @contextmanager
        def _ctx():
            self._session_id = "fake-session"
            yield self._session_id
            if auto_close:
                self._session_id = None

        return _ctx()

    def _fake_execute_prompt_transaction(
        self,
        *,
        session_id,
        frame_idx,
        text,
        boxes,
        box_labels,
        mask_inputs,
        mask_labels,
        points,
        point_labels,
        obj_id,
    ):
        prompt_calls.append((int(frame_idx), boxes is not None))
        obj_ids = [1] if int(frame_idx) == 0 else [1, 2]
        return {
            "outputs": {
                "out_obj_ids": np.asarray(obj_ids, dtype=np.int64),
                "out_boxes_xywh": np.asarray(
                    [[0.0, 0.0, 1.0, 1.0] for _ in obj_ids],
                    dtype=np.float32,
                ),
                "out_binary_masks": np.asarray(
                    [np.ones((2, 2), dtype=np.uint8) for _ in obj_ids],
                    dtype=object,
                ),
            }
        }

    def _fake_handle_frame_outputs(
        self,
        *,
        frame_idx,
        outputs,
        total_frames=None,
        yielded_frames=0,
        label_hints=None,
        apply_score_threshold=True,
        merge_existing=False,
    ):
        handle_calls.append(int(frame_idx))
        obj_ids = list(np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64))
        self._frames_processed.add(int(frame_idx))
        self._frame_track_ids[int(frame_idx)] = set(int(v) for v in obj_ids)
        if obj_ids:
            self._frames_with_masks.add(int(frame_idx))
            self._frame_masks[int(frame_idx)] = {
                str(int(obj_id)): np.ones((2, 2), dtype=np.uint8) for obj_id in obj_ids
            }
            for obj_id in obj_ids:
                self._track_last_seen_frame[int(obj_id)] = int(frame_idx)
        return len(obj_ids), len(obj_ids)

    class _FakeVideoWindowPredictor:
        pass

    windows = [
        (0, 4, frames),
    ]

    monkeypatch.setattr(
        Sam3SessionManager,
        "_resolve_runtime_device",
        lambda self, target_device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "total_frames_estimate",
        lambda self: 8,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_initialize_predictor",
        lambda self, device: created_predictors.append(_FakePredictor())
        or created_predictors[-1],
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_session_scope",
        _fake_session_scope,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_reset_action_history_if_supported",
        lambda self: None,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_execute_prompt_transaction",
        _fake_execute_prompt_transaction,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_handle_frame_outputs",
        _fake_handle_frame_outputs,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_ensure_prediction_json_coverage",
        lambda self, *, expected_frames: (0, 0),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_iter_video_windows",
        lambda self, *, window_size, stride: iter(windows),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_write_window_frames",
        lambda self, window_dir, frames, previous_count=0, shift=0: len(frames),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_reacquire_frames_with_visual_and_text",
        lambda self, frame_indices, target_device=None: None,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_derive_boxes_from_neighbor_masks",
        lambda self, frame_idx, max_boxes=4: pytest.fail(
            "text-window seeding must not use neighbor-mask carry boxes"
        ),
    )

    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.video_path = str(tmp_path / "video.mp4")
    session.video_dir = str(tmp_path)
    session.text_prompt = "mouse"
    session.sliding_window_size = 4
    session.sliding_window_stride = 2
    session.use_sliding_window_for_text_prompt = True
    session.offload_video_to_cpu = True
    session.frame_shape = (4, 4, 3)
    session.frame_names = []
    session._predictor = None
    session._predictor_device = None
    session._session_id = None
    session._stop_event = None
    session._stop_requested = False
    session.id_to_labels = {}
    session.obj_id_to_label = {}
    session._frames_processed = set()
    session._frames_with_masks = set()
    session._frame_masks = {}
    session._frame_track_ids = {}
    session._track_last_seen_frame = {}
    session._global_track_next_id = 1
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session.max_num_objects = 4
    session.multiplex_count = 4
    session.compile_model = False
    session.checkpoint_path = None
    session.default_device = "cpu"
    session.score_threshold_detection = None
    session.new_det_thresh = None
    session.max_frame_num_to_track = 4
    session.propagation_direction = "forward"

    total_frames, total_masks = session._propagate_text_prompt_windowed(
        text_prompt="mouse",
        target_device="cpu",
    )

    assert total_frames == 4
    assert total_masks == 5
    assert prompt_calls == [(0, False), (2, False)]
    assert propagate_calls == [(0, 2), (3, 1)]
    assert handle_calls == [0, 1, 2, 3]
    assert len(created_predictors) == 1


def test_text_windowed_refresh_allows_new_animals_in_later_windows(
    monkeypatch, tmp_path
) -> None:
    frames = [np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(4)]
    prompt_calls: list[tuple[int, int, int, bool]] = []
    mapped_calls: list[
        tuple[int, tuple[int, ...] | None, bool, int | None, tuple[int, ...]]
    ] = []
    handle_calls: list[tuple[int, int]] = []
    created_predictors: list[object] = []
    seed_prompt_count = 0

    class _FakePredictor:
        def __init__(self) -> None:
            self.sessions: list[str] = []

        def start_session(self, *, resource_path: str, offload_video_to_cpu: bool):
            self.sessions.append(resource_path)
            return {"session_id": "fake-session"}

        def close_session(self, session_id: str):
            return {"session_id": session_id}

        def propagate_in_video(
            self,
            *,
            session_id: str,
            propagation_direction: str,
            start_frame_idx: int,
            max_frame_num_to_track: int,
        ):
            if False:
                yield {}

    def _fake_session_scope(self, target_device=None, *, auto_close=True):
        @contextmanager
        def _ctx():
            self._session_id = "fake-session"
            yield self._session_id
            if auto_close:
                self._session_id = None

        return _ctx()

    def _fake_execute_prompt_transaction(
        self,
        *,
        session_id,
        frame_idx,
        text,
        boxes,
        box_labels,
        mask_inputs,
        mask_labels,
        points,
        point_labels,
        obj_id,
    ):
        nonlocal seed_prompt_count
        prompt_calls.append(
            (
                int(frame_idx),
                0 if boxes is None else len(boxes),
                0 if mask_inputs is None else len(mask_inputs),
                bool(text),
            )
        )
        if int(frame_idx) == 0:
            seed_prompt_count += 1
            if seed_prompt_count == 1:
                obj_ids = [1]
            elif seed_prompt_count == 2:
                obj_ids = [1, 2]
            else:
                obj_ids = []
        else:
            obj_ids = []
        return {
            "outputs": {
                "out_obj_ids": np.asarray(obj_ids, dtype=np.int64),
                "out_boxes_xywh": np.asarray(
                    [[0.0, 0.0, 1.0, 1.0] for _ in obj_ids],
                    dtype=np.float32,
                ),
                "out_binary_masks": np.asarray(
                    [np.ones((2, 2), dtype=np.uint8) for _ in obj_ids],
                    dtype=object,
                ),
            }
        }

    def _fake_map_outputs(
        self,
        outputs,
        *,
        frame_idx,
        allowed_gids=None,
        allow_new_ids=True,
        max_new_ids=None,
        session_id=None,
    ):
        obj_ids = tuple(
            int(v) for v in np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64)
        )
        mapped_calls.append(
            (
                int(frame_idx) if frame_idx is not None else -1,
                tuple(sorted(int(v) for v in allowed_gids)) if allowed_gids else None,
                bool(allow_new_ids),
                None if max_new_ids is None else int(max_new_ids),
                obj_ids,
            )
        )
        return outputs

    def _fake_handle_frame_outputs(
        self,
        *,
        frame_idx,
        outputs,
        total_frames=None,
        yielded_frames=0,
        label_hints=None,
        apply_score_threshold=True,
        merge_existing=False,
    ):
        obj_ids = list(np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64))
        handle_calls.append((int(frame_idx), len(obj_ids)))
        self._frames_processed.add(int(frame_idx))
        self._frame_track_ids[int(frame_idx)] = set(int(v) for v in obj_ids)
        if obj_ids:
            self._frames_with_masks.add(int(frame_idx))
            self._frame_masks[int(frame_idx)] = {
                str(int(obj_id)): np.ones((2, 2), dtype=np.uint8) for obj_id in obj_ids
            }
            for obj_id in obj_ids:
                self._track_last_seen_frame[int(obj_id)] = int(frame_idx)
        return len(obj_ids), len(obj_ids)

    monkeypatch.setattr(
        Sam3SessionManager,
        "_resolve_runtime_device",
        lambda self, target_device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "total_frames_estimate",
        lambda self: 8,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_initialize_predictor",
        lambda self, device: created_predictors.append(_FakePredictor())
        or created_predictors[-1],
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_session_scope",
        _fake_session_scope,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_reset_action_history_if_supported",
        lambda self: None,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_execute_prompt_transaction",
        _fake_execute_prompt_transaction,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_map_outputs_to_global_ids_at_frame",
        _fake_map_outputs,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_handle_frame_outputs",
        _fake_handle_frame_outputs,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_ensure_prediction_json_coverage",
        lambda self, *, expected_frames: (0, 0),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_iter_video_windows",
        lambda self, *, window_size, stride: iter([(0, 4, frames), (4, 8, frames)]),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_write_window_frames",
        lambda self, window_dir, frames, previous_count=0, shift=0: len(frames),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_reacquire_frames_with_visual_and_text",
        lambda self, frame_indices, target_device=None: None,
    )

    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.video_path = str(tmp_path / "video.mp4")
    session.video_dir = str(tmp_path)
    session.text_prompt = "mouse"
    session.sliding_window_size = 4
    session.sliding_window_stride = 2
    session.use_sliding_window_for_text_prompt = True
    session.offload_video_to_cpu = True
    session.frame_shape = (4, 4, 3)
    session.frame_names = []
    session._predictor = None
    session._predictor_device = None
    session._session_id = None
    session._stop_event = None
    session._stop_requested = False
    session.id_to_labels = {}
    session.obj_id_to_label = {}
    session._frames_processed = set()
    session._frames_with_masks = set()
    session._frame_masks = {}
    session._frame_track_ids = {}
    session._track_last_seen_frame = {}
    session._global_track_next_id = 1
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session.max_num_objects = 4
    session.multiplex_count = 4
    session.compile_model = False
    session.checkpoint_path = None
    session.default_device = "cpu"
    session.score_threshold_detection = None
    session.new_det_thresh = None
    session.max_frame_num_to_track = 4
    session.propagation_direction = "forward"

    total_frames, total_masks = session._propagate_text_prompt_windowed(
        text_prompt="mouse",
        target_device="cpu",
    )

    assert total_frames == 4
    assert total_masks == 3
    assert len(prompt_calls) == 4
    assert prompt_calls[2][1] == 0
    assert prompt_calls[2][2] == 0
    assert any(
        call[0] == 4 and call[1] == (1,) and call[2] is True and call[3] == 1
        for call in mapped_calls
    )
    assert any(call[0] == 4 and call[4] == (1, 2) for call in mapped_calls)
    assert handle_calls == [(0, 1), (2, 0), (4, 2), (6, 0)]
    assert len(created_predictors) == 1


def test_text_windowed_propagation_reuses_recent_masks_for_empty_outputs(
    monkeypatch, tmp_path
) -> None:
    frames = [np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(4)]
    handle_calls: list[tuple[int, tuple[int, ...]]] = []
    created_predictors: list[object] = []

    class _FakePredictor:
        def __init__(self) -> None:
            self.sessions: list[str] = []

        def start_session(self, *, resource_path: str, offload_video_to_cpu: bool):
            self.sessions.append(resource_path)
            return {"session_id": "fake-session"}

        def close_session(self, session_id: str):
            return {"session_id": session_id}

        def propagate_in_video(
            self,
            *,
            session_id: str,
            propagation_direction: str,
            start_frame_idx: int,
            max_frame_num_to_track: int,
        ):
            yield {
                "frame_index": 1,
                "outputs": {
                    "out_obj_ids": np.asarray([], dtype=np.int64),
                    "out_boxes_xywh": np.asarray([], dtype=np.float32),
                    "out_binary_masks": np.asarray([], dtype=object),
                },
            }

    def _fake_execute_prompt_transaction(
        self,
        *,
        session_id,
        frame_idx,
        text,
        boxes,
        box_labels,
        mask_inputs,
        mask_labels,
        points,
        point_labels,
        obj_id,
    ):
        return {
            "outputs": {
                "out_obj_ids": np.asarray([1], dtype=np.int64),
                "out_boxes_xywh": np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                "out_binary_masks": np.asarray(
                    [np.ones((2, 2), dtype=np.uint8)],
                    dtype=object,
                ),
            }
        }

    def _fake_map_outputs(
        self,
        outputs,
        *,
        frame_idx,
        allowed_gids=None,
        allow_new_ids=True,
        max_new_ids=None,
        session_id=None,
    ):
        return outputs

    def _fake_handle_frame_outputs(
        self,
        *,
        frame_idx,
        outputs,
        total_frames=None,
        yielded_frames=0,
        label_hints=None,
        apply_score_threshold=True,
        merge_existing=False,
    ):
        obj_ids = tuple(
            int(v) for v in np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64)
        )
        handle_calls.append((int(frame_idx), obj_ids))
        self._frames_processed.add(int(frame_idx))
        self._frame_track_ids[int(frame_idx)] = set(obj_ids)
        if obj_ids:
            self._frames_with_masks.add(int(frame_idx))
            self._frame_masks[int(frame_idx)] = {
                str(int(obj_id)): np.ones((2, 2), dtype=np.uint8) for obj_id in obj_ids
            }
            for obj_id in obj_ids:
                self._track_last_seen_frame[int(obj_id)] = int(frame_idx)
        return len(obj_ids), len(obj_ids)

    monkeypatch.setattr(
        Sam3SessionManager,
        "_resolve_runtime_device",
        lambda self, target_device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "total_frames_estimate",
        lambda self: 4,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_initialize_predictor",
        lambda self, device: created_predictors.append(_FakePredictor())
        or created_predictors[-1],
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_reset_action_history_if_supported",
        lambda self: None,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_execute_prompt_transaction",
        _fake_execute_prompt_transaction,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_map_outputs_to_global_ids_at_frame",
        _fake_map_outputs,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_handle_frame_outputs",
        _fake_handle_frame_outputs,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_ensure_prediction_json_coverage",
        lambda self, *, expected_frames: (0, 0),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_iter_video_windows",
        lambda self, *, window_size, stride: iter([(0, 4, frames)]),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_write_window_frames",
        lambda self, window_dir, frames, previous_count=0, shift=0: len(frames),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_reacquire_frames_with_visual_and_text",
        lambda self, frame_indices, target_device=None: None,
    )

    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.video_path = str(tmp_path / "video.mp4")
    session.video_dir = str(tmp_path)
    session.text_prompt = "mouse"
    session.sliding_window_size = 4
    session.sliding_window_stride = 2
    session.use_sliding_window_for_text_prompt = True
    session.offload_video_to_cpu = True
    session.frame_shape = (4, 4, 3)
    session.frame_names = []
    session._predictor = None
    session._predictor_device = None
    session._session_id = None
    session._stop_event = None
    session._stop_requested = False
    session.id_to_labels = {}
    session.obj_id_to_label = {}
    session._frames_processed = set()
    session._frames_with_masks = set()
    session._frame_masks = {}
    session._frame_track_ids = {}
    session._track_last_seen_frame = {}
    session._global_track_next_id = 1
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session.max_num_objects = 4
    session.multiplex_count = 4
    session.compile_model = False
    session.checkpoint_path = None
    session.default_device = "cpu"
    session.score_threshold_detection = None
    session.new_det_thresh = None
    session.max_frame_num_to_track = 4
    session.propagation_direction = "forward"

    total_frames, total_masks = session._propagate_text_prompt_windowed(
        text_prompt="mouse",
        target_device="cpu",
    )

    assert total_frames == 3
    assert total_masks == 3
    assert handle_calls == [(0, (1,)), (1, (1,)), (2, (1,))]
    assert len(created_predictors) == 1


def test_annotated_window_falls_back_to_text_when_no_local_annotations(
    monkeypatch, tmp_path
) -> None:
    frames = [np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(4)]
    prompt_calls: list[tuple[int, bool]] = []
    created_predictors: list[object] = []
    carry_forward_calls: list[int] = []

    class _FakePredictor:
        def __init__(self) -> None:
            self.sessions: list[str] = []

        def start_session(self, *, resource_path: str, offload_video_to_cpu: bool):
            self.sessions.append(resource_path)
            return {"session_id": "fake-session"}

        def close_session(self, session_id: str):
            return {"session_id": session_id}

        def propagate_in_video(
            self,
            *,
            session_id: str,
            propagation_direction: str,
            start_frame_idx: int,
            max_frame_num_to_track: int,
        ):
            yield {
                "frame_index": int(start_frame_idx),
                "outputs": {
                    "out_obj_ids": np.asarray([1], dtype=np.int64),
                    "out_boxes_xywh": np.asarray(
                        [[0.0, 0.0, 1.0, 1.0]], dtype=np.float32
                    ),
                    "out_binary_masks": np.asarray(
                        [np.ones((2, 2), dtype=np.uint8)],
                        dtype=object,
                    ),
                },
            }

    def _fake_session_scope(self, target_device=None, *, auto_close=True):
        @contextmanager
        def _ctx():
            self._session_id = "fake-session"
            yield self._session_id
            if auto_close:
                self._session_id = None

        return _ctx()

    def _fake_execute_prompt_transaction(
        self,
        *,
        session_id,
        frame_idx,
        text,
        boxes,
        box_labels,
        mask_inputs,
        mask_labels,
        points,
        point_labels,
        point_obj_ids=None,
        obj_id=None,
        label_hints=None,
        record_outputs=True,
        merge_existing_on_record=False,
    ):
        prompt_calls.append((int(frame_idx), bool(boxes)))
        return {
            "outputs": {
                "out_obj_ids": np.asarray([1], dtype=np.int64),
                "out_boxes_xywh": np.asarray([[0.0, 0.0, 1.0, 1.0]], dtype=np.float32),
                "out_binary_masks": np.asarray(
                    [np.ones((2, 2), dtype=np.uint8)],
                    dtype=object,
                ),
            }
        }

    def _fake_handle_frame_outputs(
        self,
        *,
        frame_idx,
        outputs,
        total_frames=None,
        yielded_frames=0,
        label_hints=None,
        apply_score_threshold=True,
        merge_existing=False,
    ):
        self._frames_processed.add(int(frame_idx))
        self._frames_with_masks.add(int(frame_idx))
        self._frame_track_ids[int(frame_idx)] = {1}
        self._frame_masks[int(frame_idx)] = {"1": np.ones((2, 2), dtype=np.uint8)}
        return 1, 1

    windows = [(0, 4, frames), (4, 8, frames)]

    monkeypatch.setattr(
        Sam3SessionManager,
        "_resolve_runtime_device",
        lambda self, target_device: torch.device("cpu"),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "total_frames_estimate",
        lambda self: 4,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_initialize_predictor",
        lambda self, device: created_predictors.append(_FakePredictor())
        or created_predictors[-1],
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_session_scope",
        _fake_session_scope,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_reset_action_history_if_supported",
        lambda self: None,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_execute_prompt_transaction",
        _fake_execute_prompt_transaction,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_handle_frame_outputs",
        _fake_handle_frame_outputs,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_ensure_prediction_json_coverage",
        lambda self, *, expected_frames: (0, 0),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_iter_video_windows",
        lambda self, *, window_size, stride: iter(windows),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_write_window_frames",
        lambda self, window_dir, frames, previous_count=0, shift=0: len(frames),
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_reacquire_frames_with_visual_and_text",
        lambda self, frame_indices, target_device=None: None,
    )
    monkeypatch.setattr(
        Sam3SessionManager,
        "_carry_forward_window_state",
        lambda self, previous_state, shift: carry_forward_calls.append(int(shift)),
    )

    session = Sam3SessionManager.__new__(Sam3SessionManager)
    session.video_path = str(tmp_path / "video.mp4")
    session.video_dir = str(tmp_path)
    session.text_prompt = "mouse"
    session.sliding_window_size = 4
    session.sliding_window_stride = 2
    session.use_sliding_window_for_text_prompt = True
    session.offload_video_to_cpu = True
    session.frame_shape = (4, 4, 3)
    session.frame_names = []
    session._predictor = None
    session._predictor_device = None
    session._session_id = None
    session._stop_event = None
    session._stop_requested = False
    session.id_to_labels = {}
    session.obj_id_to_label = {}
    session._frames_processed = set()
    session._frames_with_masks = set()
    session._frame_masks = {}
    session._frame_track_ids = {}
    session._track_last_seen_frame = {}
    session._global_track_next_id = 1
    session._global_track_last_box = {}
    session._global_track_last_seen_frame = {}
    session._global_track_history = {}
    session.max_num_objects = 4
    session.multiplex_count = 4
    session.compile_model = False
    session.checkpoint_path = None
    session.default_device = "cpu"
    session.score_threshold_detection = None
    session.new_det_thresh = None
    session.max_frame_num_to_track = 4
    session.propagation_direction = "forward"

    total_frames, total_masks = session._propagate_annotations_windowed(
        annotations=[
            {
                "type": "box",
                "ann_frame_idx": 2,
                "box": [1, 1, 2, 2],
                "labels": [1],
                "obj_id": 1,
            }
        ],
        target_device="cpu",
        propagation_direction="forward",
        max_frame_num_to_track=4,
    )

    assert total_frames == 2
    assert total_masks == 2
    assert prompt_calls[0] == (2, True)
    assert prompt_calls[1][0] == 0
    assert len(created_predictors) == 1
    assert carry_forward_calls == []


def test_vlm_boundary_box_labels_reuse_boundary_track_ids() -> None:
    labels, hints = agent_video_orchestrator._resolve_vlm_boundary_box_labels(
        candidate_boxes_xyxy=[
            [0.0, 0.0, 20.0, 20.0],
            [60.0, 60.0, 90.0, 90.0],
        ],
        boundary_boxes_norm_xywh=[
            [0.58, 0.58, 0.3, 0.3],
            [0.0, 0.0, 0.2, 0.2],
        ],
        boundary_track_ids=[3, 7],
        frame_width=100.0,
        frame_height=100.0,
        iou_threshold=0.2,
        prompt_prefix="mouse",
        id_to_labels={3: "mouse_3", 7: "mouse_7"},
    )

    assert labels == [7, 3]
    assert hints == ["mouse_7", "mouse_3"]


def test_agent_orchestrator_applies_boundary_bundle_to_seed_mapping(
    monkeypatch, tmp_path
) -> None:
    windows = [
        (0, 4, [np.full((40, 40, 3), 0, dtype=np.uint8) for _ in range(4)]),
        (4, 8, [np.full((40, 40, 3), 1, dtype=np.uint8) for _ in range(4)]),
    ]
    created_sessions: list[object] = []

    class _Bundle:
        def __init__(self) -> None:
            self.boxes = [[0.2, 0.25, 0.5, 0.5]]
            self.track_ids = [11]
            self.box_labels = [1]
            self.mask_inputs = []
            self.mask_labels = []
            self.label_hints = ["mouse_11"]

    class _FakeSession:
        def __init__(self, video_dir, id_to_labels, config):
            self.video_dir = video_dir
            self.id_to_labels = dict(id_to_labels)
            self.id_to_labels.setdefault(11, "mouse_11")
            self.config = config
            self._session_id = "sess"
            self._frames_processed = set()
            self._frames_with_masks = set()
            self._frame_masks = {}
            self.seed_calls: list[dict] = []
            created_sessions.append(self)

        @contextmanager
        def _session_scope(self, target_device=None, auto_close=True):
            yield self._session_id

        def _reset_action_history_if_supported(self) -> None:
            return None

        def _build_boundary_reseed_prompt_bundle(
            self,
            *,
            frame_idx,
            frame_width,
            frame_height,
            source_frame_idx=None,
        ):
            if int(frame_idx) <= 0:
                return None
            return _Bundle()

        def _carry_forward_window_state(self, previous_window_state, shift):
            return None

        def _get_active_session_state(self):
            return {"ok": True}

        def _map_outputs_to_global_ids_at_frame(self, outputs, *, frame_idx):
            return outputs

        def add_prompt_boxes_abs(self, frame_idx, boxes_abs, box_labels, **kwargs):
            self.seed_calls.append(
                {
                    "frame_idx": int(frame_idx),
                    "boxes_abs": [list(v) for v in boxes_abs],
                    "box_labels": [int(v) for v in box_labels],
                    "boundary_bundle": kwargs.get("boundary_bundle"),
                    "boundary_allowed_gids": kwargs.get("boundary_allowed_gids"),
                }
            )
            self._frames_processed.add(int(frame_idx))
            self._frames_with_masks.add(int(frame_idx))
            self._frame_masks.setdefault(int(frame_idx), {})["1"] = np.ones(
                (2, 2), dtype=np.uint8
            )
            return {"outputs": {"out_obj_ids": np.asarray(box_labels, dtype=np.int64)}}

        def propagate(
            self, *, start_frame_idx, propagation_direction, max_frame_num_to_track
        ):
            start = int(start_frame_idx)
            stop = start + int(max_frame_num_to_track)
            for frame_idx in range(start, stop):
                self._frames_processed.add(frame_idx)
                self._frames_with_masks.add(frame_idx)
                self._frame_masks.setdefault(frame_idx, {})["1"] = np.ones(
                    (2, 2), dtype=np.uint8
                )
            span = max(0, stop - start)
            return span, span

    monkeypatch.setattr(
        agent_video_orchestrator,
        "Sam3SessionManager",
        _FakeSession,
    )
    monkeypatch.setattr(
        agent_video_orchestrator,
        "_iter_video_windows",
        lambda **kwargs: iter(windows),
    )
    monkeypatch.setattr(
        agent_video_orchestrator,
        "_run_agent_on_frame",
        lambda frame_path, agent_cfg: (
            [[8.0, 10.0, 28.0, 30.0]]
            if frame_path.name.endswith("000000004.png")
            else [[1.0, 1.0, 11.0, 11.0]],
            [0.99],
        ),
    )

    agent_cfg = agent_video_orchestrator.AgentConfig(
        prompt="mouse",
        window_size=4,
        stride=4,
        output_dir=str(tmp_path / "agent"),
    )
    tracking_cfg = agent_video_orchestrator.TrackingConfig(
        checkpoint_path="checkpoint.pt",
        device="cpu",
        max_frame_num_to_track=1,
        use_explicit_window_reseed=True,
        use_vlm_boundary_id_correction=True,
        vlm_boundary_match_iou_threshold=0.2,
    )

    frames, masks = agent_video_orchestrator.run_agent_seeded_sam3_video(
        video_path=str(tmp_path / "video.mp4"),
        agent_cfg=agent_cfg,
        tracking_cfg=tracking_cfg,
    )

    assert frames == 8
    assert masks == 7
    assert len(created_sessions) == 1
    seed_calls = created_sessions[0].seed_calls
    assert seed_calls[0]["frame_idx"] == 0
    assert seed_calls[1]["frame_idx"] == 4
    assert seed_calls[0]["box_labels"] == [1]
    assert seed_calls[1]["box_labels"] == [11]
    assert seed_calls[1]["boundary_bundle"] is not None
    assert seed_calls[1]["boundary_allowed_gids"] == {11}
    # xyxy -> xywh conversion must happen before add_prompt_boxes_abs.
    assert seed_calls[1]["boxes_abs"] == [[8.0, 10.0, 20.0, 20.0]]
