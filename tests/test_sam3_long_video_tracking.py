from __future__ import annotations

from collections import deque
from contextlib import contextmanager

import cv2
import numpy as np
import pytest
import torch
import importlib
from PIL import Image
import pycocotools.mask as mask_utils

from annolid.segmentation.SAM.sam3 import agent_video_orchestrator
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

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([42], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [[30.0, 0.0, 10.0, 10.0]],
                dtype=np.float32,
            ),
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

    outputs = session._map_outputs_to_global_ids_at_frame(
        {
            "out_obj_ids": np.asarray([7], dtype=np.int64),
            "out_boxes_xywh": np.asarray(
                [[52.0, 30.0, 12.0, 12.0]],
                dtype=np.float32,
            ),
        },
        frame_idx=14,
    )

    assert outputs["out_obj_ids"].tolist() == [1]
    assert session._global_track_next_id == 2


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


def test_window_seed_segments_are_ordered_and_text_anchored() -> None:
    assert Sam3SessionManager._build_window_seed_segments(
        [5, 2, 5], 8, has_text_prompt=False
    ) == [(2, 5), (5, 8)]
    assert Sam3SessionManager._build_window_seed_segments(
        [], 8, has_text_prompt=True
    ) == [(0, 8)]
    assert Sam3SessionManager._build_window_seed_segments(
        [3, 6], 8, has_text_prompt=True
    ) == [(0, 3), (3, 6), (6, 8)]


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


def test_annotated_window_falls_back_to_text_when_no_local_annotations(
    monkeypatch, tmp_path
) -> None:
    frames = [np.full((4, 4, 3), idx, dtype=np.uint8) for idx in range(4)]
    prompt_calls: list[tuple[int, bool]] = []
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
                "frame_index": 0,
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

    windows = [(0, 4, frames)]

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
                "ann_frame_idx": 99,
                "box": [1, 1, 2, 2],
                "labels": [1],
                "obj_id": 1,
            }
        ],
        target_device="cpu",
        propagation_direction="forward",
        max_frame_num_to_track=4,
    )

    assert total_frames == 1
    assert total_masks == 1
    assert prompt_calls == [(0, False)]
    assert len(created_predictors) == 1
