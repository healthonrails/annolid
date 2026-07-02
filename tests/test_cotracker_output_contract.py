from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("shapely")

from annolid.tracker.cotracker.track import CoTrackerProcessor


def test_normalize_cotracker_two_value_output_preserves_bool_visibility() -> None:
    tracks = torch.zeros((1, 2, 3, 2), dtype=torch.float32)
    visibility = torch.tensor([[[True, False, True], [False, True, True]]])

    norm_tracks, norm_visibility = CoTrackerProcessor._normalize_model_output(
        (tracks, visibility),
        online=False,
    )

    assert norm_tracks is tracks
    assert norm_visibility is visibility


def test_normalize_cotracker_four_value_offline_output_ignores_auxiliary_values() -> (
    None
):
    tracks = torch.zeros((1, 2, 3, 2), dtype=torch.float32)
    visibility = torch.tensor([[[0.95, 0.89, 0.10], [0.91, 0.90, 0.20]]])
    confidence = torch.ones_like(visibility)
    train_data = {"unused": True}

    norm_tracks, norm_visibility = CoTrackerProcessor._normalize_model_output(
        (tracks, visibility, confidence, train_data),
        online=False,
    )

    assert norm_tracks is tracks
    assert norm_visibility.dtype == torch.bool
    assert norm_visibility.tolist() == [[[True, False, False], [True, False, False]]]


def test_normalize_cotracker_online_output_combines_visibility_and_confidence() -> None:
    tracks = torch.zeros((1, 1, 3, 2), dtype=torch.float32)
    visibility = torch.tensor([[[0.8, 0.8, 0.8]]])
    confidence = torch.tensor([[[0.8, 0.7, 0.6]]])

    _, norm_visibility = CoTrackerProcessor._normalize_model_output(
        (tracks, visibility, confidence, {"unused": True}),
        online=True,
    )

    assert norm_visibility.tolist() == [[[True, False, False]]]


def test_bidirectional_tracking_accepts_extra_cotracker_outputs() -> None:
    processor = CoTrackerProcessor.__new__(CoTrackerProcessor)
    processor.device = torch.device("cpu")
    processor.start_frame = 10
    processor.end_frame = 11
    processor.mask = None
    processor.mask_label = None
    processor.queries = torch.tensor([[[0.0, 1.0, 2.0]]])
    processor._stop_triggered = False
    processor.pred_worker = None
    processor._should_stop_callback = lambda: False

    class FakeVideoLoader:
        def get_frames_between(self, start_frame: int, end_frame: int) -> np.ndarray:
            assert (start_frame, end_frame) == (10, 11)
            return np.zeros((2, 8, 8, 3), dtype=np.uint8)

    tracks = torch.zeros((1, 2, 1, 2), dtype=torch.float32)
    visibility = torch.tensor([[[0.95], [0.1]]])
    confidence = torch.ones_like(visibility)

    class FakeModel:
        def __call__(self, video: torch.Tensor, **kwargs):
            assert video.shape == (1, 2, 3, 8, 8)
            assert kwargs["backward_tracking"] is False
            return tracks, visibility, confidence, {"unused": True}

    processor.video_loader = FakeVideoLoader()
    processor._ensure_model = lambda: FakeModel()

    pred_tracks, pred_visibility, video = processor._process_video_bidirectional(
        start_frame=10,
        end_frame=11,
        grid_size=0,
        grid_query_frame=0,
    )

    assert pred_tracks is tracks
    assert pred_visibility.dtype == torch.bool
    assert pred_visibility.tolist() == [[[True], [False]]]
    assert video.shape == (1, 2, 3, 8, 8)


def test_bidirectional_tracking_retries_forward_only_on_upstream_arity_bug() -> None:
    processor = CoTrackerProcessor.__new__(CoTrackerProcessor)
    processor.device = torch.device("cpu")
    processor.start_frame = 10
    processor.end_frame = 11
    processor.mask = None
    processor.mask_label = None
    processor.queries = torch.tensor([[11.0, 1.0, 2.0]])
    processor._stop_triggered = False
    processor.pred_worker = None
    processor._should_stop_callback = lambda: False

    class FakeVideoLoader:
        def get_frames_between(self, start_frame: int, end_frame: int) -> np.ndarray:
            return np.zeros((2, 8, 8, 3), dtype=np.uint8)

    tracks = torch.zeros((1, 2, 1, 2), dtype=torch.float32)
    visibility = torch.ones((1, 2, 1), dtype=torch.float32)
    calls: list[bool] = []

    class FakeModel:
        def __call__(self, video: torch.Tensor, **kwargs):
            calls.append(bool(kwargs["backward_tracking"]))
            if kwargs["backward_tracking"]:
                raise ValueError("too many values to unpack (expected 3)")
            return tracks, visibility

    processor.video_loader = FakeVideoLoader()
    processor._ensure_model = lambda: FakeModel()

    pred_tracks, pred_visibility, _ = processor._process_video_bidirectional(
        start_frame=10,
        end_frame=11,
        grid_size=0,
        grid_query_frame=0,
    )

    assert calls == [True, False]
    assert pred_tracks is tracks
    assert pred_visibility.tolist() == [[[True], [True]]]
