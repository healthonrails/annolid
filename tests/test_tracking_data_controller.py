from __future__ import annotations

from pathlib import Path

import pandas as pd

from annolid.gui.controllers.tracking_data import TrackingDataController


class _DummyBehaviorController:
    def __init__(self) -> None:
        self.events_count = 0
        self.behavior_names: list[str] = []

    def clear(self) -> None:
        return None

    def load_events_from_store(self) -> None:
        return None


class _DummyBehaviorLogWidget:
    def clear(self) -> None:
        return None

    def set_events(self, *args, **kwargs) -> None:
        return None


class _DummyWindow:
    def __init__(self) -> None:
        self.behavior_controller = _DummyBehaviorController()
        self.behavior_log_widget = _DummyBehaviorLogWidget()
        self.pinned_flags = {}
        self._df = None
        self.video_results_folder = None
        self.seekbar = None
        self.fps = 0

    def _load_behavior(self, *_args, **_kwargs) -> None:
        return None


def test_tracking_rows_for_frame_uses_precomputed_frame_index(tmp_path: Path) -> None:
    window = _DummyWindow()
    controller = TrackingDataController(window)

    df = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "a", "score": 0.9},
            {"frame_number": 0, "instance_name": "b", "score": 0.7},
            {"frame_number": 2, "instance_name": "c", "score": 0.8},
        ]
    )

    controller._tracking_df = df
    controller._tracking_frame_indices = controller._build_tracking_frame_index(df)

    assert controller.tracking_rows_for_frame(0) == [
        {"frame_number": 0, "instance_name": "a", "score": 0.9},
        {"frame_number": 0, "instance_name": "b", "score": 0.7},
    ]
    assert controller.tracking_rows_for_frame(2) == [
        {"frame_number": 2, "instance_name": "c", "score": 0.8}
    ]
    assert controller.tracking_rows_for_frame(1) == []


def test_load_tracking_results_builds_frame_index(tmp_path: Path) -> None:
    window = _DummyWindow()
    controller = TrackingDataController(window)

    video_name = "sample"
    csv_path = tmp_path / f"{video_name}_tracking.csv"
    pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "a", "score": 0.9},
            {"frame_number": 1, "instance_name": "b", "score": 0.7},
            {"frame_number": 1, "instance_name": "c", "score": 0.6},
        ]
    ).to_csv(csv_path, index=False)

    controller.load_tracking_results(tmp_path, f"{video_name}.mp4")

    assert controller.tracking_rows_for_frame(0) == [
        {"frame_number": 0, "instance_name": "a", "score": 0.9}
    ]
    assert controller.tracking_rows_for_frame(1) == [
        {"frame_number": 1, "instance_name": "b", "score": 0.7},
        {"frame_number": 1, "instance_name": "c", "score": 0.6},
    ]
    assert window._df is not None


def test_tracking_controller_shutdown_clears_cached_state() -> None:
    window = _DummyWindow()
    controller = TrackingDataController(window)
    controller._tracking_df = pd.DataFrame([{"frame_number": 1}])
    controller._tracking_frame_slices = {1: (0, 1)}
    controller._tracking_frame_indices = {1: (0,)}
    controller._tracking_csv_path = Path("/tmp/example_tracking.csv")
    controller._sidecar_request_token = "token-1"

    controller.shutdown()

    assert controller.tracking_dataframe is None
    assert controller._tracking_frame_slices is None
    assert controller._tracking_frame_indices is None
    assert controller._tracking_csv_path is None
    assert controller._sidecar_request_token == ""


def test_sidecar_finished_ignores_stale_token_without_mutating_active_context() -> None:
    window = _DummyWindow()
    controller = TrackingDataController(window)
    controller._sidecar_request_token = "active-token"
    controller._sidecar_video_name = "sample_video"
    controller._sidecar_behavior_candidates = [Path("/tmp/sample_timestamps.csv")]

    calls: list[dict] = []

    def _capture_apply(*, payload, video_name, behavior_candidates) -> None:
        calls.append(
            {
                "payload": payload,
                "video_name": video_name,
                "behavior_candidates": list(behavior_candidates),
            }
        )

    controller._apply_sidecar_payload = _capture_apply  # type: ignore[method-assign]

    controller._handle_sidecar_worker_finished(
        payload={"loaded_behavior": True},
        request_token="stale-token",
    )

    assert calls == []
    assert controller._sidecar_request_token == "active-token"
    assert controller._sidecar_video_name == "sample_video"
    assert controller._sidecar_behavior_candidates == [
        Path("/tmp/sample_timestamps.csv")
    ]
