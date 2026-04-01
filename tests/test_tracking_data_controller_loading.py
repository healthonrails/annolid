from __future__ import annotations

from pathlib import Path

import pandas as pd

from annolid.gui.controllers.tracking_data import TrackingDataController


class _DummyBehaviorController:
    def clear(self) -> None:
        pass

    def load_events_from_store(self) -> None:
        pass

    def attach_slider(self, _slider) -> None:
        pass

    @property
    def events_count(self) -> int:
        return 0

    @property
    def behavior_names(self):
        return []

    def iter_events(self):
        return []


class _DummyBehaviorLogWidget:
    def clear(self) -> None:
        pass

    def set_events(self, _events, fps: float) -> None:
        _ = fps


class _DummyWindow:
    def __init__(self) -> None:
        self.behavior_controller = _DummyBehaviorController()
        self.behavior_log_widget = _DummyBehaviorLogWidget()
        self.pinned_flags = {}
        self._df = None
        self.video_results_folder = None
        self.seekbar = None
        self.fps = 30.0

    def _load_behavior(self, _path: str) -> None:
        return

    def _load_labels(self, _path: Path) -> None:
        return

    def loadFlags(self, _flags: dict) -> None:
        return


def test_behavior_csv_filter_skips_tracking_sidecars(tmp_path: Path):
    assert (
        TrackingDataController._is_likely_behavior_csv(
            tmp_path / "sample_tracking.csv", "sample"
        )
        is False
    )
    assert (
        TrackingDataController._is_likely_behavior_csv(
            tmp_path / "sample_tracked.csv", "sample"
        )
        is False
    )
    assert (
        TrackingDataController._is_likely_behavior_csv(
            tmp_path / "sample_gaps_report.csv", "sample"
        )
        is False
    )
    assert (
        TrackingDataController._is_likely_behavior_csv(
            tmp_path / "sample_timestamps.csv", "sample"
        )
        is True
    )


def test_tracking_csv_is_not_loaded_during_video_open(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    tracking_path = tmp_path / "sample_tracking.csv"
    pd.DataFrame([{"frame_number": 0, "x1": 1, "y1": 2, "x2": 3, "y2": 4}]).to_csv(
        tracking_path, index=False
    )

    read_called = {"value": False}

    def _fail_read_csv(*_args, **_kwargs):
        read_called["value"] = True
        raise AssertionError("read_csv should not run while opening a video.")

    monkeypatch.setattr(
        "annolid.gui.controllers.tracking_data.pd.read_csv", _fail_read_csv
    )

    window = _DummyWindow()
    controller = TrackingDataController(window)
    controller.load_tracking_results(tmp_path, str(video_path))

    assert controller._tracking_csv_path == tracking_path
    assert read_called["value"] is False
    assert window._df is None


def test_tracking_csv_loads_lazily_on_first_lookup(tmp_path: Path):
    video_path = tmp_path / "small.mp4"
    video_path.write_bytes(b"")

    tracking_path = tmp_path / "small_tracking.csv"
    pd.DataFrame([{"frame_number": 0, "x1": 1, "y1": 2, "x2": 3, "y2": 4}]).to_csv(
        tracking_path, index=False
    )

    window = _DummyWindow()
    controller = TrackingDataController(window)
    controller.load_tracking_results(tmp_path, str(video_path))

    assert controller.tracking_dataframe is None
    assert window._df is None

    rows = controller.tracking_rows_for_frame(0)
    assert len(rows) == 1
    assert controller.tracking_dataframe is not None
    assert window._df is not None
