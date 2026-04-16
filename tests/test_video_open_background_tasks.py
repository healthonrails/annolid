from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets

from annolid.gui.mixins.video_workflow_mixin import VideoWorkflowMixin


class _ActionStub:
    def __init__(self) -> None:
        self.enabled_states: list[bool] = []

    def setEnabled(self, enabled: bool) -> None:
        self.enabled_states.append(bool(enabled))


class _VideoOpenHost(VideoWorkflowMixin):
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.video_results_folder = Path("/tmp/fake_video")
        self.frame_number = 0
        self.num_frames = 6
        self.filename = "/tmp/fake_video_000000000.png"
        self.video_file = "/tmp/fake_video.mp4"
        self.caption_widget = None
        self.open_segment_editor_action = _ActionStub()

    def _refresh_manual_seed_slider_marks(self, _folder) -> None:
        self.calls.append("seed_marks")

    def _refresh_missing_instance_slider_marks_from_tracking_stats(
        self, _folder
    ) -> None:
        self.calls.append("missing_marks")

    def _prefetch_label_for_frame(self, frame_idx: int, _fallback_path) -> None:
        self.calls.append(f"prefetch_{frame_idx}")

    def load_tracking_results(self, _cur_video_folder, _video_filename) -> None:
        self.calls.append("tracking")

    def _load_segments_for_active_video(self) -> None:
        self.calls.append("segments")

    def _load_data_in_caption_widget(self, _video_file) -> None:
        self.calls.append("caption_data")

    def _emit_live_frame_update(self) -> None:
        self.calls.append("frame_update")

    def tr(self, text: str) -> str:
        return str(text)


def test_video_open_background_tasks_run_in_order(monkeypatch) -> None:
    host = _VideoOpenHost()

    monkeypatch.setattr(QtCore.QTimer, "singleShot", lambda _msec, fn: fn())

    host._schedule_video_open_background_tasks(
        open_started_ts=0.0,
        programmatic_call=False,
        cur_video_folder=Path("/tmp"),
        video_filename=host.video_file,
    )

    assert host.calls == [
        "seed_marks",
        "missing_marks",
        "prefetch_1",
        "prefetch_2",
        "prefetch_3",
        "prefetch_4",
        "tracking",
        "segments",
        "caption_data",
        "frame_update",
    ]
    assert host.open_segment_editor_action.enabled_states[-1] is True


def test_video_open_background_tasks_skip_when_video_changes(monkeypatch) -> None:
    host = _VideoOpenHost()
    queued: list = []

    def _queue_callback(_msec, callback):
        queued.append(callback)

    monkeypatch.setattr(QtCore.QTimer, "singleShot", _queue_callback)

    host._schedule_video_open_background_tasks(
        open_started_ts=0.0,
        programmatic_call=True,
        cur_video_folder=Path("/tmp"),
        video_filename=host.video_file,
    )

    assert queued
    host.video_file = "/tmp/other_video.mp4"
    queued[0]()
    assert host.calls == []


def test_resolve_video_filename_prefers_video_dir(monkeypatch) -> None:
    host = _VideoOpenHost()
    host.video_file = "/data/session/eVLS47.mp4"
    host.filename = "/data/session/eVLS47/eVLS47_000000001.png"

    captured = {}

    def _fake_get_open_file_name(_parent, _title, start_dir, _filters):
        captured["start_dir"] = str(start_dir)
        return ("", "")

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        _fake_get_open_file_name,
    )

    result = host._resolve_video_filename(from_video_list=False, video_path=None)

    assert result == ""
    assert captured["start_dir"] == "/data/session"


def test_resolve_video_filename_uses_last_open_dir(monkeypatch) -> None:
    host = _VideoOpenHost()
    host.video_file = ""
    host.filename = "/tmp/frames/session_000000001.png"
    host._last_video_open_dir = "/mnt/videos"

    captured = {}

    def _fake_get_open_file_name(_parent, _title, start_dir, _filters):
        captured["start_dir"] = str(start_dir)
        return ("", "")

    monkeypatch.setattr(
        QtWidgets.QFileDialog,
        "getOpenFileName",
        _fake_get_open_file_name,
    )

    host._resolve_video_filename(from_video_list=False, video_path=None)

    assert captured["start_dir"] == "/mnt/videos"


def test_resolve_video_filename_remembers_dir_for_video_list() -> None:
    host = _VideoOpenHost()
    selected = "/Users/demo/videos/mouse.mp4"

    resolved = host._resolve_video_filename(from_video_list=True, video_path=selected)

    assert resolved == selected
    assert host._last_video_open_dir == "/Users/demo/videos"
