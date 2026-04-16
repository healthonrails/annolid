import json
from annolid.gui.widgets.behavior_describe_widget import (
    BehaviorDescribeWidget,
    BehaviorSegmentDialog,
)
import os

import pytest

qtpy = pytest.importorskip("qtpy")
from qtpy import QtCore, QtWidgets  # noqa: E402


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_behavior_segment_dialog_timeline_window_count():
    _ensure_qapp()

    dialog = BehaviorSegmentDialog(
        parent=None,
        start_time=QtCore.QTime(0, 0, 0),
        end_time=QtCore.QTime(0, 0, 5),
        notes=None,
        prompt_text="test",
        video_fps=30.0,
        total_frames=300,
    )

    # Switch to timeline mode.
    dialog.run_mode_combo.setCurrentIndex(1)
    dialog.timeline_step_spin.setValue(1)

    assert dialog.run_mode() == "timeline"
    assert dialog._timeline_window_count() == 6


def test_behavior_segment_dialog_prompt_includes_behavior_context():
    _ensure_qapp()

    dialog = BehaviorSegmentDialog(
        parent=None,
        start_time=QtCore.QTime(0, 0, 0),
        end_time=QtCore.QTime(0, 0, 5),
        notes="trial segment",
        video_description="Two mice in an open field arena.",
        instance_count=2,
        experiment_context="Resident-intruder social interaction trial.",
        behavior_definitions="Aggression bout: any slap in the face, run away, or fight initiation.",
        focus_points="Count aggression bouts and identify initiator.",
        prompt_text="",
        video_fps=30.0,
        total_frames=300,
    )

    prompt = dialog._render_template()
    assert "Video context: Two mice in an open field arena." in prompt
    assert "Track 2 instance(s)" in prompt
    assert "Experiment context: Resident-intruder social interaction trial." in prompt
    assert "Behavior definitions to apply:" in prompt
    assert "Aggression bout: any slap in the face" in prompt
    assert (
        "Focus specifically on: Count aggression bouts and identify initiator."
        in prompt
    )
    assert "initiator/responder" in prompt


def test_behavior_segment_dialog_instance_count_auto_means_none():
    _ensure_qapp()

    dialog = BehaviorSegmentDialog(
        parent=None,
        start_time=QtCore.QTime(0, 0, 0),
        end_time=QtCore.QTime(0, 0, 1),
        notes=None,
        prompt_text="",
        video_fps=30.0,
        total_frames=300,
    )

    dialog.instance_count_spin.setValue(0)
    assert dialog.instance_count() is None


class _DummyCaption(QtWidgets.QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.text_edit = QtWidgets.QPlainTextEdit(self)
        self._allow_empty_caption = False

    def create_button(
        self,
        icon_name: str = "",
        color: str = "",
        hover_color: str = "",
    ) -> QtWidgets.QPushButton:
        del icon_name, color, hover_color
        return QtWidgets.QPushButton(self)

    def set_caption(self, text: str) -> None:
        self.text_edit.setPlainText(str(text or ""))


class _DummyTimelinePanel:
    def __init__(self) -> None:
        self.current_frame = None

    def set_current_frame(self, frame: int) -> None:
        self.current_frame = int(frame)


class _DummyBehaviorController:
    def __init__(self) -> None:
        self.marks = []
        self.intervals = []

    def add_generic_mark(
        self, frame: int, mark_type: str = "", color: str = ""
    ) -> None:
        self.marks.append((int(frame), str(mark_type), str(color)))

    def create_interval(
        self,
        *,
        behavior: str,
        start_frame: int,
        end_frame: int,
        subject: str = "",
        timestamp_provider=None,
    ) -> None:
        ts_start = (
            timestamp_provider(start_frame) if callable(timestamp_provider) else None
        )
        ts_end = timestamp_provider(end_frame) if callable(timestamp_provider) else None
        self.intervals.append(
            (
                str(behavior),
                int(start_frame),
                int(end_frame),
                str(subject),
                ts_start,
                ts_end,
            )
        )


class _DummySeekbar:
    def __init__(self) -> None:
        self._val_max = 1000
        self.value = None

    def setValue(self, value: int) -> None:
        self.value = int(value)


class _DummyHost:
    def __init__(self) -> None:
        self.timeline_panel = _DummyTimelinePanel()
        self.behavior_controller = _DummyBehaviorController()
        self.seekbar = _DummySeekbar()

    def _refresh_behavior_log(self) -> None:
        return None


def test_behavior_describe_merge_writes_segment_log_json(tmp_path):
    _ensure_qapp()
    caption = _DummyCaption()
    widget = BehaviorDescribeWidget(caption)

    video_path = tmp_path / "mouse.mp4"
    video_path.write_bytes(b"")
    widget.set_video_context(str(video_path), fps=30.0, num_frames=90)

    payload = {
        "video_path": str(video_path),
        "step_seconds": 1,
        "prompt": "Describe behavior every second.",
        "end_frame": 89,
        "total_points": 3,
        "skipped_points": 1,
        "final": True,
        "points": [
            {"frame": 0, "timestamp": "00:00:00", "description": "walking"},
            {"frame": 30, "timestamp": "00:00:01", "description": "grooming"},
            {"frame": 60, "timestamp": "00:00:02", "description": "rearing"},
        ],
    }
    widget.merge_timeline_result(json.dumps(payload))

    csv_path = tmp_path / "mouse_timestamps.csv"
    log_path = tmp_path / "mouse_behavior_segment_labels.json"
    assert csv_path.exists()
    assert log_path.exists()

    log_payload = json.loads(log_path.read_text(encoding="utf-8"))
    assert log_payload["mode"] == "timeline_describe"
    assert log_payload["segment_seconds"] == 1.0
    assert log_payload["evaluated_segments"] == 3
    assert log_payload["skipped_segments"] == 1
    assert log_payload["labeled_segments"] == 3
    assert len(log_payload["predictions"]) == 3
    assert log_payload["predictions"][0]["label"] == "walking"
    assert log_payload["predictions"][0]["start_frame"] == 0
    assert log_payload["predictions"][0]["end_frame"] == 29


def test_behavior_describe_progress_updates_gui_elements(tmp_path):
    _ensure_qapp()
    caption = _DummyCaption()
    host = _DummyHost()
    caption.host_window_widget = host
    widget = BehaviorDescribeWidget(caption)

    video_path = tmp_path / "mouse.mp4"
    video_path.write_bytes(b"")
    widget.set_video_context(str(video_path), fps=30.0, num_frames=90)

    progress_payload = {
        "video_path": str(video_path),
        "total_points": 5,
        "processed_points": 2,
        "next_frame": 31,
        "latest_point": {
            "frame": 30,
            "timestamp": "00:00:01",
            "description": "grooming",
        },
    }
    widget.update_timeline_gui_progress(json.dumps(progress_payload))

    assert widget._current_frame == 30
    assert host.timeline_panel.current_frame == 30
    assert host.behavior_controller.marks[-1][:2] == (30, "prediction_progress")
    assert "00:00:01: grooming" in caption.text_edit.toPlainText()
    assert host.seekbar.value == 31


def test_behavior_describe_merge_syncs_intervals_to_behavior_controller(tmp_path):
    _ensure_qapp()
    caption = _DummyCaption()
    host = _DummyHost()
    caption.host_window_widget = host
    widget = BehaviorDescribeWidget(caption)

    video_path = tmp_path / "mouse.mp4"
    video_path.write_bytes(b"")
    widget.set_video_context(str(video_path), fps=30.0, num_frames=90)

    payload = {
        "video_path": str(video_path),
        "step_seconds": 1,
        "prompt": "Describe behavior every second.",
        "end_frame": 89,
        "total_points": 2,
        "processed_points": 1,
        "skipped_points": 0,
        "next_frame": 30,
        "points": [
            {"frame": 0, "timestamp": "00:00:00", "description": "walking"},
        ],
    }
    widget.merge_timeline_result(json.dumps(payload))

    assert host.behavior_controller.intervals
    behavior, start_frame, end_frame, subject, ts_start, ts_end = (
        host.behavior_controller.intervals[-1]
    )
    assert behavior == "walking"
    assert start_frame == 0
    assert end_frame == 29
    assert subject == "Agent"
    assert ts_start == 0.0
    assert ts_end == pytest.approx(29 / 30.0)
