from types import SimpleNamespace

from qtpy import QtWidgets

from annolid.gui.mixins.prediction_execution_mixin import PredictionExecutionMixin


def test_cutie_start_prefers_existing_predictions() -> None:
    start = PredictionExecutionMixin._resolve_cutie_start_frame_from_seed_state(
        current_start_frame=1001,
        max_existing_frame=1200,
        manual_seed_max=1000,
    )
    assert start == 1201


def test_cutie_start_prefers_latest_manual_seed_when_no_existing_outputs() -> None:
    start = PredictionExecutionMixin._resolve_cutie_start_frame_from_seed_state(
        current_start_frame=1001,
        max_existing_frame=-1,
        manual_seed_max=1000,
    )
    assert start == 1001


def test_cutie_start_keeps_frame_zero_bootstrap_for_single_seed() -> None:
    start = PredictionExecutionMixin._resolve_cutie_start_frame_from_seed_state(
        current_start_frame=1,
        max_existing_frame=-1,
        manual_seed_max=0,
    )
    assert start == 0


def test_cutie_brightness_contrast_kwargs_reads_video_settings() -> None:
    class _Host(PredictionExecutionMixin):
        video_file = "/tmp/demo.mp4"

        @staticmethod
        def get_video_brightness_contrast_values(_video_path):
            return (11, -9)

    host = _Host()
    assert host._cutie_brightness_contrast_kwargs() == {
        "brightness": 11,
        "contrast": -9,
    }


def test_cutie_brightness_contrast_kwargs_clamps_out_of_range_values() -> None:
    class _Host(PredictionExecutionMixin):
        video_file = "/tmp/demo.mp4"

        @staticmethod
        def get_video_brightness_contrast_values(_video_path):
            return (999, -999)

    host = _Host()
    assert host._cutie_brightness_contrast_kwargs() == {
        "brightness": 100,
        "contrast": -100,
    }


def test_lost_tracking_instance_does_not_restart_when_automatic_pause_enabled(
    monkeypatch,
) -> None:
    class _Button:
        def __init__(self) -> None:
            self.text = None
            self.style = None
            self.enabled = None

        def setText(self, text: str) -> None:
            self.text = text

        def setStyleSheet(self, style: str) -> None:
            self.style = style

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    button = _Button()
    active_view_calls = []
    image_calls = []

    class _Seekbar:
        def __init__(self) -> None:
            self.value = None
            self.signals_blocked = False

        def blockSignals(self, enabled: bool) -> None:
            self.signals_blocked = bool(enabled)

        def setValue(self, value: int) -> None:
            self.value = value

    class _VideoLoader:
        def load_frame(self, frame_number: int):
            import numpy as np

            return np.zeros((4, 4, 3), dtype=np.uint8) + int(frame_number % 255)

    class _Host(PredictionExecutionMixin):
        pass

    host = _Host()
    host.automatic_pause_enabled = True
    host.stepSizeWidget = SimpleNamespace(predict_button=button)
    host.seekbar = _Seekbar()
    host.video_loader = _VideoLoader()
    host.frame_number = 12
    host._pending_prediction_restart_frame = 99
    host._prediction_stop_requested = False
    host._skip_tracking_csv_overwrite_for_keypoint_round = False
    host._is_cutie_tracking_model = lambda: True
    host._set_active_view = lambda mode: active_view_calls.append(mode)
    host._frame_image_path = lambda frame: f"/tmp/frame_{frame:09d}.png"
    host.timeline_panel = SimpleNamespace(
        set_current_frame=lambda frame: image_calls.append(("timeline", frame))
    )
    host._update_audio_playhead = lambda frame: image_calls.append(("audio", frame))
    host.image_to_canvas = lambda qimage, frame_path, frame: image_calls.append(
        ("canvas", frame_path, frame)
    )
    host._finalize_prediction_progress = lambda *args, **kwargs: None

    messages = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: messages.append((args, kwargs)),
    )

    PredictionExecutionMixin.lost_tracking_instance(
        host,
        "There is 1 missing instance in the current frame (12).\n\nMissing or occluded: mouse#12",
    )

    assert getattr(host, "_pending_prediction_restart_frame", None) is None
    assert getattr(host, "_pending_prediction_pause_frame", None) == 12
    assert active_view_calls == []
    assert host.seekbar.value is None
    assert host.seekbar.signals_blocked is False
    assert image_calls == []
    assert button.text == "Pred"
    assert button.enabled is True
    assert messages

    PredictionExecutionMixin.predict_is_ready(host, "Stopped after pause")

    assert getattr(host, "_pending_prediction_pause_frame", None) is None
    assert active_view_calls == ["canvas"]
    assert host.seekbar.value == 12
    assert host.seekbar.signals_blocked is False
    assert ("timeline", 12) in image_calls
    assert ("audio", 12) in image_calls
    assert any(item[0] == "canvas" for item in image_calls)


def test_lost_tracking_instance_leaves_occlusion_flow_alone_when_disabled(
    monkeypatch,
) -> None:
    class _Button:
        def __init__(self) -> None:
            self.text = None
            self.style = None
            self.enabled = None

        def setText(self, text: str) -> None:
            self.text = text

        def setStyleSheet(self, style: str) -> None:
            self.style = style

        def setEnabled(self, enabled: bool) -> None:
            self.enabled = enabled

    button = _Button()
    host = SimpleNamespace(
        automatic_pause_enabled=False,
        stepSizeWidget=SimpleNamespace(predict_button=button),
        frame_number=12,
        _pending_prediction_restart_frame=99,
        _is_cutie_tracking_model=lambda: True,
    )

    messages = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: messages.append((args, kwargs)),
    )

    PredictionExecutionMixin.lost_tracking_instance(
        host,
        "There is 1 missing instance in the current frame (12).\n\nMissing or occluded: mouse#12",
    )

    assert getattr(host, "_pending_prediction_restart_frame", None) == 99
    assert button.text == "Pred"
    assert button.enabled is True
    assert messages == []


def test_lost_tracking_instance_records_missing_frame_mark() -> None:
    class _Button:
        def setText(self, _text: str) -> None:
            return

        def setStyleSheet(self, _style: str) -> None:
            return

        def setEnabled(self, _enabled: bool) -> None:
            return

    marked_frames = []
    host = SimpleNamespace(
        automatic_pause_enabled=False,
        stepSizeWidget=SimpleNamespace(predict_button=_Button()),
        frame_number=12,
        _is_cutie_tracking_model=lambda: True,
        _mark_missing_instance_frame=lambda frame: marked_frames.append(int(frame)),
    )

    PredictionExecutionMixin.lost_tracking_instance(
        host,
        "There is 1 missing instance in the current frame (12).\n\nMissing or occluded: mouse#12",
    )

    assert marked_frames == [12]
