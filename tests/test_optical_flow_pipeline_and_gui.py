from __future__ import annotations

import json
import os
from pathlib import Path
import threading

import cv2
import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.mixins.canvas_workflow_mixin import CanvasWorkflowMixin
from annolid.gui.widgets.optical_flow_manager import OpticalFlowManager
from annolid.gui.widgets.optical_flow_tool import OpticalFlowTool
from annolid.motion import flow_runner, flow_viz
from annolid.motion.optical_flow import (
    OpticalFlowPreferences,
    optical_flow_settings_from,
)
from annolid.utils.draw import draw_flow

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _write_synthetic_video(
    path: Path, *, num_frames: int = 5, width: int = 64, height: int = 48
) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (width, height),
    )
    assert writer.isOpened()
    try:
        for idx in range(num_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.circle(frame, (16 + idx * 4, 24), 8, (240, 180, 60), -1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_flow_runner_cli_main_and_atomic_overwrite(tmp_path: Path) -> None:
    video_path = _write_synthetic_video(tmp_path / "clip.mp4", num_frames=5)
    csv_path = tmp_path / "stats.csv"
    ndjson_path = tmp_path / "flow.ndjson"

    # First CLI run (regression: previously crashed with AttributeError on args.torch_farneback)
    flow_runner.main(
        [
            "--video",
            str(video_path),
            "--backend",
            "farneback",
            "--csv",
            str(csv_path),
            "--ndjson",
            str(ndjson_path),
            "--viz",
            "hsv",
        ]
    )

    first_lines = ndjson_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(first_lines) == 4
    assert csv_path.exists()

    # Re-running to the same NDJSON path must overwrite atomically, never append duplicates.
    flow_runner.main(
        [
            "--video",
            str(video_path),
            "--backend",
            "farneback",
            "--ndjson",
            str(ndjson_path),
        ]
    )
    second_lines = ndjson_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(second_lines) == 4


def test_flow_runner_large_flow_no_float16_overflow_and_decode_roundtrip() -> None:
    rng = np.random.default_rng(7)
    flow = rng.uniform(-10.0, 10.0, size=(32, 40, 2)).astype(np.float32)
    # Magnitudes > 256 previously overflowed float16 dx**2 + dy**2 to inf.
    flow[5, 6] = np.array([400.0, -300.0], dtype=np.float32)

    rec_q = flow_runner._build_flow_record("test.mp4", 1, flow, 32, 40, quantize=True)
    # Valid strict JSON (no NaN/Infinity tokens).
    dumped = json.dumps(rec_q, allow_nan=False)
    loaded = json.loads(dumped)
    decoded_q = flow_runner.decode_flow_record(loaded)
    assert decoded_q is not None
    assert decoded_q.shape == (32, 40, 2)
    assert np.isfinite(decoded_q).all()
    np.testing.assert_allclose(decoded_q[5, 6], [400.0, -300.0], atol=0.05)

    rec_f16 = flow_runner._build_flow_record(
        "test.mp4", 2, flow, 32, 40, quantize=False
    )
    # Regression: float16 component with missing scale must NOT be clipped to all zeros.
    rec_f16["otherData"]["flow_dx_raw"].pop("scale", None)
    decoded_f16 = flow_runner.decode_flow_record(rec_f16)
    assert decoded_f16 is not None
    np.testing.assert_allclose(decoded_f16[5, 6], [400.0, -300.0], atol=0.5)


def test_flow_runner_cooperative_stop_event_preserves_existing_ndjson(
    tmp_path: Path,
) -> None:
    video_path = _write_synthetic_video(tmp_path / "clip.mp4", num_frames=6)
    ndjson_path = tmp_path / "existing.ndjson"
    ndjson_path.write_text('{"frame_index": 1}\n', encoding="utf-8")

    stop_event = threading.Event()
    seen_frames: list[int] = []

    def _preview(payload: dict) -> None:
        seen_frames.append(int(payload["frame_index"]))
        stop_event.set()

    flow_runner.process_video_flow(
        str(video_path),
        save_ndjson=str(ndjson_path),
        preview_callback=_preview,
        stop_event=stop_event,
    )

    assert seen_frames == [1]
    # Existing completed NDJSON is preserved intact and no temp file leaks behind.
    assert ndjson_path.read_text(encoding="utf-8") == '{"frame_index": 1}\n'
    assert not list(tmp_path.glob("*.tmp"))


def test_draw_flow_handles_nonfinite_zero_step_and_grayscale() -> None:
    img_bgr = np.zeros((48, 64, 3), dtype=np.uint8)
    flow = np.zeros((48, 64, 2), dtype=np.float32)
    flow[8, 8] = [np.nan, np.inf]
    flow[24, 24] = [3.0, -2.0]

    vis = draw_flow(img_bgr, flow, step=16)
    assert vis.shape == img_bgr.shape
    assert vis.dtype == np.uint8
    assert int(vis.sum()) > 0

    # Zero/negative step must not crash.
    vis_step0 = draw_flow(img_bgr, flow, step=0)
    assert vis_step0.shape == img_bgr.shape

    # Grayscale canvas support.
    img_gray = np.zeros((48, 64), dtype=np.uint8)
    vis_gray = draw_flow(img_gray, flow, step=16)
    assert vis_gray.shape == img_gray.shape
    assert int(vis_gray.sum()) > 0


def test_flow_viz_clip_flow_preserves_negative_motion_and_sanitizes_nan() -> None:
    flow_neg = np.full((16, 16, 2), -5.0, dtype=np.float32)
    flow_pos = np.full((16, 16, 2), 5.0, dtype=np.float32)

    img_neg = flow_viz.flow_to_image(flow_neg, clip_flow=2.0)
    img_pos = flow_viz.flow_to_image(flow_pos, clip_flow=2.0)
    # Opposite motion directions must produce distinct non-white colors.
    assert not np.array_equal(img_neg, img_pos)
    assert not np.all(img_neg == 255)

    flow_bad = np.array([[[np.nan, np.inf]]], dtype=np.float32)
    img_bad = flow_viz.flow_to_image(flow_bad)
    assert img_bad.shape == (1, 1, 3)


def test_optical_flow_manager_qsettings_bool_false_and_tool_sync(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    ini_path = str(tmp_path / "settings.ini")

    class _CanvasStub:
        def __init__(self) -> None:
            self.overlays: list[object] = []
            self._flow_preview_pixmap = None

        def setFlowPreviewOverlay(self, overlay) -> None:
            self.overlays.append(overlay)
            self._flow_preview_pixmap = overlay

    class _WindowStub(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.settings = QtCore.QSettings(ini_path, QtCore.QSettings.IniFormat)
            self.canvas = _CanvasStub()
            self.frame_number = 1

    # Persist string "false" into QSettings INI (matches Qt INI serialization).
    seed_settings = QtCore.QSettings(ini_path, QtCore.QSettings.IniFormat)
    seed_settings.setValue("optical_flow/stable_hsv", "false")
    seed_settings.setValue("optical_flow/backend", "farneback_torch")
    seed_settings.sync()

    window = _WindowStub()
    manager = OpticalFlowManager(window)
    window.optical_flow_manager = manager

    assert manager.flow_stable_hsv is False
    assert window.flow_stable_hsv is False
    assert manager.optical_flow_backend == "farneback_torch"

    # Updating preferences via OpticalFlowTool must sync back to OpticalFlowManager
    # so prediction/tracking via optical_flow_settings_from(manager) sees the update.
    updated = OpticalFlowPreferences(
        backend="raft",
        raft_model="large",
        visualization="quiver",
        opacity=85,
        farneback_winsize=21,
        farneback_poly_n=7,
    )
    manager.optical_flow_tool._apply_preferences_to_window(updated)
    manager.optical_flow_tool._persist_preferences(updated)

    settings_snapshot = optical_flow_settings_from(manager)
    assert settings_snapshot["optical_flow_backend"] == "raft"
    assert settings_snapshot["optical_flow_raft_model"] == "large"
    assert settings_snapshot["flow_farneback_winsize"] == 21
    assert settings_snapshot["flow_farneback_poly_n"] == 7


def test_optical_flow_tool_indexed_playback_tolerates_corrupt_line(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    class _CanvasStub:
        def __init__(self) -> None:
            self.last_overlay = None

        def setFlowPreviewOverlay(self, overlay) -> None:
            self.last_overlay = overlay

    class _WindowStub(QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.canvas = _CanvasStub()
            self.flow_visualization = "hsv"
            self.flow_opacity = 60
            self.flow_stable_hsv = True
            self.flow_quiver_step = 8
            self.flow_quiver_gain = 1.5
            self.frame_number = 1

    ndjson_path = tmp_path / "sample_flow.ndjson"
    flow1 = np.full((24, 32, 2), 1.5, dtype=np.float32)
    flow2 = np.full((24, 32, 2), -2.0, dtype=np.float32)
    rec1 = flow_runner._build_flow_record("sample.mp4", 1, flow1, 24, 32)
    rec2 = flow_runner._build_flow_record("sample.mp4", 2, flow2, 24, 32)
    ndjson_path.write_text(
        json.dumps(rec1) + "\n" + "{corrupt truncated json\n" + json.dumps(rec2) + "\n",
        encoding="utf-8",
    )

    window = _WindowStub()
    tool = OpticalFlowTool(window)
    tool._load_records_from_path(ndjson_path)

    # Indexed without holding full JSON payloads in self._records.
    assert tool._records == {}
    assert tool.has_overlay_for_frame(1) is True
    assert tool.has_overlay_for_frame(2) is True
    assert tool.has_overlay_for_frame(99) is False

    tool.update_overlay_for_frame(1)
    assert window.canvas.last_overlay is not None
    assert window.canvas.last_overlay.shape == (24, 32, 4)

    # Switching visualization style uses cached decoded flow cleanly.
    window.flow_visualization = "quiver"
    tool.refresh_current_frame_overlay()
    assert window.canvas.last_overlay is not None
    assert window.canvas.last_overlay.shape == (24, 32, 4)

    # Frame without a record clears the overlay.
    tool.update_overlay_for_frame(0)
    assert window.canvas.last_overlay is None


def test_canvas_workflow_clears_stale_flow_overlay_on_frame_without_record() -> None:
    _ensure_qapp()

    class _CanvasStub:
        def __init__(self) -> None:
            self._flow_preview_pixmap = object()
            self.cleared = False

        def setFlowPreviewOverlay(self, overlay) -> None:
            self._flow_preview_pixmap = overlay
            if overlay is None:
                self.cleared = True

    class _ManagerStub:
        def has_overlay_for_frame(self, _frame_number: int) -> bool:
            return False

        def is_live_running(self) -> bool:
            return False

        def update_overlay_for_frame(self, _frame_number: int, _frame_rgb) -> None:
            raise AssertionError("Should not invoke update when frame has no record")

    class _Host(CanvasWorkflowMixin):
        def __init__(self) -> None:
            self.canvas = _CanvasStub()
            self.large_image_view = None
            self.depth_manager = None
            self.optical_flow_manager = _ManagerStub()

        def _refresh_behavior_overlay(self, frame_number: int | None = None) -> None:
            _ = frame_number

    host = _Host()
    img = QtGui.QImage(8, 8, QtGui.QImage.Format_RGB32)
    img.fill(QtGui.qRgb(10, 20, 30))
    host._run_visual_overlay_enrichment(frame_number=0, qimage=img)

    assert host.canvas.cleared is True
    assert host.canvas._flow_preview_pixmap is None
