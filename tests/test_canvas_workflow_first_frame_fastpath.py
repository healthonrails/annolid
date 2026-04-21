from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets

import annolid.gui.mixins.canvas_workflow_mixin as canvas_workflow_module
from annolid.gui.mixins.canvas_workflow_mixin import CanvasWorkflowMixin


class _CanvasStub:
    def __init__(self) -> None:
        self.shapes = []
        self.pixmap = None

    def setEnabled(self, _enabled: bool) -> None:
        return

    def setPatchSimilarityOverlay(self, _overlay) -> None:
        return

    def loadPixmap(self, pixmap) -> None:
        self.pixmap = pixmap


class _ZoomStub:
    def value(self) -> int:
        return 100


class _Host(CanvasWorkflowMixin):
    def __init__(self) -> None:
        self.canvas = _CanvasStub()
        self._viewer_stack = None
        self._config = {"keep_prev": False, "keep_prev_scale": False}
        self.imagePath = ""
        self.filename = ""
        self.image = None
        self.imageData = None
        self.video_loader = object()
        self.video_file = "/tmp/sample.mp4"
        self._fit_window_applied_video_key = None
        self.zoom_values = {}
        self.zoomMode = "fit_window"
        self.zoomWidget = _ZoomStub()
        self.scroll_values = {}
        self.large_image_view = None
        self.depth_manager = None
        self.optical_flow_manager = None
        self.frame_number = 0
        self.load_predict_calls: list[tuple[int, str]] = []
        self.behavior_overlay_calls: list[int] = []

    def resetState(self) -> None:
        return

    def _deactivate_pca_map(self) -> None:
        return

    def noShapes(self) -> bool:
        return True

    def setDirty(self) -> None:
        return

    def setClean(self) -> None:
        return

    def setFitWindow(self, _enabled: bool) -> None:
        return

    def adjustScale(self, initial: bool = False) -> None:
        _ = initial
        return

    def setZoom(self, _value: int) -> None:
        return

    def setScroll(self, _orientation, _value) -> None:
        return

    def loadPredictShapes(self, frame_number, filename) -> None:
        self.load_predict_calls.append((int(frame_number), str(filename)))

    def _refresh_behavior_overlay(self, frame_number: int | None = None) -> None:
        if frame_number is None:
            frame_number = int(self.frame_number)
        self.behavior_overlay_calls.append(int(frame_number))


class _OverlayManagerStub:
    def __init__(self, has_overlay: bool) -> None:
        self._has_overlay = bool(has_overlay)
        self.update_calls = 0

    def has_overlay_for_frame(self, _frame_number: int) -> bool:
        return self._has_overlay

    def update_overlay_for_frame(self, _frame_number: int, _frame_rgb) -> None:
        self.update_calls += 1


def _sample_qimage() -> QtGui.QImage:
    image = QtGui.QImage(8, 8, QtGui.QImage.Format_RGB32)
    image.fill(QtGui.qRgb(20, 30, 40))
    return image


def test_first_frame_enrichment_is_deferred(monkeypatch) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    host = _Host()
    host._defer_first_frame_enrichment = True

    scheduled = []

    def _capture_timer(delay_ms, callback):
        scheduled.append((int(delay_ms), callback))

    monkeypatch.setattr(QtCore.QTimer, "singleShot", _capture_timer)

    host.image_to_canvas(_sample_qimage(), Path("/tmp/frame_000000000.png"), 0)

    assert host.load_predict_calls == [(0, "/tmp/frame_000000000.png")]
    assert host.behavior_overlay_calls == []
    assert len(scheduled) == 1
    assert scheduled[0][0] == int(host._FIRST_FRAME_ENRICHMENT_DELAY_MS)

    scheduled[0][1]()
    assert host.load_predict_calls == [(0, "/tmp/frame_000000000.png")]
    assert host.behavior_overlay_calls == [0]


def test_enrichment_runs_inline_without_defer() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    host = _Host()
    host._defer_first_frame_enrichment = False

    host.image_to_canvas(_sample_qimage(), Path("/tmp/frame_000000001.png"), 1)

    assert host.load_predict_calls == [(1, "/tmp/frame_000000001.png")]
    assert host.behavior_overlay_calls == [1]


def test_visual_overlay_enrichment_skips_rgb_conversion_without_overlays(
    monkeypatch,
) -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    host = _Host()
    host.depth_manager = _OverlayManagerStub(has_overlay=False)
    host.optical_flow_manager = _OverlayManagerStub(has_overlay=False)

    conversion_calls = []

    def _convert_stub(_qimage):
        conversion_calls.append(True)
        raise AssertionError("RGB conversion should be skipped without overlays")

    monkeypatch.setattr(
        canvas_workflow_module, "convert_qt_image_to_rgb_cv_image", _convert_stub
    )

    host._run_visual_overlay_enrichment(frame_number=0, qimage=_sample_qimage())

    assert conversion_calls == []
    assert host.depth_manager.update_calls == 0
    assert host.optical_flow_manager.update_calls == 0
    assert host.behavior_overlay_calls == [0]
