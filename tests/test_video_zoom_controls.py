from __future__ import annotations

from qtpy import QtGui, QtWidgets

from annolid.gui.window_base import AnnolidWindowBase
from annolid.gui.mixins.core_interaction_mixin import CoreInteractionMixin
from annolid.gui.widgets.realtime_manager import RealtimeManager
from annolid.gui.widgets.threejs_viewer import ThreeJsViewerWidget


class _ImageStub:
    def isNull(self) -> bool:
        return False


class _ZoomStub:
    def value(self) -> int:
        return 250


class _CanvasStub:
    def __init__(self) -> None:
        self.scale = 1.0
        self.adjust_size_calls = 0
        self.update_geometry_calls = 0
        self.update_calls = 0
        self.loaded_pixmaps = []

    def updateGeometry(self) -> None:
        self.update_geometry_calls += 1

    def adjustSize(self) -> None:
        self.adjust_size_calls += 1

    def update(self) -> None:
        self.update_calls += 1

    def loadPixmap(self, pixmap, clear_shapes=True) -> None:
        self.loaded_pixmaps.append((pixmap, clear_shapes))

    def setRealtimeShapes(self, shapes) -> None:
        self.realtime_shapes = list(shapes or [])


class _PaintHost(CoreInteractionMixin):
    def __init__(self) -> None:
        self._active_image_view = "canvas"
        self.image = _ImageStub()
        self.canvas = _CanvasStub()
        self.zoomWidget = _ZoomStub()
        self._viewer_stack = None
        self.isPlaying = True
        self.video_loader = object()


class _StatusBarStub:
    def __init__(self) -> None:
        self.messages = []

    def showMessage(self, text: str, *args) -> None:
        self.messages.append(str(text))


class _RealtimeWindowStub:
    def __init__(self) -> None:
        self.image = QtGui.QImage()
        self._active_image_view = "canvas"
        self.canvas = _CanvasStub()
        self.paint_canvas_calls = 0
        self._status = _StatusBarStub()
        self.threejs_manager = None

    def paintCanvas(self) -> None:
        self.paint_canvas_calls += 1

    def statusBar(self):
        return self._status

    def tr(self, text: str) -> str:
        return text

    def _get_rgb_by_label(self, _label: str):
        return (0, 255, 0)


class _RealtimeControlStub:
    def __init__(self) -> None:
        self.messages = []

    def set_status_text(self, text: str) -> None:
        self.messages.append(str(text))


class _FakePage:
    def __init__(self) -> None:
        self.scripts = []

    def runJavaScript(self, script: str) -> None:
        self.scripts.append(script)


class _FakeWebView:
    def __init__(self) -> None:
        self.page_obj = _FakePage()

    def page(self):
        return self.page_obj


class _FakeStack:
    def __init__(self, current) -> None:
        self._current = current

    def currentWidget(self):
        return self._current


class _FakeThreeJsManager:
    def __init__(self, viewer) -> None:
        self._viewer = viewer

    def viewer_widget(self):
        return self._viewer


class _FakeThreeJsViewer:
    def __init__(self) -> None:
        self.zoom_factors = []
        self.reset_calls = 0

    def zoom_view(self, factor: float) -> None:
        self.zoom_factors.append(float(factor))

    def reset_view(self) -> None:
        self.reset_calls += 1


class _VisibleRealtimeThreeJsViewer:
    def __init__(self) -> None:
        self.updates = []

    def isVisible(self) -> bool:
        return True

    def update_realtime_data(self, qimage, detections) -> None:
        self.updates.append((qimage, list(detections or [])))


class _VisibleRealtimeThreeJsManager:
    def __init__(self, viewer) -> None:
        self.viewer = viewer

    def viewer_widget(self):
        return self.viewer


class _RealtimeThreeJsViewer:
    def __init__(self) -> None:
        self.init_calls = []

    def init_viewer(self, **kwargs) -> None:
        self.init_calls.append(dict(kwargs))


class _RealtimeThreeJsManager:
    def __init__(self) -> None:
        self.viewer = _RealtimeThreeJsViewer()
        self.ensure_calls = 0

    def ensure_threejs_viewer(self):
        self.ensure_calls += 1
        return self.viewer


class _RealtimeViewerWindow:
    def __init__(self) -> None:
        self.threejs_manager = None
        self.manager = _RealtimeThreeJsManager()
        self.active_views = []

    def ensure_threejs_manager(self):
        self.threejs_manager = self.manager
        return self.manager

    def _set_active_view(self, view: str) -> None:
        self.active_views.append(str(view))


def test_video_zoom_repaints_canvas_size_during_playback() -> None:
    host = _PaintHost()

    host.paintCanvas()

    assert host.canvas.scale == 2.5
    assert host.canvas.update_geometry_calls == 1
    assert host.canvas.adjust_size_calls == 1
    assert host.canvas.update_calls == 1


def test_realtime_frame_syncs_window_image_before_repaint() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    manager = RealtimeManager.__new__(RealtimeManager)
    manager.window = _RealtimeWindowStub()
    manager.realtime_running = True
    manager._classify_eye_blinks = False
    manager._bot_watch_labels = set()
    manager._bot_report_enabled = False
    manager.realtime_log_fp = None
    manager.realtime_control_widget = _RealtimeControlStub()
    manager._realtime_shapes = []

    image = QtGui.QImage(16, 10, QtGui.QImage.Format_RGB888)
    image.fill(QtGui.QColor("black"))

    RealtimeManager._on_realtime_frame(
        manager,
        image,
        {"frame_index": 3, "camera_id": "camera0"},
        [],
    )

    assert manager.window.image.isNull() is False
    assert manager.window.image.size() == image.size()
    assert manager.window.canvas.loaded_pixmaps
    assert manager.window.canvas.loaded_pixmaps[-1][1] is False
    assert manager.window.paint_canvas_calls == 1


def test_realtime_frame_does_not_repaint_canvas_when_threejs_is_visible() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    manager = RealtimeManager.__new__(RealtimeManager)
    window = _RealtimeWindowStub()
    viewer = _VisibleRealtimeThreeJsViewer()
    window.threejs_manager = _VisibleRealtimeThreeJsManager(viewer)
    manager.window = window
    manager.realtime_running = True
    manager._classify_eye_blinks = False
    manager._bot_watch_labels = set()
    manager._bot_report_enabled = False
    manager.realtime_log_fp = None
    manager.realtime_control_widget = _RealtimeControlStub()
    manager._realtime_shapes = []

    image = QtGui.QImage(16, 10, QtGui.QImage.Format_RGB888)
    image.fill(QtGui.QColor("black"))

    RealtimeManager._on_realtime_frame(
        manager,
        image,
        {"frame_index": 4, "camera_id": "camera0"},
        [],
    )

    assert window.image.isNull() is True
    assert window.canvas.loaded_pixmaps == []
    assert window.paint_canvas_calls == 0
    assert len(viewer.updates) == 1
    assert viewer.updates[0][0] is image


def test_threejs_viewer_accepts_toolbar_zoom_commands() -> None:
    host = type("Host", (), {})()
    host._web_view = _FakeWebView()

    ThreeJsViewerWidget.zoom_view(host, 1.15)
    ThreeJsViewerWidget.reset_view(host)

    scripts = host._web_view.page_obj.scripts
    assert "window.annolidZoomView" in scripts[0]
    assert "1.15" in scripts[0]
    assert "window.annolidResetView" in scripts[1]


def test_window_toolbar_zoom_routes_to_active_threejs_viewer() -> None:
    viewer = _FakeThreeJsViewer()
    host = type("Host", (), {})()
    host._viewer_stack = _FakeStack(viewer)
    host.threejs_manager = _FakeThreeJsManager(viewer)
    host._active_threejs_viewer = lambda: AnnolidWindowBase._active_threejs_viewer(host)

    assert AnnolidWindowBase._apply_threejs_toolbar_zoom(host, 10) is True
    assert AnnolidWindowBase._reset_threejs_toolbar_view(host) is True
    assert viewer.zoom_factors == [1.15]
    assert viewer.reset_calls == 1


def test_realtime_preferred_threejs_viewer_is_created_when_missing() -> None:
    manager = RealtimeManager.__new__(RealtimeManager)
    manager.window = _RealtimeViewerWindow()

    RealtimeManager._activate_realtime_viewer(
        manager,
        "threejs",
        {"enable_eye_control": True, "enable_hand_control": False},
    )

    assert manager.window.threejs_manager is manager.window.manager
    assert manager.window.manager.ensure_calls == 1
    assert manager.window.manager.viewer.init_calls == [
        {"enable_eye_control": True, "enable_hand_control": False}
    ]
    assert manager.window.active_views == ["threejs"]
