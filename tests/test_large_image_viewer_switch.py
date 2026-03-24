from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.mixins.annotation_loading_mixin import AnnotationLoadingMixin
from annolid.gui.mixins.canvas_workflow_mixin import CanvasWorkflowMixin
from annolid.gui.mixins.file_browser_mixin import FileBrowserMixin
from annolid.gui.mixins.frame_playback_mixin import FramePlaybackMixin
from annolid.gui.mixins.label_panel_mixin import LabelPanelMixin
from annolid.gui.mixins.navigation_workflow_mixin import NavigationWorkflowMixin
from annolid.gui.mixins.core_interaction_mixin import CoreInteractionMixin
from annolid.gui.mixins.playback_draw_mixin import PlaybackDrawMixin
from annolid.gui.shape import Shape
from annolid.gui.widgets.tiled_image_view import TiledImageView, _ShapeGraphicsItem
from annolid.gui.window_base import (
    AnnolidLabelListItem,
    AnnolidLabelListWidget,
    AnnolidWindowBase,
)
import annolid.gui.window_base as window_base_module
from annolid.io.large_image.base import LargeImageBackendCapabilities


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def _send_mouse_drag(view, start: QtCore.QPoint, end: QtCore.QPoint) -> None:
    viewport = view.viewport()
    global_start = viewport.mapToGlobal(start)
    global_end = viewport.mapToGlobal(end)
    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(start),
        QtCore.QPointF(global_start),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    move = QtGui.QMouseEvent(
        QtCore.QEvent.MouseMove,
        QtCore.QPointF(end),
        QtCore.QPointF(global_end),
        QtCore.Qt.NoButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    release = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease,
        QtCore.QPointF(end),
        QtCore.QPointF(global_end),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    view.mousePressEvent(press)
    view.mouseMoveEvent(move)
    view.mouseReleaseEvent(release)


def _send_mouse_click(view, pos: QtCore.QPoint) -> None:
    viewport = view.viewport()
    global_pos = viewport.mapToGlobal(pos)
    press = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonPress,
        QtCore.QPointF(pos),
        QtCore.QPointF(global_pos),
        QtCore.Qt.LeftButton,
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoModifier,
    )
    release = QtGui.QMouseEvent(
        QtCore.QEvent.MouseButtonRelease,
        QtCore.QPointF(pos),
        QtCore.QPointF(global_pos),
        QtCore.Qt.LeftButton,
        QtCore.Qt.NoButton,
        QtCore.Qt.NoModifier,
    )
    view.mousePressEvent(press)
    view.mouseReleaseEvent(release)


class _CanvasStub(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.last_pixmap = None
        self.shapes = []
        self.last_load_clear_shapes = None
        self.editing_values = []
        self.context_menu_builds = 0
        self.current_behavior_text = ""

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.last_pixmap = pixmap
        self.last_load_clear_shapes = bool(clear_shapes)
        if clear_shapes:
            self.shapes = []

    def setEditing(self, value=True):
        self.editing_values.append(bool(value))

    def _build_context_menu(self, _main_window):
        self.context_menu_builds += 1
        return QtWidgets.QMenu(self)

    def setShapeVisible(self, shape, value):
        try:
            shape.visible = bool(value)
        except Exception:
            pass

    def setBehaviorText(self, value):
        self.current_behavior_text = value or ""


class _WindowStub(AnnolidWindowBase):
    def __init__(self):
        self._toggle_calls = []
        self._clean_calls = 0
        self._status_messages = []
        self._dirty_calls = 0
        super().__init__(config={})
        self.canvas = _CanvasStub()
        self.large_image_view = TiledImageView(self)
        self.large_image_view.set_host_window(self)
        self._viewer_stack = QtWidgets.QStackedWidget()
        self._viewer_stack.addWidget(self.canvas)
        self._viewer_stack.addWidget(self.large_image_view)
        self.fileListWidget = QtWidgets.QListWidget()

    def loadShapes(self, shapes, replace=True):
        self.canvas.shapes = list(shapes or [])

    def toggleActions(self, value):
        self._toggle_calls.append(bool(value))

    def setClean(self):
        self._clean_calls += 1
        self.dirty = False

    def setDirty(self):
        self.dirty = True
        self._dirty_calls += 1

    def status(self, message, delay=5000):
        self._status_messages.append(str(message))


class _WindowNavigationStub(
    NavigationWorkflowMixin,
    FramePlaybackMixin,
    PlaybackDrawMixin,
    _WindowStub,
):
    def __init__(self):
        self._active_view_requests = []
        super().__init__()
        self.video_loader = None
        self.isPlaying = False
        self.caption_widget = None
        self.behavior_controller = type(
            "_BehaviorController", (), {"highlighted_mark": None}
        )()
        self.playButton = QtWidgets.QPushButton("Play", self)
        self.seekbar = None
        self.timer = None
        self.fps = 8.0
        self.frame_number = 0
        self.num_frames = 0

    def sync_large_image_page_state(self) -> None:
        if self.large_image_backend is None:
            return
        self.frame_number = int(
            getattr(self.large_image_backend, "get_current_page", lambda: 0)() or 0
        )
        self.num_frames = int(
            getattr(self.large_image_backend, "get_page_count", lambda: 1)() or 1
        )

    def _set_active_view(self, mode: str = "canvas") -> None:
        self._active_view_requests.append(str(mode))

    def _update_frame_display_and_emit_update(self) -> None:
        return None


class _CanvasFitWindowStub(
    CanvasWorkflowMixin,
    CoreInteractionMixin,
    _WindowStub,
):
    def __init__(self):
        super().__init__()
        self._active_image_view = "tiled"
        self.large_image_backend = object()
        self._fit_calls = 0

        def _fit_to_window():
            self._fit_calls += 1

        self.large_image_view.fit_to_window = _fit_to_window


class _PageAnnotationWindow(
    AnnotationLoadingMixin, FileBrowserMixin, AnnolidWindowBase
):
    def __init__(self):
        self._toggle_calls = []
        self._clean_calls = 0
        self._status_messages = []
        super().__init__(config={"label_flags": {}, "store_data": False})
        self.canvas = _CanvasStub()
        self.large_image_view = TiledImageView(self)
        self.large_image_view.set_host_window(self)
        self._viewer_stack = QtWidgets.QStackedWidget()
        self._viewer_stack.addWidget(self.canvas)
        self._viewer_stack.addWidget(self.large_image_view)
        self.fileListWidget = QtWidgets.QListWidget()
        self.labelList = AnnolidLabelListWidget()
        self.uniqLabelList = QtWidgets.QListWidget()
        self.caption_widget = None
        self.flag_widget = None
        self._known_file_paths = set()

    def loadShapes(self, shapes, replace=True):
        self.canvas.shapes = list(shapes or [])
        if getattr(self, "large_image_view", None) is not None:
            self.large_image_view.set_shapes(self.canvas.shapes)

    def toggleActions(self, value):
        self._toggle_calls.append(bool(value))

    def setClean(self):
        self._clean_calls += 1
        self.dirty = False

    def update_flags_from_file(self, label_file):
        return None

    def status(self, message, delay=5000):
        self._status_messages.append(str(message))


class _SettingsStub:
    def __init__(self, values=None):
        self._values = dict(values or {})

    def value(self, key, default=None, type=None):
        value = self._values.get(key, default)
        if type is None or value is None:
            return value
        return type(value)

    def setValue(self, key, value):
        self._values[key] = value


class _VisibilityHost(LabelPanelMixin):
    def __init__(self, backend):
        self._noSelectionSlot = False
        self.canvas = _CanvasStub()
        self.large_image_view = TiledImageView()
        self.large_image_view.set_backend(backend)
        self.labelList = AnnolidLabelListWidget()
        self.uniqLabelList = QtWidgets.QListWidget()
        self.actions = type(
            "_Actions",
            (),
            {"onShapesPresent": [], "deleteShapes": None, "duplicateShapes": None},
        )()
        self.labelDialog = type(
            "_LabelDialog", (), {"addLabelHistory": lambda self, text: None}
        )()
        self._refresh_overlay_dock_calls = 0
        self._dirty_calls = 0
        self._setup_label_list_connections()

    def _refreshVectorOverlayDock(self):
        self._refresh_overlay_dock_calls += 1

    def setDirty(self):
        self._dirty_calls += 1

    def editLabel(self, *_args, **_kwargs):
        return None

    def deleteSelectedShapes(self, *_args, **_kwargs):
        return None


def _make_visible_shape() -> Shape:
    shape = Shape("atlas", shape_type="polygon")
    shape.addPoint(QtCore.QPointF(10, 10))
    shape.addPoint(QtCore.QPointF(80, 10))
    shape.addPoint(QtCore.QPointF(80, 80))
    shape.addPoint(QtCore.QPointF(10, 80))
    shape.close()
    shape.other_data = {
        "overlay_stroke": "#112233",
        "overlay_fill": "#445566",
        "overlay_opacity": 0.4,
        "overlay_z_order": 2,
        "overlay_visible": True,
    }
    return shape


def _write_page_label_json(path: Path, *, label: str, image_name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            "{\n"
            '  "version": "test",\n'
            '  "flags": {},\n'
            f'  "imagePath": "{image_name}",\n'
            '  "imageData": null,\n'
            '  "imageHeight": 40,\n'
            '  "imageWidth": 60,\n'
            '  "shapes": [\n'
            "    {\n"
            f'      "label": "{label}",\n'
            '      "points": [[10, 10], [20, 10], [20, 20], [10, 20]],\n'
            '      "group_id": null,\n'
            '      "shape_type": "polygon",\n'
            '      "flags": {},\n'
            '      "description": "",\n'
            '      "mask": null,\n'
            '      "visible": true\n'
            "    }\n"
            "  ]\n"
            "}\n"
        ),
        encoding="utf-8",
    )


def test_large_tiff_load_switches_to_tiled_view_with_base_window(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "atlas.ome.tiff"
    data = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    tifffile.imwrite(image_path, data, ome=True)

    window = _WindowStub()
    try:
        window.loadFile(str(image_path))

        assert window._active_image_view == "tiled"
        assert window.large_image_backend is not None
        assert window._viewer_stack.currentWidget() is window.large_image_view
        assert window.large_image_view.content_size() == (256, 256)
        assert window.canvas.isEnabled() is False
        assert window._toggle_calls[-1] is True
        assert window._clean_calls == 1
    finally:
        window.close()


def test_multipage_tiff_load_sets_up_large_image_page_navigation(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "multipage_stack.tif"
    data = np.stack(
        [
            np.full((40, 60), 5, dtype=np.uint8),
            np.full((40, 60), 180, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data)

    window = _WindowStub()
    try:
        window.loadFile(str(image_path))

        assert window._active_image_view == "tiled"
        assert window.large_image_backend is not None
        assert window._has_large_image_page_navigation() is True
        assert window.seekbar is not None
        assert window.playButton is not None
        assert window.playButton.text() == "Play"
        assert window.seekbar._val_max == 1
        assert window.setLargeImagePageNumber(1) is True
        assert window.frame_number == 1
        assert window.num_frames == 2
        assert window.large_image_backend.get_current_page() == 1
    finally:
        window.close()


def test_large_image_page_navigation_respects_backend_capabilities(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "multipage_stack_capabilities.tif"
    data = np.stack(
        [
            np.full((40, 60), 5, dtype=np.uint8),
            np.full((40, 60), 180, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data)

    backend = window_base_module.open_large_image(image_path)
    monkeypatch.setattr(
        backend,
        "capabilities",
        lambda: LargeImageBackendCapabilities(
            supports_pages=False,
            supports_pyramids=False,
            supports_region_reads=True,
            supports_label_stack=True,
            supports_metadata_axes=True,
            supports_cache_optimization=True,
        ),
    )

    window = _WindowStub()
    try:
        window._setup_large_image_stack_navigation(backend)

        assert window._has_large_image_page_navigation() is False
        assert getattr(window, "seekbar", None) is None
        assert getattr(window, "playButton", None) is None
    finally:
        window.close()


def test_large_image_next_prev_navigation_stays_in_tiled_view(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "multipage_stack.tif"
    data = np.stack(
        [
            np.full((40, 60), 5, dtype=np.uint8),
            np.full((40, 60), 180, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data)

    window = _WindowNavigationStub()
    try:
        window.loadFile(str(image_path))
        window.sync_large_image_page_state()
        window._seekbar_owner = "large_image_stack"

        assert window._active_image_view == "tiled"
        assert window._has_large_image_page_navigation() is True
        window.openNextImg()
        assert window._active_image_view == "tiled"
        assert window.frame_number == 1
        assert window.large_image_backend.get_current_page() == 1
        assert window._active_view_requests == []

        window.openPrevImg()
        assert window._active_image_view == "tiled"
        assert window.frame_number == 0
        assert window.large_image_backend.get_current_page() == 0
        assert window._active_view_requests == []
    finally:
        if window.timer is not None:
            window.timer.stop()
        window.close()


def test_large_image_playback_advances_pages_and_stops_at_end(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "multipage_stack.tif"
    data = np.stack(
        [
            np.full((40, 60), 5, dtype=np.uint8),
            np.full((40, 60), 180, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data)

    window = _WindowNavigationStub()
    try:
        window.loadFile(str(image_path))
        window.sync_large_image_page_state()
        window._seekbar_owner = "large_image_stack"

        assert window._has_large_image_page_navigation() is True
        window.togglePlay()
        assert window.isPlaying is True
        assert window.timer is not None and window.timer.isActive() is True

        window.openNextImg()
        assert window.frame_number == 1
        assert window.isPlaying is True

        window.openNextImg()
        assert window.frame_number == 1
        assert window.isPlaying is False
        assert window.timer is not None and window.timer.isActive() is False
        assert window.playButton.text() == "Play"
    finally:
        if window.timer is not None:
            window.timer.stop()
        window.close()


def test_multipage_tiff_loads_page_specific_annotations_and_switches_pages(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "multipage_stack.tif"
    data = np.stack(
        [
            np.full((40, 60), 5, dtype=np.uint8),
            np.full((40, 60), 180, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data)
    annotation_dir = tmp_path / "multipage_stack"
    _write_page_label_json(
        annotation_dir / "multipage_stack_000000000.json",
        label="page0",
        image_name=image_path.name,
    )
    _write_page_label_json(
        annotation_dir / "multipage_stack_000000001.json",
        label="page1",
        image_name=image_path.name,
    )

    window = _PageAnnotationWindow()
    try:
        window.loadFile(str(image_path))

        assert window._active_image_view == "tiled"
        assert len(window.canvas.shapes) == 1
        assert window.canvas.shapes[0].label == "page0"
        assert window._getLabelFile(str(image_path)).endswith(
            "multipage_stack_000000000.json"
        )

        assert window.setLargeImagePageNumber(1) is True
        assert len(window.canvas.shapes) == 1
        assert window.canvas.shapes[0].label == "page1"
        assert window._getLabelFile(str(image_path)).endswith(
            "multipage_stack_000000001.json"
        )
    finally:
        window.close()


def test_multipage_tiff_page_without_annotation_clears_shapes(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "multipage_stack.tif"
    data = np.stack(
        [
            np.full((40, 60), 5, dtype=np.uint8),
            np.full((40, 60), 180, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data)
    annotation_dir = tmp_path / "multipage_stack"
    _write_page_label_json(
        annotation_dir / "multipage_stack_000000000.json",
        label="page0",
        image_name=image_path.name,
    )

    window = _PageAnnotationWindow()
    try:
        window.loadFile(str(image_path))
        assert len(window.canvas.shapes) == 1

        assert window.setLargeImagePageNumber(1) is True
        assert window.canvas.shapes == []
    finally:
        window.close()


def test_large_tiff_load_hides_unrelated_docks_and_keeps_annotation_docks(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "atlas.ome.tiff"
    data = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    tifffile.imwrite(image_path, data, ome=True)

    window = _WindowStub()
    try:
        window.timeline_dock = QtWidgets.QDockWidget("Timeline", window)
        window.audio_dock = QtWidgets.QDockWidget("Audio", window)
        window.caption_dock = QtWidgets.QDockWidget("Caption", window)
        window.video_dock = QtWidgets.QDockWidget("Video List", window)
        for dock in (
            window.timeline_dock,
            window.audio_dock,
            window.caption_dock,
            window.video_dock,
        ):
            window.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
            dock.show()

        window.label_dock.hide()
        window.shape_dock.hide()

        window.loadFile(str(image_path))

        assert window._active_image_view == "tiled"
        assert window.timeline_dock.isHidden() is True
        assert window.audio_dock.isHidden() is True
        assert window.caption_dock.isHidden() is True
        assert window.video_dock.isHidden() is True
        assert window.file_dock.isHidden() is False
        assert window.flag_dock.isHidden() is False
        assert window.label_dock.isHidden() is False
        assert window.shape_dock.isHidden() is False

        window.setLargeImageDocksActive(False)
        assert window.timeline_dock.isHidden() is False
        assert window.audio_dock.isHidden() is False
        assert window.caption_dock.isHidden() is False
        assert window.video_dock.isHidden() is False
        assert window.label_dock.isHidden() is True
        assert window.shape_dock.isHidden() is True
        assert window.file_dock.isHidden() is False
        assert window.flag_dock.isHidden() is False
    finally:
        window.close()


def test_large_tiff_load_hides_keypoint_sequencer_dock_by_default(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "atlas.ome.tiff"
    data = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    tifffile.imwrite(image_path, data, ome=True)

    window = _WindowStub()
    try:
        window.keypoint_sequence_dock = QtWidgets.QDockWidget(
            "Keypoint Sequencer", window
        )
        window.addDockWidget(
            QtCore.Qt.RightDockWidgetArea, window.keypoint_sequence_dock
        )
        window.keypoint_sequence_dock.setVisible(True)
        assert window.keypoint_sequence_dock.isHidden() is False

        window.loadFile(str(image_path))

        assert window._active_image_view == "tiled"
        assert window.keypoint_sequence_dock.isHidden() is True
    finally:
        window.close()


def test_fit_window_uses_tiled_view_when_large_image_is_active() -> None:
    _ensure_qapp()

    window = _CanvasFitWindowStub()
    try:
        window.setFitWindow(True)

        assert window.zoomMode == window.FIT_WINDOW
        assert window.actions.fitWindow.isChecked() is True
        assert window._fit_calls == 1
    finally:
        window.close()


def test_large_tiff_load_replaces_existing_video_status_controls(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "multipage_stack.tif"
    data = np.stack(
        [
            np.full((40, 60), 5, dtype=np.uint8),
            np.full((40, 60), 180, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data)

    window = _WindowStub()
    try:
        old_seekbar = QtWidgets.QSlider(window)
        old_play = QtWidgets.QPushButton("Old Play", window)
        old_save = QtWidgets.QPushButton("Old Save", window)
        window.seekbar = old_seekbar
        window.playButton = old_play
        window.saveButton = old_save
        window._seekbar_owner = "video"
        window._play_button_owner = "video"
        window.statusBar().addPermanentWidget(old_play)
        window.statusBar().addPermanentWidget(old_seekbar)
        window.statusBar().addPermanentWidget(old_save)

        window.loadFile(str(image_path))

        assert window._has_large_image_page_navigation() is True
        assert window.seekbar is not old_seekbar
        assert window.playButton is not old_play
        assert window.saveButton is None
        assert window.playButton.text() == "Play"
    finally:
        window.close()


def test_large_image_seekbar_drag_commits_page_change_on_release(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "multipage_stack.tif"
    data = np.stack(
        [
            np.full((20, 30), 5, dtype=np.uint8),
            np.full((20, 30), 60, dtype=np.uint8),
        ],
        axis=0,
    )
    tifffile.imwrite(image_path, data)

    window = _WindowNavigationStub()
    requested_pages = []
    try:
        window.loadFile(str(image_path))

        original_request = window.requestLargeImagePageNumber

        def _recording_request(page_index: int) -> bool:
            requested_pages.append(int(page_index))
            return original_request(page_index)

        window.requestLargeImagePageNumber = _recording_request

        seekbar = window.seekbar
        assert seekbar is not None

        seekbar.mousePressed.emit(0.0, 0.0)
        seekbar.setValue(1)
        assert requested_pages == []

        seekbar.mouseReleased.emit(0.0, 0.0)
        assert requested_pages == [1]
        assert window.frame_number == 1
        assert window.large_image_backend.get_current_page() == 1
    finally:
        if window.timer is not None:
            window.timer.stop()
        window.close()


def test_regular_image_load_uses_canvas_view_with_base_window(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "plain.png"
    image = QtGui.QImage(64, 32, QtGui.QImage.Format_RGB32)
    image.fill(QtGui.QColor(10, 20, 30))
    assert image.save(str(image_path))

    window = _WindowStub()
    try:
        window.loadFile(str(image_path))

        assert window._active_image_view == "canvas"
        assert window.large_image_backend is None
        assert window._viewer_stack.currentWidget() is window.canvas
        assert window.canvas.isEnabled() is True
        assert window.canvas.last_pixmap is not None
        assert window.canvas.last_pixmap.size() == image.size()
    finally:
        window.close()


def test_activate_large_image_canvas_edit_mode_preserves_shapes(tmp_path: Path) -> None:
    _ensure_qapp()

    window = _WindowStub()
    try:
        window.imagePath = str(tmp_path / "atlas.ome.tiff")
        window.image = QtGui.QImage(64, 32, QtGui.QImage.Format_RGB32)
        window.image.fill(QtGui.QColor(10, 20, 30))
        shape = _make_visible_shape()
        window.canvas.shapes = [shape]
        window._active_image_view = "tiled"
        window._viewer_stack.setCurrentWidget(window.large_image_view)
        window.large_image_backend = object()

        changed = window.activateLargeImageCanvasEditMode(reason="overlay editing")

        assert changed is True
        assert window._active_image_view == "canvas"
        assert window._viewer_stack.currentWidget() is window.canvas
        assert window.canvas.last_pixmap is not None
        assert window.canvas.last_load_clear_shapes is False
        assert window.canvas.shapes == [shape]
        assert window.canvas.editing_values[-1] is True
        assert window.large_image_backend is not None
        assert window._status_messages == []
    finally:
        window.close()


def test_large_tiff_load_avoids_duplicate_generic_loader_path(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "atlas.ome.tiff"
    data = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    tifffile.imwrite(image_path, data, ome=True)

    original_open_large_image = window_base_module.open_large_image
    open_calls = []
    generic_loader_calls = []

    def counting_open_large_image(path):
        open_calls.append(str(path))
        return original_open_large_image(path)

    def fail_generic_loader(path):
        generic_loader_calls.append(str(path))
        raise AssertionError(
            "load_image_with_backends should not run for large TIFF open"
        )

    monkeypatch.setattr(
        window_base_module, "open_large_image", counting_open_large_image
    )
    monkeypatch.setattr(
        window_base_module, "load_image_with_backends", fail_generic_loader
    )

    window = _WindowStub()
    try:
        window.loadFile(str(image_path))

        assert window._active_image_view == "tiled"
        assert open_calls == [str(image_path)]
        assert generic_loader_calls == []
    finally:
        window.close()


def test_large_tiff_load_uses_fresh_optimized_cache_when_available(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    source_path = tmp_path / "atlas.ome.tiff"
    cached_path = tmp_path / "atlas_cached.tif"
    data = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    tifffile.imwrite(source_path, data, ome=True)
    tifffile.imwrite(cached_path, data)

    original_open_large_image = window_base_module.open_large_image
    open_calls = []

    def counting_open_large_image(path):
        open_calls.append(str(path))
        return original_open_large_image(path)

    monkeypatch.setattr(
        window_base_module,
        "resolve_fresh_optimized_large_image_path",
        lambda path: cached_path,
    )
    monkeypatch.setattr(
        window_base_module, "open_large_image", counting_open_large_image
    )

    window = _WindowStub()
    try:
        window.loadFile(str(source_path))

        assert window._active_image_view == "tiled"
        assert open_calls == [str(cached_path)]
        assert window.imagePath == str(source_path)
        assert window.otherData["large_image"]["optimized_cache_path"] == str(
            cached_path
        )
    finally:
        window.close()


def test_optimize_large_image_for_viewing_updates_metadata_and_status(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    source_path = tmp_path / "atlas.ome.tiff"
    cached_path = tmp_path / "atlas_cached.tif"
    data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    tifffile.imwrite(source_path, data, ome=True)
    original_open_large_image = window_base_module.open_large_image
    monkeypatch.setattr(
        window_base_module,
        "open_large_image",
        lambda path: original_open_large_image(path),
    )

    window = _WindowStub()
    try:
        window.otherData = {}
        window.canvas.shapes = [_make_visible_shape()]
        result = window._applyOptimizedLargeImageCache(source_path, cached_path)

        assert result == str(cached_path)
        assert window.otherData["large_image"]["optimized_cache_path"] == str(
            cached_path
        )
        assert window._status_messages
        assert "optimized pyramidal tiff cache" in window._status_messages[-1].lower()
        assert window._dirty_calls == 1
    finally:
        window.close()


def test_finish_large_image_optimization_cleans_up_and_applies_cache(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    source_path = tmp_path / "atlas.ome.tiff"
    cached_path = tmp_path / "atlas_cached.tif"
    data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    tifffile.imwrite(source_path, data, ome=True)
    tifffile.imwrite(cached_path, data)
    original_open_large_image = window_base_module.open_large_image
    monkeypatch.setattr(
        window_base_module,
        "open_large_image",
        lambda path: original_open_large_image(path),
    )

    class _ThreadStub:
        def __init__(self):
            self.quit_called = 0
            self.wait_calls = []
            self.delete_calls = 0

        def quit(self):
            self.quit_called += 1

        def wait(self, timeout):
            self.wait_calls.append(timeout)

        def deleteLater(self):
            self.delete_calls += 1

    class _WorkerStub:
        def __init__(self):
            self.delete_calls = 0

        def deleteLater(self):
            self.delete_calls += 1

    class _ProgressStub:
        def __init__(self):
            self.close_calls = 0

        def close(self):
            self.close_calls += 1

    window = _WindowStub()
    try:
        window.canvas.shapes = [_make_visible_shape()]
        window._large_image_opt_source_path = source_path
        window._large_image_opt_thread = _ThreadStub()
        window._large_image_opt_worker = _WorkerStub()
        window._large_image_opt_progress = _ProgressStub()

        window._finishLargeImageOptimization(str(cached_path))

        assert window.otherData["large_image"]["optimized_cache_path"] == str(
            cached_path
        )
        assert window._large_image_opt_thread is None
        assert window._large_image_opt_worker is None
        assert window._large_image_opt_progress is None
        assert window._status_messages
    finally:
        window.close()


def test_show_large_image_cache_info_reports_current_cache(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    cache_root = tmp_path / "cache_root"
    cached_path = cache_root / "atlas_cached.pyramidal.tif"
    cached_path.parent.mkdir(parents=True, exist_ok=True)
    cached_path.write_bytes(b"x" * 128)

    monkeypatch.setattr(
        window_base_module, "large_image_cache_root", lambda: cache_root
    )
    monkeypatch.setattr(
        window_base_module,
        "list_large_image_cache_entries",
        lambda: [],
    )
    monkeypatch.setattr(window_base_module, "large_image_cache_size_bytes", lambda: 128)

    captured = {}

    def fake_information(_parent, title, text):
        captured["title"] = str(title)
        captured["text"] = str(text)
        return QtWidgets.QMessageBox.Ok

    monkeypatch.setattr(QtWidgets.QMessageBox, "information", fake_information)

    window = _WindowStub()
    try:
        window.settings = _SettingsStub(
            {
                window_base_module.LARGE_IMAGE_CACHE_MAX_ENTRIES_KEY: 7,
                window_base_module.LARGE_IMAGE_CACHE_MAX_SIZE_GB_KEY: 9,
            }
        )
        window.otherData = {"large_image": {"optimized_cache_path": str(cached_path)}}

        info = window.showLargeImageCacheInfo()

        assert info["current_cache_exists"] is True
        assert info["current_cache_path"] == str(cached_path)
        assert info["policy"]["max_entries"] == 7
        assert "Large Image Cache" in captured["title"]
        assert str(cached_path) in captured["text"]
        assert "128 B" in captured["text"]
        assert "7 file(s), 9 GB" in captured["text"]
    finally:
        window.close()


def test_large_image_cache_policy_uses_defaults_and_persists_settings() -> None:
    _ensure_qapp()

    window = _WindowStub()
    try:
        window.settings = _SettingsStub()

        defaults = window.largeImageCachePolicy()
        saved = window.setLargeImageCachePolicy(max_entries=5, max_size_gb=8)

        assert defaults["max_entries"] >= 1
        assert defaults["max_size_gb"] >= 1
        assert saved["max_entries"] == 5
        assert saved["max_size_gb"] == 8
        assert (
            window.settings.value(window_base_module.LARGE_IMAGE_CACHE_MAX_ENTRIES_KEY)
            == 5
        )
        assert (
            window.settings.value(window_base_module.LARGE_IMAGE_CACHE_MAX_SIZE_GB_KEY)
            == 8
        )
    finally:
        window.close()


def test_configure_large_image_cache_policy_updates_settings_and_status(
    monkeypatch,
) -> None:
    _ensure_qapp()

    responses = iter([(6, True), (11, True)])
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getInt",
        lambda *args, **kwargs: next(responses),
    )

    window = _WindowStub()
    try:
        window.settings = _SettingsStub()

        result = window.configureLargeImageCachePolicy()

        assert result == {
            "max_entries": 6,
            "max_size_gb": 11,
            "max_size_bytes": 11 * 1024 * 1024 * 1024,
        }
        assert "6 file(s), 11 GB" in window._status_messages[-1]
    finally:
        window.close()


def test_large_image_cache_optimize_options_use_configured_limits() -> None:
    _ensure_qapp()

    window = _WindowStub()
    try:
        window.settings = _SettingsStub(
            {
                window_base_module.LARGE_IMAGE_CACHE_MAX_ENTRIES_KEY: 4,
                window_base_module.LARGE_IMAGE_CACHE_MAX_SIZE_GB_KEY: 13,
            }
        )

        options = window.largeImageCacheOptimizeOptions()

        assert options == {
            "max_cache_entries": 4,
            "max_cache_size_bytes": 13 * 1024 * 1024 * 1024,
        }
    finally:
        window.close()


def test_open_large_image_cache_folder_reports_success(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    cache_root = tmp_path / "cache_root"
    monkeypatch.setattr(
        window_base_module, "large_image_cache_root", lambda: cache_root
    )

    opened = {}

    def fake_open_url(url):
        opened["url"] = url.toLocalFile()
        return True

    monkeypatch.setattr(QtGui.QDesktopServices, "openUrl", fake_open_url)

    window = _WindowStub()
    try:
        result = window.openLargeImageCacheFolder()

        assert result == str(cache_root)
        assert opened["url"] == str(cache_root)
        assert cache_root.exists()
    finally:
        window.close()


def test_clear_current_large_image_cache_reloads_source(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    source_path = tmp_path / "atlas.ome.tiff"
    cached_path = tmp_path / "atlas_cached.pyramidal.tif"
    data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    tifffile.imwrite(source_path, data, ome=True)
    tifffile.imwrite(cached_path, data)

    original_open_large_image = window_base_module.open_large_image
    open_calls = []

    def counting_open_large_image(path):
        open_calls.append(str(path))
        return original_open_large_image(path)

    monkeypatch.setattr(
        window_base_module, "open_large_image", counting_open_large_image
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Yes,
    )

    window = _WindowStub()
    try:
        window.imagePath = str(source_path)
        window.otherData = {"large_image": {"optimized_cache_path": str(cached_path)}}
        window.canvas.shapes = [_make_visible_shape()]

        removed = window.clearCurrentLargeImageCache()

        assert removed == 1
        assert not cached_path.exists()
        assert "optimized_cache_path" not in window.otherData["large_image"]
        assert open_calls == [str(source_path)]
        assert window._active_image_view == "tiled"
    finally:
        window.close()


def test_clear_all_large_image_caches_clears_current_reference(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    source_path = tmp_path / "atlas.ome.tiff"
    cached_path = tmp_path / "atlas_cached.pyramidal.tif"
    extra_cached_path = tmp_path / "atlas_extra.pyramidal.tif"
    data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    tifffile.imwrite(source_path, data, ome=True)
    tifffile.imwrite(cached_path, data)
    tifffile.imwrite(extra_cached_path, data)

    class _Entry:
        def __init__(self, path):
            self.path = path

    monkeypatch.setattr(
        window_base_module,
        "list_large_image_cache_entries",
        lambda: [_Entry(cached_path), _Entry(extra_cached_path)],
    )
    monkeypatch.setattr(
        window_base_module,
        "clear_all_large_image_caches",
        lambda: (cached_path.unlink(), extra_cached_path.unlink(), 2)[-1],
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: QtWidgets.QMessageBox.Yes,
    )

    original_open_large_image = window_base_module.open_large_image
    open_calls = []

    def counting_open_large_image(path):
        open_calls.append(str(path))
        return original_open_large_image(path)

    monkeypatch.setattr(
        window_base_module, "open_large_image", counting_open_large_image
    )

    window = _WindowStub()
    try:
        window.imagePath = str(source_path)
        window.otherData = {"large_image": {"optimized_cache_path": str(cached_path)}}
        removed = window.clearAllLargeImageCaches()

        assert removed == 2
        assert not cached_path.exists()
        assert not extra_cached_path.exists()
        assert "optimized_cache_path" not in window.otherData["large_image"]
        assert open_calls == [str(source_path)]
    finally:
        window.close()


def test_tiled_image_view_clamps_tiles_and_renders_visible_shapes(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()

        view.set_backend(backend)
        view.set_zoom_percent(100)

        tile_keys = view.visible_tile_keys(level=0)
        max_tx = (520 - 1) // 128
        max_ty = (300 - 1) // 128

        assert tile_keys
        assert all(0 <= key.tx <= max_tx for key in tile_keys)
        assert all(0 <= key.ty <= max_ty for key in tile_keys)
        assert len(view._tile_items) > 0

        visible = _make_visible_shape()
        hidden = _make_visible_shape()
        hidden.visible = False
        overlay_point = Shape("overlay_landmark", shape_type="point")
        overlay_point.addPoint(QtCore.QPointF(20, 20))
        overlay_point.other_data = {
            "overlay_id": "overlay_a",
            "overlay_landmark_pair_id": "pair_1",
            "overlay_visible": True,
        }
        image_point = Shape("image_landmark", shape_type="point")
        image_point.addPoint(QtCore.QPointF(140, 120))
        image_point.other_data = {"overlay_landmark_pair_id": "pair_1"}
        view.set_shapes([visible, hidden])

        assert len(view._overlay_items) == 1
        assert abs(view._overlay_items[0].opacity() - 0.4) < 1e-6
        assert abs(view._overlay_items[0].zValue() - 102.0) < 1e-6
        base_scale = view._overlay_items[0].current_scale
        base_point_pixels = view._overlay_items[0]._visual_metrics()

        view.set_zoom_percent(400)

        assert view._overlay_items[0].current_scale > base_scale
        assert view._overlay_items[0]._visual_metrics() < base_point_pixels

        view.set_shapes([visible, overlay_point, image_point])

        assert len(view._pair_items) == 1
        assert isinstance(view._pair_items[0], QtWidgets.QGraphicsLineItem)
        assert len(view._pair_endpoint_items) == 2
        assert abs(view._pair_items[0].pen().widthF() - 2.0) < 1e-6

        received = []
        view.overlayLandmarkPairSelected.connect(received.append)
        view.set_selected_landmark_pair("pair_1")

        assert abs(view._pair_items[0].pen().widthF() - 3.0) < 1e-6
        assert all(
            abs(item.pen().widthF() - 2.0) < 1e-6 for item in view._pair_endpoint_items
        )

        item_center = view.mapFromScene(view._pair_items[0].line().pointAt(0.5))
        hit_item = view._pair_item_at_view_pos(item_center)
        assert hit_item is not None
        view.set_selected_landmark_pair(hit_item.pair_id)
        view.overlayLandmarkPairSelected.emit(hit_item.pair_id)

        assert received[-1] == "pair_1"

        view.clear()

        assert view.content_size() == (0, 0)
        assert len(view._tile_items) == 0
        assert len(view._overlay_items) == 0
        assert len(view._pair_items) == 0
        assert len(view._pair_endpoint_items) == 0
        assert len(view.tile_cache) == 0
    finally:
        view.close()


def test_tiled_image_view_tile_scheduler_reuses_cached_tiles(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "scheduler.ome.tiff"
    data = np.arange(512 * 512, dtype=np.uint16).reshape(512, 512)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()

        view.set_backend(backend)
        first_stats = view.tile_scheduler_stats()["raster"]

        assert first_stats["loads_completed"] >= 1

        view.refresh_visible_tiles()
        second_stats = view.tile_scheduler_stats()["raster"]

        assert second_stats["cache_hits"] > first_stats["cache_hits"]
        assert second_stats["loads_completed"] == first_stats["loads_completed"]
    finally:
        view.close()


def test_label_visibility_toggle_refreshes_tiled_view(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_visibility_sync.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    host = _VisibilityHost(backend)
    try:
        shape = _make_visible_shape()
        host.canvas.shapes = [shape]
        host.large_image_view.set_shapes(host.canvas.shapes)

        assert len(host.large_image_view._overlay_items) == 1

        item = AnnolidLabelListItem("atlas", shape)
        item.setFlags(
            item.flags() | QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsSelectable
        )
        item.setCheckState(QtCore.Qt.Checked)
        item.setData(host.labelList.VISIBILITY_STATE_ROLE, True)
        host.labelList.addItem(item)

        item.setCheckState(QtCore.Qt.Unchecked)
        _ensure_qapp().processEvents()

        assert shape.visible is False
        assert len(host.large_image_view._overlay_items) == 0
        assert host._refresh_overlay_dock_calls >= 1
        assert host._dirty_calls >= 1
    finally:
        host.large_image_view.close()
        host.labelList.close()


def test_tiled_overlay_item_reduces_highlight_multiplier_at_high_zoom() -> None:
    shape = _make_visible_shape()
    item = _ShapeGraphicsItem(
        shape,
        content_size=(24000, 16000),
        current_scale=6.0,
    )

    settings = item._effective_highlight_settings()

    assert (
        settings[Shape.NEAR_VERTEX][0] < shape._highlightSettings[Shape.NEAR_VERTEX][0]
    )
    assert (
        settings[Shape.MOVE_VERTEX][0] < shape._highlightSettings[Shape.MOVE_VERTEX][0]
    )
    overrides = item._effective_vertex_render_overrides()
    assert overrides["glow_alpha_mult"] < 1.0
    assert overrides["halo_alpha_mult"] < 1.0
    assert overrides["glow_scale"] < 1.75


def test_tiled_image_view_clicking_polygon_edge_inserts_point(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_insert_edge.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)

        shape = _make_visible_shape()
        view.set_shapes([shape])

        moved = []
        view.shapeMoved.connect(lambda: moved.append(True))

        edge_click = view.mapFromScene(QtCore.QPointF(45, 10))
        _send_mouse_click(view, edge_click)

        assert len(shape.points) == 5
        assert moved
        assert (round(shape.points[1].x(), 1), round(shape.points[1].y(), 1)) == (
            45.0,
            10.0,
        )
        assert view.selectedShapes == [shape]
    finally:
        view.close()


def test_tiled_image_view_reuses_overlay_graphics_items_across_geometry_updates(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_reuse_overlay_items.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)

        shape = _make_visible_shape()
        view.set_shapes([shape])

        assert len(view._overlay_items) == 1
        first_item = view._overlay_items[0]

        # Mutate geometry as a drag operation would do, then resync.
        shape.moveBy(QtCore.QPointF(7.0, 5.0))
        view.set_shapes([shape])

        assert len(view._overlay_items) == 1
        assert view._overlay_items[0] is first_item
    finally:
        view.close()


def test_tiled_image_view_supports_native_overlay_selection_and_drag(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_native_edit.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)

        shape = _make_visible_shape()
        shape.other_data["overlay_id"] = "overlay_a"
        view.set_shapes([shape])

        selected = []
        moved = []
        view.selectionChanged.connect(
            lambda shapes: selected.append(list(shapes or []))
        )
        view.shapeMoved.connect(lambda: moved.append(True))

        click_pos = view.mapFromScene(QtCore.QPointF(20, 20))
        _send_mouse_click(view, click_pos)

        assert selected
        assert selected[-1] == [shape]
        assert view.selectedShapes == [shape]
        assert shape.selected is True
        assert shape.selected is True
        assert shape._highlightIndex is None

        drag_end = view.mapFromScene(QtCore.QPointF(35, 32))
        _send_mouse_drag(view, click_pos, drag_end)

        assert moved
        assert [(round(p.x(), 1), round(p.y(), 1)) for p in shape.points] == [
            (25.0, 22.0),
            (95.0, 22.0),
            (95.0, 92.0),
            (25.0, 92.0),
        ]
    finally:
        view.close()


def test_tiled_image_view_supports_native_overlay_vertex_drag(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_native_vertex.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)

        shape = _make_visible_shape()
        shape.other_data["overlay_id"] = "overlay_a"
        view.set_shapes([shape])
        view.set_selected_shapes([shape])

        vertex_start = view.mapFromScene(QtCore.QPointF(10, 10))
        vertex_end = view.mapFromScene(QtCore.QPointF(18, 16))
        _send_mouse_drag(view, vertex_start, vertex_end)

        assert (round(shape.points[0].x(), 1), round(shape.points[0].y(), 1)) == (
            18.0,
            16.0,
        )
        assert shape._highlightIndex is None
    finally:
        view.close()


def test_tiled_image_view_remove_selected_vertex_updates_polygon(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_remove_vertex.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)

        shape = _make_visible_shape()
        shape.other_data["overlay_id"] = "overlay_a"
        view.set_shapes([shape])
        view.set_selected_shapes([shape])
        view._set_selected_vertex(shape, 1)

        moved = []
        view.shapeMoved.connect(lambda: moved.append(True))

        removed = view.removeSelectedPoint()

        assert removed is True
        assert len(shape.points) == 3
        assert moved
    finally:
        view.close()


def test_tiled_image_view_supports_native_point_creation_and_labeling(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_native_point.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)
        view.setEditing(False)
        view.createMode = "point"

        created = []
        view.newShape.connect(lambda: created.append(True))

        click_pos = view.mapFromScene(QtCore.QPointF(44, 55))
        _send_mouse_click(view, click_pos)

        assert created
        assert len(view._shapes) == 1
        assert view._shapes[0].shape_type == "point"
        assert (
            round(view._shapes[0].points[0].x(), 1),
            round(view._shapes[0].points[0].y(), 1),
        ) == (44.0, 55.0)

        labeled = view.setLastLabel("landmark_a", {})

        assert labeled[0].label == "landmark_a"
    finally:
        view.close()


def test_tiled_image_view_supports_native_polygon_creation(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_native_polygon.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)
        view.setEditing(False)
        view.createMode = "polygon"
        first_click = view.mapFromScene(QtCore.QPointF(20, 20))
        second_pos = view.mapFromScene(QtCore.QPointF(70, 20))
        third_pos = view.mapFromScene(QtCore.QPointF(70, 65))
        _send_mouse_click(view, first_click)
        move_event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(second_pos),
            QtCore.QPointF(view.viewport().mapToGlobal(second_pos)),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
        view.mouseMoveEvent(move_event)

        assert not view._preview_item.path().isEmpty()
        assert not view._preview_vertices_item.path().isEmpty()
        assert view._preview_close_item.path().isEmpty()

        _send_mouse_click(view, second_pos)
        _send_mouse_click(view, third_pos)

        close_hover = view.mapFromScene(QtCore.QPointF(21, 21))
        close_move_event = QtGui.QMouseEvent(
            QtCore.QEvent.MouseMove,
            QtCore.QPointF(close_hover),
            QtCore.QPointF(view.viewport().mapToGlobal(close_hover)),
            QtCore.Qt.NoButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
        view.mouseMoveEvent(close_move_event)

        assert not view._preview_item.path().isEmpty()
        assert not view._preview_vertices_item.path().isEmpty()
        assert not view._preview_close_item.path().isEmpty()

        _send_mouse_click(view, close_hover)

        assert len(view._shapes) == 1
        assert view._shapes[0].shape_type == "polygon"
        assert view._shapes[0].isClosed() is True
        assert len(view._shapes[0].points) == 3
        assert view._preview_item.path().isEmpty()
        assert view._preview_vertices_item.path().isEmpty()
        assert view._preview_close_item.path().isEmpty()
    finally:
        view.close()


def test_tiled_image_view_context_menu_uses_canvas_builder(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_context_menu.ome.tiff"
    data = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    window = _WindowStub()
    executed = []

    def fake_exec(self, pos):
        executed.append(pos)
        return None

    monkeypatch.setattr(QtWidgets.QMenu, "exec_", fake_exec)

    try:
        view = window.large_image_view
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)

        assert view._show_context_menu(QtCore.QPoint(10, 10)) is True
        assert executed
        assert window.canvas.context_menu_builds == 1
    finally:
        window.close()


def test_tiled_image_view_can_edit_native_non_overlay_polygon(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_native_edit_polygon.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image
    from annolid.gui.shape import Shape

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)

        shape = Shape("native_polygon", shape_type="polygon")
        shape.addPoint(QtCore.QPointF(20, 20))
        shape.addPoint(QtCore.QPointF(70, 20))
        shape.addPoint(QtCore.QPointF(70, 65))
        shape.close()
        view.set_shapes([shape])

        click_pos = view.mapFromScene(QtCore.QPointF(60, 30))
        drag_end = view.mapFromScene(QtCore.QPointF(74, 36))
        _send_mouse_drag(view, click_pos, drag_end)

        assert shape.selected is True
        assert [(round(p.x(), 1), round(p.y(), 1)) for p in shape.points] == [
            (34.0, 26.0),
            (84.0, 26.0),
            (84.0, 71.0),
        ]
    finally:
        view.close()


def test_tiled_image_view_supports_native_rectangle_creation(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_native_rectangle.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)
        view.setEditing(False)
        view.createMode = "rectangle"

        from annolid.gui.shape import Shape

        view.current = Shape(shape_type="rectangle")
        view.current.addPoint(QtCore.QPointF(25, 30))
        view.current.addPoint(QtCore.QPointF(95, 90))
        view.finalise()

        assert len(view._shapes) == 1
        shape = view._shapes[0]
        assert shape.shape_type == "rectangle"
        assert shape.isClosed() is True
        assert [(round(p.x(), 1), round(p.y(), 1)) for p in shape.points] == [
            (25.0, 30.0),
            (95.0, 90.0),
        ]
    finally:
        view.close()


def test_tiled_image_view_supports_native_circle_creation(tmp_path: Path) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_native_circle.ome.tiff"
    data = np.arange(300 * 520, dtype=np.uint16).reshape(300, 520)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    view = TiledImageView(tile_size=128)
    try:
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)
        view.set_zoom_percent(100)
        view.setEditing(False)
        view.createMode = "circle"

        from annolid.gui.shape import Shape

        view.current = Shape(shape_type="circle")
        view.current.addPoint(QtCore.QPointF(60, 60))
        view.current.addPoint(QtCore.QPointF(90, 60))
        view.finalise()

        assert len(view._shapes) == 1
        shape = view._shapes[0]
        assert shape.shape_type == "circle"
        assert len(shape.points) == 2
        assert [(round(p.x(), 1), round(p.y(), 1)) for p in shape.points] == [
            (60.0, 60.0),
            (90.0, 60.0),
        ]
    finally:
        view.close()


def test_large_image_mode_widgets_show_surface_fallback_and_return(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_mode_widgets.ome.tiff"
    data = np.arange(120 * 180, dtype=np.uint16).reshape(120, 180)
    tifffile.imwrite(image_path, data, ome=True)

    window = _WindowNavigationStub()
    try:
        image = QtGui.QImage(180, 120, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(25, 35, 45))
        window.image = image
        window.canvas.loadPixmap(QtGui.QPixmap.fromImage(image), clear_shapes=False)
        window.loadFile(str(image_path))

        surface_label = window._large_image_surface_label
        mode_label = window._large_image_mode_label
        return_button = window._large_image_return_button
        assert surface_label is not None
        assert mode_label is not None
        assert return_button is not None
        assert surface_label.parent() is window
        assert mode_label.parent() is window
        assert return_button.parent() is window

        window.toggleDrawMode(False, createMode="ai_polygon")

        assert window._active_image_view == "canvas"
        assert surface_label.isHidden() is True
        assert mode_label.isHidden() is True
        assert return_button.isHidden() is True

        changed = window.returnToLargeImageTiledView()

        assert changed is True
        assert window._active_image_view == "tiled"
        assert surface_label.isHidden() is True
        assert return_button.isHidden() is True
    finally:
        if window.timer is not None:
            window.timer.stop()
        window.close()


def test_tiled_image_view_debug_status_reports_backend_tiles_and_cache(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    image_path = tmp_path / "sample_debug_status.ome.tiff"
    data = np.arange(220 * 340, dtype=np.uint16).reshape(220, 340)
    tifffile.imwrite(image_path, data, ome=True)

    from annolid.io.large_image import open_large_image

    backend = open_large_image(image_path)
    window = _WindowStub()
    try:
        window.imagePath = str(image_path)
        window.large_image_backend = backend
        window._active_image_view = "tiled"
        window._viewer_stack.setCurrentWidget(window.large_image_view)
        view = window.large_image_view
        view.resize(320, 240)
        view.show()
        _ensure_qapp().processEvents()
        view.set_backend(backend)

        status_text = view.debug_status_text()
        overlay_text = view._status_overlay.current_text()

        assert "backend=" in status_text
        assert "page=" in status_text
        assert "zoom=" in status_text
        assert "tiles=" in status_text
        assert "cache=" in status_text
        assert "Tiled Viewer" in overlay_text
        assert "visible tiles=" in overlay_text
    finally:
        window.close()
