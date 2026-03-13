from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import tifffile
from qtpy import QtCore, QtGui, QtTest, QtWidgets

from annolid.gui.shape import Shape
from annolid.gui.widgets.tiled_image_view import TiledImageView
from annolid.gui.window_base import AnnolidWindowBase
import annolid.gui.window_base as window_base_module


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
    QtWidgets.QApplication.sendEvent(viewport, press)
    QtWidgets.QApplication.sendEvent(viewport, move)
    QtWidgets.QApplication.sendEvent(viewport, release)


class _CanvasStub(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.last_pixmap = None
        self.shapes = []
        self.last_load_clear_shapes = None
        self.editing_values = []

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.last_pixmap = pixmap
        self.last_load_clear_shapes = bool(clear_shapes)
        if clear_shapes:
            self.shapes = []

    def setEditing(self, value=True):
        self.editing_values.append(bool(value))


class _WindowStub(AnnolidWindowBase):
    def __init__(self):
        self._toggle_calls = []
        self._clean_calls = 0
        self._status_messages = []
        self._dirty_calls = 0
        super().__init__(config={})
        self.canvas = _CanvasStub()
        self.large_image_view = TiledImageView(self)
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
        assert "canvas preview mode" in window._status_messages[-1]
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
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.LeftButton, pos=item_center)

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
        QtTest.QTest.mousePress(view.viewport(), QtCore.Qt.LeftButton, pos=click_pos)
        QtTest.QTest.mouseRelease(view.viewport(), QtCore.Qt.LeftButton, pos=click_pos)

        assert selected
        assert selected[-1] == [shape]
        assert view.selectedShapes == [shape]
        assert shape.selected is True
        assert len(view._vertex_items) == len(shape.points)

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
        assert len(view._vertex_items) == len(shape.points)
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
        QtTest.QTest.mouseClick(view.viewport(), QtCore.Qt.LeftButton, pos=click_pos)

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

        from annolid.gui.shape import Shape

        view.current = Shape(shape_type="polygon")
        for scene_point in (
            QtCore.QPointF(20, 20),
            QtCore.QPointF(70, 20),
            QtCore.QPointF(70, 65),
        ):
            view.current.addPoint(QtCore.QPointF(scene_point))

        view.finalise()

        assert len(view._shapes) == 1
        assert view._shapes[0].shape_type == "polygon"
        assert view._shapes[0].isClosed() is True
        assert len(view._shapes[0].points) == 3
    finally:
        view.close()
