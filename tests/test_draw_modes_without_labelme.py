import builtins
import importlib
import os
import threading
from types import SimpleNamespace

import numpy as np
from qtpy import QtCore, QtWidgets, QtGui


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_draw_mode_actions_toggle_canvas_modes():
    _ensure_qapp()

    # Import lazily after QT_QPA_PLATFORM is set.
    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        img = QtGui.QImage(120, 80, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(100, 120, 140))
        w.image_to_canvas(img, "dummy.png", 0)

        assert w.canvas.editing()

        w.actions.createRectangleMode.setChecked(True)
        assert w.canvas.drawing()
        assert w.canvas.createMode == "rectangle"

        w.actions.editMode.setChecked(True)
        assert w.canvas.editing()

        w.actions.createMode.setChecked(True)
        assert w.canvas.drawing()
        assert w.canvas.createMode == "polygon"

        w.actions.createAiPolygonMode.setChecked(True)
        assert w.canvas.drawing()
        assert w.canvas.createMode == "ai_polygon"
    finally:
        w.close()


def test_annotation_compat_provides_ai_models_without_labelme(monkeypatch):
    """AI polygon mode should still have usable models when labelme is absent."""

    _real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "labelme" or name.startswith("labelme."):
            raise ModuleNotFoundError("forced missing labelme")
        return _real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    import annolid.utils.annotation_compat as ac

    importlib.reload(ac)

    names = [m.name for m in ac.AI_MODELS]
    assert "SegmentAnything (Edge)" in names
    assert "EfficientSam (speed)" in names

    # Basic sanity check: the Edge model can be instantiated and run.
    model_cls = next(m for m in ac.AI_MODELS if m.name == "SegmentAnything (Edge)")
    model = model_cls()
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[16:48, 16:48] = 255
    model.set_image(image)
    mask = model.predict_mask_from_points(points=[[32, 32]], point_labels=[1])
    assert getattr(mask, "shape", None) == (64, 64)


def test_ai_model_image_refreshes_after_pixmap_switch():
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        first = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        first.fill(QtGui.QColor(10, 20, 30))
        w.image_to_canvas(first, "first.png", 0)

        class FakeAiModel:
            name = "fake"

            def __init__(self):
                self.images = []

            def set_image(self, image):
                self.images.append(np.asarray(image).copy())

        fake_model = FakeAiModel()
        w.canvas._ai_model = fake_model
        w.canvas._ai_model_pixmap_key = None
        w.canvas._sync_ai_model_image(force=True)
        assert len(fake_model.images) == 1

        second = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        second.fill(QtGui.QColor(200, 100, 50))
        w.canvas.loadPixmap(QtGui.QPixmap.fromImage(second), clear_shapes=False)
        assert len(fake_model.images) == 2
        assert not np.array_equal(fake_model.images[0], fake_model.images[1])
    finally:
        w.close()


def test_switching_ai_models_closes_previous_instance(monkeypatch):
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow
    import annolid.gui.widgets.canvas as canvas_mod

    class NewAiModel:
        name = "NewTestModel"

        def set_image(self, image):
            _ = image

    class OldAiModel:
        name = "OldTestModel"

        def __init__(self):
            self.closed = False

        def close(self):
            self.closed = True

        def set_image(self, image):
            _ = image

    monkeypatch.setattr(canvas_mod, "AI_MODELS", [NewAiModel])

    w = AnnolidWindow(config={})
    try:
        img = QtGui.QImage(96, 64, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(80, 90, 100))
        w.image_to_canvas(img, "switch_test.png", 0)

        old_model = OldAiModel()
        w.canvas._ai_model = old_model
        w.canvas._ai_model_pixmap_key = None

        w.canvas.initializeAiModel("NewTestModel")

        assert old_model.closed is True
        assert isinstance(w.canvas._ai_model, NewAiModel)
    finally:
        w.close()


def test_keypoint_sequence_toolbar_action_syncs_with_checkbox():
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        action = getattr(w, "toggle_keypoint_sequence_action", None)
        assert action is not None
        assert action.shortcut().toString() in {"Ctrl+Shift+K", "Meta+Shift+K"}

        # Action -> checkbox
        action.setChecked(True)
        assert w.keypoint_sequence_widget.enable_checkbox.isChecked() is True

        # Checkbox -> action
        w.keypoint_sequence_widget.enable_checkbox.setChecked(False)
        assert action.isChecked() is False
    finally:
        w.close()


def test_draw_mode_actions_expose_mode_switch_shortcuts():
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        create_keys = set(w.actions.createMode.shortcuts())
        point_keys = set(w.actions.createPointMode.shortcuts())
        edit_keys = set(w.actions.editMode.shortcuts())
        create_texts = {seq.toString() for seq in create_keys}
        point_texts = {seq.toString() for seq in point_keys}
        edit_texts = {seq.toString() for seq in edit_keys}

        assert create_texts & {"Ctrl+N", "Meta+N"}
        assert point_texts & {"Ctrl+I", "Meta+I"}
        assert edit_texts & {"Ctrl+J", "Meta+J"}
    finally:
        w.close()


class _ActivePainterProbe:
    def __init__(self, active_states, *, points=None, point_labels=None):
        self._active_states = active_states
        self.points = list(points or [])
        self.point_labels = list(point_labels or [])
        self.fill = False
        self.selected = False

    def paint(self, painter, *args):
        self._active_states.append(bool(painter.isActive()))

    def copy(self):
        return _ActivePainterProbe(
            self._active_states,
            points=self.points,
            point_labels=self.point_labels,
        )

    def addPoint(self, point, label=1):
        self.points.append(point)
        self.point_labels.append(label)

    def setShapeRefined(self, **kwargs):
        _ = kwargs


def test_ai_polygon_preview_paints_with_active_painter(monkeypatch):
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    active_states = []
    canvas = Canvas()
    try:
        image = QtGui.QImage(96, 64, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(90, 100, 110))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.resize(96, 64)
        canvas.show()
        _ensure_qapp().processEvents()
        canvas.createMode = "ai_polygon"
        canvas.current = _ActivePainterProbe(
            active_states,
            points=[QtCore.QPointF(10, 10)],
            point_labels=[1],
        )
        canvas.line = _ActivePainterProbe(
            active_states,
            points=[QtCore.QPointF(10, 10), QtCore.QPointF(40, 40)],
            point_labels=[1, 1],
        )
        monkeypatch.setattr(canvas, "_ensure_ai_model_initialized", lambda: True)

        class _FakeAiModel:
            def predict_polygon_from_points(self, points, point_labels):
                _ = point_labels
                return points + [[20.0, 40.0], [10.0, 40.0]]

        canvas._ai_model = _FakeAiModel()
        target = QtGui.QPixmap(canvas.size())
        target.fill(QtCore.Qt.transparent)
        canvas.render(target)
    finally:
        canvas.close()

    assert active_states
    assert all(active_states)


def test_ai_mask_preview_paints_with_active_painter(monkeypatch):
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    active_states = []
    canvas = Canvas()
    try:
        image = QtGui.QImage(96, 64, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(120, 130, 140))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.resize(96, 64)
        canvas.show()
        _ensure_qapp().processEvents()
        canvas.createMode = "ai_mask"
        canvas.current = _ActivePainterProbe(
            active_states,
            points=[QtCore.QPointF(20, 20)],
            point_labels=[1],
        )
        canvas.line = _ActivePainterProbe(
            active_states,
            points=[QtCore.QPointF(20, 20), QtCore.QPointF(50, 45)],
            point_labels=[1, 1],
        )
        monkeypatch.setattr(canvas, "_ensure_ai_model_initialized", lambda: True)

        class _FakeAiModel:
            def predict_mask_from_points(self, points, point_labels):
                _ = points, point_labels
                mask = np.zeros((64, 96), dtype=np.uint8)
                mask[15:48, 18:60] = 1
                return mask

        canvas._ai_model = _FakeAiModel()
        target = QtGui.QPixmap(canvas.size())
        target.fill(QtCore.Qt.transparent)
        canvas.render(target)
    finally:
        canvas.close()

    assert active_states
    assert all(active_states)


def test_playback_scroll_request_accepts_raw_orientation_int() -> None:
    from annolid.gui.mixins.playback_draw_mixin import PlaybackDrawMixin

    class _DummyBar:
        def value(self):
            return 10

        def singleStep(self):
            return 2

    class _DummyHost(PlaybackDrawMixin):
        def __init__(self):
            self.scrollBars = {
                QtCore.Qt.Horizontal: _DummyBar(),
                QtCore.Qt.Vertical: _DummyBar(),
            }
            self.calls = []

        def setScroll(self, orientation, value):
            self.calls.append((orientation, value))

    host = _DummyHost()
    host.scrollRequest(5, 0)

    assert host.calls == [(QtCore.Qt.Horizontal, 9.0)]


def test_canvas_key_release_accepts_qt_no_modifier() -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        canvas.mode = canvas.CREATE
        canvas.snapping = False
        event = SimpleNamespace(modifiers=lambda: QtCore.Qt.NoModifier)
        canvas.keyReleaseEvent(event)
        assert canvas.snapping is True
    finally:
        canvas.close()


def test_polygon_preview_line_uses_last_committed_point_not_stale_line_start() -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas
    from annolid.gui.shape import Shape

    canvas = Canvas()
    try:
        current = Shape(shape_type="polygon")
        current.points = [
            QtCore.QPointF(10, 10),
            QtCore.QPointF(20, 15),
            QtCore.QPointF(30, 40),
        ]
        current.point_labels = [1, 1, 1]
        canvas.current = current
        canvas.line.points = [QtCore.QPointF(10, 10), QtCore.QPointF(80, 70)]

        preview = canvas._build_polygon_preview_line()

        assert preview is not None
        assert preview.points[0] == QtCore.QPointF(30, 40)
        assert preview.points[1] == QtCore.QPointF(80, 70)
    finally:
        canvas.close()


def test_stale_preview_line_is_not_painted_in_edit_mode() -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    calls = []

    class _DummyPreviewLine:
        points = [QtCore.QPointF(10, 10), QtCore.QPointF(30, 30)]

        def paint(self, painter):
            calls.append(bool(painter.isActive()))

    canvas.line = _DummyPreviewLine()
    canvas.mode = canvas.EDIT

    target = QtGui.QPixmap(32, 32)
    target.fill(QtCore.Qt.transparent)
    painter = QtGui.QPainter(target)
    try:
        canvas._paint_live_preview(painter)
    finally:
        painter.end()
        canvas.close()

    assert calls == []


def test_post_status_message_is_safe_from_background_thread() -> None:
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    window = AnnolidWindow(config={})
    messages = []
    try:
        original = window.statusBar().showMessage

        def capture(message, timeout=0):
            messages.append((message, timeout))
            return original(message, timeout)

        window.statusBar().showMessage = capture  # type: ignore[method-assign]

        thread = threading.Thread(
            target=lambda: window.post_status_message("background update", 1234)
        )
        thread.start()
        thread.join(timeout=2)
        _ensure_qapp().processEvents()

        assert ("background update", 1234) in messages
    finally:
        window.close()


def test_startup_annolid_bot_respects_disable_env(monkeypatch) -> None:
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    window = AnnolidWindow(config={})
    called = {"count": 0}
    try:
        manager = getattr(window, "ai_chat_manager", None)
        assert manager is not None

        def _unexpected_start(*args, **kwargs):
            _ = args, kwargs
            called["count"] += 1

        monkeypatch.setattr(manager, "initialize_annolid_bot", _unexpected_start)
        monkeypatch.setenv("ANNOLID_DISABLE_BOT_AUTOSTART", "1")
        window._startup_annolid_bot()
        assert called["count"] == 0
    finally:
        window.close()


def test_window_close_runs_cleanup_once(monkeypatch) -> None:
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    window = AnnolidWindow(config={})
    calls = {"count": 0}
    try:
        manager = getattr(window, "ai_chat_manager", None)
        assert manager is not None

        def _cleanup_once():
            calls["count"] += 1

        monkeypatch.setattr(manager, "cleanup", _cleanup_once)
        window.close()
        _ensure_qapp().processEvents()
        window.clean_up()

        assert calls["count"] == 1
    finally:
        if window.isVisible():
            window.close()
