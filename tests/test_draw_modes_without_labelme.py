import builtins
import importlib
import os
import threading
from types import SimpleNamespace
from pathlib import Path

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


def test_unsupported_large_image_ai_mode_switches_to_canvas_with_message():
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        img = QtGui.QImage(120, 80, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(100, 120, 140))
        w.image = img
        w.canvas.loadPixmap(QtGui.QPixmap.fromImage(img), clear_shapes=False)
        w._active_image_view = "tiled"
        w._viewer_stack.setCurrentWidget(w.large_image_view)

        w.toggleDrawMode(False, createMode="ai_polygon")

        assert w._active_image_view == "canvas"
        assert w._viewer_stack.currentWidget() is w.canvas
        assert w.canvas.createMode == "ai_polygon"
        message = w.statusBar().currentMessage()
        assert "canvas preview mode" not in message.lower()
        assert "ai polygon tool" not in message.lower()
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
        w.canvas._ai_model_image_signature = None
        w.canvas._sync_ai_model_image(force=True)
        assert len(fake_model.images) == 1

        second = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        second.fill(QtGui.QColor(200, 100, 50))
        w.filename = "second.png"
        w.imagePath = str(Path("second.png").parent)
        w.canvas.loadPixmap(QtGui.QPixmap.fromImage(second), clear_shapes=False)
        assert len(fake_model.images) == 2
        assert not np.array_equal(fake_model.images[0], fake_model.images[1])
    finally:
        w.close()


def test_ai_model_image_does_not_refresh_for_same_pixmap_twice():
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        img = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(10, 20, 30))

        class FakeAiModel:
            name = "fake"

            def __init__(self):
                self.images = []

            def set_image(self, image):
                self.images.append(np.asarray(image).copy())

        fake_model = FakeAiModel()
        w.canvas._ai_model = fake_model
        w.canvas._ai_model_pixmap_key = None

        pixmap = QtGui.QPixmap.fromImage(img)
        w.canvas.loadPixmap(pixmap, clear_shapes=False)
        w.canvas.loadPixmap(pixmap, clear_shapes=False)

        assert len(fake_model.images) == 1
    finally:
        w.close()


def test_ai_model_image_does_not_refresh_for_same_frame_new_pixmap_instance():
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        first = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        first.fill(QtGui.QColor(10, 20, 30))
        w.image_to_canvas(first, "frame_000000123.png", 123)

        class FakeAiModel:
            name = "fake"

            def __init__(self):
                self.images = []

            def set_image(self, image):
                self.images.append(np.asarray(image).copy())

        fake_model = FakeAiModel()
        w.canvas._ai_model = fake_model
        w.canvas._ai_model_pixmap_key = None
        w.canvas._ai_model_image_signature = None
        w.canvas._sync_ai_model_image(force=True)

        second = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        second.fill(QtGui.QColor(10, 20, 30))
        w.canvas.loadPixmap(QtGui.QPixmap.fromImage(second), clear_shapes=False)

        assert len(fake_model.images) == 1
    finally:
        w.close()


def test_ai_model_image_does_not_refresh_for_same_image_without_filename():
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow

    w = AnnolidWindow(config={})
    try:
        first = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        first.fill(QtGui.QColor(10, 20, 30))

        class FakeAiModel:
            name = "fake"

            def __init__(self):
                self.images = []

            def set_image(self, image):
                self.images.append(np.asarray(image).copy())

        fake_model = FakeAiModel()
        w.canvas._ai_model = fake_model
        w.canvas._ai_model_pixmap_key = None
        w.canvas._ai_model_image_signature = None

        w.canvas.loadPixmap(QtGui.QPixmap.fromImage(first), clear_shapes=False)

        second = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        second.fill(QtGui.QColor(10, 20, 30))
        w.canvas.loadPixmap(QtGui.QPixmap.fromImage(second), clear_shapes=False)

        assert len(fake_model.images) == 1
    finally:
        w.close()


def test_load_pixmap_clears_transient_ai_prompt_state_when_frame_changes() -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        first = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        first.fill(QtGui.QColor(10, 20, 30))
        second = QtGui.QImage(64, 64, QtGui.QImage.Format_RGB32)
        second.fill(QtGui.QColor(20, 30, 40))

        canvas.loadPixmap(QtGui.QPixmap.fromImage(first), clear_shapes=False)
        canvas.current = SimpleNamespace()
        canvas.sam_mask.logits = np.ones((4, 4), dtype=np.float32)
        canvas.sam_image_scale = 0.5

        canvas.loadPixmap(QtGui.QPixmap.fromImage(second), clear_shapes=False)

        assert canvas.current is None
        assert canvas.sam_mask.logits is None
        assert not hasattr(canvas, "sam_image_scale")
    finally:
        canvas.close()


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


def test_initialize_ai_model_reuses_embedding_for_same_model_and_frame(
    monkeypatch,
) -> None:
    _ensure_qapp()

    from annolid.gui.app import AnnolidWindow
    import annolid.gui.widgets.canvas as canvas_mod

    class ReusableAiModel:
        name = "ReusableAiModel"

        def set_image(self, image):
            _ = image

    monkeypatch.setattr(canvas_mod, "AI_MODELS", [ReusableAiModel])

    w = AnnolidWindow(config={})
    try:
        img = QtGui.QImage(96, 64, QtGui.QImage.Format_RGB32)
        img.fill(QtGui.QColor(80, 90, 100))
        w.image_to_canvas(img, "reuse_test.png", 0)

        w.canvas.initializeAiModel("ReusableAiModel")
        assert isinstance(w.canvas._ai_model, ReusableAiModel)

        sync_forces = []

        def _capture_sync(*, force=False):
            sync_forces.append(bool(force))
            return True

        monkeypatch.setattr(w.canvas, "_sync_ai_model_image", _capture_sync)
        w.canvas.initializeAiModel("ReusableAiModel")

        assert sync_forces == [False]
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
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

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
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

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


def test_canvas_context_menu_ai_polygon_uses_local_action_and_resets_prompt_state():
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        shared_ai_action = QtWidgets.QAction("AI Polygon", canvas)
        shared_ai_action.setEnabled(True)

        class _WindowStub:
            def __init__(self):
                self.actions = SimpleNamespace(createAiPolygonMode=shared_ai_action)
                self.calls = []

            def toggleDrawMode(self, edit=True, createMode="polygon"):
                self.calls.append(
                    (
                        bool(edit),
                        createMode,
                        canvas.current is None,
                        canvas.sam_mask.logits is None,
                    )
                )

        canvas.current = SimpleNamespace(points=[QtCore.QPointF(10, 10)])
        canvas.sam_mask.logits = np.ones((2, 2), dtype=np.float32)
        window = _WindowStub()

        menu = canvas._build_context_menu(window)
        ai_action = next(
            action for action in menu.actions() if action.text() == "AI Polygon"
        )

        assert ai_action is not shared_ai_action

        ai_action.trigger()

        assert window.calls == [(False, "ai_polygon", True, True)]
    finally:
        canvas.close()


def test_toggle_draw_mode_initializes_ai_model_once_for_ai_modes() -> None:
    from annolid.gui.mixins.playback_draw_mixin import PlaybackDrawMixin

    class _DummyAction:
        def __init__(self):
            self.enabled = []
            self.checked = False
            self._signals_blocked = False

        def setEnabled(self, value):
            self.enabled.append(bool(value))

        def isCheckable(self):
            return True

        def isChecked(self):
            return self.checked

        def setChecked(self, value):
            self.checked = bool(value)

        def blockSignals(self, value):
            previous = self._signals_blocked
            self._signals_blocked = bool(value)
            return previous

    class _DummyCanvas:
        def __init__(self):
            self.calls = []
            self.editing = []
            self.createMode = None
            self.cancel_calls = []

        def setEditing(self, value):
            self.editing.append(bool(value))

        def cancelCurrentDrawing(self, *, clear_sam_mask=False):
            self.cancel_calls.append(bool(clear_sam_mask))

        def initializeAiModel(self, name, _custom_ai_models=None):
            self.calls.append((name, _custom_ai_models))

    class _DummyHost(PlaybackDrawMixin):
        def __init__(self):
            self.actions = SimpleNamespace(
                editMode=_DummyAction(),
                createAiPolygonMode=_DummyAction(),
            )
            self.canvas = _DummyCanvas()
            self._selectAiModelComboBox = SimpleNamespace(
                currentText=lambda: "EfficientSam (speed)"
            )
            self.ai_model_manager = SimpleNamespace(
                custom_model_names={"custom": "model"}
            )
            self._active_image_view = "canvas"
            self.large_image_view = None

        def tr(self, text):
            return text

    host = _DummyHost()
    host.toggleDrawMode(False, createMode="ai_polygon")

    assert host.canvas.calls == [("EfficientSam (speed)", {"custom": "model"})]
    assert host.canvas.cancel_calls == [True]


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


def test_ai_polygon_finalise_clamps_points_to_image_bounds(monkeypatch):
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(100, 80, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.createMode = "ai_polygon"
        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(20, 20), label=1)
        shape.addPoint(QtCore.QPointF(30, 25), label=1)
        canvas.current = shape
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        class _FakeAiModel:
            def predict_mask_from_points(self, points, point_labels):
                _ = points, point_labels
                mask = np.zeros((80, 100), dtype=np.uint8)
                mask[3:76, 2:98] = 1
                return mask

        canvas._ai_model = _FakeAiModel()
        canvas.finalise()
        assert len(canvas.shapes) == 1
        polygon = canvas.shapes[0]
        assert len(polygon.points) >= 3
        assert all(0.0 <= point.x() <= 99.0 for point in polygon.points)
        assert all(0.0 <= point.y() <= 79.0 for point in polygon.points)
    finally:
        canvas.close()


def test_polygon_double_click_finalise_trims_tail_point() -> None:
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(120, 90, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.mode = canvas.CREATE
        canvas.createMode = "polygon"
        shape = Shape(shape_type="polygon")
        shape.addPoint(QtCore.QPointF(10, 10))
        shape.addPoint(QtCore.QPointF(40, 10))
        shape.addPoint(QtCore.QPointF(40, 35))
        # Simulate the redundant trailing click point added just before double-click.
        shape.addPoint(QtCore.QPointF(30, 25))
        canvas.current = shape

        canvas.mouseDoubleClickEvent(None)

        assert canvas.current is None
        assert len(canvas.shapes) == 1
        assert len(canvas.shapes[0].points) == 3
    finally:
        canvas.close()


def test_ai_polygon_double_click_does_not_finalise_when_not_closeable(monkeypatch):
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(100, 80, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.mode = canvas.CREATE
        canvas.createMode = "ai_polygon"
        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(20, 20), label=1)
        shape.addPoint(QtCore.QPointF(30, 25), label=1)
        canvas.current = shape
        called = {"count": 0}
        monkeypatch.setattr(
            canvas,
            "finalise",
            lambda: called.__setitem__("count", called["count"] + 1),
        )

        canvas.mouseDoubleClickEvent(None)

        assert called["count"] == 0
    finally:
        canvas.close()


def test_ai_polygon_finalise_inference_error_keeps_editing(monkeypatch):
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(100, 80, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.mode = canvas.CREATE
        canvas.createMode = "ai_polygon"
        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(10, 10), label=1)
        shape.addPoint(QtCore.QPointF(25, 20), label=1)
        shape.addPoint(QtCore.QPointF(35, 40), label=1)
        canvas.current = shape
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        class _FailingModel:
            def predict_polygon_from_points(self, points, point_labels):
                _ = points, point_labels
                raise RuntimeError("model failed")

        canvas._ai_model = _FailingModel()

        canvas.finalise()

        assert canvas.current is shape
        assert canvas.current.shape_type == "points"
        assert canvas.shapes == []
    finally:
        canvas.close()


def test_shape_pop_point_keeps_point_labels_aligned() -> None:
    from annolid.gui.shape import Shape

    shape = Shape(shape_type="points")
    shape.addPoint(QtCore.QPointF(10, 10), label=1)
    shape.addPoint(QtCore.QPointF(20, 20), label=0)
    shape.addPoint(QtCore.QPointF(30, 30), label=1)

    popped = shape.popPoint()

    assert popped == QtCore.QPointF(30, 30)
    assert len(shape.points) == 2
    assert len(shape.point_labels) == 2
    assert shape.point_labels == [1, 0]


def test_ai_polygon_double_click_trim_keeps_prompt_labels_aligned() -> None:
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas(epsilon=5.0)
    try:
        image = QtGui.QImage(100, 80, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.mode = canvas.CREATE
        canvas.createMode = "ai_polygon"
        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(10, 10), label=1)
        shape.addPoint(QtCore.QPointF(20, 20), label=1)
        # Near-duplicate tail point (trim target).
        shape.addPoint(QtCore.QPointF(20.2, 20.1), label=1)
        canvas.current = shape

        canvas._trim_double_click_tail_point()

        assert len(canvas.current.points) == 2
        assert len(canvas.current.point_labels) == 2
    finally:
        canvas.close()


def test_polygon_from_prompt_mask_uses_largest_refined_mask_component() -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        mask = np.zeros((120, 120), dtype=np.uint8)
        mask[10:35, 10:35] = 1
        mask[55:110, 60:115] = 1

        points = canvas._polygon_from_prompt_mask(
            mask,
            prompt_points=[[90, 90], [30, 30]],
            point_labels=[1, 0],
        )

        assert len(points) >= 3
        xs = [p.x() for p in points]
        ys = [p.y() for p in points]
        assert min(xs) >= 55
        assert min(ys) >= 55
    finally:
        canvas.close()


def test_ai_polygon_finalise_uses_largest_refined_mask_component(monkeypatch) -> None:
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(140, 120, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.mode = canvas.CREATE
        canvas.createMode = "ai_polygon"
        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(100, 95), label=1)
        shape.addPoint(QtCore.QPointF(20, 20), label=0)
        canvas.current = shape
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        class _FakeAiModel:
            def predict_mask_from_points(self, points, point_labels):
                _ = points, point_labels
                mask = np.zeros((120, 140), dtype=np.uint8)
                mask[5:40, 5:35] = 1
                mask[55:115, 70:135] = 1
                return mask

        canvas._ai_model = _FakeAiModel()
        canvas.finalise()

        assert len(canvas.shapes) == 1
        polygon = canvas.shapes[0]
        xs = [p.x() for p in polygon.points]
        ys = [p.y() for p in polygon.points]
        assert min(xs) >= 65
        assert min(ys) >= 50
    finally:
        canvas.close()


def test_ai_polygon_finalise_prefers_sam_mask_over_polygon_output(monkeypatch) -> None:
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(140, 120, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.mode = canvas.CREATE
        canvas.createMode = "ai_polygon"
        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(15, 15), label=1)
        shape.addPoint(QtCore.QPointF(110, 20), label=1)
        shape.addPoint(QtCore.QPointF(115, 90), label=1)
        shape.addPoint(QtCore.QPointF(20, 100), label=1)
        canvas.current = shape
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        calls = []

        class _FakeAiModel:
            def predict_polygon_from_points(self, points, point_labels):
                calls.append("polygon")
                _ = points, point_labels
                return np.array(
                    [
                        [5.0, 5.0],
                        [135.0, 5.0],
                        [135.0, 115.0],
                        [5.0, 115.0],
                    ],
                    dtype=np.float32,
                )

            def predict_mask_from_points(self, points, point_labels):
                calls.append("mask")
                _ = points, point_labels
                mask = np.zeros((120, 140), dtype=np.uint8)
                mask[20:100, 25:120] = 1
                return mask

        canvas._ai_model = _FakeAiModel()
        canvas.finalise()

        assert calls == ["mask"]
        assert len(canvas.shapes) == 1
        polygon = canvas.shapes[0]
        assert polygon.shape_type == "polygon"
        xs = [point.x() for point in polygon.points]
        ys = [point.y() for point in polygon.points]
        assert min(xs) >= 20
        assert max(xs) <= 121
        assert min(ys) >= 15
        assert max(ys) <= 101

    finally:
        canvas.close()


def test_ai_polygon_does_not_fallback_to_polygon_when_mask_predictor_errors(
    monkeypatch,
) -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(140, 120, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        calls = []

        class _FailingMaskModel:
            def predict_mask_from_points(self, points, point_labels):
                calls.append("mask")
                _ = points, point_labels
                raise RuntimeError("temporary predictor failure")

            def predict_polygon_from_points(self, points, point_labels):
                calls.append("polygon")
                _ = points, point_labels
                return np.array(
                    [
                        [-100.0, -100.0],
                        [280.0, -100.0],
                        [280.0, 220.0],
                        [-100.0, 220.0],
                    ],
                    dtype=np.float32,
                )

        canvas._ai_model = _FailingMaskModel()
        points = canvas._predict_ai_polygon_points(
            prompt_points=[[75.0, 65.0], [85.0, 78.0]],
            point_labels=[1, 1],
        )

        assert calls == ["mask"]
        assert points == []
    finally:
        canvas.close()


def test_ai_polygon_refines_polygon_as_more_prompts_are_added(monkeypatch) -> None:
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(140, 120, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        class _RefiningMaskModel:
            def predict_polygon_from_points(self, points, point_labels):
                raise AssertionError("mask path should drive AI polygon generation")

            def predict_mask_from_points(self, points, point_labels):
                _ = point_labels
                mask = np.zeros((120, 140), dtype=np.uint8)
                if len(points) == 1:
                    mask[50:80, 60:90] = 1
                elif len(points) == 2:
                    mask[45:90, 50:105] = 1
                else:
                    mask[40:95, 45:110] = 1
                return mask

        canvas._ai_model = _RefiningMaskModel()

        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(75, 65), label=1)
        canvas.current = shape

        polygon1 = canvas._predict_ai_polygon_points(
            prompt_points=[[75.0, 65.0]],
            point_labels=[1],
        )
        shape.addPoint(QtCore.QPointF(85, 78), label=1)
        polygon2 = canvas._predict_ai_polygon_points(
            prompt_points=[[75.0, 65.0], [85.0, 78.0]],
            point_labels=[1, 1],
        )
        shape.addPoint(QtCore.QPointF(55, 55), label=1)
        polygon3 = canvas._predict_ai_polygon_points(
            prompt_points=[[75.0, 65.0], [85.0, 78.0], [55.0, 55.0]],
            point_labels=[1, 1, 1],
        )

        def _bbox(points):
            xs = [point.x() for point in points]
            ys = [point.y() for point in points]
            return min(xs), min(ys), max(xs), max(ys)

        bbox1 = _bbox(polygon1)
        bbox2 = _bbox(polygon2)
        bbox3 = _bbox(polygon3)

        assert len(polygon1) >= 3
        assert len(polygon2) >= 3
        assert len(polygon3) >= 3
        assert bbox2[0] <= bbox1[0]
        assert bbox2[1] <= bbox1[1]
        assert bbox2[2] >= bbox1[2]
        assert bbox2[3] >= bbox1[3]
        assert bbox3[0] <= bbox2[0]
        assert bbox3[1] <= bbox2[1]
        assert bbox3[2] >= bbox2[2]
        assert bbox3[3] >= bbox2[3]
    finally:
        canvas.close()


def test_ai_polygon_ignores_full_width_strip_mask_artifacts(monkeypatch) -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(420, 220, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        class _StripArtifactMaskModel:
            def predict_mask_from_points(self, points, point_labels):
                _ = points, point_labels
                mask = np.zeros((220, 420), dtype=np.uint8)
                # Artifact component: nearly full-width thin strip.
                mask[150:163, 0:420] = 1
                # Real object near the prompt point.
                mask[120:175, 270:330] = 1
                return mask

        canvas._ai_model = _StripArtifactMaskModel()
        polygon = canvas._predict_ai_polygon_points(
            prompt_points=[[300.0, 145.0]],
            point_labels=[1],
        )

        assert len(polygon) >= 3
        xs = [point.x() for point in polygon]
        ys = [point.y() for point in polygon]
        assert min(xs) >= 260
        assert max(xs) <= 340
        assert min(ys) >= 110
        assert max(ys) <= 185
    finally:
        canvas.close()


def test_ai_polygon_retries_with_normalized_prompt_coordinates(monkeypatch) -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(420, 220, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        class _NormalizedPromptMaskModel:
            """Model that expects points normalized to [0, 1]."""

            def predict_mask_from_points(self, points, point_labels):
                _ = point_labels
                arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
                mask = np.zeros((220, 420), dtype=np.uint8)
                if arr.size == 0:
                    return mask
                # If pixel-space points are passed by mistake, clip pushes to right edge.
                px = int(round(float(np.clip(arr[0, 0], 0.0, 1.0)) * 419.0))
                py = int(round(float(np.clip(arr[0, 1], 0.0, 1.0)) * 219.0))
                x1 = max(0, px - 18)
                x2 = min(420, px + 18)
                y1 = max(0, py - 14)
                y2 = min(220, py + 14)
                mask[y1:y2, x1:x2] = 1
                return mask

        canvas._ai_model = _NormalizedPromptMaskModel()
        polygon = canvas._predict_ai_polygon_points(
            prompt_points=[[96.0, 118.0]],
            point_labels=[1],
        )

        assert len(polygon) >= 3
        xs = [point.x() for point in polygon]
        ys = [point.y() for point in polygon]
        # Should stay close to prompt location, not drift to right edge.
        assert max(xs) < 180
        assert min(xs) > 40
        assert max(ys) < 170
        assert min(ys) > 70
    finally:
        canvas.close()


def test_ai_polygon_finalise_materializes_stable_polygon_shape(monkeypatch) -> None:
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(140, 120, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.mode = canvas.CREATE
        canvas.createMode = "ai_polygon"
        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(75, 65), label=1)
        shape.addPoint(QtCore.QPointF(85, 78), label=1)
        shape.other_data["source"] = "ai"
        canvas.current = shape
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        class _StableMaskModel:
            def predict_mask_from_points(self, points, point_labels):
                _ = points, point_labels
                mask = np.zeros((120, 140), dtype=np.uint8)
                mask[45:90, 50:105] = 1
                return mask

        canvas._ai_model = _StableMaskModel()
        canvas.finalise()

        assert len(canvas.shapes) == 1
        polygon = canvas.shapes[0]
        assert polygon.shape_type == "polygon"
        assert polygon.isClosed() is True
        assert polygon._shape_raw is None
        assert len(polygon.points) == len(polygon.point_labels)
        assert set(polygon.point_labels) == {1}
        assert polygon.other_data["source"] == "ai"
    finally:
        canvas.close()


def test_ai_polygon_modifier_key_after_finalise_does_not_mutate_shape(
    monkeypatch,
) -> None:
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(140, 120, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.mode = canvas.CREATE
        canvas.createMode = "ai_polygon"
        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(75, 65), label=1)
        shape.addPoint(QtCore.QPointF(85, 78), label=1)
        canvas.current = shape
        monkeypatch.setattr(
            canvas, "_ensure_ai_model_initialized", lambda **kwargs: True
        )

        class _StableMaskModel:
            def predict_mask_from_points(self, points, point_labels):
                _ = points, point_labels
                mask = np.zeros((120, 140), dtype=np.uint8)
                mask[45:90, 50:105] = 1
                return mask

        canvas._ai_model = _StableMaskModel()
        canvas.finalise()
        polygon = canvas.shapes[0]
        before = [(point.x(), point.y()) for point in polygon.points]

        press = QtGui.QKeyEvent(
            QtCore.QEvent.KeyPress,
            QtCore.Qt.Key_Meta,
            QtCore.Qt.MetaModifier,
        )
        release = QtGui.QKeyEvent(
            QtCore.QEvent.KeyRelease,
            QtCore.Qt.Key_Meta,
            QtCore.Qt.NoModifier,
        )
        canvas.keyPressEvent(press)
        canvas.keyReleaseEvent(release)

        after = [(point.x(), point.y()) for point in polygon.points]
        assert after == before
    finally:
        canvas.close()


def test_ai_polygon_finalise_does_not_force_resync_when_pixmap_unchanged(
    monkeypatch,
) -> None:
    _ensure_qapp()

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    canvas = Canvas()
    try:
        image = QtGui.QImage(140, 120, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(20, 30, 40))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas.mode = canvas.CREATE
        canvas.createMode = "ai_polygon"
        shape = Shape(shape_type="points")
        shape.addPoint(QtCore.QPointF(75, 65), label=1)
        shape.addPoint(QtCore.QPointF(85, 78), label=1)
        canvas.current = shape

        class _StableMaskModel:
            def predict_mask_from_points(self, points, point_labels):
                _ = points, point_labels
                mask = np.zeros((120, 140), dtype=np.uint8)
                mask[45:90, 50:105] = 1
                return mask

        canvas._ai_model = _StableMaskModel()
        canvas._ai_model_pixmap_key = int(canvas.pixmap.cacheKey())
        sync_forces = []

        def _fake_sync(*, force=False):
            sync_forces.append(bool(force))
            return True

        monkeypatch.setattr(canvas, "_sync_ai_model_image", _fake_sync)

        canvas.finalise()

        assert sync_forces
        assert all(force is False for force in sync_forces)
    finally:
        canvas.close()


def test_sync_ai_model_image_skips_duplicate_embed_for_same_pixmap_key(
    monkeypatch,
) -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    class _TrackingAiModel:
        def __init__(self):
            self.calls = 0

        def set_image(self, image):
            _ = image
            self.calls += 1

    canvas = Canvas()
    try:
        image = QtGui.QImage(64, 48, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(10, 20, 30))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas._ai_model = _TrackingAiModel()
        monkeypatch.setattr(
            canvas, "_ai_model_image_signature_value", lambda: ("frame", 1)
        )

        assert canvas._sync_ai_model_image(force=False) is True
        assert canvas._sync_ai_model_image(force=False) is True
        assert canvas._ai_model.calls == 1
    finally:
        canvas.close()


def test_sync_ai_model_image_refreshes_when_signature_changes_even_if_pixmap_key_matches(
    monkeypatch,
) -> None:
    _ensure_qapp()

    from annolid.gui.widgets.canvas import Canvas

    class _TrackingAiModel:
        def __init__(self):
            self.calls = 0

        def set_image(self, image):
            _ = image
            self.calls += 1

    canvas = Canvas()
    try:
        image = QtGui.QImage(64, 48, QtGui.QImage.Format_RGB32)
        image.fill(QtGui.QColor(10, 20, 30))
        canvas.loadPixmap(QtGui.QPixmap.fromImage(image))
        canvas._ai_model = _TrackingAiModel()

        signature_values = iter([("frame", 1), ("frame", 2)])
        monkeypatch.setattr(
            canvas,
            "_ai_model_image_signature_value",
            lambda: next(signature_values),
        )

        assert canvas._sync_ai_model_image(force=False) is True
        # Simulate a reused pixmap key path while the actual frame identity changed.
        canvas._ai_model_pixmap_key = int(canvas.pixmap.cacheKey())
        assert canvas._sync_ai_model_image(force=False) is True
        assert canvas._ai_model.calls == 2
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
