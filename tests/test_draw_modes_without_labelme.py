import builtins
import importlib
import os

import numpy as np
from qtpy import QtWidgets, QtGui


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
