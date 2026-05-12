import os
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

from qtpy import QtGui, QtWidgets  # noqa: E402


_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def test_grab_current_frame_image_supports_memoryview_qimage_bits():
    _ensure_qapp()

    from annolid.gui.controllers.dino import DinoController

    qimage = QtGui.QImage(2, 1, QtGui.QImage.Format_RGBA8888)
    qimage.setPixelColor(0, 0, QtGui.QColor(10, 20, 30, 255))
    qimage.setPixelColor(1, 0, QtGui.QColor(40, 50, 60, 128))

    controller = DinoController.__new__(DinoController)
    controller._window = SimpleNamespace(
        canvas=SimpleNamespace(pixmap=QtGui.QPixmap.fromImage(qimage))
    )

    image = controller._grab_current_frame_image()

    assert image is not None
    assert image.mode == "RGB"
    assert image.size == (2, 1)
    assert image.getpixel((0, 0)) == (10, 20, 30)
    assert image.getpixel((1, 0)) == (40, 50, 60)


def test_qimage_rgba_array_returns_none_for_invalid_image():
    from annolid.gui.controllers.dino import DinoController

    assert DinoController._qimage_rgba_array(QtGui.QImage()) is None
