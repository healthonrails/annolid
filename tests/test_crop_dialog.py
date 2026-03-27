from __future__ import annotations

from pathlib import Path

from qtpy import QtWidgets
from qtpy.QtCore import QRectF
from qtpy.QtGui import QColor, QImage

from annolid.gui.widgets.crop_dialog import CropFrameWidget, CropDialog


def _ensure_qapp():
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_test_image(path: Path, width: int = 200, height: int = 120) -> None:
    image = QImage(width, height, QImage.Format_RGB32)
    image.fill(QColor("white"))
    assert image.save(str(path))


def test_crop_rect_is_clipped_to_input_image_and_ignores_zoom(tmp_path: Path) -> None:
    _ensure_qapp()
    image_path = tmp_path / "frame.png"
    _make_test_image(image_path)

    widget = CropFrameWidget(str(image_path))
    widget.scale(2.0, 2.0)
    widget.setCropRectFromSceneRect(QRectF(10, 15, 70, 40))

    rect = widget.getCropRect()
    assert rect is not None
    assert rect.x() == 10
    assert rect.y() == 15
    assert rect.width() == 70
    assert rect.height() == 40

    widget.scale(0.5, 0.5)
    widget.setCropRectFromSceneRect(QRectF(-10, -5, 50, 30))

    rect = widget.getCropRect()
    assert rect is not None
    assert rect.x() == 0
    assert rect.y() == 0
    assert rect.width() == 40
    assert rect.height() == 25


def test_crop_dialog_returns_image_space_coordinates(tmp_path: Path) -> None:
    _ensure_qapp()
    image_path = tmp_path / "frame.png"
    _make_test_image(image_path)

    dialog = CropDialog(str(image_path))
    dialog.crop_widget.setCropRectFromSceneRect(QRectF(5, 6, 33, 44))

    assert dialog.getCropCoordinates() == (5, 6, 33, 44)
