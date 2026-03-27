from __future__ import annotations

from qtpy.QtGui import QGuiApplication
from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QPainter, QPen, QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from annolid.gui.widgets.crop_region import CropRegion


class _CropSelectionController:
    def __init__(self) -> None:
        self._start_pos = QPointF()
        self._selection_rect = QRectF()
        self._has_selection = False

    def begin(self, scene_pos: QPointF) -> None:
        self._start_pos = QPointF(scene_pos)
        self._selection_rect = QRectF()
        self._has_selection = True

    def update(self, scene_pos: QPointF, bounds: QRectF) -> QRectF | None:
        if not self._has_selection:
            return None
        raw_rect = QRectF(self._start_pos, scene_pos).normalized()
        clipped = raw_rect.intersected(bounds)
        if clipped.isEmpty():
            self._selection_rect = QRectF()
            return None
        self._selection_rect = clipped
        return clipped

    def finish(self, scene_pos: QPointF, bounds: QRectF) -> CropRegion | None:
        rect = self.update(scene_pos, bounds)
        return CropRegion.from_qrectf(rect, bounds=bounds)

    def clear(self) -> None:
        self._start_pos = QPointF()
        self._selection_rect = QRectF()
        self._has_selection = False

    def preview_rect(self) -> QRectF:
        return QRectF(self._selection_rect)


class CropFrameWidget(QGraphicsView):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Crop Region")
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setAlignment(Qt.AlignCenter)
        self.setFrameShape(QGraphicsView.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap = QPixmap(image_path)
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.pixmap_item)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())

        self.crop_region: CropRegion | None = None
        self._selection = _CropSelectionController()
        self._auto_fit = True
        self._zoom_factor = 1.15

        self.rect_item = QGraphicsRectItem()
        pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
        self.rect_item.setPen(pen)
        self.scene.addItem(self.rect_item)
        self.rect_item.setVisible(False)

    def setCropRectFromSceneRect(self, selection_rect: QRectF | None) -> None:
        bounds = self.pixmap_item.boundingRect()
        self.crop_region = CropRegion.from_qrectf(selection_rect, bounds=bounds)
        if self.crop_region is not None:
            self._selection._selection_rect = self.crop_region.as_qrectf()
            self._selection._has_selection = True
        else:
            self._selection.clear()
        if self.crop_region is None:
            self.rect_item.setRect(QRectF())
            self.rect_item.setVisible(False)
            return
        self.rect_item.setRect(self.crop_region.as_qrectf())
        self.rect_item.setVisible(True)

    def fit_to_window(self):
        if self.pixmap_item.pixmap().isNull():
            return
        self.resetTransform()
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        self._auto_fit = True

    def clearCropRect(self):
        self.crop_region = None
        self._selection.clear()
        self.rect_item.setRect(QRectF())
        self.rect_item.setVisible(False)

    def showEvent(self, event):
        super().showEvent(event)
        if self._auto_fit:
            self.fit_to_window()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._selection.begin(self.mapToScene(event.pos()))
            self.rect_item.setVisible(True)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            bounds = self.pixmap_item.boundingRect()
            rect = self._selection.update(self.mapToScene(event.pos()), bounds)
            if rect is not None:
                self.rect_item.setRect(rect)
                self.rect_item.setVisible(True)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            bounds = self.pixmap_item.boundingRect()
            self.crop_region = self._selection.finish(
                self.mapToScene(event.pos()), bounds
            )
            if self.crop_region is None:
                self.rect_item.setRect(QRectF())
                self.rect_item.setVisible(False)
            else:
                self.rect_item.setRect(self.crop_region.as_qrectf())
                self.rect_item.setVisible(True)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if event.angleDelta().y() == 0:
            super().wheelEvent(event)
            return
        self._auto_fit = False
        factor = (
            self._zoom_factor if event.angleDelta().y() > 0 else 1.0 / self._zoom_factor
        )
        self.scale(factor, factor)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_to_window()

    def getCropRect(self):
        if self.crop_region is None:
            return None
        return self.crop_region.as_qrectf()

    def getCropRegion(self) -> CropRegion | None:
        return self.crop_region


class CropDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Frame")
        self.setModal(True)
        self.setSizeGripEnabled(True)
        self.crop_widget = CropFrameWidget(image_path, parent=self)
        self.crop_widget.setMinimumSize(720, 480)
        self.instructions_label = QLabel(
            "Drag to draw the crop region. Scroll to zoom. Use Fit to Window to reset the view."
        )
        self.instructions_label.setWordWrap(True)
        self.fit_button = QPushButton("Fit to Window")
        self.fit_button.clicked.connect(self.crop_widget.fit_to_window)
        self.clear_button = QPushButton("Clear Selection")
        self.clear_button.clicked.connect(self.crop_widget.clearCropRect)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)

        button_row = QHBoxLayout()
        button_row.addWidget(self.fit_button)
        button_row.addWidget(self.clear_button)
        button_row.addStretch(1)
        button_row.addWidget(self.cancel_button)
        button_row.addWidget(self.ok_button)

        layout = QVBoxLayout()
        layout.addWidget(self.instructions_label)
        layout.addWidget(self.crop_widget)
        layout.addLayout(button_row)
        self.setLayout(layout)
        self._resize_for_image(image_path)

    def _resize_for_image(self, image_path):
        pixmap = QPixmap(image_path)
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(1100, 800)
            return

        available = screen.availableGeometry()
        max_width = int(available.width() * 0.9)
        max_height = int(available.height() * 0.88)

        if pixmap.isNull():
            self.resize(min(1100, max_width), min(800, max_height))
            return

        chrome_width = 120
        chrome_height = 180
        target_width = min(max(pixmap.width() + chrome_width, 960), max_width)
        target_height = min(max(pixmap.height() + chrome_height, 720), max_height)
        self.resize(target_width, target_height)

    def getCropCoordinates(self):
        region = self.crop_widget.getCropRegion()
        return None if region is None else region.as_tuple()
