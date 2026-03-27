from __future__ import annotations

from qtpy.QtCore import QPointF, QRectF, Qt
from qtpy.QtGui import QColor, QPen, QPixmap
from qtpy.QtWidgets import (
    QDialog,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QPushButton,
    QVBoxLayout,
)


class CropFrameWidget(QGraphicsView):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Crop Region")
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)

        self.pixmap = QPixmap(image_path)
        self.pixmap_item = QGraphicsPixmapItem(self.pixmap)
        self.scene.addItem(self.pixmap_item)

        self.crop_rect = None
        self.start_pos = QPointF()
        self.end_pos = QPointF()

        self.rect_item = QGraphicsRectItem()
        pen = QPen(QColor(255, 0, 0), 2, Qt.DashLine)
        self.rect_item.setPen(pen)
        self.scene.addItem(self.rect_item)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = self.mapToScene(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton:
            self.end_pos = self.mapToScene(event.pos())
            rect = QRectF(self.start_pos, self.end_pos).normalized()
            self.rect_item.setRect(rect)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.end_pos = self.mapToScene(event.pos())
            self.crop_rect = self.rect_item.rect()
        super().mouseReleaseEvent(event)

    def getCropRect(self):
        return self.crop_rect


class CropDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Crop Frame")
        self.crop_widget = CropFrameWidget(image_path)
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)

        layout = QVBoxLayout()
        layout.addWidget(self.crop_widget)
        layout.addWidget(self.ok_button)
        self.setLayout(layout)

    def getCropCoordinates(self):
        rect = self.crop_widget.getCropRect()
        if rect:
            return int(rect.x()), int(rect.y()), int(rect.width()), int(rect.height())
        return None
