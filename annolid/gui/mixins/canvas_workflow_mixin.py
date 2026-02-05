from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, ImageEnhance
from qtpy import QtCore, QtGui, QtWidgets

from annolid.utils.logger import logger
from annolid.utils.qt2cv import convert_qt_image_to_rgb_cv_image


class CanvasWorkflowMixin:
    """Canvas rendering and brightness/contrast workflow mixin."""

    def image_to_canvas(self, qimage, filename, frame_number):
        self.resetState()
        self.canvas.setEnabled(True)
        self.canvas.setPatchSimilarityOverlay(None)
        self._deactivate_pca_map()
        if isinstance(filename, str):
            filename = Path(filename)
        self.imagePath = str(filename.parent)
        self.filename = str(filename)
        self.image = qimage
        self.imageData = qimage
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.canvas.loadPixmap(pixmap)
        try:
            frame_rgb = convert_qt_image_to_rgb_cv_image(qimage).copy()
        except Exception:
            frame_rgb = None
        if getattr(self, "depth_manager", None) is not None:
            self.depth_manager.update_overlay_for_frame(frame_number, frame_rgb)
        if getattr(self, "optical_flow_manager", None) is not None:
            self.optical_flow_manager.update_overlay_for_frame(frame_number, frame_rgb)
        if self._config["keep_prev"] and self.noShapes():
            self.loadShapes(prev_shapes, replace=False)
            self.setDirty()
        else:
            self.setClean()

        video_file_key_for_zoom = (
            str(self.video_file) if self.video_file else str(self.filename)
        )
        if not self._config["keep_prev_scale"]:
            if self.video_loader is not None and self.video_file:
                if self._fit_window_applied_video_key != video_file_key_for_zoom:
                    self.setFitWindow(True)
                    self.adjustScale(initial=True)
                    self._fit_window_applied_video_key = video_file_key_for_zoom
            else:
                self.adjustScale(initial=True)
        elif video_file_key_for_zoom in self.zoom_values:
            self.zoomMode = self.zoom_values[video_file_key_for_zoom][0]
            self.setZoom(self.zoom_values[video_file_key_for_zoom][1])
        else:
            self.adjustScale(initial=True)

        if video_file_key_for_zoom:
            self.zoom_values[video_file_key_for_zoom] = (
                self.zoomMode,
                self.zoomWidget.value(),
            )

        for orientation in self.scroll_values:
            if self.filename in self.scroll_values[orientation]:
                self.setScroll(
                    orientation, self.scroll_values[orientation][self.filename]
                )
        try:
            self.loadPredictShapes(frame_number, filename)
        except Exception:
            logger.debug(
                "Failed to load shapes for frame %s", frame_number, exc_info=True
            )
        self._refresh_behavior_overlay()

    @staticmethod
    def _qimage_to_bytes(qimage: QtGui.QImage, fmt: str = "PNG"):
        """Serialize a QImage into raw bytes compatible with LabelMe utilities."""
        if qimage is None or qimage.isNull():
            return None

        buffer = QtCore.QBuffer()
        if not buffer.open(QtCore.QIODevice.WriteOnly):
            logger.warning("Unable to open buffer for QImage serialization.")
            return None

        succeeded = qimage.save(buffer, fmt)
        buffer.close()

        if not succeeded:
            logger.warning("Failed to serialize QImage to %s", fmt)
            return None

        return bytes(buffer.data())

    def brightnessContrast(self, value):
        """Interactive brightness/contrast adjustment with live preview."""
        _ = value
        if self.image is None or self.image.isNull():
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Brightness/Contrast"),
                self.tr("Open an image or video frame first."),
            )
            return

        original_image = QtGui.QImage(self.image)
        original_data = self.imageData
        key = self.filename or self.imagePath or "__current__"
        init_brightness, init_contrast = self.brightnessContrast_values.get(key, (0, 0))
        init_brightness = int(init_brightness or 0)
        init_contrast = int(init_contrast or 0)

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("Brightness / Contrast"))
        layout = QtWidgets.QVBoxLayout(dialog)

        brightness_label = QtWidgets.QLabel(self.tr("Brightness: 0"), dialog)
        brightness_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, dialog)
        brightness_slider.setRange(-100, 100)
        brightness_slider.setValue(init_brightness)

        contrast_label = QtWidgets.QLabel(self.tr("Contrast: 0"), dialog)
        contrast_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, dialog)
        contrast_slider.setRange(-100, 100)
        contrast_slider.setValue(init_contrast)

        layout.addWidget(brightness_label)
        layout.addWidget(brightness_slider)
        layout.addWidget(contrast_label)
        layout.addWidget(contrast_slider)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Reset,
            QtCore.Qt.Horizontal,
            dialog,
        )
        layout.addWidget(buttons)

        def _apply_preview(brightness: int, contrast: int) -> None:
            adjusted = self._apply_brightness_contrast_qimage(
                original_image, brightness, contrast
            )
            if adjusted is None or adjusted.isNull():
                return
            self.image = adjusted
            self.imageData = adjusted
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(adjusted))
            self.paintCanvas()

        def _on_change() -> None:
            b = int(brightness_slider.value())
            c = int(contrast_slider.value())
            brightness_label.setText(self.tr("Brightness: %d") % b)
            contrast_label.setText(self.tr("Contrast: %d") % c)
            _apply_preview(b, c)

        brightness_slider.valueChanged.connect(_on_change)
        contrast_slider.valueChanged.connect(_on_change)
        _on_change()

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        reset_button = buttons.button(QtWidgets.QDialogButtonBox.Reset)
        if reset_button is not None:
            reset_button.clicked.connect(
                lambda: (brightness_slider.setValue(0), contrast_slider.setValue(0))
            )

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            b = int(brightness_slider.value())
            c = int(contrast_slider.value())
            self.brightnessContrast_values[key] = (b, c)
        else:
            self.image = original_image
            self.imageData = original_data
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(original_image))
            self.paintCanvas()

    def _apply_brightness_contrast_qimage(
        self, image: QtGui.QImage, brightness: int, contrast: int
    ) -> QtGui.QImage:
        """Apply brightness/contrast and return a new QImage."""
        image_bytes = self._qimage_to_bytes(image)
        if image_bytes is None:
            return QtGui.QImage()
        try:
            pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            b_factor = max(0.0, 1.0 + (float(brightness) / 100.0))
            c_factor = max(0.0, 1.0 + (float(contrast) / 100.0))
            if b_factor != 1.0:
                pil_img = ImageEnhance.Brightness(pil_img).enhance(b_factor)
            if c_factor != 1.0:
                pil_img = ImageEnhance.Contrast(pil_img).enhance(c_factor)
            out_buf = io.BytesIO()
            pil_img.save(out_buf, format="PNG")
            qimg = QtGui.QImage.fromData(out_buf.getvalue(), "PNG")
            return qimg
        except Exception:
            logger.debug("Failed to apply brightness/contrast.", exc_info=True)
            return QtGui.QImage()

    def adjustScale(self, initial=False):
        """Safely adjust zoom while handling cases with no active pixmap."""
        canvas_pixmap = getattr(self.canvas, "pixmap", None)
        if canvas_pixmap is None or canvas_pixmap.isNull():
            logger.debug("adjustScale skipped: canvas pixmap not ready.")
            return

        if not getattr(self, "filename", None):
            logger.debug("adjustScale skipped: no active filename.")
            return

        frame_number = getattr(self, "frame_number", None)
        filename = getattr(self, "filename", None)
        if frame_number is None or filename is None:
            logger.debug("adjustScale skipped: missing frame context.")
            return

        super().adjustScale(initial=initial)
        self.addRecentFile(self.filename)
        self.toggleActions(True)
        if self._df_deeplabcut is not None:
            self._load_deeplabcut_results(frame_number)
        return True
