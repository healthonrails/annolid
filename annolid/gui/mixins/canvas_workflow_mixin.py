from __future__ import annotations

import json
import hashlib
from pathlib import Path
import time

from qtpy import QtCore, QtGui, QtWidgets

from annolid.utils.image_adjustments import (
    apply_brightness_contrast_uint8,
    normalize_brightness_contrast_value,
)
from annolid.utils.logger import logger
from annolid.utils.qt2cv import (
    convert_cv_image_to_qt_image,
    convert_qt_image_to_rgb_cv_image,
)


class CanvasWorkflowMixin:
    """Canvas rendering and brightness/contrast workflow mixin."""

    _FIRST_FRAME_ENRICHMENT_DELAY_MS = 150

    def image_to_canvas(self, qimage, filename, frame_number):
        render_started_ts = time.perf_counter()
        self.resetState()
        self.canvas.setEnabled(True)
        self._active_image_view = "canvas"
        if getattr(self, "_viewer_stack", None) is not None:
            self._viewer_stack.setCurrentWidget(self.canvas)
        self.canvas.setPatchSimilarityOverlay(None)
        self._deactivate_pca_map()
        if isinstance(filename, str):
            filename = Path(filename)
        self.imagePath = str(filename.parent)
        self.filename = str(filename)
        self.image = qimage
        self.imageData = qimage
        sync_frame_jump = getattr(self, "_sync_frame_jump_input", None)
        if callable(sync_frame_jump):
            sync_frame_jump(int(frame_number))
        if self._config["keep_prev"]:
            prev_shapes = self.canvas.shapes
        pixmap = QtGui.QPixmap.fromImage(qimage)
        self.canvas.loadPixmap(pixmap)
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

        should_defer_enrichment = bool(
            getattr(self, "_defer_first_frame_enrichment", False)
        ) and bool(getattr(self, "video_file", None))
        if should_defer_enrichment:
            setattr(self, "_defer_first_frame_enrichment", False)
            self._run_fast_annotation_enrichment(
                frame_number=int(frame_number),
                filename=filename,
            )
            self._schedule_frame_enrichment(
                frame_number=int(frame_number),
                filename=filename,
                qimage=qimage,
                delay_ms=int(self._FIRST_FRAME_ENRICHMENT_DELAY_MS),
            )
            logger.info(
                "Frame %s painted in %.1fms; deferred enrichment scheduled (%dms).",
                frame_number,
                (time.perf_counter() - render_started_ts) * 1000.0,
                int(self._FIRST_FRAME_ENRICHMENT_DELAY_MS),
            )
            return
        self._run_frame_enrichment(
            frame_number=int(frame_number),
            filename=filename,
            qimage=qimage,
        )

    def _run_fast_annotation_enrichment(
        self,
        *,
        frame_number: int,
        filename: Path,
    ) -> None:
        try:
            self.loadPredictShapes(frame_number, filename)
        except Exception:
            logger.debug(
                "Failed to load shapes for frame %s", frame_number, exc_info=True
            )

    def _schedule_frame_enrichment(
        self,
        *,
        frame_number: int,
        filename: Path,
        qimage: QtGui.QImage,
        delay_ms: int,
    ) -> None:
        token = f"{frame_number}|{filename}|{time.time_ns()}"
        setattr(self, "_frame_enrichment_token", token)
        QtCore.QTimer.singleShot(
            max(0, int(delay_ms)),
            lambda: self._run_scheduled_frame_enrichment(
                token=token,
                frame_number=frame_number,
                filename=filename,
                qimage=qimage,
            ),
        )

    def _run_scheduled_frame_enrichment(
        self,
        *,
        token: str,
        frame_number: int,
        filename: Path,
        qimage: QtGui.QImage,
    ) -> None:
        active_token = str(getattr(self, "_frame_enrichment_token", "") or "")
        if active_token != str(token):
            return
        current_frame = getattr(self, "frame_number", None)
        try:
            current_frame_int = int(current_frame) if current_frame is not None else -1
        except Exception:
            current_frame_int = -1
        if current_frame_int != int(frame_number):
            return
        if str(getattr(self, "filename", "") or "") != str(filename):
            return
        self._run_visual_overlay_enrichment(
            frame_number=int(frame_number),
            qimage=qimage,
        )

    def _run_frame_enrichment(
        self,
        *,
        frame_number: int,
        filename: Path,
        qimage: QtGui.QImage,
    ) -> None:
        enrich_started_ts = time.perf_counter()
        self._run_fast_annotation_enrichment(
            frame_number=int(frame_number),
            filename=filename,
        )
        self._run_visual_overlay_enrichment(
            frame_number=int(frame_number),
            qimage=qimage,
        )
        logger.debug(
            "Frame %s enrichment finished in %.1fms.",
            frame_number,
            (time.perf_counter() - enrich_started_ts) * 1000.0,
        )

    def _run_visual_overlay_enrichment(
        self,
        *,
        frame_number: int,
        qimage: QtGui.QImage,
    ) -> None:
        depth_manager = getattr(self, "depth_manager", None)
        optical_flow_manager = getattr(self, "optical_flow_manager", None)
        needs_depth_overlay = False
        needs_flow_overlay = False
        if depth_manager is not None:
            has_depth_overlay = getattr(depth_manager, "has_overlay_for_frame", None)
            needs_depth_overlay = (
                bool(has_depth_overlay(frame_number))
                if callable(has_depth_overlay)
                else True
            )
        if optical_flow_manager is not None:
            has_flow_overlay = getattr(
                optical_flow_manager, "has_overlay_for_frame", None
            )
            needs_flow_overlay = (
                bool(has_flow_overlay(frame_number))
                if callable(has_flow_overlay)
                else True
            )

        if getattr(self, "large_image_view", None) is not None:
            self.large_image_view.clear()
        frame_rgb = None
        if needs_depth_overlay or needs_flow_overlay:
            try:
                frame_rgb = convert_qt_image_to_rgb_cv_image(qimage).copy()
            except Exception:
                frame_rgb = None
        if needs_depth_overlay and depth_manager is not None:
            depth_manager.update_overlay_for_frame(frame_number, frame_rgb)
        if needs_flow_overlay and optical_flow_manager is not None:
            optical_flow_manager.update_overlay_for_frame(frame_number, frame_rgb)
        self._refresh_behavior_overlay(frame_number=int(frame_number))

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

    @staticmethod
    def _sanitize_brightness_contrast_value(value: int | float | None) -> int:
        return normalize_brightness_contrast_value(value)

    def _video_brightness_contrast_key(
        self, video_path: str | None = None
    ) -> str | None:
        candidate = str(video_path or getattr(self, "video_file", "") or "").strip()
        if not candidate:
            return None
        try:
            return str(Path(candidate).expanduser().resolve())
        except Exception:
            return candidate

    @staticmethod
    def _video_brightness_contrast_settings_key(video_key: str) -> str:
        digest = hashlib.sha1(video_key.encode("utf-8")).hexdigest()
        return f"video/brightness_contrast/{digest}"

    def get_video_brightness_contrast_values(
        self, video_path: str | None = None
    ) -> tuple[int, int]:
        video_key = self._video_brightness_contrast_key(video_path)
        if not video_key:
            return (0, 0)

        cache = getattr(self, "video_brightness_contrast_values", None)
        if isinstance(cache, dict) and video_key in cache:
            cached_b, cached_c = cache.get(video_key, (0, 0))
            return (
                self._sanitize_brightness_contrast_value(cached_b),
                self._sanitize_brightness_contrast_value(cached_c),
            )

        settings = getattr(self, "settings", None)
        payload = None
        if settings is not None and hasattr(settings, "value"):
            try:
                payload = settings.value(
                    self._video_brightness_contrast_settings_key(video_key),
                    "",
                    type=str,
                )
            except Exception:
                payload = None

        brightness = 0
        contrast = 0
        if payload:
            try:
                data = json.loads(str(payload))
                if isinstance(data, dict):
                    brightness = data.get("brightness", 0)
                    contrast = data.get("contrast", 0)
                elif isinstance(data, (list, tuple)) and len(data) >= 2:
                    brightness = data[0]
                    contrast = data[1]
            except Exception:
                brightness = 0
                contrast = 0

        brightness = self._sanitize_brightness_contrast_value(brightness)
        contrast = self._sanitize_brightness_contrast_value(contrast)
        if isinstance(cache, dict):
            cache[video_key] = (brightness, contrast)
        return (brightness, contrast)

    def set_video_brightness_contrast_values(
        self, brightness: int, contrast: int, video_path: str | None = None
    ) -> None:
        video_key = self._video_brightness_contrast_key(video_path)
        if not video_key:
            return

        brightness = self._sanitize_brightness_contrast_value(brightness)
        contrast = self._sanitize_brightness_contrast_value(contrast)
        cache = getattr(self, "video_brightness_contrast_values", None)
        if isinstance(cache, dict):
            cache[video_key] = (brightness, contrast)

        settings = getattr(self, "settings", None)
        if settings is not None and hasattr(settings, "setValue"):
            try:
                settings.setValue(
                    self._video_brightness_contrast_settings_key(video_key),
                    json.dumps(
                        {"brightness": int(brightness), "contrast": int(contrast)},
                        separators=(",", ":"),
                    ),
                )
            except Exception:
                logger.debug(
                    "Failed to persist video brightness/contrast for %s",
                    video_key,
                    exc_info=True,
                )

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
        frame_key = self.filename or self.imagePath or "__current__"
        video_key = self._video_brightness_contrast_key()
        video_brightness, video_contrast = self.get_video_brightness_contrast_values(
            video_key
        )
        init_brightness, init_contrast = self.brightnessContrast_values.get(
            frame_key,
            (video_brightness, video_contrast),
        )
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
            self.brightnessContrast_values[frame_key] = (b, c)
            if video_key:
                self.set_video_brightness_contrast_values(
                    brightness=b,
                    contrast=c,
                    video_path=video_key,
                )
        else:
            self.image = original_image
            self.imageData = original_data
            self.canvas.loadPixmap(QtGui.QPixmap.fromImage(original_image))
            self.paintCanvas()

    def _apply_brightness_contrast_qimage(
        self, image: QtGui.QImage, brightness: int, contrast: int
    ) -> QtGui.QImage:
        """Apply brightness/contrast and return a new QImage."""
        if image is None or image.isNull():
            return QtGui.QImage()
        try:
            rgb_image = convert_qt_image_to_rgb_cv_image(image)
            adjusted = apply_brightness_contrast_uint8(
                rgb_image,
                brightness=brightness,
                contrast=contrast,
            )
            if adjusted is None:
                return QtGui.QImage()
            return convert_cv_image_to_qt_image(adjusted)
        except Exception:
            logger.debug("Failed to apply brightness/contrast.", exc_info=True)
            return QtGui.QImage()

    def adjustScale(self, initial=False):
        """Safely adjust zoom while handling cases with no active pixmap."""
        if getattr(self, "_active_image_view", "canvas") == "tiled":
            if getattr(self, "large_image_view", None) is None:
                logger.debug("adjustScale skipped: tiled view is not available.")
                return False
            if getattr(self, "large_image_backend", None) is None:
                logger.debug("adjustScale skipped: tiled backend is not ready.")
                return False
            super().adjustScale(initial=initial)
            return True
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
