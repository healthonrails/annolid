from __future__ import annotations

from typing import Optional, TYPE_CHECKING

import numpy as np
from PIL import Image, ImageDraw
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.dino_patch_service import (
    DinoPatchRequest,
    DinoPatchSimilarityService,
    DinoPCAMapService,
    DinoPCARequest,
)
from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS, PATCH_SIMILARITY_DEFAULT_MODEL
from annolid.utils.logger import logger

if TYPE_CHECKING:
    from annolid.gui.app import AnnolidWindow


class DinoController(QtCore.QObject):
    """Encapsulate patch similarity and PCA map tooling built on DINO embeddings."""

    def __init__(self, window: "AnnolidWindow") -> None:
        super().__init__(window)
        self._window = window

        self.patch_similarity_service = DinoPatchSimilarityService(window)
        self.patch_similarity_service.started.connect(
            self._on_patch_similarity_started
        )
        self.patch_similarity_service.finished.connect(
            self._on_patch_similarity_finished
        )
        self.patch_similarity_service.error.connect(
            self._on_patch_similarity_error
        )

        self.pca_map_service = DinoPCAMapService(window)
        self.pca_map_service.started.connect(self._on_pca_map_started)
        self.pca_map_service.finished.connect(self._on_pca_map_finished)
        self.pca_map_service.error.connect(self._on_pca_map_error)

    # ------------------------------------------------------------------ #
    # Initialization helpers
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        window = self._window
        settings = window.settings

        window.patch_similarity_model = str(
            settings.value("patch_similarity/model",
                           PATCH_SIMILARITY_DEFAULT_MODEL)
        )
        window.patch_similarity_alpha = float(
            settings.value("patch_similarity/alpha", 0.55)
        )
        window.patch_similarity_alpha = min(
            max(window.patch_similarity_alpha, 0.05), 1.0
        )

        window.pca_map_model = str(
            settings.value(
                "pca_map/model",
                window.patch_similarity_model or PATCH_SIMILARITY_DEFAULT_MODEL,
            )
        )
        window.pca_map_alpha = float(settings.value("pca_map/alpha", 0.65))
        window.pca_map_alpha = min(max(window.pca_map_alpha, 0.05), 1.0)
        window.pca_map_clusters = int(settings.value("pca_map/clusters", 0))
        if window.pca_map_clusters < 0:
            window.pca_map_clusters = 0

    # ------------------------------------------------------------------ #
    # Patch similarity public API
    # ------------------------------------------------------------------ #
    def toggle_patch_similarity(self, checked: Optional[bool] = None) -> None:
        window = self._window
        state = bool(checked) if isinstance(
            checked, bool) else window.patch_similarity_action.isChecked()
        if not state:
            self.deactivate_patch_similarity()
            return

        if window.canvas.pixmap is None or window.canvas.pixmap.isNull():
            QtWidgets.QMessageBox.information(
                window,
                window.tr("Patch Similarity"),
                window.tr(
                    "Load an image or video frame before starting patch similarity."),
            )
            window.patch_similarity_action.setChecked(False)
            return

        if not window.patch_similarity_model:
            self.open_patch_similarity_settings()
            if not window.patch_similarity_model:
                window.patch_similarity_action.setChecked(False)
                return

        self.deactivate_pca_map()
        window.canvas.enablePatchSimilarityMode(self.request_patch_similarity)
        window.statusBar().showMessage(
            window.tr(
                "Patch similarity active – click on the frame to query patches."),
            5000,
        )

    def deactivate_patch_similarity(self) -> None:
        window = self._window
        if hasattr(window, "patch_similarity_action"):
            window.patch_similarity_action.setChecked(False)
        if hasattr(window, "canvas") and window.canvas is not None:
            window.canvas.disablePatchSimilarityMode()
            window.canvas.setPatchSimilarityOverlay(None)

    def request_patch_similarity(self, x: int, y: int) -> None:
        window = self._window
        service = self.patch_similarity_service
        if service.is_busy():
            window.statusBar().showMessage(
                window.tr("Patch similarity is already running…"), 2000
            )
            return

        pil_image = self._grab_current_frame_image()
        if pil_image is None:
            QtWidgets.QMessageBox.warning(
                window,
                window.tr("Patch Similarity"),
                window.tr("Failed to access the current frame."),
            )
            self.deactivate_patch_similarity()
            return

        window.canvas.setPatchSimilarityOverlay(None)
        request = DinoPatchRequest(
            image=pil_image,
            click_xy=(int(x), int(y)),
            model_name=window.patch_similarity_model,
            short_side=768,
            device=None,
            alpha=float(window.patch_similarity_alpha),
        )
        if not service.request(request):
            window.statusBar().showMessage(
                window.tr("Patch similarity is already running…"), 2000
            )

    def open_patch_similarity_settings(self) -> None:
        window = self._window
        dialog = QtWidgets.QDialog(window)
        dialog.setWindowTitle(window.tr("Patch Similarity Settings"))
        layout = QtWidgets.QFormLayout(dialog)

        model_combo = QtWidgets.QComboBox(dialog)
        for cfg in PATCH_SIMILARITY_MODELS:
            model_combo.addItem(cfg.display_name, cfg.identifier)

        current_index = model_combo.findData(window.patch_similarity_model)
        if current_index >= 0:
            model_combo.setCurrentIndex(current_index)

        alpha_spin = QtWidgets.QDoubleSpinBox(dialog)
        alpha_spin.setRange(0.05, 1.0)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setValue(window.patch_similarity_alpha)

        layout.addRow(window.tr("Model"), model_combo)
        layout.addRow(window.tr("Overlay opacity"), alpha_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        layout.addRow(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            window.patch_similarity_model = model_combo.currentData()
            window.patch_similarity_alpha = alpha_spin.value()
            window.settings.setValue(
                "patch_similarity/model", window.patch_similarity_model
            )
            window.settings.setValue(
                "patch_similarity/alpha", window.patch_similarity_alpha
            )
            window.statusBar().showMessage(
                window.tr("Patch similarity model updated."),
                3000,
            )

    def open_pca_map_settings(self) -> None:
        window = self._window
        dialog = QtWidgets.QDialog(window)
        dialog.setWindowTitle(window.tr("PCA Feature Map Settings"))
        layout = QtWidgets.QFormLayout(dialog)

        model_combo = QtWidgets.QComboBox(dialog)
        for cfg in PATCH_SIMILARITY_MODELS:
            model_combo.addItem(cfg.display_name, cfg.identifier)

        current_index = model_combo.findData(window.pca_map_model)
        if current_index >= 0:
            model_combo.setCurrentIndex(current_index)

        alpha_spin = QtWidgets.QDoubleSpinBox(dialog)
        alpha_spin.setRange(0.05, 1.0)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setValue(window.pca_map_alpha)

        cluster_spin = QtWidgets.QSpinBox(dialog)
        cluster_spin.setRange(0, 12)
        cluster_spin.setValue(window.pca_map_clusters)

        layout.addRow(window.tr("Model"), model_combo)
        layout.addRow(window.tr("Overlay opacity"), alpha_spin)
        layout.addRow(window.tr("Clusters"), cluster_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        layout.addRow(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            window.pca_map_model = model_combo.currentData()
            window.pca_map_alpha = alpha_spin.value()
            window.pca_map_clusters = cluster_spin.value()
            window.settings.setValue("pca_map/model", window.pca_map_model)
            window.settings.setValue("pca_map/alpha", window.pca_map_alpha)
            window.settings.setValue(
                "pca_map/clusters", window.pca_map_clusters)
            window.statusBar().showMessage(
                window.tr("PCA settings updated."),
                3000,
            )

    # ------------------------------------------------------------------ #
    # PCA map public API
    # ------------------------------------------------------------------ #
    def toggle_pca_map(self, checked: Optional[bool] = None) -> None:
        window = self._window
        state = bool(checked) if isinstance(
            checked, bool) else window.pca_map_action.isChecked()
        if not state:
            self.deactivate_pca_map()
            return

        if window.canvas.pixmap is None or window.canvas.pixmap.isNull():
            QtWidgets.QMessageBox.information(
                window,
                window.tr("PCA Feature Map"),
                window.tr(
                    "Load an image or video frame before generating a PCA map."),
            )
            window.pca_map_action.setChecked(False)
            return

        if not window.pca_map_model:
            self.open_pca_map_settings()
            if not window.pca_map_model:
                window.pca_map_action.setChecked(False)
                return

        self.request_pca_map()

    def deactivate_pca_map(self) -> None:
        window = self._window
        if hasattr(window, "pca_map_action"):
            window.pca_map_action.setChecked(False)
        if hasattr(window, "canvas") and window.canvas is not None:
            window.canvas.setPCAMapOverlay(None)

    def request_pca_map(self) -> None:
        window = self._window
        service = self.pca_map_service
        if service.is_busy():
            window.statusBar().showMessage(
                window.tr("PCA map is already running…"), 2000
            )
            return

        window.canvas.setPCAMapOverlay(None)
        pil_image = self._grab_current_frame_image()
        if pil_image is None:
            QtWidgets.QMessageBox.warning(
                window,
                window.tr("PCA Feature Map"),
                window.tr("Failed to access the current frame."),
            )
            self.deactivate_pca_map()
            return

        device = None  # Let the service choose

        mask_bool = None
        selected_polygons = [
            shape
            for shape in getattr(window.canvas, "selectedShapes", [])
            if getattr(shape, "shape_type", "") == "polygon" and len(shape.points) >= 3
        ]
        if selected_polygons:
            mask_img = Image.new("L", pil_image.size, 0)
            draw = ImageDraw.Draw(mask_img)
            for polygon in selected_polygons:
                coords = [(float(pt.x()), float(pt.y()))
                          for pt in polygon.points]
                draw.polygon(coords, fill=255)
            mask_bool = np.array(mask_img) > 0

        cluster_k = window.pca_map_clusters if window.pca_map_clusters > 1 else None
        request = DinoPCARequest(
            image=pil_image,
            model_name=window.pca_map_model,
            short_side=768,
            device=device,
            output_size="input",
            components=3,
            clip_percentile=1.0,
            alpha=float(window.pca_map_alpha),
            mask=mask_bool,
            cluster_k=cluster_k,
        )
        if not service.request(request):
            window.statusBar().showMessage(
                window.tr("PCA map is already running…"), 2000
            )

    # ------------------------------------------------------------------ #
    # Service callbacks
    # ------------------------------------------------------------------ #
    def _on_patch_similarity_started(self):
        self._window.statusBar().showMessage(
            self._window.tr("Computing patch similarity…"))

    def _on_patch_similarity_finished(self, payload: dict) -> None:
        overlay = payload.get("overlay_rgba")
        self._window.canvas.setPatchSimilarityOverlay(overlay)
        self._window.statusBar().showMessage(
            self._window.tr("Patch similarity ready."),
            4000,
        )

    def _on_patch_similarity_error(self, message: str) -> None:
        self._window.canvas.setPatchSimilarityOverlay(None)
        QtWidgets.QMessageBox.warning(
            self._window,
            self._window.tr("Patch Similarity"),
            message,
        )
        self.deactivate_patch_similarity()

    def _on_pca_map_started(self):
        self._window.statusBar().showMessage(
            self._window.tr("Generating PCA feature map…"))

    def _on_pca_map_finished(self, payload: dict) -> None:
        overlay = payload.get("overlay_rgba")
        self._window.canvas.setPCAMapOverlay(overlay)
        self._window.statusBar().showMessage(
            self._window.tr("PCA feature map ready."),
            4000,
        )

    def _on_pca_map_error(self, message: str) -> None:
        self._window.canvas.setPCAMapOverlay(None)
        QtWidgets.QMessageBox.warning(
            self._window,
            self._window.tr("PCA Feature Map"),
            message,
        )
        self.deactivate_pca_map()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _grab_current_frame_image(self):
        window = self._window
        if window.canvas.pixmap is None or window.canvas.pixmap.isNull():
            return None
        qimage = window.canvas.pixmap.toImage().convertToFormat(
            QtGui.QImage.Format_RGBA8888)
        ptr = qimage.constBits()
        ptr.setsize(qimage.sizeInBytes())
        array = np.frombuffer(ptr, dtype=np.uint8).reshape(
            qimage.height(), qimage.width(), 4)
        try:
            return Image.fromarray(array, mode="RGBA").convert("RGB")
        except Exception as exc:
            logger.error("Failed to grab current frame image: %s", exc)
            return None
