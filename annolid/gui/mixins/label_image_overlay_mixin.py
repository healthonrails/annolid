from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets

from annolid.gui.label_image_overlay import label_entry_text, load_label_mapping_csv
from annolid.gui.status import post_window_status
from annolid.gui.large_image import open_large_image
from annolid.gui.viewer_layers import RasterLabelLayer, raster_label_layer_from_state


class LabelImageOverlayMixin:
    """Large-image label TIFF overlay workflow inspired by napari labels."""

    def _postLabelImageStatus(self, message: str, timeout: int = 4000) -> None:
        post_window_status(self, message, timeout)

    def currentLabelImageLayer(self) -> RasterLabelLayer | None:
        large_view = getattr(self, "large_image_view", None)
        if large_view is None or not hasattr(large_view, "label_overlay_state"):
            return None
        state = dict(large_view.label_overlay_state() or {})
        mapping = {}
        if hasattr(large_view, "_label_mapping"):
            mapping = dict(getattr(large_view, "_label_mapping", {}) or {})
        return raster_label_layer_from_state(state, mapping_table=mapping)

    def describeLabelImageOverlayValue(self, label_value: int) -> str:
        large_view = getattr(self, "large_image_view", None)
        mapping = {}
        if large_view is not None and hasattr(large_view, "_label_mapping"):
            mapping = dict(getattr(large_view, "_label_mapping", {}) or {})
        return self.tr("Region %s") % label_entry_text(int(label_value), mapping)

    def importLabelImageOverlay(self, _value: bool = False) -> None:
        image_path = str(getattr(self, "imagePath", "") or "")
        large_view = getattr(self, "large_image_view", None)
        if (
            not image_path
            or large_view is None
            or getattr(self, "_active_image_view", "") != "tiled"
        ):
            self.errorMessage(
                self.tr("Label Overlay"),
                self.tr(
                    "Open a large TIFF image before importing a label image overlay."
                ),
            )
            return
        start_dir = str(Path(image_path).parent)
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Import Label Image Overlay"),
            start_dir,
            self.tr(
                "Label TIFF files (*.tif *.tiff *.ome.tif *.ome.tiff);;All files (*)"
            ),
        )
        if not filename:
            return
        try:
            backend = open_large_image(filename)
        except Exception as exc:
            self.errorMessage(
                self.tr("Label Overlay"),
                self.tr("Failed to open label image overlay: %s") % str(exc),
            )
            return
        selected_page = self._resolve_initial_label_overlay_page(backend)
        try:
            backend.set_page(int(selected_page))
        except Exception:
            selected_page = int(getattr(backend, "get_current_page", lambda: 0)() or 0)

        base_backend = getattr(self, "large_image_backend", None)
        if base_backend is not None:
            try:
                base_shape = tuple(base_backend.get_level_shape(0))
                label_shape = tuple(backend.get_level_shape(0))
            except Exception:
                base_shape = ()
                label_shape = ()
            if base_shape and label_shape and base_shape != label_shape:
                QtWidgets.QMessageBox.information(
                    self,
                    self.tr("Label Overlay"),
                    self.tr(
                        "The label image size (%d × %d) does not exactly match the base image size (%d × %d). "
                        "Annolid will still load it as a best-effort overlay."
                    )
                    % (
                        int(label_shape[0]),
                        int(label_shape[1]),
                        int(base_shape[0]),
                        int(base_shape[1]),
                    ),
                )
        record = dict(
            (getattr(self, "otherData", {}) or {}).get("label_image_overlay") or {}
        )
        opacity = float(record.get("opacity", 0.45) or 0.45)
        visible = bool(record.get("visible", True))
        selected_label = record.get("selected_label")
        transform = self._default_label_overlay_transform(backend)
        mapping = {}
        mapping_path = str(record.get("mapping_path", "") or "")
        if mapping_path:
            try:
                mapping = load_label_mapping_csv(mapping_path)
            except Exception:
                mapping = {}
        large_view.set_label_layer(
            backend,
            opacity=opacity,
            mapping=mapping,
            source_path=filename,
            mapping_path=mapping_path,
            visible=visible,
            page_index=selected_page,
            transform=transform,
        )
        large_view.set_selected_label_value(selected_label)
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        self.otherData["label_image_overlay"] = {
            "source_path": str(filename),
            "mapping_path": mapping_path,
            "opacity": opacity,
            "visible": visible,
            "selected_label": selected_label,
            "page_index": int(selected_page),
            "transform": dict(transform),
        }
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        self._postLabelImageStatus(
            self.tr("Loaded label image overlay from %s (page %d)")
            % (Path(filename).name, int(selected_page) + 1)
        )
        self._syncLabelImageOverlayActionState()

    def importLabelImageMapping(self, _value: bool = False) -> None:
        large_view = getattr(self, "large_image_view", None)
        if (
            large_view is None
            or getattr(large_view, "label_layer_backend", lambda: None)() is None
        ):
            self.errorMessage(
                self.tr("Label Mapping"),
                self.tr(
                    "Load a label image overlay before importing a label mapping file."
                ),
            )
            return
        source_path = str(getattr(self, "imagePath", "") or "")
        start_dir = str(Path(source_path).parent) if source_path else str(Path.home())
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Import Label Mapping"),
            start_dir,
            self.tr("CSV/TSV files (*.csv *.tsv *.txt);;All files (*)"),
        )
        if not filename:
            return
        try:
            mapping = load_label_mapping_csv(filename)
        except Exception as exc:
            self.errorMessage(
                self.tr("Label Mapping"),
                self.tr("Failed to load label mapping: %s") % str(exc),
            )
            return
        large_view.set_label_mapping(mapping, mapping_path=filename)
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        record = dict(self.otherData.get("label_image_overlay") or {})
        record["mapping_path"] = str(filename)
        record["selected_label"] = large_view.selected_label_value()
        record["opacity"] = float(record.get("opacity", 0.45) or 0.45)
        record["visible"] = bool(record.get("visible", True))
        record["source_path"] = str(record.get("source_path", "") or "")
        record["page_index"] = int(
            large_view.label_overlay_state().get("page_index", 0) or 0
        )
        record["transform"] = dict(
            large_view.label_overlay_state().get("transform") or {}
        )
        self.otherData["label_image_overlay"] = record
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        self._postLabelImageStatus(
            self.tr("Loaded %d label ids from %s") % (len(mapping), Path(filename).name)
        )
        self._syncLabelImageOverlayActionState()

    def setLabelImageOverlayOpacity(self, _value: bool = False) -> None:
        large_view = getattr(self, "large_image_view", None)
        if (
            large_view is None
            or getattr(large_view, "label_layer_backend", lambda: None)() is None
        ):
            self.errorMessage(
                self.tr("Label Overlay"),
                self.tr("Load a label image overlay before changing its opacity."),
            )
            return
        current = int(
            round(float(large_view.label_overlay_state().get("opacity", 0.45)) * 100.0)
        )
        value, ok = QtWidgets.QInputDialog.getInt(
            self,
            self.tr("Label Overlay Opacity"),
            self.tr("Opacity percent:"),
            current,
            0,
            100,
            5,
        )
        if not ok:
            return
        opacity = float(value) / 100.0
        large_view.set_label_layer_opacity(opacity)
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        record = dict(self.otherData.get("label_image_overlay") or {})
        record["opacity"] = opacity
        record["visible"] = bool(record.get("visible", True))
        record["selected_label"] = large_view.selected_label_value()
        record["mapping_path"] = str(record.get("mapping_path", "") or "")
        record["source_path"] = str(record.get("source_path", "") or "")
        record["page_index"] = int(
            large_view.label_overlay_state().get("page_index", 0) or 0
        )
        record["transform"] = dict(
            large_view.label_overlay_state().get("transform") or {}
        )
        self.otherData["label_image_overlay"] = record
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        self._postLabelImageStatus(
            self.tr("Updated label overlay opacity to %d%%") % int(value)
        )
        self._syncLabelImageOverlayActionState()

    def setLabelImageOverlayVisible(self, visible: bool) -> bool:
        large_view = getattr(self, "large_image_view", None)
        if (
            large_view is None
            or getattr(large_view, "label_layer_backend", lambda: None)() is None
        ):
            return False
        visible_flag = bool(visible)
        large_view.set_label_layer_visible(visible_flag)
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        record = dict(self.otherData.get("label_image_overlay") or {})
        state = dict(large_view.label_overlay_state() or {})
        record["visible"] = visible_flag
        record["selected_label"] = large_view.selected_label_value()
        record["opacity"] = float(state.get("opacity", 0.45) or 0.45)
        record["mapping_path"] = str(state.get("mapping_path", "") or "")
        record["source_path"] = str(state.get("source_path", "") or "")
        record["page_index"] = int(state.get("page_index", 0) or 0)
        record["transform"] = dict(state.get("transform") or {})
        self.otherData["label_image_overlay"] = record
        self._syncLabelImageOverlayActionState()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        self._postLabelImageStatus(
            self.tr("Label image overlay %s")
            % (self.tr("shown") if visible_flag else self.tr("hidden"))
        )
        return True

    def toggleLabelImageOverlayVisible(self, value: bool = False) -> None:
        if not self.setLabelImageOverlayVisible(bool(value)):
            self._syncLabelImageOverlayActionState()

    def clearLabelImageOverlay(self, _value: bool = False) -> None:
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None:
            large_view.clear_label_layer()
        if isinstance(getattr(self, "otherData", None), dict):
            self.otherData.pop("label_image_overlay", None)
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        self._postLabelImageStatus(self.tr("Cleared label image overlay"))
        self._syncLabelImageOverlayActionState()

    def _restoreLabelImageOverlayFromState(self) -> None:
        large_view = getattr(self, "large_image_view", None)
        if large_view is None or getattr(self, "_active_image_view", "") != "tiled":
            if large_view is not None:
                large_view.clear_label_layer()
            return
        record = dict(
            (getattr(self, "otherData", {}) or {}).get("label_image_overlay") or {}
        )
        source_path = str(record.get("source_path", "") or "")
        if not source_path:
            large_view.clear_label_layer()
            return
        path = Path(source_path).expanduser()
        if not path.exists():
            large_view.clear_label_layer()
            self._postLabelImageStatus(
                self.tr("Label overlay file is missing: %s") % str(path.name),
                2500,
            )
            return
        try:
            backend = open_large_image(path)
        except Exception:
            large_view.clear_label_layer()
            return
        base_backend = getattr(self, "large_image_backend", None)
        selected_page = int(record.get("page_index", 0) or 0)
        if base_backend is not None:
            try:
                page_count = int(
                    getattr(base_backend, "get_page_count", lambda: 1)() or 1
                )
                label_page_count = int(
                    getattr(backend, "get_page_count", lambda: 1)() or 1
                )
                page_index = int(
                    getattr(base_backend, "get_current_page", lambda: 0)() or 0
                )
                if label_page_count == page_count and page_count > 1:
                    selected_page = page_index
                    backend.set_page(page_index)
                else:
                    backend.set_page(selected_page)
            except Exception:
                pass
        mapping = {}
        mapping_path = str(record.get("mapping_path", "") or "")
        if mapping_path:
            try:
                mapping = load_label_mapping_csv(mapping_path)
            except Exception:
                mapping = {}
        large_view.set_label_layer(
            backend,
            opacity=float(record.get("opacity", 0.45) or 0.45),
            mapping=mapping,
            source_path=str(path),
            mapping_path=mapping_path,
            visible=bool(record.get("visible", True)),
            page_index=selected_page,
            transform=dict(record.get("transform") or {}),
        )
        large_view.set_selected_label_value(record.get("selected_label"))
        self._syncLabelImageOverlayActionState()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()

    def _syncLabelImageOverlayActionState(self) -> None:
        actions = getattr(self, "actions", None)
        action = getattr(actions, "toggle_label_image_overlay_visible", None)
        if action is None:
            return
        layer = self.currentLabelImageLayer()
        has_overlay = layer is not None
        visible = bool(layer.visible) if layer is not None else False
        if isinstance(action, QtCore.QObject):
            with QtCore.QSignalBlocker(action):
                action.setEnabled(bool(has_overlay))
                action.setChecked(bool(visible))
            return
        action.setEnabled(bool(has_overlay))
        action.setChecked(bool(visible))

    def _default_label_overlay_transform(self, label_backend) -> dict[str, float]:
        large_view = getattr(self, "large_image_view", None)
        if large_view is None:
            return {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0}
        base_w, base_h = tuple(
            getattr(large_view, "content_size", lambda: (0, 0))() or (0, 0)
        )
        try:
            label_w, label_h = tuple(label_backend.get_level_shape(0))
        except Exception:
            label_w, label_h = (0, 0)
        if base_w <= 0 or base_h <= 0 or label_w <= 0 or label_h <= 0:
            return {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0}
        width_ratio = max(base_w, label_w) / max(1.0, min(base_w, label_w))
        height_ratio = max(base_h, label_h) / max(1.0, min(base_h, label_h))
        if width_ratio <= 1.5 and height_ratio <= 1.5:
            return {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0}
        scale = min((base_w * 0.9) / float(label_w), (base_h * 0.9) / float(label_h))
        scaled_w = float(label_w) * scale
        scaled_h = float(label_h) * scale
        tx = (float(base_w) - scaled_w) / 2.0
        ty = (float(base_h) - scaled_h) / 2.0
        return {"tx": tx, "ty": ty, "sx": scale, "sy": scale}

    def _resolve_initial_label_overlay_page(self, label_backend) -> int:
        page_count = max(
            1, int(getattr(label_backend, "get_page_count", lambda: 1)() or 1)
        )
        if page_count <= 1:
            return 0
        base_backend = getattr(self, "large_image_backend", None)
        if base_backend is not None:
            try:
                base_page_count = int(
                    getattr(base_backend, "get_page_count", lambda: 1)() or 1
                )
                base_page_index = int(
                    getattr(base_backend, "get_current_page", lambda: 0)() or 0
                )
                if base_page_count == page_count:
                    return min(max(base_page_index, 0), page_count - 1)
            except Exception:
                pass
        best_page = 0
        best_score = -1
        current_page = int(getattr(label_backend, "get_current_page", lambda: 0)() or 0)
        try:
            label_w, label_h = tuple(label_backend.get_level_shape(0))
            sample_w = min(int(label_w), 512)
            sample_h = min(int(label_h), 512)
            for page_idx in range(page_count):
                try:
                    label_backend.set_page(page_idx)
                    sample = label_backend.read_region(
                        0, 0, sample_w, sample_h, level=0
                    )
                    score = int((sample != 0).sum())
                except Exception:
                    score = -1
                if score > best_score:
                    best_page = page_idx
                    best_score = score
        finally:
            try:
                label_backend.set_page(current_page)
            except Exception:
                pass
        return int(best_page)
