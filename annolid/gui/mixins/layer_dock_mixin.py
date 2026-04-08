from __future__ import annotations

from qtpy import QtCore

from annolid.gui.widgets.layer_dock import ViewerLayerDockWidget
from annolid.gui.viewer_layers import (
    AnnotationLayer,
    LandmarkLayer,
    RasterImageLayer,
    RasterLabelLayer,
    VectorOverlayLayer,
)


class LayerDockMixin:
    def _is_raster_overlay_layer_id(self, layer_id: str) -> bool:
        target = str(layer_id or "")
        if not target or target == "raster_image":
            return False
        for layer in list(getattr(self, "viewerLayerModels", lambda: [])() or []):
            if (
                isinstance(layer, RasterImageLayer)
                and str(getattr(layer, "id", "") or "") == target
                and target != "raster_image"
            ):
                return True
        return False

    def _ensureViewerLayerDock(self) -> ViewerLayerDockWidget:
        dock = getattr(self, "viewer_layer_dock", None)
        if isinstance(dock, ViewerLayerDockWidget):
            return dock
        dock = ViewerLayerDockWidget(self)
        dock.layerVisibilityChanged.connect(self._onViewerLayerVisibilityChanged)
        dock.layerOpacityChanged.connect(self._onViewerLayerOpacityChanged)
        dock.layerSelected.connect(self._onViewerLayerSelected)
        dock.layerMoveRequested.connect(self._onViewerLayerMoveRequested)
        dock.layerRenameRequested.connect(self._onViewerLayerRenameRequested)
        dock.layerRemoveRequested.connect(self._onViewerLayerRemoveRequested)
        dock.layerMoveToTopRequested.connect(self._onViewerLayerMoveToTopRequested)
        dock.layerMoveToBottomRequested.connect(
            self._onViewerLayerMoveToBottomRequested
        )
        self.viewer_layer_dock = dock
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        vector_dock = getattr(self, "vector_overlay_dock", None)
        if vector_dock is not None:
            try:
                self.tabifyDockWidget(dock, vector_dock)
            except Exception:
                pass
        return dock

    def _viewerLayerEntries(self) -> list[dict]:
        entries = []
        for layer in list(getattr(self, "viewerLayerModels", lambda: [])() or []):
            details = []
            is_base_raster = (
                isinstance(layer, RasterImageLayer) and str(layer.id) == "raster_image"
            )
            is_raster_overlay = (
                isinstance(layer, RasterImageLayer) and not is_base_raster
            )
            supports_opacity = (
                isinstance(layer, (RasterLabelLayer, VectorOverlayLayer))
                or is_raster_overlay
            )
            checkable = bool(
                is_base_raster
                or not isinstance(layer, RasterImageLayer)
                or is_raster_overlay
            )
            if isinstance(layer, RasterImageLayer):
                if str(layer.id) == "raster_image":
                    details.append("Base raster image")
                else:
                    details.append("Raster overlay image")
                details.append(f"page {int(layer.backend_page_index) + 1}")
            elif isinstance(layer, RasterLabelLayer):
                details.append("Label overlay")
                details.append(f"page {int(layer.page_index) + 1}")
            elif isinstance(layer, VectorOverlayLayer):
                details.append(f"{int(layer.shape_count)} shapes")
                details.append(f"source: {str(layer.source_kind or 'svg')}")
            elif isinstance(layer, LandmarkLayer):
                details.append(f"{len(list(layer.pairs or []))} landmark pairs")
            elif isinstance(layer, AnnotationLayer):
                details.append(f"{len(list(layer.shapes or []))} annotations")
            entries.append(
                {
                    "id": str(layer.id),
                    "name": str(layer.name),
                    "visible": bool(getattr(layer, "visible", True)),
                    "opacity": float(getattr(layer, "opacity", 1.0)),
                    "supports_opacity": supports_opacity,
                    "supports_reorder": is_raster_overlay,
                    "checkable": checkable,
                    "details": " | ".join(details),
                }
            )
        return entries

    def _refreshViewerLayerDock(self) -> None:
        dock = self._ensureViewerLayerDock()
        entries = self._viewerLayerEntries()
        dock.set_layers(entries)
        dock.setVisible(bool(entries))

    def _setAnnotationLayerVisible(self, visible: bool) -> bool:
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return False
        changed = False
        for shape in list(getattr(canvas, "shapes", []) or []):
            other = dict(getattr(shape, "other_data", {}) or {})
            if "overlay_id" in other:
                continue
            if bool(getattr(shape, "visible", True)) == bool(visible):
                continue
            if hasattr(canvas, "setShapeVisible"):
                canvas.setShapeVisible(shape, bool(visible))
            else:
                shape.visible = bool(visible)
            changed = True
        if not changed:
            return False
        try:
            canvas.update()
        except Exception:
            pass
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None:
            large_view.set_shapes(getattr(canvas, "shapes", []))
        if hasattr(self, "setDirty"):
            self.setDirty()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def _onViewerLayerVisibilityChanged(self, layer_id: str, visible: bool) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        changed = False
        if layer_id == "label_image_overlay" and hasattr(
            self, "setLabelImageOverlayVisible"
        ):
            changed = bool(self.setLabelImageOverlayVisible(bool(visible)))
        elif layer_id == "raster_image" and hasattr(self, "setBaseRasterImageVisible"):
            changed = bool(self.setBaseRasterImageVisible(bool(visible)))
        elif self._is_raster_overlay_layer_id(layer_id) and hasattr(
            self, "setRasterImageLayerVisible"
        ):
            changed = bool(self.setRasterImageLayerVisible(layer_id, bool(visible)))
        elif layer_id == "annotations":
            changed = self._setAnnotationLayerVisible(bool(visible))
        elif layer_id.endswith("_landmarks") and hasattr(
            self, "setVectorOverlayLandmarkLayerVisible"
        ):
            overlay_id = layer_id[: -len("_landmarks")]
            changed = bool(
                self.setVectorOverlayLandmarkLayerVisible(overlay_id, bool(visible))
            )
        elif hasattr(self, "setVectorOverlayTransform"):
            changed = bool(
                self.setVectorOverlayTransform(str(layer_id), visible=bool(visible))
            )
        if changed:
            self._refreshViewerLayerDock()

    def _setLabelImageOverlayOpacity(self, opacity: float) -> bool:
        large_view = getattr(self, "large_image_view", None)
        if (
            large_view is None
            or getattr(large_view, "label_layer_backend", lambda: None)() is None
        ):
            return False
        large_view.set_label_layer_opacity(float(opacity))
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        state = dict(large_view.label_overlay_state() or {})
        record = dict(self.otherData.get("label_image_overlay") or {})
        record["opacity"] = float(state.get("opacity", opacity) or opacity)
        record["visible"] = bool(state.get("visible", True))
        record["selected_label"] = state.get("selected_label")
        record["mapping_path"] = str(state.get("mapping_path", "") or "")
        record["source_path"] = str(state.get("source_path", "") or "")
        record["page_index"] = int(state.get("page_index", 0) or 0)
        record["transform"] = dict(state.get("transform") or {})
        self.otherData["label_image_overlay"] = record
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def _onViewerLayerOpacityChanged(self, layer_id: str, opacity: float) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        changed = False
        if layer_id == "label_image_overlay":
            changed = self._setLabelImageOverlayOpacity(float(opacity))
        elif self._is_raster_overlay_layer_id(layer_id) and hasattr(
            self, "setRasterImageLayerOpacity"
        ):
            changed = bool(self.setRasterImageLayerOpacity(layer_id, float(opacity)))
        elif hasattr(self, "setVectorOverlayTransform"):
            changed = bool(
                self.setVectorOverlayTransform(str(layer_id), opacity=float(opacity))
            )
        if changed:
            self._refreshViewerLayerDock()

    def _onViewerLayerMoveRequested(self, layer_id: str, direction: int) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        changed = False
        if self._is_raster_overlay_layer_id(layer_id) and hasattr(
            self, "moveRasterImageLayer"
        ):
            changed = bool(self.moveRasterImageLayer(layer_id, int(direction)))
        if changed:
            self._refreshViewerLayerDock()

    def _onViewerLayerRenameRequested(self, layer_id: str, new_name: str) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        changed = False
        if self._is_raster_overlay_layer_id(layer_id) and hasattr(
            self, "renameRasterImageLayer"
        ):
            changed = bool(self.renameRasterImageLayer(layer_id, str(new_name or "")))
        if changed:
            self._refreshViewerLayerDock()

    def _onViewerLayerRemoveRequested(self, layer_id: str) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        changed = False
        if self._is_raster_overlay_layer_id(layer_id) and hasattr(
            self, "removeRasterImageLayer"
        ):
            changed = bool(self.removeRasterImageLayer(layer_id))
        if changed:
            self._refreshViewerLayerDock()

    def _onViewerLayerMoveToTopRequested(self, layer_id: str) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        changed = False
        if self._is_raster_overlay_layer_id(layer_id) and hasattr(
            self, "moveRasterImageLayerToEdge"
        ):
            changed = bool(self.moveRasterImageLayerToEdge(layer_id, "top"))
        if changed:
            self._refreshViewerLayerDock()

    def _onViewerLayerMoveToBottomRequested(self, layer_id: str) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        changed = False
        if self._is_raster_overlay_layer_id(layer_id) and hasattr(
            self, "moveRasterImageLayerToEdge"
        ):
            changed = bool(self.moveRasterImageLayerToEdge(layer_id, "bottom"))
        if changed:
            self._refreshViewerLayerDock()

    def _onViewerLayerSelected(self, layer_id: str) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        vector_dock = getattr(self, "vector_overlay_dock", None)
        if layer_id.endswith("_landmarks"):
            overlay_id = layer_id[: -len("_landmarks")]
            if vector_dock is not None and hasattr(vector_dock, "set_current_overlay"):
                vector_dock.set_current_overlay(overlay_id)
            if hasattr(self, "_setSelectedVectorOverlayLandmarkPair"):
                self._setSelectedVectorOverlayLandmarkPair(None)
            return
        overlays = {
            str(layer.id)
            for layer in list(getattr(self, "vectorOverlayLayers", lambda: [])() or [])
        }
        if (
            layer_id in overlays
            and vector_dock is not None
            and hasattr(vector_dock, "set_current_overlay")
        ):
            vector_dock.set_current_overlay(layer_id)
