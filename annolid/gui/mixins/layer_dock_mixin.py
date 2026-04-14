from __future__ import annotations

from pathlib import Path

from qtpy import QtCore
from qtpy import QtGui

from annolid.gui.widgets.layer_dock import ViewerLayerDockWidget
from annolid.gui.viewer_layers import (
    AnnotationLayer,
    LandmarkLayer,
    RasterImageLayer,
    RasterLabelLayer,
    VectorOverlayLayer,
)


class LayerDockMixin:
    def _currentLayerSettingsFilePath(self) -> str:
        candidates: list[str] = []
        label_file = getattr(self, "labelFile", None)
        for raw in (
            getattr(label_file, "filename", ""),
            getattr(self, "filename", ""),
        ):
            value = str(raw or "").strip()
            if value.lower().endswith(".json"):
                candidates.append(value)
        image_path = str(getattr(self, "imagePath", "") or "").strip()
        if image_path:
            try:
                candidates.append(
                    str(Path(image_path).expanduser().with_suffix(".json"))
                )
            except Exception:
                pass
        for candidate in candidates:
            path = Path(candidate).expanduser()
            if path.exists():
                return str(path)
        return str(candidates[0]) if candidates else ""

    def _rasterOverlaySettingsPath(self, record: dict | None) -> str:
        settings_path = str((record or {}).get("settings_path") or "").strip()
        if settings_path:
            return settings_path
        current_settings_path = self._currentLayerSettingsFilePath()
        if current_settings_path:
            return current_settings_path
        return str((record or {}).get("source_path") or "").strip()

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
        dock.layerTranslateRequested.connect(self._onViewerLayerTranslateRequested)
        dock.layerResetTransformRequested.connect(
            self._onViewerLayerResetTransformRequested
        )
        dock.layerMoveRequested.connect(self._onViewerLayerMoveRequested)
        dock.layerRenameRequested.connect(self._onViewerLayerRenameRequested)
        dock.layerRemoveRequested.connect(self._onViewerLayerRemoveRequested)
        dock.layerMoveToTopRequested.connect(self._onViewerLayerMoveToTopRequested)
        dock.layerMoveToBottomRequested.connect(
            self._onViewerLayerMoveToBottomRequested
        )
        dock.layerOpenSourceRequested.connect(self._onViewerLayerOpenSourceRequested)
        dock.layerOpenSourceFolderRequested.connect(
            self._onViewerLayerOpenSourceFolderRequested
        )
        dock.layerApplySettingsRequested.connect(
            self._onViewerLayerApplySettingsRequested
        )
        dock.layerSaveSettingsRequested.connect(
            self._onViewerLayerSaveSettingsRequested
        )
        dock.layerQuickTransformRequested.connect(
            self._onViewerLayerQuickTransformRequested
        )
        dock.layerInteractiveResizeToggled.connect(
            self._onViewerLayerInteractiveResizeToggled
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
            page_index = 0
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
            supports_translate = bool(is_raster_overlay)
            checkable = bool(
                is_base_raster
                or not isinstance(layer, RasterImageLayer)
                or is_raster_overlay
            )
            if isinstance(layer, RasterImageLayer):
                source_path = ""
                page_index = int(layer.backend_page_index)
                for record in list(
                    getattr(self, "_raster_layer_records", lambda: [])() or []
                ):
                    if str((record or {}).get("id") or "") != str(layer.id):
                        continue
                    source_path = self._rasterOverlaySettingsPath(record)
                    page_index = int(
                        (record or {}).get("page_index", page_index) or page_index
                    )
                    break
                if str(layer.id) == "raster_image":
                    details.append("Base raster image")
                else:
                    details.append("Raster overlay image")
                details.append(f"page {int(page_index) + 1}")
                transform = getattr(layer, "transform", None)
                if transform is not None:
                    try:
                        tx = float(getattr(transform, "tx", 0.0) or 0.0)
                        ty = float(getattr(transform, "ty", 0.0) or 0.0)
                        sx = float(getattr(transform, "sx", 1.0) or 1.0)
                        sy = float(getattr(transform, "sy", 1.0) or 1.0)
                        if (
                            abs(tx) > 1e-6
                            or abs(ty) > 1e-6
                            or abs(sx - 1.0) > 1e-6
                            or abs(sy - 1.0) > 1e-6
                        ):
                            details.append(
                                f"offset=({tx:.1f}, {ty:.1f}) scale=({sx:.3f}, {sy:.3f})"
                            )
                    except Exception:
                        pass
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
                    "supports_translate": supports_translate,
                    "supports_settings": is_raster_overlay,
                    "supports_reorder": is_raster_overlay,
                    "checkable": checkable,
                    "source_path": source_path if is_raster_overlay else "",
                    "page_index": int(page_index),
                    "tx": float(
                        getattr(getattr(layer, "transform", None), "tx", 0.0) or 0.0
                    ),
                    "ty": float(
                        getattr(getattr(layer, "transform", None), "ty", 0.0) or 0.0
                    ),
                    "sx": float(
                        getattr(getattr(layer, "transform", None), "sx", 1.0) or 1.0
                    ),
                    "sy": float(
                        getattr(getattr(layer, "transform", None), "sy", 1.0) or 1.0
                    ),
                    "details": " | ".join(details),
                }
            )
        return entries

    def _rasterOverlayLayerRecord(self, layer_id: str) -> dict | None:
        target = str(layer_id or "")
        if not target or not hasattr(self, "_raster_layer_records"):
            return None
        for record in list(self._raster_layer_records() or []):
            if str((record or {}).get("id") or "") == target:
                return dict(record or {})
        return None

    def _refreshViewerLayerDock(self) -> None:
        dock = self._ensureViewerLayerDock()
        entries = self._viewerLayerEntries()
        dock.set_layers(entries)
        dock.setVisible(bool(entries))

    def selectedViewerLayerId(self) -> str | None:
        layer_id = str(getattr(self, "_selected_viewer_layer_id", "") or "")
        return layer_id or None

    def selectedRasterOverlayLayerId(self) -> str | None:
        layer_id = self.selectedViewerLayerId()
        if not layer_id:
            return None
        if self._is_raster_overlay_layer_id(layer_id):
            return layer_id
        return None

    def _setAnnotationLayerVisible(self, visible: bool) -> bool:
        canvas = getattr(self, "canvas", None)
        if canvas is None:
            return False
        visible_flag = bool(visible)
        annotation_shapes = [
            shape
            for shape in list(getattr(canvas, "shapes", []) or [])
            if "overlay_id" not in dict(getattr(shape, "other_data", {}) or {})
        ]
        changed = False
        for shape in annotation_shapes:
            if bool(getattr(shape, "visible", True)) == visible_flag:
                continue
            if hasattr(canvas, "setShapeVisible"):
                canvas.setShapeVisible(shape, visible_flag)
            else:
                shape.visible = visible_flag
            changed = True
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None:
            for shape in annotation_shapes:
                if hasattr(large_view, "setShapeVisible"):
                    try:
                        large_view.setShapeVisible(shape, visible_flag)
                    except Exception:
                        pass
        if not changed:
            return False
        try:
            canvas.update()
        except Exception:
            pass
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

    def _onViewerLayerTranslateRequested(
        self, layer_id: str, dx: float, dy: float
    ) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        changed = False
        if self._is_raster_overlay_layer_id(layer_id) and hasattr(
            self, "translateRasterImageLayer"
        ):
            changed = bool(
                self.translateRasterImageLayer(layer_id, float(dx), float(dy))
            )
        if changed:
            self._refreshViewerLayerDock()

    def _onViewerLayerResetTransformRequested(self, layer_id: str) -> None:
        layer_id = str(layer_id or "")
        if not layer_id:
            return
        changed = False
        if self._is_raster_overlay_layer_id(layer_id) and hasattr(
            self, "setRasterImageLayerTransform"
        ):
            changed = bool(
                self.setRasterImageLayerTransform(
                    layer_id, tx=0.0, ty=0.0, sx=1.0, sy=1.0
                )
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
            self._selected_viewer_layer_id = None
            return
        self._selected_viewer_layer_id = layer_id
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

    def _onViewerLayerOpenSourceRequested(self, layer_id: str) -> None:
        layer_id = str(layer_id or "")
        if not self._is_raster_overlay_layer_id(layer_id):
            return
        record = self._rasterOverlayLayerRecord(layer_id)
        source_path = Path(self._rasterOverlaySettingsPath(record)).expanduser()
        if not source_path.exists():
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(source_path)))

    def _onViewerLayerOpenSourceFolderRequested(self, layer_id: str) -> None:
        layer_id = str(layer_id or "")
        if not self._is_raster_overlay_layer_id(layer_id):
            return
        record = self._rasterOverlayLayerRecord(layer_id)
        source_path = Path(self._rasterOverlaySettingsPath(record)).expanduser()
        if not source_path.exists():
            return
        QtGui.QDesktopServices.openUrl(
            QtCore.QUrl.fromLocalFile(str(source_path.parent))
        )

    def _applyViewerLayerSettings(self, layer_id: str, settings: dict) -> bool:
        layer_id = str(layer_id or "")
        if not self._is_raster_overlay_layer_id(layer_id):
            return False
        payload = dict(settings or {})
        changed = False
        if hasattr(self, "renameRasterImageLayer"):
            changed = (
                bool(
                    self.renameRasterImageLayer(
                        layer_id, str(payload.get("name") or "")
                    )
                )
                or changed
            )
        if hasattr(self, "setRasterImageLayerPageIndex"):
            changed = (
                bool(
                    self.setRasterImageLayerPageIndex(
                        layer_id,
                        int(payload.get("page_index", 0) or 0),
                    )
                )
                or changed
            )
        if hasattr(self, "setRasterImageLayerTransform"):
            changed = (
                bool(
                    self.setRasterImageLayerTransform(
                        layer_id,
                        tx=float(payload.get("tx", 0.0) or 0.0),
                        ty=float(payload.get("ty", 0.0) or 0.0),
                        sx=float(payload.get("sx", 1.0) or 1.0),
                        sy=float(payload.get("sy", 1.0) or 1.0),
                    )
                )
                or changed
            )
        if changed:
            self._refreshViewerLayerDock()
        return changed

    def _onViewerLayerApplySettingsRequested(
        self, layer_id: str, settings: dict
    ) -> None:
        self._applyViewerLayerSettings(layer_id, settings)

    def _onViewerLayerSaveSettingsRequested(
        self, layer_id: str, settings: dict
    ) -> None:
        self._applyViewerLayerSettings(layer_id, settings)

    def _onViewerLayerQuickTransformRequested(
        self, layer_id: str, payload: dict
    ) -> None:
        layer_id = str(layer_id or "")
        if not self._is_raster_overlay_layer_id(layer_id):
            return
        data = dict(payload or {})
        action = str(data.get("action") or "").strip().lower()
        changed = False
        if action == "scale" and hasattr(self, "scaleRasterImageLayer"):
            changed = bool(
                self.scaleRasterImageLayer(
                    layer_id,
                    factor=float(data.get("factor", 1.0) or 1.0),
                    keep_center=bool(data.get("keep_center", True)),
                )
            )
        elif action == "align" and hasattr(self, "alignRasterImageLayer"):
            changed = bool(
                self.alignRasterImageLayer(
                    layer_id,
                    horizontal=data.get("horizontal"),
                    vertical=data.get("vertical"),
                )
            )
        if changed:
            self._refreshViewerLayerDock()

    def _onViewerLayerInteractiveResizeToggled(self, enabled: bool) -> None:
        large_view = getattr(self, "large_image_view", None)
        if large_view is None or not hasattr(
            large_view, "set_raster_overlay_arrow_mode"
        ):
            return
        try:
            large_view.set_raster_overlay_arrow_mode(bool(enabled))
        except Exception:
            return
