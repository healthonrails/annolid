from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import numpy as np
from qtpy import QtCore, QtWidgets

from annolid.gui.affine import (
    apply_affine_to_shape_points,
    solve_affine_from_landmarks,
)
from annolid.gui.overlay import (
    OverlayDocument,
    OverlayLandmarkPairState,
    OverlayRecordModel,
    OverlayTransform,
    VectorShape,
    overlay_delta_matrix,
    overlay_landmark_pair_to_dict,
    overlay_record_from_dict,
    overlay_record_to_dict,
    overlay_transform_from_dict,
    overlay_transform_to_dict,
    points_bounds_center,
)
from annolid.gui.viewer_layers import (
    AnnotationLayer,
    LandmarkLayer,
    LandmarkPair,
    VectorOverlayLayer,
    vector_overlay_layer_from_record,
)
from annolid.gui.status import post_window_status
from annolid.io.vector import (
    export_overlay_document_json,
    export_overlay_document_labelme,
    export_overlay_document_svg,
)
from annolid.gui.svg_overlay import import_vector_shapes
from annolid.gui.widgets.vector_overlay_dock import VectorOverlayDockWidget


class VectorOverlayMixin:
    """Import external vector overlays as standard editable Annolid shapes."""

    def _postVectorOverlayStatus(self, message: str, timeout: int = 4000) -> None:
        post_window_status(self, message, timeout)

    def _currentOverlayImportImageSize(self) -> tuple[float, float] | None:
        large_image_view = getattr(self, "large_image_view", None)
        if large_image_view is not None and hasattr(large_image_view, "content_size"):
            try:
                width, height = large_image_view.content_size()
                if width and height:
                    return float(width), float(height)
            except Exception:
                pass
        image = getattr(self, "image", None)
        if image is not None and hasattr(image, "isNull") and not image.isNull():
            try:
                return float(image.width()), float(image.height())
            except Exception:
                pass
        return None

    def _overlay_records(self) -> list[OverlayRecordModel]:
        if not isinstance(getattr(self, "otherData", None), dict):
            return []
        records = []
        for record in list(self.otherData.get("svg_overlays") or []):
            model = overlay_record_from_dict(record)
            if model is not None:
                records.append(model)
        return records

    def _set_overlay_records(self, records: list[OverlayRecordModel]) -> None:
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        self.otherData["svg_overlays"] = [
            overlay_record_to_dict(record) for record in list(records or [])
        ]

    def _overlay_record_model(
        self, overlay_id: str
    ) -> tuple[int | None, OverlayRecordModel | None]:
        for index, record in enumerate(self._overlay_records()):
            if str(record.id or "") == str(overlay_id or ""):
                return index, record
        return None, None

    @staticmethod
    def _shape_identity_key(shape) -> str:
        other = dict(getattr(shape, "other_data", {}) or {})
        element_id = str(other.get("overlay_element_id", "") or "")
        if element_id:
            return element_id
        label = str(getattr(shape, "label", "") or "")
        points = getattr(shape, "points", []) or []
        first = points[0] if points else None
        if first is not None:
            return f"{label}:{float(first.x()):.3f}:{float(first.y()):.3f}"
        return label or f"shape_{id(shape)}"

    @classmethod
    def _landmark_pair_state_from_shapes(cls, overlay_shape, image_shape, pair_id: str):
        overlay_other = dict(getattr(overlay_shape, "other_data", {}) or {})
        return OverlayLandmarkPairState(
            pair_id=str(pair_id or ""),
            overlay_label=str(getattr(overlay_shape, "label", "") or "") or None,
            image_label=str(getattr(image_shape, "label", "") or "") or None,
            overlay_element_id=str(overlay_other.get("overlay_element_id", "") or "")
            or None,
            image_shape_key=cls._shape_identity_key(image_shape) or None,
        )

    def _sync_overlay_record_landmark_pairs(self, overlay_id: str) -> None:
        record_index, record = self._overlay_record_model(overlay_id)
        if record_index is None or record is None:
            return
        pair_states = []
        seen = set()
        for shape in self._iter_overlay_shapes(overlay_id):
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if not pair_id or pair_id in seen:
                continue
            image_shape = self._find_image_landmark_pair_shape(pair_id)
            if image_shape is None:
                continue
            seen.add(pair_id)
            pair_states.append(
                self._landmark_pair_state_from_shapes(shape, image_shape, pair_id)
            )
        records = self._overlay_records()
        updated = record
        updated.landmark_pairs = pair_states
        records[record_index] = updated
        self._set_overlay_records(records)

    def _shape_backed_overlay_landmark_pairs(
        self, overlay_id: str
    ) -> list[OverlayLandmarkPairState]:
        pair_states = []
        seen = set()
        for shape in self._iter_overlay_shapes(overlay_id):
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if not pair_id or pair_id in seen:
                continue
            image_shape = self._find_image_landmark_pair_shape(pair_id)
            if image_shape is None:
                continue
            seen.add(pair_id)
            pair_states.append(
                self._landmark_pair_state_from_shapes(shape, image_shape, pair_id)
            )
        return pair_states

    def _editable_shape_vector(self, shape) -> VectorShape:
        other = dict(getattr(shape, "other_data", {}) or {})
        shape_type = str(getattr(shape, "shape_type", "") or "").lower()
        if shape_type == "polygon":
            kind = "polygon"
        elif shape_type == "point":
            kind = "point"
        else:
            kind = "polyline"
        return VectorShape(
            id=str(
                other.get("overlay_element_id")
                or getattr(shape, "label", "")
                or f"shape_{id(shape)}"
            ),
            kind=kind,
            points=[
                (float(point.x()), float(point.y()))
                for point in getattr(shape, "points", []) or []
            ],
            label=str(getattr(shape, "label", "") or "") or None,
            stroke=other.get("overlay_stroke"),
            fill=other.get("overlay_fill"),
            text=other.get("overlay_text"),
            locked=bool(other.get("overlay_locked", False)),
            source_tag=other.get("overlay_element"),
            layer_name=other.get("overlay_layer"),
            source_path=other.get("overlay_source"),
        )

    @staticmethod
    def _shape_bounds(shapes) -> tuple[float, float, float, float] | None:
        xs = []
        ys = []
        for shape in list(shapes or []):
            for point in getattr(shape, "points", []) or []:
                xs.append(float(point.x()))
                ys.append(float(point.y()))
        if not xs or not ys:
            return None
        return min(xs), min(ys), max(xs), max(ys)

    def _fitImportedOverlayToImage(self, result) -> None:
        image_size = self._currentOverlayImportImageSize()
        bounds = self._shape_bounds(getattr(result, "shapes", []))
        if image_size is None or bounds is None:
            return
        image_width, image_height = image_size
        min_x, min_y, max_x, max_y = bounds
        overlay_width = max(1e-6, max_x - min_x)
        overlay_height = max(1e-6, max_y - min_y)
        image_max = max(image_width, image_height)
        overlay_max = max(overlay_width, overlay_height)
        is_far_outside = (
            min_x < (-0.1 * image_width)
            or min_y < (-0.1 * image_height)
            or max_x > (1.1 * image_width)
            or max_y > (1.1 * image_height)
        )
        scale_ratio = image_max / overlay_max if overlay_max > 0 else 1.0
        if not is_far_outside and 0.35 <= scale_ratio <= 2.5:
            return

        target_scale = min(
            (image_width * 0.9) / overlay_width, (image_height * 0.9) / overlay_height
        )
        if target_scale <= 0:
            return
        current_center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        target_center = (image_width / 2.0, image_height / 2.0)
        target_transform = OverlayTransform(
            tx=float(target_center[0] - current_center[0]),
            ty=float(target_center[1] - current_center[1]),
            sx=float(target_scale),
            sy=float(target_scale),
            rotation_deg=0.0,
            opacity=float(
                (result.metadata or {}).get("transform", {}).get("opacity", 0.5)
            ),
            visible=bool(
                (result.metadata or {}).get("transform", {}).get("visible", True)
            ),
            z_order=int((result.metadata or {}).get("transform", {}).get("z_order", 0)),
        )
        affine_delta = overlay_delta_matrix(
            OverlayTransform(),
            target_transform,
            pivot=current_center,
        )
        for shape in list(getattr(result, "shapes", []) or []):
            apply_affine_to_shape_points(shape, affine_delta)
            other = dict(getattr(shape, "other_data", {}) or {})
            other["overlay_document_transform"] = overlay_transform_to_dict(
                target_transform
            )
            shape.other_data = other
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict):
            metadata["transform"] = overlay_transform_to_dict(target_transform)
            metadata["initial_fit_to_image"] = True

    def setupVectorOverlayDock(self) -> None:
        if not hasattr(self, "_selected_overlay_landmark_pair_id"):
            self._selected_overlay_landmark_pair_id = None
        if not hasattr(self, "_selected_overlay_landmark_overlay_id"):
            self._selected_overlay_landmark_overlay_id = None
        if not hasattr(self, "_vector_overlay_pair_sync_in_progress"):
            self._vector_overlay_pair_sync_in_progress = False
        dock = getattr(self, "vector_overlay_dock", None)
        if isinstance(dock, VectorOverlayDockWidget):
            self._refreshVectorOverlayDock()
            return
        dock = VectorOverlayDockWidget(self)
        dock.overlayApplyRequested.connect(self._applyVectorOverlayFromDock)
        dock.overlayResetRequested.connect(self._resetVectorOverlayFromDock)
        dock.overlayPairSelectedRequested.connect(self._pairVectorOverlayFromDock)
        dock.overlayPairSelectionChanged.connect(self._selectVectorOverlayPairFromDock)
        dock.overlayRemovePairRequested.connect(self._removeVectorOverlayPairFromDock)
        dock.overlayClearPairsRequested.connect(self._clearVectorOverlayPairsFromDock)
        dock.overlayLandmarkAlignRequested.connect(self._alignVectorOverlayFromDock)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, dock)
        self.vector_overlay_dock = dock
        if hasattr(self, "shape_dock"):
            try:
                self.tabifyDockWidget(self.shape_dock, dock)
            except Exception:
                pass
        canvas = getattr(self, "canvas", None)
        if canvas is not None and hasattr(canvas, "selectionChanged"):
            try:
                canvas.selectionChanged.connect(
                    self._onVectorOverlayCanvasSelectionChanged
                )
            except Exception:
                pass
        if canvas is not None and hasattr(canvas, "overlayLandmarkPairSelected"):
            try:
                canvas.overlayLandmarkPairSelected.connect(
                    lambda pair_id: self._selectCurrentVectorOverlayPair(str(pair_id))
                )
            except Exception:
                pass
        large_image_view = getattr(self, "large_image_view", None)
        if large_image_view is not None and hasattr(
            large_image_view, "overlayLandmarkPairSelected"
        ):
            try:
                large_image_view.overlayLandmarkPairSelected.connect(
                    lambda pair_id: self._selectCurrentVectorOverlayPair(str(pair_id))
                )
            except Exception:
                pass
        self._refreshVectorOverlayDock()

    def _onVectorOverlayCanvasSelectionChanged(self, shapes) -> None:
        self._refreshVectorOverlayDock()
        if getattr(self, "_vector_overlay_pair_sync_in_progress", False):
            if hasattr(self, "_syncLargeImageDocument"):
                self._syncLargeImageDocument()
            return
        self._syncSelectedVectorOverlayPairFromShapes(shapes)
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()

    def _refreshVectorOverlayDock(self) -> None:
        dock = getattr(self, "vector_overlay_dock", None)
        self._syncVectorOverlayActionState()
        if isinstance(dock, VectorOverlayDockWidget):
            overlays = self.listVectorOverlays()
            dock.set_overlays(overlays)
            dock.setVisible(bool(overlays))
            overlay_id = dock._selected_overlay_id()
            self._applySelectedVectorOverlayPairHighlight(
                overlay_id, getattr(self, "_selected_overlay_landmark_pair_id", None)
            )

    def _applyVectorOverlayFromDock(self, overlay_id: str, payload: dict) -> None:
        self.setVectorOverlayTransform(overlay_id, **dict(payload or {}))

    def _resetVectorOverlayFromDock(self, overlay_id: str) -> None:
        self.setVectorOverlayTransform(
            overlay_id,
            tx=0.0,
            ty=0.0,
            sx=1.0,
            sy=1.0,
            rotation_deg=0.0,
            opacity=0.5,
            visible=True,
            z_order=0,
        )

    def _alignVectorOverlayFromDock(self, overlay_id: str) -> None:
        try:
            pair_count = self.alignVectorOverlayFromLandmarks(overlay_id)
        except Exception as exc:
            if hasattr(self, "errorMessage"):
                self.errorMessage(
                    self.tr("Landmark Alignment"),
                    self.tr("Failed to align overlay landmarks: %s") % str(exc),
                )
            return
        self._postVectorOverlayStatus(
            self.tr("Aligned overlay from %d landmark pairs") % int(pair_count)
        )

    def _pairVectorOverlayFromDock(self, overlay_id: str) -> None:
        try:
            self.pairSelectedVectorOverlayLandmarks(overlay_id)
        except Exception as exc:
            if hasattr(self, "errorMessage"):
                self.errorMessage(
                    self.tr("Landmark Pairing"),
                    self.tr("Failed to pair selected landmarks: %s") % str(exc),
                )
            return
        self._postVectorOverlayStatus(
            self.tr("Paired selected overlay/image landmarks")
        )

    def _removeVectorOverlayPairFromDock(self, overlay_id: str, pair_id: str) -> None:
        removed = self.removeVectorOverlayLandmarkPair(overlay_id, pair_id)
        if removed:
            self._postVectorOverlayStatus(self.tr("Removed selected landmark pair"))

    def _clearVectorOverlayPairsFromDock(self, overlay_id: str) -> None:
        removed = self.clearVectorOverlayLandmarkPairs(overlay_id)
        if removed:
            self._postVectorOverlayStatus(
                self.tr("Cleared %d explicit landmark pair(s)") % int(removed)
            )

    def _selectVectorOverlayPairFromDock(self, overlay_id: str, pair_id: str) -> None:
        self._applySelectedVectorOverlayPairHighlight(overlay_id, pair_id or None)

    def _selectCurrentVectorOverlayPair(self, pair_id: str | None) -> None:
        overlay_id = self._currentVectorOverlayId()
        self._applySelectedVectorOverlayPairHighlight(overlay_id, pair_id or None)

    def _overlay_id_for_pair_id(self, pair_id: str) -> str:
        target_pair_id = str(pair_id or "")
        if not target_pair_id:
            return ""
        for overlay in self.listVectorOverlays():
            overlay_id = str((overlay or {}).get("id") or "")
            landmark_summary = dict((overlay or {}).get("landmark_summary") or {})
            for pair in list(landmark_summary.get("explicit_pairs") or []):
                if str((pair or {}).get("pair_id") or "") == target_pair_id:
                    return overlay_id
        return ""

    def _selectedVectorOverlayPairFromShapes(self, shapes) -> tuple[str, str] | None:
        selected = list(shapes or [])
        point_shapes = [
            shape
            for shape in selected
            if str(getattr(shape, "shape_type", "") or "").lower() == "point"
            and getattr(shape, "points", None)
        ]
        if not point_shapes:
            return None
        pair_ids = set()
        overlay_ids = set()
        for shape in point_shapes:
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if not pair_id:
                continue
            pair_ids.add(pair_id)
            overlay_id = str(other.get("overlay_id") or "")
            if overlay_id:
                overlay_ids.add(overlay_id)
        if len(pair_ids) != 1:
            return None
        pair_id = next(iter(pair_ids))
        overlay_id = next(iter(overlay_ids), "")
        if not overlay_id:
            overlay_id = self._overlay_id_for_pair_id(pair_id)
        if not overlay_id:
            return None
        return overlay_id, pair_id

    def _syncSelectedVectorOverlayPairFromShapes(self, shapes) -> None:
        pair = self._selectedVectorOverlayPairFromShapes(shapes)
        if pair is None:
            if getattr(self, "_selected_overlay_landmark_pair_id", None):
                self._applySelectedVectorOverlayPairHighlight(None, None)
            return
        overlay_id, pair_id = pair
        if str(getattr(self, "_selected_overlay_landmark_overlay_id", "") or "") == str(
            overlay_id
        ) and str(getattr(self, "_selected_overlay_landmark_pair_id", "") or "") == str(
            pair_id
        ):
            return
        self._applySelectedVectorOverlayPairHighlight(overlay_id, pair_id)

    def _applySelectedVectorOverlayPairHighlight(
        self, overlay_id: str | None, pair_id: str | None
    ) -> None:
        selected_overlay_id = str(overlay_id or "")
        selected_pair_id = str(pair_id or "")
        if not selected_overlay_id or not selected_pair_id:
            selected_pair_id = ""
        else:
            valid_pair_ids = {
                str(pair.get("pair_id") or "")
                for overlay in self.listVectorOverlays()
                if str((overlay or {}).get("id") or "") == selected_overlay_id
                for pair in list(
                    dict((overlay or {}).get("landmark_summary") or {}).get(
                        "explicit_pairs"
                    )
                    or []
                )
            }
            if selected_pair_id not in valid_pair_ids:
                selected_pair_id = ""
        current_overlay_id = str(
            getattr(self, "_selected_overlay_landmark_overlay_id", "") or ""
        )
        current_pair_id = str(
            getattr(self, "_selected_overlay_landmark_pair_id", "") or ""
        )
        canvas = getattr(self, "canvas", None)
        large_image_view = getattr(self, "large_image_view", None)
        dock = getattr(self, "vector_overlay_dock", None)
        if (
            current_overlay_id == selected_overlay_id
            and current_pair_id == selected_pair_id
        ):
            if canvas is not None and hasattr(canvas, "setSelectedOverlayLandmarkPair"):
                canvas.setSelectedOverlayLandmarkPair(selected_pair_id)
            if large_image_view is not None and hasattr(
                large_image_view, "set_selected_landmark_pair"
            ):
                large_image_view.set_selected_landmark_pair(selected_pair_id)
            if isinstance(dock, VectorOverlayDockWidget):
                dock.set_selected_pair(
                    selected_overlay_id or current_overlay_id, selected_pair_id
                )
            return
        self._selected_overlay_landmark_overlay_id = selected_overlay_id or None
        self._selected_overlay_landmark_pair_id = selected_pair_id or None
        if canvas is not None and hasattr(canvas, "setSelectedOverlayLandmarkPair"):
            canvas.setSelectedOverlayLandmarkPair(selected_pair_id)
        if selected_overlay_id and selected_pair_id:
            self._selectVectorOverlayPairShapes(selected_overlay_id, selected_pair_id)
        if large_image_view is not None and hasattr(
            large_image_view, "set_selected_landmark_pair"
        ):
            large_image_view.set_selected_landmark_pair(selected_pair_id)
        if isinstance(dock, VectorOverlayDockWidget):
            dock.set_selected_pair(
                selected_overlay_id or current_overlay_id, selected_pair_id
            )

    def _selectVectorOverlayPairShapes(self, overlay_id: str, pair_id: str) -> None:
        overlay_shape = None
        for shape in self._iter_overlay_shapes(overlay_id):
            other = dict(getattr(shape, "other_data", {}) or {})
            if str(other.get("overlay_landmark_pair_id") or "") == str(pair_id):
                overlay_shape = shape
                break
        image_shape = self._find_image_landmark_pair_shape(pair_id)
        selected_shapes = [
            shape for shape in (overlay_shape, image_shape) if shape is not None
        ]
        if len(selected_shapes) != 2:
            return
        canvas = getattr(self, "canvas", None)
        self._vector_overlay_pair_sync_in_progress = True
        try:
            if canvas is not None and hasattr(canvas, "selectShapes"):
                canvas.selectShapes(selected_shapes)
            elif canvas is not None:
                canvas.selectedShapes = list(selected_shapes)
            if hasattr(self, "shapeSelectionChanged"):
                self.shapeSelectionChanged(selected_shapes)
        finally:
            self._vector_overlay_pair_sync_in_progress = False

    def _currentVectorOverlayId(self) -> str | None:
        dock = getattr(self, "vector_overlay_dock", None)
        if isinstance(dock, VectorOverlayDockWidget):
            return dock._selected_overlay_id()
        overlays = self.listVectorOverlays()
        if len(overlays) == 1:
            return str((overlays[0] or {}).get("id") or "")
        return None

    def importSvgOverlay(self, _value: bool = False) -> None:
        if self.image is None or self.image.isNull():
            self.errorMessage(
                self.tr("Import SVG Overlay"),
                self.tr("Open an image before importing an SVG overlay."),
            )
            return

        start_dir = self.lastOpenDir or str(Path.home())
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Import Vector Overlay"),
            start_dir,
            self.tr("Vector files (*.svg *.ai *.pdf)"),
        )
        if not filename:
            return

        try:
            result = import_vector_shapes(filename)
        except Exception as exc:
            self.errorMessage(
                self.tr("Import Vector Overlay"),
                self.tr("Failed to import vector overlay: %s") % str(exc),
            )
            return

        if not result.shapes:
            self.errorMessage(
                self.tr("Import Vector Overlay"),
                self.tr("No supported vector elements were found."),
            )
            return

        self._fitImportedOverlayToImage(result)
        self.loadShapes(result.shapes, replace=False)
        self.lastOpenDir = str(Path(filename).parent)

        record = OverlayRecordModel(
            id=str(result.metadata.get("id") or ""),
            source_path=str(filename),
            source_kind=str(result.metadata.get("source_kind", "svg") or "svg"),
            shape_count=int(len(result.shapes)),
            transform=overlay_transform_from_dict(result.metadata.get("transform")),
            metadata=dict(result.metadata or {}),
            source_shapes=list(result.metadata.get("source_shapes") or []),
            editable_shapes=[
                self._editable_shape_vector(shape)
                for shape in list(result.shapes or [])
            ],
            landmark_pairs=[],
        )
        records = self._overlay_records()
        records.append(record)
        self._set_overlay_records(records)
        self._refreshVectorOverlayDock()
        self.setDirty()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        self._postVectorOverlayStatus(
            self.tr("Imported %d vector overlay shapes from %s")
            % (len(result.shapes), Path(filename).name)
        )

    def listVectorOverlays(self) -> list[dict]:
        normalized = []
        record_map = {record.id: record for record in self._overlay_records()}
        for layer in self.vectorOverlayLayers():
            overlay_id = str(layer.id or "")
            record_model = record_map.get(overlay_id)
            pair_state_models = list(getattr(record_model, "landmark_pairs", []) or [])
            if not pair_state_models and overlay_id:
                pair_state_models = self._shape_backed_overlay_landmark_pairs(
                    overlay_id
                )
            record = {
                "id": layer.id,
                "source": layer.source_path,
                "source_kind": str(
                    getattr(record_model, "source_kind", "")
                    or dict(layer.metadata or {}).get("source_kind")
                    or "svg"
                ),
                "shape_count": int(layer.shape_count),
                "metadata": dict(layer.metadata or {}),
                "transform": overlay_transform_to_dict(
                    OverlayTransform(
                        tx=layer.transform.tx,
                        ty=layer.transform.ty,
                        sx=layer.transform.sx,
                        sy=layer.transform.sy,
                        rotation_deg=layer.transform.rotation_deg,
                        opacity=layer.opacity,
                        visible=layer.visible,
                        z_order=layer.z_index,
                    )
                ),
                "name": layer.name,
                "source_locked": bool(
                    dict(layer.metadata or {}).get("locked_source", True)
                ),
                "editable_layer_name": str(
                    dict(layer.metadata or {}).get("editable_layer_name")
                    or "Corrections"
                ),
            }
            pairs = (
                self._collect_overlay_landmark_pairs(overlay_id) if overlay_id else []
            )
            explicit_pairs = [
                overlay_landmark_pair_to_dict(pair) for pair in pair_state_models
            ]
            explicit_count = len(explicit_pairs)
            candidate = self._selected_point_pair_candidate(overlay_id)
            record["landmark_summary"] = {
                "matched_count": len(pairs),
                "explicit_count": explicit_count,
                "auto_count": max(0, len(pairs) - explicit_count),
                "labels": [
                    key[0] if not key[1] else f"{key[0]} ({key[1]})"
                    for _src, _dst, key in pairs
                ],
                "explicit_pairs": explicit_pairs,
                "pair_candidate": (
                    {
                        "overlay_label": str(getattr(candidate[0], "label", "") or ""),
                        "image_label": str(getattr(candidate[1], "label", "") or ""),
                    }
                    if candidate is not None
                    else {}
                ),
            }
            normalized.append(record)
        return normalized

    def vectorOverlayLayers(self) -> list[VectorOverlayLayer]:
        layers = []
        for record_model in self._overlay_records():
            overlay_id = str(record_model.id or "")
            editable_shapes = []
            for shape in self._iter_overlay_shapes(overlay_id):
                editable_shapes.append(self._editable_shape_vector(shape))
            record = overlay_record_to_dict(record_model)
            record["editable_shapes"] = editable_shapes
            layer = vector_overlay_layer_from_record(record, shapes=editable_shapes)
            if layer is not None:
                layers.append(layer)
        return layers

    def vectorOverlayLandmarkLayers(self) -> list[LandmarkLayer]:
        layers = []
        record_map = {record.id: record for record in self._overlay_records()}
        for overlay_layer in self.vectorOverlayLayers():
            pairs = []
            record_model = record_map.get(overlay_layer.id)
            landmarks_visible = True
            if record_model is not None:
                landmarks_visible = bool(
                    dict(getattr(record_model, "metadata", {}) or {}).get(
                        "landmarks_visible", True
                    )
                )
            pair_id_order = [
                str(pair.pair_id or "")
                for pair in list(getattr(record_model, "landmark_pairs", []) or [])
            ]
            for pair_index, (src, dst, key) in enumerate(
                self._collect_overlay_landmark_pairs(overlay_layer.id)
            ):
                pair_id = (
                    pair_id_order[pair_index]
                    if pair_index < len(pair_id_order) and pair_id_order[pair_index]
                    else f"{overlay_layer.id}:{pair_index}"
                )
                pairs.append(
                    LandmarkPair(
                        pair_id=pair_id,
                        source=(float(src[0]), float(src[1])),
                        target=(float(dst[0]), float(dst[1])),
                        key=(str(key[0]), str(key[1])),
                    )
                )
            layers.append(
                LandmarkLayer(
                    id=f"{overlay_layer.id}_landmarks",
                    name=f"{overlay_layer.name} landmarks",
                    visible=landmarks_visible,
                    opacity=1.0,
                    locked=False,
                    z_index=max(overlay_layer.z_index + 1, 1),
                    pairs=pairs,
                )
            )
        return layers

    def setVectorOverlayLandmarkLayerVisible(
        self, overlay_id: str, visible: bool
    ) -> bool:
        overlay_index, current_record = self._overlay_record_model(overlay_id)
        if overlay_index is None or current_record is None:
            return False
        visible_flag = bool(visible)
        metadata = dict(current_record.metadata or {})
        if bool(metadata.get("landmarks_visible", True)) == visible_flag:
            return False
        metadata["landmarks_visible"] = visible_flag
        updated_record = OverlayRecordModel(
            id=current_record.id,
            source_path=current_record.source_path,
            source_kind=current_record.source_kind,
            transform=current_record.transform,
            shape_count=current_record.shape_count,
            metadata=metadata,
            source_shapes=list(current_record.source_shapes or []),
            landmark_pairs=list(current_record.landmark_pairs or []),
        )
        records = self._overlay_records()
        records[overlay_index] = updated_record
        self._set_overlay_records(records)
        for shape in self._iter_overlay_shapes(overlay_id):
            other = dict(getattr(shape, "other_data", {}) or {})
            other["overlay_landmarks_visible"] = visible_flag
            shape.other_data = other
        if not visible_flag and str(
            getattr(self, "_selected_overlay_landmark_pair_id", "") or ""
        ):
            active_pair = str(
                getattr(self, "_selected_overlay_landmark_pair_id", "") or ""
            )
            for pair in self.vectorOverlayLandmarkLayers():
                if pair.id == f"{overlay_id}_landmarks":
                    if any(
                        str(item.pair_id or "") == active_pair
                        for item in list(pair.pairs or [])
                    ):
                        self._setSelectedVectorOverlayLandmarkPair(None)
                        break
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.update()
        if getattr(self, "large_image_view", None) is not None:
            self.large_image_view.set_shapes(getattr(canvas, "shapes", []))
        self._refreshVectorOverlayDock()
        if hasattr(self, "setDirty"):
            self.setDirty()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def currentAnnotationLayer(self) -> AnnotationLayer:
        canvas = getattr(self, "canvas", None)
        shapes = [
            shape
            for shape in list(getattr(canvas, "shapes", []) or [])
            if "overlay_id" not in dict(getattr(shape, "other_data", {}) or {})
        ]
        return AnnotationLayer(
            id="annotations",
            name="Annotations",
            visible=True,
            opacity=1.0,
            locked=False,
            z_index=100,
            shapes=shapes,
        )

    def _iter_overlay_shapes(self, overlay_id: str):
        canvas = getattr(self, "canvas", None)
        for shape in list(getattr(canvas, "shapes", []) or []):
            other = dict(getattr(shape, "other_data", {}) or {})
            if other.get("overlay_id") == overlay_id:
                yield shape

    def _iter_non_overlay_shapes(self):
        canvas = getattr(self, "canvas", None)
        for shape in list(getattr(canvas, "shapes", []) or []):
            other = dict(getattr(shape, "other_data", {}) or {})
            if "overlay_id" not in other:
                yield shape

    def _selected_point_pair_candidate(self, overlay_id: str):
        canvas = getattr(self, "canvas", None)
        selected = list(getattr(canvas, "selectedShapes", []) or [])
        if len(selected) != 2:
            return None
        point_shapes = [
            shape
            for shape in selected
            if str(getattr(shape, "shape_type", "") or "").lower() == "point"
            and getattr(shape, "points", None)
        ]
        if len(point_shapes) != 2:
            return None
        overlay_shape = None
        image_shape = None
        for shape in point_shapes:
            other = dict(getattr(shape, "other_data", {}) or {})
            if other.get("overlay_id") == overlay_id:
                overlay_shape = shape
            elif "overlay_id" not in other:
                image_shape = shape
        if overlay_shape is None or image_shape is None:
            return None
        return overlay_shape, image_shape

    def _find_image_landmark_pair_shape(self, pair_id: str):
        if not pair_id:
            return None
        for shape in self._iter_non_overlay_shapes():
            other = dict(getattr(shape, "other_data", {}) or {})
            if str(other.get("overlay_landmark_pair_id") or "") == str(pair_id):
                return shape
        return None

    def _landmark_key_for_shape(self, shape) -> tuple[str, str]:
        label = str(getattr(shape, "label", "") or "").strip()
        group_id = getattr(shape, "group_id", None)
        return (label, "" if group_id in (None, "") else str(group_id))

    def _collect_overlay_landmark_pairs(
        self, overlay_id: str
    ) -> list[tuple[tuple[float, float], tuple[float, float], tuple[str, str]]]:
        explicit_pairs = []
        explicit_overlay = {}
        explicit_image = {}
        overlay_points = {}
        for shape in self._iter_overlay_shapes(overlay_id):
            if str(getattr(shape, "shape_type", "") or "").lower() != "point":
                continue
            if not getattr(shape, "points", None):
                continue
            coords = (
                float(shape.points[0].x()),
                float(shape.points[0].y()),
            )
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if pair_id:
                explicit_overlay[pair_id] = (
                    coords,
                    self._landmark_key_for_shape(shape),
                )
                continue
            overlay_points[self._landmark_key_for_shape(shape)] = coords

        image_points = {}
        for shape in self._iter_non_overlay_shapes():
            if str(getattr(shape, "shape_type", "") or "").lower() != "point":
                continue
            if not getattr(shape, "points", None):
                continue
            coords = (
                float(shape.points[0].x()),
                float(shape.points[0].y()),
            )
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if pair_id:
                explicit_image[pair_id] = (coords, self._landmark_key_for_shape(shape))
                continue
            image_points[self._landmark_key_for_shape(shape)] = coords

        for pair_id, (src, key) in explicit_overlay.items():
            image_entry = explicit_image.get(pair_id)
            if image_entry is None:
                continue
            explicit_pairs.append((src, image_entry[0], key))

        pairs = []
        for key, src in overlay_points.items():
            dst = image_points.get(key)
            if dst is None:
                continue
            pairs.append((src, dst, key))
        return explicit_pairs + pairs

    def _overlay_record(self, overlay_id: str) -> tuple[int, dict] | tuple[None, None]:
        if not isinstance(getattr(self, "otherData", None), dict):
            return None, None
        overlays = list(self.otherData.get("svg_overlays") or [])
        for idx, overlay in enumerate(overlays):
            if str((overlay or {}).get("id") or "") == str(overlay_id):
                return idx, dict(overlay or {})
        return None, None

    def buildVectorOverlayDocument(self, overlay_id: str) -> OverlayDocument | None:
        _overlay_index, record_model = self._overlay_record_model(overlay_id)
        if record_model is None:
            return None
        metadata = dict(record_model.metadata or {})
        layer = next(
            (item for item in self.vectorOverlayLayers() if item.id == str(overlay_id)),
            None,
        )
        if layer is None:
            return None
        return OverlayDocument(
            source_path=str(layer.source_path or record_model.source_path or ""),
            layer_name=str(
                metadata.get("layer_name") or Path(str(layer.source_path or "")).stem
            ),
            transform=record_model.transform,
            shapes=list(layer.shapes),
            source_kind=str(record_model.source_kind or "svg"),
            source_shapes=list(record_model.source_shapes or []),
            landmark_pairs=[
                overlay_landmark_pair_to_dict(pair)
                for pair in list(record_model.landmark_pairs or [])
            ],
        )

    def _overlay_bounds_center(self, overlay_id: str) -> tuple[float, float]:
        points = []
        for shape in self._iter_overlay_shapes(overlay_id):
            points.extend(
                [
                    (float(point.x()), float(point.y()))
                    for point in getattr(shape, "points", [])
                ]
            )
        return points_bounds_center(points)

    def _sync_overlay_visibility_items(self, overlay_id: str, visible: bool) -> None:
        label_list = getattr(self, "labelList", None)
        if label_list is None:
            return
        with QtCore.QSignalBlocker(label_list):
            for index in range(label_list.count()):
                item = label_list.item(index)
                shape = item.shape() if hasattr(item, "shape") else None
                other = dict(getattr(shape, "other_data", {}) or {})
                if other.get("overlay_id") != overlay_id:
                    continue
                item.setCheckState(
                    QtCore.Qt.Checked if visible else QtCore.Qt.Unchecked
                )

    def setAllVectorOverlaysVisible(self, visible: bool) -> int:
        if not isinstance(getattr(self, "otherData", None), dict):
            return 0
        overlays = list(self.otherData.get("svg_overlays") or [])
        if not overlays:
            self._syncVectorOverlayActionState()
            return 0

        visible_flag = bool(visible)
        changed = 0
        for idx, overlay in enumerate(overlays):
            current = dict(overlay or {})
            overlay_id = str(current.get("id") or "")
            transform = overlay_transform_from_dict(current.get("transform"))
            if transform.visible == visible_flag:
                continue
            target_transform = OverlayTransform(
                tx=transform.tx,
                ty=transform.ty,
                sx=transform.sx,
                sy=transform.sy,
                rotation_deg=transform.rotation_deg,
                opacity=transform.opacity,
                visible=visible_flag,
                z_order=transform.z_order,
            )
            current["transform"] = overlay_transform_to_dict(target_transform)
            metadata = dict(current.get("metadata") or {})
            metadata["transform"] = dict(current["transform"])
            current["metadata"] = metadata
            overlays[idx] = current
            for shape in self._iter_overlay_shapes(overlay_id):
                other = dict(getattr(shape, "other_data", {}) or {})
                other["overlay_visible"] = visible_flag
                other["overlay_document_transform"] = overlay_transform_to_dict(
                    target_transform
                )
                shape.other_data = other
                shape.visible = visible_flag
            self._sync_overlay_visibility_items(overlay_id, visible_flag)
            changed += 1

        if changed <= 0:
            self._syncVectorOverlayActionState()
            return 0

        self.otherData["svg_overlays"] = overlays
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.update()
        if getattr(self, "large_image_view", None) is not None:
            self.large_image_view.set_shapes(getattr(canvas, "shapes", []))
        self._refreshVectorOverlayDock()
        self._syncVectorOverlayActionState()
        if hasattr(self, "setDirty"):
            self.setDirty()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return changed

    def toggleVectorOverlaysVisible(self, value: bool = False) -> None:
        self.setAllVectorOverlaysVisible(bool(value))

    def _syncVectorOverlayActionState(self) -> None:
        actions = getattr(self, "actions", None)
        action = getattr(actions, "toggle_vector_overlays_visible", None)
        if action is None:
            return
        overlays = self.listVectorOverlays()
        enabled = bool(overlays)
        checked = enabled and all(
            bool(dict(overlay.get("transform") or {}).get("visible", True))
            for overlay in overlays
        )
        with QtCore.QSignalBlocker(action):
            action.setEnabled(enabled)
            action.setChecked(bool(checked))

    def setVectorOverlayTransform(
        self,
        overlay_id: str,
        *,
        tx: float | None = None,
        ty: float | None = None,
        sx: float | None = None,
        sy: float | None = None,
        rotation_deg: float | None = None,
        opacity: float | None = None,
        visible: bool | None = None,
        z_order: int | None = None,
    ) -> bool:
        overlay_index, current_record = self._overlay_record_model(overlay_id)
        if overlay_index is None or current_record is None:
            return False

        current_transform = current_record.transform
        target_transform = OverlayTransform(
            tx=current_transform.tx if tx is None else float(tx),
            ty=current_transform.ty if ty is None else float(ty),
            sx=current_transform.sx if sx is None else float(sx),
            sy=current_transform.sy if sy is None else float(sy),
            rotation_deg=(
                current_transform.rotation_deg
                if rotation_deg is None
                else float(rotation_deg)
            ),
            opacity=current_transform.opacity if opacity is None else float(opacity),
            visible=current_transform.visible if visible is None else bool(visible),
            z_order=current_transform.z_order if z_order is None else int(z_order),
        )
        if abs(target_transform.sx) < 1e-8 or abs(target_transform.sy) < 1e-8:
            raise ValueError("Overlay scale must be non-zero")

        pivot = self._overlay_bounds_center(overlay_id)
        affine_delta = overlay_delta_matrix(
            current_transform, target_transform, pivot=pivot
        )
        has_geometry_delta = not (
            current_transform.tx == target_transform.tx
            and current_transform.ty == target_transform.ty
            and current_transform.sx == target_transform.sx
            and current_transform.sy == target_transform.sy
            and current_transform.rotation_deg == target_transform.rotation_deg
        )
        for shape in self._iter_overlay_shapes(overlay_id):
            if has_geometry_delta:
                apply_affine_to_shape_points(shape, affine_delta)
            other = dict(getattr(shape, "other_data", {}) or {})
            other["overlay_visible"] = target_transform.visible
            other["overlay_opacity"] = target_transform.opacity
            other["overlay_z_order"] = target_transform.z_order
            other["overlay_document_transform"] = overlay_transform_to_dict(
                target_transform
            )
            shape.other_data = other
            shape.visible = bool(target_transform.visible)

        records = self._overlay_records()
        current_record.transform = target_transform
        metadata = dict(current_record.metadata or {})
        metadata["transform"] = overlay_transform_to_dict(target_transform)
        current_record.metadata = metadata
        current_record.editable_shapes = [
            self._editable_shape_vector(shape)
            for shape in self._iter_overlay_shapes(overlay_id)
        ]
        records[overlay_index] = current_record
        self._set_overlay_records(records)

        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.update()
        if getattr(self, "large_image_view", None) is not None:
            self.large_image_view.set_shapes(getattr(canvas, "shapes", []))
            self._applySelectedVectorOverlayPairHighlight(
                overlay_id, getattr(self, "_selected_overlay_landmark_pair_id", None)
            )
        self._sync_overlay_visibility_items(overlay_id, target_transform.visible)
        self._refreshVectorOverlayDock()
        self._syncVectorOverlayActionState()
        if hasattr(self, "setDirty"):
            self.setDirty()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def alignVectorOverlayFromLandmarks(self, overlay_id: str) -> int:
        overlay_index, current_record = self._overlay_record_model(overlay_id)
        if overlay_index is None or current_record is None:
            return 0

        pairs = self._collect_overlay_landmark_pairs(overlay_id)
        if len(pairs) < 3:
            raise ValueError(
                "Need at least 3 matching point landmarks shared by label/group_id"
            )

        source = [src for src, _dst, _key in pairs]
        target = [dst for _src, dst, _key in pairs]
        matrix = solve_affine_from_landmarks(source, target)

        for shape in self._iter_overlay_shapes(overlay_id):
            apply_affine_to_shape_points(shape, matrix)
            other = dict(getattr(shape, "other_data", {}) or {})
            other["overlay_landmark_alignment_matrix"] = matrix.tolist()
            other["overlay_landmark_pair_count"] = len(pairs)
            shape.other_data = other

        current_transform = current_record.transform
        reset_transform = OverlayTransform(
            opacity=current_transform.opacity,
            visible=current_transform.visible,
            z_order=current_transform.z_order,
        )
        metadata = dict(current_record.metadata or {})
        metadata["transform"] = overlay_transform_to_dict(reset_transform)
        metadata["landmark_alignment"] = {
            "affine_matrix": np.asarray(matrix, dtype=float).tolist(),
            "pair_count": len(pairs),
            "keys": [list(key) for _src, _dst, key in pairs],
        }
        records = self._overlay_records()
        current_record.transform = reset_transform
        current_record.metadata = metadata
        current_record.editable_shapes = [
            self._editable_shape_vector(shape)
            for shape in self._iter_overlay_shapes(overlay_id)
        ]
        records[overlay_index] = current_record
        self._set_overlay_records(records)
        self._sync_overlay_record_landmark_pairs(overlay_id)

        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.update()
        if getattr(self, "large_image_view", None) is not None:
            self.large_image_view.set_shapes(getattr(canvas, "shapes", []))
            self._applySelectedVectorOverlayPairHighlight(
                overlay_id, getattr(self, "_selected_overlay_landmark_pair_id", None)
            )
        self._refreshVectorOverlayDock()
        if hasattr(self, "setDirty"):
            self.setDirty()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return len(pairs)

    def pairSelectedVectorOverlayLandmarks(self, overlay_id: str) -> str:
        candidate = self._selected_point_pair_candidate(overlay_id)
        if candidate is None:
            raise ValueError(
                "Select exactly two point shapes: one overlay point and one image point."
            )
        overlay_shape, image_shape = candidate
        pair_id = f"overlay_pair_{uuid4().hex}"
        for shape in (overlay_shape, image_shape):
            other = dict(getattr(shape, "other_data", {}) or {})
            other["overlay_landmark_pair_id"] = pair_id
            shape.other_data = other
        canvas = getattr(self, "canvas", None)
        if canvas is not None:
            canvas.update()
        self._sync_overlay_record_landmark_pairs(overlay_id)
        self._applySelectedVectorOverlayPairHighlight(overlay_id, pair_id)
        self._refreshVectorOverlayDock()
        if hasattr(self, "setDirty"):
            self.setDirty()
        return pair_id

    def removeVectorOverlayLandmarkPair(self, overlay_id: str, pair_id: str) -> bool:
        removed = False
        for shape in list(self._iter_overlay_shapes(overlay_id)) + list(
            self._iter_non_overlay_shapes()
        ):
            other = dict(getattr(shape, "other_data", {}) or {})
            if str(other.get("overlay_landmark_pair_id") or "") != str(pair_id):
                continue
            other.pop("overlay_landmark_pair_id", None)
            shape.other_data = other
            removed = True
        if removed:
            canvas = getattr(self, "canvas", None)
            if canvas is not None:
                canvas.update()
            self._sync_overlay_record_landmark_pairs(overlay_id)
            if str(
                getattr(self, "_selected_overlay_landmark_pair_id", "") or ""
            ) == str(pair_id):
                self._applySelectedVectorOverlayPairHighlight(overlay_id, None)
            self._refreshVectorOverlayDock()
            if hasattr(self, "setDirty"):
                self.setDirty()
        return removed

    def clearVectorOverlayLandmarkPairs(self, overlay_id: str) -> int:
        pair_ids = set()
        for shape in self._iter_overlay_shapes(overlay_id):
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if pair_id:
                pair_ids.add(pair_id)
        removed = 0
        for pair_id in pair_ids:
            if self.removeVectorOverlayLandmarkPair(overlay_id, pair_id):
                removed += 1
        return removed

    def exportVectorOverlay(
        self,
        _value: bool = False,
        *,
        overlay_id: str | None = None,
        output_path: str | None = None,
    ) -> str | None:
        resolved_overlay_id = str(overlay_id or self._currentVectorOverlayId() or "")
        if not resolved_overlay_id:
            if hasattr(self, "errorMessage"):
                self.errorMessage(
                    self.tr("Export Vector Overlay"),
                    self.tr("Select or import an overlay before exporting."),
                )
            return None
        document = self.buildVectorOverlayDocument(resolved_overlay_id)
        if document is None:
            if hasattr(self, "errorMessage"):
                self.errorMessage(
                    self.tr("Export Vector Overlay"),
                    self.tr(
                        "Could not find the selected overlay in the current project."
                    ),
                )
            return None

        target = output_path
        if not target:
            suggested = f"{Path(document.source_path or 'overlay').stem}_corrected.svg"
            target, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                self.tr("Export Vector Overlay"),
                str(Path(self.lastOpenDir or Path.home()) / suggested),
                self.tr(
                    "SVG files (*.svg);;Overlay JSON files (*.json);;LabelMe JSON files (*.labelme.json)"
                ),
            )
        if not target:
            return None

        suffix = Path(target).suffix.lower()
        target_name = Path(target).name.lower()
        if target_name.endswith(".labelme.json"):
            exported = export_overlay_document_labelme(document, target)
        elif suffix == ".json":
            exported = export_overlay_document_json(document, target)
        else:
            exported = export_overlay_document_svg(document, target)
        self.lastOpenDir = str(exported.parent)
        self._postVectorOverlayStatus(
            self.tr("Exported corrected overlay to %s") % str(exported.name)
        )
        return str(exported)
