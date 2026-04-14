from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from qtpy import QtWidgets

from annolid.gui.large_image import open_large_image
from annolid.gui.status import post_window_status
from annolid.gui.viewer_layers import AffineTransform, RasterImageLayer


class RasterLayerMixin:
    def _base_raster_layer_z_index(self) -> float:
        raster = getattr(self, "currentRasterImageLayer", lambda: None)()
        try:
            return float(getattr(raster, "z_index", -100.0) or -100.0)
        except Exception:
            return -100.0

    def _reference_raster_layer_id(self, layer_id: str) -> str:
        target = str(layer_id or "").strip()
        if not target:
            return "raster_image"
        layers = sorted(
            list(getattr(self, "viewerLayerModels", lambda: [])() or []),
            key=lambda layer: float(getattr(layer, "z_index", 0.0) or 0.0),
        )
        current_index = next(
            (
                idx
                for idx, layer in enumerate(layers)
                if isinstance(layer, RasterImageLayer) and str(layer.id or "") == target
            ),
            -1,
        )
        if current_index < 0:
            return "raster_image"
        for index in range(current_index - 1, -1, -1):
            layer = layers[index]
            if isinstance(layer, RasterImageLayer):
                return str(layer.id or "raster_image")
        return "raster_image"

    def _ensure_raster_alignment_context(self, layer_id: str) -> None:
        target = str(layer_id or "").strip()
        if not target:
            return
        if hasattr(self, "setRasterImageLayerVisible"):
            try:
                self.setRasterImageLayerVisible(target, True)
            except Exception:
                pass
        reference_layer_id = self._reference_raster_layer_id(target)
        if reference_layer_id == "raster_image":
            try:
                self.setBaseRasterImageVisible(True)
            except Exception:
                pass
            return
        if hasattr(self, "setRasterImageLayerVisible"):
            try:
                self.setRasterImageLayerVisible(reference_layer_id, True)
            except Exception:
                pass

    def setBaseRasterImageVisible(self, visible: bool) -> bool:
        large_view = getattr(self, "large_image_view", None)
        if large_view is None or not hasattr(large_view, "set_base_raster_visible"):
            return False
        changed = bool(large_view.set_base_raster_visible(bool(visible)))
        if not changed:
            return False
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        self.otherData["base_raster_visible"] = bool(visible)
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def _restoreBaseRasterImageVisibilityFromState(self) -> None:
        large_view = getattr(self, "large_image_view", None)
        if large_view is None or not hasattr(large_view, "set_base_raster_visible"):
            return
        data = getattr(self, "otherData", None)
        if not isinstance(data, dict):
            return
        visible = bool(data.get("base_raster_visible", True))
        try:
            large_view.set_base_raster_visible(visible)
        except Exception:
            pass

    def addRasterImageLayersFromPaths(self, paths: list[str]) -> int:
        normalized_paths = []
        for raw in list(paths or []):
            path = Path(str(raw or "").strip()).expanduser()
            if not path:
                continue
            try:
                resolved = str(path.resolve())
            except Exception:
                resolved = str(path)
            if resolved:
                normalized_paths.append(resolved)
        if not normalized_paths:
            return 0
        records = list(self._raster_layer_records())
        existing_sources = {
            str((record or {}).get("source_path") or "").strip() for record in records
        }
        added = 0
        for source_path in normalized_paths:
            if source_path in existing_sources:
                continue
            existing_sources.add(source_path)
            layer_name = Path(source_path).name
            records.append(
                {
                    "id": f"raster_overlay_{uuid4().hex[:10]}",
                    "name": layer_name,
                    "source_path": source_path,
                    "visible": True,
                    "opacity": 1.0,
                    "page_index": 0,
                    "z_index": 10 + len(records),
                    "tx": 0.0,
                    "ty": 0.0,
                    "sx": 1.0,
                    "sy": 1.0,
                }
            )
            added += 1
        if added <= 0:
            return 0
        self._set_raster_layer_records(records)
        self._restoreRasterImageLayersFromState()
        return int(added)

    def rasterOverlayLayers(self) -> list[RasterImageLayer]:
        large_view = getattr(self, "large_image_view", None)
        if large_view is None or not hasattr(large_view, "raster_overlay_layers_state"):
            return []
        try:
            state_layers = list(large_view.raster_overlay_layers_state() or [])
        except Exception:
            state_layers = []
        layers: list[RasterImageLayer] = []
        ordered_layers = sorted(
            state_layers,
            key=lambda item: float((item or {}).get("z_index", 0.0) or 0.0),
        )
        for index, record in enumerate(ordered_layers):
            layer_id = str(record.get("id") or "")
            if not layer_id:
                continue
            name = str(
                record.get("name")
                or Path(str(record.get("source_path") or "")).name
                or layer_id
            )
            layers.append(
                RasterImageLayer(
                    id=layer_id,
                    name=name,
                    visible=bool(record.get("visible", True)),
                    opacity=float(record.get("opacity", 1.0) or 1.0),
                    locked=False,
                    z_index=int(
                        float(record.get("z_index", 10 + index) or (10 + index))
                    ),
                    transform=AffineTransform(
                        tx=float(record.get("tx", 0.0) or 0.0),
                        ty=float(record.get("ty", 0.0) or 0.0),
                        sx=max(1e-6, float(record.get("sx", 1.0) or 1.0)),
                        sy=max(1e-6, float(record.get("sy", 1.0) or 1.0)),
                    ),
                    backend_page_index=int(record.get("page_index", 0) or 0),
                    channel=None,
                )
            )
        return layers

    def importRasterImageLayers(self, _value: bool = False) -> None:
        image_path = str(getattr(self, "imagePath", "") or "")
        large_view = getattr(self, "large_image_view", None)
        if (
            not image_path
            or large_view is None
            or getattr(self, "_active_image_view", "") != "tiled"
        ):
            self.errorMessage(
                self.tr("TIFF Layers"),
                self.tr(
                    "Open a large TIFF image before importing additional TIFF layers."
                ),
            )
            return
        start_dir = str(Path(image_path).parent)
        filenames, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            self.tr("Import TIFF Layers"),
            start_dir,
            self.tr("TIFF files (*.tif *.tiff *.ome.tif *.ome.tiff);;All files (*)"),
        )
        if not filenames:
            return
        added = self.addRasterImageLayersFromPaths(list(filenames or []))
        if added <= 0:
            return
        post_window_status(self, self.tr("Loaded %d TIFF layer(s)") % int(added))

    def setRasterImageLayerVisible(self, layer_id: str, visible: bool) -> bool:
        layer_id = str(layer_id or "").strip()
        if not layer_id:
            return False
        large_view = getattr(self, "large_image_view", None)
        if large_view is None or not hasattr(
            large_view, "set_raster_overlay_layer_visible"
        ):
            return False
        changed = bool(
            large_view.set_raster_overlay_layer_visible(layer_id, bool(visible))
        )
        if not changed:
            return False
        records = list(self._raster_layer_records())
        updated = False
        for record in records:
            if str((record or {}).get("id") or "") != layer_id:
                continue
            record["visible"] = bool(visible)
            updated = True
            break
        if updated:
            self._set_raster_layer_records(records)
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def setRasterImageLayerOpacity(self, layer_id: str, opacity: float) -> bool:
        layer_id = str(layer_id or "").strip()
        if not layer_id:
            return False
        self._ensure_raster_alignment_context(layer_id)
        large_view = getattr(self, "large_image_view", None)
        if large_view is None or not hasattr(
            large_view, "set_raster_overlay_layer_opacity"
        ):
            return False
        normalized = max(0.0, min(1.0, float(opacity)))
        changed = bool(
            large_view.set_raster_overlay_layer_opacity(layer_id, float(normalized))
        )
        if not changed:
            return False
        records = list(self._raster_layer_records())
        updated = False
        for record in records:
            if str((record or {}).get("id") or "") != layer_id:
                continue
            record["opacity"] = float(normalized)
            updated = True
            break
        if updated:
            self._set_raster_layer_records(records)
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def setRasterImageLayerTransform(
        self,
        layer_id: str,
        *,
        tx: float | None = None,
        ty: float | None = None,
        sx: float | None = None,
        sy: float | None = None,
    ) -> bool:
        layer_id = str(layer_id or "").strip()
        if not layer_id:
            return False
        self._ensure_raster_alignment_context(layer_id)
        records = list(self._raster_layer_records())
        target_record: dict | None = None
        for record in records:
            if str((record or {}).get("id") or "") != layer_id:
                continue
            target_record = record
            break
        if target_record is None:
            return False

        next_tx = float(target_record.get("tx", 0.0) if tx is None else tx)
        next_ty = float(target_record.get("ty", 0.0) if ty is None else ty)
        next_sx = max(1e-6, float(target_record.get("sx", 1.0) if sx is None else sx))
        next_sy = max(1e-6, float(target_record.get("sy", 1.0) if sy is None else sy))
        if (
            abs(float(target_record.get("tx", 0.0) or 0.0) - next_tx) < 1e-9
            and abs(float(target_record.get("ty", 0.0) or 0.0) - next_ty) < 1e-9
            and abs(float(target_record.get("sx", 1.0) or 1.0) - next_sx) < 1e-9
            and abs(float(target_record.get("sy", 1.0) or 1.0) - next_sy) < 1e-9
        ):
            return False

        target_record["tx"] = float(next_tx)
        target_record["ty"] = float(next_ty)
        target_record["sx"] = float(next_sx)
        target_record["sy"] = float(next_sy)

        large_view = getattr(self, "large_image_view", None)
        if large_view is not None and hasattr(
            large_view, "set_raster_overlay_layer_transform"
        ):
            changed = bool(
                large_view.set_raster_overlay_layer_transform(
                    layer_id,
                    tx=float(next_tx),
                    ty=float(next_ty),
                    sx=float(next_sx),
                    sy=float(next_sy),
                )
            )
            if not changed:
                return False

        self._set_raster_layer_records(records)
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def setRasterImageLayerPageIndex(self, layer_id: str, page_index: int) -> bool:
        layer_id = str(layer_id or "").strip()
        if not layer_id:
            return False
        records = list(self._raster_layer_records())
        target_record: dict | None = None
        for record in records:
            if str((record or {}).get("id") or "") != layer_id:
                continue
            target_record = record
            break
        if target_record is None:
            return False
        source_path = Path(str(target_record.get("source_path") or "")).expanduser()
        if not source_path.exists():
            return False
        normalized_page = max(0, int(page_index))
        try:
            backend = open_large_image(source_path)
            page_count = int(getattr(backend, "get_page_count", lambda: 1)() or 1)
            normalized_page = max(0, min(normalized_page, max(0, page_count - 1)))
        except Exception:
            pass
        current_page = int(target_record.get("page_index", 0) or 0)
        if current_page == normalized_page:
            return False
        target_record["page_index"] = int(normalized_page)
        self._set_raster_layer_records(records)
        self._restoreRasterImageLayersFromState()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def translateRasterImageLayer(
        self, layer_id: str, dx: float = 0.0, dy: float = 0.0
    ) -> bool:
        layer_id = str(layer_id or "").strip()
        if not layer_id:
            return False
        records = list(self._raster_layer_records())
        target_record: dict | None = None
        for record in records:
            if str((record or {}).get("id") or "") == layer_id:
                target_record = record
                break
        if target_record is None:
            return False
        next_tx = float(target_record.get("tx", 0.0) or 0.0) + float(dx)
        next_ty = float(target_record.get("ty", 0.0) or 0.0) + float(dy)
        return bool(
            self.setRasterImageLayerTransform(
                layer_id,
                tx=next_tx,
                ty=next_ty,
                sx=float(target_record.get("sx", 1.0) or 1.0),
                sy=float(target_record.get("sy", 1.0) or 1.0),
            )
        )

    def moveRasterImageLayer(self, layer_id: str, direction: int) -> bool:
        layer_id = str(layer_id or "").strip()
        step = int(direction)
        if not layer_id or step not in {-1, 1}:
            return False
        all_layers = list(getattr(self, "viewerLayerModels", lambda: [])() or [])
        if not all_layers:
            return False
        ordered = sorted(
            all_layers,
            key=lambda layer: float(getattr(layer, "z_index", 0.0) or 0.0),
        )
        current_index = next(
            (idx for idx, layer in enumerate(ordered) if str(layer.id) == layer_id),
            -1,
        )
        if current_index < 0:
            return False
        target_index = current_index + (1 if step < 0 else -1)
        if target_index < 0 or target_index >= len(ordered):
            return False
        target_layer = ordered[target_index]
        target_z = float(getattr(target_layer, "z_index", 0.0) or 0.0)
        base_z = self._base_raster_layer_z_index()
        if step < 0:
            # Move "up" toward the front of the stack: raise z-order.
            next_index = min(len(ordered) - 1, target_index + 1)
            if next_index == target_index:
                new_z = target_z + 1.0
            else:
                next_z = float(getattr(ordered[next_index], "z_index", 0.0) or 0.0)
                new_z = max(target_z + 0.1, next_z + 0.1)
        else:
            # Move "down" toward the back of the stack: lower z-order.
            prev_index = max(0, target_index - 1)
            if prev_index == target_index:
                new_z = target_z - 1.0
            else:
                prev_z = float(getattr(ordered[prev_index], "z_index", 0.0) or 0.0)
                new_z = min(target_z - 0.1, prev_z - 0.1)
            new_z = max(base_z + 0.1, new_z)
        records = list(self._raster_layer_records())
        updated = False
        for record in records:
            if str((record or {}).get("id") or "") != layer_id:
                continue
            record["z_index"] = float(new_z)
            updated = True
            break
        if not updated:
            return False
        self._set_raster_layer_records(records)
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None and hasattr(
            large_view, "set_raster_overlay_layer_z_index"
        ):
            changed = bool(large_view.set_raster_overlay_layer_z_index(layer_id, new_z))
            if not changed:
                return False
        else:
            self._restoreRasterImageLayersFromState()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def moveRasterImageLayerToEdge(self, layer_id: str, where: str) -> bool:
        layer_id = str(layer_id or "").strip()
        edge = str(where or "").strip().lower()
        if not layer_id or edge not in {"top", "bottom"}:
            return False
        all_layers = list(getattr(self, "viewerLayerModels", lambda: [])() or [])
        if not all_layers:
            return False
        ordered = sorted(
            all_layers,
            key=lambda layer: float(getattr(layer, "z_index", 0.0) or 0.0),
        )
        current_index = next(
            (idx for idx, layer in enumerate(ordered) if str(layer.id) == layer_id),
            -1,
        )
        if current_index < 0:
            return False
        if edge == "top":
            target_z = float(getattr(ordered[-1], "z_index", 0.0) or 0.0) + 1.0
        else:
            target_z = max(
                self._base_raster_layer_z_index() + 0.1,
                float(getattr(ordered[0], "z_index", 0.0) or 0.0) - 1.0,
            )
        records = list(self._raster_layer_records())
        updated = False
        for record in records:
            if str((record or {}).get("id") or "") != layer_id:
                continue
            record["z_index"] = float(target_z)
            updated = True
            break
        if not updated:
            return False
        self._set_raster_layer_records(records)
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None and hasattr(
            large_view, "set_raster_overlay_layer_z_index"
        ):
            changed = bool(
                large_view.set_raster_overlay_layer_z_index(layer_id, target_z)
            )
            if not changed:
                return False
        else:
            self._restoreRasterImageLayersFromState()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def renameRasterImageLayer(self, layer_id: str, name: str) -> bool:
        layer_id = str(layer_id or "").strip()
        normalized = str(name or "").strip()
        if not layer_id or not normalized:
            return False
        records = list(self._raster_layer_records())
        updated = False
        for record in records:
            if str((record or {}).get("id") or "") != layer_id:
                continue
            if str((record or {}).get("name") or "") == normalized:
                return False
            record["name"] = normalized
            updated = True
            break
        if not updated:
            return False
        self._set_raster_layer_records(records)
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None and hasattr(
            large_view, "set_raster_overlay_layer_name"
        ):
            large_view.set_raster_overlay_layer_name(layer_id, normalized)
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def removeRasterImageLayer(self, layer_id: str) -> bool:
        layer_id = str(layer_id or "").strip()
        if not layer_id:
            return False
        records = list(self._raster_layer_records())
        kept = [
            record
            for record in records
            if str((record or {}).get("id") or "") != layer_id
        ]
        if len(kept) == len(records):
            return False
        for idx, item in enumerate(
            sorted(
                kept,
                key=lambda record: float((record or {}).get("z_index", 0.0) or 0.0),
            )
        ):
            item["z_index"] = 10 + idx
        self._set_raster_layer_records(kept)
        large_view = getattr(self, "large_image_view", None)
        if large_view is not None and hasattr(
            large_view, "remove_raster_overlay_layer"
        ):
            large_view.remove_raster_overlay_layer(layer_id)
        else:
            self._restoreRasterImageLayersFromState()
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()
        return True

    def _restoreRasterImageLayersFromState(self) -> None:
        large_view = getattr(self, "large_image_view", None)
        if large_view is None:
            return
        if getattr(self, "_active_image_view", "") != "tiled":
            if hasattr(large_view, "clear_raster_overlay_layers"):
                large_view.clear_raster_overlay_layers()
            return
        records = list(self._raster_layer_records())
        if not records:
            if hasattr(large_view, "clear_raster_overlay_layers"):
                large_view.clear_raster_overlay_layers()
            return
        layer_payloads: list[dict] = []
        base_backend = getattr(self, "large_image_backend", None)
        base_page = int(getattr(base_backend, "get_current_page", lambda: 0)() or 0)
        base_page_count = int(getattr(base_backend, "get_page_count", lambda: 1)() or 1)
        valid_records: list[dict] = []
        ordered_records = sorted(
            records,
            key=lambda item: float((item or {}).get("z_index", 0.0) or 0.0),
        )
        for index, record in enumerate(ordered_records):
            source_path = str((record or {}).get("source_path") or "").strip()
            if not source_path:
                continue
            path = Path(source_path).expanduser()
            if not path.exists():
                continue
            try:
                backend = open_large_image(path)
            except Exception:
                continue
            target_page = int(record.get("page_index", 0) or 0)
            try:
                overlay_page_count = int(
                    getattr(backend, "get_page_count", lambda: 1)() or 1
                )
            except Exception:
                overlay_page_count = 1
            if overlay_page_count == base_page_count and overlay_page_count > 1:
                target_page = base_page
            target_page = max(0, min(target_page, max(0, overlay_page_count - 1)))
            try:
                backend.set_page(target_page)
            except Exception:
                target_page = int(
                    getattr(backend, "get_current_page", lambda: 0)() or 0
                )
            layer_id = str(record.get("id") or "")
            if not layer_id:
                layer_id = f"raster_overlay_{uuid4().hex[:10]}"
            layer_name = str(record.get("name") or path.name or layer_id)
            opacity = max(0.0, min(1.0, float(record.get("opacity", 1.0) or 1.0)))
            visible = bool(record.get("visible", True))
            z_index = float(record.get("z_index", 10 + index) or (10 + index))
            layer_payloads.append(
                {
                    "id": layer_id,
                    "name": layer_name,
                    "source_path": str(path),
                    "backend": backend,
                    "visible": visible,
                    "opacity": opacity,
                    "z_index": z_index,
                    "page_index": int(target_page),
                    "tx": float(record.get("tx", 0.0) or 0.0),
                    "ty": float(record.get("ty", 0.0) or 0.0),
                    "sx": max(1e-6, float(record.get("sx", 1.0) or 1.0)),
                    "sy": max(1e-6, float(record.get("sy", 1.0) or 1.0)),
                }
            )
            valid_records.append(
                {
                    "id": layer_id,
                    "name": layer_name,
                    "source_path": str(path),
                    "visible": visible,
                    "opacity": opacity,
                    "z_index": z_index,
                    "page_index": int(target_page),
                    "tx": float(record.get("tx", 0.0) or 0.0),
                    "ty": float(record.get("ty", 0.0) or 0.0),
                    "sx": max(1e-6, float(record.get("sx", 1.0) or 1.0)),
                    "sy": max(1e-6, float(record.get("sy", 1.0) or 1.0)),
                }
            )
        if hasattr(large_view, "set_raster_overlay_layers"):
            large_view.set_raster_overlay_layers(layer_payloads)
        self._set_raster_layer_records(valid_records)
        if hasattr(self, "_syncLargeImageDocument"):
            self._syncLargeImageDocument()

    def _syncRasterImageLayerPages(self) -> None:
        large_view = getattr(self, "large_image_view", None)
        base_backend = getattr(self, "large_image_backend", None)
        if (
            large_view is None
            or base_backend is None
            or not hasattr(large_view, "sync_raster_overlay_pages")
        ):
            return
        base_page = int(getattr(base_backend, "get_current_page", lambda: 0)() or 0)
        base_page_count = int(getattr(base_backend, "get_page_count", lambda: 1)() or 1)
        large_view.sync_raster_overlay_pages(
            base_page=int(base_page),
            base_page_count=int(base_page_count),
        )
        state = (
            list(large_view.raster_overlay_layers_state() or [])
            if hasattr(large_view, "raster_overlay_layers_state")
            else []
        )
        if state:
            self._set_raster_layer_records(state)

    def _raster_layer_records(self) -> list[dict]:
        data = getattr(self, "otherData", None)
        if not isinstance(data, dict):
            return []
        return list(data.get("raster_image_layers") or [])

    def _set_raster_layer_records(self, records: list[dict]) -> None:
        if not isinstance(getattr(self, "otherData", None), dict):
            self.otherData = {}
        self.otherData["raster_image_layers"] = list(records or [])
