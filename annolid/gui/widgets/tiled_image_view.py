from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets

from annolid.large_image.gui.status_overlay import LargeImageStatusOverlay
from annolid.gui.label_image_overlay import colorize_label_image, label_entry_text
from annolid.gui.large_image_modes import (
    is_tile_native_create_mode,
    large_image_draw_mode_label,
)
from annolid.gui.shape import Shape
from annolid.gui.tile_scheduler import TileRenderPlan, TileRequestScheduler
from annolid.io.large_image import LargeImageBackend
from annolid.io.large_image.common import array_to_qimage


@dataclass(frozen=True)
class TileKey:
    level: int
    tx: int
    ty: int


class TileCache:
    def __init__(self, max_items: int = 128):
        self.max_items = max(1, int(max_items))
        self._items: OrderedDict[TileKey, QtGui.QImage] = OrderedDict()

    def get(self, key: TileKey):
        image = self._items.get(key)
        if image is not None:
            self._items.move_to_end(key)
        return image

    def put(self, key: TileKey, image):
        self._items[key] = image
        self._items.move_to_end(key)
        self.trim(self.max_items)

    def trim(self, max_items: int | None = None):
        limit = self.max_items if max_items is None else max(1, int(max_items))
        while len(self._items) > limit:
            self._items.popitem(last=False)

    def __len__(self) -> int:
        return len(self._items)


class _LandmarkPairItem(QtWidgets.QGraphicsLineItem):
    def __init__(self, pair_id: str, x1: float, y1: float, x2: float, y2: float):
        super().__init__(x1, y1, x2, y2)
        self.pair_id = str(pair_id or "")

    @staticmethod
    def pair_pen(*, selected: bool = False) -> QtGui.QPen:
        color = (
            QtGui.QColor(255, 140, 0, 240)
            if selected
            else QtGui.QColor(255, 215, 0, 220)
        )
        pen = QtGui.QPen(color)
        pen.setWidthF(3.0 if selected else 2.0)
        pen.setStyle(QtCore.Qt.DashLine)
        return pen

    def set_selected(self, selected: bool) -> None:
        self.setPen(self.pair_pen(selected=selected))
        self.setZValue(95.0 if selected else 90.0)


class _LandmarkPairEndpointItem(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, pair_id: str, center: QtCore.QPointF, radius: float = 7.0):
        super().__init__(
            center.x() - radius,
            center.y() - radius,
            radius * 2.0,
            radius * 2.0,
        )
        self.pair_id = str(pair_id or "")
        self._center = QtCore.QPointF(center)
        self._radius = float(radius)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        fill = (
            QtGui.QColor(255, 140, 0, 180)
            if selected
            else QtGui.QColor(255, 215, 0, 110)
        )
        outline = QtGui.QColor(255, 255, 255, 230)
        pen = QtGui.QPen(outline)
        pen.setWidthF(2.0 if selected else 1.5)
        self.setPen(pen)
        self.setBrush(QtGui.QBrush(fill))
        self.setZValue(96.0 if selected else 91.0)


class _ShapeGraphicsItem(QtWidgets.QGraphicsItem):
    def __init__(
        self,
        shape,
        *,
        content_size: tuple[int, int],
        current_scale: float,
    ):
        super().__init__()
        self._ann_shape = shape
        self.content_size = tuple(content_size or (0, 0))
        self.current_scale = max(0.01, float(current_scale))
        self.setAcceptedMouseButtons(QtCore.Qt.NoButton)

    def set_current_scale(self, current_scale: float) -> None:
        normalized = max(0.01, float(current_scale))
        if abs(normalized - self.current_scale) < 1e-6:
            return
        self.prepareGeometryChange()
        self.current_scale = normalized
        self.update()

    def _visual_metrics(self) -> float:
        width, height = self.content_size
        diagonal = math.hypot(float(width or 0), float(height or 0))
        diagonal_factor = min(
            1.0, max(0.0, (diagonal - 2048.0) / max(1.0, 32768.0 - 2048.0))
        )
        scale = max(0.01, self.current_scale)
        point_pixels = 6.0 + (1.25 * diagonal_factor)
        if scale < 0.2:
            point_pixels += 2.0
        elif scale < 0.5:
            point_pixels += 1.0
        elif scale > 8.0:
            point_pixels -= 3.25
        elif scale > 5.0:
            point_pixels -= 2.5
        elif scale > 4.0:
            point_pixels -= 2.0
        elif scale > 2.0:
            point_pixels -= 1.5
        elif scale > 1.0:
            point_pixels -= 0.75
        point_pixels = min(9.0, max(2.0, point_pixels))
        return point_pixels

    def _effective_highlight_settings(self) -> dict:
        base = dict(getattr(self._ann_shape, "_highlightSettings", {}) or {})
        if not base:
            return base
        scale = max(0.01, self.current_scale)
        if scale >= 8.0:
            near_scale = 1.4
            move_scale = 1.15
        elif scale >= 4.0:
            near_scale = 1.7
            move_scale = 1.2
        elif scale >= 2.0:
            near_scale = 2.2
            move_scale = 1.3
        else:
            near_scale = 2.8
            move_scale = 1.4
        if getattr(Shape, "NEAR_VERTEX", None) in base:
            _, near_shape = base[Shape.NEAR_VERTEX]
            base[Shape.NEAR_VERTEX] = (near_scale, near_shape)
        if getattr(Shape, "MOVE_VERTEX", None) in base:
            _, move_shape = base[Shape.MOVE_VERTEX]
            base[Shape.MOVE_VERTEX] = (move_scale, move_shape)
        return base

    def _effective_vertex_render_overrides(self) -> dict:
        scale = max(0.01, self.current_scale)
        if scale >= 8.0:
            return {
                "highlight_settings": self._effective_highlight_settings(),
                "glow_scale": 1.2,
                "halo_scale": 0.7,
                "glow_alpha_mult": 0.35,
                "halo_alpha_mult": 0.3,
                "inner_alpha_mult": 0.85,
                "highlight_lighter": 112,
            }
        if scale >= 4.0:
            return {
                "highlight_settings": self._effective_highlight_settings(),
                "glow_scale": 1.3,
                "halo_scale": 0.8,
                "glow_alpha_mult": 0.45,
                "halo_alpha_mult": 0.38,
                "inner_alpha_mult": 0.9,
                "highlight_lighter": 116,
            }
        if scale >= 2.0:
            return {
                "highlight_settings": self._effective_highlight_settings(),
                "glow_scale": 1.45,
                "halo_scale": 0.92,
                "glow_alpha_mult": 0.6,
                "halo_alpha_mult": 0.5,
                "inner_alpha_mult": 0.95,
                "highlight_lighter": 122,
            }
        return {"highlight_settings": self._effective_highlight_settings()}

    def boundingRect(self) -> QtCore.QRectF:
        rect = self._ann_shape.boundingRect()
        if rect is None or not rect.isValid():
            points = list(getattr(self._ann_shape, "points", []) or [])
            if not points:
                return QtCore.QRectF()
            point = points[0]
            rect = QtCore.QRectF(float(point.x()), float(point.y()), 0.0, 0.0)
        margin = max(8.0, self._visual_metrics() * 2.75)
        return rect.adjusted(-margin, -margin, margin, margin)

    def shape(self) -> QtGui.QPainterPath:
        path = QtGui.QPainterPath()
        rect = self.boundingRect()
        if rect.isValid():
            path.addRect(rect)
        return path

    def contains(self, point: QtCore.QPointF) -> bool:
        try:
            return self.shape().contains(point)
        except Exception:
            return False

    def paint(self, painter, option, widget=None) -> None:
        _ = option
        _ = widget
        other = dict(getattr(self._ann_shape, "other_data", {}) or {})
        width, height = self.content_size
        original_line = getattr(self._ann_shape, "line_color", None)
        original_fill = getattr(self._ann_shape, "fill_color", None)
        original_fill_flag = bool(getattr(self._ann_shape, "fill", False))
        original_scale = getattr(self._ann_shape, "scale", Shape.scale)
        original_point_size = getattr(self._ann_shape, "point_size", Shape.point_size)
        had_vertex_render_overrides = hasattr(
            self._ann_shape, "_vertex_render_overrides"
        )
        original_vertex_render_overrides = dict(
            getattr(self._ann_shape, "_vertex_render_overrides", {}) or {}
        )
        try:
            point_pixels = self._visual_metrics()
            stroke = QtGui.QColor(str(other.get("overlay_stroke") or ""))
            fill = QtGui.QColor(str(other.get("overlay_fill") or ""))
            if stroke.isValid():
                stroke.setAlpha(max(stroke.alpha(), 220))
                self._ann_shape.line_color = stroke
            if fill.isValid():
                fill.setAlpha(max(fill.alpha(), 40))
                self._ann_shape.fill_color = fill
                self._ann_shape.fill = True
            elif getattr(self._ann_shape, "selected", False):
                self._ann_shape.fill = True
            self._ann_shape.scale = self.current_scale
            self._ann_shape.point_size = point_pixels
            self._ann_shape._vertex_render_overrides = (
                self._effective_vertex_render_overrides()
            )
            self._ann_shape.paint(painter, width or None, height or None)
        finally:
            self._ann_shape.line_color = original_line
            self._ann_shape.fill_color = original_fill
            self._ann_shape.fill = original_fill_flag
            self._ann_shape.scale = original_scale
            self._ann_shape.point_size = original_point_size
            if had_vertex_render_overrides:
                self._ann_shape._vertex_render_overrides = (
                    original_vertex_render_overrides
                )
            elif hasattr(self._ann_shape, "_vertex_render_overrides"):
                delattr(self._ann_shape, "_vertex_render_overrides")


class TiledImageView(QtWidgets.QGraphicsView):
    """Foundation widget for large-image tile rendering."""

    overlayLandmarkPairSelected = QtCore.Signal(str)
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    newShape = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)

    CREATE, EDIT = 0, 1

    def __init__(self, parent=None, tile_size: int = 512):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.tile_size = max(128, int(tile_size))
        self.tile_cache = TileCache()
        self._tile_scheduler = TileRequestScheduler(
            cache_get=self.tile_cache.get,
            cache_put=self.tile_cache.put,
            load_tile=self._load_raster_tile,
            async_load=True,
        )
        self.backend: LargeImageBackend | None = None
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._preview_item = QtWidgets.QGraphicsPathItem()
        self._preview_item.setZValue(130.0)
        self._scene.addItem(self._preview_item)
        self._preview_vertices_item = QtWidgets.QGraphicsPathItem()
        self._preview_vertices_item.setZValue(131.0)
        self._scene.addItem(self._preview_vertices_item)
        self._preview_close_item = QtWidgets.QGraphicsPathItem()
        self._preview_close_item.setZValue(132.0)
        self._scene.addItem(self._preview_close_item)
        self._tile_items: dict[TileKey, QtWidgets.QGraphicsPixmapItem] = {}
        self._label_tile_items: dict[TileKey, QtWidgets.QGraphicsPixmapItem] = {}
        self._overlay_items: list[QtWidgets.QGraphicsItem] = []
        self._pair_items: list[QtWidgets.QGraphicsItem] = []
        self._pair_endpoint_items: list[QtWidgets.QGraphicsItem] = []
        self._selected_overlay_landmark_pair_id: str | None = None
        self._last_raster_level: int = 0
        self._last_visible_tile_count: int = 0
        self._shapes = []
        self.selectedShapes = []
        self._active_shape = None
        self._active_vertex_index: int | None = None
        self._dragging_shape = False
        self._shape_moved_during_drag = False
        self._last_scene_pos: QtCore.QPointF | None = None
        self._host_window = None
        self._content_size: tuple[int, int] = (0, 0)
        self._fit_mode: str = "fit_window"
        self._label_backend: LargeImageBackend | None = None
        self._label_tile_cache = TileCache()
        self._last_label_level: int = 0
        self._last_label_visible_tile_count: int = 0
        self._label_tile_scheduler = TileRequestScheduler(
            cache_get=self._label_tile_cache.get,
            cache_put=self._label_tile_cache.put,
            load_tile=self._load_label_tile,
            async_load=True,
        )
        self._label_value_tile_cache: OrderedDict[tuple[int, int], np.ndarray] = (
            OrderedDict()
        )
        self._label_mapping: dict[int, dict] = {}
        self._label_overlay_opacity: float = 0.45
        self._label_overlay_visible: bool = True
        self._selected_label_value: int | None = None
        self._label_source_path: str | None = None
        self._label_mapping_path: str | None = None
        self._label_page_index: int = 0
        self._label_transform: dict[str, float] = {
            "tx": 0.0,
            "ty": 0.0,
            "sx": 1.0,
            "sy": 1.0,
        }
        self._last_hovered_label_value: int | None = None
        self.mode = self.EDIT
        self.createMode = "polygon"
        self.current = None
        self.epsilon = 10.0
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.setMouseTracking(True)
        self._status_overlay = LargeImageStatusOverlay(self.viewport())
        self._status_overlay.raise_()
        self._tile_result_timer = QtCore.QTimer(self)
        self._tile_result_timer.setInterval(30)
        self._tile_result_timer.timeout.connect(self._poll_pending_tile_results)
        self._current_visible_raster_keys: tuple[TileKey, ...] = ()
        self._current_visible_label_keys: tuple[TileKey, ...] = ()

    def set_host_window(self, window) -> None:
        self._host_window = window

    def _notify_host_large_image_document_changed(self) -> None:
        host = self._host_window
        if host is None:
            return
        sync = getattr(host, "_syncLargeImageDocument", None)
        if callable(sync):
            try:
                sync()
            except Exception:
                pass
        self._refresh_status_overlay()

    def viewport_state(self) -> dict[str, float | int | str]:
        center = self.mapToScene(self.viewport().rect().center())
        zoom_percent = int(round(self.current_scale() * 100.0))
        return {
            "zoom_percent": max(1, zoom_percent),
            "center_x": float(center.x()),
            "center_y": float(center.y()),
            "fit_mode": str(self._fit_mode or "manual"),
        }

    def apply_viewport_state(self, state: dict | None) -> None:
        data = dict(state or {})
        fit_mode = str(data.get("fit_mode", self._fit_mode or "manual") or "manual")
        if fit_mode == "fit_window":
            self.fit_to_window()
        elif fit_mode == "fit_width":
            self.fit_to_width()
        else:
            self.set_zoom_percent(int(data.get("zoom_percent", 100) or 100))
        if "center_x" in data and "center_y" in data:
            self.centerOn(
                float(data.get("center_x", 0.0)), float(data.get("center_y", 0.0))
            )
            self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def set_backend(self, backend: LargeImageBackend) -> None:
        try:
            self._tile_scheduler.shutdown()
        except Exception:
            pass
        self.backend = backend
        self.tile_cache = TileCache()
        self._tile_scheduler = TileRequestScheduler(
            cache_get=self.tile_cache.get,
            cache_put=self.tile_cache.put,
            load_tile=self._load_raster_tile,
        )
        self._clear_tile_items()
        self._clear_label_value_cache()
        thumbnail = backend.get_thumbnail(max_size=2048)
        if isinstance(thumbnail, QtGui.QImage):
            image = thumbnail
        else:
            image = array_to_qimage(thumbnail)
        full_w, full_h = backend.get_level_shape(0)
        self._content_size = (int(full_w), int(full_h))
        pixmap = QtGui.QPixmap.fromImage(image)
        self._pixmap_item.setPixmap(pixmap)
        self._pixmap_item.setOffset(0, 0)
        if pixmap.width() > 0 and pixmap.height() > 0:
            self._pixmap_item.setScale(full_w / pixmap.width())
        self._pixmap_item.setZValue(-100.0)
        self._scene.setSceneRect(0.0, 0.0, float(full_w), float(full_h))
        self.fit_to_window()
        self.refresh_visible_tiles()
        self._refresh_status_overlay()
        self._notify_host_large_image_document_changed()

    def set_label_layer(
        self,
        backend: LargeImageBackend,
        *,
        opacity: float = 0.45,
        mapping: dict[int, dict] | None = None,
        source_path: str | None = None,
        mapping_path: str | None = None,
        visible: bool = True,
        page_index: int | None = None,
        transform: dict | None = None,
    ) -> None:
        try:
            self._label_tile_scheduler.shutdown()
        except Exception:
            pass
        self._label_backend = backend
        self._label_overlay_opacity = max(0.0, min(1.0, float(opacity)))
        self._label_mapping = dict(mapping or {})
        self._label_source_path = str(source_path or "")
        self._label_mapping_path = str(mapping_path or "")
        self._label_overlay_visible = bool(visible)
        self._label_page_index = int(page_index or 0)
        if isinstance(transform, dict):
            self._label_transform = {
                "tx": float(transform.get("tx", 0.0) or 0.0),
                "ty": float(transform.get("ty", 0.0) or 0.0),
                "sx": max(1e-6, float(transform.get("sx", 1.0) or 1.0)),
                "sy": max(1e-6, float(transform.get("sy", 1.0) or 1.0)),
            }
        else:
            self._label_transform = {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0}
        self._label_tile_cache = TileCache()
        self._label_tile_scheduler = TileRequestScheduler(
            cache_get=self._label_tile_cache.get,
            cache_put=self._label_tile_cache.put,
            load_tile=self._load_label_tile,
            async_load=True,
        )
        self._clear_label_value_cache()
        self._clear_label_tile_items()
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def clear_label_layer(self) -> None:
        try:
            self._label_tile_scheduler.shutdown()
        except Exception:
            pass
        self._label_backend = None
        self._label_tile_cache = TileCache()
        self._label_tile_scheduler = TileRequestScheduler(
            cache_get=self._label_tile_cache.get,
            cache_put=self._label_tile_cache.put,
            load_tile=self._load_label_tile,
            async_load=True,
        )
        self._clear_label_value_cache()
        self._label_mapping = {}
        self._label_overlay_visible = True
        self._selected_label_value = None
        self._label_source_path = None
        self._label_mapping_path = None
        self._label_page_index = 0
        self._label_transform = {"tx": 0.0, "ty": 0.0, "sx": 1.0, "sy": 1.0}
        self._last_hovered_label_value = None
        self._last_label_level = 0
        self._last_label_visible_tile_count = 0
        self._clear_label_tile_items()
        self._refresh_status_overlay()
        self._notify_host_large_image_document_changed()

    def label_layer_backend(self) -> LargeImageBackend | None:
        return self._label_backend

    def set_label_mapping(
        self, mapping: dict[int, dict] | None, *, mapping_path: str | None = None
    ) -> None:
        self._label_mapping = dict(mapping or {})
        self._label_mapping_path = str(mapping_path or "")
        self._notify_host_large_image_document_changed()

    def set_label_layer_opacity(self, opacity: float) -> None:
        normalized = max(0.0, min(1.0, float(opacity)))
        if abs(normalized - self._label_overlay_opacity) < 1e-6:
            return
        self._label_overlay_opacity = normalized
        self._label_tile_cache = TileCache()
        self._clear_label_tile_items()
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def set_label_layer_visible(self, visible: bool) -> None:
        visible_flag = bool(visible)
        if visible_flag == self._label_overlay_visible:
            return
        self._label_overlay_visible = visible_flag
        if not visible_flag:
            self._clear_label_tile_items()
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def label_layer_visible(self) -> bool:
        return bool(self._label_overlay_visible)

    def set_selected_label_value(self, label_value: int | None) -> None:
        normalized = None
        if label_value is not None:
            value = int(label_value)
            normalized = value if value > 0 else None
        if normalized == self._selected_label_value:
            return
        self._selected_label_value = normalized
        self._label_tile_cache = TileCache()
        self._clear_label_tile_items()
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def selected_label_value(self) -> int | None:
        return self._selected_label_value

    def label_overlay_state(self) -> dict:
        return {
            "source_path": str(self._label_source_path or ""),
            "mapping_path": str(self._label_mapping_path or ""),
            "opacity": float(self._label_overlay_opacity),
            "visible": bool(self._label_overlay_visible),
            "selected_label": self._selected_label_value,
            "page_index": int(self._label_page_index),
            "transform": dict(self._label_transform),
        }

    def content_size(self) -> tuple[int, int]:
        return self._content_size

    def clear(self) -> None:
        try:
            self._tile_scheduler.shutdown()
        except Exception:
            pass
        try:
            self._label_tile_scheduler.shutdown()
        except Exception:
            pass
        self._tile_result_timer.stop()
        self.backend = None
        self._content_size = (0, 0)
        self.tile_cache = TileCache()
        self._tile_scheduler = TileRequestScheduler(
            cache_get=self.tile_cache.get,
            cache_put=self.tile_cache.put,
            load_tile=self._load_raster_tile,
            async_load=True,
        )
        self._selected_overlay_landmark_pair_id = None
        self._shapes = []
        self.selectedShapes = []
        self._active_shape = None
        self._active_vertex_index = None
        self._dragging_shape = False
        self._shape_moved_during_drag = False
        self._last_scene_pos = None
        self.current = None
        self.mode = self.EDIT
        self._pixmap_item.setPixmap(QtGui.QPixmap())
        self._preview_item.setPath(QtGui.QPainterPath())
        self._preview_vertices_item.setPath(QtGui.QPainterPath())
        self._preview_close_item.setPath(QtGui.QPainterPath())
        self._clear_tile_items()
        self.clear_label_layer()
        self.set_shapes([])
        self._scene.setSceneRect(QtCore.QRectF())
        self.resetTransform()
        self._last_raster_level = 0
        self._last_visible_tile_count = 0
        self._last_label_level = 0
        self._last_label_visible_tile_count = 0
        self._current_visible_raster_keys = ()
        self._current_visible_label_keys = ()
        self._refresh_status_overlay()
        self._notify_host_large_image_document_changed()

    def drawing(self) -> bool:
        return self.mode == self.CREATE

    def editing(self) -> bool:
        return self.mode == self.EDIT

    def setEditing(self, value=True) -> None:
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            self.current = None
            self._preview_item.setPath(QtGui.QPainterPath())
            self._preview_vertices_item.setPath(QtGui.QPainterPath())
            self._preview_close_item.setPath(QtGui.QPainterPath())
            self.drawingPolygon.emit(False)
        self._notify_host_large_image_document_changed()

    def _supports_create_mode(self, create_mode: str) -> bool:
        return is_tile_native_create_mode(create_mode)

    def _clear_tile_items(self) -> None:
        for item in self._tile_items.values():
            self._scene.removeItem(item)
        self._tile_items.clear()

    def _clear_label_tile_items(self) -> None:
        for item in self._label_tile_items.values():
            self._scene.removeItem(item)
        self._label_tile_items.clear()

    def _clear_label_value_cache(self) -> None:
        self._label_value_tile_cache.clear()

    def set_shapes(self, shapes) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []
        for item in self._pair_items:
            self._scene.removeItem(item)
        self._pair_items = []
        for item in self._pair_endpoint_items:
            self._scene.removeItem(item)
        self._pair_endpoint_items = []
        shapes_list = shapes if isinstance(shapes, list) else list(shapes or [])
        self._shapes = shapes_list
        selected_ids = {id(shape) for shape in (self.selectedShapes or [])}
        self.selectedShapes = [
            shape for shape in shapes_list if id(shape) in selected_ids
        ]
        for item in self._make_pair_overlay_items(shapes_list):
            self._scene.addItem(item)
            self._pair_items.append(item)
        for item in self._make_pair_endpoint_items(shapes_list):
            self._scene.addItem(item)
            self._pair_endpoint_items.append(item)
        for shape in shapes_list:
            item = self._make_overlay_item(shape)
            if item is not None:
                self._scene.addItem(item)
                self._overlay_items.append(item)
        self._refresh_overlay_render_metrics()
        self._notify_host_large_image_document_changed()

    def _refresh_overlay_render_metrics(self) -> None:
        scale = self.current_scale()
        for item in self._overlay_items:
            if isinstance(item, _ShapeGraphicsItem):
                item.set_current_scale(scale)

    def set_selected_landmark_pair(self, pair_id: str | None) -> None:
        self._selected_overlay_landmark_pair_id = str(pair_id or "") or None
        for item in self._pair_items:
            if isinstance(item, _LandmarkPairItem):
                item.set_selected(
                    item.pair_id == self._selected_overlay_landmark_pair_id
                )
        for item in self._pair_endpoint_items:
            if isinstance(item, _LandmarkPairEndpointItem):
                item.set_selected(
                    item.pair_id == self._selected_overlay_landmark_pair_id
                )

    def _apply_selection(self, shapes, *, emit_signal: bool = False) -> None:
        selected = list(shapes or [])
        selected_ids = {id(shape) for shape in selected}
        self.selectedShapes = [
            shape for shape in self._shapes if id(shape) in selected_ids
        ]
        for shape in self._shapes:
            try:
                shape.selected = id(shape) in selected_ids
            except Exception:
                pass
            try:
                if id(shape) not in selected_ids:
                    shape.highlightClear()
            except Exception:
                pass
        self.set_shapes(self._shapes)
        if emit_signal:
            self.selectionChanged.emit(list(self.selectedShapes))
        self._notify_host_large_image_document_changed()

    def _scene_pos_from_event(self, event) -> QtCore.QPointF:
        pos = event.pos() if hasattr(event, "pos") else event.position().toPoint()
        return self.mapToScene(pos)

    def _scene_contains(self, point: QtCore.QPointF) -> bool:
        return self.sceneRect().contains(point)

    def _close_enough(self, p1: QtCore.QPointF, p2: QtCore.QPointF) -> bool:
        return QtCore.QLineF(p1, p2).length() < (
            self.epsilon / max(self.current_scale(), 0.01)
        )

    def _store_canvas_shapes_backup(self) -> None:
        host = self.window()
        canvas = getattr(host, "canvas", None)
        if canvas is not None and hasattr(canvas, "storeShapes"):
            try:
                canvas.storeShapes()
            except Exception:
                pass

    def _update_drawing_preview(self, scene_pos: QtCore.QPointF | None = None) -> None:
        path = QtGui.QPainterPath()
        if self.current is None:
            self._preview_item.setPath(path)
            self._preview_vertices_item.setPath(QtGui.QPainterPath())
            self._preview_close_item.setPath(QtGui.QPainterPath())
            return
        points = list(getattr(self.current, "points", []) or [])
        if not points:
            self._preview_item.setPath(path)
            self._preview_vertices_item.setPath(QtGui.QPainterPath())
            self._preview_close_item.setPath(QtGui.QPainterPath())
            return
        mode = str(self.createMode or "").lower()
        raw_target = (
            QtCore.QPointF(scene_pos)
            if scene_pos is not None
            else QtCore.QPointF(points[-1])
        )
        close_target = self._polygon_close_target(raw_target)
        scene_target = (
            QtCore.QPointF(close_target) if close_target is not None else raw_target
        )
        path.moveTo(points[0])
        if mode == "point":
            radius = 4.5
            path.addEllipse(points[0], radius, radius)
        elif mode in {"line", "rectangle", "circle"}:
            if len(points) == 1:
                path.lineTo(scene_target)
            else:
                if mode == "line":
                    path.lineTo(points[1])
                elif mode == "rectangle":
                    rect = QtCore.QRectF(points[0], points[1]).normalized()
                    path.addRect(rect)
                else:
                    dx = float(points[1].x()) - float(points[0].x())
                    dy = float(points[1].y()) - float(points[0].y())
                    radius = max(0.0, (dx * dx + dy * dy) ** 0.5)
                    path.addEllipse(points[0], radius, radius)
        elif mode == "linestrip":
            for point in points[1:]:
                path.lineTo(point)
            path.lineTo(scene_target)
        else:
            for point in points[1:]:
                path.lineTo(point)
            path.lineTo(scene_target)
        preview_color = QtGui.QColor(0, 255, 255, 235)
        pen = QtGui.QPen(preview_color)
        pen.setWidthF(max(2.0, 2.6 / max(self.current_scale(), 0.25)))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        self._preview_item.setPen(pen)
        self._preview_item.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        self._preview_item.setPath(path)
        vertex_path = QtGui.QPainterPath()
        vertex_radius = max(2.0, min(5.0, 5.0 / max(self.current_scale(), 0.5)))
        for point in points:
            vertex_path.addEllipse(point, vertex_radius, vertex_radius)
        if mode in {"polygon", "linestrip"} and scene_pos is not None:
            vertex_path.addEllipse(
                scene_target, vertex_radius * 0.75, vertex_radius * 0.75
            )
        vertex_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 245))
        vertex_pen.setWidthF(max(1.0, 1.4 / max(self.current_scale(), 0.5)))
        self._preview_vertices_item.setPen(vertex_pen)
        self._preview_vertices_item.setBrush(QtGui.QBrush(preview_color))
        self._preview_vertices_item.setPath(vertex_path)
        close_path = QtGui.QPainterPath()
        if close_target is not None and points:
            close_radius = vertex_radius * 1.9
            close_path.addEllipse(points[0], close_radius, close_radius)
            close_pen = QtGui.QPen(QtGui.QColor(255, 215, 0, 245))
            close_pen.setWidthF(max(1.5, 2.2 / max(self.current_scale(), 0.5)))
            self._preview_close_item.setPen(close_pen)
            self._preview_close_item.setBrush(QtGui.QBrush(QtCore.Qt.NoBrush))
        self._preview_close_item.setPath(close_path)

    def _polygon_close_target(
        self, scene_pos: QtCore.QPointF | None
    ) -> QtCore.QPointF | None:
        if self.current is None:
            return None
        if str(self.createMode or "").lower() != "polygon":
            return None
        points = list(getattr(self.current, "points", []) or [])
        if len(points) < 2 or scene_pos is None:
            return None
        first_point = QtCore.QPointF(points[0])
        if self._close_enough(scene_pos, first_point):
            return first_point
        return None

    def finalise(self) -> None:
        if self.current is None:
            return
        if not self._supports_create_mode(self.createMode):
            return
        mode = str(self.createMode or "").lower()
        if mode == "polygon":
            if len(self.current.points) < 3:
                return
            self.current.close()
        elif mode in {"line", "rectangle", "circle"}:
            if len(self.current.points) < 2:
                return
            if mode == "rectangle":
                self.current.close()
        elif mode == "linestrip":
            if len(self.current.points) < 2:
                return
            self.current.setOpen()
        elif mode == "point":
            if len(self.current.points) < 1:
                return
        self._shapes.append(self.current)
        self._store_canvas_shapes_backup()
        finished = self.current
        self.current = None
        self._preview_item.setPath(QtGui.QPainterPath())
        self._preview_vertices_item.setPath(QtGui.QPainterPath())
        self._preview_close_item.setPath(QtGui.QPainterPath())
        self.drawingPolygon.emit(False)
        self.set_shapes(self._shapes)
        self._apply_selection([finished], emit_signal=True)
        self.newShape.emit()

    def setLastLabel(self, text, flags):
        assert text
        if not self._shapes:
            return []
        self._shapes[-1].label = text
        self._shapes[-1].flags = flags
        self.set_shapes(self._shapes)
        return [self._shapes[-1]]

    def undoLastLine(self):
        if self.current is not None:
            self.current = None
            self._preview_item.setPath(QtGui.QPainterPath())
            self._preview_vertices_item.setPath(QtGui.QPainterPath())
            self._preview_close_item.setPath(QtGui.QPainterPath())
            self.drawingPolygon.emit(False)
            return
        if not self._shapes:
            return
        self._shapes.pop()
        self.set_shapes(self._shapes)

    def undoLastPoint(self):
        if self.current is None:
            return
        if getattr(self.current, "points", None):
            self.current.popPoint()
        if not getattr(self.current, "points", None):
            self.current = None
            self._preview_item.setPath(QtGui.QPainterPath())
            self._preview_vertices_item.setPath(QtGui.QPainterPath())
            self._preview_close_item.setPath(QtGui.QPainterPath())
            self.drawingPolygon.emit(False)
            return
        self._update_drawing_preview(self.current.points[-1])

    def _show_context_menu(self, global_pos) -> bool:
        host = getattr(self, "_host_window", None)
        if host is not None:
            canvas = getattr(host, "canvas", None)
            builder = getattr(canvas, "_build_context_menu", None)
            if callable(builder):
                try:
                    menu = builder(host)
                except Exception:
                    return False
                if menu is None:
                    return False
                menu.exec_(global_pos)
                return True
        seen = set()
        queue = [self, self.parentWidget(), self.parent(), self.window()]
        while queue:
            candidate = queue.pop(0)
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            canvas = getattr(candidate, "canvas", None)
            builder = getattr(canvas, "_build_context_menu", None)
            if callable(builder):
                try:
                    menu = builder(candidate)
                except Exception:
                    return False
                if menu is None:
                    return False
                menu.exec_(global_pos)
                return True
            queue.append(getattr(candidate, "parentWidget", lambda: None)())
            queue.append(getattr(candidate, "parent", lambda: None)())
        return False

    def set_selected_shapes(self, shapes) -> None:
        self._apply_selection(shapes, emit_signal=False)

    @staticmethod
    def _is_editable_shape(shape) -> bool:
        other = dict(getattr(shape, "other_data", {}) or {})
        if not getattr(shape, "visible", True) or not bool(
            other.get("overlay_visible", True)
        ):
            return False
        if bool(other.get("overlay_locked", False)):
            return False
        if str(getattr(shape, "shape_type", "") or "").lower() == "mask":
            return False
        return bool(getattr(shape, "points", []) or [])

    def _shape_hit_test(self, scene_pos: QtCore.QPointF):
        scale = max(self.current_scale(), 0.01)
        width, height = self._content_size
        diagonal = math.hypot(float(width or 0), float(height or 0))
        diagonal_factor = min(
            1.0, max(0.0, (diagonal - 2048.0) / max(1.0, 32768.0 - 2048.0))
        )
        point_pixels = min(12.0, max(6.0, 8.0 + (2.0 * diagonal_factor)))
        epsilon = max(4.0, (point_pixels * 0.95) / scale)
        for shape in reversed(list(self._shapes or [])):
            if not self._is_editable_shape(shape):
                continue
            vertex_index = None
            try:
                vertex_index = shape.nearestVertex(scene_pos, epsilon)
            except Exception:
                vertex_index = None
            if vertex_index is not None:
                return shape, vertex_index, "vertex"
            shape_type = str(getattr(shape, "shape_type", "") or "").lower()
            if shape_type == "point":
                point = shape.points[0]
                distance = QtCore.QLineF(scene_pos, point).length()
                if distance <= epsilon:
                    return shape, 0, "shape"
            elif shape_type in {"line", "linestrip"}:
                try:
                    edge = shape.nearestEdge(scene_pos, epsilon)
                except Exception:
                    edge = None
                if edge is not None and bool(
                    getattr(shape, "canAddPoint", lambda: False)()
                ):
                    return shape, edge, "edge"
                if edge is not None:
                    return shape, None, "shape"
            else:
                try:
                    edge = shape.nearestEdge(scene_pos, epsilon)
                except Exception:
                    edge = None
                if edge is not None and bool(
                    getattr(shape, "canAddPoint", lambda: False)()
                ):
                    return shape, edge, "edge"
                try:
                    if shape.containsPoint(scene_pos):
                        return shape, None, "shape"
                except Exception:
                    pass
        return None, None, None

    def _clamp_scene_point(self, point: QtCore.QPointF) -> QtCore.QPointF:
        rect = self.sceneRect()
        if rect.isNull():
            return QtCore.QPointF(point)
        return QtCore.QPointF(
            min(max(float(point.x()), rect.left()), rect.right()),
            min(max(float(point.y()), rect.top()), rect.bottom()),
        )

    def _bounded_move_selected_shapes(self, delta: QtCore.QPointF) -> bool:
        if not self.selectedShapes:
            return False
        left = None
        top = None
        right = None
        bottom = None
        for shape in self.selectedShapes:
            rect = shape.boundingRect()
            if rect is None:
                continue
            left = rect.left() if left is None else min(left, rect.left())
            top = rect.top() if top is None else min(top, rect.top())
            right = rect.right() if right is None else max(right, rect.right())
            bottom = rect.bottom() if bottom is None else max(bottom, rect.bottom())
        if left is None:
            return False
        scene = self.sceneRect()
        dx = float(delta.x())
        dy = float(delta.y())
        if left + dx < scene.left():
            dx += scene.left() - (left + dx)
        if right + dx > scene.right():
            dx -= (right + dx) - scene.right()
        if top + dy < scene.top():
            dy += scene.top() - (top + dy)
        if bottom + dy > scene.bottom():
            dy -= (bottom + dy) - scene.bottom()
        bounded = QtCore.QPointF(dx, dy)
        if abs(bounded.x()) < 1e-8 and abs(bounded.y()) < 1e-8:
            return False
        for shape in self.selectedShapes:
            shape.moveBy(bounded)
        return True

    def _pair_item_at_view_pos(self, pos) -> _LandmarkPairItem | None:
        query = self.mapToScene(pos)
        threshold = max(6.0, 10.0 / max(self.current_scale(), 0.01))
        best_item = None
        best_distance = None
        for item in self._pair_items:
            if not isinstance(item, _LandmarkPairItem):
                continue
            line = item.line()
            length = line.length()
            if length <= 0:
                continue
            dx = line.dx()
            dy = line.dy()
            t = (((query.x() - line.x1()) * dx) + ((query.y() - line.y1()) * dy)) / (
                length * length
            )
            t = max(0.0, min(1.0, t))
            closest = QtCore.QPointF(line.x1() + (dx * t), line.y1() + (dy * t))
            distance = QtCore.QLineF(query, closest).length()
            if distance > threshold:
                continue
            if best_distance is None or distance < best_distance:
                best_item = item
                best_distance = distance
        return best_item

    def _make_pair_overlay_items(self, shapes) -> list[QtWidgets.QGraphicsItem]:
        overlay_points = {}
        image_points = {}
        pair_visible = {}
        for shape in list(shapes or []):
            if str(getattr(shape, "shape_type", "") or "").lower() != "point":
                continue
            points = getattr(shape, "points", None) or []
            if not points:
                continue
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if not pair_id:
                continue
            point = points[0]
            coords = (float(point.x()), float(point.y()))
            if "overlay_id" in other:
                overlay_points[pair_id] = coords
                pair_visible[pair_id] = bool(
                    other.get("overlay_landmarks_visible", True)
                )
            else:
                image_points[pair_id] = coords
        items = []
        for pair_id, src in overlay_points.items():
            dst = image_points.get(pair_id)
            if dst is None or not bool(pair_visible.get(pair_id, True)):
                continue
            line_item = _LandmarkPairItem(pair_id, src[0], src[1], dst[0], dst[1])
            line_item.set_selected(pair_id == self._selected_overlay_landmark_pair_id)
            items.append(line_item)
        return items

    def _make_pair_endpoint_items(self, shapes) -> list[QtWidgets.QGraphicsItem]:
        overlay_points = {}
        image_points = {}
        pair_visible = {}
        for shape in list(shapes or []):
            if str(getattr(shape, "shape_type", "") or "").lower() != "point":
                continue
            points = getattr(shape, "points", None) or []
            if not points:
                continue
            other = dict(getattr(shape, "other_data", {}) or {})
            pair_id = str(other.get("overlay_landmark_pair_id") or "")
            if not pair_id:
                continue
            point = QtCore.QPointF(float(points[0].x()), float(points[0].y()))
            if "overlay_id" in other:
                overlay_points[pair_id] = point
                pair_visible[pair_id] = bool(
                    other.get("overlay_landmarks_visible", True)
                )
            else:
                image_points[pair_id] = point
        items = []
        for pair_id, src in overlay_points.items():
            dst = image_points.get(pair_id)
            if dst is None or not bool(pair_visible.get(pair_id, True)):
                continue
            for point in (src, dst):
                endpoint_item = _LandmarkPairEndpointItem(pair_id, point)
                endpoint_item.set_selected(
                    pair_id == self._selected_overlay_landmark_pair_id
                )
                items.append(endpoint_item)
        return items

    def _make_overlay_item(self, shape):
        other = dict(getattr(shape, "other_data", {}) or {})
        if not getattr(shape, "visible", True) or not bool(
            other.get("overlay_visible", True)
        ):
            return None
        if not getattr(shape, "points", []):
            return None
        item = _ShapeGraphicsItem(
            shape,
            content_size=self._content_size,
            current_scale=self.current_scale(),
        )
        item.setOpacity(max(0.0, min(1.0, float(other.get("overlay_opacity", 1.0)))))
        item.setZValue(100.0 + float(other.get("overlay_z_order", 0)))
        return item

    def visible_tile_keys(self, level: int = 0) -> list[TileKey]:
        return self._visible_tile_keys_for_backend(self.backend, level=level)

    def _prefetch_tile_keys(
        self,
        visible_keys: Iterable[TileKey],
        *,
        backend: LargeImageBackend | None,
        level: int,
        limit: int = 12,
    ) -> list[TileKey]:
        if backend is None:
            return []
        try:
            level_w, level_h = backend.get_level_shape(level)
        except Exception:
            return []
        max_tx = max(0, (int(level_w) - 1) // self.tile_size)
        max_ty = max(0, (int(level_h) - 1) // self.tile_size)
        visible = list(visible_keys)
        visible_set = set(visible)
        queued = []
        seen = set()
        for key in visible:
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    neighbor = TileKey(
                        level=key.level,
                        tx=max(0, min(max_tx, key.tx + dx)),
                        ty=max(0, min(max_ty, key.ty + dy)),
                    )
                    if neighbor in visible_set or neighbor in seen:
                        continue
                    seen.add(neighbor)
                    queued.append(neighbor)
                    if len(queued) >= max(0, int(limit)):
                        return queued
        return queued

    def _build_tile_render_plan(
        self,
        *,
        backend: LargeImageBackend | None,
        level: int,
        current_items: dict[TileKey, QtWidgets.QGraphicsPixmapItem],
    ) -> TileRenderPlan[TileKey]:
        visible_keys = tuple(self._visible_tile_keys_for_backend(backend, level=level))
        visible_key_set = set(visible_keys)
        stale_keys = tuple(
            key for key in list(current_items) if key not in visible_key_set
        )
        prefetch_keys = tuple(
            self._prefetch_tile_keys(visible_keys, backend=backend, level=level)
        )
        return TileRenderPlan(
            visible_keys=visible_keys,
            prefetch_keys=prefetch_keys,
            stale_keys=stale_keys,
        )

    def _remove_stale_tile_items(
        self,
        items: dict[TileKey, QtWidgets.QGraphicsPixmapItem],
        stale_keys: Iterable[TileKey],
    ) -> None:
        for key in stale_keys:
            item = items.pop(key, None)
            if item is not None:
                self._scene.removeItem(item)

    def _tile_scene_metrics(
        self,
        backend: LargeImageBackend,
        key: TileKey,
        *,
        tx: float = 0.0,
        ty: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> tuple[QtCore.QPointF, QtGui.QTransform]:
        full_w, full_h = self._content_size
        level_w, level_h = backend.get_level_shape(key.level)
        scale_x = full_w / max(1, level_w)
        scale_y = full_h / max(1, level_h)
        position = QtCore.QPointF(
            tx + (key.tx * self.tile_size * scale_x * sx),
            ty + (key.ty * self.tile_size * scale_y * sy),
        )
        transform = QtGui.QTransform.fromScale(scale_x * sx, scale_y * sy)
        return position, transform

    def _apply_tile_images(
        self,
        *,
        backend: LargeImageBackend,
        tile_images: dict[TileKey, QtGui.QImage],
        visible_keys: Iterable[TileKey],
        item_map: dict[TileKey, QtWidgets.QGraphicsPixmapItem],
        z_value_for_key,
        tx: float = 0.0,
        ty: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
    ) -> None:
        for key in visible_keys:
            if key in item_map:
                continue
            cached = tile_images.get(key)
            if cached is None:
                continue
            item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(cached))
            position, transform = self._tile_scene_metrics(
                backend, key, tx=tx, ty=ty, sx=sx, sy=sy
            )
            item.setPos(position)
            item.setTransform(transform)
            item.setZValue(float(z_value_for_key(key)))
            self._scene.addItem(item)
            item_map[key] = item

    def _visible_tile_keys_for_backend(
        self, backend: LargeImageBackend | None, *, level: int = 0
    ) -> list[TileKey]:
        if backend is None:
            return []
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        full_w, full_h = self._content_size
        level_w, level_h = backend.get_level_shape(level)
        if level_w <= 0 or level_h <= 0:
            return []
        scale_x = full_w / max(1, level_w)
        scale_y = full_h / max(1, level_h)
        max_tx = max(0, (level_w - 1) // self.tile_size)
        max_ty = max(0, (level_h - 1) // self.tile_size)
        left = max(0, int(rect.left() / scale_x) // self.tile_size)
        top = max(0, int(rect.top() / scale_y) // self.tile_size)
        right = min(max_tx, max(0, int(rect.right() / scale_x) // self.tile_size))
        bottom = min(max_ty, max(0, int(rect.bottom() / scale_y) // self.tile_size))
        return [
            TileKey(level=level, tx=tx, ty=ty)
            for ty in range(top, bottom + 1)
            for tx in range(left, right + 1)
        ]

    def current_scale(self) -> float:
        transform = self.transform()
        return float(transform.m11())

    def _select_level(self) -> int:
        return self._select_level_for_backend(self.backend)

    def _select_level_for_backend(self, backend: LargeImageBackend | None) -> int:
        if backend is None:
            return 0
        levels = max(1, int(backend.get_level_count()))
        scale = self.current_scale()
        if levels <= 1:
            return 0
        level = 0
        while level + 1 < levels and scale < 0.5:
            level += 1
            scale *= 2.0
        return level

    def _prioritize_visible_keys(
        self, keys: list[TileKey], *, backend: LargeImageBackend | None
    ) -> tuple[TileKey, ...]:
        if not keys or backend is None:
            return tuple(keys)
        center = self.mapToScene(self.viewport().rect().center())
        try:
            level_w, level_h = backend.get_level_shape(int(keys[0].level))
        except Exception:
            return tuple(keys)
        full_w = max(1, int(self._content_size[0] or 1))
        full_h = max(1, int(self._content_size[1] or 1))
        scale_x = full_w / max(1, int(level_w))
        scale_y = full_h / max(1, int(level_h))

        def _distance(tile_key: TileKey) -> float:
            tile_center_x = (
                (tile_key.tx * self.tile_size) + (self.tile_size / 2.0)
            ) * scale_x
            tile_center_y = (
                (tile_key.ty * self.tile_size) + (self.tile_size / 2.0)
            ) * scale_y
            return math.hypot(center.x() - tile_center_x, center.y() - tile_center_y)

        return tuple(sorted(keys, key=_distance))

    def _poll_pending_tile_results(self) -> None:
        updated = False
        raster_ready = self._tile_scheduler.take_completed()
        if raster_ready and self.backend is not None:
            visible = tuple(
                key for key in self._current_visible_raster_keys if key in raster_ready
            )
            if visible:
                self._apply_tile_images(
                    backend=self.backend,
                    tile_images={key: raster_ready[key] for key in visible},
                    visible_keys=visible,
                    item_map=self._tile_items,
                    z_value_for_key=lambda key: -10.0 - key.level,
                )
                updated = True
        label_ready = self._label_tile_scheduler.take_completed()
        if label_ready and self._label_backend is not None:
            visible = tuple(
                key for key in self._current_visible_label_keys if key in label_ready
            )
            if visible:
                self._apply_tile_images(
                    backend=self._label_backend,
                    tile_images={key: label_ready[key] for key in visible},
                    visible_keys=visible,
                    item_map=self._label_tile_items,
                    z_value_for_key=lambda key: 20.0 + float(key.level) * -0.1,
                    tx=float(self._label_transform["tx"]),
                    ty=float(self._label_transform["ty"]),
                    sx=float(self._label_transform["sx"]),
                    sy=float(self._label_transform["sy"]),
                )
                updated = True
        if updated:
            self.viewport().update()
            self._notify_host_large_image_document_changed()
        stats = self.tile_scheduler_stats()
        if not int(stats["raster"].get("outstanding_requests", 0)) and not int(
            stats["label"].get("outstanding_requests", 0)
        ):
            self._tile_result_timer.stop()
        self._refresh_status_overlay()

    def refresh_visible_tiles(self) -> None:
        if self.backend is None:
            self._last_visible_tile_count = 0
            self._current_visible_raster_keys = ()
            self._refresh_status_overlay()
            return
        # Show the preview thumbnail when zoomed out; lazily add tiles once zoom is meaningful.
        if self.current_scale() < 0.2 and self.backend.get_level_count() <= 1:
            self._last_raster_level = 0
            self._last_visible_tile_count = 0
            self._current_visible_raster_keys = ()
            self._refresh_status_overlay()
            return
        level = self._select_level()
        plan = self._build_tile_render_plan(
            backend=self.backend,
            level=level,
            current_items=self._tile_items,
        )
        self._last_raster_level = int(level)
        self._last_visible_tile_count = len(plan.visible_keys)
        ordered_visible = self._prioritize_visible_keys(
            list(plan.visible_keys), backend=self.backend
        )
        self._current_visible_raster_keys = ordered_visible
        self._remove_stale_tile_items(self._tile_items, plan.stale_keys)
        tile_images = self._tile_scheduler.schedule(
            ordered_visible,
            prefetch_keys=plan.prefetch_keys,
            prime_keys=ordered_visible[:4],
        )
        tile_images.update(
            {
                key: cached
                for key in ordered_visible
                if (cached := self.tile_cache.get(key)) is not None
            }
        )
        self._apply_tile_images(
            backend=self.backend,
            tile_images=tile_images,
            visible_keys=ordered_visible,
            item_map=self._tile_items,
            z_value_for_key=lambda key: -10.0 - key.level,
        )
        if (
            int(self.tile_scheduler_stats()["raster"].get("outstanding_requests", 0))
            > 0
        ):
            self._tile_result_timer.start()
        self._refresh_visible_label_tiles()
        self._refresh_status_overlay()

    def _refresh_visible_label_tiles(self) -> None:
        backend = self._label_backend
        if backend is None or not self._label_overlay_visible:
            self._clear_label_tile_items()
            self._last_label_level = 0
            self._last_label_visible_tile_count = 0
            self._current_visible_label_keys = ()
            self._refresh_status_overlay()
            return
        if not self._content_size[0] or not self._content_size[1]:
            return
        if self.current_scale() < 0.2 and backend.get_level_count() <= 1:
            self._last_label_level = 0
            self._last_label_visible_tile_count = 0
            self._current_visible_label_keys = ()
            self._refresh_status_overlay()
            return
        level = self._select_level_for_backend(backend)
        plan = self._build_tile_render_plan(
            backend=backend,
            level=level,
            current_items=self._label_tile_items,
        )
        self._last_label_level = int(level)
        self._last_label_visible_tile_count = len(plan.visible_keys)
        ordered_visible = self._prioritize_visible_keys(
            list(plan.visible_keys), backend=backend
        )
        self._current_visible_label_keys = ordered_visible
        self._remove_stale_tile_items(self._label_tile_items, plan.stale_keys)
        label_images = self._label_tile_scheduler.schedule(
            ordered_visible,
            prefetch_keys=plan.prefetch_keys,
            prime_keys=ordered_visible[:2],
        )
        label_images.update(
            {
                key: cached
                for key in ordered_visible
                if (cached := self._label_tile_cache.get(key)) is not None
            }
        )
        self._apply_tile_images(
            backend=backend,
            tile_images=label_images,
            visible_keys=ordered_visible,
            item_map=self._label_tile_items,
            z_value_for_key=lambda key: 20.0 + float(key.level) * -0.1,
            tx=float(self._label_transform["tx"]),
            ty=float(self._label_transform["ty"]),
            sx=float(self._label_transform["sx"]),
            sy=float(self._label_transform["sy"]),
        )
        if int(self.tile_scheduler_stats()["label"].get("outstanding_requests", 0)) > 0:
            self._tile_result_timer.start()
        self._refresh_status_overlay()

    def _load_raster_tile(self, key: TileKey) -> QtGui.QImage:
        if self.backend is None:
            return QtGui.QImage()
        x = key.tx * self.tile_size
        y = key.ty * self.tile_size
        region = self.backend.read_region(
            x, y, self.tile_size, self.tile_size, level=key.level
        )
        return region if isinstance(region, QtGui.QImage) else array_to_qimage(region)

    def _load_label_tile(self, key: TileKey) -> QtGui.QImage:
        backend = self._label_backend
        if backend is None:
            return QtGui.QImage()
        x = key.tx * self.tile_size
        y = key.ty * self.tile_size
        region = backend.read_region(
            x, y, self.tile_size, self.tile_size, level=key.level
        )
        rgba = colorize_label_image(
            np.asarray(region),
            opacity=self._label_overlay_opacity,
            selected_label=self._selected_label_value,
        )
        return array_to_qimage(rgba)

    def tile_scheduler_stats(self) -> dict[str, dict[str, int]]:
        raster = self._tile_scheduler.stats()
        label = self._label_tile_scheduler.stats()
        return {
            "raster": dict(raster.__dict__),
            "label": dict(label.__dict__),
        }

    def debug_status_text(self) -> str:
        document = None
        host = self._host_window
        if host is not None:
            getter = getattr(host, "currentLargeImageDocument", None)
            if callable(getter):
                try:
                    document = getter()
                except Exception:
                    document = None
        backend_name = "unknown"
        cache_name = "source"
        page_text = "1/1"
        surface_text = self._status_overlay_surface_text()
        if document is not None:
            backend_name = str(document.backend_name or backend_name)
            page_text = "%d/%d" % (
                int(getattr(document, "current_page", 0) or 0) + 1,
                max(1, int(getattr(document, "page_count", 1) or 1)),
            )
            cache_metadata = dict(getattr(document, "cache_metadata", {}) or {})
            cache_path = str(
                cache_metadata.get("optimized_cache_path", "") or ""
            ).strip()
            if cache_path:
                cache_name = Path(cache_path).name
        elif self.backend is not None:
            backend_name = str(
                getattr(self.backend, "name", backend_name) or backend_name
            )
            if hasattr(self.backend, "get_current_page") and hasattr(
                self.backend, "get_page_count"
            ):
                try:
                    page_text = "%d/%d" % (
                        int(self.backend.get_current_page() or 0) + 1,
                        max(1, int(self.backend.get_page_count() or 1)),
                    )
                except Exception:
                    page_text = "1/1"
        stats = self.tile_scheduler_stats()
        raster_stats = dict(stats.get("raster", {}) or {})
        label_stats = dict(stats.get("label", {}) or {})
        return "\n".join(
            [
                surface_text,
                "backend=%s page=%s level=%d zoom=%d%%"
                % (
                    backend_name,
                    page_text,
                    int(self._last_raster_level),
                    int(round(self.current_scale() * 100.0)),
                ),
                "tiles=%d label_tiles=%d pending=%d hits=%d misses=%d"
                % (
                    int(self._last_visible_tile_count),
                    int(self._last_label_visible_tile_count),
                    int(raster_stats.get("outstanding_requests", 0))
                    + int(label_stats.get("outstanding_requests", 0)),
                    int(raster_stats.get("cache_hits", 0))
                    + int(label_stats.get("cache_hits", 0)),
                    int(raster_stats.get("cache_misses", 0))
                    + int(label_stats.get("cache_misses", 0)),
                ),
                "cache=%s" % cache_name,
            ]
        )

    def _status_overlay_surface_text(self) -> str:
        host = self._host_window
        document = None
        if host is not None:
            getter = getattr(host, "currentLargeImageDocument", None)
            if callable(getter):
                try:
                    document = getter()
                except Exception:
                    document = None
        surface = str(getattr(document, "surface", "tiled") or "tiled")
        draw_mode = str(
            getattr(document, "draw_mode", self.createMode or "polygon") or "polygon"
        )
        if surface == "canvas":
            return "Canvas Preview | %s" % large_image_draw_mode_label(draw_mode)
        if bool(getattr(document, "editing", True)):
            return "Tiled Viewer | Editing"
        return "Tiled Viewer | %s" % large_image_draw_mode_label(draw_mode)

    def _position_status_overlay(self) -> None:
        overlay = getattr(self, "_status_overlay", None)
        if overlay is None:
            return
        margin = 12
        overlay.adjustSize()
        size = overlay.sizeHint()
        overlay.resize(size)
        viewport = self.viewport().rect()
        overlay.move(
            max(margin, viewport.left() + margin),
            max(margin, viewport.top() + margin),
        )

    def _refresh_status_overlay(self) -> None:
        overlay = getattr(self, "_status_overlay", None)
        if overlay is None:
            return
        if self.backend is None or not any(self._content_size):
            overlay.set_status(surface_text="", details_text="", visible=False)
            return
        stats = self.tile_scheduler_stats()
        raster_stats = dict(stats.get("raster", {}) or {})
        label_stats = dict(stats.get("label", {}) or {})
        details = [
            "backend=%s page=%s level=%d zoom=%d%%"
            % (
                str(getattr(self.backend, "name", "unknown") or "unknown"),
                "%d/%d"
                % (
                    int(getattr(self.backend, "get_current_page", lambda: 0)() or 0)
                    + 1,
                    max(
                        1,
                        int(getattr(self.backend, "get_page_count", lambda: 1)() or 1),
                    ),
                ),
                int(self._last_raster_level),
                int(round(self.current_scale() * 100.0)),
            ),
            "visible tiles=%d label tiles=%d"
            % (
                int(self._last_visible_tile_count),
                int(self._last_label_visible_tile_count),
            ),
            "requests=%d hits=%d misses=%d"
            % (
                int(raster_stats.get("outstanding_requests", 0))
                + int(label_stats.get("outstanding_requests", 0)),
                int(raster_stats.get("cache_hits", 0))
                + int(label_stats.get("cache_hits", 0)),
                int(raster_stats.get("cache_misses", 0))
                + int(label_stats.get("cache_misses", 0)),
            ),
        ]
        host = self._host_window
        if host is not None:
            getter = getattr(host, "currentLargeImageDocument", None)
            if callable(getter):
                try:
                    document = getter()
                except Exception:
                    document = None
                else:
                    cache_metadata = dict(getattr(document, "cache_metadata", {}) or {})
                    cache_path = str(
                        cache_metadata.get("optimized_cache_path", "") or ""
                    ).strip()
                    details.append(
                        "cache=%s"
                        % (Path(cache_path).name if cache_path else "source image")
                    )
        overlay.set_status(
            surface_text=self._status_overlay_surface_text(),
            details_text="\n".join(details),
            visible=True,
        )
        self._position_status_overlay()

    def _label_scene_to_level0(
        self, scene_pos: QtCore.QPointF
    ) -> tuple[int, int] | None:
        backend = self._label_backend
        if backend is None or not self._content_size[0] or not self._content_size[1]:
            return None
        level0_w, level0_h = backend.get_level_shape(0)
        if level0_w <= 0 or level0_h <= 0:
            return None
        sx = max(1e-6, float(self._label_transform.get("sx", 1.0) or 1.0))
        sy = max(1e-6, float(self._label_transform.get("sy", 1.0) or 1.0))
        tx = float(self._label_transform.get("tx", 0.0) or 0.0)
        ty = float(self._label_transform.get("ty", 0.0) or 0.0)
        normalized_x = (float(scene_pos.x()) - tx) / sx
        normalized_y = (float(scene_pos.y()) - ty) / sy
        x = int(
            round(
                (normalized_x / max(1.0, float(self._content_size[0])))
                * max(1, level0_w - 1)
            )
        )
        y = int(
            round(
                (normalized_y / max(1.0, float(self._content_size[1])))
                * max(1, level0_h - 1)
            )
        )
        x = min(max(x, 0), max(0, level0_w - 1))
        y = min(max(y, 0), max(0, level0_h - 1))
        return x, y

    def label_value_at(self, scene_pos: QtCore.QPointF) -> int | None:
        backend = self._label_backend
        if backend is None:
            return None
        coords = self._label_scene_to_level0(scene_pos)
        if coords is None:
            return None
        x, y = coords
        tile_x = x // self.tile_size
        tile_y = y // self.tile_size
        key = (tile_x, tile_y)
        cached = self._label_value_tile_cache.get(key)
        if cached is None:
            region = backend.read_region(
                tile_x * self.tile_size,
                tile_y * self.tile_size,
                self.tile_size,
                self.tile_size,
                level=0,
            )
            cached = np.asarray(region)
            self._label_value_tile_cache[key] = cached
            self._label_value_tile_cache.move_to_end(key)
            while len(self._label_value_tile_cache) > 64:
                self._label_value_tile_cache.popitem(last=False)
        local_y = y - (tile_y * self.tile_size)
        local_x = x - (tile_x * self.tile_size)
        if cached.ndim > 2:
            cached = np.squeeze(cached)
        if cached.ndim != 2:
            return None
        if (
            local_y < 0
            or local_y >= cached.shape[0]
            or local_x < 0
            or local_x >= cached.shape[1]
        ):
            return None
        try:
            return int(cached[local_y, local_x])
        except Exception:
            return None

    def _update_label_hover_status(self, scene_pos: QtCore.QPointF) -> None:
        backend = self._label_backend
        host = getattr(self, "_host_window", None)
        if backend is None or host is None:
            self._last_hovered_label_value = None
            return
        label_value = self.label_value_at(scene_pos)
        if label_value == self._last_hovered_label_value:
            return
        self._last_hovered_label_value = label_value
        if label_value is None:
            return
        if hasattr(host, "describeLabelImageOverlayValue"):
            try:
                text = host.describeLabelImageOverlayValue(label_value)
            except Exception:
                text = label_entry_text(label_value, self._label_mapping)
        else:
            text = label_entry_text(label_value, self._label_mapping)
        post_status = getattr(host, "_post_window_status", None)
        if callable(post_status):
            post_status(str(text), 1500)

    def set_zoom_percent(self, percent: int) -> None:
        if not self._content_size[0] or not self._content_size[1]:
            return
        self._fit_mode = "manual"
        self.resetTransform()
        factor = max(0.01, float(percent) / 100.0)
        self.scale(factor, factor)
        self._refresh_overlay_render_metrics()
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def fit_to_window(self) -> None:
        self._fit_mode = "fit_window"
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self._refresh_overlay_render_metrics()
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def fit_to_width(self) -> None:
        self._fit_mode = "fit_width"
        rect = self.sceneRect()
        if rect.width() <= 0:
            return
        self.resetTransform()
        viewport_width = max(1, self.viewport().width())
        factor = viewport_width / rect.width()
        self.scale(factor, factor)
        self._refresh_overlay_render_metrics()
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode == "fit_window":
            self.fit_to_window()
        elif self._fit_mode == "fit_width":
            self.fit_to_width()
        else:
            self.refresh_visible_tiles()
            self._notify_host_large_image_document_changed()
        self._position_status_overlay()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self._refresh_overlay_render_metrics()
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()

    def mousePressEvent(self, event):
        pos = event.pos() if hasattr(event, "pos") else event.position().toPoint()
        scene_pos = self.mapToScene(pos)
        if event.button() == QtCore.Qt.LeftButton and self.drawing():
            if not self._supports_create_mode(
                self.createMode
            ) or not self._scene_contains(scene_pos):
                event.accept()
                return
            mode = str(self.createMode or "").lower()
            if self.current is None:
                from annolid.gui.shape import Shape

                self.current = Shape(
                    shape_type=mode if mode != "polygon" else "polygon"
                )
                self.current.addPoint(QtCore.QPointF(scene_pos))
                if mode == "point":
                    self.finalise()
                else:
                    self.drawingPolygon.emit(True)
                    self._update_drawing_preview(scene_pos)
                event.accept()
                return
            if mode == "polygon":
                if self._polygon_close_target(scene_pos) is not None:
                    self.finalise()
                else:
                    self.current.addPoint(QtCore.QPointF(scene_pos))
                    self._update_drawing_preview(scene_pos)
            elif mode == "linestrip":
                self.current.addPoint(QtCore.QPointF(scene_pos))
                self._update_drawing_preview(scene_pos)
                if event.modifiers() & QtCore.Qt.ControlModifier:
                    self.finalise()
            elif mode in {"line", "rectangle", "circle"}:
                if len(self.current.points) == 1:
                    self.current.addPoint(QtCore.QPointF(scene_pos))
                else:
                    self.current.points[1] = QtCore.QPointF(scene_pos)
                self.finalise()
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            shape, vertex_index, hit_kind = self._shape_hit_test(scene_pos)
            if shape is not None:
                if hit_kind == "edge" and bool(
                    getattr(shape, "canAddPoint", lambda: False)()
                ):
                    insert_index = int(vertex_index)
                    target_point = self._clamp_scene_point(scene_pos)
                    shape.insertPoint(insert_index, QtCore.QPointF(target_point))
                    try:
                        shape.highlightVertex(insert_index, shape.MOVE_VERTEX)
                    except Exception:
                        pass
                    self._active_shape = shape
                    self._active_vertex_index = insert_index
                    self._dragging_shape = True
                    self._shape_moved_during_drag = True
                    self._last_scene_pos = QtCore.QPointF(target_point)
                    self._apply_selection([shape], emit_signal=True)
                    self.set_shapes(self._shapes)
                    event.accept()
                    return
                multi = bool(event.modifiers() & QtCore.Qt.ControlModifier)
                current = list(self.selectedShapes or [])
                if multi:
                    current_ids = {id(item) for item in current}
                    if id(shape) in current_ids:
                        selected = [item for item in current if id(item) != id(shape)]
                    else:
                        selected = current + [shape]
                else:
                    selected = [shape]
                self._active_shape = shape
                self._active_vertex_index = (
                    vertex_index if hit_kind == "vertex" else None
                )
                self._dragging_shape = True
                self._shape_moved_during_drag = False
                self._last_scene_pos = QtCore.QPointF(scene_pos)
                try:
                    if self._active_vertex_index is not None:
                        shape.highlightVertex(
                            self._active_vertex_index, shape.MOVE_VERTEX
                        )
                    else:
                        shape.highlightClear()
                except Exception:
                    pass
                self._apply_selection(selected, emit_signal=True)
                event.accept()
                return
            label_value = self.label_value_at(scene_pos)
            if label_value is not None and int(label_value) > 0:
                self.set_selected_label_value(label_value)
                self._update_label_hover_status(scene_pos)
            self._apply_selection([], emit_signal=True)
        item = self._pair_item_at_view_pos(pos)
        if isinstance(item, _LandmarkPairItem):
            self.set_selected_landmark_pair(item.pair_id)
            self.overlayLandmarkPairSelected.emit(item.pair_id)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.pos() if hasattr(event, "pos") else event.position().toPoint()
        scene_pos = self.mapToScene(pos)
        if self.drawing() and self.current is not None:
            self._update_drawing_preview(self._clamp_scene_point(scene_pos))
            event.accept()
            return
        if (
            self._dragging_shape
            and self._active_shape is not None
            and self._last_scene_pos is not None
            and (event.buttons() & QtCore.Qt.LeftButton)
        ):
            delta = QtCore.QPointF(
                float(scene_pos.x()) - float(self._last_scene_pos.x()),
                float(scene_pos.y()) - float(self._last_scene_pos.y()),
            )
            moved = False
            if self._active_vertex_index is not None:
                old_point = QtCore.QPointF(
                    self._active_shape.points[self._active_vertex_index]
                )
                target = self._clamp_scene_point(old_point + delta)
                bounded = target - old_point
                if abs(bounded.x()) > 1e-8 or abs(bounded.y()) > 1e-8:
                    self._active_shape.moveVertexBy(self._active_vertex_index, bounded)
                    moved = True
            else:
                moved = self._bounded_move_selected_shapes(delta)
            if moved:
                self._shape_moved_during_drag = True
                self._last_scene_pos = QtCore.QPointF(scene_pos)
                self.set_shapes(self._shapes)
            event.accept()
            return
        self._update_label_hover_status(scene_pos)
        super().mouseMoveEvent(event)

    def mouseDoubleClickEvent(self, event):
        if self.drawing() and str(self.createMode or "").lower() in {
            "polygon",
            "linestrip",
        }:
            self.finalise()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.RightButton:
            global_pos = event.globalPos() if hasattr(event, "globalPos") else None
            if global_pos is not None and self._show_context_menu(global_pos):
                event.accept()
                return
        if event.button() == QtCore.Qt.LeftButton and self._dragging_shape:
            moved = bool(self._shape_moved_during_drag)
            if self._active_shape is not None:
                try:
                    self._active_shape.highlightClear()
                except Exception:
                    pass
            self._dragging_shape = False
            self._active_shape = None
            self._active_vertex_index = None
            self._last_scene_pos = None
            self._shape_moved_during_drag = False
            self.set_shapes(self._shapes)
            if moved:
                self.shapeMoved.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def contextMenuEvent(self, event):
        if self._show_context_menu(event.globalPos()):
            event.accept()
            return
        super().contextMenuEvent(event)

    def keyPressEvent(self, event):
        if self.drawing():
            if event.key() == QtCore.Qt.Key_Escape:
                self.current = None
                self._preview_item.setPath(QtGui.QPainterPath())
                self._preview_vertices_item.setPath(QtGui.QPainterPath())
                self._preview_close_item.setPath(QtGui.QPainterPath())
                self.drawingPolygon.emit(False)
                event.accept()
                return
            if event.key() in (QtCore.Qt.Key_Return, QtCore.Qt.Key_Enter):
                self.finalise()
                event.accept()
                return
            if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                self.undoLastPoint()
                event.accept()
                return
        super().keyPressEvent(event)
