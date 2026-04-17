from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
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
from annolid.gui.mixins.shared_polygon_edit_mixin import SharedPolygonEditMixin
from annolid.gui.shape import Shape
from annolid.gui.shared_vertices import SharedTopologyRegistry
from annolid.gui.tile_scheduler import TileRenderPlan, TileRequestScheduler
from annolid.io.large_image import LargeImageBackend
from annolid.io.large_image.common import array_to_qimage

CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor


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


@dataclass
class _RasterOverlayRuntime:
    layer_id: str
    name: str
    backend: LargeImageBackend
    source_path: str
    visible: bool = True
    opacity: float = 1.0
    z_index: float = 10.0
    page_index: int = 0
    tx: float = 0.0
    ty: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    rotation_deg: float = 0.0
    tile_cache: TileCache = field(default_factory=TileCache)
    tile_scheduler: TileRequestScheduler | None = None
    tile_items: dict[TileKey, QtWidgets.QGraphicsPixmapItem] = field(
        default_factory=dict
    )
    last_level: int = 0
    last_visible_tile_count: int = 0
    current_visible_keys: tuple[TileKey, ...] = ()


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

    def sync_shape_geometry(self) -> None:
        # Shape points are mutated externally while editing. Notify QGraphicsView
        # about geometry changes before repainting to avoid stale cached bounds.
        self.prepareGeometryChange()
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
            stroke = QtGui.QColor(
                str(
                    other.get("brain3d_overlay_stroke")
                    or other.get("overlay_stroke")
                    or ""
                )
            )
            fill = QtGui.QColor(
                str(
                    other.get("brain3d_overlay_fill") or other.get("overlay_fill") or ""
                )
            )
            if stroke.isValid():
                stroke.setAlpha(max(stroke.alpha(), 220))
                self._ann_shape.line_color = stroke
            if fill.isValid():
                fill.setAlpha(max(fill.alpha(), 80))
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


class TiledImageView(SharedPolygonEditMixin, QtWidgets.QGraphicsView):
    """Foundation widget for large-image tile rendering."""

    overlayLandmarkPairSelected = QtCore.Signal(str)
    selectionChanged = QtCore.Signal(list)
    vertexSelected = QtCore.Signal(bool)
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
        self._active_edge_index: int | None = None
        self._selected_vertex_shape = None
        self._selected_vertex_index: int | None = None
        self._dragging_shape = False
        self._shape_moved_during_drag = False
        self._last_scene_pos: QtCore.QPointF | None = None
        self._dragging_raster_overlay = False
        self._raster_overlay_drag_layer_id: str | None = None
        self._raster_overlay_drag_start_pos: QtCore.QPointF | None = None
        self._raster_overlay_drag_start_tx: float = 0.0
        self._raster_overlay_drag_start_ty: float = 0.0
        self._raster_overlay_arrow_mode: bool = False
        self._raster_overlay_arrow_dragging: bool = False
        self._raster_overlay_arrow_handle: str | None = None
        self._raster_overlay_arrow_layer_id: str | None = None
        self._raster_overlay_arrow_start_pos: QtCore.QPointF | None = None
        self._raster_overlay_arrow_start_tx: float = 0.0
        self._raster_overlay_arrow_start_ty: float = 0.0
        self._raster_overlay_arrow_start_sx: float = 1.0
        self._raster_overlay_arrow_start_sy: float = 1.0
        self._raster_overlay_arrow_start_rotation: float = 0.0
        self._raster_overlay_arrow_start_angle_deg: float | None = None
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
        self._raster_overlay_layers: dict[str, _RasterOverlayRuntime] = {}
        self._base_raster_visible: bool = True
        self._label_transform: dict[str, float] = {
            "tx": 0.0,
            "ty": 0.0,
            "sx": 1.0,
            "sy": 1.0,
        }
        self._last_hovered_label_value: int | None = None
        self._shared_topology_registry = SharedTopologyRegistry.from_shapes([])
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
        self._visible_tiles_refresh_timer = QtCore.QTimer(self)
        self._visible_tiles_refresh_timer.setSingleShot(True)
        self._visible_tiles_refresh_timer.setInterval(40)
        self._visible_tiles_refresh_timer.timeout.connect(
            self._refresh_visible_tiles_now
        )
        self._current_visible_raster_keys: tuple[TileKey, ...] = ()
        self._current_visible_label_keys: tuple[TileKey, ...] = ()
        self._adjoining_source_shape: Shape | None = None
        self._adjoining_default_point_pending: bool = False
        self._shared_boundary_reshape_mode: bool = False
        self._shared_boundary_shape: Shape | None = None
        self._shared_boundary_edge_index: int | None = None
        self._dragging_shared_boundary: bool = False
        self._shared_boundary_last_pos: QtCore.QPointF | None = None
        self._cursor = CURSOR_DEFAULT
        self._host_document_sync_pending = False

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

    def _defer_host_large_image_document_changed(self) -> None:
        if self._host_document_sync_pending:
            return
        self._host_document_sync_pending = True

        def _flush() -> None:
            self._host_document_sync_pending = False
            self._notify_host_large_image_document_changed()

        QtCore.QTimer.singleShot(0, _flush)

    def _queue_visible_tiles_refresh(self) -> None:
        timer = getattr(self, "_visible_tiles_refresh_timer", None)
        if timer is None:
            self._refresh_visible_tiles_now()
            return
        try:
            timer.start()
        except Exception:
            self._refresh_visible_tiles_now()

    def overrideCursor(self, cursor) -> None:
        self._cursor = cursor
        try:
            self.setCursor(cursor)
        except Exception:
            pass
        viewport = getattr(self, "viewport", None)
        if callable(viewport):
            try:
                vp = viewport()
                if vp is not None:
                    vp.setCursor(cursor)
            except Exception:
                pass

    def restoreCursor(self) -> None:
        try:
            self.unsetCursor()
        except Exception:
            pass
        viewport = getattr(self, "viewport", None)
        if callable(viewport):
            try:
                vp = viewport()
                if vp is not None:
                    vp.unsetCursor()
            except Exception:
                pass
        self._refresh_status_overlay()

    def enterEvent(self, event):
        self.overrideCursor(self._cursor)
        super().enterEvent(event)

    def focusOutEvent(self, event):
        self._clearSharedBoundaryReshape()
        self.restoreCursor()
        super().focusOutEvent(event)

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
            self._queue_visible_tiles_refresh()
        self._notify_host_large_image_document_changed()

    def set_backend(self, backend: LargeImageBackend) -> None:
        try:
            self._tile_scheduler.shutdown()
        except Exception:
            pass
        self.clear_raster_overlay_layers(notify=False)
        self.backend = backend
        self.tile_cache = TileCache()
        self._tile_scheduler = TileRequestScheduler(
            cache_get=self.tile_cache.get,
            cache_put=self.tile_cache.put,
            load_tile=self._load_raster_tile,
        )
        self._base_raster_visible = True
        self._pixmap_item.setVisible(True)
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
        self._refresh_visible_tiles_now()
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
        self._refresh_visible_tiles_now()
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
        self._refresh_visible_tiles_now()
        self._notify_host_large_image_document_changed()

    def set_label_layer_visible(self, visible: bool) -> None:
        visible_flag = bool(visible)
        if visible_flag == self._label_overlay_visible:
            return
        self._label_overlay_visible = visible_flag
        if not visible_flag:
            self._clear_label_tile_items()
        self._refresh_visible_tiles_now()
        self._notify_host_large_image_document_changed()

    def label_layer_visible(self) -> bool:
        return bool(self._label_overlay_visible)

    def set_raster_overlay_layers(self, layers: list[dict]) -> None:
        self.clear_raster_overlay_layers(notify=False)
        ordered_layers = sorted(
            list(layers or []),
            key=lambda layer: float((layer or {}).get("z_index", 0.0) or 0.0),
        )
        for layer in ordered_layers:
            layer_id = str(layer.get("id") or "").strip()
            backend = layer.get("backend")
            if not layer_id or backend is None:
                continue
            runtime = _RasterOverlayRuntime(
                layer_id=layer_id,
                name=str(layer.get("name") or layer_id),
                backend=backend,
                source_path=str(layer.get("source_path") or ""),
                visible=bool(layer.get("visible", True)),
                opacity=max(0.0, min(1.0, float(layer.get("opacity", 1.0) or 1.0))),
                z_index=float(layer.get("z_index", 10.0) or 10.0),
                page_index=int(layer.get("page_index", 0) or 0),
                tx=float(layer.get("tx", 0.0) or 0.0),
                ty=float(layer.get("ty", 0.0) or 0.0),
                sx=max(1e-6, float(layer.get("sx", 1.0) or 1.0)),
                sy=max(1e-6, float(layer.get("sy", 1.0) or 1.0)),
                rotation_deg=float(layer.get("rotation_deg", 0.0) or 0.0),
            )
            runtime.tile_scheduler = TileRequestScheduler(
                cache_get=runtime.tile_cache.get,
                cache_put=runtime.tile_cache.put,
                load_tile=lambda key,
                _layer_id=layer_id: self._load_raster_overlay_tile(_layer_id, key),
                async_load=True,
            )
            self._raster_overlay_layers[layer_id] = runtime
        self._refresh_visible_tiles_now()
        self._notify_host_large_image_document_changed()

    def clear_raster_overlay_layers(self, *, notify: bool = True) -> None:
        for runtime in list(self._raster_overlay_layers.values()):
            scheduler = runtime.tile_scheduler
            if scheduler is not None:
                try:
                    scheduler.shutdown()
                except Exception:
                    pass
            for item in list(runtime.tile_items.values()):
                self._scene.removeItem(item)
            runtime.tile_items.clear()
        self._raster_overlay_layers.clear()
        if notify:
            self.refresh_visible_tiles()
            self._notify_host_large_image_document_changed()

    def set_raster_overlay_layer_visible(self, layer_id: str, visible: bool) -> bool:
        runtime = self._raster_overlay_layers.get(str(layer_id or ""))
        if runtime is None:
            return False
        visible_flag = bool(visible)
        if visible_flag == bool(runtime.visible):
            return False
        runtime.visible = visible_flag
        if not visible_flag:
            self._remove_stale_tile_items(
                runtime.tile_items,
                list(runtime.tile_items.keys()),
            )
            runtime.current_visible_keys = ()
            runtime.last_visible_tile_count = 0
        self._refresh_visible_tiles_now()
        self._notify_host_large_image_document_changed()
        return True

    def set_base_raster_visible(self, visible: bool) -> bool:
        visible_flag = bool(visible)
        if visible_flag == bool(self._base_raster_visible):
            return False
        self._base_raster_visible = visible_flag
        self._pixmap_item.setVisible(visible_flag)
        if not visible_flag:
            self._remove_stale_tile_items(
                self._tile_items, list(self._tile_items.keys())
            )
            self._current_visible_raster_keys = ()
            self._last_visible_tile_count = 0
        self._refresh_visible_tiles_now()
        self._notify_host_large_image_document_changed()
        return True

    def base_raster_visible(self) -> bool:
        return bool(self._base_raster_visible)

    def set_raster_overlay_layer_opacity(self, layer_id: str, opacity: float) -> bool:
        runtime = self._raster_overlay_layers.get(str(layer_id or ""))
        if runtime is None:
            return False
        normalized = max(0.0, min(1.0, float(opacity)))
        if abs(float(runtime.opacity) - normalized) < 1e-6:
            return False
        runtime.opacity = float(normalized)
        self._update_existing_tile_items(
            backend=runtime.backend,
            item_map=runtime.tile_items,
            z_value_for_key=lambda key, z=runtime.z_index: float(z)
            + (float(key.level) * -0.05),
            tx=float(runtime.tx),
            ty=float(runtime.ty),
            sx=float(runtime.sx),
            sy=float(runtime.sy),
            rotation_deg=float(runtime.rotation_deg),
            item_opacity=runtime.opacity,
        )
        self.viewport().update()
        self._notify_host_large_image_document_changed()
        return True

    def set_raster_overlay_layer_name(self, layer_id: str, name: str) -> bool:
        runtime = self._raster_overlay_layers.get(str(layer_id or ""))
        if runtime is None:
            return False
        normalized = str(name or "").strip()
        if not normalized:
            return False
        if str(runtime.name) == normalized:
            return False
        runtime.name = normalized
        self._notify_host_large_image_document_changed()
        return True

    def set_raster_overlay_layer_z_index(self, layer_id: str, z_index: float) -> bool:
        runtime = self._raster_overlay_layers.get(str(layer_id or ""))
        if runtime is None:
            return False
        normalized = float(z_index)
        if abs(float(runtime.z_index) - normalized) < 1e-9:
            return False
        runtime.z_index = normalized
        for key, item in list(runtime.tile_items.items()):
            try:
                item.setZValue(float(normalized) + (float(key.level) * -0.05))
            except Exception:
                continue
        self.viewport().update()
        self._notify_host_large_image_document_changed()
        return True

    def set_raster_overlay_layer_transform(
        self,
        layer_id: str,
        *,
        tx: float | None = None,
        ty: float | None = None,
        sx: float | None = None,
        sy: float | None = None,
        rotation_deg: float | None = None,
    ) -> bool:
        runtime = self._raster_overlay_layers.get(str(layer_id or ""))
        if runtime is None:
            return False
        next_tx = float(runtime.tx if tx is None else tx)
        next_ty = float(runtime.ty if ty is None else ty)
        next_sx = max(1e-6, float(runtime.sx if sx is None else sx))
        next_sy = max(1e-6, float(runtime.sy if sy is None else sy))
        next_rotation = float(
            runtime.rotation_deg if rotation_deg is None else rotation_deg
        )
        if (
            abs(float(runtime.tx) - next_tx) < 1e-9
            and abs(float(runtime.ty) - next_ty) < 1e-9
            and abs(float(runtime.sx) - next_sx) < 1e-9
            and abs(float(runtime.sy) - next_sy) < 1e-9
            and abs(float(runtime.rotation_deg) - next_rotation) < 1e-9
        ):
            return False
        runtime.tx = next_tx
        runtime.ty = next_ty
        runtime.sx = next_sx
        runtime.sy = next_sy
        runtime.rotation_deg = next_rotation
        self._update_existing_tile_items(
            backend=runtime.backend,
            item_map=runtime.tile_items,
            z_value_for_key=lambda key, z=runtime.z_index: float(z)
            + (float(key.level) * -0.05),
            tx=float(runtime.tx),
            ty=float(runtime.ty),
            sx=float(runtime.sx),
            sy=float(runtime.sy),
            rotation_deg=float(runtime.rotation_deg),
            item_opacity=runtime.opacity,
        )
        self.viewport().update()
        self._queue_visible_tiles_refresh()
        self._notify_host_large_image_document_changed()
        return True

    def remove_raster_overlay_layer(self, layer_id: str) -> bool:
        runtime = self._raster_overlay_layers.pop(str(layer_id or ""), None)
        if runtime is None:
            return False
        scheduler = runtime.tile_scheduler
        if scheduler is not None:
            try:
                scheduler.shutdown()
            except Exception:
                pass
        self._remove_stale_tile_items(
            runtime.tile_items,
            list(runtime.tile_items.keys()),
        )
        runtime.tile_items.clear()
        runtime.current_visible_keys = ()
        runtime.last_visible_tile_count = 0
        self.refresh_visible_tiles()
        self._notify_host_large_image_document_changed()
        return True

    def raster_overlay_layers_state(self) -> list[dict]:
        state = []
        ordered_runtimes = sorted(
            self._raster_overlay_layers.values(),
            key=lambda runtime: float(runtime.z_index),
        )
        for runtime in ordered_runtimes:
            state.append(
                {
                    "id": str(runtime.layer_id),
                    "name": str(runtime.name),
                    "source_path": str(runtime.source_path),
                    "visible": bool(runtime.visible),
                    "opacity": float(runtime.opacity),
                    "z_index": float(runtime.z_index),
                    "page_index": int(runtime.page_index),
                    "tx": float(runtime.tx),
                    "ty": float(runtime.ty),
                    "sx": float(runtime.sx),
                    "sy": float(runtime.sy),
                    "rotation_deg": float(runtime.rotation_deg),
                }
            )
        return state

    def sync_raster_overlay_pages(
        self,
        *,
        base_page: int,
        base_page_count: int,
    ) -> None:
        changed = False
        for runtime in self._raster_overlay_layers.values():
            backend = runtime.backend
            try:
                overlay_page_count = int(
                    getattr(backend, "get_page_count", lambda: 1)() or 1
                )
            except Exception:
                overlay_page_count = 1
            if overlay_page_count != int(base_page_count) or overlay_page_count <= 1:
                continue
            try:
                backend.set_page(int(base_page))
            except Exception:
                continue
            runtime.page_index = int(base_page)
            runtime.tile_cache = TileCache()
            runtime.tile_items.clear()
            runtime.current_visible_keys = ()
            runtime.last_visible_tile_count = 0
            changed = True
        if changed:
            self.refresh_visible_tiles()
            self._notify_host_large_image_document_changed()

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
        self._selected_vertex_shape = None
        self._selected_vertex_index = None
        self._dragging_shape = False
        self._shape_moved_during_drag = False
        self._last_scene_pos = None
        self._adjoining_default_point_pending = False
        self._clear_adjoining_source()
        self.current = None
        self.mode = self.EDIT
        self._pixmap_item.setPixmap(QtGui.QPixmap())
        self._preview_item.setPath(QtGui.QPainterPath())
        self._preview_vertices_item.setPath(QtGui.QPainterPath())
        self._preview_close_item.setPath(QtGui.QPainterPath())
        self._clear_tile_items()
        self._base_raster_visible = True
        self.clear_raster_overlay_layers(notify=False)
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
            self._adjoining_default_point_pending = False
            self._clear_adjoining_source()
            self._clearSharedBoundaryReshape()
            self._preview_item.setPath(QtGui.QPainterPath())
            self._preview_vertices_item.setPath(QtGui.QPainterPath())
            self._preview_close_item.setPath(QtGui.QPainterPath())
            self.drawingPolygon.emit(False)
            self.restoreCursor()
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
        self._clearSharedBoundaryReshape()
        previous_overlay_items = list(self._overlay_items or [])
        previous_by_shape_id: dict[int, _ShapeGraphicsItem] = {}
        for item in previous_overlay_items:
            if isinstance(item, _ShapeGraphicsItem):
                previous_by_shape_id[id(item._ann_shape)] = item
        self._overlay_items = []
        for item in self._pair_items:
            self._scene.removeItem(item)
        self._pair_items = []
        for item in self._pair_endpoint_items:
            self._scene.removeItem(item)
        self._pair_endpoint_items = []
        shapes_list = shapes if isinstance(shapes, list) else list(shapes or [])
        self._shapes = shapes_list
        self._shared_finalize_topology_edit()
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
        used_shape_ids: set[int] = set()
        for shape in shapes_list:
            shape_id = id(shape)
            used_shape_ids.add(shape_id)
            reused = previous_by_shape_id.get(shape_id)
            if reused is not None:
                other = dict(getattr(shape, "other_data", {}) or {})
                visible = bool(getattr(shape, "visible", True)) and bool(
                    other.get("overlay_visible", True)
                )
                if visible and bool(getattr(shape, "points", []) or []):
                    reused.setOpacity(
                        max(0.0, min(1.0, float(other.get("overlay_opacity", 1.0))))
                    )
                    reused.setZValue(100.0 + float(other.get("overlay_z_order", 0)))
                    reused.sync_shape_geometry()
                    self._overlay_items.append(reused)
                else:
                    self._scene.removeItem(reused)
                continue
            item = self._make_overlay_item(shape)
            if item is not None:
                self._scene.addItem(item)
                self._overlay_items.append(item)
        for item in previous_overlay_items:
            if not isinstance(item, _ShapeGraphicsItem):
                continue
            if id(item._ann_shape) in used_shape_ids:
                continue
            self._scene.removeItem(item)
        self._refresh_overlay_render_metrics()
        self._notify_host_large_image_document_changed()

    def _refresh_overlay_render_metrics(self) -> None:
        scale = self.current_scale()
        for item in self._overlay_items:
            if isinstance(item, _ShapeGraphicsItem):
                item.set_current_scale(scale)

    def _refresh_overlay_geometry(self) -> None:
        for item in self._overlay_items:
            if isinstance(item, _ShapeGraphicsItem):
                # Hidden or detached items are intentionally left untouched.
                # Calling prepareGeometryChange on those items can crash Qt.
                if not item.isVisible() or item.scene() is None:
                    continue
                try:
                    item.sync_shape_geometry()
                except Exception:
                    continue

    def _selected_raster_overlay_layer_id(self) -> str | None:
        host = getattr(self, "_host_window", None)
        getter = getattr(host, "selectedRasterOverlayLayerId", None)
        if callable(getter):
            try:
                layer_id = str(getter() or "")
            except Exception:
                layer_id = ""
            if layer_id:
                return layer_id
        return None

    def _raster_overlay_runtime(self, layer_id: str):
        return self._raster_overlay_layers.get(str(layer_id or ""))

    def set_raster_overlay_arrow_mode(self, enabled: bool) -> None:
        enabled_flag = bool(enabled)
        if enabled_flag == bool(self._raster_overlay_arrow_mode):
            return
        self._raster_overlay_arrow_mode = enabled_flag
        if not enabled_flag:
            self._end_raster_overlay_arrow_drag()
        self.viewport().update()

    def _selected_raster_overlay_runtime(
        self,
    ) -> tuple[str, _RasterOverlayRuntime] | None:
        layer_id = self._selected_raster_overlay_layer_id()
        if not layer_id:
            return None
        runtime = self._raster_overlay_runtime(layer_id)
        if runtime is None:
            return None
        return str(layer_id), runtime

    def _raster_overlay_scene_bounds(
        self, runtime: _RasterOverlayRuntime
    ) -> QtCore.QRectF:
        full_w = max(1.0, float(self._content_size[0] or 1))
        full_h = max(1.0, float(self._content_size[1] or 1))
        width = full_w * max(1e-6, float(runtime.sx))
        height = full_h * max(1e-6, float(runtime.sy))
        return QtCore.QRectF(float(runtime.tx), float(runtime.ty), width, height)

    def _raster_overlay_scene_center(
        self, runtime: _RasterOverlayRuntime
    ) -> QtCore.QPointF:
        full_w = max(1.0, float(self._content_size[0] or 1))
        full_h = max(1.0, float(self._content_size[1] or 1))
        return QtCore.QPointF(
            float(runtime.tx) + ((full_w * max(1e-6, float(runtime.sx))) * 0.5),
            float(runtime.ty) + ((full_h * max(1e-6, float(runtime.sy))) * 0.5),
        )

    @staticmethod
    def _overlay_affine_matrix(
        *, sx: float, sy: float, rotation_deg: float
    ) -> tuple[float, float, float, float]:
        theta = math.radians(float(rotation_deg))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        m11 = float(sx) * cos_t
        m12 = float(sx) * sin_t
        m21 = -float(sy) * sin_t
        m22 = float(sy) * cos_t
        return m11, m12, m21, m22

    def _overlay_scene_from_local(
        self,
        *,
        local_x: float,
        local_y: float,
        tx: float,
        ty: float,
        sx: float,
        sy: float,
        rotation_deg: float,
    ) -> QtCore.QPointF:
        full_w = max(1.0, float(self._content_size[0] or 1))
        full_h = max(1.0, float(self._content_size[1] or 1))
        center_local_x = full_w * 0.5
        center_local_y = full_h * 0.5
        center_scene_x = float(tx) + ((full_w * float(sx)) * 0.5)
        center_scene_y = float(ty) + ((full_h * float(sy)) * 0.5)
        m11, m12, m21, m22 = self._overlay_affine_matrix(
            sx=float(sx),
            sy=float(sy),
            rotation_deg=float(rotation_deg),
        )
        dx = float(local_x) - center_local_x
        dy = float(local_y) - center_local_y
        return QtCore.QPointF(
            center_scene_x + (m11 * dx) + (m21 * dy),
            center_scene_y + (m12 * dx) + (m22 * dy),
        )

    def _raster_overlay_scene_corners(
        self, runtime: _RasterOverlayRuntime
    ) -> dict[str, QtCore.QPointF]:
        full_w = max(1.0, float(self._content_size[0] or 1))
        full_h = max(1.0, float(self._content_size[1] or 1))
        params = {
            "tx": float(runtime.tx),
            "ty": float(runtime.ty),
            "sx": max(1e-6, float(runtime.sx)),
            "sy": max(1e-6, float(runtime.sy)),
            "rotation_deg": float(runtime.rotation_deg),
        }
        return {
            "nw": self._overlay_scene_from_local(local_x=0.0, local_y=0.0, **params),
            "ne": self._overlay_scene_from_local(local_x=full_w, local_y=0.0, **params),
            "se": self._overlay_scene_from_local(
                local_x=full_w, local_y=full_h, **params
            ),
            "sw": self._overlay_scene_from_local(local_x=0.0, local_y=full_h, **params),
        }

    def _raster_overlay_axis_scene_vectors(
        self, runtime: _RasterOverlayRuntime
    ) -> tuple[QtCore.QPointF, QtCore.QPointF]:
        m11, m12, m21, m22 = self._overlay_affine_matrix(
            sx=max(1e-6, float(runtime.sx)),
            sy=max(1e-6, float(runtime.sy)),
            rotation_deg=float(runtime.rotation_deg),
        )
        return QtCore.QPointF(m11, m12), QtCore.QPointF(m21, m22)

    def _raster_overlay_arrow_handles(
        self, runtime: _RasterOverlayRuntime
    ) -> dict[str, dict]:
        corners = self._raster_overlay_scene_corners(runtime)
        nw = corners["nw"]
        ne = corners["ne"]
        se = corners["se"]
        sw = corners["sw"]
        center = self._raster_overlay_scene_center(runtime)
        top_center = QtCore.QPointF(
            (float(nw.x()) + float(ne.x())) * 0.5,
            (float(nw.y()) + float(ne.y())) * 0.5,
        )
        right_center = QtCore.QPointF(
            (float(ne.x()) + float(se.x())) * 0.5,
            (float(ne.y()) + float(se.y())) * 0.5,
        )
        bottom_center = QtCore.QPointF(
            (float(sw.x()) + float(se.x())) * 0.5,
            (float(sw.y()) + float(se.y())) * 0.5,
        )
        left_center = QtCore.QPointF(
            (float(nw.x()) + float(sw.x())) * 0.5,
            (float(nw.y()) + float(sw.y())) * 0.5,
        )
        # Keep handles visually stable across zoom levels by converting
        # desired screen-space pixels into scene units.
        scale = max(1e-6, float(self.current_scale() or 1.0))
        scene_per_px = 1.0 / scale
        margin = 14.0 * scene_per_px
        size = 16.0 * scene_per_px
        top_dir_x = float(top_center.x()) - float(center.x())
        top_dir_y = float(top_center.y()) - float(center.y())
        top_dir_norm = math.hypot(top_dir_x, top_dir_y)
        if top_dir_norm <= 1e-9:
            top_unit_x, top_unit_y = 0.0, -1.0
        else:
            top_unit_x = top_dir_x / top_dir_norm
            top_unit_y = top_dir_y / top_dir_norm
        rotate_center = QtCore.QPointF(
            float(top_center.x()) + (top_unit_x * (margin + size * 1.6)),
            float(top_center.y()) + (top_unit_y * (margin + size * 1.6)),
        )
        offset_x = top_unit_x * margin
        offset_y = top_unit_y * margin
        return {
            "left": {
                "center": QtCore.QPointF(
                    float(left_center.x()) + offset_x,
                    float(left_center.y()) + offset_y,
                ),
                "hit": QtCore.QRectF(
                    float(left_center.x()) + offset_x - size,
                    float(left_center.y()) + offset_y - size,
                    size * 2.0,
                    size * 2.0,
                ),
                "cursor": QtCore.Qt.SizeHorCursor,
            },
            "right": {
                "center": QtCore.QPointF(
                    float(right_center.x()) - offset_x,
                    float(right_center.y()) - offset_y,
                ),
                "hit": QtCore.QRectF(
                    float(right_center.x()) - offset_x - size,
                    float(right_center.y()) - offset_y - size,
                    size * 2.0,
                    size * 2.0,
                ),
                "cursor": QtCore.Qt.SizeHorCursor,
            },
            "top": {
                "center": QtCore.QPointF(
                    float(top_center.x()) + offset_x,
                    float(top_center.y()) + offset_y,
                ),
                "hit": QtCore.QRectF(
                    float(top_center.x()) + offset_x - size,
                    float(top_center.y()) + offset_y - size,
                    size * 2.0,
                    size * 2.0,
                ),
                "cursor": QtCore.Qt.SizeVerCursor,
            },
            "bottom": {
                "center": QtCore.QPointF(
                    float(bottom_center.x()) - offset_x,
                    float(bottom_center.y()) - offset_y,
                ),
                "hit": QtCore.QRectF(
                    float(bottom_center.x()) - offset_x - size,
                    float(bottom_center.y()) - offset_y - size,
                    size * 2.0,
                    size * 2.0,
                ),
                "cursor": QtCore.Qt.SizeVerCursor,
            },
            "nw": {
                "center": QtCore.QPointF(
                    float(nw.x()) + (offset_x * 0.9),
                    float(nw.y()) + (offset_y * 0.9),
                ),
                "hit": QtCore.QRectF(
                    float(nw.x()) + (offset_x * 0.9) - size,
                    float(nw.y()) + (offset_y * 0.9) - size,
                    size * 2.0,
                    size * 2.0,
                ),
                "cursor": QtCore.Qt.SizeFDiagCursor,
            },
            "ne": {
                "center": QtCore.QPointF(
                    float(ne.x()) + (offset_x * 0.9),
                    float(ne.y()) + (offset_y * 0.9),
                ),
                "hit": QtCore.QRectF(
                    float(ne.x()) + (offset_x * 0.9) - size,
                    float(ne.y()) + (offset_y * 0.9) - size,
                    size * 2.0,
                    size * 2.0,
                ),
                "cursor": QtCore.Qt.SizeBDiagCursor,
            },
            "sw": {
                "center": QtCore.QPointF(
                    float(sw.x()) - (offset_x * 0.9),
                    float(sw.y()) - (offset_y * 0.9),
                ),
                "hit": QtCore.QRectF(
                    float(sw.x()) - (offset_x * 0.9) - size,
                    float(sw.y()) - (offset_y * 0.9) - size,
                    size * 2.0,
                    size * 2.0,
                ),
                "cursor": QtCore.Qt.SizeBDiagCursor,
            },
            "se": {
                "center": QtCore.QPointF(
                    float(se.x()) - (offset_x * 0.9),
                    float(se.y()) - (offset_y * 0.9),
                ),
                "hit": QtCore.QRectF(
                    float(se.x()) - (offset_x * 0.9) - size,
                    float(se.y()) - (offset_y * 0.9) - size,
                    size * 2.0,
                    size * 2.0,
                ),
                "cursor": QtCore.Qt.SizeFDiagCursor,
            },
            "rotate": {
                "center": rotate_center,
                "hit": QtCore.QRectF(
                    float(rotate_center.x()) - size,
                    float(rotate_center.y()) - size,
                    size * 2.0,
                    size * 2.0,
                ),
                "cursor": QtCore.Qt.OpenHandCursor,
            },
        }

    def _rotation_angle_deg(
        self, *, center: QtCore.QPointF, target: QtCore.QPointF
    ) -> float:
        dx = float(target.x()) - float(center.x())
        dy = float(target.y()) - float(center.y())
        return math.degrees(math.atan2(dy, dx))

    @staticmethod
    def _normalize_angle_deg(value: float) -> float:
        angle = float(value)
        while angle > 180.0:
            angle -= 360.0
        while angle <= -180.0:
            angle += 360.0
        return angle

    def _hit_raster_overlay_arrow_handle(self, scene_pos: QtCore.QPointF) -> str | None:
        selected = self._selected_raster_overlay_runtime()
        if selected is None:
            return None
        _layer_id, runtime = selected
        handles = self._raster_overlay_arrow_handles(runtime)
        for handle, payload in handles.items():
            hit_rect = payload.get("hit")
            if isinstance(hit_rect, QtCore.QRectF) and hit_rect.contains(scene_pos):
                return str(handle)
        return None

    def _start_raster_overlay_arrow_drag(
        self, handle: str, scene_pos: QtCore.QPointF
    ) -> bool:
        selected = self._selected_raster_overlay_runtime()
        if selected is None:
            return False
        layer_id, runtime = selected
        handle_name = str(handle or "").strip().lower()
        if handle_name not in {
            "left",
            "right",
            "top",
            "bottom",
            "nw",
            "ne",
            "sw",
            "se",
            "rotate",
        }:
            return False
        self._raster_overlay_arrow_dragging = True
        self._raster_overlay_arrow_handle = handle_name
        self._raster_overlay_arrow_layer_id = str(layer_id)
        self._raster_overlay_arrow_start_pos = QtCore.QPointF(scene_pos)
        self._raster_overlay_arrow_start_tx = float(runtime.tx)
        self._raster_overlay_arrow_start_ty = float(runtime.ty)
        self._raster_overlay_arrow_start_sx = max(1e-6, float(runtime.sx))
        self._raster_overlay_arrow_start_sy = max(1e-6, float(runtime.sy))
        self._raster_overlay_arrow_start_rotation = float(runtime.rotation_deg)
        self._raster_overlay_arrow_start_angle_deg = None
        if handle_name == "rotate":
            center = self._raster_overlay_scene_center(runtime)
            self._raster_overlay_arrow_start_angle_deg = self._rotation_angle_deg(
                center=center, target=scene_pos
            )
        cursor = QtCore.Qt.SizeVerCursor
        if handle_name in {"left", "right"}:
            cursor = QtCore.Qt.SizeHorCursor
        elif handle_name in {"nw", "se"}:
            cursor = QtCore.Qt.SizeFDiagCursor
        elif handle_name in {"ne", "sw"}:
            cursor = QtCore.Qt.SizeBDiagCursor
        elif handle_name == "rotate":
            cursor = QtCore.Qt.ClosedHandCursor
        self.overrideCursor(cursor)
        if handle_name == "rotate":
            self._set_hover_feedback(self.tr("Drag handle to rotate raster overlay"))
        else:
            self._set_hover_feedback(self.tr("Drag arrow to resize raster overlay"))
        return True

    def _apply_raster_overlay_arrow_drag(self, scene_pos: QtCore.QPointF) -> bool:
        layer_id = str(self._raster_overlay_arrow_layer_id or "")
        handle = str(self._raster_overlay_arrow_handle or "")
        if (
            not self._raster_overlay_arrow_dragging
            or not layer_id
            or handle
            not in {"left", "right", "top", "bottom", "nw", "ne", "sw", "se", "rotate"}
        ):
            return False
        host = getattr(self, "_host_window", None)
        setter = getattr(host, "setRasterImageLayerTransform", None)
        if not callable(setter):
            return False
        full_w = max(1.0, float(self._content_size[0] or 1))
        full_h = max(1.0, float(self._content_size[1] or 1))
        start_tx = float(self._raster_overlay_arrow_start_tx)
        start_ty = float(self._raster_overlay_arrow_start_ty)
        start_sx = max(1e-6, float(self._raster_overlay_arrow_start_sx))
        start_sy = max(1e-6, float(self._raster_overlay_arrow_start_sy))
        start_pos = self._raster_overlay_arrow_start_pos
        if start_pos is None:
            return False
        delta_x = float(scene_pos.x()) - float(start_pos.x())
        delta_y = float(scene_pos.y()) - float(start_pos.y())
        next_tx = start_tx
        next_ty = start_ty
        next_sx = start_sx
        next_sy = start_sy
        next_rotation = float(self._raster_overlay_arrow_start_rotation)
        min_scale = 1e-6
        if handle == "rotate":
            center = QtCore.QPointF(
                start_tx + ((full_w * start_sx) * 0.5),
                start_ty + ((full_h * start_sy) * 0.5),
            )
            start_angle = self._raster_overlay_arrow_start_angle_deg
            if start_angle is None:
                start_angle = self._rotation_angle_deg(center=center, target=start_pos)
                self._raster_overlay_arrow_start_angle_deg = float(start_angle)
            current_angle = self._rotation_angle_deg(center=center, target=scene_pos)
            delta_angle = self._normalize_angle_deg(current_angle - float(start_angle))
            next_rotation = self._normalize_angle_deg(
                float(self._raster_overlay_arrow_start_rotation) + delta_angle
            )
        else:
            m11, m12, m21, m22 = self._overlay_affine_matrix(
                sx=start_sx,
                sy=start_sy,
                rotation_deg=float(self._raster_overlay_arrow_start_rotation),
            )
            ux_len = max(1e-12, math.hypot(m11, m12))
            uy_len = max(1e-12, math.hypot(m21, m22))
            ux_x = m11 / ux_len
            ux_y = m12 / ux_len
            uy_x = m21 / uy_len
            uy_y = m22 / uy_len
            proj_x = (delta_x * ux_x) + (delta_y * ux_y)
            proj_y = (delta_x * uy_x) + (delta_y * uy_y)
            start_width = full_w * ux_len
            start_height = full_h * uy_len
            if start_width <= 0.0 or start_height <= 0.0:
                return False
            if handle == "left":
                next_sx = max(min_scale, start_sx - (proj_x / full_w))
            elif handle == "right":
                next_sx = max(min_scale, start_sx + (proj_x / full_w))
            elif handle == "top":
                next_sy = max(min_scale, start_sy - (proj_y / full_h))
            elif handle == "bottom":
                next_sy = max(min_scale, start_sy + (proj_y / full_h))
            else:
                is_east = handle in {"ne", "se"}
                is_south = handle in {"sw", "se"}
                ratio_x = 1.0 + (
                    (proj_x / start_width) if is_east else (-proj_x / start_width)
                )
                ratio_y = 1.0 + (
                    (proj_y / start_height) if is_south else (-proj_y / start_height)
                )
                ratio = ratio_x if abs(ratio_x - 1.0) >= abs(ratio_y - 1.0) else ratio_y
                ratio = max(min_scale, ratio)
                next_sx = start_sx * ratio
                next_sy = start_sy * ratio

            anchor_local_x = 0.0
            anchor_local_y = 0.0
            if handle == "left":
                anchor_local_x = full_w
                anchor_local_y = full_h * 0.5
            elif handle == "right":
                anchor_local_x = 0.0
                anchor_local_y = full_h * 0.5
            elif handle == "top":
                anchor_local_x = full_w * 0.5
                anchor_local_y = full_h
            elif handle == "bottom":
                anchor_local_x = full_w * 0.5
                anchor_local_y = 0.0
            elif handle == "nw":
                anchor_local_x = full_w
                anchor_local_y = full_h
            elif handle == "ne":
                anchor_local_x = 0.0
                anchor_local_y = full_h
            elif handle == "sw":
                anchor_local_x = full_w
                anchor_local_y = 0.0
            elif handle == "se":
                anchor_local_x = 0.0
                anchor_local_y = 0.0

            anchor_scene = self._overlay_scene_from_local(
                local_x=anchor_local_x,
                local_y=anchor_local_y,
                tx=start_tx,
                ty=start_ty,
                sx=start_sx,
                sy=start_sy,
                rotation_deg=float(self._raster_overlay_arrow_start_rotation),
            )
            n11, n12, n21, n22 = self._overlay_affine_matrix(
                sx=next_sx,
                sy=next_sy,
                rotation_deg=float(self._raster_overlay_arrow_start_rotation),
            )
            center_local_x = full_w * 0.5
            center_local_y = full_h * 0.5
            anchor_dx = anchor_local_x - center_local_x
            anchor_dy = anchor_local_y - center_local_y
            center_scene_x = float(anchor_scene.x()) - (
                (n11 * anchor_dx) + (n21 * anchor_dy)
            )
            center_scene_y = float(anchor_scene.y()) - (
                (n12 * anchor_dx) + (n22 * anchor_dy)
            )
            next_tx = center_scene_x - ((full_w * next_sx) * 0.5)
            next_ty = center_scene_y - ((full_h * next_sy) * 0.5)
        try:
            changed = bool(
                setter(
                    layer_id,
                    tx=float(next_tx),
                    ty=float(next_ty),
                    sx=float(next_sx),
                    sy=float(next_sy),
                    rotation_deg=float(next_rotation),
                )
            )
        except Exception:
            changed = False
        if changed:
            self.viewport().update()
        return changed

    def _end_raster_overlay_arrow_drag(self) -> None:
        if not self._raster_overlay_arrow_dragging:
            return
        self._raster_overlay_arrow_dragging = False
        self._raster_overlay_arrow_handle = None
        self._raster_overlay_arrow_layer_id = None
        self._raster_overlay_arrow_start_pos = None
        self._raster_overlay_arrow_start_tx = 0.0
        self._raster_overlay_arrow_start_ty = 0.0
        self._raster_overlay_arrow_start_sx = 1.0
        self._raster_overlay_arrow_start_sy = 1.0
        self._raster_overlay_arrow_start_rotation = 0.0
        self._raster_overlay_arrow_start_angle_deg = None
        self._notify_host_large_image_document_changed()
        self.viewport().update()

    def _start_raster_overlay_drag(
        self, layer_id: str, scene_pos: QtCore.QPointF
    ) -> bool:
        runtime = self._raster_overlay_runtime(layer_id)
        if runtime is None:
            return False
        host = getattr(self, "_host_window", None)
        ensure_context = getattr(host, "_ensure_raster_alignment_context", None)
        if callable(ensure_context):
            try:
                ensure_context(layer_id)
            except Exception:
                pass
        self._dragging_raster_overlay = True
        self._raster_overlay_drag_layer_id = str(layer_id or "")
        self._raster_overlay_drag_start_pos = QtCore.QPointF(scene_pos)
        self._raster_overlay_drag_start_tx = float(runtime.tx)
        self._raster_overlay_drag_start_ty = float(runtime.ty)
        self._active_shape = None
        self._active_vertex_index = None
        self._active_edge_index = None
        self._dragging_shape = False
        self._dragging_shared_boundary = False
        self.overrideCursor(CURSOR_MOVE)
        self._set_hover_feedback(self.tr("Drag to align raster overlay"))
        return True

    def _apply_raster_overlay_drag(self, scene_pos: QtCore.QPointF) -> bool:
        layer_id = str(self._raster_overlay_drag_layer_id or "")
        runtime = self._raster_overlay_runtime(layer_id)
        if (
            not layer_id
            or runtime is None
            or self._raster_overlay_drag_start_pos is None
        ):
            return False
        delta_x = float(scene_pos.x()) - float(self._raster_overlay_drag_start_pos.x())
        delta_y = float(scene_pos.y()) - float(self._raster_overlay_drag_start_pos.y())
        next_tx = float(self._raster_overlay_drag_start_tx) + delta_x
        next_ty = float(self._raster_overlay_drag_start_ty) + delta_y
        host = getattr(self, "_host_window", None)
        setter = getattr(host, "setRasterImageLayerTransform", None)
        if not callable(setter):
            return False
        try:
            moved = bool(
                setter(
                    layer_id,
                    tx=next_tx,
                    ty=next_ty,
                    sx=float(runtime.sx),
                    sy=float(runtime.sy),
                )
            )
        except Exception:
            moved = False
        if not moved:
            return False
        return True

    def _end_raster_overlay_drag(self) -> None:
        if not self._dragging_raster_overlay:
            return
        self._dragging_raster_overlay = False
        self._raster_overlay_drag_layer_id = None
        self._raster_overlay_drag_start_pos = None
        self._raster_overlay_drag_start_tx = 0.0
        self._raster_overlay_drag_start_ty = 0.0
        self._notify_host_large_image_document_changed()

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
        if (
            self._selected_vertex_shape is not None
            and id(self._selected_vertex_shape) not in selected_ids
        ):
            self._set_selected_vertex(None, None)
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

    def _set_selected_vertex(
        self, shape, index: int | None, *, apply_highlight: bool = True
    ) -> None:
        previous_shape = self._selected_vertex_shape
        if shape is None or index is None:
            self._selected_vertex_shape = None
            self._selected_vertex_index = None
            if previous_shape is not None and previous_shape is not shape:
                try:
                    previous_shape.highlightClear()
                except Exception:
                    pass
            self.vertexSelected.emit(False)
            return
        points = list(getattr(shape, "points", []) or [])
        normalized = int(index)
        if normalized < 0 or normalized >= len(points):
            self._selected_vertex_shape = None
            self._selected_vertex_index = None
            if previous_shape is not None and previous_shape is not shape:
                try:
                    previous_shape.highlightClear()
                except Exception:
                    pass
            self.vertexSelected.emit(False)
            return
        if previous_shape is not None and previous_shape is not shape:
            try:
                previous_shape.highlightClear()
            except Exception:
                pass
        self._selected_vertex_shape = shape
        self._selected_vertex_index = normalized
        if apply_highlight:
            try:
                shape.highlightVertex(normalized, shape.MOVE_VERTEX)
            except Exception:
                pass
        self.vertexSelected.emit(True)

    def selectedVertex(self) -> bool:
        shape = self._selected_vertex_shape
        index = self._selected_vertex_index
        if shape is None or index is None:
            return False
        if not any(item is shape for item in self._shapes):
            return False
        points = list(getattr(shape, "points", []) or [])
        return 0 <= int(index) < len(points)

    def _boundary_polygon_source(self):
        if (
            self._active_shape is not None
            and str(getattr(self._active_shape, "shape_type", "") or "").lower()
            == "polygon"
            and len(getattr(self._active_shape, "points", []) or []) >= 2
            and self._active_edge_index is not None
        ):
            return self._active_shape, self._active_edge_index
        return None, None

    def _selected_polygon_source(self):
        candidates = list(self.selectedShapes or [])
        if not candidates:
            host = getattr(self, "_host_window", None)
            canvas = getattr(host, "canvas", None) if host is not None else None
            candidates = list(getattr(canvas, "selectedShapes", []) or [])
        for item in candidates:
            if (
                str(getattr(item, "shape_type", "") or "").lower() == "polygon"
                and len(getattr(item, "points", []) or []) >= 2
            ):
                return item
        return None

    def _selected_shared_boundary_candidate(self):
        shape = self._selected_polygon_source()
        if shape is None:
            return None
        registry = getattr(self, "_shared_topology_registry", None)
        if not isinstance(registry, SharedTopologyRegistry):
            return None
        for edge_id in list(getattr(shape, "shared_edge_ids", []) or []):
            if not edge_id:
                continue
            try:
                if len(registry.edge_occurrences(edge_id)) >= 2:
                    return shape
            except Exception:
                continue
        return None

    def _adjoining_boundary_source(self):
        source = getattr(self, "_adjoining_source_shape", None)
        if source is not None and source in (self._shapes or []):
            return source
        return self._selected_polygon_source()

    def _clear_adjoining_source(self):
        self._adjoining_source_shape = None

    def canStartAdjoiningPolygon(self) -> bool:
        return self._selected_polygon_source() is not None

    def _shared_boundary_source(self):
        shape = self._active_shape
        edge_index = self._active_edge_index
        if shape is None or edge_index is None:
            return None, None
        if str(getattr(shape, "shape_type", "") or "").lower() != "polygon":
            return None, None
        try:
            edge_id = shape.shared_edge_id(int(edge_index))
        except Exception:
            edge_id = None
        if not edge_id:
            return None, None
        registry = getattr(self, "_shared_topology_registry", None)
        if isinstance(registry, SharedTopologyRegistry):
            if len(registry.edge_occurrences(edge_id)) < 2:
                return None, None
        return shape, int(edge_index)

    def canStartSharedBoundaryReshape(self) -> bool:
        shape, edge_index = self._shared_boundary_source()
        if shape is not None and edge_index is not None:
            return True
        return self._selected_shared_boundary_candidate() is not None

    def startSharedBoundaryReshape(self) -> bool:
        shape, edge_index = self._shared_boundary_source()
        if shape is None or edge_index is None:
            shape = self._selected_shared_boundary_candidate()
            edge_index = None
        if shape is None:
            return False
        self._shared_boundary_reshape_mode = True
        self._shared_boundary_shape = shape
        self._shared_boundary_edge_index = (
            int(edge_index) if edge_index is not None else None
        )
        self._dragging_shared_boundary = False
        self._shared_boundary_last_pos = None
        self.overrideCursor(CURSOR_MOVE)
        self._set_hover_feedback(self.tr("Drag shared boundary to reshape"))
        self._notify_host_large_image_document_changed()
        return True

    def _clearSharedBoundaryReshape(self) -> None:
        self._shared_boundary_reshape_mode = False
        self._shared_boundary_shape = None
        self._shared_boundary_edge_index = None
        self._dragging_shared_boundary = False
        self._shared_boundary_last_pos = None

    def _reshapeSharedBoundaryBy(self, delta) -> bool:
        shape = self._shared_boundary_shape
        edge_index = self._shared_boundary_edge_index
        if shape is None or edge_index is None:
            return False
        if not self._shared_reshape_boundary(shape, int(edge_index), delta):
            return False
        self._refresh_overlay_geometry()
        return True

    def adjoiningPolygonSeed(self, edge_index=None):
        _ = edge_index
        return None

    def beginAdjoiningPolygonFromSeed(self, seed_shape) -> bool:
        if seed_shape is None:
            return False
        seed = seed_shape
        self.current = seed
        self.current.setOpen()
        self.current.fill = bool(getattr(seed_shape, "fill", False))
        try:
            self.current.highlightClear()
        except Exception:
            pass
        self._active_shape = None
        self._active_vertex_index = None
        self._active_edge_index = None
        self._set_selected_vertex(None, None)
        self._apply_selection([], emit_signal=True)
        self.mode = self.CREATE
        self.createMode = "polygon"
        self.overrideCursor(CURSOR_DRAW)
        last_point = QtCore.QPointF(seed.points[-1])
        self.current.addPoint(QtCore.QPointF(last_point))
        self._adjoining_default_point_pending = True
        self._update_drawing_preview(last_point)
        self.drawingPolygon.emit(True)
        self._notify_host_large_image_document_changed()
        return True

    def startAdjoiningPolygonFromSelection(self, edge_index=None) -> bool:
        shape = self._selected_polygon_source()
        if shape is None:
            return False
        seed = self._shared_adjoining_seed_for_shape(shape, edge_index=edge_index)
        if seed is None:
            return False
        self._adjoining_source_shape = shape
        self._adjoining_default_point_pending = False
        self.beginAdjoiningPolygonFromSeed(seed)
        return True

    def removeSelectedPoint(self) -> bool:
        shape = self._selected_vertex_shape
        index = self._selected_vertex_index
        if shape is None or index is None:
            return False
        points = list(getattr(shape, "points", []) or [])
        normalized = int(index)
        if normalized < 0 or normalized >= len(points):
            return False
        if not self._shared_remove_vertex(shape, normalized):
            return False
        updated_points = list(getattr(shape, "points", []) or [])
        try:
            shape.highlightClear()
        except Exception:
            pass
        if not updated_points:
            self._shapes = [item for item in self._shapes if item is not shape]
            self._set_selected_vertex(None, None)
            selected = [item for item in self.selectedShapes if item is not shape]
            self._apply_selection(selected, emit_signal=True)
            self.set_shapes(self._shapes)
            self.shapeMoved.emit()
            return True
        next_index = min(normalized, len(updated_points) - 1)
        self._set_selected_vertex(shape, next_index)
        if not any(item is shape for item in self.selectedShapes):
            self._apply_selection([shape], emit_signal=True)
        else:
            self.set_shapes(self._shapes)
        self.shapeMoved.emit()
        return True

    def setShapeVisible(self, shape, value, *, emit_selection: bool = True):
        visible_flag = bool(value)
        try:
            shape.visible = visible_flag
        except Exception:
            pass
        matched_item = None
        for item in list(self._overlay_items or []):
            if getattr(item, "_ann_shape", None) is shape:
                matched_item = item
                item.setVisible(visible_flag)
                # Only sync geometry when showing; syncing on hidden items can
                # cause crashes in Qt's graphics system (prepareGeometryChange
                # on invisible items is unsafe).
                if visible_flag:
                    try:
                        item.sync_shape_geometry()
                    except Exception:
                        pass
                break
        if visible_flag and matched_item is None:
            item = self._make_overlay_item(shape)
            if item is not None:
                self._scene.addItem(item)
                self._overlay_items.append(item)
                matched_item = item
        if not visible_flag:
            try:
                shape.highlightClear()
            except Exception:
                pass
            selected_changed = False
            next_selected = []
            for selected_shape in list(self.selectedShapes or []):
                if selected_shape is shape:
                    selected_changed = True
                    continue
                next_selected.append(selected_shape)
            if selected_changed:
                self.selectedShapes = next_selected
                if emit_selection:
                    try:
                        self.selectionChanged.emit(list(self.selectedShapes))
                    except Exception:
                        pass
                if self._selected_vertex_shape is shape:
                    self._set_selected_vertex(None, None)
        if visible_flag:
            self._refresh_overlay_geometry()
            self._notify_host_large_image_document_changed()
        else:
            # Avoid immediate geometry sync on hide; deferred repaint is safer
            # for Qt's graphics scene when visibility toggles happen rapidly.
            try:
                viewport = self.viewport()
                if viewport is not None:
                    viewport.update()
            except Exception:
                pass
            self._defer_host_large_image_document_changed()

    def _scene_pos_from_event(self, event) -> QtCore.QPointF:
        pos = event.pos() if hasattr(event, "pos") else event.position().toPoint()
        return self.mapToScene(pos)

    def _scene_contains(self, point: QtCore.QPointF) -> bool:
        return self.sceneRect().contains(point)

    def _close_enough(self, p1: QtCore.QPointF, p2: QtCore.QPointF) -> bool:
        return QtCore.QLineF(p1, p2).length() < (
            self.epsilon / max(self.current_scale(), 0.01)
        )

    def _nearest_shape_vertex(
        self,
        scene_pos: QtCore.QPointF,
        *,
        exclude_shape=None,
    ) -> QtCore.QPointF | None:
        scale = max(self.current_scale(), 0.01)
        epsilon = self.epsilon / scale
        best_point = None
        best_distance = None
        for shape in reversed(list(self._shapes or [])):
            if shape is exclude_shape:
                continue
            if not self._is_editable_shape(shape):
                continue
            points = list(getattr(shape, "points", []) or [])
            if not points:
                continue
            vertex_index = None
            try:
                vertex_index = shape.nearestVertex(scene_pos, epsilon)
            except Exception:
                vertex_index = None
            if vertex_index is None:
                continue
            try:
                candidate = QtCore.QPointF(points[int(vertex_index)])
            except Exception:
                continue
            distance = QtCore.QLineF(scene_pos, candidate).length()
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_point = candidate
        return best_point

    def _adjoining_boundary_feature(
        self, scene_pos: QtCore.QPointF
    ) -> tuple[Shape | None, dict | None]:
        source = self._adjoining_boundary_source()
        if source is None or str(self.createMode or "").lower() != "polygon":
            return None, None
        epsilon = max(1.0, 12.0 / max(self.current_scale(), 0.01))
        feature = getattr(source, "nearest_boundary_feature", None)
        if not callable(feature):
            return source, None
        try:
            boundary = feature(scene_pos, epsilon)
        except Exception:
            boundary = None
        if not isinstance(boundary, dict):
            return source, None
        return source, boundary

    def _drawing_snap_target(self, scene_pos: QtCore.QPointF) -> QtCore.QPointF:
        mode = str(self.createMode or "").lower()
        source, boundary = self._adjoining_boundary_feature(scene_pos)
        if source is not None and boundary is not None and mode == "polygon":
            return QtCore.QPointF(boundary["point"])
        if self.current is None:
            return QtCore.QPointF(scene_pos)
        if mode not in {"polygon", "linestrip"}:
            return QtCore.QPointF(scene_pos)
        close_target = self._polygon_close_target(scene_pos)
        if close_target is not None:
            return QtCore.QPointF(close_target)
        vertex_target = self._nearest_shape_vertex(
            scene_pos, exclude_shape=self.current
        )
        if vertex_target is not None:
            return QtCore.QPointF(vertex_target)
        return QtCore.QPointF(scene_pos)

    def _sync_shared_vertex(self, shape, index, point=None):
        try:
            return self._shared_sync_vertex(shape, index, point=point)
        except Exception:
            return None

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
        pending_default = bool(
            mode == "polygon" and self._adjoining_default_point_pending
        )
        raw_target = (
            QtCore.QPointF(scene_pos)
            if scene_pos is not None
            else QtCore.QPointF(points[-1])
        )
        scene_target = self._drawing_snap_target(raw_target)
        close_target = self._polygon_close_target(scene_target)
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
            if not pending_default:
                path.lineTo(scene_target)
        scale = max(self.current_scale(), 0.01)
        base_color = getattr(self.current, "line_color", None)
        if isinstance(base_color, QtGui.QColor):
            preview_color = QtGui.QColor(base_color)
        else:
            preview_color = QtGui.QColor(0, 255, 255, 235)
        preview_color.setAlpha(max(200, int(preview_color.alpha())))
        pen = QtGui.QPen(preview_color)
        pen.setWidthF(float(max(1, int(round(2.0 / scale)))))
        pen.setCapStyle(QtCore.Qt.RoundCap)
        pen.setJoinStyle(QtCore.Qt.RoundJoin)
        preview_brush = QtGui.QBrush(QtCore.Qt.NoBrush)
        if mode == "polygon" and bool(getattr(self.current, "fill", False)):
            fill_color = getattr(self.current, "fill_color", None)
            if isinstance(fill_color, QtGui.QColor):
                fill = QtGui.QColor(fill_color)
            else:
                fill = QtGui.QColor(preview_color)
            fill.setAlpha(max(50, min(165, int(fill.alpha()))))
            preview_brush = QtGui.QBrush(fill)
            if not close_target and len(points) >= 2:
                fill_path = QtGui.QPainterPath(points[0])
                for point in points[1:]:
                    fill_path.lineTo(point)
                fill_path.lineTo(scene_target)
                fill_path.closeSubpath()
                path = fill_path
        self._preview_item.setPen(pen)
        self._preview_item.setBrush(preview_brush)
        self._preview_item.setPath(path)
        vertex_path = QtGui.QPainterPath()
        point_size = float(getattr(self.current, "point_size", Shape.point_size))
        vertex_radius = max(1.5, point_size / (2.0 * scale))
        for point in points:
            vertex_path.addEllipse(point, vertex_radius, vertex_radius)
        if (
            mode in {"polygon", "linestrip"}
            and scene_pos is not None
            and not pending_default
        ):
            vertex_path.addEllipse(
                scene_target, vertex_radius * 0.75, vertex_radius * 0.75
            )
        vertex_pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 245))
        vertex_pen.setWidthF(float(max(1, int(round(1.2 / scale)))))
        self._preview_vertices_item.setPen(vertex_pen)
        self._preview_vertices_item.setBrush(QtGui.QBrush(preview_color))
        self._preview_vertices_item.setPath(vertex_path)
        close_path = QtGui.QPainterPath()
        if close_target is not None and points:
            close_radius = vertex_radius * 1.9
            close_path.addEllipse(points[0], close_radius, close_radius)
            close_pen = QtGui.QPen(QtGui.QColor(255, 215, 0, 245))
            close_pen.setWidthF(float(max(1, int(round(2.0 / scale)))))
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
            if self._adjoining_default_point_pending and self.current.points:
                self.current.popPoint()
                self._adjoining_default_point_pending = False
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
        self._adjoining_default_point_pending = False
        self._clear_adjoining_source()
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
        self._clearSharedBoundaryReshape()
        if self.current is not None:
            self.current = None
            self._adjoining_default_point_pending = False
            self._clear_adjoining_source()
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
        self._clearSharedBoundaryReshape()
        if self.current is None:
            return
        if self._adjoining_default_point_pending and len(self.current.points) >= 3:
            self.current.popPoint()
            self._adjoining_default_point_pending = False
            self._update_drawing_preview(self.current.points[-1])
            return
        if getattr(self.current, "points", None):
            self.current.popPoint()
        if not getattr(self.current, "points", None):
            self.current = None
            self._adjoining_default_point_pending = False
            self._clear_adjoining_source()
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
        return self._shared_move_selected_shapes(self.selectedShapes, bounded)

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
        tx: float = 0.0,
        ty: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
        rotation_deg: float = 0.0,
    ) -> TileRenderPlan[TileKey]:
        visible_keys = tuple(
            self._visible_tile_keys_for_backend(
                backend,
                level=level,
                tx=float(tx),
                ty=float(ty),
                sx=float(sx),
                sy=float(sy),
                rotation_deg=float(rotation_deg),
            )
        )
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

    def _scene_scale_for_backend(
        self, backend: LargeImageBackend
    ) -> tuple[float, float]:
        full_w, full_h = self._content_size
        level_w, level_h = backend.get_level_shape(0)
        scale_x = float(full_w) / max(1, int(level_w))
        scale_y = float(full_h) / max(1, int(level_h))
        return scale_x, scale_y

    def _tile_scene_metrics(
        self,
        backend: LargeImageBackend,
        key: TileKey,
        *,
        tx: float = 0.0,
        ty: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
        rotation_deg: float = 0.0,
    ) -> tuple[QtCore.QPointF, QtGui.QTransform]:
        full_w, full_h = self._content_size
        level_w, level_h = backend.get_level_shape(key.level)
        base_scale_x = full_w / max(1, level_w)
        base_scale_y = full_h / max(1, level_h)
        scale_x = float(base_scale_x) * float(sx)
        scale_y = float(base_scale_y) * float(sy)
        theta = math.radians(float(rotation_deg))
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        m11 = scale_x * cos_t
        m12 = scale_x * sin_t
        m21 = -scale_y * sin_t
        m22 = scale_y * cos_t
        center_scene_x = float(tx) + (float(full_w) * float(sx) * 0.5)
        center_scene_y = float(ty) + (float(full_h) * float(sy) * 0.5)
        center_level_x = float(level_w) * 0.5
        center_level_y = float(level_h) * 0.5
        t_x = center_scene_x - ((m11 * center_level_x) + (m21 * center_level_y))
        t_y = center_scene_y - ((m12 * center_level_x) + (m22 * center_level_y))
        tile_origin_x = float(key.tx * self.tile_size)
        tile_origin_y = float(key.ty * self.tile_size)
        position = QtCore.QPointF(
            t_x + (m11 * tile_origin_x) + (m21 * tile_origin_y),
            t_y + (m12 * tile_origin_x) + (m22 * tile_origin_y),
        )
        transform = QtGui.QTransform(m11, m12, m21, m22, 0.0, 0.0)
        return position, transform

    def _update_existing_tile_items(
        self,
        *,
        backend: LargeImageBackend,
        item_map: dict[TileKey, QtWidgets.QGraphicsPixmapItem],
        z_value_for_key,
        tx: float = 0.0,
        ty: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
        rotation_deg: float = 0.0,
        item_opacity: float | None = None,
    ) -> None:
        for key, item in list(item_map.items()):
            position, transform = self._tile_scene_metrics(
                backend,
                key,
                tx=tx,
                ty=ty,
                sx=sx,
                sy=sy,
                rotation_deg=rotation_deg,
            )
            item.setPos(position)
            item.setTransform(transform)
            item.setZValue(float(z_value_for_key(key)))
            if item_opacity is not None:
                item.setOpacity(max(0.0, min(1.0, float(item_opacity))))

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
        rotation_deg: float = 0.0,
        item_opacity: float | None = None,
    ) -> None:
        for key in visible_keys:
            if key in item_map:
                continue
            cached = tile_images.get(key)
            if cached is None:
                continue
            item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(cached))
            position, transform = self._tile_scene_metrics(
                backend,
                key,
                tx=tx,
                ty=ty,
                sx=sx,
                sy=sy,
                rotation_deg=rotation_deg,
            )
            item.setPos(position)
            item.setTransform(transform)
            item.setZValue(float(z_value_for_key(key)))
            if item_opacity is not None:
                item.setOpacity(max(0.0, min(1.0, float(item_opacity))))
            self._scene.addItem(item)
            item_map[key] = item

    def _visible_tile_keys_for_backend(
        self,
        backend: LargeImageBackend | None,
        *,
        level: int = 0,
        tx: float = 0.0,
        ty: float = 0.0,
        sx: float = 1.0,
        sy: float = 1.0,
        rotation_deg: float = 0.0,
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
        scene_scale_x = max(1e-12, float(scale_x) * float(sx))
        scene_scale_y = max(1e-12, float(scale_y) * float(sy))
        if abs(float(rotation_deg)) > 1e-9:
            theta = math.radians(float(rotation_deg))
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            m11 = scene_scale_x * cos_t
            m12 = scene_scale_x * sin_t
            m21 = -scene_scale_y * sin_t
            m22 = scene_scale_y * cos_t
            center_scene_x = float(tx) + (float(full_w) * float(sx) * 0.5)
            center_scene_y = float(ty) + (float(full_h) * float(sy) * 0.5)
            center_level_x = float(level_w) * 0.5
            center_level_y = float(level_h) * 0.5
            t_x = center_scene_x - ((m11 * center_level_x) + (m21 * center_level_y))
            t_y = center_scene_y - ((m12 * center_level_x) + (m22 * center_level_y))
            det = (m11 * m22) - (m12 * m21)
            if abs(det) < 1e-12:
                return []
            inv11 = m22 / det
            inv12 = -m12 / det
            inv21 = -m21 / det
            inv22 = m11 / det
            corners = [
                rect.topLeft(),
                rect.topRight(),
                rect.bottomLeft(),
                rect.bottomRight(),
            ]
            level_points: list[QtCore.QPointF] = []
            for corner in corners:
                qx = float(corner.x()) - t_x
                qy = float(corner.y()) - t_y
                px = (inv11 * qx) + (inv21 * qy)
                py = (inv12 * qx) + (inv22 * qy)
                level_points.append(QtCore.QPointF(px, py))
            min_x = min(point.x() for point in level_points)
            max_x = max(point.x() for point in level_points)
            min_y = min(point.y() for point in level_points)
            max_y = max(point.y() for point in level_points)
            max_tx = max(0, (level_w - 1) // self.tile_size)
            max_ty = max(0, (level_h - 1) // self.tile_size)
            left = max(0, int(math.floor(min_x / self.tile_size)))
            top = max(0, int(math.floor(min_y / self.tile_size)))
            right = min(max_tx, max(0, int(math.floor(max_x / self.tile_size))))
            bottom = min(max_ty, max(0, int(math.floor(max_y / self.tile_size))))
            return [
                TileKey(level=level, tx=tx_idx, ty=ty_idx)
                for ty_idx in range(top, bottom + 1)
                for tx_idx in range(left, right + 1)
            ]
        max_tx = max(0, (level_w - 1) // self.tile_size)
        max_ty = max(0, (level_h - 1) // self.tile_size)
        left = max(0, int((rect.left() - float(tx)) / scene_scale_x) // self.tile_size)
        top = max(0, int((rect.top() - float(ty)) / scene_scale_y) // self.tile_size)
        right = min(
            max_tx,
            max(0, int((rect.right() - float(tx)) / scene_scale_x) // self.tile_size),
        )
        bottom = min(
            max_ty,
            max(0, int((rect.bottom() - float(ty)) / scene_scale_y) // self.tile_size),
        )
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
        if raster_ready and self.backend is not None and self._base_raster_visible:
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
                    rotation_deg=0.0,
                )
                updated = True
        for runtime in self._raster_overlay_layers.values():
            scheduler = runtime.tile_scheduler
            if scheduler is None:
                continue
            overlay_ready = scheduler.take_completed()
            if not overlay_ready:
                continue
            visible = tuple(
                key for key in runtime.current_visible_keys if key in overlay_ready
            )
            if not visible:
                continue
            self._apply_tile_images(
                backend=runtime.backend,
                tile_images={key: overlay_ready[key] for key in visible},
                visible_keys=visible,
                item_map=runtime.tile_items,
                z_value_for_key=lambda key, z=runtime.z_index: float(z)
                + (float(key.level) * -0.05),
                tx=float(runtime.tx),
                ty=float(runtime.ty),
                sx=float(runtime.sx),
                sy=float(runtime.sy),
                rotation_deg=float(runtime.rotation_deg),
                item_opacity=runtime.opacity,
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
                    rotation_deg=0.0,
                )
                updated = True
        if updated:
            self.viewport().update()
            self._notify_host_large_image_document_changed()
        stats = self.tile_scheduler_stats()
        if (
            not int(stats["raster"].get("outstanding_requests", 0))
            and not int(stats["label"].get("outstanding_requests", 0))
            and not int(stats["raster_overlays"].get("outstanding_requests", 0))
        ):
            self._tile_result_timer.stop()
        self._refresh_status_overlay()

    def _refresh_visible_tiles_now(self) -> None:
        if self.backend is None:
            self._last_visible_tile_count = 0
            self._current_visible_raster_keys = ()
            self._refresh_status_overlay()
            return
        if not self._base_raster_visible:
            self._last_visible_tile_count = 0
            self._current_visible_raster_keys = ()
            self._remove_stale_tile_items(
                self._tile_items, list(self._tile_items.keys())
            )
            self._refresh_visible_raster_overlay_tiles()
            self._refresh_visible_label_tiles()
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
            rotation_deg=0.0,
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
            rotation_deg=0.0,
        )
        if (
            int(self.tile_scheduler_stats()["raster"].get("outstanding_requests", 0))
            > 0
        ):
            self._tile_result_timer.start()
        self._refresh_visible_raster_overlay_tiles()
        self._refresh_visible_label_tiles()
        self._refresh_status_overlay()

    def refresh_visible_tiles(self) -> None:
        self._refresh_visible_tiles_now()

    def _refresh_visible_raster_overlay_tiles(self) -> None:
        for runtime in self._raster_overlay_layers.values():
            if not runtime.visible:
                self._remove_stale_tile_items(
                    runtime.tile_items,
                    list(runtime.tile_items.keys()),
                )
                runtime.last_level = 0
                runtime.last_visible_tile_count = 0
                runtime.current_visible_keys = ()
                continue
            backend = runtime.backend
            if backend is None:
                continue
            if not self._content_size[0] or not self._content_size[1]:
                continue
            # Unlike the base raster, overlays do not have a thumbnail pixmap
            # fallback. Keep scheduling visible tiles at low zoom so a checked
            # overlay still appears when the base TIFF is hidden.
            level = self._select_level_for_backend(backend)
            plan = self._build_tile_render_plan(
                backend=backend,
                level=level,
                current_items=runtime.tile_items,
                tx=float(runtime.tx),
                ty=float(runtime.ty),
                sx=float(runtime.sx),
                sy=float(runtime.sy),
                rotation_deg=float(runtime.rotation_deg),
            )
            runtime.last_level = int(level)
            runtime.last_visible_tile_count = len(plan.visible_keys)
            ordered_visible = self._prioritize_visible_keys(
                list(plan.visible_keys), backend=backend
            )
            runtime.current_visible_keys = ordered_visible
            self._remove_stale_tile_items(runtime.tile_items, plan.stale_keys)
            scheduler = runtime.tile_scheduler
            if scheduler is None:
                continue
            overlay_images = scheduler.schedule(
                ordered_visible,
                prefetch_keys=plan.prefetch_keys,
                prime_keys=ordered_visible[:2],
            )
            overlay_images.update(
                {
                    key: cached
                    for key in ordered_visible
                    if (cached := runtime.tile_cache.get(key)) is not None
                }
            )
            self._apply_tile_images(
                backend=backend,
                tile_images=overlay_images,
                visible_keys=ordered_visible,
                item_map=runtime.tile_items,
                z_value_for_key=lambda key, z=runtime.z_index: float(z)
                + (float(key.level) * -0.05),
                tx=float(runtime.tx),
                ty=float(runtime.ty),
                sx=float(runtime.sx),
                sy=float(runtime.sy),
                rotation_deg=float(runtime.rotation_deg),
                item_opacity=runtime.opacity,
            )
            if int(scheduler.stats().outstanding_requests) > 0:
                self._tile_result_timer.start()

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
            tx=float(self._label_transform["tx"]),
            ty=float(self._label_transform["ty"]),
            sx=float(self._label_transform["sx"]),
            sy=float(self._label_transform["sy"]),
            rotation_deg=0.0,
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
            rotation_deg=0.0,
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

    def _load_raster_overlay_tile(self, layer_id: str, key: TileKey) -> QtGui.QImage:
        runtime = self._raster_overlay_layers.get(str(layer_id or ""))
        if runtime is None:
            return QtGui.QImage()
        backend = runtime.backend
        x = key.tx * self.tile_size
        y = key.ty * self.tile_size
        region = backend.read_region(
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
        overlay_outstanding = 0
        overlay_hits = 0
        overlay_misses = 0
        for runtime in self._raster_overlay_layers.values():
            scheduler = runtime.tile_scheduler
            if scheduler is None:
                continue
            stats = scheduler.stats()
            overlay_outstanding += int(stats.outstanding_requests)
            overlay_hits += int(stats.cache_hits)
            overlay_misses += int(stats.cache_misses)
        return {
            "raster": dict(raster.__dict__),
            "label": dict(label.__dict__),
            "raster_overlays": {
                "outstanding_requests": int(overlay_outstanding),
                "cache_hits": int(overlay_hits),
                "cache_misses": int(overlay_misses),
            },
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
        overlay_stats = dict(stats.get("raster_overlays", {}) or {})
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
                "raster_overlay_layers=%d overlay_pending=%d overlay_hits=%d overlay_misses=%d"
                % (
                    len(self._raster_overlay_layers),
                    int(overlay_stats.get("outstanding_requests", 0)),
                    int(overlay_stats.get("cache_hits", 0)),
                    int(overlay_stats.get("cache_misses", 0)),
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

    def _scene_xy_text(self, scene_pos: QtCore.QPointF) -> str:
        return f"x:{float(scene_pos.x()):.1f},y:{float(scene_pos.y()):.1f}"

    def _set_hover_feedback(self, tooltip: str, *, status: str | None = None) -> None:
        self.setToolTip(str(tooltip or ""))
        try:
            self.setStatusTip(str(tooltip or ""))
        except Exception:
            pass
        if status is None:
            status = tooltip
        host = getattr(self, "_host_window", None)
        post_status = getattr(host, "_post_window_status", None) if host else None
        if callable(post_status) and status:
            post_status(str(status), 1500)

    def drawForeground(self, painter, rect):  # pragma: no cover - paint path
        super().drawForeground(painter, rect)
        if not bool(self._raster_overlay_arrow_mode):
            return
        selected = self._selected_raster_overlay_runtime()
        if selected is None:
            return
        _layer_id, runtime = selected
        if not bool(getattr(runtime, "visible", True)):
            return
        corners = self._raster_overlay_scene_corners(runtime)
        nw = corners["nw"]
        ne = corners["ne"]
        se = corners["se"]
        sw = corners["sw"]
        handles = self._raster_overlay_arrow_handles(runtime)
        pen = QtGui.QPen(QtGui.QColor("#2aa3ff"))
        pen.setWidthF(1.5)
        pen.setCosmetic(True)
        painter.save()
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setPen(pen)
        painter.setBrush(QtGui.QBrush(QtGui.QColor(42, 163, 255, 80)))
        outline = QtGui.QPolygonF([nw, ne, se, sw])
        painter.drawPolygon(outline)
        top_center = handles.get("top", {}).get("center")
        rotate_center = handles.get("rotate", {}).get("center")
        if isinstance(top_center, QtCore.QPointF) and isinstance(
            rotate_center, QtCore.QPointF
        ):
            painter.drawLine(top_center, rotate_center)
        for handle_name, payload in handles.items():
            center = payload.get("center")
            if not isinstance(center, QtCore.QPointF):
                continue
            scale = max(1e-6, float(self.current_scale() or 1.0))
            arrow_size = 9.0 * (1.0 / scale)
            path = QtGui.QPainterPath()
            if handle_name == "left":
                path.moveTo(center.x() - arrow_size, center.y())
                path.lineTo(center.x() + arrow_size, center.y() - (arrow_size * 0.75))
                path.lineTo(center.x() + arrow_size, center.y() + (arrow_size * 0.75))
            elif handle_name == "right":
                path.moveTo(center.x() + arrow_size, center.y())
                path.lineTo(center.x() - arrow_size, center.y() - (arrow_size * 0.75))
                path.lineTo(center.x() - arrow_size, center.y() + (arrow_size * 0.75))
            elif handle_name == "top":
                path.moveTo(center.x(), center.y() - arrow_size)
                path.lineTo(center.x() - (arrow_size * 0.75), center.y() + arrow_size)
                path.lineTo(center.x() + (arrow_size * 0.75), center.y() + arrow_size)
            elif handle_name == "bottom":
                path.moveTo(center.x(), center.y() + arrow_size)
                path.lineTo(center.x() - (arrow_size * 0.75), center.y() - arrow_size)
                path.lineTo(center.x() + (arrow_size * 0.75), center.y() - arrow_size)
            elif handle_name == "rotate":
                painter.setBrush(QtGui.QBrush(QtGui.QColor("#0f87e8")))
                painter.drawEllipse(center, arrow_size * 0.7, arrow_size * 0.7)
                continue
            else:
                d = arrow_size * 0.85
                path.moveTo(center.x(), center.y() - d)
                path.lineTo(center.x() + d, center.y())
                path.lineTo(center.x(), center.y() + d)
                path.lineTo(center.x() - d, center.y())
            path.closeSubpath()
            painter.setBrush(QtGui.QBrush(QtGui.QColor("#2aa3ff")))
            painter.drawPath(path)
        painter.restore()

    def set_zoom_percent(self, percent: int) -> None:
        if not self._content_size[0] or not self._content_size[1]:
            return
        self._fit_mode = "manual"
        self.resetTransform()
        factor = max(0.01, float(percent) / 100.0)
        self.scale(factor, factor)
        self._refresh_overlay_render_metrics()
        self._queue_visible_tiles_refresh()
        self._notify_host_large_image_document_changed()

    def fit_to_window(self) -> None:
        self._fit_mode = "fit_window"
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self._refresh_overlay_render_metrics()
        self._queue_visible_tiles_refresh()
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
        self._queue_visible_tiles_refresh()
        self._notify_host_large_image_document_changed()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode == "fit_window":
            self.fit_to_window()
        elif self._fit_mode == "fit_width":
            self.fit_to_width()
        else:
            self._queue_visible_tiles_refresh()
            self._notify_host_large_image_document_changed()
        self._position_status_overlay()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self._queue_visible_tiles_refresh()
        self._notify_host_large_image_document_changed()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self._refresh_overlay_render_metrics()
        self._queue_visible_tiles_refresh()
        self._notify_host_large_image_document_changed()

    def mousePressEvent(self, event):
        pos = event.pos() if hasattr(event, "pos") else event.position().toPoint()
        scene_pos = self.mapToScene(pos)
        if (
            event.button() == QtCore.Qt.LeftButton
            and self._shared_boundary_reshape_mode
        ):
            shape, vertex_index, hit_kind = self._shape_hit_test(scene_pos)
            if (
                shape is not None
                and hit_kind == "edge"
                and self._shared_boundary_shape is not None
                and shape is self._shared_boundary_shape
            ):
                registry = getattr(self, "_shared_topology_registry", None)
                edge_id = None
                try:
                    edge_id = shape.shared_edge_id(int(vertex_index))
                except Exception:
                    edge_id = None
                if (
                    edge_id
                    and isinstance(registry, SharedTopologyRegistry)
                    and len(registry.edge_occurrences(edge_id)) >= 2
                ):
                    self._shared_boundary_shape = shape
                    self._shared_boundary_edge_index = int(vertex_index)
                    self._dragging_shared_boundary = True
                    self._shared_boundary_last_pos = QtCore.QPointF(scene_pos)
                    self._shape_moved_during_drag = False
                    self.overrideCursor(CURSOR_MOVE)
                    event.accept()
                    return
            event.accept()
            return
        if (
            event.button() == QtCore.Qt.LeftButton
            and not self.drawing()
            and self._selected_raster_overlay_layer_id() is not None
        ):
            if bool(self._raster_overlay_arrow_mode):
                handle = self._hit_raster_overlay_arrow_handle(scene_pos)
                if handle and self._start_raster_overlay_arrow_drag(
                    handle, self._clamp_scene_point(scene_pos)
                ):
                    event.accept()
                    return
            layer_id = self._selected_raster_overlay_layer_id()
            if (
                not bool(self._raster_overlay_arrow_mode)
                and layer_id
                and self._start_raster_overlay_drag(
                    layer_id, self._clamp_scene_point(scene_pos)
                )
            ):
                event.accept()
                return
        if event.button() == QtCore.Qt.LeftButton and self.drawing():
            if not self._supports_create_mode(
                self.createMode
            ) or not self._scene_contains(scene_pos):
                event.accept()
                return
            mode = str(self.createMode or "").lower()
            clamped_scene_pos = self._clamp_scene_point(scene_pos)
            adjoining_source, adjoining_feature = self._adjoining_boundary_feature(
                clamped_scene_pos
            )
            snapped_scene_pos = self._drawing_snap_target(clamped_scene_pos)
            if self.current is None:
                from annolid.gui.shape import Shape

                self.current = Shape(
                    shape_type=mode if mode != "polygon" else "polygon"
                )
                self.current.addPoint(QtCore.QPointF(snapped_scene_pos))
                if mode == "point":
                    self.finalise()
                else:
                    self.drawingPolygon.emit(True)
                    self._update_drawing_preview(snapped_scene_pos)
                event.accept()
                return
            if mode == "polygon":
                if self._polygon_close_target(snapped_scene_pos) is not None:
                    self.finalise()
                else:
                    if self._adjoining_default_point_pending and self.current.points:
                        self.current.points[-1] = QtCore.QPointF(snapped_scene_pos)
                        current_index = len(self.current.points) - 1
                        self._adjoining_default_point_pending = False
                    else:
                        self.current.addPoint(QtCore.QPointF(snapped_scene_pos))
                        current_index = len(self.current.points) - 1
                    if adjoining_feature is not None and adjoining_source is not None:
                        self._shared_link_adjoining_point(
                            self.current,
                            current_index,
                            snapped_scene_pos,
                            adjoining_source,
                            adjoining_feature,
                        )
                    self._update_drawing_preview(snapped_scene_pos)
            elif mode == "linestrip":
                self._adjoining_default_point_pending = False
                self.current.addPoint(QtCore.QPointF(snapped_scene_pos))
                self._update_drawing_preview(snapped_scene_pos)
                if event.modifiers() & QtCore.Qt.ControlModifier:
                    self.finalise()
            elif mode in {"line", "rectangle", "circle"}:
                self._adjoining_default_point_pending = False
                if len(self.current.points) == 1:
                    self.current.addPoint(QtCore.QPointF(snapped_scene_pos))
                else:
                    self.current.points[1] = QtCore.QPointF(snapped_scene_pos)
                self.finalise()
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton:
            shape, vertex_index, hit_kind = self._shape_hit_test(scene_pos)
            if shape is not None:
                if self._shared_boundary_reshape_mode:
                    shared_shape, shared_edge_index = self._shared_boundary_source()
                    if shared_shape is not None and shared_edge_index is not None:
                        self._shared_boundary_shape = shared_shape
                        self._shared_boundary_edge_index = shared_edge_index
                        self._dragging_shared_boundary = True
                        self._shared_boundary_last_pos = QtCore.QPointF(scene_pos)
                        self.overrideCursor(CURSOR_MOVE)
                        event.accept()
                        return
                if hit_kind == "edge" and bool(
                    getattr(shape, "canAddPoint", lambda: False)()
                ):
                    insert_index = int(vertex_index)
                    target_point = self._clamp_scene_point(scene_pos)
                    self._shared_insert_vertex_on_edge(
                        shape,
                        insert_index,
                        QtCore.QPointF(target_point),
                    )
                    try:
                        shape.highlightVertex(insert_index, shape.MOVE_VERTEX)
                    except Exception:
                        pass
                    self._active_shape = shape
                    self._active_vertex_index = insert_index
                    self._set_selected_vertex(shape, insert_index)
                    self._dragging_shape = True
                    self._shape_moved_during_drag = True
                    self._last_scene_pos = QtCore.QPointF(target_point)
                    self._apply_selection([shape], emit_signal=True)
                    self._refresh_overlay_geometry()
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
                self._active_edge_index = (
                    int(vertex_index) if hit_kind == "edge" else None
                )
                if hit_kind == "vertex":
                    self._set_selected_vertex(shape, int(vertex_index))
                else:
                    self._set_selected_vertex(None, None)
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
            self._set_selected_vertex(None, None)
            self._active_edge_index = None
            self._apply_selection([], emit_signal=True)
        elif event.button() == QtCore.Qt.RightButton:
            shape, vertex_index, hit_kind = self._shape_hit_test(scene_pos)
            if shape is not None:
                self._active_shape = shape
                self._active_vertex_index = (
                    int(vertex_index) if hit_kind == "vertex" else None
                )
                self._active_edge_index = (
                    int(vertex_index) if hit_kind == "edge" else None
                )
            else:
                self._active_shape = None
                self._active_vertex_index = None
                self._active_edge_index = None
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
        if (
            self._raster_overlay_arrow_dragging
            and self._raster_overlay_arrow_layer_id is not None
            and (event.buttons() & QtCore.Qt.LeftButton)
        ):
            if self._apply_raster_overlay_arrow_drag(
                self._clamp_scene_point(scene_pos)
            ):
                event.accept()
                return
        if (
            self._dragging_raster_overlay
            and self._raster_overlay_drag_layer_id is not None
            and (event.buttons() & QtCore.Qt.LeftButton)
        ):
            if self._apply_raster_overlay_drag(self._clamp_scene_point(scene_pos)):
                event.accept()
                return
        if self.drawing():
            mode = str(self.createMode or "").lower()
            snapped_scene_pos = self._drawing_snap_target(
                self._clamp_scene_point(scene_pos)
            )
            if (
                mode == "polygon"
                and self._adjoining_default_point_pending
                and self.current is not None
                and self.current.points
            ):
                self.current.points[-1] = QtCore.QPointF(snapped_scene_pos)
            if mode == "polygon":
                close_target = self._polygon_close_target(snapped_scene_pos)
                vertex_target = self._nearest_shape_vertex(
                    snapped_scene_pos, exclude_shape=self.current
                )
                self.overrideCursor(
                    CURSOR_POINT if close_target is not None else CURSOR_DRAW
                )
                if close_target is not None:
                    self._set_hover_feedback(
                        self.tr("Click to close polygon"),
                        status=self.tr("Close polygon"),
                    )
                elif vertex_target is not None:
                    self.overrideCursor(CURSOR_POINT)
                    self._set_hover_feedback(
                        self.tr("Click to create point"),
                        status=self.tr("Snap to shared vertex"),
                    )
                else:
                    self._set_hover_feedback(
                        self.tr("Click to create point"),
                        status=self.tr("Add polygon point"),
                    )
            else:
                self.overrideCursor(CURSOR_DRAW)
                self._set_hover_feedback(self.tr("Click to create point"))
            if self.current is not None:
                self._update_drawing_preview(snapped_scene_pos)
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
                self.overrideCursor(CURSOR_POINT)
                old_point = QtCore.QPointF(
                    self._active_shape.points[self._active_vertex_index]
                )
                target = self._clamp_scene_point(old_point + delta)
                bounded = target - old_point
                if abs(bounded.x()) > 1e-8 or abs(bounded.y()) > 1e-8:
                    self._active_shape.moveVertexBy(self._active_vertex_index, bounded)
                    self._sync_shared_vertex(
                        self._active_shape,
                        self._active_vertex_index,
                        self._active_shape.points[self._active_vertex_index],
                    )
                    moved = True
            else:
                self.overrideCursor(CURSOR_MOVE)
                moved = self._bounded_move_selected_shapes(delta)
            if moved:
                self._shape_moved_during_drag = True
                self._last_scene_pos = QtCore.QPointF(scene_pos)
                self._refresh_overlay_geometry()
                event.accept()
                return
        if (
            self._dragging_shared_boundary
            and self._shared_boundary_last_pos is not None
            and (event.buttons() & QtCore.Qt.LeftButton)
        ):
            delta = QtCore.QPointF(
                float(scene_pos.x()) - float(self._shared_boundary_last_pos.x()),
                float(scene_pos.y()) - float(self._shared_boundary_last_pos.y()),
            )
            if self._reshapeSharedBoundaryBy(delta):
                self._shared_boundary_last_pos = QtCore.QPointF(scene_pos)
                self._shape_moved_during_drag = True
                self._refresh_overlay_geometry()
                self.update()
                event.accept()
                return
        if self.editing():
            if bool(self._raster_overlay_arrow_mode):
                handle = self._hit_raster_overlay_arrow_handle(scene_pos)
                if handle in {"left", "right"}:
                    self.overrideCursor(QtCore.Qt.SizeHorCursor)
                    self._set_hover_feedback(
                        self.tr("Drag arrow to resize left/right"),
                        status=self._scene_xy_text(scene_pos),
                    )
                    event.accept()
                    return
                if handle in {"nw", "se"}:
                    self.overrideCursor(QtCore.Qt.SizeFDiagCursor)
                    self._set_hover_feedback(
                        self.tr("Drag corner to resize proportionally"),
                        status=self._scene_xy_text(scene_pos),
                    )
                    event.accept()
                    return
                if handle in {"ne", "sw"}:
                    self.overrideCursor(QtCore.Qt.SizeBDiagCursor)
                    self._set_hover_feedback(
                        self.tr("Drag corner to resize proportionally"),
                        status=self._scene_xy_text(scene_pos),
                    )
                    event.accept()
                    return
                if handle in {"top", "bottom"}:
                    self.overrideCursor(QtCore.Qt.SizeVerCursor)
                    self._set_hover_feedback(
                        self.tr("Drag arrow to resize top/bottom"),
                        status=self._scene_xy_text(scene_pos),
                    )
                    event.accept()
                    return
                if handle == "rotate":
                    self.overrideCursor(QtCore.Qt.OpenHandCursor)
                    self._set_hover_feedback(
                        self.tr("Drag handle to rotate raster overlay"),
                        status=self._scene_xy_text(scene_pos),
                    )
                    event.accept()
                    return
            shape, vertex_index, hit_kind = self._shape_hit_test(scene_pos)
            if shape is not None:
                label = str(getattr(shape, "label", "") or "")
                if hit_kind == "vertex":
                    self.overrideCursor(CURSOR_POINT)
                    self._set_hover_feedback(
                        self.tr("Click & drag to move point"),
                        status=(
                            f"{label},{self._scene_xy_text(scene_pos)}"
                            if label
                            else self._scene_xy_text(scene_pos)
                        ),
                    )
                elif hit_kind == "edge":
                    self.overrideCursor(CURSOR_POINT)
                    self._set_hover_feedback(
                        self.tr("Click to create point"),
                        status=(
                            f"{label},{self._scene_xy_text(scene_pos)}"
                            if label
                            else self._scene_xy_text(scene_pos)
                        ),
                    )
                else:
                    self.overrideCursor(CURSOR_GRAB)
                    self._set_hover_feedback(
                        self.tr("Click & drag to move shape '%s'") % label
                        if label
                        else self.tr("Click & drag to move shape"),
                        status=(
                            f"{label},{self._scene_xy_text(scene_pos)}"
                            if label
                            else self._scene_xy_text(scene_pos)
                        ),
                    )
            else:
                self.restoreCursor()
                self._set_hover_feedback(
                    self.tr("Image"),
                    status=self._scene_xy_text(scene_pos),
                )
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
        if event.button() == QtCore.Qt.LeftButton and self._dragging_raster_overlay:
            self._end_raster_overlay_drag()
            event.accept()
            return
        if (
            event.button() == QtCore.Qt.LeftButton
            and self._raster_overlay_arrow_dragging
        ):
            self._end_raster_overlay_arrow_drag()
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton and self._dragging_shared_boundary:
            moved = bool(self._shape_moved_during_drag)
            self._dragging_shared_boundary = False
            self._shared_boundary_last_pos = None
            self._clearSharedBoundaryReshape()
            self._shared_finalize_topology_edit()
            self._refresh_overlay_geometry()
            if moved:
                self.shapeMoved.emit()
            event.accept()
            return
        if event.button() == QtCore.Qt.LeftButton and self._dragging_shape:
            moved = bool(self._shape_moved_during_drag)
            released_shape = self._active_shape
            released_vertex = self._active_vertex_index
            if self._active_shape is not None:
                try:
                    self._active_shape.highlightClear()
                except Exception:
                    pass
            self._dragging_shape = False
            self._active_shape = None
            self._active_vertex_index = None
            self._active_edge_index = None
            self._last_scene_pos = None
            self._shape_moved_during_drag = False
            if released_shape is not None and released_vertex is not None:
                self._set_selected_vertex(
                    released_shape, int(released_vertex), apply_highlight=False
                )
            if moved:
                self._shared_finalize_topology_edit()
            self._refresh_overlay_geometry()
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
                self.restoreCursor()
                self._clear_adjoining_source()
                self._clearSharedBoundaryReshape()
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
        elif self.editing():
            if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
                if self.selectedVertex() and self.removeSelectedPoint():
                    event.accept()
                    return
                host = getattr(self, "_host_window", None) or self.window()
                delete_fn = getattr(host, "deleteSelectedShapes", None)
                if callable(delete_fn):
                    try:
                        delete_fn()
                        event.accept()
                        return
                    except Exception:
                        pass
        if event.key() == QtCore.Qt.Key_Escape and self._shared_boundary_reshape_mode:
            self._clearSharedBoundaryReshape()
            self.restoreCursor()
            event.accept()
            return
        super().keyPressEvent(event)

    def leaveEvent(self, event):
        self._clearSharedBoundaryReshape()
        self.restoreCursor()
        self.setToolTip(self.tr("Image"))
        super().leaveEvent(event)
