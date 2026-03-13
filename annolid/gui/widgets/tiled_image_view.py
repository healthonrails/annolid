from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from qtpy import QtCore, QtGui, QtWidgets

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
        self.backend: LargeImageBackend | None = None
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._preview_item = QtWidgets.QGraphicsPathItem()
        self._preview_item.setZValue(130.0)
        self._scene.addItem(self._preview_item)
        self._tile_items: dict[TileKey, QtWidgets.QGraphicsPixmapItem] = {}
        self._overlay_items: list[QtWidgets.QGraphicsItem] = []
        self._vertex_items: list[QtWidgets.QGraphicsItem] = []
        self._pair_items: list[QtWidgets.QGraphicsItem] = []
        self._pair_endpoint_items: list[QtWidgets.QGraphicsItem] = []
        self._selected_overlay_landmark_pair_id: str | None = None
        self._shapes = []
        self.selectedShapes = []
        self._active_shape = None
        self._active_vertex_index: int | None = None
        self._dragging_shape = False
        self._shape_moved_during_drag = False
        self._last_scene_pos: QtCore.QPointF | None = None
        self._content_size: tuple[int, int] = (0, 0)
        self._fit_mode: str = "fit_window"
        self.mode = self.EDIT
        self.createMode = "polygon"
        self.current = None
        self.epsilon = 10.0
        self.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setRenderHint(QtGui.QPainter.SmoothPixmapTransform, True)
        self.setMouseTracking(True)

    def set_backend(self, backend: LargeImageBackend) -> None:
        self.backend = backend
        self.tile_cache = TileCache()
        self._clear_tile_items()
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

    def content_size(self) -> tuple[int, int]:
        return self._content_size

    def clear(self) -> None:
        self.backend = None
        self._content_size = (0, 0)
        self.tile_cache = TileCache()
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
        self._clear_tile_items()
        self.set_shapes([])
        self._scene.setSceneRect(QtCore.QRectF())
        self.resetTransform()

    def drawing(self) -> bool:
        return self.mode == self.CREATE

    def editing(self) -> bool:
        return self.mode == self.EDIT

    def setEditing(self, value=True) -> None:
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            self.current = None
            self._preview_item.setPath(QtGui.QPainterPath())
            self.drawingPolygon.emit(False)

    def _supports_create_mode(self, create_mode: str) -> bool:
        return str(create_mode or "").lower() in {
            "point",
            "line",
            "linestrip",
            "polygon",
        }

    def _clear_tile_items(self) -> None:
        for item in self._tile_items.values():
            self._scene.removeItem(item)
        self._tile_items.clear()

    def set_shapes(self, shapes) -> None:
        for item in self._overlay_items:
            self._scene.removeItem(item)
        self._overlay_items = []
        for item in self._vertex_items:
            self._scene.removeItem(item)
        self._vertex_items = []
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
            for vertex_item in self._make_vertex_handle_items(shape):
                self._scene.addItem(vertex_item)
                self._vertex_items.append(vertex_item)

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
            return
        points = list(getattr(self.current, "points", []) or [])
        if not points:
            self._preview_item.setPath(path)
            return
        mode = str(self.createMode or "").lower()
        scene_target = (
            QtCore.QPointF(scene_pos)
            if scene_pos is not None
            else QtCore.QPointF(points[-1])
        )
        path.moveTo(points[0])
        if mode == "point":
            radius = 4.5
            path.addEllipse(points[0], radius, radius)
        elif mode == "line":
            if len(points) == 1:
                path.lineTo(scene_target)
            else:
                path.lineTo(points[1])
        elif mode == "linestrip":
            for point in points[1:]:
                path.lineTo(point)
            path.lineTo(scene_target)
        else:
            for point in points[1:]:
                path.lineTo(point)
            path.lineTo(scene_target)
        pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 240))
        pen.setWidthF(2.0)
        self._preview_item.setPen(pen)
        self._preview_item.setBrush(QtCore.Qt.NoBrush)
        self._preview_item.setPath(path)

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
        elif mode == "line":
            if len(self.current.points) < 2:
                return
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
            self.drawingPolygon.emit(False)
            return
        self._update_drawing_preview(self.current.points[-1])

    def set_selected_shapes(self, shapes) -> None:
        self._apply_selection(shapes, emit_signal=False)

    @staticmethod
    def _is_editable_overlay_shape(shape) -> bool:
        other = dict(getattr(shape, "other_data", {}) or {})
        if "overlay_id" not in other:
            return False
        if not getattr(shape, "visible", True) or not bool(
            other.get("overlay_visible", True)
        ):
            return False
        if bool(other.get("overlay_locked", False)):
            return False
        return bool(getattr(shape, "points", []) or [])

    def _shape_hit_test(self, scene_pos: QtCore.QPointF):
        epsilon = max(4.0, 8.0 / max(self.current_scale(), 0.01))
        for shape in reversed(list(self._shapes or [])):
            if not self._is_editable_overlay_shape(shape):
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
                if edge is not None:
                    return shape, None, "shape"
            else:
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

    def _make_vertex_handle_items(self, shape) -> list[QtWidgets.QGraphicsItem]:
        if shape not in self.selectedShapes:
            return []
        items = []
        highlight_index = getattr(shape, "_highlightIndex", None)
        for index, point in enumerate(getattr(shape, "points", []) or []):
            radius = 6.0 if index == highlight_index else 4.5
            item = QtWidgets.QGraphicsEllipseItem(
                float(point.x()) - radius,
                float(point.y()) - radius,
                radius * 2.0,
                radius * 2.0,
            )
            item.setPen(QtGui.QPen(QtGui.QColor(255, 255, 255, 230), 1.5))
            item.setBrush(
                QtGui.QBrush(
                    QtGui.QColor(255, 140, 0, 220)
                    if index == highlight_index
                    else QtGui.QColor(0, 255, 255, 210)
                )
            )
            item.setZValue(120.0)
            items.append(item)
        return items

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
            else:
                image_points[pair_id] = coords
        items = []
        for pair_id, src in overlay_points.items():
            dst = image_points.get(pair_id)
            if dst is None:
                continue
            line_item = _LandmarkPairItem(pair_id, src[0], src[1], dst[0], dst[1])
            line_item.set_selected(pair_id == self._selected_overlay_landmark_pair_id)
            items.append(line_item)
        return items

    def _make_pair_endpoint_items(self, shapes) -> list[QtWidgets.QGraphicsItem]:
        overlay_points = {}
        image_points = {}
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
            else:
                image_points[pair_id] = point
        items = []
        for pair_id, src in overlay_points.items():
            dst = image_points.get(pair_id)
            if dst is None:
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
        points = [
            QtCore.QPointF(float(p.x()), float(p.y()))
            for p in getattr(shape, "points", [])
        ]
        if not points:
            return None
        stroke = QtGui.QColor(str(other.get("overlay_stroke") or "#00ff00"))
        if not stroke.isValid():
            stroke = QtGui.QColor(0, 255, 0, 220)
        else:
            stroke.setAlpha(220)
        fill = QtGui.QColor(str(other.get("overlay_fill") or "#00ff00"))
        if not fill.isValid():
            fill = QtGui.QColor(0, 255, 0, 40)
        else:
            fill.setAlpha(40)
        pen = QtGui.QPen(stroke)
        pen.setWidthF(2.0)
        brush = QtGui.QBrush(fill)
        shape_type = str(getattr(shape, "shape_type", "") or "").lower()
        if shape in self.selectedShapes or getattr(shape, "selected", False):
            pen = QtGui.QPen(QtGui.QColor(255, 255, 255, 245))
            pen.setWidthF(3.0)
            if shape_type != "point":
                fill = QtGui.QColor(fill)
                fill.setAlpha(max(fill.alpha(), 90))
                brush = QtGui.QBrush(fill)
        item = None
        if shape_type == "point":
            p = points[0]
            item = QtWidgets.QGraphicsEllipseItem(p.x() - 4, p.y() - 4, 8, 8)
            item.setPen(pen)
            item.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 220)))
        elif shape_type == "line":
            if len(points) < 2:
                return None
            item = QtWidgets.QGraphicsLineItem(
                points[0].x(), points[0].y(), points[1].x(), points[1].y()
            )
            item.setPen(pen)
        elif shape_type == "polygon":
            polygon = QtGui.QPolygonF(points)
            item = QtWidgets.QGraphicsPolygonItem(polygon)
            item.setPen(pen)
            item.setBrush(brush)
        else:
            path = QtGui.QPainterPath()
            path.moveTo(points[0])
            for point in points[1:]:
                path.lineTo(point)
            item = QtWidgets.QGraphicsPathItem(path)
            item.setPen(pen)
        item.setOpacity(max(0.0, min(1.0, float(other.get("overlay_opacity", 1.0)))))
        item.setZValue(100.0 + float(other.get("overlay_z_order", 0)))
        return item

    def visible_tile_keys(self, level: int = 0) -> list[TileKey]:
        if self.backend is None:
            return []
        rect = self.mapToScene(self.viewport().rect()).boundingRect()
        full_w, full_h = self._content_size
        level_w, level_h = self.backend.get_level_shape(level)
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
        if self.backend is None:
            return 0
        levels = max(1, int(self.backend.get_level_count()))
        scale = self.current_scale()
        if levels <= 1:
            return 0
        level = 0
        while level + 1 < levels and scale < 0.5:
            level += 1
            scale *= 2.0
        return level

    def refresh_visible_tiles(self) -> None:
        if self.backend is None:
            return
        # Show the preview thumbnail when zoomed out; lazily add tiles once zoom is meaningful.
        if self.current_scale() < 0.2 and self.backend.get_level_count() <= 1:
            return
        level = self._select_level()
        visible_keys = set(self.visible_tile_keys(level=level))
        for key in list(self._tile_items):
            if key not in visible_keys:
                self._scene.removeItem(self._tile_items.pop(key))
        for key in visible_keys:
            if key in self._tile_items:
                continue
            cached = self.tile_cache.get(key)
            if cached is None:
                x = key.tx * self.tile_size
                y = key.ty * self.tile_size
                region = self.backend.read_region(
                    x, y, self.tile_size, self.tile_size, level=key.level
                )
                cached = (
                    region
                    if isinstance(region, QtGui.QImage)
                    else array_to_qimage(region)
                )
                self.tile_cache.put(key, cached)
            item = QtWidgets.QGraphicsPixmapItem(QtGui.QPixmap.fromImage(cached))
            full_w, full_h = self._content_size
            level_w, level_h = self.backend.get_level_shape(key.level)
            scale_x = full_w / max(1, level_w)
            scale_y = full_h / max(1, level_h)
            item.setPos(
                key.tx * self.tile_size * scale_x, key.ty * self.tile_size * scale_y
            )
            item.setTransform(QtGui.QTransform.fromScale(scale_x, scale_y))
            item.setZValue(-10.0 - key.level)
            self._scene.addItem(item)
            self._tile_items[key] = item

    def set_zoom_percent(self, percent: int) -> None:
        if not self._content_size[0] or not self._content_size[1]:
            return
        self._fit_mode = "manual"
        self.resetTransform()
        factor = max(0.01, float(percent) / 100.0)
        self.scale(factor, factor)
        self.refresh_visible_tiles()

    def fit_to_window(self) -> None:
        self._fit_mode = "fit_window"
        self.fitInView(self.sceneRect(), QtCore.Qt.KeepAspectRatio)
        self.refresh_visible_tiles()

    def fit_to_width(self) -> None:
        self._fit_mode = "fit_width"
        rect = self.sceneRect()
        if rect.width() <= 0:
            return
        self.resetTransform()
        viewport_width = max(1, self.viewport().width())
        factor = viewport_width / rect.width()
        self.scale(factor, factor)
        self.refresh_visible_tiles()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._fit_mode == "fit_window":
            self.fit_to_window()
        elif self._fit_mode == "fit_width":
            self.fit_to_width()
        else:
            self.refresh_visible_tiles()

    def scrollContentsBy(self, dx: int, dy: int) -> None:
        super().scrollContentsBy(dx, dy)
        self.refresh_visible_tiles()

    def wheelEvent(self, event):
        super().wheelEvent(event)
        self.refresh_visible_tiles()

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
                if len(self.current.points) >= 2 and self._close_enough(
                    scene_pos, self.current.points[0]
                ):
                    self.finalise()
                else:
                    self.current.addPoint(QtCore.QPointF(scene_pos))
                    self._update_drawing_preview(scene_pos)
            elif mode == "linestrip":
                self.current.addPoint(QtCore.QPointF(scene_pos))
                self._update_drawing_preview(scene_pos)
                if event.modifiers() & QtCore.Qt.ControlModifier:
                    self.finalise()
            elif mode == "line":
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
            self._apply_selection([], emit_signal=True)
        item = self.itemAt(pos)
        if not isinstance(item, _LandmarkPairItem):
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

    def keyPressEvent(self, event):
        if self.drawing():
            if event.key() == QtCore.Qt.Key_Escape:
                self.current = None
                self._preview_item.setPath(QtGui.QPainterPath())
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
