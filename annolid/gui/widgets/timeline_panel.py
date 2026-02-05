from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from qtpy import QtCore, QtGui, QtWidgets


class RefreshingComboBox(QtWidgets.QComboBox):
    popupAboutToShow = QtCore.Signal()

    def showPopup(self) -> None:
        self.popupAboutToShow.emit()
        return super().showPopup()


def _format_timecode(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_min = total_s // 60
    m = total_min % 60
    h = total_min // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


@dataclass(frozen=True)
class TimelineTrack:
    track_id: str
    label: str
    kind: str = "behavior"
    color: Optional[QtGui.QColor] = None
    behaviors: Tuple[str, ...] = ()
    subject: Optional[str] = None


@dataclass(frozen=True)
class TimelineEvent:
    track_id: str
    start_frame: int
    end_frame: Optional[int]
    label: str
    color: Optional[QtGui.QColor] = None
    confidence: Optional[float] = None
    kind: str = "behavior"
    behavior: Optional[str] = None
    subject: Optional[str] = None


@dataclass(frozen=True)
class TimelineEventContext:
    event: TimelineEvent
    original_start: int
    original_end: Optional[int]


class TimelineModel(QtCore.QObject):
    changed = QtCore.Signal()

    def __init__(self, parent: Optional[QtCore.QObject] = None) -> None:
        super().__init__(parent)
        self._tracks: List[TimelineTrack] = []
        self._events: List[TimelineEvent] = []

    def set_tracks(self, tracks: Iterable[TimelineTrack]) -> None:
        self._tracks = list(tracks)
        self.changed.emit()

    def set_events(self, events: Iterable[TimelineEvent]) -> None:
        self._events = list(events)
        self.changed.emit()

    def clear(self) -> None:
        self._tracks = []
        self._events = []
        self.changed.emit()

    @property
    def tracks(self) -> List[TimelineTrack]:
        return list(self._tracks)

    @property
    def events(self) -> List[TimelineEvent]:
        return list(self._events)


class TimelineSegmentItem(QtWidgets.QGraphicsRectItem):
    """A timeline segment with rounded corners and a subtle border."""

    def __init__(self, rect: QtCore.QRectF) -> None:
        super().__init__(rect)
        self._border_color = QtGui.QColor(0, 0, 0, 55)
        self._border_width = 1.0
        self.setPen(QtGui.QPen(QtCore.Qt.NoPen))

    def set_border_color(self, color: QtGui.QColor) -> None:
        self._border_color = QtGui.QColor(color)
        self.update()

    def set_border_width(self, width: float) -> None:
        self._border_width = max(0.0, float(width))
        self.update()

    def shape(self) -> QtGui.QPainterPath:
        rect = self.rect()
        radius = max(1.0, min(6.0, rect.height() * 0.45))
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, radius, radius)
        return path

    def paint(
        self,
        painter: QtGui.QPainter,
        option: QtWidgets.QStyleOptionGraphicsItem,
        widget: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        rect = self.rect().adjusted(0.5, 0.5, -0.5, -0.5)
        radius = max(1.0, min(6.0, rect.height() * 0.45))
        path = QtGui.QPainterPath()
        path.addRoundedRect(rect, radius, radius)

        painter.fillPath(path, self.brush())

        if option.state & QtWidgets.QStyle.State_Selected:
            outline = QtGui.QPen(QtGui.QColor(255, 255, 255, 210), 1.5)
            outline.setCosmetic(True)
            painter.setPen(outline)
            painter.drawPath(path)
            glow = QtGui.QPen(QtGui.QColor(60, 120, 220, 160), 2.5)
            glow.setCosmetic(True)
            painter.setPen(glow)
            painter.drawPath(path)
            return

        if self._border_width > 0:
            border = QtGui.QPen(self._border_color, self._border_width)
            border.setCosmetic(True)
            painter.setPen(border)
            painter.drawPath(path)


class TimelineEventItem(TimelineSegmentItem):
    HANDLE_WIDTH = 5.0

    def __init__(
        self,
        rect: QtCore.QRectF,
        context: TimelineEventContext,
        pixels_per_frame: float,
        min_frame: int,
        max_frame: int,
        edit_callback: Callable[[str, TimelineEventContext, int, int], None],
    ) -> None:
        super().__init__(rect)
        self._context = context
        self._pixels_per_frame = max(1e-6, float(pixels_per_frame))
        self._min_frame = int(min_frame)
        self._max_frame = int(max_frame)
        self._edit_callback = edit_callback
        self._drag_mode: Optional[str] = None
        self._drag_start_pos: Optional[QtCore.QPointF] = None
        self._drag_start_rect: Optional[QtCore.QRectF] = None
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsSelectable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, False)
        self.setAcceptHoverEvents(True)

    def hoverMoveEvent(self, event: QtWidgets.QGraphicsSceneHoverEvent) -> None:
        pos = event.pos().x()
        rect = self.rect()
        if abs(pos - rect.left()) <= self.HANDLE_WIDTH:
            self.setCursor(QtCore.Qt.SizeHorCursor)
        elif abs(pos - rect.right()) <= self.HANDLE_WIDTH:
            self.setCursor(QtCore.Qt.SizeHorCursor)
        else:
            self.setCursor(QtCore.Qt.OpenHandCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if event.button() != QtCore.Qt.LeftButton:
            return super().mousePressEvent(event)
        pos = event.pos().x()
        rect = self.rect()
        if abs(pos - rect.left()) <= self.HANDLE_WIDTH:
            self._drag_mode = "resize_left"
        elif abs(pos - rect.right()) <= self.HANDLE_WIDTH:
            self._drag_mode = "resize_right"
        else:
            self._drag_mode = "move"
        self._drag_start_pos = event.scenePos()
        self._drag_start_rect = QtCore.QRectF(rect)
        self.setCursor(QtCore.Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if (
            self._drag_mode is None
            or self._drag_start_pos is None
            or self._drag_start_rect is None
        ):
            return super().mouseMoveEvent(event)
        delta = event.scenePos().x() - self._drag_start_pos.x()
        rect = QtCore.QRectF(self._drag_start_rect)
        if self._drag_mode == "move":
            rect.moveLeft(rect.left() + delta)
        elif self._drag_mode == "resize_left":
            rect.setLeft(rect.left() + delta)
        elif self._drag_mode == "resize_right":
            rect.setRight(rect.right() + delta)
        rect = self._normalize_rect(rect)
        self.setRect(rect)
        event.accept()

    def mouseReleaseEvent(self, event: QtWidgets.QGraphicsSceneMouseEvent) -> None:
        if self._drag_mode is None:
            return super().mouseReleaseEvent(event)
        rect = self._normalize_rect(self.rect())
        self.setRect(rect)
        start_frame, end_frame = self._frames_for_rect(rect)
        QtCore.QTimer.singleShot(
            0,
            lambda: self._edit_callback(
                "update", self._context, start_frame, end_frame
            ),
        )
        self._drag_mode = None
        self._drag_start_pos = None
        self._drag_start_rect = None
        self.setCursor(QtCore.Qt.OpenHandCursor)
        event.accept()

    def contextMenuEvent(self, event: QtWidgets.QGraphicsSceneContextMenuEvent) -> None:
        menu = QtWidgets.QMenu()
        delete_action = menu.addAction("Delete Segment")
        chosen = menu.exec_(event.screenPos())
        if chosen is delete_action:
            start = self._context.original_start
            end = (
                self._context.original_end
                if self._context.original_end is not None
                else start
            )
            QtCore.QTimer.singleShot(
                0,
                lambda: self._edit_callback(
                    "delete", self._context, int(start), int(end)
                ),
            )
            event.accept()
            return
        super().contextMenuEvent(event)

    def _normalize_rect(self, rect: QtCore.QRectF) -> QtCore.QRectF:
        if rect.width() < self._pixels_per_frame:
            rect.setWidth(self._pixels_per_frame)
        if rect.left() < 0:
            rect.moveLeft(0)
        max_width = (self._max_frame - self._min_frame + 1) * self._pixels_per_frame
        if rect.right() > max_width:
            rect.moveRight(max_width)
        return rect

    def _frames_for_rect(self, rect: QtCore.QRectF) -> Tuple[int, int]:
        start = int(round(rect.left() / self._pixels_per_frame)) + self._min_frame
        end = int(round(rect.right() / self._pixels_per_frame)) + self._min_frame - 1
        start = max(self._min_frame, min(self._max_frame, start))
        end = max(start, min(self._max_frame, end))
        return start, end


class TimelineGraphicsView(QtWidgets.QGraphicsView):
    frameSelected = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QtGui.QPainter.Antialiasing, True)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.setMouseTracking(True)

        self._model: Optional[TimelineModel] = None
        self._min_frame = 0
        self._max_frame = 0
        self._zoom_factor = 1.0
        self._row_height = 18
        self._row_gap = 6
        self._header_height = 20
        self._current_frame = 0
        self._base_pixels_per_frame = 1.0
        self._edit_mode = False
        self._edit_callback: Optional[
            Callable[[str, TimelineEventContext, int, int], None]
        ] = None
        self._track_order: List[TimelineTrack] = []
        self._new_item: Optional[QtWidgets.QGraphicsRectItem] = None
        self._new_item_row: Optional[int] = None
        self._new_item_start_frame: Optional[int] = None
        self._frame_to_time: Optional[Callable[[int], Optional[float]]] = None
        self._deferred_rebuild = False
        self._playhead_line: Optional[QtWidgets.QGraphicsLineItem] = None
        self._playhead_triangle: Optional[QtWidgets.QGraphicsPolygonItem] = None
        self._last_pixels_per_frame: float = 1.0
        self._last_scene_height: float = 0.0

    def set_edit_mode(self, enabled: bool) -> None:
        self._edit_mode = bool(enabled)

    def set_edit_callback(
        self,
        callback: Optional[Callable[[str, TimelineEventContext, int, int], None]],
    ) -> None:
        self._edit_callback = callback

    def set_frame_to_time(
        self, provider: Optional[Callable[[int], Optional[float]]]
    ) -> None:
        self._frame_to_time = provider
        self.rebuild_scene()

    def set_model(self, model: Optional[TimelineModel]) -> None:
        if self._model is model:
            return
        if self._model is not None:
            try:
                self._model.changed.disconnect(self.rebuild_scene)
            except Exception:
                pass
        self._model = model
        if self._model is not None:
            self._model.changed.connect(self.rebuild_scene)
        self.rebuild_scene()

    def set_time_range(self, min_frame: int, max_frame: int) -> None:
        self._min_frame = int(min_frame)
        self._max_frame = max(int(max_frame), self._min_frame)
        self.rebuild_scene()

    def set_zoom_factor(self, zoom_factor: float) -> None:
        self._zoom_factor = max(0.1, float(zoom_factor))
        self.rebuild_scene()

    def set_current_frame(self, frame: int) -> None:
        self._current_frame = int(frame)
        self._update_playhead()

    def set_row_metrics(self, height: int, gap: int) -> None:
        self._row_height = max(8, int(height))
        self._row_gap = max(0, int(gap))
        self.rebuild_scene()

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        self.rebuild_scene()

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        if event.modifiers() & QtCore.Qt.ControlModifier:
            delta = 0.1 if event.angleDelta().y() > 0 else -0.1
            self.set_zoom_factor(self._zoom_factor + delta)
            event.accept()
            return
        super().wheelEvent(event)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            event.button() == QtCore.Qt.LeftButton
            and self._edit_mode
            and self._edit_callback
        ):
            item = self._event_item_at(event.pos())
            if item is not None:
                return super().mousePressEvent(event)
            row_idx = self._row_index_for_pos(self.mapToScene(event.pos()).y())
            if row_idx is not None:
                frame = self._x_to_frame(self.mapToScene(event.pos()).x())
                self._new_item_row = row_idx
                self._new_item_start_frame = frame
                self._new_item = self._scene.addRect(
                    QtCore.QRectF(
                        self._frame_to_x(
                            frame, self._base_pixels_per_frame * self._zoom_factor
                        ),
                        self._row_y(row_idx),
                        1,
                        self._row_height,
                    ),
                    QtGui.QPen(QtCore.Qt.NoPen),
                    QtGui.QBrush(QtGui.QColor(120, 180, 220, 120)),
                )
                event.accept()
                return
        if event.button() == QtCore.Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())
            frame = self._x_to_frame(scene_pos.x())
            self.frameSelected.emit(frame)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> None:
        if self._new_item is not None and self._new_item_start_frame is not None:
            # The scene may rebuild due to external updates; if the draft item was deleted,
            # abort the interaction instead of crashing.
            try:
                _ = self._new_item.rect()
            except RuntimeError:
                self._new_item = None
                self._new_item_row = None
                self._new_item_start_frame = None
                super().mouseMoveEvent(event)
                return
            scene_pos = self.mapToScene(event.pos())
            end_frame = self._x_to_frame(scene_pos.x())
            start = min(self._new_item_start_frame, end_frame)
            end = max(self._new_item_start_frame, end_frame)
            x0 = self._frame_to_x(
                start, self._base_pixels_per_frame * self._zoom_factor
            )
            x1 = self._frame_to_x(
                end + 1, self._base_pixels_per_frame * self._zoom_factor
            )
            rect = self._new_item.rect()
            rect.setLeft(x0)
            rect.setWidth(max(1.0, x1 - x0))
            self._new_item.setRect(rect)
            event.accept()
            return
        if self._frame_to_time is not None:
            scene_pos = self.mapToScene(event.pos())
            if scene_pos.y() >= 0:
                frame = self._x_to_frame(scene_pos.x())
                time_s = self._frame_to_time(frame)
                if time_s is not None:
                    QtWidgets.QToolTip.showText(
                        event.globalPos(), f"{time_s:.3f}s (frame {frame})"
                    )
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> None:
        if (
            self._new_item is not None
            and self._new_item_start_frame is not None
            and self._new_item_row is not None
        ):
            scene_pos = self.mapToScene(event.pos())
            end_frame = self._x_to_frame(scene_pos.x())
            start = min(self._new_item_start_frame, end_frame)
            end = max(self._new_item_start_frame, end_frame)
            # Keep the in-progress rectangle visible until the callback refreshes the model.
            # This avoids a "nothing happened" flicker if the refresh is async.
            pending_item = self._new_item
            pending_item.setBrush(QtGui.QBrush(QtGui.QColor(120, 180, 220, 160)))
            pending_item.setPen(QtGui.QPen(QtCore.Qt.NoPen))
            self._new_item = None
            row_idx = self._new_item_row
            self._new_item_row = None
            self._new_item_start_frame = None
            track = (
                self._track_order[row_idx]
                if 0 <= row_idx < len(self._track_order)
                else None
            )
            if track is not None and self._edit_callback:
                temp_event = TimelineEvent(
                    track_id=track.track_id,
                    start_frame=start,
                    end_frame=end,
                    label=track.label,
                    behavior=track.behaviors[0] if track.behaviors else None,
                    subject=track.subject,
                    kind=track.kind,
                )
                context = TimelineEventContext(
                    event=temp_event,
                    original_start=start,
                    original_end=end,
                )
                QtCore.QTimer.singleShot(
                    0,
                    lambda ctx=context, s=start, e=end: self._edit_callback(
                        "create", ctx, s, e
                    ),
                )
            else:
                # No valid track or callback; remove the pending item.
                try:
                    self._scene.removeItem(pending_item)
                except Exception:
                    pass
            if self._deferred_rebuild:
                # Apply any skipped rebuild after the interaction finishes.
                self._deferred_rebuild = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if (
            self._edit_mode
            and self._edit_callback is not None
            and event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace)
        ):
            selected = [
                item
                for item in self.scene().selectedItems()
                if isinstance(item, TimelineEventItem)
            ]
            if selected:
                for item in selected:
                    ctx = item._context
                    start = ctx.original_start
                    end = ctx.original_end if ctx.original_end is not None else start
                    self._edit_callback("delete", ctx, int(start), int(end))
                event.accept()
                return
        super().keyPressEvent(event)

    def rebuild_scene(self) -> None:
        # Don't rebuild while the user is drawing a new segment; clearing the scene
        # deletes the draft item and can crash on the next mouse move.
        if self._new_item is not None:
            self._deferred_rebuild = True
            return

        self._playhead_line = None
        self._playhead_triangle = None
        self._scene.clear()
        frame_span = max(1, self._max_frame - self._min_frame + 1)
        view_width = max(1, self.viewport().width())
        self._base_pixels_per_frame = max(1.0, view_width / frame_span)
        pixels_per_frame = self._base_pixels_per_frame * self._zoom_factor
        self._last_pixels_per_frame = pixels_per_frame

        tracks = self._model.tracks if self._model is not None else []
        events = self._model.events if self._model is not None else []
        self._track_order = list(tracks)
        row_pitch = self._row_height + self._row_gap
        total_rows = len(tracks)
        scene_height = self._header_height + total_rows * row_pitch + self._row_gap
        scene_width = frame_span * pixels_per_frame
        self._last_scene_height = scene_height
        self._scene.setSceneRect(0, 0, scene_width, scene_height)

        header_rect = QtCore.QRectF(0, 0, scene_width, self._header_height)
        header_gradient = QtGui.QLinearGradient(
            header_rect.topLeft(), header_rect.bottomLeft()
        )
        header_gradient.setColorAt(0.0, QtGui.QColor(250, 251, 253))
        header_gradient.setColorAt(1.0, QtGui.QColor(238, 241, 245))
        self._scene.addRect(
            header_rect,
            QtGui.QPen(QtCore.Qt.NoPen),
            QtGui.QBrush(header_gradient),
        )

        tick_pen = QtGui.QPen(QtGui.QColor(170, 176, 186))
        tick_pen.setCosmetic(True)
        tick_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        tick_font.setPointSize(max(8, self.font().pointSize() - 1))
        major_ticks = self._tick_frames()
        for tick_frame in major_ticks:
            x = self._frame_to_x(tick_frame, pixels_per_frame)
            self._scene.addLine(x, 0, x, scene_height, tick_pen)
            label_text = str(tick_frame)
            if self._frame_to_time is not None:
                value = self._frame_to_time(tick_frame)
                if value is not None:
                    label_text = _format_timecode(value)
            label = self._scene.addText(label_text)
            label.setPos(x + 2, 1)
            label.setFont(tick_font)
            label.setDefaultTextColor(QtGui.QColor(70, 75, 82))

        # Minor ticks: 4 subdivisions between major ticks.
        minor_pen = QtGui.QPen(QtGui.QColor(215, 220, 227))
        minor_pen.setCosmetic(True)
        if len(major_ticks) >= 2:
            for a, b in zip(major_ticks[:-1], major_ticks[1:]):
                step = max(1, int(round((b - a) / 4)))
                for f in range(a + step, b, step):
                    x = self._frame_to_x(f, pixels_per_frame)
                    self._scene.addLine(x, 0, x, self._header_height, minor_pen)

        for idx, track in enumerate(tracks):
            y = self._header_height + idx * row_pitch
            row_rect = QtCore.QRectF(0, y, scene_width, self._row_height)
            base_color = (
                QtGui.QColor(255, 255, 255)
                if idx % 2 == 0
                else QtGui.QColor(248, 249, 251)
            )
            self._scene.addRect(
                row_rect, QtGui.QPen(QtCore.Qt.NoPen), QtGui.QBrush(base_color)
            )

        track_index = {track.track_id: idx for idx, track in enumerate(tracks)}
        for event in events:
            row_idx = track_index.get(event.track_id)
            if row_idx is None:
                continue
            y = self._header_height + row_idx * row_pitch
            end_frame = (
                event.end_frame if event.end_frame is not None else self._max_frame
            )
            start_frame = max(self._min_frame, event.start_frame)
            end_frame = max(start_frame, min(self._max_frame, end_frame))
            x0 = self._frame_to_x(start_frame, pixels_per_frame)
            x1 = self._frame_to_x(end_frame + 1, pixels_per_frame)
            width = max(1.0, x1 - x0)
            rect = QtCore.QRectF(x0, y, width, self._row_height)
            color = event.color or QtGui.QColor(100, 140, 200)
            brush = QtGui.QBrush(self._apply_confidence(color, event.confidence))
            context = TimelineEventContext(
                event=event,
                original_start=event.start_frame,
                original_end=event.end_frame,
            )
            is_editable_event = (
                self._edit_mode
                and self._edit_callback
                and event.end_frame is not None
                and event.kind == "behavior"
                and bool(event.behavior)
            )
            segment_item: TimelineSegmentItem
            if is_editable_event:
                segment_item = TimelineEventItem(
                    rect,
                    context,
                    pixels_per_frame,
                    self._min_frame,
                    self._max_frame,
                    self._edit_callback,
                )
                segment_item.setBrush(brush)
                self._scene.addItem(segment_item)
            else:
                segment_item = TimelineSegmentItem(rect)
                segment_item.setBrush(brush)
                self._scene.addItem(segment_item)

            # Subtle border that follows the fill color.
            border_color = QtGui.QColor(0, 0, 0, 70)
            if color.isValid():
                border_color = QtGui.QColor(color)
                border_color.setAlpha(90)
            segment_item.set_border_color(border_color)

            label_padding = 6
            if width > 34:
                label_font = QtGui.QFont(self.font())
                label_font.setPointSize(max(8, self.font().pointSize() - 1))
                metrics = QtGui.QFontMetrics(label_font)
                elided = metrics.elidedText(
                    event.label,
                    QtCore.Qt.ElideRight,
                    int(max(0.0, width - 2 * label_padding)),
                )
                if elided:
                    text_item = QtWidgets.QGraphicsSimpleTextItem(elided, segment_item)
                    text_item.setFont(label_font)
                    text_item.setPos(label_padding, 1)
                    text_item.setBrush(QtGui.QBrush(_ideal_text_color(brush.color())))

        self._create_or_update_playhead(playhead_frame=self._current_frame)

    def _frame_to_x(self, frame: int, pixels_per_frame: float) -> float:
        return max(0.0, (frame - self._min_frame) * pixels_per_frame)

    def _x_to_frame(self, x: float) -> int:
        pixels_per_frame = self._base_pixels_per_frame * self._zoom_factor
        if pixels_per_frame <= 0:
            return self._min_frame
        frame = int(round(x / pixels_per_frame)) + self._min_frame
        return max(self._min_frame, min(self._max_frame, frame))

    def _tick_frames(self) -> List[int]:
        span = max(1, self._max_frame - self._min_frame + 1)
        step = _nice_tick_step(span)
        start = (self._min_frame // step) * step
        return list(range(start, self._max_frame + 1, step))

    def _row_index_for_pos(self, y_pos: float) -> Optional[int]:
        if y_pos < self._header_height:
            return None
        row_pitch = self._row_height + self._row_gap
        idx = int((y_pos - self._header_height) // row_pitch)
        if idx < 0 or idx >= len(self._track_order):
            return None
        return idx

    def _row_y(self, row_idx: int) -> float:
        return self._header_height + row_idx * (self._row_height + self._row_gap)

    def _event_item_at(self, view_pos: QtCore.QPoint) -> Optional[TimelineEventItem]:
        item = self.itemAt(view_pos)
        while item is not None:
            if isinstance(item, TimelineEventItem):
                return item
            item = item.parentItem()
        return None

    def _update_playhead(self) -> None:
        if self._new_item is not None:
            # Avoid interfering with interactive drawing.
            return
        if self._playhead_line is None or self._playhead_triangle is None:
            # Scene not built yet (or was cleared).
            self.rebuild_scene()
            return
        self._create_or_update_playhead(playhead_frame=self._current_frame)

    def _create_or_update_playhead(self, *, playhead_frame: int) -> None:
        playhead_x = self._frame_to_x(int(playhead_frame), self._last_pixels_per_frame)
        playhead_pen = QtGui.QPen(QtGui.QColor(220, 60, 60))
        playhead_pen.setCosmetic(True)

        if self._playhead_line is None:
            self._playhead_line = self._scene.addLine(
                playhead_x,
                0,
                playhead_x,
                self._last_scene_height,
                playhead_pen,
            )
        else:
            self._playhead_line.setLine(
                playhead_x, 0, playhead_x, self._last_scene_height
            )

        # Triangle playhead marker (VIA-like).
        tri_size = 7.0
        tri = QtGui.QPolygonF(
            [
                QtCore.QPointF(playhead_x, self._header_height - 1),
                QtCore.QPointF(playhead_x - tri_size, 1),
                QtCore.QPointF(playhead_x + tri_size, 1),
            ]
        )
        if self._playhead_triangle is None:
            self._playhead_triangle = self._scene.addPolygon(
                tri,
                QtGui.QPen(QtCore.Qt.NoPen),
                QtGui.QBrush(QtGui.QColor(220, 60, 60)),
            )
        else:
            self._playhead_triangle.setPolygon(tri)

    @staticmethod
    def _apply_confidence(
        color: QtGui.QColor, confidence: Optional[float]
    ) -> QtGui.QColor:
        if confidence is None:
            return color
        value = max(0.0, min(1.0, float(confidence)))
        alpha = int(80 + 175 * value)
        adjusted = QtGui.QColor(color)
        adjusted.setAlpha(alpha)
        return adjusted


class TimelinePanel(QtWidgets.QWidget):
    frameSelected = QtCore.Signal(int)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._model = TimelineModel(self)
        self._color_getter: Optional[Callable[[str], QtGui.QColor]] = None
        self._behavior_controller = None
        self._row_mode = "Behavior"
        self._custom_rows: List[Dict[str, Any]] = []
        self._timestamp_provider: Optional[Callable[[int], Optional[float]]] = None
        self._settings = QtCore.QSettings()
        self._active_behavior: Optional[str] = None
        self._defined_behaviors: List[str] = []
        self._catalog_provider: Optional[Callable[[], List[str]]] = None
        self._catalog_adder: Optional[Callable[[str], None]] = None
        self._hidden_behaviors: set[str] = set()
        self._catalog_behaviors: List[str] = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(8)

        left_widget = QtWidgets.QFrame()
        left_widget.setObjectName("TimelineSidebar")
        left_widget.setFrameShape(QtWidgets.QFrame.NoFrame)
        left_layout = QtWidgets.QVBoxLayout(left_widget)
        left_layout.setContentsMargins(10, 10, 10, 10)
        left_layout.setSpacing(10)

        header_row = QtWidgets.QHBoxLayout()
        header_row.setSpacing(8)
        self._row_mode_combo = QtWidgets.QComboBox()
        self._row_mode_combo.addItems(["Behavior", "Custom"])
        self._row_mode_combo.currentTextChanged.connect(self._on_row_mode_changed)
        header_row.addWidget(self._row_mode_combo, 1)

        self._edit_rows_button = QtWidgets.QToolButton()
        try:
            self._edit_rows_button.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
            )
        except Exception:
            pass
        self._edit_rows_button.setToolTip("Configure rows")
        self._edit_rows_button.clicked.connect(self._open_row_config_dialog)
        header_row.addWidget(self._edit_rows_button)
        left_layout.addLayout(header_row)

        self._track_list = QtWidgets.QListWidget()
        self._track_list.setObjectName("TimelineTrackList")
        self._track_list.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._track_list.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self._track_list.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._track_list.currentRowChanged.connect(self._on_track_selected)
        left_layout.addWidget(self._track_list, 1)

        manage_row = QtWidgets.QHBoxLayout()
        manage_row.setSpacing(8)
        self._behavior_combo = RefreshingComboBox()
        self._behavior_combo.setEditable(True)
        self._behavior_combo.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self._behavior_combo.setMinimumWidth(150)
        self._behavior_combo.currentTextChanged.connect(self._on_behavior_selected)
        self._behavior_combo.popupAboutToShow.connect(self.refresh_behavior_catalog)
        manage_row.addWidget(self._behavior_combo, 1)

        self._add_behavior_button = QtWidgets.QToolButton()
        self._add_behavior_button.setIcon(
            self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        )
        self._add_behavior_button.setToolTip("Add behavior to catalog")
        self._add_behavior_button.clicked.connect(self._define_behavior)
        manage_row.addWidget(self._add_behavior_button)

        self._hide_behavior_button = QtWidgets.QToolButton()
        trash_icon = None
        try:
            trash_icon = self.style().standardIcon(QtWidgets.QStyle.SP_TrashIcon)
        except Exception:
            trash_icon = None
        self._hide_behavior_button.setIcon(
            trash_icon
            if trash_icon is not None and not trash_icon.isNull()
            else self.style().standardIcon(QtWidgets.QStyle.SP_DialogCancelButton)
        )
        self._hide_behavior_button.setToolTip("Hide selected behavior")
        self._hide_behavior_button.clicked.connect(self._hide_selected_behavior)
        manage_row.addWidget(self._hide_behavior_button)

        self._reset_hidden_button = QtWidgets.QToolButton()
        try:
            self._reset_hidden_button.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.SP_BrowserReload)
            )
        except Exception:
            self._reset_hidden_button.setIcon(
                self.style().standardIcon(QtWidgets.QStyle.SP_DialogResetButton)
            )
        self._reset_hidden_button.setToolTip("Restore hidden behaviors")
        self._reset_hidden_button.clicked.connect(self._reset_hidden_behaviors)
        manage_row.addWidget(self._reset_hidden_button)

        left_layout.addLayout(manage_row)

        tools_row = QtWidgets.QHBoxLayout()
        tools_row.setSpacing(10)
        self._edit_toggle = QtWidgets.QToolButton()
        self._edit_toggle.setText("Edit")
        self._edit_toggle.setCheckable(True)
        self._edit_toggle.toggled.connect(self._on_edit_toggled)
        tools_row.addWidget(self._edit_toggle)

        zoom_label = QtWidgets.QLabel("Zoom")
        zoom_label.setObjectName("TimelineZoomLabel")
        tools_row.addWidget(zoom_label)
        self._zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._zoom_slider.setRange(1, 20)
        self._zoom_slider.setValue(3)
        self._zoom_slider.valueChanged.connect(self._on_zoom_changed)
        tools_row.addWidget(self._zoom_slider, 1)
        left_layout.addLayout(tools_row)

        splitter.addWidget(left_widget)

        self._view = TimelineGraphicsView()
        self._view.setObjectName("TimelineView")
        self._view.set_model(self._model)
        self._view.frameSelected.connect(self.frameSelected.emit)
        self._view.set_edit_callback(self._handle_event_edit)
        self._view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._view.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(246, 247, 249)))
        splitter.addWidget(self._view)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter, 1)
        self.setMinimumSize(0, 0)

        self.setObjectName("TimelinePanel")
        self.setStyleSheet(_timeline_panel_stylesheet())

        self._view.verticalScrollBar().valueChanged.connect(
            self._track_list.verticalScrollBar().setValue
        )
        self._track_list.verticalScrollBar().valueChanged.connect(
            self._view.verticalScrollBar().setValue
        )

        self._load_settings()
        self._apply_row_mode_ui()
        self._edit_toggle.setChecked(True)
        self.refresh_behavior_catalog()
        self._install_behavior_shortcuts()

    def set_behavior_catalog(
        self,
        *,
        provider: Optional[Callable[[], List[str]]] = None,
        adder: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._catalog_provider = provider
        self._catalog_adder = adder
        self.refresh_behavior_catalog()

    def set_time_range(self, min_frame: int, max_frame: int) -> None:
        self._view.set_time_range(min_frame, max_frame)

    def set_current_frame(self, frame: int) -> None:
        self._view.set_current_frame(frame)

    def set_behavior_controller(
        self,
        controller,
        *,
        color_getter: Optional[Callable[[str], QtGui.QColor]] = None,
    ) -> None:
        if self._behavior_controller is controller:
            return
        if self._behavior_controller is not None:
            try:
                self._behavior_controller.remove_change_listener(
                    self.refresh_from_behavior_controller
                )
            except Exception:
                pass
        self._behavior_controller = controller
        self._color_getter = color_getter
        if controller is not None:
            controller.add_change_listener(self.refresh_from_behavior_controller)
        self.refresh_from_behavior_controller()

    def set_timestamp_provider(
        self, provider: Optional[Callable[[int], Optional[float]]]
    ) -> None:
        self._timestamp_provider = provider
        self._view.set_frame_to_time(provider)

    def refresh_behavior_catalog(self) -> None:
        self._sync_behavior_choices()
        self._refresh_timeline_model()

    def refresh_from_behavior_controller(self) -> None:
        if self._behavior_controller is None:
            self._model.clear()
            self._track_list.clear()
            self._behavior_combo.clear()
            return

        self.refresh_behavior_catalog()

    def _refresh_timeline_model(self) -> None:
        if self._row_mode == "Behavior":
            self._refresh_behavior_rows()
            return
        if self._behavior_controller is None:
            self._model.clear()
            return
        event_list = list(self._behavior_controller.iter_events())
        tracks, events = self._build_tracks_and_events(event_list)
        self._model.set_tracks(tracks)
        self._model.set_events(events)
        self._refresh_track_list(tracks)

    def _refresh_behavior_rows(self) -> None:
        tracks: List[TimelineTrack] = []
        for behavior in self._catalog_behaviors:
            color = _color_from_getter(self._color_getter, behavior)
            tracks.append(
                TimelineTrack(
                    track_id=behavior,
                    label=behavior,
                    color=color,
                    behaviors=(behavior,),
                    kind="behavior",
                )
            )

        events: List[TimelineEvent] = []
        if self._behavior_controller is not None:
            try:
                ranges = list(self._behavior_controller.timeline.iter_ranges())
            except Exception:
                ranges = []
            for behavior, start, end in ranges:
                if behavior in self._hidden_behaviors:
                    continue
                if self._catalog_behaviors and behavior not in self._catalog_behaviors:
                    continue
                color = _color_from_getter(self._color_getter, behavior)
                events.append(
                    TimelineEvent(
                        track_id=behavior,
                        start_frame=int(start),
                        end_frame=int(end) if end is not None else None,
                        label=behavior,
                        color=color,
                        behavior=behavior,
                        subject=None,
                        kind="behavior",
                    )
                )

        self._model.set_tracks(tracks)
        self._model.set_events(events)
        self._refresh_track_list(tracks)

    def clear(self) -> None:
        self._model.clear()
        self._track_list.clear()
        self._behavior_combo.clear()

    def _refresh_track_list(self, tracks: List[TimelineTrack]) -> None:
        self._track_list.blockSignals(True)
        self._track_list.clear()

        row_height = self._view._row_height + self._view._row_gap
        if not tracks:
            placeholder = QtWidgets.QListWidgetItem("No rows.")
            placeholder.setFlags(QtCore.Qt.NoItemFlags)
            placeholder.setSizeHint(QtCore.QSize(self._track_list.width(), row_height))
            self._track_list.addItem(placeholder)
            self._track_list.blockSignals(False)
            return

        for idx, track in enumerate(tracks, start=1):
            display = (
                f"{idx}. {track.label}" if track.kind == "behavior" else track.label
            )
            item = QtWidgets.QListWidgetItem(display)
            item.setData(QtCore.Qt.UserRole, track.track_id)
            item.setSizeHint(QtCore.QSize(self._track_list.width(), row_height))

            swatch_color = track.color
            if swatch_color is None and track.behaviors:
                swatch_color = _color_from_getter(
                    self._color_getter, track.behaviors[0]
                )
            if swatch_color is not None:
                pix = QtGui.QPixmap(12, 12)
                pix.fill(QtGui.QColor(swatch_color))
                item.setIcon(QtGui.QIcon(pix))

            self._track_list.addItem(item)

        self._track_list.blockSignals(False)

        # Keep the selection aligned with the active behavior when possible.
        if self._active_behavior and self._row_mode == "Behavior":
            self._select_behavior_in_track_list(self._active_behavior)

    def _on_zoom_changed(self, value: int) -> None:
        self._view.set_zoom_factor(max(0.1, float(value)))

    def _on_behavior_selected(self, text: str) -> None:
        self._active_behavior = text if text else None

    def _on_edit_toggled(self, enabled: bool) -> None:
        if self._row_mode == "Event":
            self._edit_toggle.setChecked(False)
            self._view.set_edit_mode(False)
            return
        self._view.set_edit_mode(enabled)

    def _on_row_mode_changed(self, text: str) -> None:
        self._row_mode = text
        self._apply_row_mode_ui()
        self.refresh_from_behavior_controller()
        self._save_settings()

    def _apply_row_mode_ui(self) -> None:
        is_custom = self._row_mode == "Custom"
        self._edit_rows_button.setEnabled(is_custom)
        self._edit_toggle.setEnabled(True)
        self._behavior_combo.setEnabled(True)
        self._add_behavior_button.setEnabled(True)
        self._hide_behavior_button.setEnabled(True)

    def _load_settings(self) -> None:
        mode = self._settings.value("timeline/row_mode", "Behavior", type=str)
        if mode:
            self._row_mode = mode
        rows_json = self._settings.value("timeline/custom_rows", "[]", type=str)
        try:
            rows = json.loads(rows_json) if rows_json else []
        except json.JSONDecodeError:
            rows = []
        self._custom_rows = rows if isinstance(rows, list) else []
        defined_json = self._settings.value(
            "timeline/defined_behaviors", "[]", type=str
        )
        try:
            defined = json.loads(defined_json) if defined_json else []
        except json.JSONDecodeError:
            defined = []
        self._defined_behaviors = (
            [str(x) for x in defined] if isinstance(defined, list) else []
        )
        hidden_json = self._settings.value("timeline/hidden_behaviors", "[]", type=str)
        try:
            hidden = json.loads(hidden_json) if hidden_json else []
        except json.JSONDecodeError:
            hidden = []
        self._hidden_behaviors = (
            set(str(x) for x in hidden) if isinstance(hidden, list) else set()
        )
        # Ensure settings and the visible combo stay in sync. Older installs may have
        # persisted modes that are no longer exposed; default back to Behavior.
        idx = self._row_mode_combo.findText(self._row_mode)
        blocker = QtCore.QSignalBlocker(self._row_mode_combo)
        try:
            if idx >= 0:
                self._row_mode_combo.setCurrentIndex(idx)
            else:
                self._row_mode = "Behavior"
                self._row_mode_combo.setCurrentText("Behavior")
                self._settings.setValue("timeline/row_mode", self._row_mode)
        finally:
            del blocker

    def _save_settings(self) -> None:
        self._settings.setValue("timeline/row_mode", self._row_mode)
        try:
            self._settings.setValue(
                "timeline/custom_rows", json.dumps(self._custom_rows)
            )
        except (TypeError, ValueError):
            self._settings.setValue("timeline/custom_rows", "[]")
        try:
            self._settings.setValue(
                "timeline/defined_behaviors", json.dumps(self._defined_behaviors)
            )
        except (TypeError, ValueError):
            self._settings.setValue("timeline/defined_behaviors", "[]")
        try:
            self._settings.setValue(
                "timeline/hidden_behaviors", json.dumps(sorted(self._hidden_behaviors))
            )
        except (TypeError, ValueError):
            self._settings.setValue("timeline/hidden_behaviors", "[]")

    def _open_row_config_dialog(self) -> None:
        dialog = TimelineRowConfigDialog(self._custom_rows, parent=self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        self._custom_rows = dialog.rows()
        self._save_settings()
        self.refresh_from_behavior_controller()

    def _build_tracks_and_events(
        self, events: Sequence
    ) -> Tuple[List[TimelineTrack], List[TimelineEvent]]:
        behaviors = (
            list(self._catalog_behaviors)
            if self._catalog_behaviors
            else sorted({event.behavior for event in events if event.behavior})
        )
        subjects = sorted({event.subject or "Subject 1" for event in events})

        if self._row_mode == "Event":
            tracks = [
                TimelineTrack(track_id="start", label="Start", kind="event"),
                TimelineTrack(track_id="end", label="End", kind="event"),
            ]
            timeline_events = []
            for event in events:
                track_id = "start" if event.event == "start" else "end"
                color = _color_from_getter(self._color_getter, event.behavior)
                timeline_events.append(
                    TimelineEvent(
                        track_id=track_id,
                        start_frame=event.frame,
                        end_frame=event.frame,
                        label=event.behavior,
                        color=color,
                        kind="event",
                        behavior=event.behavior,
                        subject=event.subject,
                    )
                )
            return tracks, timeline_events

        if self._row_mode == "Instance":
            tracks = [
                TimelineTrack(
                    track_id=subject,
                    label=subject,
                    kind="subject",
                    subject=subject,
                    behaviors=tuple(behaviors),
                )
                for subject in subjects
            ]
            ranges = _build_ranges(events, group_by=("behavior", "subject"))
            timeline_events = []
            for behavior, subject, start, end in ranges:
                color = _color_from_getter(self._color_getter, behavior)
                track_id = subject or "Subject 1"
                timeline_events.append(
                    TimelineEvent(
                        track_id=track_id,
                        start_frame=start,
                        end_frame=end,
                        label=behavior,
                        color=color,
                        behavior=behavior,
                        subject=subject,
                    )
                )
            return tracks, timeline_events

        if self._row_mode == "Behavior+Instance":
            tracks = []
            for behavior in behaviors:
                for subject in subjects:
                    label = f"{behavior} ({subject})"
                    tracks.append(
                        TimelineTrack(
                            track_id=f"{behavior}::{subject}",
                            label=label,
                            kind="behavior_subject",
                            subject=subject,
                            behaviors=(behavior,),
                        )
                    )
            ranges = _build_ranges(events, group_by=("behavior", "subject"))
            timeline_events = []
            for behavior, subject, start, end in ranges:
                color = _color_from_getter(self._color_getter, behavior)
                track_id = f"{behavior}::{subject or 'Subject 1'}"
                timeline_events.append(
                    TimelineEvent(
                        track_id=track_id,
                        start_frame=start,
                        end_frame=end,
                        label=behavior,
                        color=color,
                        behavior=behavior,
                        subject=subject,
                    )
                )
            return tracks, timeline_events

        if self._row_mode == "Custom":
            tracks: List[TimelineTrack] = []
            for idx, row in enumerate(self._custom_rows):
                label = row.get("label") or f"Row {idx + 1}"
                behaviors_for_row = tuple(row.get("behaviors") or [])
                subject = row.get("subject")
                row_color = (
                    _color_from_getter(self._color_getter, behaviors_for_row[0])
                    if behaviors_for_row
                    else None
                )
                tracks.append(
                    TimelineTrack(
                        track_id=row.get("id") or f"custom_{idx}",
                        label=label,
                        kind="custom",
                        color=row_color,
                        behaviors=behaviors_for_row,
                        subject=subject,
                    )
                )

            ranges = _build_ranges(events, group_by=("behavior",))
            timeline_events = []
            for behavior, _subject, start, end in ranges:
                if behavior in self._hidden_behaviors:
                    continue
                for track in tracks:
                    if track.behaviors and behavior not in track.behaviors:
                        continue
                    color = _color_from_getter(self._color_getter, behavior)
                    timeline_events.append(
                        TimelineEvent(
                            track_id=track.track_id,
                            start_frame=start,
                            end_frame=end,
                            label=behavior,
                            color=color,
                            behavior=behavior,
                            subject=None,
                            kind="custom",
                        )
                    )
            return tracks, timeline_events

        tracks = [
            TimelineTrack(
                track_id=behavior,
                label=behavior,
                color=_color_from_getter(self._color_getter, behavior),
                behaviors=(behavior,),
            )
            for behavior in behaviors
        ]
        ranges = _build_ranges(events, group_by=("behavior",))
        timeline_events = []
        for behavior, _subject, start, end in ranges:
            color = _color_from_getter(self._color_getter, behavior)
            label = behavior
            timeline_events.append(
                TimelineEvent(
                    track_id=behavior,
                    start_frame=start,
                    end_frame=end,
                    label=label,
                    color=color,
                    behavior=behavior,
                    subject=None,
                )
            )
        return tracks, timeline_events

    def _handle_event_edit(
        self, action: str, context: TimelineEventContext, start: int, end: int
    ) -> None:
        if self._behavior_controller is None:
            return
        event = context.event
        behavior = event.behavior
        if action == "delete":
            if behavior is None:
                return
            self._behavior_controller.delete_interval(
                behavior=behavior,
                start_frame=context.original_start,
                end_frame=context.original_end
                if context.original_end is not None
                else context.original_start,
            )
            return
        if action == "create":
            behavior = self._resolve_behavior_for_track(event.track_id, behavior)
            if behavior is None:
                return
            self._behavior_controller.create_interval(
                behavior=behavior,
                start_frame=start,
                end_frame=end,
                subject=None,
                timestamp_provider=self._timestamp_provider,
            )
            return
        if behavior is None:
            return
        self._behavior_controller.update_interval(
            behavior=behavior,
            old_start=context.original_start,
            old_end=context.original_end,
            new_start=start,
            new_end=end,
            subject=None,
            timestamp_provider=self._timestamp_provider,
        )

    def _resolve_behavior_for_track(
        self, track_id: str, fallback: Optional[str]
    ) -> Optional[str]:
        tracks = self._model.tracks
        track = next((t for t in tracks if t.track_id == track_id), None)
        if self._active_behavior:
            if (
                track is None
                or not track.behaviors
                or self._active_behavior in track.behaviors
            ):
                return self._active_behavior
        if track is None:
            return fallback
        if track.behaviors:
            if len(track.behaviors) == 1:
                return track.behaviors[0]
            choice, ok = QtWidgets.QInputDialog.getItem(
                self,
                "Select Behavior",
                "Behavior:",
                list(track.behaviors),
                editable=False,
            )
            return choice if ok else None
        if fallback:
            return fallback
        if self._behavior_controller is None:
            return None
        choices = sorted(self._behavior_controller.behavior_names)
        if not choices:
            return None
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Select Behavior",
            "Behavior:",
            choices,
            editable=False,
        )
        return choice if ok else None

    def _sync_behavior_choices(self) -> None:
        behaviors: List[str] = []
        if self._catalog_provider is not None:
            try:
                behaviors.extend(self._catalog_provider() or [])
            except Exception:
                pass
        if self._behavior_controller is not None:
            behaviors.extend(list(self._behavior_controller.behavior_names))
        behaviors.extend(self._defined_behaviors)
        for row in self._custom_rows:
            behaviors.extend(list(row.get("behaviors") or []))
        unique: List[str] = []
        for name in sorted({b.strip() for b in behaviors if b and str(b).strip()}):
            if name in self._hidden_behaviors:
                continue
            unique.append(name)
        self._catalog_behaviors = list(unique)
        current = self._active_behavior
        self._behavior_combo.blockSignals(True)
        self._behavior_combo.clear()
        for behavior in unique:
            self._behavior_combo.addItem(behavior)
        if current and current in unique:
            self._behavior_combo.setCurrentText(current)
        elif unique:
            self._behavior_combo.setCurrentIndex(0)
            self._active_behavior = self._behavior_combo.currentText()
        self._behavior_combo.blockSignals(False)
        self._reset_hidden_button.setVisible(bool(self._hidden_behaviors))

    def _define_behavior(self) -> None:
        name = str(self._behavior_combo.currentText() or "").strip()
        if not name:
            return
        if not name[0].isalpha() and name[0] != "_":
            QtWidgets.QMessageBox.warning(
                self, "Invalid name", "Behavior must start with a letter or underscore."
            )
            return
        if self._catalog_adder is not None:
            try:
                self._catalog_adder(name)
            except Exception:
                pass
        if name not in self._defined_behaviors:
            self._defined_behaviors.append(name)
            self._defined_behaviors.sort()
            self._save_settings()
        self.refresh_behavior_catalog()
        self._behavior_combo.setCurrentText(name)

    def _install_behavior_shortcuts(self) -> None:
        for i in range(1, 10):
            sc = QtWidgets.QShortcut(QtGui.QKeySequence(str(i)), self)
            sc.activated.connect(lambda idx=i: self._select_behavior_by_index(idx))

    def _select_behavior_by_index(self, one_based_index: int) -> None:
        if self._behavior_combo.count() <= 1:
            return
        idx = max(1, min(one_based_index, self._behavior_combo.count() - 1))
        self._behavior_combo.setCurrentIndex(idx)

    def set_active_behavior(self, behavior: str) -> None:
        behavior = str(behavior).strip() if behavior is not None else ""
        if not behavior:
            return
        if (
            behavior not in self._defined_behaviors
            and behavior not in self._hidden_behaviors
        ):
            # Ensure a behavior selected from the Flags dock appears in the catalog.
            self._defined_behaviors.append(behavior)
            self._defined_behaviors.sort()
            self._save_settings()
        self.refresh_behavior_catalog()
        self._behavior_combo.setCurrentText(behavior)
        self._select_behavior_in_track_list(behavior)

    def _select_behavior_in_track_list(self, behavior: str) -> None:
        for row in range(self._track_list.count()):
            item = self._track_list.item(row)
            if not item:
                continue
            raw = item.data(QtCore.Qt.UserRole)
            if raw == behavior:
                self._track_list.setCurrentRow(row)
                break

    def _on_track_selected(self, row: int) -> None:
        item = self._track_list.item(row) if row >= 0 else None
        if item is None:
            return
        track_id = item.data(QtCore.Qt.UserRole)
        if not isinstance(track_id, str) or not track_id:
            return
        if self._row_mode == "Behavior":
            self._behavior_combo.setCurrentText(track_id)
            return
        track = next((t for t in self._model.tracks if t.track_id == track_id), None)
        if track is None:
            return
        if len(track.behaviors) == 1:
            self._behavior_combo.setCurrentText(track.behaviors[0])

    def _hide_selected_behavior(self) -> None:
        behavior: Optional[str] = None
        if self._row_mode == "Behavior":
            item = self._track_list.currentItem()
            if item is not None:
                raw = item.data(QtCore.Qt.UserRole)
                if isinstance(raw, str) and raw:
                    behavior = raw
        if not behavior:
            behavior = self._active_behavior
        if not behavior:
            return
        self._hidden_behaviors.add(behavior)
        if behavior in self._defined_behaviors:
            self._defined_behaviors.remove(behavior)
        self._save_settings()
        if self._active_behavior == behavior:
            self._active_behavior = None
        self.refresh_behavior_catalog()

    def _reset_hidden_behaviors(self) -> None:
        if not self._hidden_behaviors:
            return
        self._hidden_behaviors.clear()
        self._save_settings()
        self.refresh_behavior_catalog()


def _nice_tick_step(span: int) -> int:
    if span <= 10:
        return 1
    rough = max(1, span // 8)
    magnitude = 10 ** (len(str(int(rough))) - 1)
    for step in (1, 2, 5, 10):
        candidate = step * magnitude
        if candidate >= rough:
            return candidate
    return magnitude * 10


def _color_from_getter(
    getter: Optional[Callable[[str], QtGui.QColor]], label: str
) -> QtGui.QColor:
    if getter is None:
        return QtGui.QColor(100, 140, 200)
    try:
        value = getter(label)
    except Exception:
        return QtGui.QColor(100, 140, 200)
    if isinstance(value, QtGui.QColor):
        return value
    if isinstance(value, str):
        return QtGui.QColor(value)
    if isinstance(value, tuple):
        try:
            return QtGui.QColor(*value)
        except Exception:
            return QtGui.QColor(100, 140, 200)
    return QtGui.QColor(100, 140, 200)


def _build_ranges(
    events: Sequence,
    *,
    group_by: Tuple[str, ...] = ("behavior",),
) -> List[Tuple[str, Optional[str], int, int]]:
    grouped: Dict[Tuple, List] = {}
    for event in events:
        key = tuple(getattr(event, field, None) for field in group_by)
        grouped.setdefault(key, []).append(event)

    ranges: List[Tuple[str, Optional[str], int, int]] = []
    for key, group_events in grouped.items():
        behavior = getattr(group_events[0], "behavior", None)
        subject = getattr(group_events[0], "subject", None)
        group_events.sort(key=lambda evt: (evt.frame, 0 if evt.event == "start" else 1))
        open_starts: List[int] = []
        for evt in group_events:
            if evt.event == "start":
                open_starts.append(evt.frame)
            else:
                start_frame = open_starts.pop() if open_starts else evt.frame
                ranges.append((behavior, subject, start_frame, evt.frame))
        for start_frame in open_starts:
            ranges.append((behavior, subject, start_frame, start_frame))
    return ranges


def _ideal_text_color(background: QtGui.QColor) -> QtGui.QColor:
    """Choose a readable text color over a (possibly translucent) background."""
    if not isinstance(background, QtGui.QColor) or not background.isValid():
        return QtGui.QColor(20, 20, 20)
    r, g, b, a = (
        background.red(),
        background.green(),
        background.blue(),
        background.alpha(),
    )
    if a < 120:
        return QtGui.QColor(20, 20, 20)
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return QtGui.QColor(18, 18, 18) if luminance > 165 else QtGui.QColor(250, 250, 250)


def _timeline_panel_stylesheet() -> str:
    # Intentionally scoped via objectNames to avoid bleeding into the rest of the app.
    return """
QFrame#TimelineSidebar {
  background: #1d1f23;
  border: 1px solid #2a2d33;
  border-radius: 10px;
}
QFrame#TimelineSidebar QLabel {
  color: #cfd3da;
}
QFrame#TimelineSidebar QComboBox,
QFrame#TimelineSidebar QLineEdit {
  background: #15171a;
  color: #e8eaed;
  border: 1px solid #2a2d33;
  border-radius: 8px;
  padding: 6px 10px;
}
QFrame#TimelineSidebar QComboBox::drop-down {
  border: 0px;
  width: 22px;
}
QFrame#TimelineSidebar QToolButton {
  background: transparent;
  color: #e8eaed;
  border: 1px solid #2a2d33;
  border-radius: 8px;
  padding: 6px;
}
QFrame#TimelineSidebar QToolButton:hover {
  background: #24262b;
}
QFrame#TimelineSidebar QToolButton:checked {
  background: #2d4f7a;
  border-color: #3d6aa1;
}
QListWidget#TimelineTrackList {
  background: #15171a;
  border: 1px solid #2a2d33;
  border-radius: 8px;
  padding: 4px;
  outline: none;
}
QListWidget#TimelineTrackList::item {
  padding: 6px 6px;
  border-radius: 6px;
  color: #eef1f5;
}
QListWidget#TimelineTrackList::item:selected {
  background: #2d4f7a;
}
QGraphicsView#TimelineView {
  border: 1px solid #d8dde6;
  border-radius: 10px;
  background: #f6f7f9;
}
"""


class TimelineRowConfigDialog(QtWidgets.QDialog):
    def __init__(
        self, rows: List[Dict[str, Any]], parent: Optional[QtWidgets.QWidget] = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Timeline Rows")
        self._table = QtWidgets.QTableWidget(0, 3, self)
        self._table.setHorizontalHeaderLabels(["Label", "Behaviors", "Subject"])
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)

        add_btn = QtWidgets.QPushButton("Add")
        remove_btn = QtWidgets.QPushButton("Remove")
        add_btn.clicked.connect(self._add_row)
        remove_btn.clicked.connect(self._remove_selected)

        button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addWidget(add_btn)
        btn_row.addWidget(remove_btn)
        btn_row.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._table, 1)
        layout.addLayout(btn_row)
        layout.addWidget(button_box)

        for row in rows:
            self._add_row(
                label=row.get("label", ""),
                behaviors=",".join(row.get("behaviors", []) or []),
                subject=row.get("subject", "") or "",
            )

    def _add_row(self, label: str = "", behaviors: str = "", subject: str = "") -> None:
        row_idx = self._table.rowCount()
        self._table.insertRow(row_idx)
        self._table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(label))
        self._table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(behaviors))
        self._table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(subject))

    def _remove_selected(self) -> None:
        selected = sorted(
            {idx.row() for idx in self._table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row_idx in selected:
            self._table.removeRow(row_idx)

    def rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for row_idx in range(self._table.rowCount()):
            label = self._text_at(row_idx, 0)
            behaviors = [
                b.strip() for b in self._text_at(row_idx, 1).split(",") if b.strip()
            ]
            subject = self._text_at(row_idx, 2).strip()
            row_id = _slugify(label or f"row_{row_idx + 1}")
            rows.append(
                {
                    "id": row_id,
                    "label": label or row_id,
                    "behaviors": behaviors,
                    "subject": subject if subject else None,
                }
            )
        return rows

    def _text_at(self, row_idx: int, col: int) -> str:
        item = self._table.item(row_idx, col)
        return item.text() if item is not None else ""


def _slugify(text: str) -> str:
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in text.strip()
    )
    return cleaned or "row"
