"""
    Customized video slider.
    Modified from here
    https://github.com/murthylab/sleap/blob/1eb06f81eb8f0bc1beedd1c3dd10902f8ff9e724/sleap/gui/widgets/slider.py

"""
from dataclasses import dataclass
import itertools
import numpy as np
from qtpy import QtCore, QtWidgets, QtGui
from qtpy.QtGui import (QPen, QBrush, QColor, QKeyEvent,
                        QPolygonF, QPainterPath)


from typing import (Callable, Dict,
                    Iterable, List,
                    Optional, Tuple,
                    Union)


@dataclass(frozen=True)
class VideoSliderMark:
    """
    Class to hold data for an individual mark on the slider.

    Attributes:
        mark_type: Type of the mark, options are:
            * "simple"     (single value)
            * "simple_thin" (    ditto   )
            * "filled"
            * "open"
            * "predicted"
            * "tick"
            * "tick_column"
            * "event_start"
            * "event_end"
        val: Beginning of mark range
        color: Color of mark, can be string or (r, g, b) tuple.
        filled: Whether the mark is shown filled (solid color).
    """

    mark_type: str
    val: float
    end_val: float = None
    row: int = None
    _color: Union[tuple, str] = "black"

    @property
    def color(self):
        """Returns color of mark."""
        colors = dict(
            simple="black",
            simple_thin="black",
            filled="blue",
            open="blue",
            predicted=(1, 170, 247),  # light blue
            tick="lightGray",
            tick_column="gray",
            event_start="green",
            event_end="red",
        )

        if self.mark_type in colors:
            return colors[self.mark_type]
        else:
            return self._color

    @color.setter
    def color(self, val):
        """Sets color of mark."""
        self._color = val

    @property
    def QColor(self):
        """Returns color of mark as `QColor`."""
        c = self.color
        if type(c) == str:
            return QColor(c)
        else:
            return QColor(*c)

    @property
    def filled(self):
        """Returns whether mark is filled or open."""
        if self.mark_type == "open":
            return False
        else:
            return True

    @property
    def top_pad(self):
        if self.mark_type in ["tick_column"]:
            return 40
        if self.mark_type == "tick":
            return 0
        return 2

    @property
    def bottom_pad(self):
        if self.mark_type in ["tick_column"]:
            return 200
        if self.mark_type == "tick":
            return 0
        return 2

    @property
    def visual_width(self):
        if self.mark_type in ("open", "filled", "tick", "event_start", "event_end"):
            return 2
        if self.mark_type in ("tick_column", "simple", "predicted"):
            return 1
        return 2

    def get_height(self, container_height):
        height = container_height
        # if self.padded:
        height -= self.top_pad + self.bottom_pad

        return height


class VideoSlider(QtWidgets.QGraphicsView):
    """Drop-in replacement for QSlider with additional features.

    Args:
        orientation: ignored (here for compatibility with QSlider)
        min: initial minimum value
        max: initial maximum value
        val: initial value
        marks: initial set of values to mark on slider
            this can be either
            * list of values to mark

    Signals:
        mousePressed: triggered on Qt event
        mouseMoved: triggered on Qt event
        mouseReleased: triggered on Qt event
        keyPress: triggered on Qt event
        keyReleased: triggered on Qt event
        valueChanged: triggered when value of slider changes
        selectionChanged: triggered when slider range selection changes
    """

    mousePressed = QtCore.Signal(float, float)
    mouseMoved = QtCore.Signal(float, float)
    mouseReleased = QtCore.Signal(float, float)
    keyPress = QtCore.Signal(QKeyEvent)
    keyRelease = QtCore.Signal(QKeyEvent)
    valueChanged = QtCore.Signal(int)
    selectionChanged = QtCore.Signal(int, int)
    heightUpdated = QtCore.Signal()

    def __init__(
        self,
        orientation=-1,  # for compatibility with QSlider
        min=0,
        max=1,
        val=0,
        marks=None,
        *args,
        **kwargs,
    ):
        super(VideoSlider, self).__init__(*args, **kwargs)

        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)
        self.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)

        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        self.setMouseTracking(True)

        self._get_val_tooltip = None

        self.tick_index_offset = 1
        self.zoom_factor = 1

        self._header_label_height = 20
        self._header_graph_height = 40
        self._header_height = self._header_label_height  # room for frame labels
        self._min_height = 19 + self._header_height

        self._base_font = QtGui.QFont()
        self._base_font.setPixelSize(10)

        self._tick_marks = []

        # Add border rect
        outline_rect = QtCore.QRectF(0, 0, 200, self._min_height - 3)
        self.box_rect = outline_rect
        # self.outlineBox = self.scene.addRect(outline_rect)
        # self.outlineBox.setPen(QPen(QColor("black", alpha=0)))

        # Add drag handle rect
        self._handle_width = 6
        handle_rect = QtCore.QRectF(
            0, self._handle_top, self._handle_width, self._handle_height
        )
        self.setMinimumHeight(self._min_height)
        self.setMaximumHeight(self._min_height)
        self.handle = self.scene.addRect(QtCore.QRectF(
            0, self._handle_top, self._handle_width, self._handle_height
        ))
        self.handle.setPen(QPen(QColor(80, 80, 80)))
        self.handle.setBrush(QColor(128, 128, 128, 128))

        # Add (hidden) rect to highlight selection
        self.select_box = self.scene.addRect(
            QtCore.QRectF(0, 1, 0, outline_rect.height() - 2)
        )
        self.select_box.setPen(QPen(QColor(80, 80, 255)))
        self.select_box.setBrush(QColor(80, 80, 255, 128))
        self.select_box.hide()

        self.zoom_box = self.scene.addRect(
            QtCore.QRectF(0, 1, 0, outline_rect.height() - 2)
        )
        self.zoom_box.setPen(QPen(QColor(80, 80, 80, 64)))
        self.zoom_box.setBrush(QColor(80, 80, 80, 64))
        self.zoom_box.hide()

        self.scene.setBackgroundBrush(QBrush(QColor(200, 200, 200)))

        self.clearSelection()
        self.setEnabled(True)
        self.setMinimum(min)
        self.setMaximum(max)
        self.setValue(val)
        self.setMarks(marks)

        pen = QPen(QColor(80, 80, 255), 0.5)
        pen.setCosmetic(True)
        self.poly = self.scene.addPath(
            QPainterPath(), pen, self.select_box.brush())
        self.headerSeries = dict()
        self._draw_header()

    # Methods to match API for QSlider

    def value(self) -> float:
        """Returns value of slider."""
        return self._val_main

    def setValue(self, val: float) -> float:
        """Sets value of slider."""
        self._val_main = val
        x = self._toPos(val)
        self.handle.setPos(x, 0)
        self.ensureVisible(x, 0, self._handle_width, 0, 3, 0)

    def setMinimum(self, min: float) -> float:
        """Sets minimum value for slider."""
        self._val_min = min

    def setMaximum(self, max: float) -> float:
        """Sets maximum value for slider."""
        self._val_max = max

    def setEnabled(self, val: float) -> float:
        """Set whether the slider is enabled."""
        self._enabled = val

    def enabled(self):
        """Returns whether slider is enabled."""
        return self._enabled

    # Methods for working with visual positions (mapping to and from, redrawing)

    def _update_visual_positions(self):
        """Updates the visual x position of handle and slider annotations."""
        x = self._toPos(self.value())
        self.handle.setPos(x, 0)

        for mark in self._mark_items.keys():

            width = mark.visual_width

            x = self._toPos(mark.val, center=True)
            self._mark_items[mark].setPos(x, 0)

            if mark in self._mark_labels:
                label_x = max(
                    0, x - self._mark_labels[mark].boundingRect().width() // 2
                )
                self._mark_labels[mark].setPos(label_x, 4)

            rect = self._mark_items[mark].rect()
            rect.setWidth(width)
            rect.setHeight(
                mark.get_height(
                    container_height=self.box_rect.height() - self._header_height
                )
            )

            self._mark_items[mark].setRect(rect)

    def _get_min_max_slider_heights(self):

        # Start with padding height
        extra_height = 8 + self._header_height
        min_height = extra_height
        max_height = extra_height
        # Make sure min/max height is at least 19, even if few tracks
        min_height = max(self._min_height, min_height)
        max_height = max(self._min_height, max_height)

        return min_height, max_height

    def _update_slider_height(self):
        """Update the height of the slider."""

        min_height, max_height = self._get_min_max_slider_heights()

        # TODO: find the current height of the scrollbar
        # self.horizontalScrollBar().height() gives the wrong value
        scrollbar_height = 18

        self.setMaximumHeight(max_height + scrollbar_height)
        self.setMinimumHeight(min_height + scrollbar_height)

        # Redraw all marks with new height and y position
        marks = self.getMarks()
        self.setMarks(marks)

        self.resizeEvent()
        self.heightUpdated.emit()

    def _toPos(self, val: float, center=False) -> float:
        """
        Converts slider value to x position on slider.

        Args:
            val: The slider value.
            center: Whether to offset by half the width of drag handle,
                so that plotted location will light up with center of handle.

        Returns:
            x position.
        """
        x = val
        x -= self._val_min
        x /= max(1, self._val_max - self._val_min)
        x *= self._slider_width
        if center:
            x += self.handle.rect().width() / 2.0
        return x

    def _toVal(self, x: float, center=False) -> float:
        """Converts x position to slider value."""
        val = x
        val /= self._slider_width
        val *= max(1, self._val_max - self._val_min)
        val += self._val_min
        val = round(val)
        return val

    @property
    def _slider_width(self) -> float:
        """Returns visual width of slider."""
        return self.box_rect.width() - self.handle.rect().width()

    @property
    def slider_visible_value_range(self) -> float:
        """Value range that's visible given current size and zoom."""
        return self._toVal(self.width() - 1)

    @property
    def _mark_area_height(self) -> float:
        _, max_height = self._get_min_max_slider_heights()
        return max_height - 3 - self._header_height

    @property
    def value_range(self) -> float:
        return self._val_max - self._val_min

    @property
    def box_rect(self) -> QtCore.QRectF:
        return self._box_rect

    @box_rect.setter
    def box_rect(self, rect: QtCore.QRectF):
        self._box_rect = rect

        # Update the scene rect so that it matches how much space we
        # currently want for drawing everything.
        rect.setWidth(rect.width() - 1)
        self.setSceneRect(rect)

    # Methods for range selection and zoom

    def clearSelection(self):
        """Clears selection endpoints."""
        self._selection = []
        self.select_box.hide()

    def startSelection(self, val):
        """Adds initial selection endpoint.

        Called when user starts dragging to select range in slider.

        Args:
            val: value of endpoint
        """
        self._selection.append(val)

    def endSelection(self, val, update: bool = False):
        """Add final selection endpoint.

        Called during or after the user is dragging to select range.

        Args:
            val: value of endpoint
            update:
        """
        # If we want to update endpoint and there's already one, remove it
        if update and len(self._selection) % 2 == 0:
            self._selection.pop()
        # Add the selection endpoint
        self._selection.append(val)
        a, b = self._selection[-2:]
        if a == b:
            self.clearSelection()
        else:
            self._draw_selection(a, b)
        # Emit signal (even if user selected same region as before)
        self.selectionChanged.emit(*self.getSelection())

    def setSelection(self, start_val, end_val):
        """Selects clip from start_val to end_val."""
        self.startSelection(start_val)
        self.endSelection(end_val, update=True)

    def hasSelection(self) -> bool:
        """Returns True if a clip is selected, False otherwise."""
        a, b = self.getSelection()
        return a < b

    def getSelection(self):
        """Returns start and end value of current selection endpoints."""
        a, b = 0, 0
        if len(self._selection) % 2 == 0 and len(self._selection) > 0:
            a, b = self._selection[-2:]
        start = min(a, b)
        end = max(a, b)
        return start, end

    def _draw_selection(self, a: float, b: float):
        self._update_selection_box_positions(self.select_box, a, b)

    def _draw_zoom_box(self, a: float, b: float):
        self._update_selection_box_positions(self.zoom_box, a, b)

    def _update_selection_box_positions(self, box_object, a: float, b: float):
        """Update box item on slider.

        Args:
            box_object: The box to update
            a: one endpoint value
            b: other endpoint value

        Returns:
            None.
        """
        start = min(a, b)
        end = max(a, b)
        start_pos = self._toPos(start, center=True)
        end_pos = self._toPos(end, center=True)
        box_rect = QtCore.QRectF(
            start_pos,
            self._header_height,
            end_pos - start_pos,
            self.box_rect.height(),
        )

        box_object.setRect(box_rect)
        box_object.show()

    def _update_selection_boxes_on_resize(self):
        for box_object in (self.select_box, self.zoom_box):
            rect = box_object.rect()
            rect.setHeight(self._handle_height)
            box_object.setRect(rect)

        if self.select_box.isVisible():
            self._draw_selection(*self.getSelection())

    def moveSelectionAnchor(self, x: float, y: float):
        """
        Moves selection anchor in response to mouse position.

        Args:
            x: x position of mouse
            y: y position of mouse

        Returns:
            None.
        """
        x = max(x, 0)
        x = min(x, self.box_rect.width())
        anchor_val = self._toVal(x, center=True)

        if len(self._selection) % 2 == 0:
            self.startSelection(anchor_val)

        self._draw_selection(anchor_val, self._selection[-1])

    def releaseSelectionAnchor(self, x, y):
        """
        Finishes selection in response to mouse release.

        Args:
            x: x position of mouse
            y: y position of mouse

        Returns:
            None.
        """
        x = max(x, 0)
        x = min(x, self.box_rect.width())
        anchor_val = self._toVal(x)
        self.endSelection(anchor_val)

    def moveZoomDrag(self, x: float, y: float):
        if getattr(self, "_zoom_start_val", None) is None:
            self._zoom_start_val = self._toVal(x, center=True)

        current_val = self._toVal(x, center=True)

        self._draw_zoom_box(current_val, self._zoom_start_val)

    def releaseZoomDrag(self, x, y):

        self.zoom_box.hide()

        val_a = self._zoom_start_val
        val_b = self._toVal(x, center=True)

        val_start = min(val_a, val_b)
        val_end = max(val_a, val_b)

        # pad the zoom
        val_range = val_end - val_start
        val_start -= val_range * 0.05
        val_end += val_range * 0.05

        self.setZoomRange(val_start, val_end)

        self._zoom_start_val = None

    def setZoomRange(self, start_val: float, end_val: float):

        zoom_val_range = end_val - start_val
        if zoom_val_range > 0:
            self.zoom_factor = self.value_range / zoom_val_range
        else:
            self.zoom_factor = 1

        self.resizeEvent()

        center_val = start_val + zoom_val_range / 2
        center_pos = self._toPos(center_val)

        self.centerOn(center_pos, 0)

    # Methods for modifying marks on slider

    def clearMarks(self):
        """Clears all marked values for slider."""
        if hasattr(self, "_mark_items"):
            for item in self._mark_items.values():
                self.scene.removeItem(item)

        if hasattr(self, "_mark_labels"):
            for item in self._mark_labels.values():
                self.scene.removeItem(item)

        self._marks = set()  # holds mark position
        self._mark_items = dict()  # holds visual Qt object for plotting mark
        self._mark_labels = dict()

    def setMarks(self, marks: Iterable[Union[VideoSliderMark, int]]):
        """Sets all marked values for the slider.

        Args:
            marks: iterable with all values to mark

        Returns:
            None.
        """
        self.clearMarks()

        # Add tick marks first so they're behind other marks
        self._add_tick_marks()

        if marks is not None:
            for mark in marks:
                if not isinstance(mark, VideoSliderMark):
                    mark = VideoSliderMark("simple", mark)
                self.addMark(mark, update=False)

        self._update_visual_positions()

    def setTickMarks(self):
        """Resets which tick marks to show."""
        self._clear_tick_marks()
        self._add_tick_marks()

    def _clear_tick_marks(self):
        if not hasattr(self, "_tick_marks"):
            return

        for mark in self._tick_marks:
            self.removeMark(mark)

    def _add_tick_marks(self):
        val_range = self.slider_visible_value_range

        if val_range < 20:
            val_order = 1
        else:
            val_order = 10
            while val_range // val_order > 24:
                val_order *= 10

        self._tick_marks = []

        for tick_pos in range(
            self._val_min + val_order - 1, self._val_max + 1, val_order
        ):
            self._tick_marks.append(VideoSliderMark("tick", tick_pos))

        for tick_mark in self._tick_marks:
            self.addMark(tick_mark, update=False)

    def removeMark(self, mark: VideoSliderMark):
        """Removes an individual mark."""
        if mark in self._mark_labels:
            self.scene.removeItem(self._mark_labels[mark])
            del self._mark_labels[mark]
        if mark in self._mark_items:
            self.scene.removeItem(self._mark_items[mark])
            del self._mark_items[mark]
        if mark in self._marks:
            self._marks.remove(mark)

    def getMarks(self, mark_type: str = ""):
        """Returns list of marks."""
        if mark_type:
            return [mark for mark in self._marks if mark.mark_type == type]

        return self._marks

    def addMark(self, new_mark: VideoSliderMark, update: bool = True):
        """Adds a marked value to the slider.

        Args:
            new_mark: value to mark
            update: Whether to redraw slider with new mark.

        Returns:
            None.
        """
        # check if mark is within slider range
        if new_mark.val > self._val_max:
            return
        if new_mark.val < self._val_min:
            return

        self._marks.add(new_mark)

        v_top_pad = self._header_height + 1
        v_bottom_pad = 1
        v_top_pad += new_mark.top_pad
        v_bottom_pad += new_mark.bottom_pad

        width = new_mark.visual_width

        v_offset = v_top_pad

        height = new_mark.get_height(
            container_height=self.box_rect.height() - self._header_height
        )

        color = new_mark.QColor
        pen = QPen(color, 0.5)
        pen.setCosmetic(True)
        brush = QBrush(color) if new_mark.filled else QBrush()

        line = self.scene.addRect(-width // 2, v_offset,
                                  width, height, pen, brush)
        self._mark_items[new_mark] = line

        if new_mark.mark_type in ["tick", "event_start", "event_end"]:
            # Show tick mark behind other slider marks
            self._mark_items[new_mark].setZValue(0)

            # Add a text label to show in header area
            mark_label_text = (
                # sci notation if large
                f"{new_mark.val + self.tick_index_offset:g}"
            )
            self._mark_labels[new_mark] = self.scene.addSimpleText(
                mark_label_text, self._base_font
            )
        else:
            # Show in front of tick marks and behind track lines
            self._mark_items[new_mark].setZValue(1)

        if update:
            self._update_visual_positions()

    # Methods for header graph

    def setHeaderSeries(self, series: Optional[Dict[int, float]] = None):
        """Show header graph with specified series.

        Args:
            series: {frame number: series value} dict.
        Returns:
            None.
        """
        self.headerSeries = [] if series is None else series
        self._header_height = self._header_label_height + self._header_graph_height
        self._draw_header()
        self._update_slider_height()

    def clearHeader(self):
        """Remove header graph from slider."""
        self.headerSeries = []
        self._header_height = self._header_label_height
        self._update_slider_height()

    def _get_header_series_len(self):
        if hasattr(self.headerSeries, "keys"):
            series_frame_max = max(self.headerSeries.keys())
        else:
            series_frame_max = len(self.headerSeries)
        return series_frame_max

    @property
    def _header_series_items(self):
        """Yields (frame idx, val) for header series items."""
        if hasattr(self.headerSeries, "items"):
            for key, val in self.headerSeries.items():
                yield key, val
        else:
            for key in range(len(self.headerSeries)):
                val = self.headerSeries[key]
                yield key, val

    def _draw_header(self):
        """Draws the header graph."""
        if len(self.headerSeries) == 0 or self._header_height == 0:
            self.poly.setPath(QPainterPath())
            return

        series_frame_max = self._get_header_series_len()

        step = series_frame_max // int(self._slider_width)
        step = max(step, 1)
        count = series_frame_max // step * step

        sampled = np.full((count), 0.0, dtype=np.float)

        for key, val in self._header_series_items:
            if key < count:
                sampled[key] = val

        sampled = np.max(sampled.reshape(count // step, step), axis=1)
        series = {i * step: sampled[i] for i in range(count // step)}

        series_min = np.min(sampled) - 1
        series_max = np.max(sampled)
        series_scale = (self._header_graph_height) / (series_max - series_min)

        def toYPos(val):
            return self._header_height - ((val - series_min) * series_scale)

        step_chart = False  # use steps rather than smooth line

        points = []
        points.append((self._toPos(0, center=True), toYPos(series_min)))
        for idx, val in series.items():
            points.append((self._toPos(idx, center=True), toYPos(val)))
            if step_chart:
                points.append(
                    (self._toPos(idx + step, center=True), toYPos(val)))
        points.append(
            (self._toPos(max(series.keys()) + 1, center=True), toYPos(series_min))
        )

        # Convert to list of QtCore.QPointF objects
        points = list(itertools.starmap(QtCore.QPointF, points))
        self.poly.setPath(self._pointsToPath(points))

    def _pointsToPath(self, points: List[QtCore.QPointF]) -> QPainterPath:
        """Converts list of `QtCore.QPointF` objects to a `QPainterPath`."""
        path = QPainterPath()
        path.addPolygon(QPolygonF(points))
        return path

    # Methods for working with slider handle

    def mapMouseXToHandleX(self, x) -> float:
        x -= self.handle.rect().width() / 2.0
        x = max(x, 0)
        x = min(x, self.box_rect.width() - self.handle.rect().width())
        return x

    def moveHandle(self, x, y):
        """Move handle in response to mouse position.

        Emits valueChanged signal if value of slider changed.

        Args:
            x: x position of mouse
            y: y position of mouse
        """
        x = self.mapMouseXToHandleX(x)

        val = self._toVal(x)

        # snap to nearby mark within handle
        mark_vals = [mark.val for mark in self._marks]
        handle_left = self._toVal(x - self.handle.rect().width() / 2)
        handle_right = self._toVal(x + self.handle.rect().width() / 2)
        marks_in_handle = [
            mark for mark in mark_vals if handle_left < mark < handle_right
        ]
        if marks_in_handle:
            marks_in_handle.sort(key=lambda m: (abs(m - val), m > val))
            val = marks_in_handle[0]

        old = self.value()
        self.setValue(val)

        if old != val:
            self.valueChanged.emit(self._val_main)

    @property
    def _handle_top(self) -> float:
        """Returns y position of top of handle (i.e., header height)."""
        return 1 + self._header_height

    @property
    def _handle_height(self, outline_rect=None) -> float:
        """
        Returns visual height of handle.

        Args:
            outline_rect: The rect of the outline box for the slider. This
                is only required when calling during initialization (when the
                outline box doesn't yet exist).

        Returns:
            Height of handle in pixels.
        """
        return self._mark_area_height

    # Methods for selection of contiguously marked ranges of frames

    def contiguousSelectionMarksAroundVal(self, val):
        """Selects contiguously marked frames around value."""
        if not self.isMarkedVal(val):
            return

        dec_val = self.getStartContiguousMark(val)
        inc_val = self.getEndContiguousMark(val)

        self.setSelection(dec_val, inc_val)

    def getStartContiguousMark(self, val: int) -> int:
        """
        Returns first marked value in contiguously marked region around val.
        """
        last_val = val
        dec_val = self._dec_contiguous_marked_val(last_val)
        while last_val > dec_val > self._val_min:
            last_val = dec_val
            dec_val = self._dec_contiguous_marked_val(last_val)

        return dec_val

    def getEndContiguousMark(self, val: int) -> int:
        """
        Returns last marked value in contiguously marked region around val.
        """
        last_val = val
        inc_val = self._inc_contiguous_marked_val(last_val)
        while last_val < inc_val < self._val_max:
            last_val = inc_val
            inc_val = self._inc_contiguous_marked_val(last_val)

        return inc_val

    def getMarksAtVal(self, val: int) -> List[VideoSliderMark]:
        if val is None:
            return []

        return [
            mark
            for mark in self._marks
            if (mark.val == val and mark.mark_type not in ("tick", "tick_column", "event_start", "event_end"))
        ]

    def isMarkedVal(self, val: int) -> bool:
        """Returns whether value has mark."""
        if self.getMarksAtVal(val):
            return True
        return False

    def _dec_contiguous_marked_val(self, val):
        """Decrements value within contiguously marked range if possible."""
        dec_val = min(
            (
                mark.val
                for mark in self._marks
                if mark.val < val <= mark.end_val
            ),
            default=val,
        )
        if dec_val < val:
            return dec_val

        if val - 1 in [mark.val for mark in self._marks]:
            return val - 1

        # Return original value if we can't decrement it w/in contiguous range
        return val

    def _inc_contiguous_marked_val(self, val):
        """Increments value within contiguously marked range if possible."""
        inc_val = max(
            (
                mark.end_val - 1
                for mark in self._marks
                if mark.val <= val < mark.end_val
            ),
            default=val,
        )
        if inc_val > val:
            return inc_val

        if val + 1 in [mark.val for mark in self._marks]:
            return val + 1

        # Return original value if we can't decrement it w/in contiguous range
        return val

    # Method for cursor

    def setTooltipCallable(self, tooltip_callable: Callable):
        """
        Sets function to get tooltip text for given value in slider.

        Args:
            tooltip_callable: a function which takes the value which the user
                is hovering over and returns the tooltip text to show (if any)
        """
        self._get_val_tooltip = tooltip_callable

    def _update_cursor_for_event(self, event):
        if event.modifiers() == QtCore.Qt.ShiftModifier:
            self.setCursor(QtCore.Qt.CrossCursor)
        elif event.modifiers() == QtCore.Qt.AltModifier:
            self.setCursor(QtCore.Qt.SizeHorCursor)
        else:
            self.unsetCursor()

    # Methods which override QGraphicsView

    def resizeEvent(self, event=None):
        """Override method to update visual size when necessary.

        Args:
            event
        """

        outline_rect = self.box_rect
        handle_rect = self.handle.rect()

        outline_rect.setHeight(self._mark_area_height + self._header_height)

        if event is not None:
            visual_width = event.size().width() - 1
        else:
            visual_width = self.width() - 1

        drawn_width = visual_width * self.zoom_factor

        outline_rect.setWidth(drawn_width)
        self.box_rect = outline_rect

        handle_rect.setTop(self._handle_top)
        handle_rect.setHeight(self._handle_height)
        self.handle.setRect(handle_rect)

        self._update_selection_boxes_on_resize()

        self.setTickMarks()
        self._update_visual_positions()
        self._draw_header()

        super(VideoSlider, self).resizeEvent(event)

    def mousePressEvent(self, event):
        """Override method to move handle for mouse press/drag.

        Args:
            event
        """
        scenePos = self.mapToScene(event.pos())

        # Do nothing if not enabled
        if not self.enabled():
            return
        # Do nothing if click outside slider area
        if not self.box_rect.contains(scenePos):
            return

        move_function = None
        release_function = None

        self._update_cursor_for_event(event)

        # Shift : selection
        if event.modifiers() == QtCore.Qt.ShiftModifier:
            move_function = self.moveSelectionAnchor
            release_function = self.releaseSelectionAnchor

            self.clearSelection()

        # No modifier : go to frame
        elif event.modifiers() == QtCore.Qt.NoModifier:
            move_function = self.moveHandle
            release_function = None

        # Alt (option) : zoom
        elif event.modifiers() == QtCore.Qt.AltModifier:
            move_function = self.moveZoomDrag
            release_function = self.releaseZoomDrag

        else:
            event.accept()  # mouse events shouldn't be passed to video widgets

        # Connect to signals
        if move_function is not None:
            self.mouseMoved.connect(move_function)

        def done(x, y):
            self.unsetCursor()
            if release_function is not None:
                release_function(x, y)
            if move_function is not None:
                self.mouseMoved.disconnect(move_function)
            self.mouseReleased.disconnect(done)

        self.mouseReleased.connect(done)

        # Emit signal
        self.mouseMoved.emit(scenePos.x(), scenePos.y())
        self.mousePressed.emit(scenePos.x(), scenePos.y())

    def mouseMoveEvent(self, event):
        """Override method to emit mouseMoved signal on drag."""
        scenePos = self.mapToScene(event.pos())

        # Update cursor type based on current modifier key
        self._update_cursor_for_event(event)

        # Show tooltip with information about frame under mouse
        if self._get_val_tooltip:
            hover_frame_idx = self._toVal(
                self.mapMouseXToHandleX(scenePos.x()))
            tooltip = self._get_val_tooltip(hover_frame_idx)
            QtWidgets.QToolTip.showText(event.globalPos(), tooltip)

        self.mouseMoved.emit(scenePos.x(), scenePos.y())

    def mouseReleaseEvent(self, event):
        """Override method to emit mouseReleased signal on release."""
        scenePos = self.mapToScene(event.pos())

        self.mouseReleased.emit(scenePos.x(), scenePos.y())

    def mouseDoubleClickEvent(self, event):
        """Override method to move handle for mouse double-click.

        Args:
            event
        """
        scenePos = self.mapToScene(event.pos())

        # Do nothing if not enabled
        if not self.enabled():
            return
        # Do nothing if click outside slider area
        if not self.box_rect.contains(scenePos):
            return

        if event.modifiers() == QtCore.Qt.ShiftModifier:
            self.contiguousSelectionMarksAroundVal(self._toVal(scenePos.x()))

    def leaveEvent(self, event):
        self.unsetCursor()

    def keyPressEvent(self, event):
        """Catch event and emit signal so something else can handle event."""
        self._update_cursor_for_event(event)
        self.keyPress.emit(event)
        event.accept()

    def keyReleaseEvent(self, event):
        """Catch event and emit signal so something else can handle event."""
        self.unsetCursor()
        self.keyRelease.emit(event)
        event.accept()

    def boundingRect(self) -> QtCore.QRectF:
        """Method required by Qt."""
        return self.box_rect

    def paint(self, *args, **kwargs):
        """Method required by Qt."""
        super(VideoSlider, self).paint(*args, **kwargs)
