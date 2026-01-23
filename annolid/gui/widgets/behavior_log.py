from __future__ import annotations

from typing import Iterable, List, Optional

from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtGui import QKeySequence

from annolid.gui.behavior_controller import BehaviorEvent


def _format_seconds(value: Optional[float]) -> str:
    if value is None:
        return "—"
    minutes, seconds = divmod(value, 60.0)
    return f"{int(minutes):02d}:{seconds:05.2f}"


class BehaviorEventLogWidget(QtWidgets.QWidget):
    """Table view listing recorded behavior events with quick actions."""

    jumpToFrame = QtCore.Signal(int)
    behaviorSelected = QtCore.Signal(object)
    editRequested = QtCore.Signal(object)
    deleteRequested = QtCore.Signal(object)
    confirmRequested = QtCore.Signal(object)
    rejectRequested = QtCore.Signal(object)
    undoRequested = QtCore.Signal()
    clearRequested = QtCore.Signal()

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        color_getter=None,
    ) -> None:
        super().__init__(parent)
        self._events: List[BehaviorEvent] = []
        self._display_events: List[BehaviorEvent] = []
        self._fps: Optional[float] = None
        self._filter_subject: Optional[str] = None
        self._filter_category: Optional[str] = None
        self._color_getter = color_getter

        self._table = QtWidgets.QTableWidget(0, 10)
        self._table.setObjectName("behaviorEventTable")
        self._table.setHorizontalHeaderLabels(
            [
                "#",
                "Behavior",
                "Event",
                "Subject",
                "Modifiers",
                "Category",
                "Time",
                "Frame",
                "Duration",
                "Status",
            ]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.doubleClicked.connect(self._handle_double_click)

        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        filter_layout = QtWidgets.QHBoxLayout()
        filter_layout.addWidget(QtWidgets.QLabel("Subject:"))
        self._subject_filter = QtWidgets.QComboBox()
        self._subject_filter.addItem("All Subjects", None)
        self._subject_filter.currentIndexChanged.connect(
            self._on_subject_filter_changed
        )
        filter_layout.addWidget(self._subject_filter)

        filter_layout.addWidget(QtWidgets.QLabel("Category:"))
        self._category_filter = QtWidgets.QComboBox()
        self._category_filter.addItem("All Categories", None)
        self._category_filter.currentIndexChanged.connect(
            self._on_category_filter_changed
        )
        filter_layout.addWidget(self._category_filter)
        filter_layout.addStretch(1)

        self._undo_button = QtWidgets.QPushButton("Undo Last")
        self._undo_button.setToolTip("Remove the most recently added behavior event")
        self._undo_button.clicked.connect(self.undoRequested.emit)

        self._clear_button = QtWidgets.QPushButton("Clear")
        self._clear_button.setToolTip("Remove all recorded behavior events")
        self._clear_button.clicked.connect(self._confirm_clear)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self._undo_button)
        button_layout.addWidget(self._clear_button)
        button_layout.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addLayout(filter_layout)
        layout.addWidget(self._table)
        layout.addLayout(button_layout)

        self._shortcuts = [
            QtWidgets.QShortcut(QKeySequence("Ctrl+Z"), self, self.undoRequested.emit),
            QtWidgets.QShortcut(
                QKeySequence("Ctrl+Shift+Z"), self, self.undoRequested.emit
            ),
            QtWidgets.QShortcut(
                QKeySequence("Ctrl+Backspace"), self, self._confirm_clear
            ),
        ]

    def set_fps(self, fps: Optional[float]) -> None:
        if fps is not None and fps > 0:
            self._fps = float(fps)

    def set_events(
        self,
        events: Iterable[BehaviorEvent],
        *,
        fps: Optional[float] = None,
    ) -> None:
        if fps is not None and fps > 0:
            self._fps = float(fps)
        self._events = sorted(
            list(events), key=lambda evt: (evt.frame, 0 if evt.event == "start" else 1)
        )
        self._update_filter_options()
        self._refresh()

    def append_event(
        self,
        event: BehaviorEvent,
        *,
        fps: Optional[float] = None,
    ) -> None:
        if fps is not None and fps > 0:
            self._fps = float(fps)
        self._events.append(event)
        self._events.sort(key=lambda evt: (evt.frame, 0 if evt.event == "start" else 1))
        self._update_filter_options()
        self._refresh()

    def remove_event(self, key: tuple[int, str, str]) -> None:
        self._events = [evt for evt in self._events if evt.mark_key != key]
        self._update_filter_options()
        self._refresh()

    def clear(self) -> None:
        self._events.clear()
        self._display_events.clear()
        self._table.setRowCount(0)
        self._undo_button.setEnabled(False)
        self._clear_button.setEnabled(False)
        with QtCore.QSignalBlocker(self._subject_filter):
            self._subject_filter.clear()
            self._subject_filter.addItem("All Subjects", None)
        with QtCore.QSignalBlocker(self._category_filter):
            self._category_filter.clear()
            self._category_filter.addItem("All Categories", None)
        self._filter_subject = None
        self._filter_category = None

    def _refresh(self) -> None:
        display_events = [
            event for event in self._events if self._passes_filters(event)
        ]
        self._display_events = display_events
        self._table.setRowCount(len(display_events))

        # Track starts to estimate durations when matching end events arrive.
        open_starts: dict[str, BehaviorEvent] = {}

        for row, event in enumerate(display_events):
            self._table.setItem(row, 0, self._make_item(str(row + 1)))
            self._table.setItem(row, 1, self._make_item(event.behavior))
            self._table.setItem(row, 2, self._make_item(event.event.title()))
            subject_text = self._normalize_subject(event.subject)
            self._table.setItem(row, 3, self._make_item(subject_text))
            modifiers_text = ", ".join(event.modifiers) if event.modifiers else "—"
            self._table.setItem(row, 4, self._make_item(modifiers_text))
            category_text = self._normalize_category(event.category)
            self._table.setItem(row, 5, self._make_item(category_text))

            time_seconds = self._resolve_time_seconds(event)
            self._table.setItem(row, 6, self._make_item(_format_seconds(time_seconds)))
            self._table.setItem(row, 7, self._make_item(str(event.frame)))

            duration_text = "—"
            if event.event == "start":
                open_starts[event.behavior] = event
            else:
                start_event = open_starts.pop(event.behavior, None)
                if start_event is not None:
                    start_time = self._resolve_time_seconds(start_event)
                    if start_time is not None and time_seconds is not None:
                        duration_text = f"{time_seconds - start_time:.2f}"
            self._table.setItem(row, 8, self._make_item(duration_text))
            status = "Confirmed" if getattr(event, "confirmed", True) else "Auto"
            self._table.setItem(row, 9, self._make_item(status))

            # Apply a subtle color cue for start/end events to improve scanning.
            color = (
                QtGui.QColor("#2E7D32")
                if event.event == "start"
                else QtGui.QColor("#C62828")
            )
            for column in range(self._table.columnCount()):
                item = self._table.item(row, column)
                if item is not None:
                    item.setForeground(QtGui.QBrush(color))

            behavior_color = self._color_for_behavior(event.behavior)
            if behavior_color is not None:
                behavior_color.setAlpha(50)
                for column in range(self._table.columnCount()):
                    item = self._table.item(row, column)
                    if item is not None:
                        item.setBackground(QtGui.QBrush(behavior_color))

        self._undo_button.setEnabled(bool(self._events))
        self._clear_button.setEnabled(bool(self._events))

    def _resolve_time_seconds(self, event: BehaviorEvent) -> Optional[float]:
        if event.timestamp is not None:
            return float(event.timestamp)
        if self._fps and self._fps > 0:
            return event.frame / self._fps
        return None

    def _passes_filters(self, event: BehaviorEvent) -> bool:
        subject = self._normalize_subject(event.subject)
        category = self._normalize_category(event.category)
        if self._filter_subject and subject != self._filter_subject:
            return False
        if self._filter_category and category != self._filter_category:
            return False
        return True

    def set_color_getter(self, color_getter) -> None:
        self._color_getter = color_getter
        self._refresh()

    def _color_for_behavior(self, behavior: str) -> Optional[QtGui.QColor]:
        if not behavior or self._color_getter is None:
            return None
        try:
            rgb = self._color_getter(behavior)
        except Exception:
            return None
        if rgb is None:
            return None
        try:
            r, g, b = rgb
        except Exception:
            return None
        return QtGui.QColor(int(r), int(g), int(b))

    def _update_filter_options(self) -> None:
        subjects = sorted(
            {self._normalize_subject(evt.subject) for evt in self._events}
        )
        categories = sorted(
            {self._normalize_category(evt.category) for evt in self._events}
        )
        self._populate_filter_combo(
            self._subject_filter, subjects, self._filter_subject, "All Subjects"
        )
        self._populate_filter_combo(
            self._category_filter, categories, self._filter_category, "All Categories"
        )

    def _populate_filter_combo(
        self,
        combo: QtWidgets.QComboBox,
        values: List[str],
        current: Optional[str],
        all_label: str,
    ) -> None:
        with QtCore.QSignalBlocker(combo):
            combo.clear()
            combo.addItem(all_label, None)
            for value in values:
                combo.addItem(value, value)
            if current:
                index = combo.findData(current)
                if index >= 0:
                    combo.setCurrentIndex(index)
                else:
                    current = None
        if combo is self._subject_filter:
            self._filter_subject = current
        elif combo is self._category_filter:
            self._filter_category = current

    def _on_subject_filter_changed(self, index: int) -> None:
        self._filter_subject = self._subject_filter.currentData()
        self._refresh()

    def _on_category_filter_changed(self, index: int) -> None:
        self._filter_category = self._category_filter.currentData()
        self._refresh()

    @staticmethod
    def _normalize_subject(subject: Optional[str]) -> str:
        subject = (subject or "").strip()
        return subject if subject else "—"

    @staticmethod
    def _normalize_category(category: Optional[str]) -> str:
        category = (category or "").strip()
        return category if category else "—"

    @staticmethod
    def _make_item(text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
        return item

    def _handle_double_click(self, index: QtCore.QModelIndex) -> None:
        row = index.row()
        if 0 <= row < len(self._display_events):
            event = self._display_events[row]
            self.jumpToFrame.emit(event.frame)
            self.behaviorSelected.emit(event)

    def current_event(self) -> Optional[BehaviorEvent]:
        row = self._table.currentRow()
        if row < 0 or row >= len(self._display_events):
            return None
        return self._display_events[row]

    def _confirm_clear(self) -> None:
        if not self._events:
            return
        reply = QtWidgets.QMessageBox.question(
            self,
            "Clear Behavior Events",
            "Remove all recorded behavior events?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply == QtWidgets.QMessageBox.Yes:
            self.clearRequested.emit()

    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt override
        current = self.current_event()
        if current is None:
            return
        menu = QtWidgets.QMenu(self)
        menu.addAction("Jump to Frame", lambda: self.jumpToFrame.emit(current.frame))
        menu.addAction("Edit Interval…", lambda: self.editRequested.emit(current))
        menu.addAction("Delete Event", lambda: self.deleteRequested.emit(current))
        if not getattr(current, "confirmed", True):
            menu.addAction("Confirm Event", lambda: self.confirmRequested.emit(current))
        else:
            menu.addAction(
                "Mark Unconfirmed", lambda: self.rejectRequested.emit(current)
            )
        menu.exec_(event.globalPos())
