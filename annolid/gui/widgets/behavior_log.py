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
    undoRequested = QtCore.Signal()
    clearRequested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._events: List[BehaviorEvent] = []
        self._fps: Optional[float] = None

        self._table = QtWidgets.QTableWidget(0, 6)
        self._table.setObjectName("behaviorEventTable")
        self._table.setHorizontalHeaderLabels(
            ["#", "Behavior", "Event", "Time", "Frame", "Duration"]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.doubleClicked.connect(self._handle_double_click)

        header = self._table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)

        self._undo_button = QtWidgets.QPushButton("Undo Last")
        self._undo_button.setToolTip(
            "Remove the most recently added behavior event")
        self._undo_button.clicked.connect(self.undoRequested.emit)

        self._clear_button = QtWidgets.QPushButton("Clear")
        self._clear_button.setToolTip("Remove all recorded behavior events")
        self._clear_button.clicked.connect(self._confirm_clear)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self._undo_button)
        button_layout.addWidget(self._clear_button)
        button_layout.addStretch(1)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._table)
        layout.addLayout(button_layout)

        self._shortcuts = [
            QtWidgets.QShortcut(QKeySequence("Ctrl+Z"),
                                self, self.undoRequested.emit),
            QtWidgets.QShortcut(QKeySequence("Ctrl+Shift+Z"),
                                self, self.undoRequested.emit),
            QtWidgets.QShortcut(QKeySequence(
                "Ctrl+Backspace"), self, self._confirm_clear),
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
        self._events.sort(key=lambda evt: (
            evt.frame, 0 if evt.event == "start" else 1))
        self._refresh()

    def remove_event(self, key: tuple[int, str, str]) -> None:
        self._events = [evt for evt in self._events if evt.mark_key != key]
        self._refresh()

    def clear(self) -> None:
        self._events.clear()
        self._table.setRowCount(0)
        self._undo_button.setEnabled(False)
        self._clear_button.setEnabled(False)

    def _refresh(self) -> None:
        self._table.setRowCount(len(self._events))

        # Track starts to estimate durations when matching end events arrive.
        open_starts: dict[str, BehaviorEvent] = {}

        for row, event in enumerate(self._events):
            self._table.setItem(row, 0, self._make_item(str(row + 1)))
            self._table.setItem(row, 1, self._make_item(event.behavior))
            self._table.setItem(row, 2, self._make_item(event.event.title()))

            time_seconds = self._resolve_time_seconds(event)
            self._table.setItem(row, 3, self._make_item(
                _format_seconds(time_seconds)))
            self._table.setItem(row, 4, self._make_item(str(event.frame)))

            duration_text = "—"
            if event.event == "start":
                open_starts[event.behavior] = event
            else:
                start_event = open_starts.pop(event.behavior, None)
                if start_event is not None:
                    start_time = self._resolve_time_seconds(start_event)
                    if start_time is not None and time_seconds is not None:
                        duration_text = f"{time_seconds - start_time:.2f}"
            self._table.setItem(row, 5, self._make_item(duration_text))

            # Apply a subtle color cue for start/end events to improve scanning.
            color = QtGui.QColor(
                "#2E7D32") if event.event == "start" else QtGui.QColor("#C62828")
            for column in range(self._table.columnCount()):
                item = self._table.item(row, column)
                if item is not None:
                    item.setForeground(QtGui.QBrush(color))

        self._undo_button.setEnabled(bool(self._events))
        self._clear_button.setEnabled(bool(self._events))

    def _resolve_time_seconds(self, event: BehaviorEvent) -> Optional[float]:
        if event.timestamp is not None:
            return float(event.timestamp)
        if self._fps and self._fps > 0:
            return event.frame / self._fps
        return None

    @staticmethod
    def _make_item(text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        item.setFlags(item.flags() ^ QtCore.Qt.ItemIsEditable)
        return item

    def _handle_double_click(self, index: QtCore.QModelIndex) -> None:
        row = index.row()
        if 0 <= row < len(self._events):
            frame = self._events[row].frame
            self.jumpToFrame.emit(frame)

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
