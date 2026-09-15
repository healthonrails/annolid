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
        self._pairing_issues: dict[int, str] = {}

        self._search = QtWidgets.QLineEdit()
        self._search.setPlaceholderText(
            "Search behavior, subject, modifier or category…"
        )
        self._search.setClearButtonEnabled(True)
        self._search.setAccessibleName("Search behavior events")
        self._search.textChanged.connect(self._refresh)

        self._review_filter = QtWidgets.QComboBox()
        self._review_filter.addItem("All events", "all")
        self._review_filter.addItem("Unconfirmed", "unconfirmed")
        self._review_filter.addItem("Pairing issues", "issues")
        self._review_filter.setToolTip(
            "Find proposals to review or unmatched start/end events"
        )
        self._review_filter.currentIndexChanged.connect(self._refresh)

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
                "Duration (s)",
                "Status",
            ]
        )
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self._table.setAlternatingRowColors(True)
        self._table.doubleClicked.connect(self._handle_double_click)
        self._table.itemSelectionChanged.connect(self._update_actions)

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

        self._jump_button = QtWidgets.QPushButton("Jump to frame")
        self._jump_button.clicked.connect(self._jump_to_selected)
        self._edit_button = QtWidgets.QPushButton("Edit interval…")
        self._edit_button.clicked.connect(self._edit_selected)
        self._review_button = QtWidgets.QPushButton("Confirm interval")
        self._review_button.clicked.connect(self._review_selected)
        action_layout = QtWidgets.QHBoxLayout()
        for button in (self._jump_button, self._edit_button, self._review_button):
            action_layout.addWidget(button)
        self._summary = QtWidgets.QLabel()
        self._summary.setWordWrap(True)
        self._hint = QtWidgets.QLabel(
            "Select an event to review it; double-click to jump to its frame."
        )
        self._hint.setWordWrap(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._search)
        layout.addLayout(filter_layout)
        layout.addWidget(self._review_filter)
        layout.addWidget(self._summary)
        layout.addWidget(self._table)
        layout.addWidget(self._hint)
        layout.addLayout(action_layout)
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
        for shortcut in self._shortcuts:
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        self._refresh()

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
        with QtCore.QSignalBlocker(self._search):
            self._search.clear()
        with QtCore.QSignalBlocker(self._review_filter):
            self._review_filter.setCurrentIndex(0)
        self._fps = None
        self._refresh()

    def _refresh(self) -> None:
        selected = self.current_event()
        durations, self._pairing_issues = self._analyze_pairs()
        display_events = [
            event for event in self._events if self._passes_filters(event)
        ]
        self._display_events = display_events
        self._table.clearSelection()
        self._table.setCurrentCell(-1, -1)
        self._table.setRowCount(len(display_events))

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

            duration_text = durations.get(id(event), "—")
            self._table.setItem(row, 8, self._make_item(duration_text))
            status = "Confirmed" if getattr(event, "confirmed", True) else "Unconfirmed"
            issue = self._pairing_issues.get(id(event))
            if issue:
                status += f" · {issue}"
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
            if selected is event or selected == event:
                self._table.selectRow(row)

        self._undo_button.setEnabled(bool(self._events))
        self._clear_button.setEnabled(bool(self._events))
        unconfirmed = sum(not event.confirmed for event in self._events)
        self._summary.setText(
            f"{len(display_events)} of {len(self._events)} events · "
            f"{unconfirmed} unconfirmed · {len(self._pairing_issues)} pairing issues"
        )
        self._hint.setText(
            "Select an event to review it; double-click to jump to its frame."
            if display_events
            else (
                "No events match these filters. Clear search or choose All events."
                if self._events
                else "No events yet. Select a behavior in Flags, then use S to start and E to end while viewing the video."
            )
        )
        self._update_actions()

    def _analyze_pairs(self) -> tuple[dict[int, str], dict[int, str]]:
        """Inspect all boundaries before filtering, without altering annotations."""
        durations: dict[int, str] = {}
        issues: dict[int, str] = {}
        pending: dict[tuple, list[BehaviorEvent]] = {}
        for event in self._events:
            key = (event.behavior, event.subject or "", tuple(sorted(event.modifiers)))
            if event.event == "start":
                pending.setdefault(key, []).append(event)
            elif event.event == "end":
                starts = pending.pop(key, [])
                if not starts:
                    issues[id(event)] = "Missing start"
                elif len(starts) > 1:
                    for boundary in [*starts, event]:
                        issues[id(boundary)] = "Ambiguous starts"
                else:
                    start_time = self._resolve_time_seconds(starts[0])
                    end_time = self._resolve_time_seconds(event)
                    if start_time is not None and end_time is not None:
                        if end_time < start_time:
                            for boundary in (starts[0], event):
                                issues[id(boundary)] = "Time goes backwards"
                        else:
                            durations[id(event)] = f"{end_time - start_time:.2f}"
        for starts in pending.values():
            for event in starts:
                issues[id(event)] = (
                    "Missing end" if len(starts) == 1 else "Ambiguous starts"
                )
        return durations, issues

    def _update_actions(self) -> None:
        event = self.current_event()
        for button in (self._jump_button, self._edit_button, self._review_button):
            button.setEnabled(event is not None)
        self._review_button.setText(
            "Mark interval unconfirmed"
            if event and event.confirmed
            else "Confirm interval"
        )

    def _jump_to_selected(self) -> None:
        event = self.current_event()
        if event is not None:
            self.jumpToFrame.emit(event.frame)

    def _edit_selected(self) -> None:
        event = self.current_event()
        if event is not None:
            self.editRequested.emit(event)

    def _review_selected(self) -> None:
        event = self.current_event()
        if event is not None:
            signal = self.rejectRequested if event.confirmed else self.confirmRequested
            signal.emit(event)

    def _resolve_time_seconds(self, event: BehaviorEvent) -> Optional[float]:
        if event.timestamp is not None:
            return float(event.timestamp)
        if self._fps and self._fps > 0:
            return event.frame / self._fps
        return None

    def _passes_filters(self, event: BehaviorEvent) -> bool:
        query = self._search.text().strip().casefold()
        searchable = " ".join(
            [
                event.behavior,
                event.subject or "",
                event.category or "",
                *event.modifiers,
            ]
        ).casefold()
        if query and query not in searchable:
            return False
        review = self._review_filter.currentData()
        if review == "unconfirmed" and event.confirmed:
            return False
        if review == "issues" and id(event) not in self._pairing_issues:
            return False
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
