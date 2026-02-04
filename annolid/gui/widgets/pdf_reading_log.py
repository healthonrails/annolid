from __future__ import annotations

import datetime as _dt
import time
from typing import Dict, Optional

from qtpy import QtCore, QtWidgets


def _format_timestamp(ts: object) -> str:
    try:
        value = float(ts)  # seconds
        if value <= 0:
            raise ValueError
        dt = _dt.datetime.fromtimestamp(value)
        return dt.strftime("%H:%M:%S")
    except Exception:
        return "--:--:--"


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


class PdfReadingLogWidget(QtWidgets.QWidget):
    """Dock widget body: shows reading events and allows jump-to-location."""

    entry_activated = QtCore.Signal(dict)
    clear_requested = QtCore.Signal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._last_activation_id = ""
        self._last_activation_time = 0.0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QtWidgets.QHBoxLayout()
        header.setSpacing(6)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setPlaceholderText("Search reading log…")
        self.search_edit.textChanged.connect(self._apply_filter)
        self.clear_button = QtWidgets.QToolButton(self)
        self.clear_button.setText("Clear")
        self.clear_button.setToolTip("Clear this PDF's reading log")
        self.clear_button.clicked.connect(self.clear_requested.emit)
        header.addWidget(self.search_edit, 1)
        header.addWidget(self.clear_button)
        layout.addLayout(header)

        self.list_widget = QtWidgets.QListWidget(self)
        self.list_widget.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.itemActivated.connect(self._on_item_activated)
        self.list_widget.itemClicked.connect(self._on_item_clicked)
        layout.addWidget(self.list_widget, 1)

    def clear(self) -> None:
        self.list_widget.clear()

    def set_entries(self, entries: list[dict[str, object]]) -> None:
        self.list_widget.clear()
        for entry in entries:
            self.add_entry(entry, apply_filter=False)
        self._apply_filter()

    def add_entry(self, entry: Dict[str, object], *, apply_filter: bool = True) -> None:
        ts = _format_timestamp(entry.get("ts"))
        event_type = str(entry.get("type") or "event")
        page_num = _safe_int(entry.get("pageNum"), 0)
        label = str(entry.get("label") or "").strip()
        if not label:
            label = event_type
        suffix = f" · p{page_num}" if page_num > 0 else ""
        text = f"[{ts}] {label}{suffix}"

        item = QtWidgets.QListWidgetItem(text)
        item.setData(QtCore.Qt.UserRole, dict(entry))
        item.setToolTip(text)
        # Minimal iconography.
        icon = None
        if event_type in {"bookmark_add", "bookmark_remove"}:
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogYesButton)
        elif event_type.startswith("note_"):
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_FileDialogDetailedView)
        elif event_type.startswith("mark_"):
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_DialogApplyButton)
        elif event_type in {"open", "resume"}:
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_ArrowRight)
        elif event_type in {"stop"}:
            icon = self.style().standardIcon(QtWidgets.QStyle.SP_BrowserStop)
        if icon is not None:
            item.setIcon(icon)
        self.list_widget.insertItem(0, item)
        if apply_filter:
            self._apply_filter()

    def _apply_filter(self) -> None:
        query = (self.search_edit.text() or "").strip().lower()
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            payload = item.data(QtCore.Qt.UserRole) or {}
            label = str(payload.get("label") or "")
            event_type = str(payload.get("type") or "")
            snippet = str(payload.get("snippet") or payload.get("text") or "")
            hay = " ".join([label, event_type, snippet, item.text()]).lower()
            item.setHidden(bool(query) and query not in hay)

    def _on_item_activated(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return
        if not self._should_emit_activation(item):
            return
        payload = item.data(QtCore.Qt.UserRole)
        if isinstance(payload, dict):
            self.entry_activated.emit(payload)

    def _on_item_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is None:
            return
        if not self._should_emit_activation(item):
            return
        payload = item.data(QtCore.Qt.UserRole)
        if isinstance(payload, dict):
            self.entry_activated.emit(payload)

    def _should_emit_activation(self, item: QtWidgets.QListWidgetItem) -> bool:
        payload = item.data(QtCore.Qt.UserRole) or {}
        try:
            entry_id = str(payload.get("id") or item.text())
        except Exception:
            entry_id = item.text()
        now = time.monotonic()
        # Avoid double-trigger when double-click causes both clicked + activated.
        if (
            entry_id == self._last_activation_id
            and (now - self._last_activation_time) < 0.35
        ):
            return False
        self._last_activation_id = entry_id
        self._last_activation_time = now
        return True
