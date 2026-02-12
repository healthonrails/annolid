from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from qtpy import QtCore, QtWidgets

from annolid.core.agent.session_manager import (
    AgentSessionManager,
    PersistentSessionStore,
)


class ChatSessionManagerDialog(QtWidgets.QDialog):
    """Dialog for browsing and managing persisted AI chat sessions."""

    sessionSwitched = QtCore.Signal(str)
    sessionCreated = QtCore.Signal(str)
    sessionCleared = QtCore.Signal(str)
    sessionDeleted = QtCore.Signal(str)

    def __init__(
        self,
        *,
        session_manager: AgentSessionManager,
        session_store: PersistentSessionStore,
        active_session_id: str,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._session_manager = session_manager
        self._session_store = session_store
        self._active_session_id = str(active_session_id or "").strip()
        self._rows_cache: List[Dict[str, object]] = []
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        self.setWindowTitle("Session Manager")
        self.resize(920, 500)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        hint = QtWidgets.QLabel(
            "Gateway-style session control: inspect, switch, create, clear, or delete chat sessions.",
            self,
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        top_row = QtWidgets.QHBoxLayout()
        self.active_label = QtWidgets.QLabel(self)
        top_row.addWidget(self.active_label, 1)
        self.filter_edit = QtWidgets.QLineEdit(self)
        self.filter_edit.setPlaceholderText("Filter by session id…")
        top_row.addWidget(self.filter_edit, 1)
        root.addLayout(top_row)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal, self)
        root.addWidget(splitter, 1)

        left = QtWidgets.QWidget(splitter)
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        self.table = QtWidgets.QTableWidget(left)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(
            ["Session", "Updated", "Messages", "Facts"]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )
        left_layout.addWidget(self.table, 1)

        right = QtWidgets.QWidget(splitter)
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(6)

        self.details = QtWidgets.QPlainTextEdit(right)
        self.details.setReadOnly(True)
        self.details.setPlaceholderText("Select a session to preview details…")
        right_layout.addWidget(self.details, 1)

        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        button_row = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh", self)
        self.switch_btn = QtWidgets.QPushButton("Switch", self)
        self.new_btn = QtWidgets.QPushButton("New Session", self)
        self.clear_btn = QtWidgets.QPushButton("Clear Messages", self)
        self.delete_btn = QtWidgets.QPushButton("Delete Session", self)
        self.close_btn = QtWidgets.QPushButton("Close", self)
        button_row.addWidget(self.refresh_btn)
        button_row.addWidget(self.switch_btn)
        button_row.addWidget(self.new_btn)
        button_row.addWidget(self.clear_btn)
        button_row.addWidget(self.delete_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.close_btn)
        root.addLayout(button_row)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.table.itemSelectionChanged.connect(self._update_details)
        self.table.itemDoubleClicked.connect(lambda _item: self._switch_selected())
        self.refresh_btn.clicked.connect(self.refresh)
        self.switch_btn.clicked.connect(self._switch_selected)
        self.new_btn.clicked.connect(self._create_session)
        self.clear_btn.clicked.connect(self._clear_selected)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.close_btn.clicked.connect(self.accept)

    def refresh(self) -> None:
        self._rows_cache = list(self._session_manager.list_sessions())
        self.active_label.setText(f"Active session: {self._active_session_id or '-'}")
        self._apply_filter()

    def _apply_filter(self) -> None:
        pattern = str(self.filter_edit.text() or "").strip().lower()
        if not pattern:
            rows = self._rows_cache
        else:
            rows = [
                row
                for row in self._rows_cache
                if pattern in str(row.get("key") or "").lower()
            ]
        self._fill_table(rows)
        self._update_details()

    def _fill_table(self, rows: List[Dict[str, object]]) -> None:
        self.table.setRowCount(len(rows))
        select_row = -1
        for idx, row in enumerate(rows):
            key = str(row.get("key") or "")
            updated = str(row.get("updated_at") or "")
            msg_count = int(row.get("message_count") or 0)
            overview = self._session_manager.get_session_overview(key)
            fact_count = int(overview.get("fact_count") or 0)

            key_item = QtWidgets.QTableWidgetItem(key)
            key_item.setData(QtCore.Qt.UserRole, key)
            self.table.setItem(idx, 0, key_item)
            self.table.setItem(idx, 1, QtWidgets.QTableWidgetItem(updated))
            self.table.setItem(idx, 2, QtWidgets.QTableWidgetItem(str(msg_count)))
            self.table.setItem(idx, 3, QtWidgets.QTableWidgetItem(str(fact_count)))
            if key == self._active_session_id:
                select_row = idx
        if select_row >= 0:
            self.table.selectRow(select_row)
        elif rows:
            self.table.selectRow(0)

    def _selected_key(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            return ""
        item = self.table.item(row, 0)
        if item is None:
            return ""
        return str(item.data(QtCore.Qt.UserRole) or "").strip()

    def _update_details(self) -> None:
        key = self._selected_key()
        if not key:
            self.details.setPlainText("")
            return
        try:
            overview = self._session_manager.get_session_overview(key)
        except Exception as exc:
            self.details.setPlainText(f"Failed to load session details: {exc}")
            return
        facts = overview.get("facts") or {}
        metadata = overview.get("metadata") or {}
        facts_lines = "\n".join(f"- {k}: {v}" for k, v in dict(facts).items()) or "-"
        meta_lines = "\n".join(f"- {k}: {v}" for k, v in dict(metadata).items()) or "-"
        self.details.setPlainText(
            "\n".join(
                [
                    f"Session: {overview.get('key')}",
                    f"Path: {overview.get('path')}",
                    f"Created: {overview.get('created_at')}",
                    f"Updated: {overview.get('updated_at')}",
                    f"Messages: {overview.get('message_count')}",
                    f"Facts: {overview.get('fact_count')}",
                    "",
                    "Facts",
                    facts_lines,
                    "",
                    "Metadata",
                    meta_lines,
                ]
            )
        )

    def _new_session_id_suggestion(self) -> str:
        return f"gui:annolid_bot:{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    def _switch_selected(self) -> None:
        key = self._selected_key()
        if not key:
            return
        self._active_session_id = key
        self.active_label.setText(f"Active session: {self._active_session_id}")
        self.sessionSwitched.emit(key)
        self._apply_filter()

    def _create_session(self) -> None:
        value, ok = QtWidgets.QInputDialog.getText(
            self,
            "New Session",
            "Session ID",
            text=self._new_session_id_suggestion(),
        )
        if not ok:
            return
        key = str(value or "").strip()
        if not key:
            return
        session = self._session_manager.get_or_create(key)
        self._session_manager.save(session)
        self._active_session_id = key
        self.sessionCreated.emit(key)
        self.refresh()

    def _clear_selected(self) -> None:
        key = self._selected_key()
        if not key:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Clear Session Messages",
            f"Clear messages and facts for session '{key}'?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self._session_store.clear_history(key)
        self._session_store.clear_facts(key)
        self.sessionCleared.emit(key)
        self.refresh()

    def _delete_selected(self) -> None:
        key = self._selected_key()
        if not key:
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Delete Session",
            f"Delete session '{key}' from disk?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self._session_store.clear_session(key)
        self.sessionDeleted.emit(key)
        if key == self._active_session_id:
            self._active_session_id = ""
        self.refresh()
