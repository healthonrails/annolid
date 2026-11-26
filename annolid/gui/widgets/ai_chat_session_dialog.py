from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Sequence

from qtpy import QtCore, QtWidgets

from annolid.services.chat_session import (
    AgentSessionManager,
    PersistentSessionStore,
)
from annolid.utils.atomic_io import atomic_write_text

_PREVIEW_MESSAGE_LIMIT = 8
_PREVIEW_MESSAGE_MAX_CHARS = 240


def format_session_markdown(
    overview: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
) -> str:
    """Format a persisted chat session into clean, portable Markdown."""
    key = str(overview.get("key") or "session").strip()
    lines = [
        f"# Annolid Bot Session: {key}",
        "",
        f"- **Created**: {overview.get('created_at') or '-'}",
        f"- **Updated**: {overview.get('updated_at') or '-'}",
        f"- **Messages**: {overview.get('message_count', len(messages))}",
        "",
    ]
    facts = dict(overview.get("facts") or {})
    if facts:
        lines.append("## Session Facts")
        lines.append("")
        for k, v in facts.items():
            lines.append(f"- **{k}**: {v}")
        lines.append("")

    lines.append("## Transcript")
    lines.append("")
    emitted = 0
    for msg in messages:
        role = str(msg.get("role") or "").strip().lower()
        content = str(msg.get("content") or "").strip()
        if not content or role not in {"user", "assistant", "system"}:
            continue
        heading = {"user": "User", "assistant": "Annolid Bot", "system": "System"}.get(
            role, role.title()
        )
        lines.append(f"### {heading}")
        lines.append("")
        lines.append(content)
        lines.append("")
        emitted += 1
    if emitted == 0:
        lines.append("*(No messages recorded in this session.)*")
        lines.append("")
    return "\n".join(lines)


class ChatSessionManagerDialog(QtWidgets.QDialog):
    """Dialog for browsing, searching, exporting, and managing persisted AI chat sessions."""

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
        self._overview_cache: Dict[str, Dict[str, Any]] = {}
        self._history_cache: Dict[str, List[Dict[str, Any]]] = {}
        self._search_blob_cache: Dict[str, str] = {}
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        self.setWindowTitle("Session Manager")
        self.resize(940, 520)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)

        hint = QtWidgets.QLabel(
            "Inspect, search, switch, create, export, clear, or delete persisted Annolid Bot sessions.",
            self,
        )
        hint.setWordWrap(True)
        root.addWidget(hint)

        top_row = QtWidgets.QHBoxLayout()
        self.active_label = QtWidgets.QLabel(self)
        top_row.addWidget(self.active_label, 1)
        self.filter_edit = QtWidgets.QLineEdit(self)
        self.filter_edit.setClearButtonEnabled(True)
        self.filter_edit.setPlaceholderText(
            "Filter by session id, facts, or recent messages…"
        )
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
        self.export_btn = QtWidgets.QPushButton("Export Markdown…", self)
        self.clear_btn = QtWidgets.QPushButton("Clear Messages", self)
        self.delete_btn = QtWidgets.QPushButton("Delete Session", self)
        self.close_btn = QtWidgets.QPushButton("Close", self)
        button_row.addWidget(self.refresh_btn)
        button_row.addWidget(self.switch_btn)
        button_row.addWidget(self.new_btn)
        button_row.addWidget(self.export_btn)
        button_row.addWidget(self.clear_btn)
        button_row.addWidget(self.delete_btn)
        button_row.addStretch(1)
        button_row.addWidget(self.close_btn)
        root.addLayout(button_row)

        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setEnabled(False)
        root.addWidget(self.status_label)

        self.filter_edit.textChanged.connect(self._apply_filter)
        self.table.itemSelectionChanged.connect(self._update_details)
        self.table.itemDoubleClicked.connect(lambda _item: self._switch_selected())
        self.refresh_btn.clicked.connect(self.refresh)
        self.switch_btn.clicked.connect(self._switch_selected)
        self.new_btn.clicked.connect(self._create_session)
        self.export_btn.clicked.connect(self._export_selected_markdown)
        self.clear_btn.clicked.connect(self._clear_selected)
        self.delete_btn.clicked.connect(self._delete_selected)
        self.close_btn.clicked.connect(self.accept)

    def refresh(self) -> None:
        self._rows_cache = list(self._session_manager.list_sessions())
        self._overview_cache.clear()
        self._history_cache.clear()
        self._search_blob_cache.clear()
        for row in self._rows_cache:
            key = str(row.get("key") or "").strip()
            if not key:
                continue
            try:
                overview = dict(self._session_manager.get_session_overview(key) or {})
            except Exception:
                overview = {"key": key, "fact_count": 0}
            self._overview_cache[key] = overview
            try:
                history = list(self._session_store.get_history(key) or [])
            except Exception:
                history = []
            self._history_cache[key] = history
            fact_text = " ".join(
                f"{k} {v}" for k, v in dict(overview.get("facts") or {}).items()
            )
            recent_text = " ".join(
                str(msg.get("content") or "")[:160]
                for msg in history[-_PREVIEW_MESSAGE_LIMIT:]
                if str(msg.get("role") or "").strip().lower()
                in {"user", "assistant", "system"}
            )
            self._search_blob_cache[key] = f"{key}\n{fact_text}\n{recent_text}".lower()

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
                if pattern
                in self._search_blob_cache.get(
                    str(row.get("key") or "").strip(),
                    str(row.get("key") or "").lower(),
                )
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
            overview = self._overview_cache.get(key) or {}
            fact_count = int(overview.get("fact_count") or 0)

            display_key = f"{key} (active)" if key == self._active_session_id else key
            key_item = QtWidgets.QTableWidgetItem(display_key)
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
        self._sync_selection_buttons()

    def _selected_key(self) -> str:
        row = self.table.currentRow()
        if row < 0:
            return ""
        item = self.table.item(row, 0)
        if item is None:
            return ""
        return str(item.data(QtCore.Qt.UserRole) or "").strip()

    def _sync_selection_buttons(self) -> None:
        has_selection = bool(self._selected_key())
        for btn in (
            self.switch_btn,
            self.export_btn,
            self.clear_btn,
            self.delete_btn,
        ):
            btn.setEnabled(has_selection)

    def _update_details(self) -> None:
        self._sync_selection_buttons()
        key = self._selected_key()
        if not key:
            self.details.setPlainText("")
            return
        overview = self._overview_cache.get(key)
        if overview is None:
            try:
                overview = dict(self._session_manager.get_session_overview(key) or {})
            except Exception as exc:
                self.details.setPlainText(f"Failed to load session details: {exc}")
                return
            self._overview_cache[key] = overview
        facts = overview.get("facts") or {}
        metadata = overview.get("metadata") or {}
        facts_lines = "\n".join(f"- {k}: {v}" for k, v in dict(facts).items()) or "-"
        meta_lines = "\n".join(f"- {k}: {v}" for k, v in dict(metadata).items()) or "-"

        history = self._history_cache.get(key)
        if history is None:
            try:
                history = list(self._session_store.get_history(key) or [])
            except Exception:
                history = []
            self._history_cache[key] = history

        visible_turns = [
            msg
            for msg in history
            if str(msg.get("role") or "").strip().lower()
            in {"user", "assistant", "system"}
            and str(msg.get("content") or "").strip()
        ]
        preview_lines: List[str] = []
        for msg in visible_turns[-_PREVIEW_MESSAGE_LIMIT:]:
            role = str(msg.get("role") or "").strip().lower()
            tag = {"user": "You", "assistant": "Bot", "system": "System"}.get(
                role, role.title()
            )
            snippet = " ".join(str(msg.get("content") or "").split())
            if len(snippet) > _PREVIEW_MESSAGE_MAX_CHARS:
                snippet = snippet[:_PREVIEW_MESSAGE_MAX_CHARS].rstrip() + "…"
            preview_lines.append(f"[{tag}] {snippet}")
        transcript_preview = "\n".join(preview_lines) if preview_lines else "-"

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
                    "Recent Messages",
                    transcript_preview,
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

    def _export_selected_markdown(self) -> None:
        key = self._selected_key()
        if not key:
            return
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", key).strip("_") or "session"
        default_path = str(Path.home() / f"{safe_stem}.md")
        target_path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export Session Transcript",
            default_path,
            "Markdown Files (*.md);;Text Files (*.txt);;All Files (*)",
        )
        if not target_path:
            return
        overview = self._overview_cache.get(key) or {"key": key}
        history = self._history_cache.get(key)
        if history is None:
            try:
                history = list(self._session_store.get_history(key) or [])
            except Exception:
                history = []
        markdown = format_session_markdown(overview, history)
        try:
            atomic_write_text(Path(target_path), markdown)
            self.status_label.setText(f"Exported session to {Path(target_path).name}.")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Export Failed",
                f"Could not save transcript to {target_path}.\n\n{exc}",
            )

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
