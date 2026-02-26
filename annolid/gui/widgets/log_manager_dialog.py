from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

from qtpy import QtCore, QtGui, QtWidgets

from annolid.datasets.labelme_collection import default_label_index_path
from annolid.utils.log_paths import (
    resolve_annolid_logs_root,
    resolve_annolid_realtime_logs_root,
)
from annolid.utils.runs import shared_runs_root


@dataclass(frozen=True)
class _LogEntry:
    key: str
    name: str
    path_fn: Callable[[], Path]
    removable: bool = True


class LogManagerDialog(QtWidgets.QDialog):
    """Standalone logs manager for user and bot workflows."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Log Manager")
        self.resize(860, 420)
        self._entries: List[_LogEntry] = [
            _LogEntry("logs_root", "Logs Root", resolve_annolid_logs_root),
            _LogEntry(
                "realtime",
                "Realtime Logs",
                resolve_annolid_realtime_logs_root,
            ),
            _LogEntry(
                "label_index",
                "Label Index Logs",
                lambda: default_label_index_path().parent,
            ),
            _LogEntry("runs", "Runs Logs", shared_runs_root),
            _LogEntry("app", "App Logs", lambda: resolve_annolid_logs_root() / "app"),
        ]
        self._build_ui()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        info = QtWidgets.QLabel(
            "Manage Annolid log locations independently from dataset tools.\n"
            "Default layout: ~/.annolid/logs/{app,realtime,runs,label_index}"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        self.table = QtWidgets.QTableWidget(0, 4, self)
        self.table.setHorizontalHeaderLabels(["Name", "Path", "Exists", "Items"])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        self.table.horizontalHeader().setSectionResizeMode(
            2, QtWidgets.QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            3, QtWidgets.QHeaderView.ResizeToContents
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.refresh_btn = QtWidgets.QPushButton("Refresh")
        self.open_btn = QtWidgets.QPushButton("Open")
        self.copy_btn = QtWidgets.QPushButton("Copy Path")
        self.remove_btn = QtWidgets.QPushButton("Remove Folder")
        self.copy_json_btn = QtWidgets.QPushButton("Copy JSON Snapshot")
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.copy_btn)
        btn_row.addWidget(self.remove_btn)
        btn_row.addStretch(1)
        btn_row.addWidget(self.copy_json_btn)
        layout.addLayout(btn_row)

        self.status = QtWidgets.QLabel("")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        close_row = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close, self)
        close_row.rejected.connect(self.reject)
        layout.addWidget(close_row)

        self.refresh_btn.clicked.connect(self._refresh)
        self.open_btn.clicked.connect(self._open_selected)
        self.copy_btn.clicked.connect(self._copy_selected_path)
        self.remove_btn.clicked.connect(self._remove_selected)
        self.copy_json_btn.clicked.connect(self._copy_json_snapshot)
        self.table.itemSelectionChanged.connect(self._sync_buttons)

    def _resolve_entries(self) -> List[Dict[str, object]]:
        payload: List[Dict[str, object]] = []
        for entry in self._entries:
            path = Path(entry.path_fn()).expanduser().resolve()
            exists = path.exists()
            try:
                count = len(list(path.iterdir())) if exists and path.is_dir() else 0
            except Exception:
                count = -1
            payload.append(
                {
                    "key": entry.key,
                    "name": entry.name,
                    "path": str(path),
                    "exists": bool(exists),
                    "item_count": int(count),
                    "removable": bool(entry.removable),
                }
            )
        return payload

    def _refresh(self) -> None:
        rows = self._resolve_entries()
        self.table.setRowCount(len(rows))
        for r, row in enumerate(rows):
            name_item = QtWidgets.QTableWidgetItem(str(row["name"]))
            path_item = QtWidgets.QTableWidgetItem(str(row["path"]))
            exists_item = QtWidgets.QTableWidgetItem("Yes" if row["exists"] else "No")
            count = int(row.get("item_count", 0))
            count_text = str(count) if count >= 0 else "?"
            count_item = QtWidgets.QTableWidgetItem(count_text)
            path_item.setData(QtCore.Qt.UserRole, row)
            self.table.setItem(r, 0, name_item)
            self.table.setItem(r, 1, path_item)
            self.table.setItem(r, 2, exists_item)
            self.table.setItem(r, 3, count_item)
        if rows and self.table.currentRow() < 0:
            self.table.selectRow(0)
        self._sync_buttons()

    def _selected_row_payload(self) -> Dict[str, object] | None:
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 1)
        if item is None:
            return None
        payload = item.data(QtCore.Qt.UserRole)
        return payload if isinstance(payload, dict) else None

    def _sync_buttons(self) -> None:
        payload = self._selected_row_payload()
        enabled = payload is not None
        self.open_btn.setEnabled(enabled)
        self.copy_btn.setEnabled(enabled)
        self.remove_btn.setEnabled(enabled and bool(payload.get("removable")))

    def _open_selected(self) -> None:
        payload = self._selected_row_payload()
        if not payload:
            return
        folder = Path(str(payload["path"]))
        folder.mkdir(parents=True, exist_ok=True)
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _copy_selected_path(self) -> None:
        payload = self._selected_row_payload()
        if not payload:
            return
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        app.clipboard().setText(str(payload["path"]))
        self.status.setText(f"Copied path: {payload['path']}")

    def _remove_selected(self) -> None:
        payload = self._selected_row_payload()
        if not payload:
            return
        folder = Path(str(payload["path"]))
        if not folder.exists():
            QtWidgets.QMessageBox.information(
                self, "Folder not found", f"Folder does not exist:\n{folder}"
            )
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            f"Remove {payload['name']}?",
            f"Delete this folder and all contents?\n{folder}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        try:
            shutil.rmtree(folder)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self, "Delete failed", f"Failed to remove folder:\n{exc}"
            )
            return
        self.status.setText(f"Removed: {folder}")
        self._refresh()

    def _copy_json_snapshot(self) -> None:
        payload = {"logs": self._resolve_entries()}
        text = json.dumps(payload, indent=2, sort_keys=True)
        app = QtWidgets.QApplication.instance()
        if app is None:
            return
        app.clipboard().setText(text)
        self.status.setText("Copied logs JSON snapshot to clipboard.")
