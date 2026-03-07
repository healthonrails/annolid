from __future__ import annotations

import json
from typing import Optional

from qtpy import QtWidgets

from annolid.domain.memory.taxonomy import MEMORY_CATEGORIES, MEMORY_SOURCES
from annolid.interfaces.memory.registry import get_memory_service, get_retrieval_service


class _MemoryRecordEditorDialog(QtWidgets.QDialog):
    def __init__(
        self, parent=None, title: str = "Memory Record", seed: Optional[dict] = None
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(640, 460)
        self._seed = seed or {}

        layout = QtWidgets.QVBoxLayout(self)
        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.text_edit = QtWidgets.QPlainTextEdit(self)
        self.scope_edit = QtWidgets.QLineEdit(self)
        self.category_combo = QtWidgets.QComboBox(self)
        self.category_combo.addItems(list(MEMORY_CATEGORIES))
        self.source_combo = QtWidgets.QComboBox(self)
        self.source_combo.addItems(list(MEMORY_SOURCES))
        self.importance_spin = QtWidgets.QDoubleSpinBox(self)
        self.importance_spin.setRange(0.0, 1.0)
        self.importance_spin.setSingleStep(0.05)
        self.tags_edit = QtWidgets.QLineEdit(self)
        self.metadata_edit = QtWidgets.QPlainTextEdit(self)

        form.addRow("Text", self.text_edit)
        form.addRow("Scope", self.scope_edit)
        form.addRow("Category", self.category_combo)
        form.addRow("Source", self.source_combo)
        form.addRow("Importance", self.importance_spin)
        form.addRow("Tags (comma-separated)", self.tags_edit)
        form.addRow("Metadata (JSON)", self.metadata_edit)

        self.error_label = QtWidgets.QLabel(self)
        self.error_label.setStyleSheet("color: #c0392b;")
        layout.addWidget(self.error_label)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=self,
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._load_seed()

    def _load_seed(self) -> None:
        seed = self._seed
        self.text_edit.setPlainText(str(seed.get("text", "")))
        self.scope_edit.setText(str(seed.get("scope", "global")))
        self.importance_spin.setValue(float(seed.get("importance", 0.5)))
        self.tags_edit.setText(", ".join(list(seed.get("tags", []) or [])))
        metadata = seed.get("metadata", {})
        self.metadata_edit.setPlainText(json.dumps(metadata, indent=2, sort_keys=True))

        category = str(seed.get("category", "other"))
        idx = self.category_combo.findText(category)
        if idx >= 0:
            self.category_combo.setCurrentIndex(idx)
        source = str(seed.get("source", "system"))
        idx = self.source_combo.findText(source)
        if idx >= 0:
            self.source_combo.setCurrentIndex(idx)

    def _accept(self) -> None:
        self.error_label.clear()
        try:
            json.loads(self.metadata_edit.toPlainText() or "{}")
        except Exception as exc:
            self.error_label.setText(f"Invalid metadata JSON: {exc}")
            return
        if not self.text_edit.toPlainText().strip():
            self.error_label.setText("Text is required.")
            return
        if not self.scope_edit.text().strip():
            self.error_label.setText("Scope is required.")
            return
        self.accept()

    def payload(self) -> dict:
        metadata = json.loads(self.metadata_edit.toPlainText() or "{}")
        tags = [x.strip() for x in self.tags_edit.text().split(",") if x.strip()]
        return {
            "text": self.text_edit.toPlainText().strip(),
            "scope": self.scope_edit.text().strip(),
            "category": self.category_combo.currentText(),
            "source": self.source_combo.currentText(),
            "importance": float(self.importance_spin.value()),
            "tags": tags,
            "metadata": metadata,
        }


class MemoryManagerDialog(QtWidgets.QDialog):
    """Simple GUI CRUD for memory records."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Memory Manager")
        self.resize(980, 620)
        self._rows = []

        root = QtWidgets.QVBoxLayout(self)
        top = QtWidgets.QHBoxLayout()
        root.addLayout(top)
        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setPlaceholderText(
            "Search memories (leave blank to list scoped records)"
        )
        self.scope_edit = QtWidgets.QLineEdit(self)
        self.scope_edit.setPlaceholderText("Optional scope, e.g. workspace:default")
        self.search_btn = QtWidgets.QPushButton("Search", self)
        self.refresh_btn = QtWidgets.QPushButton("Refresh", self)
        top.addWidget(self.search_edit, 3)
        top.addWidget(self.scope_edit, 2)
        top.addWidget(self.search_btn)
        top.addWidget(self.refresh_btn)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Scope", "Category", "Source", "Importance", "Text"]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        root.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        root.addLayout(buttons)
        self.add_btn = QtWidgets.QPushButton("Add", self)
        self.edit_btn = QtWidgets.QPushButton("Edit", self)
        self.delete_btn = QtWidgets.QPushButton("Delete", self)
        self.close_btn = QtWidgets.QPushButton("Close", self)
        buttons.addWidget(self.add_btn)
        buttons.addWidget(self.edit_btn)
        buttons.addWidget(self.delete_btn)
        buttons.addStretch(1)
        buttons.addWidget(self.close_btn)

        self.status_label = QtWidgets.QLabel(self)
        root.addWidget(self.status_label)

        self.search_btn.clicked.connect(self.refresh_results)
        self.refresh_btn.clicked.connect(self.refresh_results)
        self.add_btn.clicked.connect(self.add_record)
        self.edit_btn.clicked.connect(self.edit_selected_record)
        self.delete_btn.clicked.connect(self.delete_selected_record)
        self.close_btn.clicked.connect(self.close)
        self.table.doubleClicked.connect(lambda _idx: self.edit_selected_record())
        self.refresh_results()

    def _set_status(self, message: str) -> None:
        self.status_label.setText(message)

    def _selected_row(self) -> Optional[dict]:
        row = self.table.currentRow()
        if row < 0 or row >= len(self._rows):
            return None
        return self._rows[row]

    def refresh_results(self) -> None:
        retrieval = get_retrieval_service()
        if not retrieval:
            self._rows = []
            self.table.setRowCount(0)
            self._set_status("Memory subsystem unavailable.")
            return
        query = self.search_edit.text().strip()
        scope = self.scope_edit.text().strip() or None
        try:
            hits = retrieval.search_memory(query=query, top_k=200, scope=scope)
        except Exception as exc:
            self._set_status(f"Failed to load memories: {exc}")
            return

        self._rows = [
            {
                "id": h.id,
                "text": h.text,
                "scope": h.scope,
                "category": h.category,
                "source": h.source,
                "importance": h.importance,
                "tags": h.tags,
                "metadata": h.metadata,
            }
            for h in hits
        ]
        self.table.setRowCount(len(self._rows))
        for r, row in enumerate(self._rows):
            self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(row["id"])))
            self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(row["scope"])))
            self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(str(row["category"])))
            self.table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(row["source"])))
            self.table.setItem(
                r, 4, QtWidgets.QTableWidgetItem(f"{float(row['importance']):.2f}")
            )
            self.table.setItem(r, 5, QtWidgets.QTableWidgetItem(str(row["text"])))
        self._set_status(f"Loaded {len(self._rows)} memories.")

    def add_record(self) -> None:
        service = get_memory_service()
        if not service:
            self._set_status("Memory service unavailable.")
            return
        dialog = _MemoryRecordEditorDialog(self, "Add Memory")
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        payload = dialog.payload()
        try:
            service.store_memory(**payload)
            self._set_status("Memory added.")
            self.refresh_results()
        except Exception as exc:
            self._set_status(f"Add failed: {exc}")

    def edit_selected_record(self) -> None:
        service = get_memory_service()
        if not service:
            self._set_status("Memory service unavailable.")
            return
        row = self._selected_row()
        if not row:
            self._set_status("Select a record first.")
            return
        dialog = _MemoryRecordEditorDialog(self, "Edit Memory", seed=row)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        payload = dialog.payload()
        patch = {
            "text": payload["text"],
            "scope": payload["scope"],
            "category": payload["category"],
            "source": payload["source"],
            "importance": payload["importance"],
            "tags": payload["tags"],
            "metadata": payload["metadata"],
        }
        ok = service.update_memory(row["id"], patch)
        if not ok:
            self._set_status("Update failed.")
            return
        self._set_status("Memory updated.")
        self.refresh_results()

    def delete_selected_record(self) -> None:
        service = get_memory_service()
        if not service:
            self._set_status("Memory service unavailable.")
            return
        row = self._selected_row()
        if not row:
            self._set_status("Select a record first.")
            return
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Delete Memory",
            f"Delete selected memory?\n\n{row['text'][:160]}",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if confirm != QtWidgets.QMessageBox.Yes:
            return
        ok = service.delete_memory(row["id"])
        if not ok:
            self._set_status("Delete failed.")
            return
        self._set_status("Memory deleted.")
        self.refresh_results()
