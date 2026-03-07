from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from qtpy import QtWidgets

from annolid.domain.memory.taxonomy import MEMORY_CATEGORIES, MEMORY_SOURCES
from annolid.infrastructure.memory.lancedb.migration import (
    collect_legacy_records,
    import_records,
)
from annolid.interfaces.memory.registry import get_memory_backend
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
    """GUI memory operations: record CRUD and migration dashboard."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Memory Manager")
        self.resize(980, 620)
        self._rows = []
        self._migration_records = []

        root = QtWidgets.QVBoxLayout(self)
        self.tabs = QtWidgets.QTabWidget(self)
        root.addWidget(self.tabs)
        self._build_records_tab()
        self._build_migration_tab()

        footer = QtWidgets.QHBoxLayout()
        root.addLayout(footer)
        self.close_btn = QtWidgets.QPushButton("Close", self)
        footer.addStretch(1)
        footer.addWidget(self.close_btn)

        self.status_label = QtWidgets.QLabel(self)
        root.addWidget(self.status_label)

        self.search_btn.clicked.connect(self.refresh_results)
        self.refresh_btn.clicked.connect(self.refresh_results)
        self.add_btn.clicked.connect(self.add_record)
        self.edit_btn.clicked.connect(self.edit_selected_record)
        self.delete_btn.clicked.connect(self.delete_selected_record)
        self.close_btn.clicked.connect(self.close)
        self.table.doubleClicked.connect(lambda _idx: self.edit_selected_record())
        self.pick_dir_btn.clicked.connect(self.pick_migration_dir)
        self.scan_btn.clicked.connect(self.scan_migration_sources)
        self.import_btn.clicked.connect(self.import_migration_sources)
        self.refresh_results()

    def _build_records_tab(self) -> None:
        page = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(page)
        top = QtWidgets.QHBoxLayout()
        layout.addLayout(top)
        self.search_edit = QtWidgets.QLineEdit(page)
        self.search_edit.setPlaceholderText(
            "Search memories (leave blank to list scoped records)"
        )
        self.scope_edit = QtWidgets.QLineEdit(page)
        self.scope_edit.setPlaceholderText("Optional scope, e.g. workspace:default")
        self.search_btn = QtWidgets.QPushButton("Search", page)
        self.refresh_btn = QtWidgets.QPushButton("Refresh", page)
        top.addWidget(self.search_edit, 3)
        top.addWidget(self.scope_edit, 2)
        top.addWidget(self.search_btn)
        top.addWidget(self.refresh_btn)

        self.table = QtWidgets.QTableWidget(page)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["ID", "Scope", "Category", "Source", "Importance", "Text"]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table)

        buttons = QtWidgets.QHBoxLayout()
        layout.addLayout(buttons)
        self.add_btn = QtWidgets.QPushButton("Add", page)
        self.edit_btn = QtWidgets.QPushButton("Edit", page)
        self.delete_btn = QtWidgets.QPushButton("Delete", page)
        buttons.addWidget(self.add_btn)
        buttons.addWidget(self.edit_btn)
        buttons.addWidget(self.delete_btn)
        buttons.addStretch(1)
        self.tabs.addTab(page, "Records")

    def _build_migration_tab(self) -> None:
        page = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(page)

        source_row = QtWidgets.QHBoxLayout()
        layout.addLayout(source_row)
        self.migration_source_edit = QtWidgets.QLineEdit(page)
        self.migration_source_edit.setPlaceholderText(
            "Legacy source root directory (JSON, markdown logs, project schemas)"
        )
        self.pick_dir_btn = QtWidgets.QPushButton("Browse…", page)
        self.scan_btn = QtWidgets.QPushButton("Scan Sources", page)
        self.import_btn = QtWidgets.QPushButton("Import Scanned", page)
        source_row.addWidget(self.migration_source_edit, 1)
        source_row.addWidget(self.pick_dir_btn)
        source_row.addWidget(self.scan_btn)
        source_row.addWidget(self.import_btn)

        counts_row = QtWidgets.QHBoxLayout()
        layout.addLayout(counts_row)
        self.json_count_label = QtWidgets.QLabel("JSON: 0", page)
        self.markdown_count_label = QtWidgets.QLabel("Markdown: 0", page)
        self.schema_count_label = QtWidgets.QLabel("Project schema: 0", page)
        self.total_count_label = QtWidgets.QLabel("Total records: 0", page)
        counts_row.addWidget(self.json_count_label)
        counts_row.addWidget(self.markdown_count_label)
        counts_row.addWidget(self.schema_count_label)
        counts_row.addWidget(self.total_count_label)
        counts_row.addStretch(1)

        self.migration_report = QtWidgets.QPlainTextEdit(page)
        self.migration_report.setReadOnly(True)
        layout.addWidget(self.migration_report)
        self.tabs.addTab(page, "Migration Dashboard")

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

    def pick_migration_dir(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Legacy Memory Source Directory",
            self.migration_source_edit.text().strip() or str(Path.home()),
        )
        if folder:
            self.migration_source_edit.setText(folder)

    def scan_migration_sources(self) -> None:
        source_text = self.migration_source_edit.text().strip()
        if not source_text:
            self._set_status("Select a migration source directory first.")
            return
        source_root = Path(source_text).expanduser()
        if not source_root.exists() or not source_root.is_dir():
            self._set_status(f"Invalid source directory: {source_root}")
            return
        records, stats = collect_legacy_records(source_root)
        self._migration_records = records
        self.json_count_label.setText(f"JSON: {stats.get('json', 0)}")
        self.markdown_count_label.setText(f"Markdown: {stats.get('markdown', 0)}")
        self.schema_count_label.setText(
            f"Project schema: {stats.get('project_schema', 0)}"
        )
        self.total_count_label.setText(f"Total records: {len(records)}")

        lines = [
            f"Source root: {source_root}",
            f"JSON records: {stats.get('json', 0)}",
            f"Markdown records: {stats.get('markdown', 0)}",
            f"Project schema records: {stats.get('project_schema', 0)}",
            f"Total records: {len(records)}",
            "",
            "Preview (up to 20 records):",
        ]
        for record in records[:20]:
            lines.append(f"- [{record.category}] {record.text[:140]}")
        self.migration_report.setPlainText("\n".join(lines))
        self._set_status(f"Scan complete: {len(records)} candidate records.")

    def import_migration_sources(self) -> None:
        if not self._migration_records:
            self.scan_migration_sources()
            if not self._migration_records:
                return
        backend = get_memory_backend()
        if not backend:
            self._set_status("Memory backend unavailable.")
            return
        result = import_records(backend, self._migration_records)
        current = self.migration_report.toPlainText().rstrip()
        summary = (
            "\n\nImport result:\n"
            f"- imported: {result.imported}\n"
            f"- failed: {result.failed}\n"
        )
        self.migration_report.setPlainText((current + summary).strip() + "\n")
        self.refresh_results()
        self._set_status(
            f"Migration import complete: imported={result.imported} failed={result.failed}"
        )
