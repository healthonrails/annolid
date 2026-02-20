from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.utils.citations import (
    BibEntry,
    entry_to_dict,
    load_bibtex,
    merge_validated_fields,
    remove_entry,
    save_bibtex,
    upsert_entry,
    validate_basic_citation_fields,
    validate_citation_metadata,
)


class CitationManagerDialog(QtWidgets.QDialog):
    def __init__(
        self,
        *,
        default_bib_path_getter: Callable[[], Path],
        save_from_context: Callable[..., Dict[str, Any]],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._default_bib_path_getter = default_bib_path_getter
        self._save_from_context = save_from_context
        self._entries: list[BibEntry] = []
        self._table_updating = False

        self.setWindowTitle("Citations")
        self.resize(900, 520)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        top = QtWidgets.QHBoxLayout()
        self.bib_path_edit = QtWidgets.QLineEdit(self)
        self.bib_path_edit.setPlaceholderText("Path to .bib file")
        browse_btn = QtWidgets.QPushButton("Browse", self)
        browse_btn.clicked.connect(self._browse_bib_path)
        open_btn = QtWidgets.QPushButton("Open File", self)
        open_btn.clicked.connect(self._open_bib_file)
        refresh_btn = QtWidgets.QPushButton("Refresh", self)
        refresh_btn.clicked.connect(self._refresh_table)
        top.addWidget(self.bib_path_edit, 1)
        top.addWidget(browse_btn)
        top.addWidget(open_btn)
        top.addWidget(refresh_btn)
        root.addLayout(top)

        self.table = QtWidgets.QTableWidget(self)
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(
            ["Key", "Type", "Title", "Year", "DOI", "Source"]
        )
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.SelectedClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.ResizeToContents
        )
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.ResizeToContents
        )
        self.table.setAcceptDrops(True)
        self.table.viewport().setAcceptDrops(True)
        self.table.dragEnterEvent = self._drag_enter_event
        self.table.dragMoveEvent = self._drag_move_event
        self.table.dropEvent = self._drop_event
        root.addWidget(self.table, 1)

        controls = QtWidgets.QHBoxLayout()
        self.key_edit = QtWidgets.QLineEdit(self)
        self.key_edit.setPlaceholderText("Optional key override for save-from-context")
        self.validate_checkbox = QtWidgets.QCheckBox("Auto validate before save", self)
        self.validate_checkbox.setChecked(True)
        self.strict_checkbox = QtWidgets.QCheckBox("Strict", self)
        self.strict_checkbox.setChecked(False)
        self.strict_checkbox.setToolTip(
            "Fail save when external metadata validation is weak."
        )
        add_empty_btn = QtWidgets.QPushButton("Add Manual Row", self)
        add_empty_btn.clicked.connect(self._add_manual_row)
        save_row_btn = QtWidgets.QPushButton("Save Row Edits", self)
        save_row_btn.clicked.connect(self._save_selected_row_edits)
        from_pdf_btn = QtWidgets.QPushButton("Save From PDF", self)
        from_pdf_btn.clicked.connect(lambda: self._save_from_active_context("pdf"))
        from_web_btn = QtWidgets.QPushButton("Save From Web", self)
        from_web_btn.clicked.connect(lambda: self._save_from_active_context("web"))
        auto_btn = QtWidgets.QPushButton("Save Auto", self)
        auto_btn.clicked.connect(lambda: self._save_from_active_context("auto"))
        validate_all_btn = QtWidgets.QPushButton("Validate All", self)
        validate_all_btn.clicked.connect(self._validate_all_rows)
        remove_btn = QtWidgets.QPushButton("Remove Selected", self)
        remove_btn.clicked.connect(self._remove_selected)
        controls.addWidget(self.key_edit, 1)
        controls.addWidget(self.validate_checkbox)
        controls.addWidget(self.strict_checkbox)
        controls.addWidget(add_empty_btn)
        controls.addWidget(save_row_btn)
        controls.addWidget(from_pdf_btn)
        controls.addWidget(from_web_btn)
        controls.addWidget(auto_btn)
        controls.addWidget(validate_all_btn)
        controls.addWidget(remove_btn)
        root.addLayout(controls)

        self.status_label = QtWidgets.QLabel("", self)
        self.duplicate_warning_label = QtWidgets.QLabel("", self)
        self.duplicate_warning_label.setStyleSheet("color: #b45309;")
        root.addWidget(self.status_label)
        root.addWidget(self.duplicate_warning_label)

        self.table.itemChanged.connect(self._on_table_item_changed)

        self._set_default_bib_path()
        self._refresh_table()

    def _set_default_bib_path(self) -> None:
        self.bib_path_edit.setText(str(self._default_bib_path_getter()))

    def _resolved_bib_path(self) -> Path:
        text = str(self.bib_path_edit.text() or "").strip()
        return Path(text).expanduser() if text else self._default_bib_path_getter()

    def _browse_bib_path(self) -> None:
        current = str(self._resolved_bib_path())
        selected, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select BibTeX File",
            current,
            "BibTeX Files (*.bib);;All Files (*)",
        )
        if selected:
            self.bib_path_edit.setText(selected)
            self._refresh_table()

    def _open_bib_file(self) -> None:
        path = self._resolved_bib_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")
        opened = QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))
        self.status_label.setText(
            f"Opened {path.name}" if opened else "Failed to open .bib file."
        )

    def _refresh_table(self) -> None:
        path = self._resolved_bib_path()
        self._entries = list(load_bibtex(path))
        self._table_updating = True
        self.table.setRowCount(len(self._entries))
        for row, entry in enumerate(self._entries):
            payload = entry_to_dict(entry)
            key_item = QtWidgets.QTableWidgetItem(str(payload["key"]))
            key_item.setData(QtCore.Qt.UserRole, str(payload["key"]))
            self.table.setItem(row, 0, key_item)
            self.table.setItem(
                row, 1, QtWidgets.QTableWidgetItem(str(payload["entry_type"]))
            )
            self.table.setItem(
                row, 2, QtWidgets.QTableWidgetItem(str(payload.get("title") or ""))
            )
            self.table.setItem(
                row, 3, QtWidgets.QTableWidgetItem(str(payload.get("year") or ""))
            )
            self.table.setItem(
                row,
                4,
                QtWidgets.QTableWidgetItem(str(entry.fields.get("doi") or "")),
            )
            source_ref = str(
                entry.fields.get("url")
                or entry.fields.get("source_path")
                or entry.fields.get("file")
                or ""
            ).strip()
            source_item = QtWidgets.QTableWidgetItem(source_ref)
            source_item.setToolTip(source_ref)
            self.table.setItem(row, 5, source_item)
        self._table_updating = False
        self._refresh_duplicate_warning()
        self.status_label.setText(f"{len(self._entries)} citation(s).")

    def _save_from_active_context(self, source: str) -> None:
        key = str(self.key_edit.text() or "").strip()
        payload = self._save_from_context(
            source=source,
            key=key,
            bib_path=self._resolved_bib_path(),
            validate_before_save=bool(self.validate_checkbox.isChecked()),
            strict_validation=bool(self.strict_checkbox.isChecked()),
        )
        if not payload.get("ok"):
            self.status_label.setText(str(payload.get("error") or "Save failed."))
            return
        self._refresh_table()
        self.key_edit.clear()
        validation = dict(payload.get("validation") or {})
        suffix = ""
        if bool(validation.get("checked")):
            provider = str(validation.get("provider") or "").strip()
            score = float(validation.get("score") or 0.0)
            state = "verified" if bool(validation.get("verified")) else "unverified"
            label = f"{provider} {state}" if provider else state
            suffix = f" | validation: {label} ({score:.2f})"
        self.status_label.setText(
            f"{'Created' if payload.get('created') else 'Updated'} {payload.get('key')}{suffix}"
        )

    def _remove_selected(self) -> None:
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            self.status_label.setText("Select citation rows to remove.")
            return
        path = self._resolved_bib_path()
        entries = load_bibtex(path)
        keys = {
            str(self.table.item(index.row(), 0).text() or "").strip()
            for index in rows
            if self.table.item(index.row(), 0) is not None
        }
        removed_any = False
        for key in keys:
            entries, removed = remove_entry(entries, key)
            removed_any = removed_any or removed
        if removed_any:
            save_bibtex(path, entries, sort_keys=True)
            self._refresh_table()
            self.status_label.setText(f"Removed {len(keys)} citation(s).")
        else:
            self.status_label.setText("No matching citation keys found.")

    def _add_manual_row(self) -> None:
        self._table_updating = True
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem("new_key"))
        self.table.setItem(row, 1, QtWidgets.QTableWidgetItem("article"))
        self.table.setItem(row, 2, QtWidgets.QTableWidgetItem(""))
        self.table.setItem(row, 3, QtWidgets.QTableWidgetItem(""))
        self.table.setItem(row, 4, QtWidgets.QTableWidgetItem(""))
        self.table.setItem(row, 5, QtWidgets.QTableWidgetItem(""))
        self.table.setCurrentCell(row, 0)
        self.table.editItem(self.table.item(row, 0))
        self._table_updating = False
        self.status_label.setText("Edit the row, then click 'Save Row Edits'.")
        self._refresh_duplicate_warning()

    def _save_selected_row_edits(self) -> None:
        row = self.table.currentRow()
        if row < 0:
            self.status_label.setText("Select a row to save edits.")
            return
        key_item = self.table.item(row, 0)
        type_item = self.table.item(row, 1)
        title_item = self.table.item(row, 2)
        year_item = self.table.item(row, 3)
        doi_item = self.table.item(row, 4)
        source_item = self.table.item(row, 5)
        new_key = str(key_item.text() if key_item else "").strip()
        if not new_key:
            self.status_label.setText("Citation key is required.")
            return
        entry_type = (
            str(type_item.text() if type_item else "").strip().lower() or "article"
        )
        title = str(title_item.text() if title_item else "").strip()
        year = str(year_item.text() if year_item else "").strip()
        doi = str(doi_item.text() if doi_item else "").strip()
        source_ref = str(source_item.text() if source_item else "").strip()
        original_key = (
            str(key_item.data(QtCore.Qt.UserRole) or "").strip() if key_item else ""
        )
        basic_errors = validate_basic_citation_fields(
            {
                "__key__": new_key,
                "year": year,
                "doi": doi,
            }
        )
        if basic_errors:
            self.status_label.setText(" ".join(basic_errors))
            return

        path = self._resolved_bib_path()
        entries = load_bibtex(path)
        base_entry: Optional[BibEntry] = None
        for existing in entries:
            if existing.key.strip().lower() == original_key.strip().lower():
                base_entry = existing
                break
        fields = dict(base_entry.fields) if base_entry is not None else {}
        fields["title"] = title
        if year:
            fields["year"] = year
        else:
            fields.pop("year", None)
        if doi:
            fields["doi"] = doi
            fields.setdefault("url", f"https://doi.org/{doi}")
        else:
            fields.pop("doi", None)
        if source_ref:
            if source_ref.startswith(("http://", "https://")):
                fields["url"] = source_ref
                fields.pop("source_path", None)
            else:
                fields["source_path"] = source_ref
                if fields.get("url", "").strip() == source_ref:
                    fields.pop("url", None)
        else:
            fields.pop("source_path", None)
        if not title:
            fields.pop("title", None)
        if bool(self.validate_checkbox.isChecked()):
            validation = validate_citation_metadata(fields, timeout_s=1.8)
            fields = merge_validated_fields(
                fields, validation, replace_when_confident=True
            )
            if bool(self.strict_checkbox.isChecked()) and not bool(
                validation.get("verified")
            ):
                self.status_label.setText(
                    (
                        "Strict validation blocked save. "
                        + str(validation.get("message") or "")
                    ).strip()
                )
                return

        if original_key:
            entries, _ = remove_entry(entries, original_key)
        entries, created = upsert_entry(
            entries,
            BibEntry(entry_type=entry_type, key=new_key, fields=fields),
        )
        save_bibtex(path, entries, sort_keys=True)
        self._refresh_table()
        self.status_label.setText(
            f"{'Created' if created else 'Updated'} row for key '{new_key}'."
        )

    def _on_table_item_changed(self, item: QtWidgets.QTableWidgetItem) -> None:
        if self._table_updating:
            return
        if item.column() not in (0, 3, 4):
            return
        row = item.row()
        key = str(
            self.table.item(row, 0).text() if self.table.item(row, 0) else ""
        ).strip()
        year = str(
            self.table.item(row, 3).text() if self.table.item(row, 3) else ""
        ).strip()
        doi = str(
            self.table.item(row, 4).text() if self.table.item(row, 4) else ""
        ).strip()
        errors = validate_basic_citation_fields(
            {"__key__": key, "year": year, "doi": doi}
        )
        default_brush = self.palette().brush(QtGui.QPalette.Text)
        for col in (0, 3, 4):
            cell = self.table.item(row, col)
            if cell is None:
                continue
            if errors and col == item.column():
                cell.setForeground(QtGui.QBrush(QtGui.QColor("#b91c1c")))
            else:
                cell.setForeground(default_brush)
            if col == item.column() and errors:
                cell.setToolTip(" ".join(errors))
            elif col == item.column():
                cell.setToolTip("")
        self._refresh_duplicate_warning()

    def _refresh_duplicate_warning(self) -> None:
        seen: set[str] = set()
        duplicates: set[str] = set()
        for row in range(self.table.rowCount()):
            key_item = self.table.item(row, 0)
            if key_item is None:
                continue
            key = str(key_item.text() or "").strip().lower()
            if not key:
                continue
            if key in seen:
                duplicates.add(key)
            seen.add(key)
        if duplicates:
            self.duplicate_warning_label.setText(
                "Duplicate keys: " + ", ".join(sorted(duplicates))
            )
        else:
            self.duplicate_warning_label.setText("")

    def _validate_all_rows(self) -> None:
        self.status_label.setText("Validating all rows... this may take a moment.")
        QtWidgets.QApplication.processEvents()
        path = self._resolved_bib_path()
        entries = load_bibtex(path)
        updated_any = False
        for entry in entries:
            validation = validate_citation_metadata(entry.fields, timeout_s=2.0)
            if validation.get("verified"):
                new_fields = merge_validated_fields(
                    entry.fields, validation, replace_when_confident=True
                )
                if new_fields != entry.fields:
                    entry.fields = new_fields
                    updated_any = True
        if updated_any:
            save_bibtex(path, entries, sort_keys=True)
            self._refresh_table()
            self.status_label.setText("Batch validation completed and updated entries.")
        else:
            self.status_label.setText("Batch validation finished (no changes needed).")

    def _drag_enter_event(self, event: QtGui.QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(".pdf"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def _drag_move_event(self, event: QtGui.QDragMoveEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                if url.isLocalFile() and url.toLocalFile().lower().endswith(".pdf"):
                    event.acceptProposedAction()
                    return
        event.ignore()

    def _drop_event(self, event: QtGui.QDropEvent) -> None:
        urls = event.mimeData().urls()
        if not urls:
            return

        path = self._resolved_bib_path()
        entries = load_bibtex(path)
        added_count = 0

        for url in urls:
            if url.isLocalFile() and url.toLocalFile().lower().endswith(".pdf"):
                local_path = url.toLocalFile()
                file_name = Path(local_path).stem
                # Clean up filename dashes/underscores for search
                title_query = file_name.replace("_", " ").replace("-", " ")

                self.status_label.setText(
                    f"Resolving citation for: {title_query[:30]}..."
                )
                QtWidgets.QApplication.processEvents()

                fields = {"title": title_query, "file": local_path}
                validation = validate_citation_metadata(fields, timeout_s=2.5)

                if validation.get("candidate"):
                    fields = merge_validated_fields(
                        fields, validation, replace_when_confident=True
                    )

                import re

                candidate_key = str(fields.get("__bibkey__") or "").strip()
                if not candidate_key:
                    candidate_key = re.sub(
                        r"[^a-zA-Z0-9:_\-.]+", "_", title_query
                    ).strip("_")[:15]

                if not candidate_key:
                    candidate_key = f"pdf_ref_{added_count}"

                entry_type = "article"

                entries, _ = upsert_entry(
                    entries,
                    BibEntry(entry_type=entry_type, key=candidate_key, fields=fields),
                )
                added_count += 1

        if added_count > 0:
            save_bibtex(path, entries, sort_keys=True)
            self._refresh_table()
            self.status_label.setText(
                f"Resolved and added {added_count} citation(s) from PDF(s)."
            )
