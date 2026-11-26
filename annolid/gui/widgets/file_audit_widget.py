import csv
import io
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.services.project_workspace import (
    PROJECT_CONFIG_FILENAME,
    audit_project_annotations,
    scan_annotation_audit,
)
from annolid.utils.atomic_io import atomic_write_text


class FileAuditWidget(QtWidgets.QWidget):
    """Widget to display, search, filter, open, and export scanned annotation files."""

    fileOpenRequested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)

        # Filters, Search & Export
        filter_layout = QtWidgets.QHBoxLayout()
        self.btn_all = QtWidgets.QRadioButton("All")
        self.btn_valid = QtWidgets.QRadioButton("Valid")
        self.btn_issues = QtWidgets.QRadioButton("Issues Only")
        self.btn_all.setChecked(True)

        self.btn_all.toggled.connect(self._apply_filter)
        self.btn_valid.toggled.connect(self._apply_filter)
        self.btn_issues.toggled.connect(self._apply_filter)

        self.search_edit = QtWidgets.QLineEdit(self)
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setPlaceholderText("Filter by filename or status…")
        self.search_edit.textChanged.connect(lambda _text: self._apply_filter())

        self.btn_export_csv = QtWidgets.QPushButton("Export CSV…")
        self.btn_export_csv.setToolTip("Export displayed file audit rows to CSV")
        self.btn_export_csv.clicked.connect(lambda: self.export_csv())

        filter_layout.addWidget(QtWidgets.QLabel("Filter:"))
        filter_layout.addWidget(self.btn_all)
        filter_layout.addWidget(self.btn_valid)
        filter_layout.addWidget(self.btn_issues)
        filter_layout.addWidget(self.search_edit, 1)
        filter_layout.addWidget(self.btn_export_csv)
        self.layout.addLayout(filter_layout)

        self.summary_label = QtWidgets.QLabel("Total: 0 | Valid: 0 | Issues: 0", self)
        self.summary_label.setEnabled(False)
        self.layout.addWidget(self.summary_label)

        # Table
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Status", "Filename", "Image", "Shapes"])
        self.table.horizontalHeader().setSectionResizeMode(
            1, QtWidgets.QHeaderView.Stretch
        )
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setAlternatingRowColors(True)
        self.table.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        self.table.itemDoubleClicked.connect(self._on_item_double_clicked)
        self.layout.addWidget(self.table)

        # Data storage
        self.items: List[Dict[str, Any]] = []

    def set_items(self, items: List[Dict[str, Any]]):
        self.items = list(items or [])
        self._apply_filter()

    def _filtered_items(self) -> List[Dict[str, Any]]:
        mode = "all"
        if self.btn_valid.isChecked():
            mode = "valid"
        elif self.btn_issues.isChecked():
            mode = "issues"
        query = str(self.search_edit.text() or "").strip().lower()

        filtered = []
        for item in self.items:
            is_valid = item.get("status") == "valid"
            if mode == "valid" and not is_valid:
                continue
            if mode == "issues" and is_valid:
                continue
            if query:
                json_name = str(item.get("json_path", "")).lower()
                status_str = str(item.get("status", "")).lower()
                if query not in json_name and query not in status_str:
                    continue
            filtered.append(item)
        return filtered

    def _update_summary(self, shown_count: int) -> None:
        total = len(self.items)
        valid = sum(1 for item in self.items if item.get("status") == "valid")
        issues = total - valid
        if shown_count != total:
            self.summary_label.setText(
                f"Showing: {shown_count}/{total} | Valid: {valid} | Issues: {issues}"
            )
        else:
            self.summary_label.setText(
                f"Total: {total} | Valid: {valid} | Issues: {issues}"
            )

    def _apply_filter(self):
        self.table.setRowCount(0)
        filtered = self._filtered_items()
        self.table.setRowCount(len(filtered))
        for row, item in enumerate(filtered):
            status_item = QtWidgets.QTableWidgetItem()
            if item["status"] == "valid":
                status_item.setText("✅")
                status_item.setToolTip("Valid annotation pair")
            elif item["status"] == "missing_image":
                status_item.setText("❌")
                status_item.setToolTip("Missing image file")
            else:
                status_item.setText("⚠️")
                status_item.setToolTip(str(item["status"]))
            status_item.setTextAlignment(QtCore.Qt.AlignCenter)
            self.table.setItem(row, 0, status_item)

            json_path = Path(item["json_path"])
            fname_item = QtWidgets.QTableWidgetItem(json_path.name)
            fname_item.setToolTip(str(json_path))
            self.table.setItem(row, 1, fname_item)

            img_text = "Yes" if item.get("image_path") else "Missing"
            img_item = QtWidgets.QTableWidgetItem(img_text)
            if not item.get("image_path"):
                img_item.setForeground(QtGui.QColor("#ef5350"))
            self.table.setItem(row, 2, img_item)

            shape_item = QtWidgets.QTableWidgetItem(str(item.get("shape_count", 0)))
            self.table.setItem(row, 3, shape_item)

            fname_item.setData(QtCore.Qt.UserRole, item)

        self._update_summary(len(filtered))

    def _on_item_double_clicked(self, item: QtWidgets.QTableWidgetItem) -> None:
        if item is None:
            return
        name_item = self.table.item(item.row(), 1)
        if name_item is None:
            return
        data = name_item.data(QtCore.Qt.UserRole)
        if not isinstance(data, dict):
            return
        target = data.get("image_path") or data.get("json_path")
        if target:
            self.fileOpenRequested.emit(str(target))

    def export_csv(self, output_path: Optional[Path | str] = None) -> Optional[Path]:
        rows = self._filtered_items()
        if output_path is None:
            chosen, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Export File Audit CSV",
                "file_audit.csv",
                "CSV Files (*.csv)",
            )
            if not chosen:
                return None
            target = Path(chosen)
        else:
            target = Path(output_path)

        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["status", "json_path", "image_path", "shape_count"])
        for item in rows:
            writer.writerow(
                [
                    item.get("status", ""),
                    str(item.get("json_path", "")),
                    str(item.get("image_path") or ""),
                    int(item.get("shape_count", 0) or 0),
                ]
            )
        try:
            atomic_write_text(target, buf.getvalue(), encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Export Failed",
                f"Failed to export file audit CSV:\n{exc}",
            )
            return None
        return target

    def _show_context_menu(self, position):
        menu = QtWidgets.QMenu()
        selected = self.table.selectedItems()
        if not selected:
            return

        row = selected[0].row()
        item_data = self.table.item(row, 1).data(QtCore.Qt.UserRole)
        json_path = Path(item_data["json_path"])

        action_open = menu.addAction("Open File")
        action_reveal = menu.addAction("Reveal in Finder/Explorer")
        action_delete = menu.addAction("Delete JSON File")

        action = menu.exec_(self.table.viewport().mapToGlobal(position))

        if action == action_open:
            target = item_data.get("image_path") or json_path
            self.fileOpenRequested.emit(str(target))
        elif action == action_reveal:
            self._reveal_file(json_path)
        elif action == action_delete:
            self._delete_file(json_path, row)

    def _reveal_file(self, path: Path):
        url = QtCore.QUrl.fromLocalFile(str(path.parent))
        QtGui.QDesktopServices.openUrl(url)

    def _delete_file(self, path: Path, row: int):
        confirm = QtWidgets.QMessageBox.question(
            self,
            "Confirm Delete",
            f"Are you sure you want to delete {path.name}?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if confirm == QtWidgets.QMessageBox.Yes:
            try:
                path.unlink()
                self.table.removeRow(row)
                self.items = [i for i in self.items if Path(i["json_path"]) != path]
                self._apply_filter()
            except Exception as e:
                QtWidgets.QMessageBox.critical(self, "Error", f"Failed to delete: {e}")


class _AnnotationAuditScanWorker(QtCore.QObject):
    """Background worker auditing LabelMe JSON pairs and NDJSON stores."""

    finished = QtCore.Signal(int, object, str)  # token, result_dict_or_none, error_msg

    def __init__(
        self,
        *,
        token: int,
        root_dir: Path,
        recursive: bool = True,
        project_config: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self._token = int(token)
        self._root_dir = Path(root_dir)
        self._recursive = bool(recursive)
        self._project_config = dict(project_config) if project_config else None
        self._cancel_event = threading.Event()

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancel_event.set()

    @QtCore.Slot()
    def run(self) -> None:
        result: Optional[Dict[str, Any]] = None
        error_msg = ""
        try:
            if (self._root_dir / PROJECT_CONFIG_FILENAME).is_file():
                result = audit_project_annotations(
                    self._root_dir,
                    config=self._project_config,
                    recursive=self._recursive,
                    cancel_check=self._cancel_event.is_set,
                )
            else:
                result = scan_annotation_audit(
                    [self._root_dir],
                    recursive=self._recursive,
                    cancel_check=self._cancel_event.is_set,
                )
        except Exception as exc:
            error_msg = str(exc)
        finally:
            self.finished.emit(self._token, result, error_msg)
            app = QtCore.QCoreApplication.instance()
            if app is not None and self.thread() != app.thread():
                try:
                    self.moveToThread(app.thread())
                except Exception:
                    pass


class FileAuditDialog(QtWidgets.QDialog):
    """Standalone/project-scoped annotation audit dialog with async scanning."""

    fileOpenRequested = QtCore.Signal(str)

    def __init__(
        self,
        *,
        initial_dir: Optional[str | Path] = None,
        project_config: Optional[Dict[str, Any]] = None,
        auto_scan: bool = True,
        parent: Optional[QtWidgets.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Project Annotation Audit")
        self.resize(860, 580)
        self._project_config = dict(project_config) if project_config else None
        self._scan_token = 0
        self._scan_thread: Optional[QtCore.QThread] = None
        self._scan_worker: Optional[_AnnotationAuditScanWorker] = None

        self._build_ui()
        if initial_dir:
            self.dir_edit.setText(str(Path(initial_dir).expanduser()))
            if auto_scan and Path(initial_dir).expanduser().is_dir():
                self.scan_directory(initial_dir)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        top_row.addWidget(QtWidgets.QLabel("Folder:", self))
        self.dir_edit = QtWidgets.QLineEdit(self)
        self.dir_edit.setPlaceholderText(
            "Select an Annolid project or annotations folder…"
        )
        top_row.addWidget(self.dir_edit, 1)

        self.browse_btn = QtWidgets.QPushButton("Browse…", self)
        self.browse_btn.clicked.connect(self._browse_folder)
        top_row.addWidget(self.browse_btn)

        self.recursive_check = QtWidgets.QCheckBox("Recursive", self)
        self.recursive_check.setChecked(True)
        top_row.addWidget(self.recursive_check)

        self.scan_btn = QtWidgets.QPushButton("Scan", self)
        self.scan_btn.clicked.connect(self._on_scan_or_cancel_clicked)
        top_row.addWidget(self.scan_btn)
        layout.addLayout(top_row)

        self.store_info_label = QtWidgets.QLabel("", self)
        self.store_info_label.setWordWrap(True)
        self.store_info_label.setVisible(False)
        layout.addWidget(self.store_info_label)

        self.audit_widget = FileAuditWidget(self)
        self.audit_widget.fileOpenRequested.connect(self.fileOpenRequested.emit)
        layout.addWidget(self.audit_widget, 1)

        footer = QtWidgets.QHBoxLayout()
        self.status_label = QtWidgets.QLabel("Select a folder and click Scan.", self)
        self.status_label.setEnabled(False)
        footer.addWidget(self.status_label, 1)
        self.close_btn = QtWidgets.QPushButton("Close", self)
        self.close_btn.clicked.connect(self.close)
        footer.addWidget(self.close_btn)
        layout.addLayout(footer)

    def _browse_folder(self) -> None:
        start = self.dir_edit.text().strip() or str(Path.home())
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Audit",
            start,
        )
        if chosen:
            self.dir_edit.setText(chosen)
            self.scan_directory(chosen)

    def _on_scan_or_cancel_clicked(self) -> None:
        if self._scan_thread is not None and self._scan_thread.isRunning():
            self.cancel_scan(wait_ms=0)
            self.status_label.setText("Cancelling scan…")
            return
        folder = self.dir_edit.text().strip()
        if not folder:
            self._browse_folder()
            return
        self.scan_directory(folder)

    def cancel_scan(self, *, wait_ms: int = 2000) -> None:
        self._scan_token += 1
        worker = self._scan_worker
        thread = self._scan_thread
        if worker is not None:
            try:
                worker.cancel()
            except RuntimeError:
                pass
        if thread is not None:
            try:
                if thread.isRunning() and wait_ms > 0:
                    thread.quit()
                    thread.wait(wait_ms)
            except RuntimeError:
                pass
        if wait_ms > 0:
            if worker is not None:
                try:
                    worker.deleteLater()
                except RuntimeError:
                    pass
            if thread is not None:
                try:
                    thread.deleteLater()
                except RuntimeError:
                    pass
            self._scan_worker = None
            self._scan_thread = None
            self.scan_btn.setText("Scan")

    def scan_directory(
        self,
        folder: str | Path,
        *,
        recursive: Optional[bool] = None,
        project_config: Optional[Dict[str, Any]] = None,
        async_scan: bool = True,
    ) -> None:
        target = Path(folder).expanduser()
        self.dir_edit.setText(str(target))
        if recursive is not None:
            self.recursive_check.setChecked(bool(recursive))
        if project_config is not None:
            self._project_config = dict(project_config)

        if not target.is_dir():
            self.status_label.setText(f"Folder does not exist: {target}")
            self.audit_widget.set_items([])
            self.store_info_label.setVisible(False)
            return

        self.cancel_scan(wait_ms=1000)
        self._scan_token += 1
        token = self._scan_token
        use_recursive = bool(self.recursive_check.isChecked())

        if not async_scan:
            try:
                if (target / PROJECT_CONFIG_FILENAME).is_file():
                    result = audit_project_annotations(
                        target,
                        config=self._project_config,
                        recursive=use_recursive,
                    )
                else:
                    result = scan_annotation_audit([target], recursive=use_recursive)
            except Exception as exc:
                self._apply_scan_result(token, None, str(exc))
            else:
                self._apply_scan_result(token, result, "")
            return

        self.status_label.setText(f"Scanning {target.name}…")
        self.scan_btn.setText("Cancel")
        self.browse_btn.setEnabled(False)

        worker = _AnnotationAuditScanWorker(
            token=token,
            root_dir=target,
            recursive=use_recursive,
            project_config=self._project_config,
        )
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        worker.finished.connect(self._on_scan_finished, QtCore.Qt.QueuedConnection)
        thread.started.connect(worker.run)

        self._scan_worker = worker
        self._scan_thread = thread
        thread.start()

    @QtCore.Slot(int, object, str)
    def _on_scan_finished(self, token: int, result: object, error_msg: str) -> None:
        thread = self._scan_thread
        worker = self._scan_worker
        self._scan_thread = None
        self._scan_worker = None
        if thread is not None:
            try:
                thread.quit()
                thread.wait(2000)
                thread.deleteLater()
            except RuntimeError:
                pass
        if worker is not None:
            try:
                worker.deleteLater()
            except RuntimeError:
                pass
        self.scan_btn.setText("Scan")
        self.browse_btn.setEnabled(True)
        if token != self._scan_token:
            if self._scan_thread is None:
                self.status_label.setText("Scan cancelled.")
            return
        self._apply_scan_result(
            token, result if isinstance(result, dict) else None, error_msg
        )

    def _apply_scan_result(
        self,
        token: int,
        result: Optional[Dict[str, Any]],
        error_msg: str,
    ) -> None:
        if token != self._scan_token:
            return
        if error_msg:
            self.status_label.setText(f"Scan failed: {error_msg}")
            self.store_info_label.setVisible(False)
            return
        if not result:
            self.status_label.setText("Scan returned no data.")
            self.store_info_label.setVisible(False)
            return
        if result.get("cancelled"):
            self.status_label.setText("Scan cancelled.")
            return

        items = list(result.get("items") or [])
        stores = list(result.get("ndjson_stores") or [])
        ndjson_frames = int(result.get("ndjson_total_frames", 0) or 0)
        self.audit_widget.set_items(items)

        if stores:
            broken = [s for s in stores if s.get("error")]
            store_msg = (
                f"AnnotationStore NDJSON: {len(stores)} store(s), "
                f"{ndjson_frames} indexed frame(s)"
            )
            if broken:
                store_msg += f" ({len(broken)} unreadable)"
            self.store_info_label.setText(store_msg)
            self.store_info_label.setVisible(True)
        else:
            self.store_info_label.setVisible(False)

        self.status_label.setText(
            f"Scanned {len(items)} LabelMe JSON file(s)"
            + (f" and {len(stores)} NDJSON store(s)" if stores else "")
            + ". Double-click any row to open in Annolid."
        )

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.cancel_scan(wait_ms=2000)
        super().closeEvent(event)
