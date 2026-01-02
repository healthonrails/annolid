from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from qtpy import QtCore, QtGui, QtWidgets

from annolid.datasets.labelme_collection import (
    DEFAULT_LABEL_INDEX_DIRNAME,
    DEFAULT_LABEL_INDEX_NAME,
    default_label_index_path,
    index_labelme_pair,
    iter_labelme_json_files,
    load_indexed_image_paths,
    resolve_image_path,
)
from annolid.gui.workers import FlexibleWorker
from annolid.datasets.builders.label_index_yolo import build_yolo_from_label_index


@dataclass(frozen=True)
class LabelIndexJob:
    sources: List[Path]
    index_file: Path
    recursive: bool
    include_empty: bool
    allow_duplicates: bool


def _build_json_list(sources: Sequence[Path], *, recursive: bool) -> List[Path]:
    json_files: List[Path] = []
    for src in sources:
        json_files.extend(
            list(iter_labelme_json_files(Path(src), recursive=recursive)))
    return sorted(set(json_files))


def _run_index_job(job: LabelIndexJob, *, pred_worker=None, stop_event=None) -> dict:
    index_file = Path(job.index_file).expanduser().resolve()
    sources = [Path(p).expanduser().resolve() for p in job.sources]

    json_files = _build_json_list(sources, recursive=job.recursive)
    total = len(json_files)

    indexed_images = set()
    if not job.allow_duplicates:
        indexed_images = load_indexed_image_paths(index_file)

    appended = 0
    skipped = 0
    missing_image = 0

    for i, json_path in enumerate(json_files, start=1):
        if stop_event is not None and stop_event.is_set():
            break

        image_path = resolve_image_path(json_path)
        if image_path is None:
            missing_image += 1
            skipped += 1
        else:
            image_abs = str(Path(image_path).expanduser().resolve())
            if (not job.allow_duplicates) and image_abs in indexed_images:
                skipped += 1
            else:
                record = index_labelme_pair(
                    json_path=json_path,
                    index_file=index_file,
                    image_path=image_path,
                    include_empty=job.include_empty,
                    source="annolid_gui_dialog",
                )
                if record is None:
                    skipped += 1
                else:
                    appended += 1
                    if not job.allow_duplicates:
                        indexed_images.add(image_abs)

        if pred_worker is not None and total > 0:
            pct = int(round((i / total) * 100))
            pred_worker.progress_signal.emit(max(0, min(100, pct)))
            pred_worker.preview_signal.emit(
                {
                    "processed": i,
                    "total": total,
                    "appended": appended,
                    "skipped": skipped,
                    "missing_image": missing_image,
                }
            )

    return {
        "index_file": str(index_file),
        "sources": [str(p) for p in sources],
        "recursive": bool(job.recursive),
        "include_empty": bool(job.include_empty),
        "allow_duplicates": bool(job.allow_duplicates),
        "total_json": total,
        "appended": appended,
        "skipped": skipped,
        "missing_image": missing_image,
    }


class LabelCollectionDialog(QtWidgets.QDialog):
    """GUI for indexing LabelMe PNG/JSON pairs into a JSONL dataset index."""

    def __init__(self, *, settings: Optional[QtCore.QSettings] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Collect Labels (Dataset Index)")
        self.setMinimumWidth(700)

        self._settings = settings
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[FlexibleWorker] = None

        self._build_ui()
        self._load_settings()
        self._refresh_index_preview()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._worker is not None or self._thread is not None:
            choice = QtWidgets.QMessageBox.question(
                self,
                "Stop collection?",
                "Label collection is running. Stop and close?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if choice != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return
            self._request_stop()
        super().closeEvent(event)

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        header = QtWidgets.QLabel(
            "Build a central dataset index without copying files.\n"
            f"Default index location: {DEFAULT_LABEL_INDEX_DIRNAME}/{DEFAULT_LABEL_INDEX_NAME}"
        )
        header.setWordWrap(True)
        layout.addWidget(header)

        self.tabs = QtWidgets.QTabWidget(self)
        layout.addWidget(self.tabs)

        self._auto_tab = QtWidgets.QWidget(self)
        self._backfill_tab = QtWidgets.QWidget(self)
        self._export_tab = QtWidgets.QWidget(self)
        self.tabs.addTab(self._auto_tab, "Auto Indexing")
        self.tabs.addTab(self._backfill_tab, "Backfill / Collect")
        self.tabs.addTab(self._export_tab, "Export")

        self._build_auto_tab()
        self._build_backfill_tab()
        self._build_export_tab()

        self.status_label = QtWidgets.QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        layout.addWidget(self.log)

        buttons = QtWidgets.QDialogButtonBox(self)
        self.close_btn = buttons.addButton(
            "Close", QtWidgets.QDialogButtonBox.RejectRole)
        self.close_btn.clicked.connect(self.reject)

        self.run_btn = buttons.addButton(
            "Collect Now", QtWidgets.QDialogButtonBox.AcceptRole)
        self.run_btn.clicked.connect(self._start_backfill_job)

        self.export_btn = buttons.addButton(
            "Convert to YOLO", QtWidgets.QDialogButtonBox.ActionRole)
        self.export_btn.clicked.connect(self._start_yolo_job)

        self.cancel_btn = buttons.addButton(
            "Cancel", QtWidgets.QDialogButtonBox.DestructiveRole)
        self.cancel_btn.clicked.connect(self._request_stop)
        self.cancel_btn.setEnabled(False)

        layout.addWidget(buttons)

        self.tabs.currentChanged.connect(self._sync_primary_actions)
        self._sync_primary_actions()

    def _build_auto_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self._auto_tab)

        self.auto_enable = QtWidgets.QCheckBox("Enable auto indexing on save")
        self.auto_enable.stateChanged.connect(self._save_settings)
        layout.addWidget(self.auto_enable)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.dataset_root_edit = QtWidgets.QLineEdit()
        self.dataset_root_btn = QtWidgets.QPushButton("Browse…")
        self.dataset_root_btn.clicked.connect(self._browse_dataset_root)

        root_row = QtWidgets.QHBoxLayout()
        root_row.addWidget(self.dataset_root_edit, 1)
        root_row.addWidget(self.dataset_root_btn)
        form.addRow("Dataset root:", root_row)

        self.index_file_edit = QtWidgets.QLineEdit()
        self.index_file_btn = QtWidgets.QPushButton("Choose file…")
        self.index_file_btn.clicked.connect(self._browse_index_file)

        index_row = QtWidgets.QHBoxLayout()
        index_row.addWidget(self.index_file_edit, 1)
        index_row.addWidget(self.index_file_btn)
        form.addRow("Index file:", index_row)

        hint = QtWidgets.QLabel(
            "Notes:\n"
            "- Environment variable ANNOLID_LABEL_INDEX_FILE overrides this setting.\n"
            "- The index file is append-only JSONL."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        open_row = QtWidgets.QHBoxLayout()
        self.open_index_btn = QtWidgets.QPushButton("Open index file")
        self.open_index_btn.clicked.connect(self._open_index_file)
        self.open_folder_btn = QtWidgets.QPushButton("Open index folder")
        self.open_folder_btn.clicked.connect(self._open_index_folder)
        open_row.addWidget(self.open_index_btn)
        open_row.addWidget(self.open_folder_btn)
        open_row.addStretch(1)
        layout.addLayout(open_row)

        self.dataset_root_edit.editingFinished.connect(
            self._on_dataset_root_changed)
        self.index_file_edit.editingFinished.connect(self._save_settings)

        layout.addStretch(1)

    def _build_backfill_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self._backfill_tab)

        self.sources_list = QtWidgets.QListWidget(self)
        self.sources_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.sources_list, 2)

        row = QtWidgets.QHBoxLayout()
        self.add_source_btn = QtWidgets.QPushButton("Add folder…")
        self.add_source_btn.clicked.connect(self._add_source_folder)
        self.remove_source_btn = QtWidgets.QPushButton("Remove selected")
        self.remove_source_btn.clicked.connect(self._remove_selected_sources)
        self.clear_sources_btn = QtWidgets.QPushButton("Clear")
        self.clear_sources_btn.clicked.connect(self.sources_list.clear)
        row.addWidget(self.add_source_btn)
        row.addWidget(self.remove_source_btn)
        row.addWidget(self.clear_sources_btn)
        row.addStretch(1)
        layout.addLayout(row)

        opts = QtWidgets.QHBoxLayout()
        self.recursive_cb = QtWidgets.QCheckBox("Recursive")
        self.recursive_cb.setChecked(True)
        self.include_empty_cb = QtWidgets.QCheckBox("Include empty JSON")
        self.include_empty_cb.setChecked(False)
        self.allow_dupes_cb = QtWidgets.QCheckBox("Allow duplicates")
        self.allow_dupes_cb.setChecked(False)
        opts.addWidget(self.recursive_cb)
        opts.addWidget(self.include_empty_cb)
        opts.addWidget(self.allow_dupes_cb)
        opts.addStretch(1)
        layout.addLayout(opts)

        layout.addStretch(1)

    def _build_export_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self._export_tab)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.yolo_output_dir_edit = QtWidgets.QLineEdit()
        self.yolo_output_dir_btn = QtWidgets.QPushButton("Browse…")
        self.yolo_output_dir_btn.clicked.connect(self._browse_yolo_output_dir)
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.yolo_output_dir_edit, 1)
        out_row.addWidget(self.yolo_output_dir_btn)
        form.addRow("Output dir:", out_row)

        self.yolo_dataset_name_edit = QtWidgets.QLineEdit("YOLO_dataset")
        form.addRow("Dataset name:", self.yolo_dataset_name_edit)
        self.yolo_output_dir_edit.editingFinished.connect(self._save_settings)
        self.yolo_dataset_name_edit.editingFinished.connect(
            self._save_settings)

        split_row = QtWidgets.QHBoxLayout()
        self.yolo_val_edit = QtWidgets.QDoubleSpinBox()
        self.yolo_val_edit.setRange(0.0, 0.9)
        self.yolo_val_edit.setSingleStep(0.05)
        self.yolo_val_edit.setValue(0.1)
        self.yolo_test_edit = QtWidgets.QDoubleSpinBox()
        self.yolo_test_edit.setRange(0.0, 0.9)
        self.yolo_test_edit.setSingleStep(0.05)
        self.yolo_test_edit.setValue(0.1)
        split_row.addWidget(QtWidgets.QLabel("val"))
        split_row.addWidget(self.yolo_val_edit)
        split_row.addSpacing(12)
        split_row.addWidget(QtWidgets.QLabel("test"))
        split_row.addWidget(self.yolo_test_edit)
        split_row.addStretch(1)
        form.addRow("Splits:", split_row)

        self.yolo_link_mode = QtWidgets.QComboBox()
        self.yolo_link_mode.addItems(["hardlink", "copy", "symlink"])
        form.addRow("Link mode:", self.yolo_link_mode)

        self.yolo_task_mode = QtWidgets.QComboBox()
        self.yolo_task_mode.addItems(["auto", "segmentation", "pose"])
        form.addRow("Task:", self.yolo_task_mode)

        flags_row = QtWidgets.QHBoxLayout()
        self.yolo_include_empty_cb = QtWidgets.QCheckBox("Include empty JSON")
        self.yolo_keep_staging_cb = QtWidgets.QCheckBox("Keep staging files")
        flags_row.addWidget(self.yolo_include_empty_cb)
        flags_row.addWidget(self.yolo_keep_staging_cb)
        flags_row.addStretch(1)
        form.addRow("Options:", flags_row)

        hint = QtWidgets.QLabel(
            "Robustness:\n"
            "- Missing/deleted JSON or images are skipped.\n"
            "- Images are staged as PNG to satisfy YOLO conversion."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        layout.addStretch(1)

    def _sync_primary_actions(self) -> None:
        tab = self.tabs.currentWidget()
        self.run_btn.setVisible(tab is self._backfill_tab)
        self.export_btn.setVisible(tab is self._export_tab)

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _open_index_file(self) -> None:
        text = self.index_file_edit.text().strip()
        if not text:
            return
        path = Path(text)
        if not path.exists():
            QtWidgets.QMessageBox.information(
                self, "Index file not found", "Index file does not exist yet.")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(path)))

    def _open_index_folder(self) -> None:
        text = self.index_file_edit.text().strip()
        if not text:
            return
        path = Path(text)
        folder = path.parent if path.suffix else path
        if not folder.exists():
            QtWidgets.QMessageBox.information(
                self, "Folder not found", "Index folder does not exist yet.")
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _browse_dataset_root(self) -> None:
        root = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select dataset root")
        if not root:
            return
        self.dataset_root_edit.setText(root)
        self._on_dataset_root_changed()

    def _browse_index_file(self) -> None:
        current = self.index_file_edit.text().strip()
        start_dir = str(Path(current).parent) if current else str(Path.home())
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select index file",
            start_dir,
            "JSONL (*.jsonl);;All files (*)",
        )
        if not path:
            return
        self.index_file_edit.setText(path)
        self._save_settings()
        self._refresh_index_preview()

    def _on_dataset_root_changed(self) -> None:
        root = self.dataset_root_edit.text().strip()
        if root:
            self.index_file_edit.setText(
                str(default_label_index_path(Path(root))))
        self._save_settings()
        self._refresh_index_preview()

    def _add_source_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Add source folder")
        if not folder:
            return
        self.sources_list.addItem(folder)
        self._save_settings()

    def _remove_selected_sources(self) -> None:
        for item in self.sources_list.selectedItems():
            row = self.sources_list.row(item)
            self.sources_list.takeItem(row)
        self._save_settings()

    def _refresh_index_preview(self) -> None:
        text = self.index_file_edit.text().strip()
        if not text:
            self.status_label.setText("")
            return
        index_file = Path(text)
        if index_file.exists():
            self.status_label.setText(f"Index file: {index_file}")
        else:
            self.status_label.setText(
                f"Index file (will be created): {index_file}")

    def _load_settings(self) -> None:
        if self._settings is None:
            return
        index_file = self._settings.value(
            "dataset/label_index_file", "", type=str)
        dataset_root = self._settings.value(
            "dataset/label_collection_dir", "", type=str)
        sources = self._settings.value(
            "dataset/label_index_sources", [], type=list)
        yolo_output_dir = self._settings.value(
            "dataset/label_yolo_output_dir", "", type=str)
        yolo_dataset_name = self._settings.value(
            "dataset/label_yolo_dataset_name", "YOLO_dataset", type=str)
        yolo_task = self._settings.value(
            "dataset/label_yolo_task", "auto", type=str)

        if index_file:
            self.index_file_edit.setText(index_file)
            self.auto_enable.setChecked(True)
        else:
            self.auto_enable.setChecked(False)

        if dataset_root:
            self.dataset_root_edit.setText(dataset_root)
            if not index_file:
                self.index_file_edit.setText(
                    str(default_label_index_path(Path(dataset_root))))

        self.sources_list.clear()
        for src in sources or []:
            if src:
                self.sources_list.addItem(str(src))

        if yolo_output_dir:
            self.yolo_output_dir_edit.setText(yolo_output_dir)
        elif dataset_root:
            self.yolo_output_dir_edit.setText(dataset_root)
        if yolo_dataset_name:
            self.yolo_dataset_name_edit.setText(yolo_dataset_name)
        if yolo_task:
            idx = self.yolo_task_mode.findText(yolo_task)
            if idx >= 0:
                self.yolo_task_mode.setCurrentIndex(idx)

    def _save_settings(self) -> None:
        if self._settings is None:
            return

        dataset_root = self.dataset_root_edit.text().strip()
        index_file = self.index_file_edit.text().strip()
        sources = [self.sources_list.item(i).text()
                   for i in range(self.sources_list.count())]

        if dataset_root:
            self._settings.setValue(
                "dataset/label_collection_dir", dataset_root)

        if self.auto_enable.isChecked() and index_file:
            self._settings.setValue("dataset/label_index_file", index_file)
        else:
            self._settings.setValue("dataset/label_index_file", "")

        self._settings.setValue("dataset/label_index_sources", sources)
        yolo_output_dir = getattr(self, "yolo_output_dir_edit", None)
        if yolo_output_dir is not None:
            value = self.yolo_output_dir_edit.text().strip()
            if value:
                self._settings.setValue("dataset/label_yolo_output_dir", value)
        yolo_name = getattr(self, "yolo_dataset_name_edit", None)
        if yolo_name is not None:
            value = self.yolo_dataset_name_edit.text().strip()
            if value:
                self._settings.setValue(
                    "dataset/label_yolo_dataset_name", value)
        yolo_task = getattr(self, "yolo_task_mode", None)
        if yolo_task is not None:
            value = str(self.yolo_task_mode.currentText())
            if value:
                self._settings.setValue("dataset/label_yolo_task", value)

    def _set_running(self, running: bool) -> None:
        self.progress.setVisible(running)
        self.cancel_btn.setEnabled(running)
        self.run_btn.setEnabled(not running)
        self.add_source_btn.setEnabled(not running)
        self.remove_source_btn.setEnabled(not running)
        self.clear_sources_btn.setEnabled(not running)
        self.export_btn.setEnabled(not running)
        self.yolo_output_dir_btn.setEnabled(not running)

    def _request_stop(self) -> None:
        if self._worker is not None:
            self._worker.request_stop()

    def _browse_yolo_output_dir(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select output directory")
        if not folder:
            return
        self.yolo_output_dir_edit.setText(folder)
        self._save_settings()

    def _start_backfill_job(self) -> None:
        if self._worker is not None or self._thread is not None:
            return

        index_file_text = self.index_file_edit.text().strip()
        if not index_file_text:
            QtWidgets.QMessageBox.warning(
                self, "Missing index file", "Please set an index file.")
            return

        sources = [self.sources_list.item(i).text()
                   for i in range(self.sources_list.count())]
        if not sources:
            QtWidgets.QMessageBox.warning(
                self, "Missing sources", "Please add at least one source folder.")
            return

        job = LabelIndexJob(
            sources=[Path(s) for s in sources],
            index_file=Path(index_file_text),
            recursive=bool(self.recursive_cb.isChecked()),
            include_empty=bool(self.include_empty_cb.isChecked()),
            allow_duplicates=bool(self.allow_dupes_cb.isChecked()),
        )

        self._append_log(f"Index file: {job.index_file}")
        for src in job.sources:
            self._append_log(f"Source: {src}")

        self.progress.setValue(0)
        self._set_running(True)

        self._worker = FlexibleWorker(_run_index_job, job)
        self._thread = QtCore.QThread(self)
        self._worker.moveToThread(self._thread)

        self._worker.progress_signal.connect(self.progress.setValue)
        self._worker.preview_signal.connect(self._on_preview)
        self._worker.finished_signal.connect(self._on_finished)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

    @QtCore.Slot(object)
    def _on_preview(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        processed = payload.get("processed", 0)
        total = payload.get("total", 0)
        appended = payload.get("appended", 0)
        skipped = payload.get("skipped", 0)
        missing = payload.get("missing_image", 0)
        self.status_label.setText(
            f"Processed {processed}/{total} • Appended {appended} • Skipped {skipped} • Missing image {missing}"
        )

    def _start_yolo_job(self) -> None:
        if self._worker is not None or self._thread is not None:
            return

        index_file_text = self.index_file_edit.text().strip()
        if not index_file_text:
            QtWidgets.QMessageBox.warning(
                self, "Missing index file", "Please set an index file.")
            return
        index_path = Path(index_file_text)
        if not index_path.exists():
            QtWidgets.QMessageBox.warning(
                self, "Index file not found", f"Index file does not exist:\n{index_path}")
            return

        output_dir_text = self.yolo_output_dir_edit.text().strip()
        if not output_dir_text:
            QtWidgets.QMessageBox.warning(
                self, "Missing output dir", "Please choose an output directory.")
            return

        dataset_name = self.yolo_dataset_name_edit.text().strip() or "YOLO_dataset"
        val_size = float(self.yolo_val_edit.value())
        test_size = float(self.yolo_test_edit.value())
        link_mode = str(self.yolo_link_mode.currentText())
        task_mode = str(self.yolo_task_mode.currentText())
        include_empty = bool(self.yolo_include_empty_cb.isChecked())
        keep_staging = bool(self.yolo_keep_staging_cb.isChecked())

        self._append_log(f"YOLO output: {output_dir_text}/{dataset_name}")
        self.progress.setValue(0)
        self._set_running(True)

        def task(*, pred_worker=None, stop_event=None):
            return build_yolo_from_label_index(
                index_file=index_path,
                output_dir=Path(output_dir_text),
                dataset_name=dataset_name,
                val_size=val_size,
                test_size=test_size,
                link_mode=link_mode,
                task=task_mode,
                include_empty=include_empty,
                keep_staging=keep_staging,
                pred_worker=pred_worker,
                stop_event=stop_event,
            )

        self._worker = FlexibleWorker(task_function=task)
        self._thread = QtCore.QThread(self)
        self._worker.moveToThread(self._thread)

        self._worker.progress_signal.connect(self.progress.setValue)
        self._worker.finished_signal.connect(self._on_finished)

        self._thread.started.connect(self._worker.run)
        self._thread.start()

    @QtCore.Slot(object)
    def _on_finished(self, result: object) -> None:
        try:
            if isinstance(result, Exception):
                self._append_log(f"Error: {result}")
            else:
                if isinstance(result, dict) and result.get("dataset_dir"):
                    self._append_log(
                        f"YOLO dataset: {result.get('dataset_dir')}")
                    self.progress.setValue(100)
                else:
                    appended = getattr(
                        result, "get", lambda *_: None)("appended") if isinstance(result, dict) else None
                    skipped = getattr(
                        result, "get", lambda *_: None)("skipped") if isinstance(result, dict) else None
                    missing = getattr(
                        result, "get", lambda *_: None)("missing_image") if isinstance(result, dict) else None
                    self._append_log(
                        f"Done. Appended={appended} Skipped={skipped} MissingImage={missing}")
                self._refresh_index_preview()
        finally:
            self._set_running(False)
            if self._thread is not None:
                self._thread.quit()
                self._thread.wait(2000)
            if self._worker is not None:
                self._worker.deleteLater()
            if self._thread is not None:
                self._thread.deleteLater()
            self._worker = None
            self._thread = None
