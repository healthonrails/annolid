from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from qtpy import QtCore, QtGui, QtWidgets

from annolid.datasets.labelme_collection import (
    DEFAULT_LABEL_INDEX_DIRNAME,
    DEFAULT_LABEL_INDEX_NAME,
    build_labelme_spec,
    default_label_index_path,
    infer_labelme_keypoint_names,
    index_labelme_pair,
    iter_labelme_pairs,
    iter_labelme_json_files,
    load_indexed_image_paths,
    resolve_image_path,
    split_labelme_pairs,
    write_labelme_index,
)
from annolid.gui.workers import FlexibleWorker
from annolid.datasets.builders.label_index_yolo import build_yolo_from_label_index
from annolid.gui.widgets.file_audit_widget import FileAuditWidget


@dataclass(frozen=True)
class LabelIndexJob:
    sources: List[Path]
    index_file: Path
    dataset_root: Path
    recursive: bool
    include_empty: bool
    allow_duplicates: bool
    write_spec: bool
    spec_path: Optional[Path]
    val_size: float
    test_size: float
    seed: int
    group_by: str
    group_regex: Optional[str]
    split_dir: str
    split_sources: Optional[dict]
    keypoint_names: Optional[List[str]]
    kpt_dims: int
    infer_flip_idx: bool
    max_keypoint_files: int
    min_keypoint_count: int


def _build_json_list(sources: Sequence[Path], *, recursive: bool) -> List[Path]:
    json_files: List[Path] = []
    for src in sources:
        json_files.extend(list(iter_labelme_json_files(Path(src), recursive=recursive)))
    return sorted(set(json_files))


def _run_index_job(job: LabelIndexJob, *, pred_worker=None, stop_event=None) -> dict:
    index_file = Path(job.index_file).expanduser().resolve()
    sources = [Path(p).expanduser().resolve() for p in job.sources]
    dataset_root = Path(job.dataset_root).expanduser().resolve()

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

    result: dict = {
        "index_file": str(index_file),
        "sources": [str(p) for p in sources],
        "dataset_root": str(dataset_root),
        "recursive": bool(job.recursive),
        "include_empty": bool(job.include_empty),
        "allow_duplicates": bool(job.allow_duplicates),
        "total_json": total,
        "appended": appended,
        "skipped": skipped,
        "missing_image": missing_image,
    }

    if job.write_spec:
        if job.split_sources:
            splits = {
                "train": iter_labelme_pairs(
                    [Path(p) for p in job.split_sources.get("train", [])],
                    recursive=bool(job.recursive),
                    include_empty=bool(job.include_empty),
                ),
                "val": iter_labelme_pairs(
                    [Path(p) for p in job.split_sources.get("val", [])],
                    recursive=bool(job.recursive),
                    include_empty=bool(job.include_empty),
                ),
                "test": iter_labelme_pairs(
                    [Path(p) for p in job.split_sources.get("test", [])],
                    recursive=bool(job.recursive),
                    include_empty=bool(job.include_empty),
                ),
            }
            pairs = splits["train"] + splits["val"] + splits["test"]
        else:
            pairs = iter_labelme_pairs(
                sources,
                recursive=bool(job.recursive),
                include_empty=bool(job.include_empty),
            )
            splits = split_labelme_pairs(
                pairs,
                val_size=float(job.val_size),
                test_size=float(job.test_size),
                seed=int(job.seed),
                group_by=str(job.group_by),
                group_regex=(str(job.group_regex) if job.group_regex else None),
            )
        split_dir = dataset_root / str(job.split_dir or DEFAULT_LABEL_INDEX_DIRNAME)
        split_dir.mkdir(parents=True, exist_ok=True)
        train_index = split_dir / "labelme_train.jsonl"
        val_index = split_dir / "labelme_val.jsonl"
        test_index = split_dir / "labelme_test.jsonl"

        write_labelme_index(splits.get("train", []), index_file=train_index)
        if splits.get("val"):
            write_labelme_index(splits.get("val", []), index_file=val_index)
        else:
            val_index = None
        if splits.get("test"):
            write_labelme_index(splits.get("test", []), index_file=test_index)
        else:
            test_index = None

        keypoint_names = list(job.keypoint_names or [])
        if not keypoint_names:
            keypoint_names = infer_labelme_keypoint_names(
                pairs,
                max_files=int(job.max_keypoint_files),
                min_count=int(job.min_keypoint_count),
            )
        if not keypoint_names:
            raise ValueError(
                "Could not infer keypoint names from LabelMe JSONs. "
                "Provide keypoint names in the dialog."
            )

        flip_idx = None
        if bool(job.infer_flip_idx):
            from annolid.segmentation.dino_kpseg.keypoints import (
                infer_flip_idx_from_names,
            )

            flip_idx = infer_flip_idx_from_names(
                keypoint_names, kpt_count=len(keypoint_names)
            )

        spec_out = job.spec_path or (dataset_root / "labelme_spec.yaml")
        spec_path = build_labelme_spec(
            dataset_root=dataset_root,
            train_index=train_index if splits.get("train") else None,
            val_index=val_index,
            test_index=test_index,
            keypoint_names=keypoint_names,
            kpt_dims=int(job.kpt_dims),
            flip_idx=flip_idx,
            output_yaml=Path(spec_out),
        )
        result["spec_path"] = str(spec_path)
        result["split_counts"] = {
            "train": len(splits.get("train", [])),
            "val": len(splits.get("val", [])),
            "test": len(splits.get("test", [])),
        }

    return result


class LabelCollectionDialog(QtWidgets.QDialog):
    """GUI for indexing LabelMe PNG/JSON pairs into a JSONL dataset index."""

    def __init__(self, *, settings: Optional[QtCore.QSettings] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Collect Labels (Dataset Index)")
        self.setMinimumWidth(700)
        self.setMinimumHeight(520)
        self.resize(820, 720)

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

        scroll = QtWidgets.QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        layout.addWidget(scroll, 1)

        container = QtWidgets.QWidget(scroll)
        content = QtWidgets.QVBoxLayout(container)

        header = QtWidgets.QLabel(
            "Build a central dataset index without copying files.\n"
            f"Default index location: {DEFAULT_LABEL_INDEX_DIRNAME}/{DEFAULT_LABEL_INDEX_NAME}"
        )
        header.setWordWrap(True)
        content.addWidget(header)

        self.tabs = QtWidgets.QTabWidget(self)
        content.addWidget(self.tabs)

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
        content.addWidget(self.status_label)

        self.progress = QtWidgets.QProgressBar(self)
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setVisible(False)
        content.addWidget(self.progress)

        self.log = QtWidgets.QPlainTextEdit(self)
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(120)
        content.addWidget(self.log)

        content.addStretch(1)
        scroll.setWidget(container)

        buttons = QtWidgets.QDialogButtonBox(self)
        self.close_btn = buttons.addButton(
            "Close", QtWidgets.QDialogButtonBox.RejectRole
        )
        self.close_btn.clicked.connect(self.reject)

        self.run_btn = buttons.addButton(
            "Collect Now", QtWidgets.QDialogButtonBox.AcceptRole
        )
        self.run_btn.clicked.connect(self._start_backfill_job)

        self.export_btn = buttons.addButton(
            "Convert to YOLO", QtWidgets.QDialogButtonBox.ActionRole
        )
        self.export_btn.clicked.connect(self._start_yolo_job)

        self.cancel_btn = buttons.addButton(
            "Cancel", QtWidgets.QDialogButtonBox.DestructiveRole
        )
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

        self.dataset_root_edit.editingFinished.connect(self._on_dataset_root_changed)
        self.index_file_edit.editingFinished.connect(self._save_settings)

        layout.addStretch(1)

    def _build_backfill_tab(self) -> None:
        layout = QtWidgets.QVBoxLayout(self._backfill_tab)

        self.sources_list = QtWidgets.QListWidget(self)
        self.sources_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
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

        # Audit Widget
        self.audit_layout = QtWidgets.QVBoxLayout()
        self.scan_btn = QtWidgets.QPushButton("Scan / Preview Files")
        self.scan_btn.clicked.connect(self._scan_sources)
        self.audit_layout.addWidget(self.scan_btn)

        self.audit_widget = FileAuditWidget()
        self.audit_layout.addWidget(self.audit_widget)

        layout.addLayout(self.audit_layout)

        spec_group = QtWidgets.QGroupBox("LabelMe spec.yaml (optional)")
        spec_form = QtWidgets.QFormLayout(spec_group)

        self.spec_enable_cb = QtWidgets.QCheckBox(
            "Generate spec.yaml + train/val/test splits"
        )
        self.spec_enable_cb.setChecked(False)
        self.spec_enable_cb.stateChanged.connect(self._sync_spec_controls)
        spec_form.addRow(self.spec_enable_cb)

        self.spec_path_edit = QtWidgets.QLineEdit()
        self.spec_path_btn = QtWidgets.QPushButton("Browse…")
        self.spec_path_btn.clicked.connect(self._browse_spec_path)
        spec_row = QtWidgets.QHBoxLayout()
        spec_row.addWidget(self.spec_path_edit, 1)
        spec_row.addWidget(self.spec_path_btn)
        spec_form.addRow("Spec path:", spec_row)

        self.spec_split_dir_edit = QtWidgets.QLineEdit(DEFAULT_LABEL_INDEX_DIRNAME)
        spec_form.addRow("Split JSONL dir:", self.spec_split_dir_edit)

        split_row = QtWidgets.QHBoxLayout()
        self.spec_val_edit = QtWidgets.QDoubleSpinBox()
        self.spec_val_edit.setRange(0.0, 0.9)
        self.spec_val_edit.setSingleStep(0.05)
        self.spec_val_edit.setValue(0.1)
        self.spec_test_edit = QtWidgets.QDoubleSpinBox()
        self.spec_test_edit.setRange(0.0, 0.9)
        self.spec_test_edit.setSingleStep(0.05)
        self.spec_test_edit.setValue(0.0)
        split_row.addWidget(QtWidgets.QLabel("val"))
        split_row.addWidget(self.spec_val_edit)
        split_row.addSpacing(12)
        split_row.addWidget(QtWidgets.QLabel("test"))
        split_row.addWidget(self.spec_test_edit)
        split_row.addStretch(1)
        spec_form.addRow("Splits:", split_row)

        self.spec_seed_spin = QtWidgets.QSpinBox()
        self.spec_seed_spin.setRange(0, 1000000)
        self.spec_seed_spin.setValue(0)
        spec_form.addRow("Seed:", self.spec_seed_spin)

        self.spec_group_by_combo = QtWidgets.QComboBox()
        self.spec_group_by_combo.addItems(
            ["parent", "grandparent", "stem_prefix", "regex", "none"]
        )
        spec_form.addRow("Group by:", self.spec_group_by_combo)
        self.spec_group_regex_edit = QtWidgets.QLineEdit()
        self.spec_group_regex_edit.setPlaceholderText(
            "Optional regex (for group_by=regex)"
        )
        spec_form.addRow("Group regex:", self.spec_group_regex_edit)

        self.spec_keypoint_names_edit = QtWidgets.QLineEdit()
        self.spec_keypoint_names_edit.setPlaceholderText(
            "nose, leftear, rightear, tailbase"
        )
        spec_form.addRow("Keypoint names:", self.spec_keypoint_names_edit)

        self.spec_kpt_dims_combo = QtWidgets.QComboBox()
        self.spec_kpt_dims_combo.addItems(["3", "2"])
        spec_form.addRow("Keypoint dims:", self.spec_kpt_dims_combo)

        self.spec_infer_flip_cb = QtWidgets.QCheckBox("Infer flip_idx from names")
        self.spec_infer_flip_cb.setChecked(True)
        spec_form.addRow("Flip index:", self.spec_infer_flip_cb)

        self.spec_max_files_spin = QtWidgets.QSpinBox()
        self.spec_max_files_spin.setRange(1, 100000)
        self.spec_max_files_spin.setValue(500)
        spec_form.addRow("Max JSONs to scan:", self.spec_max_files_spin)

        self.spec_min_count_spin = QtWidgets.QSpinBox()
        self.spec_min_count_spin.setRange(1, 1000)
        self.spec_min_count_spin.setValue(1)
        spec_form.addRow("Min keypoint count:", self.spec_min_count_spin)

        layout.addWidget(spec_group)
        self._sync_spec_controls()

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
        self.yolo_dataset_name_edit.editingFinished.connect(self._save_settings)

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

    def _sync_spec_controls(self) -> None:
        enabled = bool(self.spec_enable_cb.isChecked())
        for widget in (
            self.spec_path_edit,
            self.spec_path_btn,
            self.spec_split_dir_edit,
            self.spec_val_edit,
            self.spec_test_edit,
            self.spec_seed_spin,
            self.spec_group_by_combo,
            self.spec_group_regex_edit,
            self.spec_keypoint_names_edit,
            self.spec_kpt_dims_combo,
            self.spec_infer_flip_cb,
            self.spec_max_files_spin,
            self.spec_min_count_spin,
        ):
            widget.setEnabled(enabled)

    def _browse_spec_path(self) -> None:
        dataset_root = self.dataset_root_edit.text().strip()
        start_dir = dataset_root or str(Path.home())
        path, _filter = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Select spec.yaml",
            str(Path(start_dir) / "labelme_spec.yaml"),
            "YAML (*.yaml *.yml);;All files (*)",
        )
        if not path:
            return
        self.spec_path_edit.setText(path)
        self._save_settings()

    def _append_log(self, text: str) -> None:
        self.log.appendPlainText(text)

    def _open_index_file(self) -> None:
        text = self.index_file_edit.text().strip()
        if not text:
            return
        path = Path(text)
        if not path.exists():
            QtWidgets.QMessageBox.information(
                self, "Index file not found", "Index file does not exist yet."
            )
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
                self, "Folder not found", "Index folder does not exist yet."
            )
            return
        QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(folder)))

    def _browse_dataset_root(self) -> None:
        root = QtWidgets.QFileDialog.getExistingDirectory(self, "Select dataset root")
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
            self.index_file_edit.setText(str(default_label_index_path(Path(root))))
        self._save_settings()
        self._refresh_index_preview()

    def _add_source_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Add source folder")
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
            self.status_label.setText(f"Index file (will be created): {index_file}")

    def _load_settings(self) -> None:
        if self._settings is None:
            return
        index_file = self._settings.value("dataset/label_index_file", "", type=str)
        dataset_root = self._settings.value(
            "dataset/label_collection_dir", "", type=str
        )
        sources = self._settings.value("dataset/label_index_sources", [], type=list)
        yolo_output_dir = self._settings.value(
            "dataset/label_yolo_output_dir", "", type=str
        )
        yolo_dataset_name = self._settings.value(
            "dataset/label_yolo_dataset_name", "YOLO_dataset", type=str
        )
        yolo_task = self._settings.value("dataset/label_yolo_task", "auto", type=str)
        spec_enabled = self._settings.value(
            "dataset/label_spec_enabled", False, type=bool
        )
        spec_path = self._settings.value("dataset/label_spec_path", "", type=str)
        spec_split_dir = self._settings.value(
            "dataset/label_spec_split_dir", DEFAULT_LABEL_INDEX_DIRNAME, type=str
        )
        spec_val = self._settings.value("dataset/label_spec_val", 0.1, type=float)
        spec_test = self._settings.value("dataset/label_spec_test", 0.0, type=float)
        spec_seed = self._settings.value("dataset/label_spec_seed", 0, type=int)
        spec_group_by = self._settings.value(
            "dataset/label_spec_group_by", "parent", type=str
        )
        spec_group_regex = self._settings.value(
            "dataset/label_spec_group_regex", "", type=str
        )
        spec_keypoint_names = self._settings.value(
            "dataset/label_spec_keypoint_names", "", type=str
        )
        spec_kpt_dims = self._settings.value(
            "dataset/label_spec_kpt_dims", "3", type=str
        )
        spec_infer_flip = self._settings.value(
            "dataset/label_spec_infer_flip", True, type=bool
        )
        spec_max_files = self._settings.value(
            "dataset/label_spec_max_files", 500, type=int
        )
        spec_min_count = self._settings.value(
            "dataset/label_spec_min_count", 1, type=int
        )

        if index_file:
            self.index_file_edit.setText(index_file)
            self.auto_enable.setChecked(True)
        else:
            self.auto_enable.setChecked(False)

        if dataset_root:
            self.dataset_root_edit.setText(dataset_root)
            if not index_file:
                self.index_file_edit.setText(
                    str(default_label_index_path(Path(dataset_root)))
                )

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

        self.spec_enable_cb.setChecked(bool(spec_enabled))
        if spec_path:
            self.spec_path_edit.setText(spec_path)
        if spec_split_dir:
            self.spec_split_dir_edit.setText(spec_split_dir)
        self.spec_val_edit.setValue(float(spec_val))
        self.spec_test_edit.setValue(float(spec_test))
        self.spec_seed_spin.setValue(int(spec_seed))
        idx = self.spec_group_by_combo.findText(str(spec_group_by))
        if idx >= 0:
            self.spec_group_by_combo.setCurrentIndex(idx)
        if spec_group_regex:
            self.spec_group_regex_edit.setText(spec_group_regex)
        if spec_keypoint_names:
            self.spec_keypoint_names_edit.setText(spec_keypoint_names)
        if spec_kpt_dims:
            idx = self.spec_kpt_dims_combo.findText(str(spec_kpt_dims))
            if idx >= 0:
                self.spec_kpt_dims_combo.setCurrentIndex(idx)
        self.spec_infer_flip_cb.setChecked(bool(spec_infer_flip))
        self.spec_max_files_spin.setValue(int(spec_max_files))
        self.spec_min_count_spin.setValue(int(spec_min_count))
        self._sync_spec_controls()

    def _save_settings(self) -> None:
        if self._settings is None:
            return

        dataset_root = self.dataset_root_edit.text().strip()
        index_file = self.index_file_edit.text().strip()
        sources = [
            self.sources_list.item(i).text() for i in range(self.sources_list.count())
        ]

        if dataset_root:
            self._settings.setValue("dataset/label_collection_dir", dataset_root)

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
                self._settings.setValue("dataset/label_yolo_dataset_name", value)
        yolo_task = getattr(self, "yolo_task_mode", None)
        if yolo_task is not None:
            value = str(self.yolo_task_mode.currentText())
            if value:
                self._settings.setValue("dataset/label_yolo_task", value)

        self._settings.setValue(
            "dataset/label_spec_enabled", bool(self.spec_enable_cb.isChecked())
        )
        self._settings.setValue(
            "dataset/label_spec_path", self.spec_path_edit.text().strip()
        )
        self._settings.setValue(
            "dataset/label_spec_split_dir", self.spec_split_dir_edit.text().strip()
        )
        self._settings.setValue(
            "dataset/label_spec_val", float(self.spec_val_edit.value())
        )
        self._settings.setValue(
            "dataset/label_spec_test", float(self.spec_test_edit.value())
        )
        self._settings.setValue(
            "dataset/label_spec_seed", int(self.spec_seed_spin.value())
        )
        self._settings.setValue(
            "dataset/label_spec_group_by", str(self.spec_group_by_combo.currentText())
        )
        self._settings.setValue(
            "dataset/label_spec_group_regex", self.spec_group_regex_edit.text().strip()
        )
        self._settings.setValue(
            "dataset/label_spec_keypoint_names",
            self.spec_keypoint_names_edit.text().strip(),
        )
        self._settings.setValue(
            "dataset/label_spec_kpt_dims", str(self.spec_kpt_dims_combo.currentText())
        )
        self._settings.setValue(
            "dataset/label_spec_infer_flip", bool(self.spec_infer_flip_cb.isChecked())
        )
        self._settings.setValue(
            "dataset/label_spec_max_files", int(self.spec_max_files_spin.value())
        )
        self._settings.setValue(
            "dataset/label_spec_min_count", int(self.spec_min_count_spin.value())
        )

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
            self, "Select output directory"
        )
        if not folder:
            return
        self.yolo_output_dir_edit.setText(folder)
        self._save_settings()

    def _start_backfill_job(self) -> None:
        if self._worker is not None or self._thread is not None:
            return

        index_file_text = self.index_file_edit.text().strip()

        sources = [
            self.sources_list.item(i).text() for i in range(self.sources_list.count())
        ]
        if not sources:
            QtWidgets.QMessageBox.warning(
                self, "Missing sources", "Please add at least one source folder."
            )
            return

        source_paths = [Path(s) for s in sources]
        dataset_root_text = self.dataset_root_edit.text().strip()
        dataset_root = (
            Path(dataset_root_text).expanduser()
            if dataset_root_text
            else self._infer_dataset_root(source_paths)
        )
        if dataset_root is not None and not dataset_root_text:
            self.dataset_root_edit.setText(str(dataset_root))

        if not index_file_text:
            if dataset_root is None:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Missing dataset root",
                    "Please set a dataset root or index file.",
                )
                return
            index_path = default_label_index_path(dataset_root)
            self.index_file_edit.setText(str(index_path))
            index_file_text = str(index_path)
            self.auto_enable.setChecked(True)
            self._save_settings()
            self._refresh_index_preview()

        split_sources = None
        if self.spec_enable_cb.isChecked():
            split_sources = self._infer_split_sources(source_paths)

        job = LabelIndexJob(
            sources=source_paths,
            index_file=Path(index_file_text),
            dataset_root=Path(
                self.dataset_root_edit.text().strip() or Path(index_file_text).parent
            ),
            recursive=bool(self.recursive_cb.isChecked()),
            include_empty=bool(self.include_empty_cb.isChecked()),
            allow_duplicates=bool(self.allow_dupes_cb.isChecked()),
            write_spec=bool(self.spec_enable_cb.isChecked()),
            spec_path=(
                Path(self.spec_path_edit.text().strip()).expanduser()
                if self.spec_path_edit.text().strip()
                else None
            ),
            val_size=float(self.spec_val_edit.value()),
            test_size=float(self.spec_test_edit.value()),
            seed=int(self.spec_seed_spin.value()),
            group_by=str(self.spec_group_by_combo.currentText()),
            group_regex=self.spec_group_regex_edit.text().strip() or None,
            split_dir=str(
                self.spec_split_dir_edit.text().strip() or DEFAULT_LABEL_INDEX_DIRNAME
            ),
            split_sources=split_sources,
            keypoint_names=[
                n.strip()
                for n in self.spec_keypoint_names_edit.text().split(",")
                if n.strip()
            ]
            or None,
            kpt_dims=int(self.spec_kpt_dims_combo.currentText()),
            infer_flip_idx=bool(self.spec_infer_flip_cb.isChecked()),
            max_keypoint_files=int(self.spec_max_files_spin.value()),
            min_keypoint_count=int(self.spec_min_count_spin.value()),
        )

        self._append_log(f"Index file: {job.index_file}")
        for src in job.sources:
            self._append_log(f"Source: {src}")
        if job.write_spec:
            self._append_log(
                f"Spec output: {job.spec_path or (job.dataset_root / 'labelme_spec.yaml')}"
            )
            if split_sources:
                split_summary = ", ".join(
                    f"{key}={len(value)}"
                    for key, value in split_sources.items()
                    if value
                )
                if split_summary:
                    self._append_log(f"Using split folders: {split_summary}")

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
                self, "Missing index file", "Please set an index file."
            )
            return
        index_path = Path(index_file_text)
        if not index_path.exists():
            QtWidgets.QMessageBox.warning(
                self,
                "Index file not found",
                f"Index file does not exist:\n{index_path}",
            )
            return

        output_dir_text = self.yolo_output_dir_edit.text().strip()
        if not output_dir_text:
            QtWidgets.QMessageBox.warning(
                self, "Missing output dir", "Please choose an output directory."
            )
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

    def _infer_dataset_root(self, sources: Sequence[Path]) -> Optional[Path]:
        if not sources:
            return None
        try:
            common = os.path.commonpath([str(Path(s).resolve()) for s in sources])
        except (ValueError, OSError):
            return Path(sources[0]).resolve()
        return Path(common)

    def _infer_split_sources(self, sources: Sequence[Path]) -> Optional[dict]:
        if not sources:
            return None
        buckets = {"train": [], "val": [], "test": []}
        unmatched: List[Path] = []
        for src in sources:
            name = Path(src).name.lower()
            if "train" in name:
                buckets["train"].append(Path(src))
            elif "val" in name or "valid" in name:
                buckets["val"].append(Path(src))
            elif "test" in name:
                buckets["test"].append(Path(src))
            else:
                unmatched.append(Path(src))
        if any(buckets.values()):
            if unmatched:
                buckets["train"].extend(unmatched)
            return buckets
        return None

    @QtCore.Slot(object)
    def _on_finished(self, result: object) -> None:
        try:
            if isinstance(result, Exception):
                self._append_log(f"Error: {result}")
            else:
                if isinstance(result, dict) and result.get("dataset_dir"):
                    self._append_log(f"YOLO dataset: {result.get('dataset_dir')}")
                    self.progress.setValue(100)
                else:
                    appended = (
                        getattr(result, "get", lambda *_: None)("appended")
                        if isinstance(result, dict)
                        else None
                    )
                    skipped = (
                        getattr(result, "get", lambda *_: None)("skipped")
                        if isinstance(result, dict)
                        else None
                    )
                    missing = (
                        getattr(result, "get", lambda *_: None)("missing_image")
                        if isinstance(result, dict)
                        else None
                    )
                    self._append_log(
                        f"Done. Appended={appended} Skipped={skipped} MissingImage={missing}"
                    )
                    if isinstance(result, dict) and result.get("spec_path"):
                        self._append_log(f"LabelMe spec: {result.get('spec_path')}")
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

    def _scan_sources(self) -> None:
        """Scan selected sources and populate audit widget."""
        sources = [
            self.sources_list.item(i).text() for i in range(self.sources_list.count())
        ]
        if not sources:
            QtWidgets.QMessageBox.warning(
                self, "No sources", "Please add at least one source folder."
            )
            return

        recursive = self.recursive_cb.isChecked()

        # Use existing helper to find files
        json_files = _build_json_list([Path(s) for s in sources], recursive=recursive)

        audit_items = []
        import json

        for jf in json_files:
            status = "valid"
            shapes_len = 0
            image_abs_path = None

            try:
                # Check for corresponding image
                image_path = resolve_image_path(jf)
                image_abs_path = image_path
                if image_path is None:
                    status = "missing_image"

                with open(jf, "r") as f:
                    data = json.load(f)

                shapes = data.get("shapes", [])
                shapes_len = len(shapes)
                if not shapes:
                    if status == "valid":
                        status = "empty"

            except Exception:
                status = "error"

            audit_items.append(
                {
                    "json_path": jf,
                    "image_path": image_abs_path,
                    "status": status,
                    "shape_count": shapes_len,
                }
            )

        self.audit_widget.set_items(audit_items)
