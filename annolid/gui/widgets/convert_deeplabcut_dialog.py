from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

from qtpy import QtCore, QtWidgets

from annolid.domain import DeepLabCutTrainingImportConfig
from annolid.gui.cursor_utils import set_widget_busy_cursor
from annolid.services.export import import_deeplabcut_dataset
from annolid.services.export import (
    export_pose_dataset,
    find_deeplabcut_analysis_file,
    import_deeplabcut_analysis,
)


class _ConversionWorker(QtCore.QObject):
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)

    def __init__(self, fn, *args, **kwargs) -> None:
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    @QtCore.Slot()
    def run(self) -> None:
        try:
            result = self._fn(*self._args, **self._kwargs)
            if isinstance(result, dict):
                payload = result
            elif result is None:
                payload = {}
            else:
                payload = {"result": str(result)}
            self.finished.emit(payload)
        except Exception as exc:
            self.failed.emit(str(exc))


class ConvertDLCDialog(QtWidgets.QDialog):
    """Import from or export to DeepLabCut, Lightning Pose, and SimBA-compatible formats."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_ConversionWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("DeepLabCut / Lightning Pose Import & Export")
        self.setModal(True)
        self.resize(700, 440)

        layout = QtWidgets.QVBoxLayout(self)

        # Mode selection
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem(
            "Import DLC / Lightning Pose analysis (CSV / H5 → LabelMe)"
        )
        self.mode_combo.addItem(
            "Import DLC / Lightning Pose training data (labeled-data/CollectedData_*.csv/.h5)"
        )
        self.mode_combo.addItem(
            "Export Annolid LabelMe / Tracking → DeepLabCut / Lightning Pose / SimBA"
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        self.stack.addWidget(self._build_analysis_page())
        self.stack.addWidget(self._build_training_page())
        self.stack.addWidget(self._build_export_page())

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        btn_row.addStretch(1)
        self.run_btn = QtWidgets.QPushButton("Run")
        self.run_btn.clicked.connect(self._run)
        self.run_btn.setEnabled(False)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        btn_row.addWidget(self.run_btn)
        btn_row.addWidget(self.close_btn)
        layout.addLayout(btn_row)

        self._on_mode_changed()

    def _build_analysis_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)

        self.csv_path_edit = QtWidgets.QLineEdit()
        self.csv_path_edit.setPlaceholderText(
            "DeepLabCut / Lightning Pose .csv or .h5 file (auto-detected if video is selected)"
        )
        csv_row = QtWidgets.QHBoxLayout()
        csv_row.addWidget(self.csv_path_edit, 1)
        csv_browse = QtWidgets.QPushButton("Browse…")
        csv_browse.clicked.connect(self._browse_dlc_csv)
        csv_row.addWidget(csv_browse)
        form.addRow("DLC CSV / H5:", csv_row)

        self.video_path_edit = QtWidgets.QLineEdit()
        self.video_path_edit.setPlaceholderText(
            "Optional video file (used for image size & auto-finding *DLC_*.csv/.h5)"
        )
        video_row = QtWidgets.QHBoxLayout()
        video_row.addWidget(self.video_path_edit, 1)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse_video)
        video_row.addWidget(browse)
        form.addRow("Video file (optional):", video_row)

        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText(
            "Output directory (defaults to <video stem>/ or <csv stem>_labelme/)"
        )
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.output_dir_edit, 1)
        out_browse = QtWidgets.QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_output_dir)
        out_row.addWidget(out_browse)
        form.addRow("Output dir:", out_row)

        self.min_likelihood_spin = QtWidgets.QDoubleSpinBox()
        self.min_likelihood_spin.setRange(0.0, 1.0)
        self.min_likelihood_spin.setSingleStep(0.05)
        self.min_likelihood_spin.setDecimals(2)
        self.min_likelihood_spin.setSpecialValueText("Off (keep all finite points)")
        self.min_likelihood_spin.setValue(0.0)
        form.addRow("Min likelihood:", self.min_likelihood_spin)

        self.multi_animal_check = QtWidgets.QCheckBox(
            "Multi-animal hint (auto-detected from 3- vs 4-row header)"
        )
        form.addRow("", self.multi_animal_check)

        self.analysis_pose_schema_check = QtWidgets.QCheckBox(
            "Write pose_schema.json (uses config.yaml skeleton when found)"
        )
        self.analysis_pose_schema_check.setChecked(True)
        form.addRow("", self.analysis_pose_schema_check)

        self.analysis_save_frames_check = QtWidgets.QCheckBox(
            "Extract matching PNG frames from video"
        )
        self.analysis_save_frames_check.setChecked(False)
        form.addRow("", self.analysis_save_frames_check)

        self.csv_path_edit.textChanged.connect(self._update_enabled_state)
        self.video_path_edit.textChanged.connect(self._update_enabled_state)
        return page

    def _build_training_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)

        self.labeled_data_edit = QtWidgets.QLineEdit()
        self.labeled_data_edit.setReadOnly(True)
        ld_row = QtWidgets.QHBoxLayout()
        ld_row.addWidget(self.labeled_data_edit, 1)
        ld_browse = QtWidgets.QPushButton("Browse…")
        ld_browse.clicked.connect(self._browse_labeled_data)
        ld_row.addWidget(ld_browse)
        form.addRow("labeled-data folder:", ld_row)

        self.instance_label_edit = QtWidgets.QLineEdit("mouse")
        form.addRow("Instance label:", self.instance_label_edit)

        self.overwrite_check = QtWidgets.QCheckBox("Overwrite existing JSON files")
        self.overwrite_check.setChecked(False)
        form.addRow("", self.overwrite_check)

        self.recursive_check = QtWidgets.QCheckBox("Scan subfolders recursively")
        self.recursive_check.setChecked(True)
        form.addRow("", self.recursive_check)

        self.write_pose_schema_check = QtWidgets.QCheckBox(
            "Write pose schema (pose_schema.json)"
        )
        self.write_pose_schema_check.setChecked(True)
        self.write_pose_schema_check.toggled.connect(self._update_enabled_state)
        form.addRow("", self.write_pose_schema_check)

        self.pose_schema_out_edit = QtWidgets.QLineEdit()
        self.pose_schema_out_edit.setPlaceholderText(
            "pose_schema.json output path (defaults to <labeled-data>/pose_schema.json)"
        )
        ps_row = QtWidgets.QHBoxLayout()
        ps_row.addWidget(self.pose_schema_out_edit, 1)
        ps_browse = QtWidgets.QPushButton("Browse…")
        ps_browse.clicked.connect(self._browse_pose_schema_out)
        ps_row.addWidget(ps_browse)
        form.addRow("Pose schema:", ps_row)

        self.pose_schema_preset_combo = QtWidgets.QComboBox()
        self.pose_schema_preset_combo.addItems(["auto", "mouse", "none"])
        form.addRow("Edge preset:", self.pose_schema_preset_combo)

        self.instance_separator_edit = QtWidgets.QLineEdit("_")
        self.instance_separator_edit.setMaxLength(4)
        form.addRow("Instance separator:", self.instance_separator_edit)

        self.instance_label_edit.textChanged.connect(self._update_enabled_state)
        self.labeled_data_edit.textChanged.connect(self._update_enabled_state)
        self.pose_schema_out_edit.textChanged.connect(self._update_enabled_state)

        return page

    def _build_export_page(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(page)

        self.export_source_edit = QtWidgets.QLineEdit()
        self.export_source_edit.setPlaceholderText(
            "Select a LabelMe annotation folder or Annolid *_tracking.csv file…"
        )
        src_row = QtWidgets.QHBoxLayout()
        src_row.addWidget(self.export_source_edit, 1)
        src_dir_btn = QtWidgets.QPushButton("Folder…")
        src_dir_btn.clicked.connect(self._browse_export_source_dir)
        src_file_btn = QtWidgets.QPushButton("File…")
        src_file_btn.clicked.connect(self._browse_export_source_file)
        src_row.addWidget(src_dir_btn)
        src_row.addWidget(src_file_btn)
        form.addRow("Annolid source:", src_row)

        self.export_format_combo = QtWidgets.QComboBox()
        self.export_format_combo.addItem(
            "DeepLabCut / Lightning Pose / SimBA Analysis CSV (.csv)",
            "deeplabcut-csv",
        )
        self.export_format_combo.addItem(
            "DeepLabCut / Lightning Pose Training Project (labeled-data + config.yaml)",
            "deeplabcut-project",
        )
        self.export_format_combo.currentIndexChanged.connect(
            self._on_export_format_changed
        )
        form.addRow("Export format:", self.export_format_combo)

        self.export_output_edit = QtWidgets.QLineEdit()
        self.export_output_edit.setPlaceholderText(
            "Select output .csv file or project folder…"
        )
        exp_out_row = QtWidgets.QHBoxLayout()
        exp_out_row.addWidget(self.export_output_edit, 1)
        exp_out_browse = QtWidgets.QPushButton("Browse…")
        exp_out_browse.clicked.connect(self._browse_export_output)
        exp_out_row.addWidget(exp_out_browse)
        form.addRow("Output path:", exp_out_row)

        self.export_scorer_edit = QtWidgets.QLineEdit("annolid")
        form.addRow("Scorer name:", self.export_scorer_edit)

        self.export_schema_edit = QtWidgets.QLineEdit()
        self.export_schema_edit.setPlaceholderText(
            "Optional pose_schema.json (auto-detected if present in source folder)"
        )
        sch_row = QtWidgets.QHBoxLayout()
        sch_row.addWidget(self.export_schema_edit, 1)
        sch_btn = QtWidgets.QPushButton("Browse…")
        sch_btn.clicked.connect(self._browse_export_schema)
        sch_row.addWidget(sch_btn)
        form.addRow("Pose schema:", sch_row)

        self.export_copy_images_check = QtWidgets.QCheckBox(
            "Copy sidecar images into labeled-data/ (training project export)"
        )
        self.export_copy_images_check.setChecked(True)
        form.addRow("", self.export_copy_images_check)

        self.export_source_edit.textChanged.connect(self._update_enabled_state)
        self.export_output_edit.textChanged.connect(self._update_enabled_state)
        self._on_export_format_changed()
        return page

    def _on_mode_changed(self) -> None:
        self.stack.setCurrentIndex(self.mode_combo.currentIndex())
        self._update_enabled_state()

    def _on_export_format_changed(self) -> None:
        fmt = str(self.export_format_combo.currentData() or "deeplabcut-csv")
        is_project = fmt == "deeplabcut-project"
        self.export_copy_images_check.setEnabled(is_project)
        src = self.export_source_edit.text().strip()
        if src and not self.export_output_edit.text().strip():
            p = Path(src).expanduser()
            if is_project:
                self.export_output_edit.setText(str(p.parent / f"{p.stem}_dlc_project"))
            else:
                self.export_output_edit.setText(str(p.parent / f"{p.stem}_dlc.csv"))

    def _browse_dlc_csv(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select DeepLabCut / Lightning Pose CSV or H5",
            str(Path.home()),
            "DeepLabCut Files (*.csv *.h5 *.hdf5);;CSV Files (*.csv);;HDF5 Files (*.h5 *.hdf5);;All files (*)",
        )
        if not path:
            return
        self.csv_path_edit.setText(path)
        if not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(
                str(Path(path).parent / f"{Path(path).stem}_labelme")
            )
        self._update_enabled_state()

    def _browse_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video file",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if not path:
            return
        self.video_path_edit.setText(path)
        if not self.csv_path_edit.text().strip():
            matched = find_deeplabcut_analysis_file(path)
            if matched is not None:
                self.csv_path_edit.setText(str(matched))
        if not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(str(Path(path).with_suffix("")))
        self._update_enabled_state()

    def _browse_output_dir(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select output directory",
            self.output_dir_edit.text().strip() or str(Path.home()),
        )
        if folder:
            self.output_dir_edit.setText(folder)

    def _browse_labeled_data(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select DeepLabCut labeled-data folder",
            str(Path.home()),
        )
        if not folder:
            return
        self.labeled_data_edit.setText(folder)
        if not self.pose_schema_out_edit.text().strip():
            self.pose_schema_out_edit.setText(str(Path(folder) / "pose_schema.json"))
        self._update_enabled_state()

    def _browse_pose_schema_out(self) -> None:
        start_dir = self.pose_schema_out_edit.text().strip()
        if not start_dir:
            ld = self.labeled_data_edit.text().strip()
            start_dir = ld or str(Path.home())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save pose schema",
            start_dir,
            "Pose schema (*.json *.yaml *.yml);;All files (*)",
        )
        if path:
            self.pose_schema_out_edit.setText(path)

    def _browse_export_source_dir(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Annolid LabelMe annotation folder",
            self.export_source_edit.text().strip() or str(Path.home()),
        )
        if not folder:
            return
        self.export_source_edit.setText(folder)
        self._on_export_format_changed()
        self._update_enabled_state()

    def _browse_export_source_file(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select Annolid Tracking CSV or COCO JSON",
            str(Path.home()),
            "Annolid/COCO (*.csv *.json);;All files (*)",
        )
        if not path:
            return
        self.export_source_edit.setText(path)
        self._on_export_format_changed()
        self._update_enabled_state()

    def _browse_export_output(self) -> None:
        fmt = str(self.export_format_combo.currentData() or "deeplabcut-csv")
        base = self.export_output_edit.text().strip() or str(Path.home())
        if fmt == "deeplabcut-project":
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self, "Select output DLC project folder", base
            )
            if folder:
                self.export_output_edit.setText(folder)
        else:
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self,
                "Save DeepLabCut / Lightning Pose CSV",
                base,
                "CSV Files (*.csv);;All files (*)",
            )
            if path:
                self.export_output_edit.setText(path)

    def _browse_export_schema(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select PoseSchema file",
            str(Path.home()),
            "Pose schema (*.json *.yaml *.yml);;All files (*)",
        )
        if path:
            self.export_schema_edit.setText(path)

    def _update_enabled_state(self) -> None:
        idx = self.mode_combo.currentIndex()
        if idx == 0:
            has_input = bool(
                self.csv_path_edit.text().strip() or self.video_path_edit.text().strip()
            )
            self.run_btn.setEnabled(has_input)
            return
        if idx == 1:
            labeled_data = self.labeled_data_edit.text().strip()
            instance_label = self.instance_label_edit.text().strip()
            if not labeled_data or not instance_label:
                self.run_btn.setEnabled(False)
                return
            if self.write_pose_schema_check.isChecked():
                self.run_btn.setEnabled(bool(self.pose_schema_out_edit.text().strip()))
            else:
                self.run_btn.setEnabled(True)
            return
        has_export_src = bool(self.export_source_edit.text().strip())
        has_export_out = bool(self.export_output_edit.text().strip())
        self.run_btn.setEnabled(has_export_src and has_export_out)

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.close_btn.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        self.stack.setEnabled(not running)
        self.run_btn.setText("Running…" if running else "Run")

    def closeEvent(self, event) -> None:
        if self._thread is not None and self._thread.isRunning():
            event.ignore()
            return
        super().closeEvent(event)

    def _run(self) -> None:
        if self._thread is not None:
            return

        idx = self.mode_combo.currentIndex()
        if idx == 0:
            csv_path = self.csv_path_edit.text().strip() or None
            video_path = self.video_path_edit.text().strip() or None
            output_dir = self.output_dir_edit.text().strip() or None
            min_like_raw = float(self.min_likelihood_spin.value())
            min_likelihood = min_like_raw if min_like_raw > 0.0 else None
            is_multi = True if self.multi_animal_check.isChecked() else None
            write_schema = bool(self.analysis_pose_schema_check.isChecked())
            save_frames = bool(self.analysis_save_frames_check.isChecked())

            def task() -> Dict[str, Any]:
                return import_deeplabcut_analysis(
                    video_path=video_path,
                    output_dir=output_dir,
                    is_multi_animal=is_multi,
                    csv_path=csv_path,
                    min_likelihood=min_likelihood,
                    write_pose_schema=write_schema,
                    save_frames=save_frames,
                )

            self._start_worker(task)
            return

        if idx == 1:
            labeled_data = Path(self.labeled_data_edit.text().strip()).expanduser()
            if not labeled_data.exists():
                QtWidgets.QMessageBox.critical(
                    self, "Error", f"Folder does not exist:\n{labeled_data}"
                )
                return
            dataset_root = labeled_data.parent
            instance_label = self.instance_label_edit.text().strip() or "mouse"
            overwrite = bool(self.overwrite_check.isChecked())
            recursive = bool(self.recursive_check.isChecked())

            write_pose_schema = bool(self.write_pose_schema_check.isChecked())
            pose_schema_out = self.pose_schema_out_edit.text().strip() or str(
                labeled_data / "pose_schema.json"
            )
            preset = self.pose_schema_preset_combo.currentText().strip() or "auto"
            instance_separator = self.instance_separator_edit.text().strip() or "_"

            def task() -> Dict[str, Any]:
                return import_deeplabcut_dataset(
                    DeepLabCutTrainingImportConfig(
                        source_dir=dataset_root,
                        labeled_data_root=labeled_data,
                        instance_label=instance_label,
                        overwrite=overwrite,
                        recursive=recursive,
                    ),
                    write_pose_schema=write_pose_schema,
                    pose_schema_out=Path(pose_schema_out)
                    if write_pose_schema
                    else None,
                    pose_schema_preset=preset,
                    instance_separator=instance_separator,
                )

            self._start_worker(task)
            return

        source_path = self.export_source_edit.text().strip()
        output_path = self.export_output_edit.text().strip()
        target_format = str(self.export_format_combo.currentData() or "deeplabcut-csv")
        scorer = self.export_scorer_edit.text().strip() or "annolid"
        schema_path = self.export_schema_edit.text().strip() or None
        copy_images = bool(self.export_copy_images_check.isChecked())

        def export_task() -> Dict[str, Any]:
            return export_pose_dataset(
                source_path,
                output_path,
                target_format=target_format,
                pose_schema=schema_path,
                scorer=scorer,
                copy_images=copy_images,
            )

        self._start_worker(export_task)

    def _start_worker(self, fn) -> None:
        self._set_running(True)
        set_widget_busy_cursor(self, True)

        thread = QtCore.QThread(self)
        worker = _ConversionWorker(fn)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        try:
            thread.start()
        except Exception:
            self._cleanup_worker()
            raise

    def _on_finished(self, summary: Dict[str, Any]) -> None:
        self._cleanup_worker()
        msg = "Conversion completed successfully."
        if summary:
            msg += "\n\n" + "\n".join(f"{k}: {v}" for k, v in summary.items())
        QtWidgets.QMessageBox.information(self, "Success", msg)

    def _on_failed(self, message: str) -> None:
        self._cleanup_worker()
        QtWidgets.QMessageBox.critical(self, "Error", f"Conversion failed:\n{message}")

    def _cleanup_worker(self) -> None:
        try:
            set_widget_busy_cursor(self, False)
        except Exception:
            pass
        self._thread = None
        self._worker = None
        self._set_running(False)
        self._update_enabled_state()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    dlg = ConvertDLCDialog()
    dlg.show()
    raise SystemExit(app.exec_())
