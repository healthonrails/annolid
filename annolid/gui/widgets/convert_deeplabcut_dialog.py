from __future__ import annotations

from pathlib import Path
import sys
from typing import Any, Dict, Optional

from qtpy import QtCore, QtWidgets

from annolid.annotation.deeplabcut2labelme import deeplabcut_to_labelme_json
from annolid.datasets.importers.deeplabcut_training_data import (
    DeepLabCutTrainingImportConfig,
    import_deeplabcut_training_data,
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
    def __init__(self):
        super().__init__()
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_ConversionWorker] = None
        self._build_ui()

    def _build_ui(self) -> None:
        self.setWindowTitle("Convert DeepLabCut → LabelMe")
        self.setModal(True)
        self.resize(640, 360)

        layout = QtWidgets.QVBoxLayout(self)

        # Mode selection
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("DLC analysis (video + matching CSV)")
        self.mode_combo.addItem("DLC training data (labeled-data/CollectedData_*.csv)")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.mode_combo, 1)
        layout.addLayout(mode_row)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        self.stack.addWidget(self._build_analysis_page())
        self.stack.addWidget(self._build_training_page())

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

        self.video_path_edit = QtWidgets.QLineEdit()
        self.video_path_edit.setReadOnly(True)
        video_row = QtWidgets.QHBoxLayout()
        video_row.addWidget(self.video_path_edit, 1)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._browse_video)
        video_row.addWidget(browse)
        form.addRow("Video file:", video_row)

        self.multi_animal_check = QtWidgets.QCheckBox("Multi-animal tracking")
        form.addRow("", self.multi_animal_check)

        self.output_dir_edit = QtWidgets.QLineEdit()
        self.output_dir_edit.setPlaceholderText("Output directory (defaults to <video stem>/)")
        out_row = QtWidgets.QHBoxLayout()
        out_row.addWidget(self.output_dir_edit, 1)
        out_browse = QtWidgets.QPushButton("Browse…")
        out_browse.clicked.connect(self._browse_output_dir)
        out_row.addWidget(out_browse)
        form.addRow("Output dir:", out_row)

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

        self.write_pose_schema_check = QtWidgets.QCheckBox("Write pose schema (pose_schema.json)")
        self.write_pose_schema_check.setChecked(True)
        self.write_pose_schema_check.toggled.connect(self._update_enabled_state)
        form.addRow("", self.write_pose_schema_check)

        self.pose_schema_out_edit = QtWidgets.QLineEdit()
        self.pose_schema_out_edit.setPlaceholderText("pose_schema.json output path (defaults to <labeled-data>/pose_schema.json)")
        ps_row = QtWidgets.QHBoxLayout()
        ps_row.addWidget(self.pose_schema_out_edit, 1)
        ps_browse = QtWidgets.QPushButton("Browse…")
        ps_browse.clicked.connect(self._browse_pose_schema_out)
        ps_row.addWidget(ps_browse)
        form.addRow("Pose schema:", ps_row)

        self.pose_schema_preset_combo = QtWidgets.QComboBox()
        self.pose_schema_preset_combo.addItems(["mouse"])
        form.addRow("Edge preset:", self.pose_schema_preset_combo)

        self.instance_separator_edit = QtWidgets.QLineEdit("_")
        self.instance_separator_edit.setMaxLength(4)
        form.addRow("Instance separator:", self.instance_separator_edit)

        # Update run button availability
        self.instance_label_edit.textChanged.connect(self._update_enabled_state)
        self.labeled_data_edit.textChanged.connect(self._update_enabled_state)
        self.pose_schema_out_edit.textChanged.connect(self._update_enabled_state)

        return page

    def _on_mode_changed(self) -> None:
        self.stack.setCurrentIndex(self.mode_combo.currentIndex())
        self._update_enabled_state()

    def _browse_video(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select video file",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov);;All files (*)",
        )
        if not path:
            return
        self.video_path_edit.setText(path)
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

    def _update_enabled_state(self) -> None:
        idx = self.mode_combo.currentIndex()
        if idx == 0:
            self.run_btn.setEnabled(bool(self.video_path_edit.text().strip()))
            return
        labeled_data = self.labeled_data_edit.text().strip()
        instance_label = self.instance_label_edit.text().strip()
        if not labeled_data or not instance_label:
            self.run_btn.setEnabled(False)
            return
        if self.write_pose_schema_check.isChecked():
            self.run_btn.setEnabled(bool(self.pose_schema_out_edit.text().strip()))
        else:
            self.run_btn.setEnabled(True)

    def _set_running(self, running: bool) -> None:
        self.run_btn.setEnabled(not running)
        self.close_btn.setEnabled(not running)
        self.mode_combo.setEnabled(not running)
        self.stack.setEnabled(not running)
        self.run_btn.setText("Running…" if running else "Run")

    def _run(self) -> None:
        if self._thread is not None:
            return

        idx = self.mode_combo.currentIndex()
        if idx == 0:
            video_path = self.video_path_edit.text().strip()
            output_dir = self.output_dir_edit.text().strip() or str(Path(video_path).with_suffix(""))
            is_multi = bool(self.multi_animal_check.isChecked())

            def task() -> Dict[str, Any]:
                deeplabcut_to_labelme_json(video_path, output_dir, is_multi)
                return {"output_dir": output_dir}

            self._start_worker(task)
            return

        labeled_data = Path(self.labeled_data_edit.text().strip()).expanduser()
        if not labeled_data.exists():
            QtWidgets.QMessageBox.critical(self, "Error", f"Folder does not exist:\n{labeled_data}")
            return
        dataset_root = labeled_data.parent
        instance_label = self.instance_label_edit.text().strip() or "mouse"
        overwrite = bool(self.overwrite_check.isChecked())
        recursive = bool(self.recursive_check.isChecked())

        write_pose_schema = bool(self.write_pose_schema_check.isChecked())
        pose_schema_out = self.pose_schema_out_edit.text().strip() or str(labeled_data / "pose_schema.json")
        preset = self.pose_schema_preset_combo.currentText().strip() or "mouse"
        instance_separator = self.instance_separator_edit.text().strip() or "_"

        def task() -> Dict[str, Any]:
            return import_deeplabcut_training_data(
                DeepLabCutTrainingImportConfig(
                    source_dir=dataset_root,
                    labeled_data_root=labeled_data,
                    instance_label=instance_label,
                    overwrite=overwrite,
                    recursive=recursive,
                ),
                write_pose_schema=write_pose_schema,
                pose_schema_out=Path(pose_schema_out) if write_pose_schema else None,
                pose_schema_preset=preset,
                instance_separator=instance_separator,
            )

        self._start_worker(task)

    def _start_worker(self, fn) -> None:
        self._set_running(True)
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)

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
        thread.start()

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
            QtWidgets.QApplication.restoreOverrideCursor()
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
