from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Dict, Optional

from qtpy.QtCore import QObject, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from annolid.services.export import export_pose_dataset, import_sleap_dataset


def _suggest_out_dir(sleap_path: Path) -> Path:
    """Default output dir: <input_stem>_labelme next to the input file."""
    p = Path(sleap_path).expanduser()
    if p.suffixes[-2:] == [".pkg", ".slp"]:
        stem = p.name[: -len("".join(p.suffixes[-2:]))]
    elif p.suffixes[-2:] == [".analysis", ".h5"]:
        stem = p.name[: -len("".join(p.suffixes[-2:]))]
    else:
        stem = p.stem
    return p.parent / f"{stem}_labelme"


@dataclass(frozen=True)
class ConvertJob:
    sleap_path: Path
    out_dir: Path
    save_frames: bool = True
    video_index: Optional[int] = None
    print_every: int = 200
    video_path: Optional[Path] = None
    pose_schema_path: Optional[Path] = None
    prefer_user_instances: bool = True
    include_occluded: bool = True


@dataclass(frozen=True)
class ExportSleapJob:
    source_path: Path
    output_path: Path
    target_format: str = "sleap-h5"
    pose_schema_path: Optional[Path] = None
    video_path: Optional[Path] = None
    embed_images: bool = False


class _Worker(QObject):
    finished = Signal(dict)
    error = Signal(str)
    log = Signal(str)
    progress = Signal(int)

    def __init__(self, job: ConvertJob | ExportSleapJob):
        super().__init__()
        self._job = job

    @Slot()
    def run(self) -> None:
        try:
            if isinstance(self._job, ExportSleapJob):
                self.log.emit(f"[ui] Source: {self._job.source_path}")
                self.log.emit(f"[ui] Output: {self._job.output_path}")
                self.log.emit(
                    f"[ui] Format: {self._job.target_format} (embed_images={self._job.embed_images})"
                )
                out = export_pose_dataset(
                    self._job.source_path,
                    self._job.output_path,
                    target_format=self._job.target_format,
                    pose_schema=self._job.pose_schema_path,
                    video_path=self._job.video_path,
                    embed_images=self._job.embed_images,
                )
                self.finished.emit(out if isinstance(out, dict) else {"result": out})
                return

            self.log.emit(f"[ui] Input: {self._job.sleap_path}")
            self.log.emit(f"[ui] Output: {self._job.out_dir}")
            self.log.emit(
                f"[ui] Options: save_frames={self._job.save_frames}, "
                f"video_index={self._job.video_index}, "
                f"prefer_user={self._job.prefer_user_instances}, "
                f"include_occluded={self._job.include_occluded}"
            )

            self._job.out_dir.mkdir(parents=True, exist_ok=True)
            out = import_sleap_dataset(
                self._job.sleap_path,
                self._job.out_dir,
                save_frames=self._job.save_frames,
                video_index=self._job.video_index,
                print_every=self._job.print_every,
                pose_schema_path=self._job.pose_schema_path,
                video_path=self._job.video_path,
                prefer_user_instances=self._job.prefer_user_instances,
                include_occluded=self._job.include_occluded,
            )
            self.finished.emit(out if isinstance(out, dict) else {"result": out})
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


class ConvertSleapDialog(QDialog):
    """Bidirectional SLEAP (.slp, .pkg.slp, .analysis.h5) ↔ Annolid LabelMe converter."""

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._sleap_path: Optional[Path] = None
        self._out_dir: Optional[Path] = None
        self._thread: Optional[QThread] = None
        self._worker: Optional[_Worker] = None

        self._build_ui()
        self._wire_ui()
        self._refresh_buttons()

    # ---------------- UI ----------------

    def _build_ui(self) -> None:
        self.setWindowTitle("SLEAP (.slp / .h5) Import & Export")
        self.resize(780, 580)

        root = QVBoxLayout(self)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(
            "Import SLEAP (.slp / .pkg.slp / .analysis.h5) → LabelMe"
        )
        self.mode_combo.addItem(
            "Export Annolid LabelMe / Tracking → SLEAP (.slp / .analysis.h5)"
        )
        mode_row.addWidget(self.mode_combo, 1)
        root.addLayout(mode_row)

        self.stack = QStackedWidget()
        self.stack.addWidget(self._build_import_page())
        self.stack.addWidget(self._build_export_page())
        root.addWidget(self.stack)

        # Actions
        actions = QHBoxLayout()
        self.btn_convert = QPushButton("Convert")
        self.btn_open_out = QPushButton("Open Output Folder")
        self.btn_close = QPushButton("Close")

        actions.addWidget(self.btn_convert)
        actions.addWidget(self.btn_open_out)
        actions.addStretch(1)
        actions.addWidget(self.btn_close)

        root.addLayout(actions)

        # Log panel
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Logs will appear here…")
        root.addWidget(self.log_view, 1)

    def _build_import_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        grp_files = QGroupBox("Files")
        files_layout = QGridLayout(grp_files)

        self.txt_in = QLineEdit()
        self.txt_in.setReadOnly(True)
        self.txt_in.setPlaceholderText("Select a .slp, .pkg.slp, or .analysis.h5 file…")
        self.btn_browse_in = QPushButton("Input…")

        self.txt_out = QLineEdit()
        self.txt_out.setReadOnly(False)
        self.txt_out.setPlaceholderText("Choose or type an output folder…")
        self.btn_use_suggested = QPushButton("Use Suggested")
        self.btn_browse_out = QPushButton("Output…")

        self.txt_video = QLineEdit()
        self.txt_video.setPlaceholderText(
            "Optional external video (for non-packaged .slp or .analysis.h5)"
        )
        self.btn_browse_video = QPushButton("Video…")

        self.txt_pose_schema = QLineEdit()
        self.txt_pose_schema.setPlaceholderText(
            "Optional PoseSchema JSON/YAML to remap skeleton keypoint names"
        )
        self.btn_browse_schema = QPushButton("Schema…")

        files_layout.addWidget(QLabel("SLEAP file:"), 0, 0)
        files_layout.addWidget(self.txt_in, 0, 1)
        files_layout.addWidget(self.btn_browse_in, 0, 2)

        files_layout.addWidget(QLabel("Output dir:"), 1, 0)
        files_layout.addWidget(self.txt_out, 1, 1)
        out_btns = QHBoxLayout()
        out_btns.addWidget(self.btn_use_suggested)
        out_btns.addWidget(self.btn_browse_out)
        files_layout.addLayout(out_btns, 1, 2)

        files_layout.addWidget(QLabel("Video (optional):"), 2, 0)
        files_layout.addWidget(self.txt_video, 2, 1)
        files_layout.addWidget(self.btn_browse_video, 2, 2)

        files_layout.addWidget(QLabel("Pose schema:"), 3, 0)
        files_layout.addWidget(self.txt_pose_schema, 3, 1)
        files_layout.addWidget(self.btn_browse_schema, 3, 2)

        layout.addWidget(grp_files)

        grp_opts = QGroupBox("Options")
        opts_layout = QGridLayout(grp_opts)

        self.chk_save_frames = QCheckBox("Save embedded / video frames")
        self.chk_save_frames.setChecked(True)

        self.chk_prefer_user = QCheckBox(
            "Prefer user labels over predictions on the same frame"
        )
        self.chk_prefer_user.setChecked(True)

        self.chk_include_occluded = QCheckBox(
            "Preserve occluded keypoints (finite coords with visible=False)"
        )
        self.chk_include_occluded.setChecked(True)

        self.spin_video_index = QSpinBox()
        self.spin_video_index.setRange(-1, 9999)
        self.spin_video_index.setSpecialValueText("All videos")
        self.spin_video_index.setValue(-1)

        self.spin_print_every = QSpinBox()
        self.spin_print_every.setRange(1, 1_000_000)
        self.spin_print_every.setValue(200)

        opts_layout.addWidget(self.chk_save_frames, 0, 0, 1, 2)
        opts_layout.addWidget(self.chk_prefer_user, 1, 0, 1, 2)
        opts_layout.addWidget(self.chk_include_occluded, 2, 0, 1, 2)
        opts_layout.addWidget(QLabel("Video index:"), 3, 0)
        opts_layout.addWidget(self.spin_video_index, 3, 1)
        opts_layout.addWidget(QLabel("Print every:"), 4, 0)
        opts_layout.addWidget(self.spin_print_every, 4, 1)

        note = QLabel(
            "Supports both SLEAP project files (.slp / .pkg.slp) and SLEAP Analysis HDF5 (.analysis.h5). "
            "Multi-video .slp files automatically prefix outputs per video."
        )
        note.setWordWrap(True)
        opts_layout.addWidget(note, 5, 0, 1, 2)

        layout.addWidget(grp_opts)
        return page

    def _build_export_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)

        grp = QGroupBox("Export to SLEAP")
        grid = QGridLayout(grp)

        self.txt_export_src = QLineEdit()
        self.txt_export_src.setPlaceholderText(
            "Select a LabelMe folder or Annolid *_tracking.csv file…"
        )
        self.btn_export_src_dir = QPushButton("Folder…")
        self.btn_export_src_file = QPushButton("File…")
        src_btns = QHBoxLayout()
        src_btns.addWidget(self.btn_export_src_dir)
        src_btns.addWidget(self.btn_export_src_file)

        self.combo_export_fmt = QComboBox()
        self.combo_export_fmt.addItem("SLEAP Analysis HDF5 (.analysis.h5)", "sleap-h5")
        self.combo_export_fmt.addItem("SLEAP Project (.slp)", "sleap-slp")
        self.combo_export_fmt.addItem(
            "SLEAP Self-Contained Package (.pkg.slp with embedded frames)",
            "sleap-pkg-slp",
        )

        self.txt_export_out = QLineEdit()
        self.txt_export_out.setPlaceholderText("Choose output .h5 or .slp file…")
        self.btn_export_out = QPushButton("Output…")

        self.txt_export_schema = QLineEdit()
        self.txt_export_schema.setPlaceholderText(
            "Optional PoseSchema JSON/YAML (auto-detected if present in source folder)"
        )
        self.btn_export_schema = QPushButton("Schema…")

        self.txt_export_video = QLineEdit()
        self.txt_export_video.setPlaceholderText(
            "Optional video path stored in SLEAP metadata"
        )
        self.btn_export_video = QPushButton("Video…")

        grid.addWidget(QLabel("Annolid source:"), 0, 0)
        grid.addWidget(self.txt_export_src, 0, 1)
        grid.addLayout(src_btns, 0, 2)

        grid.addWidget(QLabel("SLEAP format:"), 1, 0)
        grid.addWidget(self.combo_export_fmt, 1, 1, 1, 2)

        grid.addWidget(QLabel("Output file:"), 2, 0)
        grid.addWidget(self.txt_export_out, 2, 1)
        grid.addWidget(self.btn_export_out, 2, 2)

        grid.addWidget(QLabel("Pose schema:"), 3, 0)
        grid.addWidget(self.txt_export_schema, 3, 1)
        grid.addWidget(self.btn_export_schema, 3, 2)

        grid.addWidget(QLabel("Video (optional):"), 4, 0)
        grid.addWidget(self.txt_export_video, 4, 1)
        grid.addWidget(self.btn_export_video, 4, 2)

        layout.addWidget(grp)
        layout.addStretch(1)
        return page

    def _wire_ui(self) -> None:
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        self.btn_browse_in.clicked.connect(self.select_input_file)
        self.btn_browse_out.clicked.connect(self.select_out_dir)
        self.btn_use_suggested.clicked.connect(self.use_suggested_out_dir)
        self.btn_browse_video.clicked.connect(self._select_import_video)
        self.btn_browse_schema.clicked.connect(self._select_import_schema)

        self.btn_export_src_dir.clicked.connect(self._select_export_src_dir)
        self.btn_export_src_file.clicked.connect(self._select_export_src_file)
        self.btn_export_out.clicked.connect(self._select_export_output)
        self.btn_export_schema.clicked.connect(self._select_export_schema)
        self.btn_export_video.clicked.connect(self._select_export_video)
        self.combo_export_fmt.currentIndexChanged.connect(self._suggest_export_out)
        self.txt_export_src.textChanged.connect(self._refresh_buttons)
        self.txt_export_out.textChanged.connect(self._refresh_buttons)

        self.btn_convert.clicked.connect(self.run_conversion)
        self.btn_open_out.clicked.connect(self.open_output_folder)
        self.btn_close.clicked.connect(self.close)
        self.txt_out.textChanged.connect(self._on_out_dir_changed)

    # ---------------- State helpers ----------------

    def _on_mode_changed(self) -> None:
        self.stack.setCurrentIndex(self.mode_combo.currentIndex())
        self._refresh_buttons()

    def _append_log(self, msg: str) -> None:
        self.log_view.append(msg)

    def _active_output_folder(self) -> Optional[Path]:
        if self.mode_combo.currentIndex() == 0:
            return self._out_dir
        raw = self.txt_export_out.text().strip()
        if not raw:
            return None
        p = Path(raw).expanduser()
        return p if p.is_dir() else p.parent

    def _refresh_buttons(self) -> None:
        busy = self._is_busy()
        self.mode_combo.setEnabled(not busy)
        if self.mode_combo.currentIndex() == 0:
            has_in = self._sleap_path is not None
            has_out = self._out_dir is not None and str(self._out_dir).strip() != ""
            can_convert = has_in and has_out and not busy
            self.btn_browse_in.setEnabled(not busy)
            self.btn_browse_out.setEnabled(has_in and not busy)
            self.btn_use_suggested.setEnabled(has_in and not busy)
            self.chk_save_frames.setEnabled(not busy)
            self.chk_prefer_user.setEnabled(not busy)
            self.chk_include_occluded.setEnabled(not busy)
            self.spin_video_index.setEnabled(not busy)
            self.spin_print_every.setEnabled(not busy)
        else:
            has_src = bool(self.txt_export_src.text().strip())
            has_out = bool(self.txt_export_out.text().strip())
            can_convert = has_src and has_out and not busy

        self.btn_convert.setEnabled(can_convert)
        self.btn_close.setEnabled(not busy)
        out_folder = self._active_output_folder()
        self.btn_open_out.setEnabled(
            out_folder is not None and out_folder.exists() and out_folder.is_dir()
        )

    def _is_busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def closeEvent(self, event) -> None:
        if self._is_busy():
            event.ignore()
            return
        super().closeEvent(event)

    def _set_sleap_path(self, path: Path) -> None:
        p = Path(path).expanduser()
        self._sleap_path = p
        self.txt_in.setText(str(p))
        self._out_dir = _suggest_out_dir(p)
        self.txt_out.setText(str(self._out_dir))
        self._append_log(f"[ui] Selected input: {p}")
        self._append_log(f"[ui] Suggested output: {self._out_dir}")
        self._refresh_buttons()

    def _on_out_dir_changed(self, text: str) -> None:
        text = text.strip()
        if not text:
            self._out_dir = None
        else:
            self._out_dir = Path(text).expanduser()
        self._refresh_buttons()

    def use_suggested_out_dir(self) -> None:
        if not self._sleap_path:
            return
        self._out_dir = _suggest_out_dir(self._sleap_path)
        self.txt_out.setText(str(self._out_dir))
        self._append_log(f"[ui] Output dir reset to suggested: {self._out_dir}")
        self._refresh_buttons()

    # ---------------- Actions ----------------

    def select_input_file(self) -> None:
        dlg = QFileDialog(self)
        dlg.setWindowTitle("Select SLEAP file")
        dlg.setFileMode(QFileDialog.ExistingFile)
        dlg.setNameFilter(
            "SLEAP files (*.slp *.h5 *.hdf5);;SLEAP project/package (*.slp);;SLEAP Analysis HDF5 (*.h5 *.hdf5);;All files (*)"
        )
        if dlg.exec_():
            paths = dlg.selectedFiles()
            if paths:
                self._set_sleap_path(Path(paths[0]))

    def select_out_dir(self) -> None:
        base = (
            str(self._out_dir)
            if self._out_dir
            else (
                str(self._sleap_path.parent) if self._sleap_path else str(Path.home())
            )
        )
        out = QFileDialog.getExistingDirectory(self, "Select Output Folder", base)
        if out:
            self.txt_out.setText(str(Path(out).expanduser()))
            self._append_log(f"[ui] Output dir set: {out}")

    def _select_import_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self.txt_video.setText(path)

    def _select_import_schema(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PoseSchema",
            str(Path.home()),
            "Pose schema (*.json *.yaml *.yml);;All files (*)",
        )
        if path:
            self.txt_pose_schema.setText(path)

    def _suggest_export_out(self) -> None:
        raw_src = self.txt_export_src.text().strip()
        if not raw_src:
            return
        src = Path(raw_src).expanduser()
        fmt = str(self.combo_export_fmt.currentData() or "sleap-h5")
        if fmt == "sleap-h5":
            ext = ".analysis.h5"
        elif fmt == "sleap-pkg-slp":
            ext = ".pkg.slp"
        else:
            ext = ".slp"
        self.txt_export_out.setText(str(src.parent / f"{src.stem}{ext}"))

    def _select_export_src_dir(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Annolid LabelMe folder",
            self.txt_export_src.text().strip() or str(Path.home()),
        )
        if folder:
            self.txt_export_src.setText(folder)
            if not self.txt_export_out.text().strip():
                self._suggest_export_out()

    def _select_export_src_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Annolid Tracking CSV or COCO JSON",
            str(Path.home()),
            "Annolid/COCO (*.csv *.json);;All files (*)",
        )
        if path:
            self.txt_export_src.setText(path)
            if not self.txt_export_out.text().strip():
                self._suggest_export_out()

    def _select_export_output(self) -> None:
        fmt = str(self.combo_export_fmt.currentData() or "sleap-h5")
        flt = (
            "SLEAP Analysis HDF5 (*.h5);;All files (*)"
            if fmt == "sleap-h5"
            else "SLEAP Project (*.slp);;All files (*)"
        )
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save SLEAP Output",
            self.txt_export_out.text().strip() or str(Path.home()),
            flt,
        )
        if path:
            self.txt_export_out.setText(path)

    def _select_export_schema(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PoseSchema",
            str(Path.home()),
            "Pose schema (*.json *.yaml *.yml);;All files (*)",
        )
        if path:
            self.txt_export_schema.setText(path)

    def _select_export_video(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            str(Path.home()),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All files (*)",
        )
        if path:
            self.txt_export_video.setText(path)

    def _make_job(self) -> ConvertJob | ExportSleapJob:
        if self.mode_combo.currentIndex() == 1:
            fmt_data = str(self.combo_export_fmt.currentData() or "sleap-h5")
            embed = fmt_data == "sleap-pkg-slp"
            target_fmt = (
                "sleap-slp"
                if fmt_data in ("sleap-slp", "sleap-pkg-slp")
                else "sleap-h5"
            )
            schema_txt = self.txt_export_schema.text().strip()
            video_txt = self.txt_export_video.text().strip()
            return ExportSleapJob(
                source_path=Path(self.txt_export_src.text().strip()).expanduser(),
                output_path=Path(self.txt_export_out.text().strip()).expanduser(),
                target_format=target_fmt,
                pose_schema_path=Path(schema_txt).expanduser() if schema_txt else None,
                video_path=Path(video_txt).expanduser() if video_txt else None,
                embed_images=embed,
            )

        assert self._sleap_path is not None
        assert self._out_dir is not None
        vid_value = int(self.spin_video_index.value())
        video_index = None if vid_value < 0 else vid_value
        video_txt = self.txt_video.text().strip()
        schema_txt = self.txt_pose_schema.text().strip()
        return ConvertJob(
            sleap_path=Path(self._sleap_path).expanduser(),
            out_dir=Path(self._out_dir).expanduser(),
            save_frames=self.chk_save_frames.isChecked(),
            video_index=video_index,
            print_every=int(self.spin_print_every.value()),
            video_path=Path(video_txt).expanduser() if video_txt else None,
            pose_schema_path=Path(schema_txt).expanduser() if schema_txt else None,
            prefer_user_instances=self.chk_prefer_user.isChecked(),
            include_occluded=self.chk_include_occluded.isChecked(),
        )

    def run_conversion(self) -> None:
        if self._is_busy():
            return
        if self.mode_combo.currentIndex() == 0:
            if not self._sleap_path:
                QMessageBox.warning(
                    self,
                    "Missing input",
                    "Please select a .slp, .pkg.slp, or .h5 file.",
                )
                return
            if not self._out_dir:
                QMessageBox.warning(
                    self, "Missing output", "Please choose an output directory."
                )
                return
        else:
            if not self.txt_export_src.text().strip():
                QMessageBox.warning(
                    self,
                    "Missing source",
                    "Please select an Annolid source folder or CSV.",
                )
                return
            if not self.txt_export_out.text().strip():
                QMessageBox.warning(
                    self, "Missing output", "Please choose an output file path."
                )
                return

        job = self._make_job()
        self._append_log("[ui] Starting conversion…")
        self._refresh_buttons()

        self._thread = QThread(self)
        self._worker = _Worker(job)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.log.connect(self._append_log)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self._worker.finished.connect(self._thread.quit)
        self._worker.error.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup_thread)

        self._thread.start()
        self._refresh_buttons()

    def _on_finished(self, result: Dict[str, Any]) -> None:
        self._append_log("[ui] Conversion completed successfully.")
        if result:
            for k, v in result.items():
                self._append_log(f"[ui]   {k}: {v}")
        QMessageBox.information(self, "Success", "Conversion completed successfully.")
        self._refresh_buttons()

    def _on_error(self, msg: str) -> None:
        self._append_log(f"[ui] ERROR: {msg}")
        QMessageBox.critical(self, "Error", f"Conversion failed:\n\n{msg}")
        self._refresh_buttons()

    def _cleanup_thread(self) -> None:
        if self._worker is not None:
            self._worker.deleteLater()
        if self._thread is not None:
            self._thread.deleteLater()
        self._worker = None
        self._thread = None
        self._refresh_buttons()

    def open_output_folder(self) -> None:
        out = self._active_output_folder()
        if not out:
            return
        if not out.exists():
            QMessageBox.information(
                self, "Not found", "Output folder does not exist yet."
            )
            return

        try:
            import os
            import platform
            import subprocess

            system = platform.system().lower()
            if "darwin" in system:
                subprocess.check_call(["open", str(out)])
            elif "windows" in system:
                os.startfile(str(out))  # type: ignore[attr-defined]
            else:
                subprocess.check_call(["xdg-open", str(out)])
        except Exception as e:
            QMessageBox.warning(self, "Could not open folder", str(e))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    dlg = ConvertSleapDialog()
    dlg.show()
    sys.exit(app.exec_())
