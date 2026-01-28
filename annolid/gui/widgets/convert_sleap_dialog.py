import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from qtpy.QtCore import QObject, QThread, Signal, Slot
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
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
    QTextEdit,
    QVBoxLayout,
)

from annolid.annotation.sleap2labelme import convert_sleap_h5_to_labelme


def _suggest_out_dir(sleap_path: Path) -> Path:
    """Default output dir: <input_stem>_labelme next to the input file."""
    p = Path(sleap_path).expanduser()
    if p.suffixes[-2:] == [".pkg", ".slp"]:
        stem = p.name[: -len("".join(p.suffixes[-2:]))]  # remove ".pkg.slp"
    else:
        stem = p.stem
    return p.parent / f"{stem}_labelme"


@dataclass(frozen=True)
class ConvertJob:
    sleap_path: Path
    out_dir: Path
    save_frames: bool = True
    video_index: int = 0
    print_every: int = 200


class _Worker(QObject):
    finished = Signal(dict)  # result
    error = Signal(str)  # error message
    log = Signal(str)  # log lines
    progress = Signal(int)  # percent (optional)

    def __init__(self, job: ConvertJob):
        super().__init__()
        self._job = job

    @Slot()
    def run(self) -> None:
        try:
            self.log.emit(f"[ui] Input: {self._job.sleap_path}")
            self.log.emit(f"[ui] Output: {self._job.out_dir}")
            self.log.emit(
                f"[ui] Options: save_frames={self._job.save_frames}, "
                f"video_index={self._job.video_index}, print_every={self._job.print_every}"
            )

            # Ensure output folder exists (convert function may also do this, but safe here)
            self._job.out_dir.mkdir(parents=True, exist_ok=True)

            out = convert_sleap_h5_to_labelme(
                self._job.sleap_path,
                self._job.out_dir,
                save_frames=self._job.save_frames,
                video_index=self._job.video_index,
                print_every=self._job.print_every,
            )
            self.finished.emit(out if isinstance(out, dict) else {"result": out})
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}")


class ConvertSleapDialog(QDialog):
    def __init__(self, parent=None):
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
        self.setWindowTitle("Convert SLEAP to Labelme")
        self.resize(760, 520)

        root = QVBoxLayout(self)

        # File selection group
        grp_files = QGroupBox("Files")
        files_layout = QGridLayout(grp_files)

        self.txt_in = QLineEdit()
        self.txt_in.setReadOnly(True)
        self.txt_in.setPlaceholderText("Select a .h5 or .pkg.slp file…")

        self.btn_browse_in = QPushButton("Input…")

        self.txt_out = QLineEdit()
        self.txt_out.setReadOnly(False)  # allow typing
        self.txt_out.setPlaceholderText("Choose or type an output folder…")

        self.btn_use_suggested = QPushButton("Use Suggested")
        self.btn_browse_out = QPushButton("Output…")

        files_layout.addWidget(QLabel("SLEAP file:"), 0, 0)
        files_layout.addWidget(self.txt_in, 0, 1)
        files_layout.addWidget(self.btn_browse_in, 0, 2)

        files_layout.addWidget(QLabel("Output dir:"), 1, 0)
        files_layout.addWidget(self.txt_out, 1, 1)
        out_btns = QHBoxLayout()
        out_btns.addWidget(self.btn_use_suggested)
        out_btns.addWidget(self.btn_browse_out)
        files_layout.addLayout(out_btns, 1, 2)

        root.addWidget(grp_files)

        # Options group
        grp_opts = QGroupBox("Options")
        opts_layout = QGridLayout(grp_opts)

        self.chk_save_frames = QCheckBox("Save embedded frames")
        self.chk_save_frames.setChecked(True)

        self.spin_video_index = QSpinBox()
        self.spin_video_index.setRange(0, 9999)
        self.spin_video_index.setValue(0)

        self.spin_print_every = QSpinBox()
        self.spin_print_every.setRange(1, 1_000_000)
        self.spin_print_every.setValue(200)

        opts_layout.addWidget(self.chk_save_frames, 0, 0, 1, 2)
        opts_layout.addWidget(QLabel("Video index:"), 1, 0)
        opts_layout.addWidget(self.spin_video_index, 1, 1)
        opts_layout.addWidget(QLabel("Print every:"), 2, 0)
        opts_layout.addWidget(self.spin_print_every, 2, 1)

        root.addWidget(grp_opts)

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

    def _wire_ui(self) -> None:
        self.btn_browse_in.clicked.connect(self.select_input_file)
        self.btn_browse_out.clicked.connect(self.select_out_dir)
        self.btn_use_suggested.clicked.connect(self.use_suggested_out_dir)
        self.btn_convert.clicked.connect(self.run_conversion)
        self.btn_open_out.clicked.connect(self.open_output_folder)
        self.btn_close.clicked.connect(self.close)

        self.txt_out.textChanged.connect(self._on_out_dir_changed)

    # ---------------- State helpers ----------------

    def _append_log(self, msg: str) -> None:
        self.log_view.append(msg)

    def _refresh_buttons(self) -> None:
        has_in = self._sleap_path is not None
        has_out = self._out_dir is not None and str(self._out_dir).strip() != ""

        self.btn_browse_out.setEnabled(has_in)
        self.btn_use_suggested.setEnabled(has_in)

        can_convert = has_in and has_out and not self._is_busy()
        self.btn_convert.setEnabled(can_convert)

        self.btn_open_out.setEnabled(
            (self._out_dir is not None)
            and self._out_dir.exists()
            and self._out_dir.is_dir()
        )

        self.btn_browse_in.setEnabled(not self._is_busy())
        self.btn_browse_out.setEnabled(has_in and not self._is_busy())
        self.btn_use_suggested.setEnabled(has_in and not self._is_busy())
        self.chk_save_frames.setEnabled(not self._is_busy())
        self.spin_video_index.setEnabled(not self._is_busy())
        self.spin_print_every.setEnabled(not self._is_busy())

    def _is_busy(self) -> bool:
        return self._thread is not None and self._thread.isRunning()

    def _set_sleap_path(self, path: Path) -> None:
        p = Path(path).expanduser()
        self._sleap_path = p
        self.txt_in.setText(str(p))

        # Suggest output dir and populate (user can still edit)
        self._out_dir = _suggest_out_dir(p)
        self.txt_out.setText(str(self._out_dir))
        self._append_log(f"[ui] Selected input: {p}")
        self._append_log(f"[ui] Suggested output: {self._out_dir}")

        self._refresh_buttons()

    def _on_out_dir_changed(self, text: str) -> None:
        text = text.strip()
        if not text:
            self._out_dir = None
            self._refresh_buttons()
            return

        p = Path(text).expanduser()
        self._out_dir = p
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
            "SLEAP files (*.h5 *.slp);;HDF5 (*.h5);;SLEAP package (*.slp);;All files (*)"
        )

        if dlg.exec_():
            paths = dlg.selectedFiles()
            if not paths:
                return
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

    def _make_job(self) -> ConvertJob:
        assert self._sleap_path is not None
        assert self._out_dir is not None
        return ConvertJob(
            sleap_path=Path(self._sleap_path).expanduser(),
            out_dir=Path(self._out_dir).expanduser(),
            save_frames=self.chk_save_frames.isChecked(),
            video_index=int(self.spin_video_index.value()),
            print_every=int(self.spin_print_every.value()),
        )

    def run_conversion(self) -> None:
        if self._is_busy():
            return
        if not self._sleap_path:
            QMessageBox.warning(
                self, "Missing input", "Please select a .h5 or .pkg.slp file."
            )
            return
        if not self._out_dir:
            QMessageBox.warning(
                self, "Missing output", "Please choose an output directory."
            )
            return

        job = self._make_job()
        self._append_log("[ui] Starting conversion…")
        self._refresh_buttons()

        # Worker thread
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

    def _on_finished(self, result: dict) -> None:
        self._append_log("[ui] Conversion completed successfully.")
        if result:
            self._append_log(f"[ui] Result: {result}")
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
        if not self._out_dir:
            return
        out = Path(self._out_dir).expanduser()
        if not out.exists():
            QMessageBox.information(
                self, "Not found", "Output folder does not exist yet."
            )
            return

        # Cross-platform open folder
        try:
            import os
            import subprocess
            import platform

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
