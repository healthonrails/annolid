from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path
from typing import Optional

from qtpy import QtCore, QtWidgets

from annolid.data.videos import download_youtube_video


class YouTubeVideoDialog(QtWidgets.QDialog):
    """Dialog to download a YouTube video and hand its local path back to the caller."""

    downloaded_path: Optional[Path]

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Open YouTube Video"))
        self.setModal(True)
        self.resize(500, 160)

        self.downloaded_path = None

        self._description_label = QtWidgets.QLabel(
            self.tr(
                "Enter a YouTube video URL to download it locally and open it in Annolid."
            ),
            self,
        )
        self._description_label.setWordWrap(True)

        self.url_edit = QtWidgets.QLineEdit(self)
        self.url_edit.setPlaceholderText(
            self.tr("https://www.youtube.com/watch?v=your_video_id")
        )

        self.status_label = QtWidgets.QLabel("", self)
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #555555;")

        self.button_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            QtCore.Qt.Horizontal,
            self,
        )
        self.button_box.accepted.connect(self._handle_download)
        self.button_box.rejected.connect(self.reject)

        self.ok_button = self.button_box.button(QtWidgets.QDialogButtonBox.Ok)
        if self.ok_button is not None:
            self.ok_button.setText(self.tr("Download"))
        cancel_button = self.button_box.button(QtWidgets.QDialogButtonBox.Cancel)
        if cancel_button is not None:
            cancel_button.setText(self.tr("Cancel"))

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._description_label)
        layout.addWidget(self.url_edit)
        layout.addWidget(self.status_label)
        layout.addWidget(self.button_box)

        self.url_edit.setFocus()
        self.url_edit.returnPressed.connect(self._handle_download)

    def _set_status(self, message: str, *, error: bool = False) -> None:
        color = "#b71c1c" if error else "#555555"
        self.status_label.setStyleSheet(f"color: {color};")
        self.status_label.setText(message)

    def _ensure_yt_dlp_installed(self) -> None:
        """Ensure the `yt-dlp` dependency is available."""
        try:
            importlib.import_module("yt_dlp")
            return
        except ImportError:
            pass

        answer = QtWidgets.QMessageBox.question(
            self,
            self.tr("Install yt-dlp"),
            self.tr(
                "The 'yt-dlp' package is required to download YouTube videos.\n"
                "Install it now?"
            ),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            raise RuntimeError(
                self.tr(
                    "YouTube download cancelled because 'yt-dlp' is not installed."
                )
            )

        python_executable = sys.executable or "python"
        self._set_status(self.tr("Installing yt-dlp..."))
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            subprocess.check_call(
                [python_executable, "-m", "pip", "install", "yt-dlp"]
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                self.tr("Failed to install yt-dlp (pip exit code %d).") % exc.returncode
            ) from exc
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

        importlib.invalidate_caches()
        try:
            importlib.import_module("yt_dlp")
        except ImportError as exc:  # pragma: no cover - defensive path
            raise RuntimeError(
                self.tr(
                    "yt-dlp installation completed but the module is unavailable."
                )
            ) from exc

    @QtCore.Slot()
    def _handle_download(self) -> None:
        """Trigger download of the specified YouTube URL."""
        url = self.url_edit.text().strip()
        if not url:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("YouTube Download"),
                self.tr("Please enter a valid YouTube video URL."),
            )
            self.url_edit.setFocus()
            return

        if self.ok_button is not None:
            self.ok_button.setEnabled(False)

        try:
            self._ensure_yt_dlp_installed()
        except RuntimeError as exc:
            self._set_status(str(exc), error=True)
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("YouTube Download"),
                str(exc),
            )
            if self.ok_button is not None:
                self.ok_button.setEnabled(True)
            return

        self._set_status(self.tr("Downloading YouTube video..."))

        downloaded_path: Optional[Path] = None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            downloaded_path = download_youtube_video(url)
        except Exception as exc:  # pragma: no cover - pass error upstream to UI
            self._set_status(str(exc), error=True)
            QtWidgets.QMessageBox.critical(
                self,
                self.tr("YouTube Download"),
                self.tr("Failed to download the video:\n%s") % exc,
            )
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()
            if self.ok_button is not None:
                self.ok_button.setEnabled(True)

        if not downloaded_path:
            return

        if not downloaded_path.exists():
            self._set_status(
                self.tr("Download completed but the video file is missing."), error=True
            )
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("YouTube Download"),
                self.tr("Download completed but the video file is missing."),
            )
            return

        self.downloaded_path = downloaded_path
        self._set_status(self.tr("Download complete."))
        self.accept()
