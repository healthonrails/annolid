"""Non-blocking Qt helper for installing optional runtime capabilities on demand."""

from __future__ import annotations

import os
import sys
import threading
from typing import Sequence

from qtpy import QtCore, QtGui, QtWidgets

from annolid.infrastructure.capabilities import (
    CapabilityStatus,
    RuntimeInstallCancelledError,
    _auto_install_optional_deps_enabled,
    capability_message,
    check_capability,
    ensure_capability,
    install_capability_stream,
    set_interactive_install_handler,
)
from annolid.utils.logger import logger


_SESSION_AUTO_CONFIRM = False


def _on_main_gui_thread() -> bool:
    app = QtWidgets.QApplication.instance()
    if app is None:
        return False
    return QtCore.QThread.currentThread() == app.thread()


class _CapabilityInstallWorker(QtCore.QThread):
    output_line = QtCore.Signal(str)
    install_succeeded = QtCore.Signal(object)
    install_cancelled = QtCore.Signal()
    install_failed = QtCore.Signal(str)

    def __init__(
        self,
        capability_names: tuple[str, ...],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._capability_names = capability_names
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:  # pragma: no cover - exercised via QThread event loop
        installed_all: list[str] = []
        try:
            for cap_name in self._capability_names:
                self.output_line.emit(f"==> Installing runtime capability: {cap_name}")
                specs = install_capability_stream(
                    cap_name,
                    auto_install=True,
                    on_output=lambda line: self.output_line.emit(line),
                    cancel_event=self.cancel_event,
                )
                for spec in specs:
                    if spec not in installed_all:
                        installed_all.append(spec)
            self.install_succeeded.emit(tuple(installed_all))
        except RuntimeInstallCancelledError:
            self.install_cancelled.emit()
        except Exception as exc:
            self.install_failed.emit(str(exc))


class RuntimeInstallDialog(QtWidgets.QDialog):
    """Modal progress/confirmation dialog that installs capabilities off the UI thread."""

    def __init__(
        self,
        statuses: Sequence[CapabilityStatus],
        *,
        feature_title: str | None = None,
        auto_start: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._statuses = tuple(statuses)
        self._feature_title = (feature_title or "").strip()
        self._worker: _CapabilityInstallWorker | None = None
        self.installed_specs: tuple[str, ...] = ()
        self.error_message: str = ""

        titles = ", ".join(st.title or st.name for st in self._statuses)
        self.setWindowTitle(self.tr("Install Optional Runtime Packages"))
        self.setMinimumWidth(620)
        self.setMinimumHeight(380)
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)

        headline = (
            f"<b>{self._feature_title}</b> requires optional runtime packages ({titles})."
            if self._feature_title
            else f"Optional runtime capability required: <b>{titles}</b>."
        )
        self._header_label = QtWidgets.QLabel(headline, self)
        self._header_label.setWordWrap(True)
        layout.addWidget(self._header_label)

        missing_specs: list[str] = []
        approx_mb = 0
        for st in self._statuses:
            approx_mb += int(st.approx_download_mb or 0)
            for spec in st.missing_package_specs:
                if spec not in missing_specs:
                    missing_specs.append(spec)

        summary_parts = [
            f"Packages to install: <code>{', '.join(missing_specs) or titles}</code>"
        ]
        if approx_mb > 0:
            summary_parts.append(
                f"Estimated download: ~{approx_mb} MB (cached by uv/pip)"
            )
        summary_parts.append(f"Target Python: <code>{sys.executable}</code>")
        self._summary_label = QtWidgets.QLabel("<br>".join(summary_parts), self)
        self._summary_label.setWordWrap(True)
        self._summary_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        layout.addWidget(self._summary_label)

        self._progress_bar = QtWidgets.QProgressBar(self)
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        layout.addWidget(self._progress_bar)

        self._log_view = QtWidgets.QPlainTextEdit(self)
        self._log_view.setReadOnly(True)
        font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        self._log_view.setFont(font)
        self._log_view.setPlaceholderText(
            self.tr("Click 'Install Now' to download and enable this feature...")
        )
        layout.addWidget(self._log_view, stretch=1)

        self._auto_confirm_checkbox = QtWidgets.QCheckBox(
            self.tr(
                "Install missing optional packages automatically for the rest of this session"
            ),
            self,
        )
        self._auto_confirm_checkbox.setChecked(_SESSION_AUTO_CONFIRM)
        layout.addWidget(self._auto_confirm_checkbox)

        btn_row = QtWidgets.QHBoxLayout()
        self._copy_cmd_btn = QtWidgets.QPushButton(self.tr("Copy pip Command"), self)
        self._copy_cmd_btn.clicked.connect(self._copy_install_command)
        btn_row.addWidget(self._copy_cmd_btn)
        btn_row.addStretch(1)

        self._cancel_btn = QtWidgets.QPushButton(self.tr("Cancel"), self)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._cancel_btn)

        self._install_btn = QtWidgets.QPushButton(self.tr("Install Now"), self)
        self._install_btn.setDefault(True)
        self._install_btn.clicked.connect(self._start_install)
        btn_row.addWidget(self._install_btn)

        layout.addLayout(btn_row)

        if auto_start:
            QtCore.QTimer.singleShot(0, self._start_install)

    def _manual_command_text(self) -> str:
        commands = [st.install_command for st in self._statuses if st.install_command]
        return "\n".join(dict.fromkeys(commands))

    def _copy_install_command(self) -> None:
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(self._manual_command_text())
            self._append_log("Copied install command to clipboard.")

    def _append_log(self, line: str) -> None:
        self._log_view.appendPlainText(line)
        scrollbar = self._log_view.verticalScrollBar()
        if scrollbar is not None:
            scrollbar.setValue(scrollbar.maximum())

    def _start_install(self) -> None:
        global _SESSION_AUTO_CONFIRM
        if self._worker is not None and self._worker.isRunning():
            return
        _SESSION_AUTO_CONFIRM = bool(self._auto_confirm_checkbox.isChecked())
        self._install_btn.setEnabled(False)
        self._auto_confirm_checkbox.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._log_view.clear()

        cap_names = tuple(st.name for st in self._statuses)
        worker = _CapabilityInstallWorker(cap_names, parent=self)
        worker.output_line.connect(self._append_log)
        worker.install_succeeded.connect(self._on_succeeded)
        worker.install_cancelled.connect(self._on_cancelled)
        worker.install_failed.connect(self._on_failed)
        self._worker = worker
        worker.start()

    def _on_succeeded(self, specs: object) -> None:
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(1)
        if isinstance(specs, tuple):
            self.installed_specs = specs
        self._append_log("Installation completed successfully.")
        self.accept()

    def _on_cancelled(self) -> None:
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self._append_log("Installation cancelled.")
        self._install_btn.setEnabled(True)
        self._auto_confirm_checkbox.setEnabled(True)
        self.reject()

    def _on_failed(self, message: str) -> None:
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self.error_message = message
        self._append_log("")
        self._append_log("Installation failed:")
        self._append_log(message)
        self._install_btn.setText(self.tr("Retry Install"))
        self._install_btn.setEnabled(True)
        self._auto_confirm_checkbox.setEnabled(True)

    def _on_cancel_clicked(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._cancel_btn.setEnabled(False)
            self._append_log("Cancelling installation...")
            self._worker.request_cancel()
            return
        self.reject()

    def reject(self) -> None:
        if self._worker is not None and self._worker.isRunning():
            self._worker.request_cancel()
            self._worker.wait(2500)
        super().reject()


def _show_readonly_capability_message(
    parent: QtWidgets.QWidget | None,
    statuses: Sequence[CapabilityStatus],
    *,
    feature_title: str | None = None,
) -> None:
    lines = [capability_message(st) for st in statuses]
    if getattr(sys, "frozen", False):
        lines.append(
            "\nFrozen desktop bundles are read-only. Install this feature in a "
            'Python environment (`pip install "annolid[gui,...]"`) and launch '
            "Annolid from that environment."
        )
    else:
        lines.append(
            "\nAutomatic optional package installation is disabled by "
            "ANNOLID_AUTO_INSTALL_OPTIONAL_DEPS=0."
        )
    box = QtWidgets.QMessageBox(parent)
    box.setIcon(QtWidgets.QMessageBox.Information)
    box.setWindowTitle(feature_title or "Optional Runtime Packages Required")
    box.setText("\n\n".join(lines))
    copy_btn = box.addButton("Copy Command", QtWidgets.QMessageBox.ActionRole)
    box.addButton(QtWidgets.QMessageBox.Ok)
    box.exec_()
    if box.clickedButton() == copy_btn:
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(
                "\n".join(dict.fromkeys(st.install_command for st in statuses))
            )


def ensure_gui_capabilities(
    parent: QtWidgets.QWidget | None,
    names: Sequence[str],
    *,
    feature_title: str | None = None,
    prompt_user: bool = True,
) -> bool:
    """Ensure one or more capabilities exist without blocking the Qt UI thread.

    Returns ``True`` when all requested capabilities are available (either
    already installed or installed during this call), and ``False`` if the user
    cancels or installation is unavailable/fails.
    """
    missing_statuses: list[CapabilityStatus] = []
    for raw_name in names:
        status = check_capability(raw_name)
        if not status.available:
            missing_statuses.append(status)

    if not missing_statuses:
        return True

    if not _on_main_gui_thread():
        try:
            for st in missing_statuses:
                ensure_capability(st.name)
            return True
        except Exception as exc:
            logger.warning("Background capability install failed: %s", exc)
            return False

    if getattr(sys, "frozen", False) or not _auto_install_optional_deps_enabled():
        _show_readonly_capability_message(
            parent, missing_statuses, feature_title=feature_title
        )
        return False

    auto_start = (not prompt_user) or _SESSION_AUTO_CONFIRM
    dlg = RuntimeInstallDialog(
        missing_statuses,
        feature_title=feature_title,
        auto_start=auto_start,
        parent=parent,
    )
    accepted = dlg.exec_() == QtWidgets.QDialog.Accepted
    if not accepted:
        return False
    return all(check_capability(st.name).available for st in missing_statuses)


def ensure_gui_capability(
    parent: QtWidgets.QWidget | None,
    name: str,
    *,
    feature_title: str | None = None,
    prompt_user: bool = True,
) -> bool:
    """Ensure a single capability exists without blocking the Qt UI thread."""
    return ensure_gui_capabilities(
        parent,
        (name,),
        feature_title=feature_title,
        prompt_user=prompt_user,
    )


def _interactive_main_thread_handler(
    capability_name: str, reason: str
) -> tuple[str, ...] | None:
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    if not _on_main_gui_thread():
        return None
    status = check_capability(capability_name)
    if status.available:
        return ()
    parent = QtWidgets.QApplication.activeWindow()
    dlg = RuntimeInstallDialog(
        (status,),
        feature_title=reason or status.title or status.name,
        auto_start=_SESSION_AUTO_CONFIRM,
        parent=parent,
    )
    if dlg.exec_() != QtWidgets.QDialog.Accepted:
        raise RuntimeInstallCancelledError(
            f"Installation of optional runtime capability '{capability_name}' was cancelled."
        )
    return dlg.installed_specs


def register_gui_capability_handler() -> None:
    """Route main-thread ensure_capability calls through the non-blocking Qt dialog."""
    set_interactive_install_handler(_interactive_main_thread_handler)


__all__ = [
    "RuntimeInstallDialog",
    "ensure_gui_capabilities",
    "ensure_gui_capability",
    "register_gui_capability_handler",
]
