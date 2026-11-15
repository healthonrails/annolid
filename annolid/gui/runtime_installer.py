"""Non-blocking Qt helper for installing optional runtime capabilities on demand."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import threading
from typing import Sequence

from qtpy import QtCore, QtGui, QtWidgets

from annolid.infrastructure.capabilities import (
    CapabilityStatus,
    FROZEN_USER_RUNTIME_ENV,
    RuntimeInstallCancelledError,
    _auto_install_optional_deps_enabled,
    can_install_in_frozen_user_runtime,
    capability_message,
    check_capability,
    ensure_capability,
    export_wheelhouse_bundle,
    frozen_user_runtime_enabled,
    install_capability_stream,
    is_offline_mode,
    resolve_wheelhouse_dir,
    set_interactive_install_handler,
    user_runtime_site_dir,
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
        *,
        wheelhouse: str | None = None,
        offline: bool | None = None,
        allow_frozen_user_runtime: bool = False,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._capability_names = capability_names
        self._wheelhouse = wheelhouse
        self._offline = offline
        self._allow_frozen_user_runtime = allow_frozen_user_runtime
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
                    wheelhouse=self._wheelhouse,
                    offline=self._offline,
                    allow_frozen_user_runtime=self._allow_frozen_user_runtime,
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


class _WheelhouseExportWorker(QtCore.QThread):
    output_line = QtCore.Signal(str)
    export_succeeded = QtCore.Signal(object)
    export_cancelled = QtCore.Signal()
    export_failed = QtCore.Signal(str)

    def __init__(
        self,
        output_dir: str,
        capability_names: tuple[str, ...],
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._output_dir = output_dir
        self._capability_names = capability_names
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:  # pragma: no cover - exercised via QThread event loop
        try:
            manifest = export_wheelhouse_bundle(
                self._output_dir,
                capability_names=self._capability_names,
                on_output=lambda line: self.output_line.emit(line),
                cancel_event=self.cancel_event,
            )
            self.export_succeeded.emit(manifest)
        except RuntimeInstallCancelledError:
            self.export_cancelled.emit()
        except Exception as exc:
            self.export_failed.emit(str(exc))


class RuntimeInstallDialog(QtWidgets.QDialog):
    """Modal progress/confirmation dialog that installs capabilities off the UI thread."""

    def __init__(
        self,
        statuses: Sequence[CapabilityStatus],
        *,
        feature_title: str | None = None,
        auto_start: bool = False,
        wheelhouse: str | None = None,
        offline: bool | None = None,
        allow_frozen_user_runtime: bool = False,
        parent: QtWidgets.QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._statuses = tuple(statuses)
        self._feature_title = (feature_title or "").strip()
        self._worker: _CapabilityInstallWorker | None = None
        self._allow_frozen_user_runtime = allow_frozen_user_runtime
        self.installed_specs: tuple[str, ...] = ()
        self.error_message: str = ""

        try:
            resolved_wh = resolve_wheelhouse_dir(wheelhouse)
        except RuntimeError:
            resolved_wh = None
        self._wheelhouse: str | None = (
            str(resolved_wh) if resolved_wh is not None else wheelhouse
        )
        self._offline: bool | None = offline

        titles = ", ".join(st.title or st.name for st in self._statuses)
        self.setWindowTitle(self.tr("Install Optional Runtime Packages"))
        self.setMinimumWidth(650)
        self.setMinimumHeight(400)
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

        self._summary_label = QtWidgets.QLabel(self)
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

        self._offline_btn = QtWidgets.QPushButton(
            self.tr("Select Offline Folder…"), self
        )
        self._offline_btn.clicked.connect(self._choose_wheelhouse_folder)
        btn_row.addWidget(self._offline_btn)

        btn_row.addStretch(1)

        self._cancel_btn = QtWidgets.QPushButton(self.tr("Cancel"), self)
        self._cancel_btn.clicked.connect(self._on_cancel_clicked)
        btn_row.addWidget(self._cancel_btn)

        self._install_btn = QtWidgets.QPushButton(self.tr("Install Now"), self)
        self._install_btn.setDefault(True)
        self._install_btn.clicked.connect(self._start_install)
        btn_row.addWidget(self._install_btn)

        layout.addLayout(btn_row)

        self._refresh_summary_and_state()

        if auto_start and self._install_btn.isEnabled():
            QtCore.QTimer.singleShot(0, self._start_install)

    def _can_install_now(self) -> bool:
        if not getattr(sys, "frozen", False):
            return True
        if not self._allow_frozen_user_runtime:
            return False
        can_install, _backend = can_install_in_frozen_user_runtime(self._wheelhouse)
        return can_install

    def _refresh_summary_and_state(self) -> None:
        titles = ", ".join(st.title or st.name for st in self._statuses)
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
        if approx_mb > 0 and not self._wheelhouse:
            summary_parts.append(
                f"Estimated download: ~{approx_mb} MB (cached by uv/pip)"
            )
        if getattr(sys, "frozen", False):
            summary_parts.append(
                f"Target User Runtime: <code>{user_runtime_site_dir()}</code> "
                "(desktop app bundle remains read-only)"
            )
        else:
            summary_parts.append(f"Target Python: <code>{sys.executable}</code>")

        if self._wheelhouse:
            mode_tag = (
                "offline (--no-index)"
                if is_offline_mode(self._offline)
                else "preferred local cache"
            )
            summary_parts.append(
                f"Offline Wheelhouse: <code>{self._wheelhouse}</code> ({mode_tag})"
            )
        self._summary_label.setText("<br>".join(summary_parts))

        can_install = self._can_install_now()
        self._install_btn.setEnabled(can_install)
        if can_install:
            self._log_view.setPlaceholderText(
                self.tr("Click 'Install Now' to enable this feature...")
            )
        else:
            self._log_view.setPlaceholderText(
                self.tr(
                    "No external Python/uv installer detected for this frozen bundle. "
                    "Click 'Select Offline Folder…' to install from a local wheelhouse directory."
                )
            )

    def set_wheelhouse_dir(self, folder: str | None, *, offline: bool = True) -> None:
        cleaned = (folder or "").strip()
        self._wheelhouse = cleaned or None
        self._offline = offline if cleaned else None
        self._refresh_summary_and_state()

    def _choose_wheelhouse_folder(self) -> None:
        start_dir = self._wheelhouse or str(Path.home())
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            self.tr("Select Offline Wheelhouse Directory"),
            start_dir,
        )
        if chosen:
            self.set_wheelhouse_dir(chosen, offline=True)
            self._append_log(f"Selected offline wheelhouse: {chosen}")

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
        self._offline_btn.setEnabled(False)
        self._auto_confirm_checkbox.setEnabled(False)
        self._progress_bar.setVisible(True)
        self._progress_bar.setRange(0, 0)
        self._log_view.clear()

        cap_names = tuple(st.name for st in self._statuses)
        worker = _CapabilityInstallWorker(
            cap_names,
            wheelhouse=self._wheelhouse,
            offline=self._offline,
            allow_frozen_user_runtime=self._allow_frozen_user_runtime,
            parent=self,
        )
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
        self._offline_btn.setEnabled(True)
        self._auto_confirm_checkbox.setEnabled(True)
        self._refresh_summary_and_state()
        self.reject()

    def _on_failed(self, message: str) -> None:
        self._progress_bar.setRange(0, 1)
        self._progress_bar.setValue(0)
        self.error_message = message
        self._append_log("")
        self._append_log("Installation failed:")
        self._append_log(message)
        self._install_btn.setText(self.tr("Retry Install"))
        self._offline_btn.setEnabled(True)
        self._auto_confirm_checkbox.setEnabled(True)
        self._refresh_summary_and_state()

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
            f"\nUser-space runtime installation is disabled ({FROZEN_USER_RUNTIME_ENV}=0). "
            "Enable user runtime extensions or install this feature in a Python "
            'environment (`pip install "annolid[gui,...]"`) and launch Annolid from there.'
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
    wheelhouse: str | None = None,
    offline: bool | None = None,
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

    is_frozen = bool(getattr(sys, "frozen", False))

    if not _on_main_gui_thread():
        try:
            for st in missing_statuses:
                ensure_capability(
                    st.name,
                    wheelhouse=wheelhouse,
                    offline=offline,
                    allow_frozen_user_runtime=is_frozen
                    and frozen_user_runtime_enabled(),
                )
            return True
        except Exception as exc:
            logger.warning("Background capability install failed: %s", exc)
            return False

    if not _auto_install_optional_deps_enabled() or (
        is_frozen and not frozen_user_runtime_enabled()
    ):
        _show_readonly_capability_message(
            parent, missing_statuses, feature_title=feature_title
        )
        return False

    auto_start = (not prompt_user) or _SESSION_AUTO_CONFIRM
    dlg = RuntimeInstallDialog(
        missing_statuses,
        feature_title=feature_title,
        auto_start=auto_start,
        wheelhouse=wheelhouse,
        offline=offline,
        allow_frozen_user_runtime=is_frozen,
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
    wheelhouse: str | None = None,
    offline: bool | None = None,
) -> bool:
    """Ensure a single capability exists without blocking the Qt UI thread."""
    return ensure_gui_capabilities(
        parent,
        (name,),
        feature_title=feature_title,
        prompt_user=prompt_user,
        wheelhouse=wheelhouse,
        offline=offline,
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
        allow_frozen_user_runtime=bool(getattr(sys, "frozen", False)),
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
    "_WheelhouseExportWorker",
    "ensure_gui_capabilities",
    "ensure_gui_capability",
    "register_gui_capability_handler",
]
