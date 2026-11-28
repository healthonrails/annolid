"""Non-blocking Qt dialog and background workers for Annolid updates."""

from __future__ import annotations

import platform
import sys
import threading
import webbrowser
from typing import Any, Optional

from qtpy import QtCore, QtGui, QtWidgets

from annolid.services.agent_update import (
    check_auto_update_if_due,
    execute_gui_agent_rollback,
    format_update_reason,
    inspect_gui_update_status,
    run_agent_update,
)
from annolid.version import get_version

_DEFAULT_RELEASES_URL = "https://github.com/healthonrails/annolid/releases"


def build_relaunch_command() -> list[str]:
    """Return the command needed to restart the current Annolid GUI process."""
    args = list(sys.argv[1:])
    if getattr(sys, "frozen", False):
        return [sys.executable, *args]
    return [sys.executable, "-m", "annolid.gui.launcher", *args]


class _UpdateCheckWorker(QtCore.QThread):
    check_succeeded = QtCore.Signal(object)
    check_failed = QtCore.Signal(str)

    def __init__(
        self,
        *,
        project: str,
        channel: str,
        timeout_s: float,
        require_signature: bool,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._channel = channel
        self._timeout_s = timeout_s
        self._require_signature = require_signature

    def run(self) -> None:  # pragma: no cover - runs on QThread
        try:
            payload = inspect_gui_update_status(
                project=self._project,
                channel=self._channel,
                timeout_s=self._timeout_s,
                require_signature=self._require_signature,
            )
            self.check_succeeded.emit(payload)
        except Exception as exc:
            self.check_failed.emit(str(exc))


class _UpdateRunWorker(QtCore.QThread):
    output_line = QtCore.Signal(str)
    run_completed = QtCore.Signal(object, int)
    run_failed = QtCore.Signal(str)

    def __init__(
        self,
        *,
        project: str,
        channel: str,
        timeout_s: float,
        require_signature: bool,
        skip_post_check: bool = False,
        force_source_sync: bool = False,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._channel = channel
        self._timeout_s = timeout_s
        self._require_signature = require_signature
        self._skip_post_check = skip_post_check
        self._force_source_sync = force_source_sync
        self.cancel_event = threading.Event()

    def request_cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:  # pragma: no cover - runs on QThread
        try:
            payload, exit_code = run_agent_update(
                project=self._project,
                channel=self._channel,
                timeout_s=self._timeout_s,
                require_signature=self._require_signature,
                execute=True,
                skip_post_check=self._skip_post_check,
                on_output=lambda line: self.output_line.emit(str(line)),
                cancel_event=self.cancel_event,
                force_source_sync=self._force_source_sync,
            )
            self.run_completed.emit(payload, int(exit_code))
        except Exception as exc:
            self.run_failed.emit(str(exc))


class _UpdateRollbackWorker(QtCore.QThread):
    output_line = QtCore.Signal(str)
    rollback_completed = QtCore.Signal(object)
    rollback_failed = QtCore.Signal(str)

    def __init__(
        self,
        *,
        project: str,
        previous_version: str,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project
        self._previous_version = previous_version

    def run(self) -> None:  # pragma: no cover - runs on QThread
        try:
            payload = execute_gui_agent_rollback(
                project=self._project,
                previous_version=self._previous_version,
                on_output=lambda line: self.output_line.emit(str(line)),
            )
            self.rollback_completed.emit(payload)
        except Exception as exc:
            self.rollback_failed.emit(str(exc))


class StartupAutoUpdateWorker(QtCore.QThread):
    """Non-intrusive background check triggered only when auto-update is enabled."""

    update_found = QtCore.Signal(object)

    def __init__(
        self,
        *,
        project: str = "annolid",
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._project = project

    def run(self) -> None:  # pragma: no cover - runs on QThread
        try:
            payload = check_auto_update_if_due(project=self._project)
            if isinstance(payload, dict) and payload.get("update_available"):
                self.update_found.emit(payload)
        except Exception:
            pass


class AnnolidUpdateDialog(QtWidgets.QDialog):
    """Unified, non-blocking dialog for checking, applying, and rolling back updates."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        *,
        project: str = "annolid",
        channel: str = "stable",
        timeout_s: float = 8.0,
        require_signature: bool = False,
        auto_check: bool = True,
        auto_prompt_update: bool = False,
        open_rollback_on_ready: bool = False,
    ) -> None:
        super().__init__(parent)
        self._project = str(project or "annolid").strip() or "annolid"
        self._timeout_s = max(1.0, float(timeout_s))
        self._require_signature = bool(require_signature)
        self._auto_prompt_update = bool(auto_prompt_update)
        self._open_rollback_on_ready = bool(open_rollback_on_ready)
        self._closing = False
        self._busy = False
        self._restart_required = False
        self._status_payload: dict[str, Any] = {}
        self._last_run_payload: dict[str, Any] = {}
        self._active_worker: Optional[QtCore.QThread] = None

        self.setWindowTitle(self.tr("Annolid Software Update"))
        self.setMinimumWidth(640)
        self.setMinimumHeight(460)
        self.setModal(True)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(16, 16, 16, 16)
        root.setSpacing(10)

        self._headline_label = QtWidgets.QLabel(
            self.tr("Checking for Annolid updates…"), self
        )
        headline_font = self._headline_label.font()
        headline_font.setPointSize(max(12, headline_font.pointSize() + 2))
        headline_font.setBold(True)
        self._headline_label.setFont(headline_font)
        self._headline_label.setWordWrap(True)
        root.addWidget(self._headline_label)

        self._detail_label = QtWidgets.QLabel("", self)
        self._detail_label.setWordWrap(True)
        self._detail_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        root.addWidget(self._detail_label)

        info_box = QtWidgets.QGroupBox(self.tr("Installation & Release Info"), self)
        form = QtWidgets.QFormLayout(info_box)
        form.setContentsMargins(12, 10, 12, 10)
        form.setSpacing(6)

        self._current_version_value = QtWidgets.QLabel(
            str(get_version() or "unknown"), info_box
        )
        self._target_version_value = QtWidgets.QLabel("—", info_box)
        self._install_mode_value = QtWidgets.QLabel("—", info_box)
        self._release_date_value = QtWidgets.QLabel("—", info_box)

        self.channel_combo = QtWidgets.QComboBox(info_box)
        self.channel_combo.addItem(self.tr("Stable"), "stable")
        self.channel_combo.addItem(self.tr("Beta (pre-releases)"), "beta")
        self.channel_combo.addItem(self.tr("Dev"), "dev")
        normalized_channel = str(channel or "stable").strip().lower()
        idx = self.channel_combo.findData(normalized_channel)
        self.channel_combo.setCurrentIndex(idx if idx >= 0 else 0)
        self.channel_combo.currentIndexChanged.connect(self._on_channel_changed)

        form.addRow(self.tr("Current version:"), self._current_version_value)
        form.addRow(self.tr("Latest available:"), self._target_version_value)
        form.addRow(self.tr("Release channel:"), self.channel_combo)
        form.addRow(self.tr("Install mode:"), self._install_mode_value)
        form.addRow(self.tr("Release date:"), self._release_date_value)
        root.addWidget(info_box)

        self._progress_bar = QtWidgets.QProgressBar(self)
        self._progress_bar.setRange(0, 0)
        self._progress_bar.setVisible(False)
        root.addWidget(self._progress_bar)

        log_header = QtWidgets.QHBoxLayout()
        log_label = QtWidgets.QLabel(self.tr("Activity & Diagnostics Log"), self)
        log_header.addWidget(log_label)
        log_header.addStretch(1)
        self._copy_diag_btn = QtWidgets.QPushButton(self.tr("Copy Diagnostics"), self)
        self._copy_diag_btn.clicked.connect(self._copy_diagnostics)
        log_header.addWidget(self._copy_diag_btn)
        root.addLayout(log_header)

        self._log_view = QtWidgets.QPlainTextEdit(self)
        self._log_view.setReadOnly(True)
        self._log_view.setFont(
            QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        )
        root.addWidget(self._log_view, stretch=1)

        btn_row = QtWidgets.QHBoxLayout()
        self._check_btn = QtWidgets.QPushButton(self.tr("Check Again"), self)
        self._check_btn.clicked.connect(self.start_check)
        btn_row.addWidget(self._check_btn)

        self._releases_btn = QtWidgets.QPushButton(self.tr("Releases Page"), self)
        self._releases_btn.clicked.connect(self._open_releases_page)
        btn_row.addWidget(self._releases_btn)

        self._rollback_btn = QtWidgets.QPushButton(self.tr("Rollback…"), self)
        self._rollback_btn.clicked.connect(self._prompt_and_start_rollback)
        btn_row.addWidget(self._rollback_btn)

        btn_row.addStretch(1)

        self._restart_btn = QtWidgets.QPushButton(self.tr("Restart Annolid"), self)
        self._restart_btn.setVisible(False)
        self._restart_btn.clicked.connect(self._restart_application)
        btn_row.addWidget(self._restart_btn)

        self._update_btn = QtWidgets.QPushButton(self.tr("Update Now"), self)
        self._update_btn.setDefault(True)
        self._update_btn.setEnabled(False)
        self._update_btn.clicked.connect(self._on_primary_update_clicked)
        btn_row.addWidget(self._update_btn)

        self._close_btn = QtWidgets.QPushButton(self.tr("Close"), self)
        self._close_btn.clicked.connect(self._on_close_or_cancel_clicked)
        btn_row.addWidget(self._close_btn)

        root.addLayout(btn_row)

        if auto_check:
            QtCore.QTimer.singleShot(0, self.start_check)
        else:
            self._headline_label.setText(self.tr("Ready to check for updates"))
            self._detail_label.setText(
                self.tr("Click 'Check Again' to query the selected release channel.")
            )

    def selected_channel(self) -> str:
        return str(self.channel_combo.currentData() or "stable").strip().lower()

    def _append_log(self, text: str) -> None:
        line = str(text or "")
        if line:
            self._log_view.appendPlainText(line)

    def _set_busy(self, busy: bool, *, cancellable: bool = False) -> None:
        self._busy = bool(busy)
        self._progress_bar.setVisible(bool(busy))
        self.channel_combo.setEnabled(not busy)
        self._check_btn.setEnabled(not busy)
        self._rollback_btn.setEnabled(not busy)
        if busy:
            self._update_btn.setEnabled(False)
            self._close_btn.setText(
                self.tr("Cancel") if cancellable else self.tr("Close")
            )
            self._close_btn.setEnabled(cancellable)
        else:
            self._close_btn.setText(self.tr("Close"))
            self._close_btn.setEnabled(True)
            self._refresh_action_buttons()

    def _on_channel_changed(self, _index: int) -> None:
        if not self._busy:
            self.start_check()

    def start_check(self) -> None:
        if self._busy:
            return
        channel = self.selected_channel()
        self._restart_required = False
        self._restart_btn.setVisible(False)
        self._headline_label.setText(
            self.tr("Checking for updates (%s channel)…") % channel
        )
        self._detail_label.setText("")
        self._append_log(f"==> Checking for updates (channel={channel})...")
        self._set_busy(True, cancellable=False)

        worker = _UpdateCheckWorker(
            project=self._project,
            channel=channel,
            timeout_s=self._timeout_s,
            require_signature=self._require_signature,
            parent=self,
        )
        self._active_worker = worker
        worker.check_succeeded.connect(self._on_check_succeeded)
        worker.check_failed.connect(self._on_check_failed)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        worker.start()

    def _on_worker_finished(self, worker: QtCore.QThread) -> None:
        if self._active_worker is worker:
            self._active_worker = None
        worker.deleteLater()

    def _on_check_succeeded(self, payload: object) -> None:
        if self._closing:
            return
        data = dict(payload) if isinstance(payload, dict) else {}
        self._status_payload = data
        self._set_busy(False)
        self._apply_check_payload(data)

        if self._open_rollback_on_ready:
            self._open_rollback_on_ready = False
            QtCore.QTimer.singleShot(0, self._prompt_and_start_rollback)
            return

        if self._auto_prompt_update:
            self._auto_prompt_update = False
            install_mode = str(data.get("install_mode") or "package").lower()
            if data.get("update_available") and install_mode != "frozen":
                QtCore.QTimer.singleShot(0, self._on_primary_update_clicked)

    def _on_check_failed(self, error_text: str) -> None:
        if self._closing:
            return
        self._set_busy(False)
        self._auto_prompt_update = False
        self._open_rollback_on_ready = False
        self._headline_label.setText(self.tr("Could not check for updates"))
        self._detail_label.setText(str(error_text))
        self._append_log(f"Check failed: {error_text}")

    def _format_install_mode_display(self, data: dict[str, Any]) -> str:
        mode = str(data.get("install_mode") or "package").lower()
        if mode == "frozen":
            return self.tr("Desktop bundle (standalone app)")
        if mode == "source":
            branch = str(data.get("git_branch") or "").strip()
            commit = str(data.get("git_commit") or "").strip()
            dirty = bool(data.get("git_dirty", False))
            extras = []
            if branch:
                extras.append(branch)
            if commit:
                extras.append(commit)
            if dirty:
                extras.append("uncommitted changes")
            suffix = f" ({', '.join(extras)})" if extras else ""
            return f"Source checkout{suffix}"
        return self.tr("Python package (%s)") % sys.executable

    def _apply_check_payload(self, data: dict[str, Any]) -> None:
        current = str(data.get("current_version") or get_version() or "unknown")
        target = str(data.get("target_version") or "unknown")
        channel = str(data.get("channel") or self.selected_channel())
        install_mode = str(data.get("install_mode") or "package").lower()
        update_available = bool(data.get("update_available", False))
        up_to_date = bool(data.get("up_to_date", False))
        reason = str(data.get("verification_reason") or "")
        reason_msg = str(
            data.get("verification_message") or format_update_reason(reason)
        )

        self._current_version_value.setText(current)
        self._target_version_value.setText(target)
        self._install_mode_value.setText(self._format_install_mode_display(data))
        release_date = str(data.get("release_date") or "").strip()
        self._release_date_value.setText(release_date[:10] if release_date else "—")

        if update_available and install_mode == "frozen":
            self._headline_label.setText(
                self.tr("Update available: v%s → v%s") % (current, target)
            )
            self._detail_label.setText(
                self.tr(
                    "Desktop app bundles cannot self-update in place. "
                    "Click 'Download Release' to get the latest build from GitHub Releases."
                )
            )
        elif update_available:
            self._headline_label.setText(
                self.tr("Update available: v%s → v%s") % (current, target)
            )
            if install_mode == "source" and data.get("git_dirty"):
                self._detail_label.setText(
                    self.tr(
                        "A newer release is available on %s, but your local git checkout "
                        "has uncommitted changes. Commit or stash them before updating."
                    )
                    % channel
                )
            else:
                self._detail_label.setText(
                    self.tr(
                        "A newer release is ready on the %s channel. "
                        "Click 'Update Now' to install and verify it."
                    )
                    % channel
                )
        elif up_to_date:
            self._headline_label.setText(
                self.tr("Annolid is up to date (v%s)") % current
            )
            if install_mode == "source":
                if data.get("git_dirty"):
                    self._detail_label.setText(
                        self.tr(
                            "Latest PyPI release on %s is v%s. Your local git checkout has "
                            "uncommitted changes."
                        )
                        % (channel, target)
                    )
                else:
                    self._detail_label.setText(
                        self.tr(
                            "Latest PyPI release on %s is v%s. Running from a git source "
                            "checkout — you can still pull the latest commits with 'Pull Latest (Git)'."
                        )
                        % (channel, target)
                    )
            else:
                self._detail_label.setText(
                    self.tr("Latest release on %s channel is v%s.") % (channel, target)
                )
        else:
            self._headline_label.setText(self.tr("Update check blocked"))
            self._detail_label.setText(reason_msg)

        self._append_log(
            f"Current: {current} | Target ({channel}): {target} | "
            f"Mode: {install_mode} | Status: {reason or 'ok'}"
        )
        commands = data.get("commands") or []
        if commands and (update_available or install_mode == "source"):
            self._append_log("Planned commands:")
            for cmd in commands:
                self._append_log("  $ " + " ".join(str(p) for p in cmd))

        self._refresh_action_buttons()

    def _refresh_action_buttons(self) -> None:
        data = self._status_payload
        install_mode = str(data.get("install_mode") or "package").lower()
        update_available = bool(data.get("update_available", False))
        up_to_date = bool(data.get("up_to_date", False))
        git_dirty = bool(data.get("git_dirty", False))
        missing_bins = list(data.get("missing_binaries") or [])

        if not data:
            self._update_btn.setText(self.tr("Update Now"))
            self._update_btn.setEnabled(False)
            return

        if install_mode == "frozen":
            self._update_btn.setText(self.tr("Download Release"))
            self._update_btn.setEnabled(True)
            return

        if update_available:
            self._update_btn.setText(self.tr("Update Now"))
            self._update_btn.setEnabled(not git_dirty and not missing_bins)
        elif up_to_date and install_mode == "source":
            self._update_btn.setText(self.tr("Pull Latest (Git)"))
            self._update_btn.setEnabled(not git_dirty and not missing_bins)
        else:
            self._update_btn.setText(self.tr("Update Now"))
            self._update_btn.setEnabled(False)

    def _on_primary_update_clicked(self) -> None:
        if self._busy:
            return
        data = self._status_payload
        install_mode = str(data.get("install_mode") or "package").lower()
        if install_mode == "frozen":
            self._open_releases_page()
            return

        update_available = bool(data.get("update_available", False))
        up_to_date = bool(data.get("up_to_date", False))
        force_source_sync = bool(
            (not update_available) and up_to_date and install_mode == "source"
        )
        if not update_available and not force_source_sync:
            return

        channel = self.selected_channel()
        current = str(data.get("current_version") or get_version() or "unknown")
        target = str(data.get("target_version") or "latest")
        if force_source_sync:
            prompt_text = self.tr(
                "Pull the latest commits and reinstall this source checkout?\n\n"
                "Repository: %s\n"
                "An app restart will be recommended afterward."
            ) % str(data.get("source_root") or ".")
        else:
            prompt_text = self.tr(
                "Update Annolid from v%s to v%s (%s channel)?\n\n"
                "This will run package/git commands and post-update health checks."
            ) % (current, target, channel)

        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("Confirm Update"),
            prompt_text,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self.start_update(force_source_sync=force_source_sync)

    def start_update(self, *, force_source_sync: bool = False) -> None:
        if self._busy:
            return
        channel = self.selected_channel()
        self._restart_required = False
        self._restart_btn.setVisible(False)
        self._headline_label.setText(
            self.tr("Pulling latest source updates…")
            if force_source_sync
            else self.tr("Updating Annolid…")
        )
        self._detail_label.setText(
            self.tr(
                "Running update commands in the background. You can cancel anytime."
            )
        )
        self._append_log("")
        self._append_log(
            f"==> Starting update (channel={channel}, force_source_sync={force_source_sync})..."
        )
        self._set_busy(True, cancellable=True)

        worker = _UpdateRunWorker(
            project=self._project,
            channel=channel,
            timeout_s=self._timeout_s,
            require_signature=self._require_signature,
            skip_post_check=False,
            force_source_sync=force_source_sync,
            parent=self,
        )
        self._active_worker = worker
        worker.output_line.connect(self._append_log)
        worker.run_completed.connect(self._on_run_completed)
        worker.run_failed.connect(self._on_run_failed)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        worker.start()

    def _on_run_completed(self, payload: object, exit_code: int) -> None:
        if self._closing:
            return
        data = dict(payload) if isinstance(payload, dict) else {}
        self._last_run_payload = data
        self._set_busy(False)

        status = str(data.get("status") or "").strip().lower()
        reason = str(data.get("reason") or status or "unknown_error").strip()
        summary = format_update_reason(reason, data)

        if int(exit_code) == 0 and status in {"updated", "staged"}:
            self._restart_required = bool(data.get("restart_required", False))
            self._headline_label.setText(self.tr("Update completed successfully"))
            if self._restart_required:
                self._detail_label.setText(
                    summary
                    + "\n"
                    + self.tr("Restart Annolid to finish applying the update.")
                )
                self._restart_btn.setVisible(True)
            else:
                self._detail_label.setText(summary)
            self._append_log(f"==> Update finished: status={status}")
            return

        if status == "up_to_date":
            self._headline_label.setText(self.tr("Annolid is already up to date"))
            self._detail_label.setText(summary)
            self._append_log(f"==> {summary}")
            return

        if status == "cancelled":
            self._headline_label.setText(self.tr("Update cancelled"))
            self._detail_label.setText(summary)
            self._append_log("==> Update cancelled by user.")
            return

        if status == "manual_required":
            self._headline_label.setText(self.tr("Manual download required"))
            self._detail_label.setText(summary)
            self._append_log(f"==> {summary}")
            return

        self._headline_label.setText(self.tr("Update did not complete"))
        self._detail_label.setText(summary)
        self._append_log(f"==> Update failed (status={status}, reason={reason})")
        rollback = data.get("rollback")
        if isinstance(rollback, dict):
            rb_ok = bool(rollback.get("ok", False))
            rb_msg = str(
                rollback.get("message")
                or rollback.get("manual_instructions")
                or rollback.get("reason")
                or ""
            ).strip()
            self._append_log(f"Rollback result: ok={rb_ok} {rb_msg}".strip())

    def _on_run_failed(self, error_text: str) -> None:
        if self._closing:
            return
        self._set_busy(False)
        self._headline_label.setText(self.tr("Update failed"))
        self._detail_label.setText(str(error_text))
        self._append_log(f"==> Update exception: {error_text}")

    def _prompt_and_start_rollback(self) -> None:
        if self._busy:
            return
        install_mode = str(
            self._status_payload.get("install_mode") or "package"
        ).lower()
        if install_mode == "frozen":
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Rollback"),
                self.tr(
                    "Desktop app bundles cannot roll back automatically. "
                    "Download a previous release from GitHub Releases."
                ),
            )
            return

        suggested = str(
            self._status_payload.get("last_known_good_version") or ""
        ).strip()
        previous_version, ok = QtWidgets.QInputDialog.getText(
            self,
            self.tr("Rollback Annolid"),
            self.tr("Previous known-good version:"),
            QtWidgets.QLineEdit.Normal,
            suggested,
        )
        if not ok:
            return
        version = str(previous_version or "").strip()
        if not version:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Rollback"),
                self.tr("Rollback cancelled: previous version was empty."),
            )
            return

        reply = QtWidgets.QMessageBox.question(
            self,
            self.tr("Confirm Rollback"),
            self.tr(
                "Run rollback to version %s now?\n\n"
                "This may run package manager commands and require an app restart."
            )
            % version,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if reply != QtWidgets.QMessageBox.Yes:
            return

        self._headline_label.setText(self.tr("Rolling back to v%s…") % version)
        self._detail_label.setText(self.tr("Running rollback in the background…"))
        self._append_log(f"\n==> Starting rollback to {version}...")
        self._set_busy(True, cancellable=False)

        worker = _UpdateRollbackWorker(
            project=self._project,
            previous_version=version,
            parent=self,
        )
        self._active_worker = worker
        worker.output_line.connect(self._append_log)
        worker.rollback_completed.connect(self._on_rollback_completed)
        worker.rollback_failed.connect(self._on_rollback_failed)
        worker.finished.connect(lambda w=worker: self._on_worker_finished(w))
        worker.start()

    def _on_rollback_completed(self, payload: object) -> None:
        if self._closing:
            return
        data = dict(payload) if isinstance(payload, dict) else {}
        self._set_busy(False)
        if bool(data.get("ok", False)):
            self._restart_required = True
            self._restart_btn.setVisible(True)
            self._headline_label.setText(self.tr("Rollback completed successfully"))
            self._detail_label.setText(
                self.tr("Restart Annolid to run the rolled-back version.")
            )
            self._append_log("==> Rollback completed successfully.")
            return
        reason = str(data.get("reason") or "rollback_failed")
        summary = format_update_reason(reason, data)
        self._headline_label.setText(self.tr("Rollback did not complete"))
        self._detail_label.setText(summary)
        self._append_log(f"==> Rollback result: {summary}")

    def _on_rollback_failed(self, error_text: str) -> None:
        if self._closing:
            return
        self._set_busy(False)
        self._headline_label.setText(self.tr("Rollback failed"))
        self._detail_label.setText(str(error_text))
        self._append_log(f"==> Rollback exception: {error_text}")

    def _open_releases_page(self) -> None:
        url = str(self._status_payload.get("release_url") or _DEFAULT_RELEASES_URL)
        try:
            webbrowser.open(url)
        except Exception:
            pass

    def _copy_diagnostics(self) -> None:
        lines = [
            f"Annolid version: {get_version()}",
            f"Python: {sys.version.split()[0]} ({sys.executable})",
            f"Platform: {platform.platform()}",
            f"Channel: {self.selected_channel()}",
            f"Install mode: {self._status_payload.get('install_mode', 'unknown')}",
            f"Source root: {self._status_payload.get('source_root', '')}",
            f"Target version: {self._status_payload.get('target_version', '')}",
            f"Verification: {self._status_payload.get('verification_reason', '')}",
            "",
            "--- Log ---",
            self._log_view.toPlainText(),
        ]
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(lines))
        self._append_log("Diagnostics copied to clipboard.")

    def _restart_application(self) -> None:
        parent_win = self.parentWidget()
        cmd = build_relaunch_command()
        self.accept()
        if parent_win is not None and not parent_win.close():
            return
        try:
            QtCore.QProcess.startDetached(cmd[0], cmd[1:])
        except Exception:
            pass
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.quit()

    def _on_close_or_cancel_clicked(self) -> None:
        worker = self._active_worker
        if self._busy and isinstance(worker, _UpdateRunWorker):
            worker.request_cancel()
            self._close_btn.setEnabled(False)
            self._append_log("Cancellation requested…")
            return
        if not self._busy:
            self.reject()

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self._closing = True
        worker = self._active_worker
        if worker is not None and worker.isRunning():
            if isinstance(worker, _UpdateRunWorker):
                worker.request_cancel()
            worker.wait(3000)
        super().closeEvent(event)
