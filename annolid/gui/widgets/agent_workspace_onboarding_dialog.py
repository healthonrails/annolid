from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from qtpy import QtWidgets

from annolid.infrastructure.agent_workspace import get_agent_workspace_path
from annolid.services.agent_cron import (
    get_agent_status,
    onboard_agent_workspace,
    restore_agent_workspace_backup,
)


class AgentWorkspaceOnboardingDialog(QtWidgets.QDialog):
    """Guided Annolid Bot workspace onboarding/update dialog."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Annolid Bot Workspace Onboarding")
        self.resize(860, 620)
        self._build_ui()
        self._refresh_status()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        intro = QtWidgets.QLabel(
            "Guide:\n"
            "1. Preview planned workspace template actions.\n"
            "2. Initialize or update templates in ~/.annolid/workspace.\n"
            "3. Optionally prune stale bootstrap-managed files.\n"
            "4. Verify workspace template status."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        path_row = QtWidgets.QHBoxLayout()
        path_row.addWidget(QtWidgets.QLabel("Workspace:"))
        self.workspace_edit = QtWidgets.QLineEdit(str(get_agent_workspace_path()))
        path_row.addWidget(self.workspace_edit, 1)
        browse = QtWidgets.QPushButton("Browse…")
        browse.clicked.connect(self._choose_workspace)
        path_row.addWidget(browse)
        layout.addLayout(path_row)

        opts = QtWidgets.QGroupBox("Options")
        opts_layout = QtWidgets.QGridLayout(opts)
        self.update_check = QtWidgets.QCheckBox("Update existing templates")
        self.prune_check = QtWidgets.QCheckBox("Prune stale bootstrap files")
        self.backup_check = QtWidgets.QCheckBox("Backup before overwrite/prune")
        self.backup_check.setChecked(True)
        self.dry_run_check = QtWidgets.QCheckBox("Dry run (no file writes)")
        opts_layout.addWidget(self.update_check, 0, 0)
        opts_layout.addWidget(self.prune_check, 0, 1)
        opts_layout.addWidget(self.backup_check, 1, 0)
        opts_layout.addWidget(self.dry_run_check, 1, 1)
        layout.addWidget(opts)

        button_row = QtWidgets.QHBoxLayout()
        self.preview_btn = QtWidgets.QPushButton("Preview")
        self.preview_btn.clicked.connect(self._run_preview)
        button_row.addWidget(self.preview_btn)
        self.apply_btn = QtWidgets.QPushButton("Apply")
        self.apply_btn.clicked.connect(self._run_apply)
        button_row.addWidget(self.apply_btn)
        self.restore_btn = QtWidgets.QPushButton("Restore Latest Backup")
        self.restore_btn.clicked.connect(self._restore_latest_backup)
        button_row.addWidget(self.restore_btn)
        self.status_btn = QtWidgets.QPushButton("Refresh Status")
        self.status_btn.clicked.connect(self._refresh_status)
        button_row.addWidget(self.status_btn)
        button_row.addStretch(1)
        self.close_btn = QtWidgets.QPushButton("Close")
        self.close_btn.clicked.connect(self.close)
        button_row.addWidget(self.close_btn)
        layout.addLayout(button_row)

        self.summary_label = QtWidgets.QLabel("No actions run yet.")
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        self.results_table = QtWidgets.QTableWidget(0, 3, self)
        self.results_table.setHorizontalHeaderLabels(["Group", "Path", "Status"])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.verticalHeader().setVisible(False)
        self.results_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.results_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        layout.addWidget(self.results_table, 1)

    def _choose_workspace(self) -> None:
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select Agent Workspace",
            self.workspace_edit.text().strip() or str(get_agent_workspace_path()),
        )
        if selected:
            self.workspace_edit.setText(selected)

    def _workspace(self) -> str:
        value = self.workspace_edit.text().strip()
        if value:
            return str(Path(value).expanduser())
        return str(get_agent_workspace_path())

    def _build_onboard_payload(self, *, force_dry_run: bool) -> Dict[str, Any]:
        overwrite = bool(self.update_check.isChecked())
        dry_run = bool(force_dry_run or self.dry_run_check.isChecked())
        return onboard_agent_workspace(
            workspace=self._workspace(),
            overwrite=overwrite,
            dry_run=dry_run,
            backup=bool(self.backup_check.isChecked()),
            prune_bootstrap=bool(self.prune_check.isChecked()),
        )

    def _run_preview(self) -> None:
        payload = self._build_onboard_payload(force_dry_run=True)
        self._render_onboard_result(payload, title="Preview completed.")
        self._refresh_status()

    def _run_apply(self) -> None:
        payload = self._build_onboard_payload(force_dry_run=False)
        self._render_onboard_result(payload, title="Apply completed.")
        self._refresh_status()

    def _restore_latest_backup(self) -> None:
        dry_run = bool(self.dry_run_check.isChecked())
        if not dry_run:
            answer = QtWidgets.QMessageBox.question(
                self,
                "Restore Latest Backup",
                "Restore files from the latest workspace bootstrap backup?\n\n"
                "Current files will be backed up before restore.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return
        payload = restore_agent_workspace_backup(
            workspace=self._workspace(),
            latest=True,
            dry_run=dry_run,
            backup_before_restore=True,
        )
        self._render_restore_result(payload)
        self._refresh_status()

    def _refresh_status(self) -> None:
        payload = get_agent_status()
        templates = dict(payload.get("workspace_templates") or {})
        rows = []
        missing = 0
        for path, ok in sorted(templates.items()):
            status = "present" if bool(ok) else "missing"
            if not ok:
                missing += 1
            rows.append(("status", str(path), status))
        self._set_rows(rows)
        self.summary_label.setText(
            f"Workspace status: {payload.get('workspace')} | "
            f"{len(templates) - missing}/{len(templates)} templates present."
        )

    def _render_onboard_result(self, payload: Dict[str, Any], *, title: str) -> None:
        rows = []
        for path, status in sorted(dict(payload.get("files") or {}).items()):
            rows.append(("sync", str(path), str(status)))
        for path, status in sorted(dict(payload.get("pruned_files") or {}).items()):
            rows.append(("prune", str(path), str(status)))
        self._set_rows(rows)

        summary = dict(payload.get("summary") or {})
        summary_text = ", ".join(f"{k}={v}" for k, v in sorted(summary.items()))
        backup_dir = payload.get("backup_dir")
        details = (
            f"{title} workspace={payload.get('workspace')} "
            f"dry_run={payload.get('dry_run')} update={payload.get('overwrite')} "
            f"prune={payload.get('prune_bootstrap')} "
            f"summary: {summary_text or 'none'}"
        )
        if backup_dir:
            details += f" | backup_dir={backup_dir}"
        self.summary_label.setText(details)

    def _render_restore_result(self, payload: Dict[str, Any]) -> None:
        if not bool(payload.get("restored")):
            self.summary_label.setText(
                f"Restore skipped: {payload.get('reason') or 'no backup found'}"
            )
            self._set_rows([])
            return
        rows = []
        for path, status in sorted(dict(payload.get("files") or {}).items()):
            rows.append(("restore", str(path), str(status)))
        self._set_rows(rows)
        summary = dict(payload.get("summary") or {})
        summary_text = ", ".join(f"{k}={v}" for k, v in sorted(summary.items()))
        details = (
            f"Restore completed from {payload.get('backup_dir')} "
            f"dry_run={payload.get('dry_run')} summary: {summary_text or 'none'}"
        )
        pre_backup = payload.get("pre_restore_backup_dir")
        if pre_backup:
            details += f" | pre_restore_backup_dir={pre_backup}"
        self.summary_label.setText(details)

    def _set_rows(self, rows: list[tuple[str, str, str]]) -> None:
        self.results_table.setRowCount(len(rows))
        for row_idx, (group, path, status) in enumerate(rows):
            self.results_table.setItem(row_idx, 0, QtWidgets.QTableWidgetItem(group))
            self.results_table.setItem(row_idx, 1, QtWidgets.QTableWidgetItem(path))
            self.results_table.setItem(row_idx, 2, QtWidgets.QTableWidgetItem(status))
        self.results_table.resizeColumnsToContents()
