"""GUI manager for inspecting and installing optional Annolid runtime packages."""

from __future__ import annotations

import platform
import shutil
import sys

from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.runtime_installer import ensure_gui_capabilities
from annolid.infrastructure.capabilities import (
    INSTALL_PROFILES,
    check_capability,
    detect_accelerator_summary,
    list_capabilities,
)


class RuntimeManagerDialog(QtWidgets.QDialog):
    """Professional runtime & optional package manager dialog."""

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(self.tr("Runtime & Optional Packages"))
        self.setMinimumSize(860, 560)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(12)

        intro = QtWidgets.QLabel(
            self.tr(
                "<b>Lean Core + On-Demand Runtimes</b> — Annolid installs a small, "
                "fast core shell upfront and downloads heavy ML, tracking, and viewer "
                "packages only when you use them or pre-install them below."
            ),
            self,
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self._env_group = QtWidgets.QGroupBox(self.tr("Active Environment"), self)
        env_layout = QtWidgets.QVBoxLayout(self._env_group)
        self._env_label = QtWidgets.QLabel(self._env_group)
        self._env_label.setWordWrap(True)
        self._env_label.setTextInteractionFlags(QtCore.Qt.TextSelectableByMouse)
        env_layout.addWidget(self._env_label)
        layout.addWidget(self._env_group)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel(self.tr("Filter:"), self))
        self._filter_edit = QtWidgets.QLineEdit(self)
        self._filter_edit.setPlaceholderText(
            self.tr("Filter by capability name, category, or package...")
        )
        self._filter_edit.textChanged.connect(self._apply_filter)
        filter_row.addWidget(self._filter_edit, stretch=1)

        filter_row.addWidget(QtWidgets.QLabel(self.tr("Quick Profile:"), self))
        self._profile_combo = QtWidgets.QComboBox(self)
        self._profile_combo.addItem(self.tr("Select profile..."), "")
        for profile_key in ("tracking", "workstation", "full"):
            meta = INSTALL_PROFILES.get(profile_key, {})
            desc = str(meta.get("description", ""))
            self._profile_combo.addItem(f"{profile_key} — {desc[:48]}", profile_key)
        self._profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        filter_row.addWidget(self._profile_combo)
        layout.addLayout(filter_row)

        self._table = QtWidgets.QTableWidget(self)
        self._table.setColumnCount(6)
        self._table.setHorizontalHeaderLabels(
            [
                self.tr("Select"),
                self.tr("Capability"),
                self.tr("Status"),
                self.tr("Category"),
                self.tr("Est. Size"),
                self.tr("Details / Missing Packages"),
            ]
        )
        self._table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self._table.verticalHeader().setVisible(False)
        header = self._table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        layout.addWidget(self._table, stretch=1)

        footer = QtWidgets.QHBoxLayout()
        self._refresh_btn = QtWidgets.QPushButton(self.tr("Refresh"), self)
        self._refresh_btn.clicked.connect(self.refresh_table)
        footer.addWidget(self._refresh_btn)

        self._copy_btn = QtWidgets.QPushButton(self.tr("Copy pip Command"), self)
        self._copy_btn.clicked.connect(self._copy_selected_commands)
        footer.addWidget(self._copy_btn)

        footer.addStretch(1)

        self._install_btn = QtWidgets.QPushButton(self.tr("Install Checked"), self)
        self._install_btn.setDefault(True)
        self._install_btn.clicked.connect(self._install_checked)
        footer.addWidget(self._install_btn)

        close_btn = QtWidgets.QPushButton(self.tr("Close"), self)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        layout.addLayout(footer)

        self.refresh_table()

    def _format_environment_summary(self) -> str:
        accel = detect_accelerator_summary()
        installer = "uv pip" if shutil.which("uv") else "python -m pip"
        if getattr(sys, "frozen", False):
            mode = "Frozen desktop bundle (read-only)"
        else:
            mode = f"Editable / virtualenv ({installer})"

        if accel.get("has_nvidia_gpu"):
            cuda_ver = accel.get("cuda_version") or "unknown"
            gpu_text = f"NVIDIA GPU ({accel.get('gpu_name')}, CUDA {cuda_ver})"
        elif accel.get("apple_silicon"):
            gpu_text = "Apple Silicon (MPS / CoreML)"
        else:
            gpu_text = f"CPU ({accel.get('platform')} / {accel.get('arch')})"

        return (
            f"Python <b>{platform.python_version()}</b> "
            f"(<code>{sys.executable}</code>)<br>"
            f"Mode: <b>{mode}</b> &nbsp;|&nbsp; Accelerator: <b>{gpu_text}</b>"
        )

    def refresh_table(self) -> None:
        self._env_label.setText(self._format_environment_summary())
        names = list_capabilities()
        self._table.setRowCount(len(names))

        available_count = 0
        for row, name in enumerate(names):
            status = check_capability(name)
            if status.available:
                available_count += 1

            chk_item = QtWidgets.QTableWidgetItem()
            chk_item.setData(QtCore.Qt.UserRole, status.name)
            if status.available or getattr(sys, "frozen", False):
                chk_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                chk_item.setCheckState(QtCore.Qt.Unchecked)
            else:
                chk_item.setFlags(
                    QtCore.Qt.ItemIsUserCheckable
                    | QtCore.Qt.ItemIsSelectable
                    | QtCore.Qt.ItemIsEnabled
                )
                chk_item.setCheckState(QtCore.Qt.Unchecked)
            self._table.setItem(row, 0, chk_item)

            title_text = (
                f"{status.title} ({status.name})" if status.title else status.name
            )
            name_item = QtWidgets.QTableWidgetItem(title_text)
            name_item.setToolTip(status.install_command)
            self._table.setItem(row, 1, name_item)

            state_label = (
                self.tr("Installed") if status.available else self.tr("On-Demand")
            )
            state_item = QtWidgets.QTableWidgetItem(state_label)
            if status.available:
                state_item.setForeground(QtGui.QBrush(QtGui.QColor("#2e7d32")))
            else:
                state_item.setForeground(QtGui.QBrush(QtGui.QColor("#ef6c00")))
            self._table.setItem(row, 2, state_item)

            cat_item = QtWidgets.QTableWidgetItem(status.category)
            self._table.setItem(row, 3, cat_item)

            size_text = (
                f"~{status.approx_download_mb} MB"
                if status.approx_download_mb > 0
                else "-"
            )
            size_item = QtWidgets.QTableWidgetItem(size_text)
            self._table.setItem(row, 4, size_item)

            if status.available:
                detail_text = status.description
            else:
                missing_str = ", ".join(status.missing_package_specs)
                detail_text = f"{status.description} [Missing: {missing_str}]"
            detail_item = QtWidgets.QTableWidgetItem(detail_text)
            detail_item.setToolTip(detail_text)
            self._table.setItem(row, 5, detail_item)

        self._install_btn.setEnabled(not getattr(sys, "frozen", False))
        self._env_group.setTitle(
            self.tr(
                f"Active Environment ({available_count}/{len(names)} optional runtimes installed)"
            )
        )
        self._apply_filter(self._filter_edit.text())

    def _apply_filter(self, text: str) -> None:
        needle = (text or "").strip().lower()
        for row in range(self._table.rowCount()):
            if not needle:
                self._table.setRowHidden(row, False)
                continue
            matched = False
            for col in (1, 2, 3, 5):
                item = self._table.item(row, col)
                if item is not None and needle in item.text().lower():
                    matched = True
                    break
            self._table.setRowHidden(row, not matched)

    def _on_profile_selected(self, index: int) -> None:
        profile_key = str(self._profile_combo.itemData(index) or "").strip()
        if not profile_key:
            return
        meta = INSTALL_PROFILES.get(profile_key, {})
        target_caps = set(meta.get("capabilities", ()))
        for row in range(self._table.rowCount()):
            chk_item = self._table.item(row, 0)
            if chk_item is None:
                continue
            cap_name = str(chk_item.data(QtCore.Qt.UserRole) or "")
            if not (chk_item.flags() & QtCore.Qt.ItemIsUserCheckable):
                continue
            chk_item.setCheckState(
                QtCore.Qt.Checked if cap_name in target_caps else QtCore.Qt.Unchecked
            )

    def checked_capabilities(self) -> list[str]:
        selected: list[str] = []
        for row in range(self._table.rowCount()):
            chk_item = self._table.item(row, 0)
            if chk_item is not None and chk_item.checkState() == QtCore.Qt.Checked:
                cap_name = str(chk_item.data(QtCore.Qt.UserRole) or "").strip()
                if cap_name:
                    selected.append(cap_name)
        return selected

    def _selected_or_checked_capabilities(self) -> list[str]:
        checked = self.checked_capabilities()
        if checked:
            return checked
        rows = sorted({idx.row() for idx in self._table.selectedIndexes()})
        out: list[str] = []
        for row in rows:
            chk_item = self._table.item(row, 0)
            if chk_item is not None:
                cap_name = str(chk_item.data(QtCore.Qt.UserRole) or "").strip()
                if cap_name:
                    out.append(cap_name)
        return out

    def _copy_selected_commands(self) -> None:
        caps = self._selected_or_checked_capabilities()
        if not caps:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Copy pip Command"),
                self.tr("Check or select one or more capabilities first."),
            )
            return
        commands = [check_capability(name).install_command for name in caps]
        clipboard = QtWidgets.QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText("\n".join(dict.fromkeys(commands)))

    def _install_checked(self) -> None:
        caps = self.checked_capabilities()
        if not caps:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Install Optional Runtimes"),
                self.tr(
                    "Check one or more 'On-Demand' capabilities (or choose a Quick Profile) to install."
                ),
            )
            return
        ensure_gui_capabilities(
            self,
            caps,
            feature_title=self.tr("Selected Optional Runtimes"),
            prompt_user=False,
        )
        self.refresh_table()


__all__ = ["RuntimeManagerDialog"]
