from __future__ import annotations

from typing import Dict, Optional, Tuple

from qtpy import QtCore, QtWidgets


class AnnolidLabelDialog(QtWidgets.QDialog):
    """Lightweight LabelMe-compatible label editor dialog.

    Annolid historically used `labelme.widgets.LabelDialog`. We keep a minimal
    in-tree version that supports:
    - label text
    - optional flags (checkboxes)
    - optional group_id
    - description string

    API compatibility used by `annolid/gui/app.py`:
    - `.edit` (QLineEdit)
    - `.popUp(...) -> (text, flags, group_id, description)`
    - `.addLabelHistory(label)`
    """

    def __init__(self, *, parent: QtWidgets.QWidget, config: Optional[dict] = None):
        super().__init__(parent)
        self._config = config or {}
        self._history: list[str] = []

        self.setWindowTitle("Label")
        self.setModal(True)

        layout = QtWidgets.QVBoxLayout(self)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.edit = QtWidgets.QLineEdit(self)
        self.edit.setPlaceholderText("Label")
        form.addRow("Label", self.edit)

        # Description: used by some Annolid tools (e.g. GroundingSAM hints).
        self._description = QtWidgets.QLineEdit(self)
        self._description.setPlaceholderText("Optional description")
        form.addRow("Description", self._description)

        # Optional group id.
        group_row = QtWidgets.QHBoxLayout()
        self._group_enabled = QtWidgets.QCheckBox("Group", self)
        self._group_spin = QtWidgets.QSpinBox(self)
        self._group_spin.setRange(0, 10_000_000)
        self._group_spin.setEnabled(False)
        self._group_enabled.toggled.connect(self._group_spin.setEnabled)
        group_row.addWidget(self._group_enabled)
        group_row.addWidget(self._group_spin, 1)
        group_widget = QtWidgets.QWidget(self)
        group_widget.setLayout(group_row)
        form.addRow("Group ID", group_widget)

        # Flags section: created dynamically in popUp.
        self._flags_box = QtWidgets.QGroupBox("Flags", self)
        self._flags_layout = QtWidgets.QGridLayout(self._flags_box)
        self._flags_box.setVisible(False)
        layout.addWidget(self._flags_box)
        self._flag_checks: Dict[str, QtWidgets.QCheckBox] = {}

        self._buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, self
        )
        self._buttons.accepted.connect(self.accept)
        self._buttons.rejected.connect(self.reject)
        layout.addWidget(self._buttons)

        # Provide a simple completion model from history + configured labels.
        self._completer = QtWidgets.QCompleter(self)
        self._completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        self._completer.setFilterMode(QtCore.Qt.MatchContains)
        self.edit.setCompleter(self._completer)

        self._refresh_completion()

    def addLabelHistory(self, label: str) -> None:
        label = str(label or "").strip()
        if not label:
            return
        if label not in self._history:
            self._history.insert(0, label)
            self._history = self._history[:200]
            self._refresh_completion()

    def _refresh_completion(self) -> None:
        configured = self._config.get("labels") or []
        items = []
        for val in list(self._history) + list(configured):
            s = str(val or "").strip()
            if s and s not in items:
                items.append(s)
        model = QtCore.QStringListModel(items, self._completer)
        self._completer.setModel(model)

    def _clear_flag_widgets(self) -> None:
        for cb in self._flag_checks.values():
            cb.deleteLater()
        self._flag_checks = {}
        while self._flags_layout.count():
            item = self._flags_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _set_flags(self, flags: Dict[str, bool]) -> None:
        self._clear_flag_widgets()
        if not flags:
            self._flags_box.setVisible(False)
            return
        self._flags_box.setVisible(True)

        # Grid: two columns, stable ordering.
        names = sorted(flags.keys())
        for idx, name in enumerate(names):
            cb = QtWidgets.QCheckBox(str(name), self._flags_box)
            cb.setChecked(bool(flags.get(name)))
            row, col = divmod(idx, 2)
            self._flags_layout.addWidget(cb, row, col)
            self._flag_checks[str(name)] = cb

    def _get_flags(self) -> Dict[str, bool]:
        out: Dict[str, bool] = {}
        for name, cb in self._flag_checks.items():
            out[name] = bool(cb.isChecked())
        return out

    def popUp(
        self,
        text: Optional[str] = None,
        flags: Optional[Dict[str, bool]] = None,
        group_id: Optional[int] = None,
        description: str = "",
    ) -> Tuple[Optional[str], Dict[str, bool], Optional[int], str]:
        """Show dialog and return edited values.

        Returns (text, flags, group_id, description). If cancelled, returns
        (None, input_flags, input_group_id, input_description).
        """
        text = "" if text is None else str(text)
        flags_in: Dict[str, bool] = dict(flags or {})

        # Merge with configured flags (defaults).
        configured_flags = self._config.get("flags") or {}
        if isinstance(configured_flags, dict):
            for k, v in configured_flags.items():
                flags_in.setdefault(str(k), bool(v))

        self.edit.setText(text)
        self._description.setText(str(description or ""))

        if group_id is None:
            self._group_enabled.setChecked(False)
            self._group_spin.setValue(0)
        else:
            self._group_enabled.setChecked(True)
            self._group_spin.setValue(int(group_id))

        self._set_flags(flags_in)

        # Focus label input by default.
        self.edit.setFocus(QtCore.Qt.OtherFocusReason)
        self.edit.selectAll()

        accepted = self.exec_() == QtWidgets.QDialog.Accepted
        if not accepted:
            return None, flags_in, group_id, str(description or "")

        out_text = str(self.edit.text() or "").strip()
        out_desc = str(self._description.text() or "").strip()
        out_group = (
            int(self._group_spin.value()) if self._group_enabled.isChecked() else None
        )
        out_flags = self._get_flags()

        return out_text, out_flags, out_group, out_desc
