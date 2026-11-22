"""LabelMe-compatible label editor dialog with history list and per-label flags."""

from __future__ import annotations

import re
from typing import Dict, Optional, Tuple

from qtpy import QtCore, QtWidgets

from annolid.annotation.shape_io import normalize_group_id


class AnnolidLabelDialog(QtWidgets.QDialog):
    """Lightweight LabelMe-compatible label editor dialog.

    Supports:
    - label text with auto-completion & clickable/filterable label history list
    - optional regex-matched per-label flags (`label_flags`) and boolean shape flags
    - optional group_id (validated int, rejecting booleans)
    - description string
    - zone type toggle

    API compatibility used by `annolid/gui/app.py`:
    - `.edit` (QLineEdit)
    - `.popUp(...) -> (text, flags, group_id, description)`
    - `.addLabelHistory(label)`
    """

    def __init__(self, *, parent: QtWidgets.QWidget, config: Optional[dict] = None):
        super().__init__(parent)
        self._config = config or {}
        self._history: list[str] = []
        self._show_flags_active = True
        self._base_flags_in: Dict[str, bool] = {}

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

        self._zone_checkbox = QtWidgets.QCheckBox("Zone type", self)
        form.addRow("Type", self._zone_checkbox)

        # LabelMe-style clickable/filterable label list.
        self.labelList = QtWidgets.QListWidget(self)
        self.labelList.setMaximumHeight(140)
        self.labelList.itemClicked.connect(self._on_label_list_clicked)
        self.labelList.itemDoubleClicked.connect(self._on_label_list_double_clicked)
        layout.addWidget(self.labelList)

        # Flags section: created dynamically in popUp and updated when label text changes.
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

        # Completion model from history + configured labels.
        self._completer = QtWidgets.QCompleter(self)
        self._completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completion_mode = str(
            self._config.get("label_completion") or "contains"
        ).lower()
        if completion_mode == "startswith":
            self._completer.setFilterMode(QtCore.Qt.MatchStartsWith)
        else:
            self._completer.setFilterMode(QtCore.Qt.MatchContains)
        self.edit.setCompleter(self._completer)
        self.edit.textChanged.connect(self._on_label_text_changed)

        self._refresh_completion()

    def addLabelHistory(self, label: str) -> None:
        label = str(label or "").strip()
        if not label:
            return
        if label in self._history:
            self._history.remove(label)
        self._history.insert(0, label)
        self._history = self._history[:200]
        self._refresh_completion()

    def _collect_known_labels(self) -> list[str]:
        configured = self._config.get("labels") or []
        items: list[str] = []
        for val in list(self._history) + list(configured):
            s = str(val or "").strip()
            if s and s not in items:
                items.append(s)
        if bool(self._config.get("sort_labels", False)):
            items.sort(key=lambda s: s.lower())
        return items

    def _refresh_completion(self) -> None:
        items = self._collect_known_labels()
        model = QtCore.QStringListModel(items, self._completer)
        self._completer.setModel(model)
        self._rebuild_label_list(items)

    def _rebuild_label_list(self, items: Optional[list[str]] = None) -> None:
        labels = self._collect_known_labels() if items is None else items
        self.labelList.blockSignals(True)
        try:
            self.labelList.clear()
            for label in labels:
                self.labelList.addItem(label)
            self.labelList.setVisible(bool(labels))
        finally:
            self.labelList.blockSignals(False)
        self._filter_label_list(self.edit.text())

    def _filter_label_list(self, query_text: str) -> None:
        query = str(query_text or "").strip().lower()
        completion_mode = str(
            self._config.get("label_completion") or "contains"
        ).lower()
        any_visible = False
        for idx in range(self.labelList.count()):
            item = self.labelList.item(idx)
            if item is None:
                continue
            item_text = str(item.text() or "").lower()
            if not query:
                matched = True
            elif completion_mode == "startswith":
                matched = item_text.startswith(query)
            else:
                matched = query in item_text
            item.setHidden(not matched)
            if matched:
                any_visible = True
        self.labelList.setVisible(self.labelList.count() > 0 and any_visible)

    def _on_label_list_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is not None:
            self.edit.setText(item.text())

    def _on_label_list_double_clicked(self, item: QtWidgets.QListWidgetItem) -> None:
        if item is not None:
            self.edit.setText(item.text())
            self.accept()

    def _label_flags_for_text(self, label_text: str) -> Dict[str, bool]:
        """Collect regex-matched per-label flags from config ``label_flags``."""
        matched: Dict[str, bool] = {}
        label_flags_cfg = self._config.get("label_flags")
        if not isinstance(label_flags_cfg, dict):
            return matched
        for pattern, keys in label_flags_cfg.items():
            try:
                if re.match(str(pattern), label_text):
                    if isinstance(keys, (list, tuple, set)):
                        for key in keys:
                            matched.setdefault(str(key), False)
                    elif isinstance(keys, dict):
                        for key, val in keys.items():
                            matched.setdefault(str(key), bool(val))
            except re.error:
                continue
        return matched

    def _on_label_text_changed(self, text: str) -> None:
        self._filter_label_list(text)
        if not self._show_flags_active:
            return
        current_values = self._get_flags()
        merged: Dict[str, bool] = dict(self._base_flags_in)
        for key, default_val in self._label_flags_for_text(text).items():
            merged.setdefault(key, default_val)
        for key in list(merged.keys()):
            if key in current_values:
                merged[key] = current_values[key]
        self._set_flags(merged)

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
        names = sorted(name for name in flags.keys() if str(name) != "zone")
        if not names or not self._show_flags_active:
            self._flags_box.setVisible(False)
            return
        self._flags_box.setVisible(True)
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
        *,
        show_flags: bool = True,
    ) -> Tuple[Optional[str], Dict[str, bool], Optional[int], str]:
        """Show dialog and return edited values.

        Returns ``(text, flags, group_id, description)``. If cancelled, returns
        ``(None, input_flags, input_group_id, input_description)``.
        """
        text = "" if text is None else str(text)
        flags_in: Dict[str, bool] = {}
        for key, value in dict(flags or {}).items():
            if isinstance(value, bool):
                flags_in[str(key)] = value

        # Merge regex-matched per-label defaults from `label_flags`.
        for key, default_val in self._label_flags_for_text(text).items():
            flags_in.setdefault(key, default_val)

        self._show_flags_active = bool(show_flags)
        self._base_flags_in = dict(flags_in)

        self.edit.blockSignals(True)
        try:
            self.edit.setText(text)
        finally:
            self.edit.blockSignals(False)
        self._filter_label_list(text)
        self._description.setText(str(description or ""))
        inferred_zone = ("zone" in text.strip().lower()) or bool(flags_in.get("zone"))
        self._zone_checkbox.setChecked(inferred_zone)

        normalized_group = normalize_group_id(group_id)
        if normalized_group is None:
            self._group_enabled.setChecked(False)
            self._group_spin.setValue(0)
        else:
            self._group_enabled.setChecked(True)
            self._group_spin.setValue(max(0, min(10_000_000, normalized_group)))

        self._set_flags(flags_in)

        self.edit.setFocus(QtCore.Qt.OtherFocusReason)
        self.edit.selectAll()

        accepted = self.exec_() == QtWidgets.QDialog.Accepted
        if not accepted:
            return None, flags_in, normalized_group, str(description or "")

        out_text = str(self.edit.text() or "").strip()
        out_desc = str(self._description.text() or "").strip()
        out_group = (
            int(self._group_spin.value()) if self._group_enabled.isChecked() else None
        )
        out_flags = self._get_flags()
        label_implies_zone = "zone" in out_text.lower()
        is_zone_shape = self._zone_checkbox.isChecked() or label_implies_zone
        if is_zone_shape:
            out_flags["zone"] = True
            out_flags["semantic_type"] = "zone"
            out_flags["shape_category"] = "zone"
            out_flags.setdefault("zone_kind", "custom")
        else:
            for key in (
                "zone",
                "semantic_type",
                "shape_category",
                "zone_kind",
                "zone_label",
            ):
                out_flags.pop(key, None)

        return out_text, out_flags, out_group, out_desc
