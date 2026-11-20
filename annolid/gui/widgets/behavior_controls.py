from __future__ import annotations

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Set

from qtpy import QtCore, QtWidgets

from annolid.domain import BehaviorDefinition, CategoryDefinition
from annolid.domain import ModifierDefinition, SubjectDefinition


class BehaviorControlsWidget(QtWidgets.QWidget):
    """BORIS-style ethogram coding pad, focal subject switcher, modifier sets, and speed controls."""

    subjectChanged = QtCore.Signal(str)
    modifierToggled = QtCore.Signal(str, bool)
    behaviorTriggerRequested = QtCore.Signal(str)
    codingModeChanged = QtCore.Signal(bool)
    autoExcludeChanged = QtCore.Signal(bool)
    playbackSpeedChanged = QtCore.Signal(float)
    stepSecondsRequested = QtCore.Signal(float)

    _SPEED_PRESETS: tuple[float, ...] = (0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 4.0)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._modifier_buttons: Dict[str, QtWidgets.QToolButton] = {}
        self._modifier_defs: Dict[str, ModifierDefinition] = {}
        self._behavior_buttons: Dict[str, QtWidgets.QPushButton] = {}
        self._behavior_defs: List[BehaviorDefinition] = []
        self._category_colors: Dict[str, str] = {}
        self._active_behaviors: Set[str] = set()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        # 1. BORIS Coding Mode + Auto-Exclusion row
        mode_row = QtWidgets.QHBoxLayout()
        mode_row.setSpacing(6)
        self._coding_mode_btn = QtWidgets.QToolButton()
        self._coding_mode_btn.setText("⌨ Live Coding: OFF")
        self._coding_mode_btn.setCheckable(True)
        self._coding_mode_btn.setToolTip(
            "When ON, pressing ethogram or focal-subject shortcut keys immediately logs State (toggle) or Point events."
        )
        self._coding_mode_btn.toggled.connect(self._on_coding_mode_toggled)
        self._apply_coding_mode_style(False)
        mode_row.addWidget(self._coding_mode_btn, 1)

        self._auto_exclude_chk = QtWidgets.QCheckBox("Auto-stop Excluded")
        self._auto_exclude_chk.setChecked(True)
        self._auto_exclude_chk.setToolTip(
            "BORIS Exclusion Matrix mode: starting a state behavior automatically closes active mutually-exclusive states for the focal subject."
        )
        self._auto_exclude_chk.toggled.connect(self.autoExcludeChanged.emit)
        mode_row.addWidget(self._auto_exclude_chk)
        layout.addLayout(mode_row)

        # 2. Variable-Speed Playback & Precision Jump Bar
        transport_row = QtWidgets.QHBoxLayout()
        transport_row.setSpacing(3)
        speed_lbl = QtWidgets.QLabel("Speed:")
        self._speed_combo = QtWidgets.QComboBox()
        for mult in self._SPEED_PRESETS:
            self._speed_combo.addItem(f"{mult:g}x", float(mult))
        self._speed_combo.setCurrentIndex(self._SPEED_PRESETS.index(1.0))
        self._speed_combo.setToolTip("Adjust video playback speed for precision coding")
        self._speed_combo.currentIndexChanged.connect(self._emit_speed_changed)
        transport_row.addWidget(speed_lbl)
        transport_row.addWidget(self._speed_combo)

        for step_s, label in (
            (-5.0, "-5s"),
            (-1.0, "-1s"),
            (1.0, "+1s"),
            (5.0, "+5s"),
        ):
            step_btn = QtWidgets.QToolButton()
            step_btn.setText(label)
            step_btn.setToolTip(f"Jump {label} in recording")
            step_btn.clicked.connect(
                lambda _checked=False, delta=step_s: self.stepSecondsRequested.emit(
                    delta
                )
            )
            transport_row.addWidget(step_btn)
        transport_row.addStretch(1)
        layout.addLayout(transport_row)

        # 3. Focal Subject selection
        subject_layout = QtWidgets.QHBoxLayout()
        subject_label = QtWidgets.QLabel("Focal Subject:")
        subject_label.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred
        )
        self._subject_combo = QtWidgets.QComboBox()
        self._subject_combo.currentIndexChanged.connect(self._on_subject_index_changed)
        subject_layout.addWidget(subject_label)
        subject_layout.addWidget(self._subject_combo, 1)
        layout.addLayout(subject_layout)

        # Category badge
        self._category_badge = QtWidgets.QLabel("No behavior selected")
        self._category_badge.setAlignment(QtCore.Qt.AlignCenter)
        self._category_badge.setFrameShape(QtWidgets.QFrame.Panel)
        self._category_badge.setFrameShadow(QtWidgets.QFrame.Sunken)
        self._category_badge.setStyleSheet(
            "QLabel { padding: 5px; border-radius: 6px; background-color: #eceff1; }"
        )
        layout.addWidget(self._category_badge)

        # 4. Interactive Ethogram Coding Pad
        ethogram_group = QtWidgets.QGroupBox("Ethogram Pad (Click or Press Key)")
        ethogram_lay = QtWidgets.QVBoxLayout(ethogram_group)
        ethogram_lay.setContentsMargins(6, 6, 6, 6)
        ethogram_lay.setSpacing(4)

        self._ethogram_filter = QtWidgets.QLineEdit()
        self._ethogram_filter.setPlaceholderText("Filter ethogram behaviors…")
        self._ethogram_filter.setClearButtonEnabled(True)
        self._ethogram_filter.textChanged.connect(self._rebuild_ethogram_buttons)
        ethogram_lay.addWidget(self._ethogram_filter)

        self._ethogram_scroll = QtWidgets.QScrollArea()
        self._ethogram_scroll.setWidgetResizable(True)
        self._ethogram_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._ethogram_container = QtWidgets.QWidget()
        self._ethogram_grid = QtWidgets.QGridLayout(self._ethogram_container)
        self._ethogram_grid.setContentsMargins(0, 0, 0, 0)
        self._ethogram_grid.setHorizontalSpacing(5)
        self._ethogram_grid.setVerticalSpacing(5)
        self._ethogram_scroll.setWidget(self._ethogram_container)
        ethogram_lay.addWidget(self._ethogram_scroll)
        layout.addWidget(ethogram_group, 2)

        # 5. Structured Modifier Sets area
        modifiers_group = QtWidgets.QGroupBox("Modifiers")
        modifiers_layout = QtWidgets.QVBoxLayout(modifiers_group)
        modifiers_layout.setContentsMargins(6, 6, 6, 6)
        modifiers_layout.setSpacing(4)

        mod_top = QtWidgets.QHBoxLayout()
        mod_top.addStretch(1)
        self._clear_mods_btn = QtWidgets.QToolButton()
        self._clear_mods_btn.setText("Clear")
        self._clear_mods_btn.setToolTip("Uncheck all selected modifiers")
        self._clear_mods_btn.clicked.connect(self.clear_checked_modifiers)
        mod_top.addWidget(self._clear_mods_btn)
        modifiers_layout.addLayout(mod_top)

        self._modifier_container = QtWidgets.QWidget()
        self._modifier_layout = QtWidgets.QVBoxLayout(self._modifier_container)
        self._modifier_layout.setContentsMargins(0, 0, 0, 0)
        self._modifier_layout.setSpacing(4)
        modifiers_layout.addWidget(self._modifier_container)
        layout.addWidget(modifiers_group, 1)

        # 6. Quick Event Comment / Note
        comment_row = QtWidgets.QHBoxLayout()
        comment_row.setSpacing(4)
        self._comment_edit = QtWidgets.QLineEdit()
        self._comment_edit.setPlaceholderText("Optional event comment / note…")
        self._comment_edit.setClearButtonEnabled(True)
        self._auto_clear_comment_chk = QtWidgets.QCheckBox("Auto-clear")
        self._auto_clear_comment_chk.setChecked(True)
        self._auto_clear_comment_chk.setToolTip(
            "Clear the comment box automatically after recording an event"
        )
        comment_row.addWidget(self._comment_edit, 1)
        comment_row.addWidget(self._auto_clear_comment_chk)
        layout.addLayout(comment_row)

        # Status / warning label
        self._warning_label = QtWidgets.QLabel("")
        self._warning_label.setStyleSheet("color: #d32f2f; font-weight: 600;")
        self._warning_label.setWordWrap(True)
        self._warning_label.hide()
        layout.addWidget(self._warning_label)

        self._rebuild_ethogram_buttons()

    # ------------------------------------------------------------------ #
    # Coding Mode & Transport
    # ------------------------------------------------------------------ #
    def is_coding_mode_enabled(self) -> bool:
        return bool(self._coding_mode_btn.isChecked())

    def set_coding_mode_enabled(self, enabled: bool) -> None:
        self._coding_mode_btn.setChecked(bool(enabled))

    def auto_stop_excluded(self) -> bool:
        return bool(self._auto_exclude_chk.isChecked())

    def set_auto_stop_excluded(self, enabled: bool) -> None:
        self._auto_exclude_chk.setChecked(bool(enabled))

    def playback_speed(self) -> float:
        data = self._speed_combo.currentData()
        return float(data) if data is not None else 1.0

    def set_playback_speed(self, speed: float) -> None:
        best_idx = 0
        best_diff = float("inf")
        for idx in range(self._speed_combo.count()):
            val = float(self._speed_combo.itemData(idx) or 1.0)
            if abs(val - float(speed)) < best_diff:
                best_diff = abs(val - float(speed))
                best_idx = idx
        with QtCore.QSignalBlocker(self._speed_combo):
            self._speed_combo.setCurrentIndex(best_idx)

    def _emit_speed_changed(self, _index: int) -> None:
        self.playbackSpeedChanged.emit(self.playback_speed())

    def _on_coding_mode_toggled(self, enabled: bool) -> None:
        self._coding_mode_btn.setText(
            "⌨ Live Coding: ON" if enabled else "⌨ Live Coding: OFF"
        )
        self._apply_coding_mode_style(enabled)
        self.codingModeChanged.emit(bool(enabled))

    def _apply_coding_mode_style(self, enabled: bool) -> None:
        if enabled:
            self._coding_mode_btn.setStyleSheet(
                "QToolButton { background-color: #16a34a; color: #ffffff; font-weight: 700; border-radius: 5px; padding: 5px 8px; }"
            )
        else:
            self._coding_mode_btn.setStyleSheet(
                "QToolButton { background-color: #e2e8f0; color: #1e293b; font-weight: 600; border-radius: 5px; padding: 5px 8px; }"
            )

    # ------------------------------------------------------------------ #
    # Event Comment
    # ------------------------------------------------------------------ #
    def current_comment(self) -> str:
        return self._comment_edit.text().strip()

    def set_comment(self, text: str) -> None:
        self._comment_edit.setText(str(text or ""))

    def take_event_comment(self) -> str:
        comment = self._comment_edit.text().strip()
        if comment and self._auto_clear_comment_chk.isChecked():
            self._comment_edit.clear()
        return comment

    # ------------------------------------------------------------------ #
    # Subjects
    # ------------------------------------------------------------------ #
    def set_subjects(
        self,
        subjects: Sequence[SubjectDefinition],
        *,
        selected: Optional[str] = None,
    ) -> None:
        with QtCore.QSignalBlocker(self._subject_combo):
            self._subject_combo.clear()
            for subject in subjects:
                base_name = subject.name or subject.id or "Subject"
                display = (
                    f"[{subject.key_binding}] {base_name}"
                    if subject.key_binding
                    else base_name
                )
                self._subject_combo.addItem(display, base_name)
            if self._subject_combo.count() == 0:
                self._subject_combo.addItem("Subject 1", "Subject 1")
            self._subject_combo.addItem("No focal subject", "No focal subject")
            if selected:
                self._select_subject_internal(selected)
        self._subject_combo.setEnabled(self._subject_combo.count() > 0)

    def select_subject(self, subject_name: str) -> bool:
        matched = self._select_subject_internal(subject_name)
        if matched:
            self.subjectChanged.emit(self.current_subject())
        return matched

    def _select_subject_internal(self, target: str) -> bool:
        norm = target.strip().lower()
        for idx in range(self._subject_combo.count()):
            data_val = str(self._subject_combo.itemData(idx) or "").strip()
            txt_val = self._subject_combo.itemText(idx).strip()
            if data_val.lower() == norm or txt_val.lower() == norm:
                self._subject_combo.setCurrentIndex(idx)
                return True
        return False

    def current_subject(self) -> str:
        data = self._subject_combo.currentData()
        if data is not None and str(data).strip():
            return str(data).strip()
        text = self._subject_combo.currentText().strip()
        return text or "Subject 1"

    def _on_subject_index_changed(self, _index: int) -> None:
        subj = self.current_subject()
        if subj:
            self.subjectChanged.emit(subj)

    # ------------------------------------------------------------------ #
    # Ethogram Pad
    # ------------------------------------------------------------------ #
    def set_behaviors(
        self,
        behaviors: Sequence[BehaviorDefinition],
        *,
        categories: Optional[Sequence[CategoryDefinition]] = None,
        active_behaviors: Optional[Set[str]] = None,
    ) -> None:
        self._behavior_defs = list(behaviors)
        self._category_colors = {
            cat.id: (cat.color or "#455a64") for cat in (categories or ()) if cat.id
        }
        if active_behaviors is not None:
            self._active_behaviors = set(active_behaviors)
        self._rebuild_ethogram_buttons()

    def set_active_behaviors(self, active_codes: Iterable[str]) -> None:
        new_active = {str(c).strip() for c in active_codes if str(c).strip()}
        if new_active == self._active_behaviors:
            return
        self._active_behaviors = new_active
        for code, btn in self._behavior_buttons.items():
            bdef = next((b for b in self._behavior_defs if b.code == code), None)
            if bdef is not None:
                self._style_behavior_button(btn, bdef, code in self._active_behaviors)

    def _rebuild_ethogram_buttons(self) -> None:
        for btn in self._behavior_buttons.values():
            btn.deleteLater()
        self._behavior_buttons.clear()
        while self._ethogram_grid.count():
            item = self._ethogram_grid.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        query = self._ethogram_filter.text().strip().lower()
        filtered = [
            b
            for b in self._behavior_defs
            if not query
            or query in (b.code or "").lower()
            or query in (b.name or "").lower()
            or query in (b.key_binding or "").lower()
        ]
        if not filtered:
            empty_lbl = QtWidgets.QLabel(
                "No behaviors defined. Use Project Schema or Flags to add behaviors."
                if not self._behavior_defs
                else "No behaviors match filter."
            )
            empty_lbl.setWordWrap(True)
            empty_lbl.setAlignment(QtCore.Qt.AlignCenter)
            self._ethogram_grid.addWidget(empty_lbl, 0, 0)
            return

        cols = 2
        for idx, bdef in enumerate(filtered):
            btn = QtWidgets.QPushButton()
            btn.setCursor(QtCore.Qt.PointingHandCursor)
            is_act = bdef.code in self._active_behaviors
            self._style_behavior_button(btn, bdef, is_act)
            tip_parts = [
                f"{bdef.name or bdef.code} ({'State' if bdef.is_state else 'Point'})"
            ]
            if bdef.key_binding:
                tip_parts.append(f"Shortcut: {bdef.key_binding}")
            if bdef.exclusive_with:
                tip_parts.append(f"Excludes: {', '.join(bdef.exclusive_with)}")
            if bdef.description:
                tip_parts.append(bdef.description)
            btn.setToolTip("\n".join(tip_parts))
            btn.clicked.connect(
                lambda _checked=False,
                code=bdef.code: self.behaviorTriggerRequested.emit(code)
            )
            self._ethogram_grid.addWidget(btn, idx // cols, idx % cols)
            self._behavior_buttons[bdef.code] = btn

    def _style_behavior_button(
        self,
        btn: QtWidgets.QPushButton,
        bdef: BehaviorDefinition,
        is_active: bool,
    ) -> None:
        key_prefix = f"[{bdef.key_binding}] " if bdef.key_binding else ""
        kind_tag = "◆" if not bdef.is_state else ("●" if is_active else "○")
        title = bdef.name or bdef.code
        btn.setText(f"{kind_tag} {key_prefix}{title}")

        accent = (
            bdef.color
            or (
                self._category_colors.get(bdef.category_id or "")
                if bdef.category_id
                else None
            )
            or "#3b82f6"
        )
        if is_active and bdef.is_state:
            btn.setStyleSheet(
                "QPushButton { background-color: #16a34a; color: #ffffff; font-weight: 700; "
                "border: 2px solid #14532d; border-radius: 5px; padding: 5px 6px; text-align: left; }"
            )
        else:
            btn.setStyleSheet(
                "QPushButton { background-color: #f8fafc; color: #0f172a; font-weight: 600; "
                f"border-left: 4px solid {accent}; border-top: 1px solid #cbd5e1; "
                "border-right: 1px solid #cbd5e1; border-bottom: 1px solid #cbd5e1; "
                "border-radius: 4px; padding: 5px 6px; text-align: left; }"
                "QPushButton:hover { background-color: #e2e8f0; }"
            )

    # ------------------------------------------------------------------ #
    # Modifiers (Single-select & Multi-select sets)
    # ------------------------------------------------------------------ #
    def set_modifiers(self, modifiers: Sequence[ModifierDefinition]) -> None:
        """Create grouped toggle buttons for each modifier definition."""
        for button in self._modifier_buttons.values():
            button.deleteLater()
        self._modifier_buttons.clear()
        self._modifier_defs = {m.id: m for m in modifiers if m.id}

        while self._modifier_layout.count():
            item = self._modifier_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if not modifiers:
            placeholder = QtWidgets.QLabel("No modifiers defined in schema.")
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            self._modifier_layout.addWidget(placeholder)
            return

        groups: Dict[str, List[ModifierDefinition]] = defaultdict(list)
        for mod in modifiers:
            grp_label = mod.group or "General"
            groups[grp_label].append(mod)

        show_group_headers = len(groups) > 1 or "General" not in groups
        for grp_name, grp_mods in groups.items():
            box = QtWidgets.QWidget()
            box_lay = QtWidgets.QVBoxLayout(box)
            box_lay.setContentsMargins(0, 0, 0, 0)
            box_lay.setSpacing(3)
            mode = grp_mods[0].selection_mode if grp_mods else "multi"
            if show_group_headers:
                mode_badge = "single" if mode == "single" else "multi"
                hdr = QtWidgets.QLabel(f"{grp_name} ({mode_badge}):")
                hdr.setStyleSheet("font-size: 11px; font-weight: 600; color: #475569;")
                box_lay.addWidget(hdr)

            grid_w = QtWidgets.QWidget()
            grid = QtWidgets.QGridLayout(grid_w)
            grid.setContentsMargins(0, 0, 0, 0)
            grid.setHorizontalSpacing(5)
            grid.setVerticalSpacing(4)

            for index, modifier in enumerate(grp_mods):
                button = QtWidgets.QToolButton()
                key_lbl = f"[{modifier.key_binding}] " if modifier.key_binding else ""
                button.setText(f"{key_lbl}{modifier.name or modifier.id}")
                button.setCheckable(True)
                button.setSizePolicy(
                    QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed
                )
                base_tooltip = modifier.description or modifier.name or modifier.id
                button.setToolTip(base_tooltip)
                button.setProperty("base_tooltip", base_tooltip)
                button.toggled.connect(
                    lambda state, mid=modifier.id: self._handle_modifier_button_toggled(
                        mid, state
                    )
                )
                grid.addWidget(button, index // 2, index % 2)
                self._modifier_buttons[modifier.id] = button

            box_lay.addWidget(grid_w)
            self._modifier_layout.addWidget(box)

    def _handle_modifier_button_toggled(self, modifier_id: str, state: bool) -> None:
        mdef = self._modifier_defs.get(modifier_id)
        if state and mdef is not None and mdef.selection_mode == "single":
            grp = mdef.group or "General"
            for other_id, other_def in self._modifier_defs.items():
                if other_id == modifier_id:
                    continue
                if (other_def.group or "General") == grp:
                    other_btn = self._modifier_buttons.get(other_id)
                    if other_btn is not None and other_btn.isChecked():
                        with QtCore.QSignalBlocker(other_btn):
                            other_btn.setChecked(False)
                        self.modifierToggled.emit(other_id, False)
        self.modifierToggled.emit(modifier_id, state)

    def clear_checked_modifiers(self) -> None:
        for mid, btn in self._modifier_buttons.items():
            if btn.isChecked():
                with QtCore.QSignalBlocker(btn):
                    btn.setChecked(False)
                self.modifierToggled.emit(mid, False)

    def set_modifier_states(
        self,
        selected: Iterable[str],
        *,
        allowed: Optional[Set[str]] = None,
    ) -> None:
        """Update modifier toggles, optionally restricting enabled buttons."""
        selected_set = set(selected or [])
        allowed_set = (
            set(self._modifier_buttons.keys()) if allowed is None else set(allowed)
        )
        for modifier_id, button in self._modifier_buttons.items():
            with QtCore.QSignalBlocker(button):
                button.setChecked(modifier_id in selected_set)
                button.setEnabled(modifier_id in allowed_set)
            base_tooltip = button.property("base_tooltip") or button.text()
            if modifier_id in allowed_set:
                button.setToolTip(base_tooltip)
            else:
                button.setToolTip(f"{base_tooltip} (not applicable for this behavior)")

    def checked_modifiers(self) -> Set[str]:
        return {
            modifier_id
            for modifier_id, button in self._modifier_buttons.items()
            if button.isEnabled() and button.isChecked()
        }

    def set_modifiers_enabled(self, modifier_ids: Set[str], enabled: bool) -> None:
        for modifier_id in modifier_ids:
            button = self._modifier_buttons.get(modifier_id)
            if button is not None:
                button.setEnabled(enabled)

    # ------------------------------------------------------------------ #
    # Category badge & warnings
    # ------------------------------------------------------------------ #
    def set_category_badge(
        self,
        label: Optional[str],
        color: Optional[str],
    ) -> None:
        text = label or "No behavior selected"
        background = color or "#eceff1"
        foreground = "#ffffff" if self._should_use_light_text(background) else "#263238"
        self._category_badge.setText(text)
        stylesheet = (
            "QLabel {{ padding: 5px; border-radius: 6px; "
            "background-color: {bg}; color: {fg}; font-weight: 600; }}"
        ).format(bg=background, fg=foreground)
        self._category_badge.setStyleSheet(stylesheet)

    @staticmethod
    def _should_use_light_text(color: str) -> bool:
        color = color.lstrip("#")
        if len(color) != 6:
            return False
        try:
            r = int(color[0:2], 16)
            g = int(color[2:4], 16)
            b = int(color[4:6], 16)
        except ValueError:
            return False
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        return luminance < 140

    def show_warning(self, message: Optional[str]) -> None:
        if message:
            self._warning_label.setText(message)
            self._warning_label.show()
        else:
            self._warning_label.hide()
            self._warning_label.clear()

    # ------------------------------------------------------------------ #
    def clear(self) -> None:
        with QtCore.QSignalBlocker(self._subject_combo):
            self._subject_combo.clear()
            self._subject_combo.addItem("Subject 1", "Subject 1")
            self._subject_combo.addItem("No focal subject", "No focal subject")
        self._behavior_defs.clear()
        self._active_behaviors.clear()
        self._rebuild_ethogram_buttons()
        for button in self._modifier_buttons.values():
            button.deleteLater()
        self._modifier_buttons.clear()
        self._modifier_defs.clear()
        while self._modifier_layout.count():
            item = self._modifier_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        placeholder = QtWidgets.QLabel("No modifiers available.")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._modifier_layout.addWidget(placeholder)
        self._comment_edit.clear()
        self._category_badge.setText("No behavior selected")
        self._category_badge.setStyleSheet(
            "QLabel { padding: 5px; border-radius: 6px; background-color: #eceff1; }"
        )
        self._warning_label.hide()
