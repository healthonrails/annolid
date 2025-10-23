from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Set

from qtpy import QtCore, QtGui, QtWidgets

from annolid.behavior.project_schema import ModifierDefinition, SubjectDefinition


class BehaviorControlsWidget(QtWidgets.QWidget):
    """Schema-driven controls for subjects, modifiers, and category badges."""

    subjectChanged = QtCore.Signal(str)
    modifierToggled = QtCore.Signal(str, bool)

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._modifier_buttons: Dict[str, QtWidgets.QToolButton] = {}

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Subject selection
        subject_layout = QtWidgets.QHBoxLayout()
        subject_label = QtWidgets.QLabel("Subject:")
        subject_label.setSizePolicy(
            QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        self._subject_combo = QtWidgets.QComboBox()
        self._subject_combo.currentTextChanged.connect(
            self._emit_subject_changed)
        subject_layout.addWidget(subject_label)
        subject_layout.addWidget(self._subject_combo, 1)
        layout.addLayout(subject_layout)

        # Category badge
        self._category_badge = QtWidgets.QLabel("No behavior selected")
        self._category_badge.setAlignment(QtCore.Qt.AlignCenter)
        self._category_badge.setFrameShape(QtWidgets.QFrame.Panel)
        self._category_badge.setFrameShadow(QtWidgets.QFrame.Sunken)
        self._category_badge.setStyleSheet(
            "QLabel { padding: 6px; border-radius: 6px; background-color: #eceff1; }"
        )
        layout.addWidget(self._category_badge)

        # Modifier buttons area
        modifiers_group = QtWidgets.QGroupBox("Modifiers")
        modifiers_layout = QtWidgets.QVBoxLayout(modifiers_group)
        self._modifier_container = QtWidgets.QWidget()
        self._modifier_layout = QtWidgets.QGridLayout(self._modifier_container)
        self._modifier_layout.setContentsMargins(0, 0, 0, 0)
        self._modifier_layout.setHorizontalSpacing(6)
        self._modifier_layout.setVerticalSpacing(6)
        modifiers_layout.addWidget(self._modifier_container)
        layout.addWidget(modifiers_group, 1)

        # Status / warning label
        self._warning_label = QtWidgets.QLabel("")
        self._warning_label.setStyleSheet("color: #d32f2f;")
        self._warning_label.setWordWrap(True)
        self._warning_label.hide()
        layout.addWidget(self._warning_label)

        layout.addStretch(1)

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
                label = subject.name or subject.id or "Subject"
                self._subject_combo.addItem(label, subject.id or subject.name)
            if selected:
                index = self._subject_combo.findText(
                    selected, QtCore.Qt.MatchFixedString)
                if index >= 0:
                    self._subject_combo.setCurrentIndex(index)
            if self._subject_combo.count() == 0:
                self._subject_combo.addItem("Subject 1", "subject_1")
        self._subject_combo.setEnabled(self._subject_combo.count() > 0)

    def current_subject(self) -> str:
        text = self._subject_combo.currentText().strip()
        return text or "Subject 1"

    def _emit_subject_changed(self, text: str) -> None:
        if text:
            self.subjectChanged.emit(text.strip())

    # ------------------------------------------------------------------ #
    # Modifiers
    # ------------------------------------------------------------------ #
    def set_modifiers(self, modifiers: Sequence[ModifierDefinition]) -> None:
        """Create toggle buttons for each modifier definition."""
        # Clear existing buttons
        for button in self._modifier_buttons.values():
            button.deleteLater()
        self._modifier_buttons.clear()

        # Rebuild layout
        while self._modifier_layout.count():
            item = self._modifier_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)

        if not modifiers:
            placeholder = QtWidgets.QLabel("No modifiers defined in schema.")
            placeholder.setAlignment(QtCore.Qt.AlignCenter)
            self._modifier_layout.addWidget(placeholder, 0, 0)
            return

        columns = 2
        for index, modifier in enumerate(modifiers):
            button = QtWidgets.QToolButton()
            button.setText(modifier.name or modifier.id)
            button.setCheckable(True)
            base_tooltip = modifier.description or modifier.name or modifier.id
            button.setToolTip(base_tooltip)
            button.setProperty("base_tooltip", base_tooltip)
            button.toggled.connect(
                lambda state, modifier_id=modifier.id: self.modifierToggled.emit(
                    modifier_id, state
                )
            )
            row = index // columns
            column = index % columns
            self._modifier_layout.addWidget(button, row, column)
            self._modifier_buttons[modifier.id] = button

    def set_modifier_states(
        self,
        selected: Iterable[str],
        *,
        allowed: Optional[Set[str]] = None,
    ) -> None:
        """Update modifier toggles, optionally restricting enabled buttons."""
        selected_set = set(selected or [])
        allowed = set(allowed or self._modifier_buttons.keys())
        for modifier_id, button in self._modifier_buttons.items():
            with QtCore.QSignalBlocker(button):
                button.setChecked(modifier_id in selected_set)
                button.setEnabled(modifier_id in allowed)
            base_tooltip = button.property("base_tooltip") or button.text()
            if modifier_id in allowed:
                button.setToolTip(base_tooltip)
            else:
                button.setToolTip(
                    f"{base_tooltip} (not applicable for this behavior)")

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
        foreground = "#ffffff" if self._should_use_light_text(
            background) else "#263238"
        self._category_badge.setText(text)
        stylesheet = (
            "QLabel {{ padding: 6px; border-radius: 6px; "
            "background-color: {bg}; color: {fg}; font-weight: 600; }}"
        ).format(bg=background, fg=foreground)
        self._category_badge.setStyleSheet(stylesheet)

    @staticmethod
    def _should_use_light_text(color: str) -> bool:
        color = color.lstrip("#")
        if len(color) != 6:
            return False
        r = int(color[0:2], 16)
        g = int(color[2:4], 16)
        b = int(color[4:6], 16)
        # Relative luminance formula
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
            self._subject_combo.addItem("Subject 1", "subject_1")
        for button in self._modifier_buttons.values():
            button.deleteLater()
        self._modifier_buttons.clear()
        while self._modifier_layout.count():
            item = self._modifier_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        placeholder = QtWidgets.QLabel("No modifiers available.")
        placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._modifier_layout.addWidget(placeholder, 0, 0)
        self._category_badge.setText("No behavior selected")
        self._category_badge.setStyleSheet(
            "QLabel { padding: 6px; border-radius: 6px; background-color: #eceff1; }"
        )
        self._warning_label.hide()
