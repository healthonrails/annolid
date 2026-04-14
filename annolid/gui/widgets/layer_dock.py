from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtGui, QtWidgets


class ViewerLayerDockWidget(QtWidgets.QDockWidget):
    layerVisibilityChanged = QtCore.Signal(str, bool)
    layerOpacityChanged = QtCore.Signal(str, float)
    layerSelected = QtCore.Signal(str)
    layerTranslateRequested = QtCore.Signal(str, float, float)
    layerResetTransformRequested = QtCore.Signal(str)
    layerMoveRequested = QtCore.Signal(str, int)
    layerRenameRequested = QtCore.Signal(str, str)
    layerRemoveRequested = QtCore.Signal(str)
    layerMoveToTopRequested = QtCore.Signal(str)
    layerMoveToBottomRequested = QtCore.Signal(str)
    layerOpenSourceRequested = QtCore.Signal(str)
    layerOpenSourceFolderRequested = QtCore.Signal(str)
    layerApplySettingsRequested = QtCore.Signal(str, dict)
    layerSaveSettingsRequested = QtCore.Signal(str, dict)
    layerQuickTransformRequested = QtCore.Signal(str, dict)
    layerInteractiveResizeToggled = QtCore.Signal(bool)

    def __init__(self, parent=None):
        super().__init__("Layers", parent)
        self.setObjectName("viewerLayerDock")
        self._layer_map: dict[str, dict] = {}
        self._shortcuts: list[QtWidgets.QShortcut] = []
        self._settings_syncing = False
        self._copied_alignment: dict | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.layer_list = QtWidgets.QListWidget(container)
        self.layer_list.setSizeAdjustPolicy(
            QtWidgets.QAbstractScrollArea.AdjustToContents
        )
        self.layer_list.itemChanged.connect(self._on_item_changed)
        self.layer_list.currentItemChanged.connect(self._on_current_item_changed)
        self.layer_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.layer_list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.layer_list, 2)

        controls_scroll = QtWidgets.QScrollArea(container)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        controls_scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        controls_container = QtWidgets.QWidget(controls_scroll)
        controls_layout = QtWidgets.QVBoxLayout(controls_container)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        controls_scroll.setWidget(controls_container)
        layout.addWidget(controls_scroll, 3)

        self.details_label = QtWidgets.QLabel("", controls_container)
        self.details_label.setWordWrap(True)
        controls_layout.addWidget(self.details_label)

        self.opacity_slider = QtWidgets.QSlider(
            QtCore.Qt.Horizontal, controls_container
        )
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        controls_layout.addWidget(self.opacity_slider)

        def _build_collapsible_section(
            parent: QtWidgets.QWidget,
            title: str,
            content: QtWidgets.QWidget,
            *,
            expanded: bool = True,
        ) -> tuple[QtWidgets.QWidget, QtWidgets.QToolButton]:
            wrapper = QtWidgets.QWidget(parent)
            wrapper_layout = QtWidgets.QVBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(0, 0, 0, 0)
            wrapper_layout.setSpacing(2)
            toggle = QtWidgets.QToolButton(wrapper)
            toggle.setText(str(title))
            toggle.setCheckable(True)
            toggle.setChecked(bool(expanded))
            toggle.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
            toggle.setArrowType(
                QtCore.Qt.DownArrow if bool(expanded) else QtCore.Qt.RightArrow
            )
            toggle.setStyleSheet("QToolButton { font-weight: 600; padding: 4px 2px; }")
            content.setVisible(bool(expanded))
            toggle.toggled.connect(content.setVisible)
            toggle.toggled.connect(
                lambda checked, button=toggle: button.setArrowType(
                    QtCore.Qt.DownArrow if checked else QtCore.Qt.RightArrow
                )
            )
            wrapper_layout.addWidget(toggle)
            wrapper_layout.addWidget(content)
            return wrapper, toggle

        translate_box = QtWidgets.QGroupBox("", controls_container)
        translate_layout = QtWidgets.QVBoxLayout(translate_box)
        translate_layout.setContentsMargins(8, 8, 8, 8)
        translate_layout.setSpacing(6)

        step_row = QtWidgets.QHBoxLayout()
        step_label = QtWidgets.QLabel("Step", translate_box)
        self.translate_step_spin = QtWidgets.QDoubleSpinBox(translate_box)
        self.translate_step_spin.setRange(0.001, 1000000.0)
        self.translate_step_spin.setDecimals(3)
        self.translate_step_spin.setSingleStep(1.0)
        self.translate_step_spin.setValue(1.0)
        self.translate_step_spin.setSuffix(" px")
        step_row.addWidget(step_label)
        step_row.addWidget(self.translate_step_spin, 1)
        translate_layout.addLayout(step_row)

        nudge_grid = QtWidgets.QGridLayout()
        self.nudge_left_button = QtWidgets.QToolButton(translate_box)
        self.nudge_left_button.setText("Left")
        self.nudge_right_button = QtWidgets.QToolButton(translate_box)
        self.nudge_right_button.setText("Right")
        self.nudge_up_button = QtWidgets.QToolButton(translate_box)
        self.nudge_up_button.setText("Up")
        self.nudge_down_button = QtWidgets.QToolButton(translate_box)
        self.nudge_down_button.setText("Down")
        self.nudge_left_button.clicked.connect(lambda: self._request_translate(-1, 0))
        self.nudge_right_button.clicked.connect(lambda: self._request_translate(1, 0))
        self.nudge_up_button.clicked.connect(lambda: self._request_translate(0, -1))
        self.nudge_down_button.clicked.connect(lambda: self._request_translate(0, 1))
        nudge_grid.addWidget(self.nudge_up_button, 0, 1)
        nudge_grid.addWidget(self.nudge_left_button, 1, 0)
        nudge_grid.addWidget(self.nudge_right_button, 1, 2)
        nudge_grid.addWidget(self.nudge_down_button, 2, 1)
        translate_layout.addLayout(nudge_grid)

        self.reset_translate_button = QtWidgets.QPushButton(
            "Reset Alignment", translate_box
        )
        self.reset_translate_button.clicked.connect(self._request_translate_reset)
        translate_layout.addWidget(self.reset_translate_button)

        quick_scale_row = QtWidgets.QHBoxLayout()
        self.scale_step_spin = QtWidgets.QDoubleSpinBox(translate_box)
        self.scale_step_spin.setRange(0.1, 100.0)
        self.scale_step_spin.setDecimals(1)
        self.scale_step_spin.setSingleStep(0.5)
        self.scale_step_spin.setValue(5.0)
        self.scale_step_spin.setSuffix(" %")
        self.shrink_button = QtWidgets.QPushButton("Shrink", translate_box)
        self.expand_button = QtWidgets.QPushButton("Expand", translate_box)
        self.shrink_button.clicked.connect(self._request_shrink)
        self.expand_button.clicked.connect(self._request_expand)
        quick_scale_row.addWidget(QtWidgets.QLabel("Scale Step", translate_box))
        quick_scale_row.addWidget(self.scale_step_spin, 1)
        quick_scale_row.addWidget(self.shrink_button)
        quick_scale_row.addWidget(self.expand_button)
        translate_layout.addLayout(quick_scale_row)

        quick_align_grid = QtWidgets.QGridLayout()
        self.align_left_button = QtWidgets.QToolButton(translate_box)
        self.align_hcenter_button = QtWidgets.QToolButton(translate_box)
        self.align_right_button = QtWidgets.QToolButton(translate_box)
        self.align_top_button = QtWidgets.QToolButton(translate_box)
        self.align_vcenter_button = QtWidgets.QToolButton(translate_box)
        self.align_bottom_button = QtWidgets.QToolButton(translate_box)
        self.align_left_button.setText("Left")
        self.align_hcenter_button.setText("H-Center")
        self.align_right_button.setText("Right")
        self.align_top_button.setText("Top")
        self.align_vcenter_button.setText("V-Center")
        self.align_bottom_button.setText("Bottom")
        self.align_left_button.clicked.connect(
            lambda: self._request_align(horizontal="left")
        )
        self.align_hcenter_button.clicked.connect(
            lambda: self._request_align(horizontal="center")
        )
        self.align_right_button.clicked.connect(
            lambda: self._request_align(horizontal="right")
        )
        self.align_top_button.clicked.connect(
            lambda: self._request_align(vertical="top")
        )
        self.align_vcenter_button.clicked.connect(
            lambda: self._request_align(vertical="center")
        )
        self.align_bottom_button.clicked.connect(
            lambda: self._request_align(vertical="bottom")
        )
        quick_align_grid.addWidget(self.align_left_button, 0, 0)
        quick_align_grid.addWidget(self.align_hcenter_button, 0, 1)
        quick_align_grid.addWidget(self.align_right_button, 0, 2)
        quick_align_grid.addWidget(self.align_top_button, 1, 0)
        quick_align_grid.addWidget(self.align_vcenter_button, 1, 1)
        quick_align_grid.addWidget(self.align_bottom_button, 1, 2)
        translate_layout.addLayout(quick_align_grid)

        self.interactive_resize_button = QtWidgets.QPushButton(
            "Interactive Resize", translate_box
        )
        self.interactive_resize_button.setCheckable(True)
        self.interactive_resize_button.toggled.connect(
            self.layerInteractiveResizeToggled.emit
        )
        translate_layout.addWidget(self.interactive_resize_button)

        self.alignment_hint_label = QtWidgets.QLabel(
            "Shortcut: Alt+Arrows nudge, Alt+0 resets", translate_box
        )
        self.alignment_hint_label.setWordWrap(True)
        self.alignment_hint_label.setProperty("class", "mutedHint")
        translate_layout.addWidget(self.alignment_hint_label)

        translate_section, self.translate_section_toggle = _build_collapsible_section(
            controls_container,
            "Align / Nudge",
            translate_box,
            expanded=True,
        )
        controls_layout.addWidget(translate_section)

        settings_box = QtWidgets.QGroupBox("", controls_container)
        settings_layout = QtWidgets.QVBoxLayout(settings_box)
        settings_layout.setContentsMargins(8, 8, 8, 8)
        settings_layout.setSpacing(6)

        source_row = QtWidgets.QHBoxLayout()
        source_label = QtWidgets.QLabel("Settings File", settings_box)
        self.source_path_edit = QtWidgets.QLineEdit(settings_box)
        self.source_path_edit.setReadOnly(True)
        self.open_source_button = QtWidgets.QToolButton(settings_box)
        self.open_source_button.setText("Open")
        self.open_source_button.clicked.connect(self._request_open_source)
        self.reveal_source_button = QtWidgets.QToolButton(settings_box)
        self.reveal_source_button.setText("Folder")
        self.reveal_source_button.clicked.connect(self._request_open_source_folder)
        source_row.addWidget(source_label)
        source_row.addWidget(self.source_path_edit, 1)
        source_row.addWidget(self.open_source_button)
        source_row.addWidget(self.reveal_source_button)
        settings_layout.addLayout(source_row)

        self.source_state_label = QtWidgets.QLabel("", settings_box)
        self.source_state_label.setWordWrap(True)
        self.source_state_label.setProperty("class", "mutedHint")
        settings_layout.addWidget(self.source_state_label)

        name_row = QtWidgets.QHBoxLayout()
        name_label = QtWidgets.QLabel("Name", settings_box)
        self.layer_name_edit = QtWidgets.QLineEdit(settings_box)
        name_row.addWidget(name_label)
        name_row.addWidget(self.layer_name_edit, 1)
        settings_layout.addLayout(name_row)

        page_row = QtWidgets.QHBoxLayout()
        page_label = QtWidgets.QLabel("Page", settings_box)
        self.page_spin = QtWidgets.QSpinBox(settings_box)
        self.page_spin.setRange(1, 1000000)
        self.page_spin.setValue(1)
        page_row.addWidget(page_label)
        page_row.addWidget(self.page_spin, 1)
        settings_layout.addLayout(page_row)

        transform_grid = QtWidgets.QGridLayout()
        tx_label = QtWidgets.QLabel("Offset X", settings_box)
        ty_label = QtWidgets.QLabel("Offset Y", settings_box)
        sx_label = QtWidgets.QLabel("Scale X", settings_box)
        sy_label = QtWidgets.QLabel("Scale Y", settings_box)
        self.tx_spin = QtWidgets.QDoubleSpinBox(settings_box)
        self.ty_spin = QtWidgets.QDoubleSpinBox(settings_box)
        self.sx_spin = QtWidgets.QDoubleSpinBox(settings_box)
        self.sy_spin = QtWidgets.QDoubleSpinBox(settings_box)
        for spin in (self.tx_spin, self.ty_spin):
            spin.setRange(-1000000.0, 1000000.0)
            spin.setDecimals(3)
            spin.setSingleStep(1.0)
            spin.setValue(0.0)
            spin.setSuffix(" px")
        for spin in (self.sx_spin, self.sy_spin):
            spin.setRange(0.000001, 1000000.0)
            spin.setDecimals(4)
            spin.setSingleStep(0.01)
            spin.setValue(1.0)
        transform_grid.addWidget(tx_label, 0, 0)
        transform_grid.addWidget(self.tx_spin, 0, 1)
        transform_grid.addWidget(ty_label, 0, 2)
        transform_grid.addWidget(self.ty_spin, 0, 3)
        transform_grid.addWidget(sx_label, 1, 0)
        transform_grid.addWidget(self.sx_spin, 1, 1)
        transform_grid.addWidget(sy_label, 1, 2)
        transform_grid.addWidget(self.sy_spin, 1, 3)
        settings_layout.addLayout(transform_grid)

        button_row = QtWidgets.QHBoxLayout()
        self.reload_settings_button = QtWidgets.QPushButton("Reload", settings_box)
        self.apply_settings_button = QtWidgets.QPushButton("Apply", settings_box)
        self.save_settings_button = QtWidgets.QPushButton("Save Settings", settings_box)
        self.reload_settings_button.clicked.connect(
            self._reload_selected_layer_settings
        )
        self.apply_settings_button.clicked.connect(self._request_apply_settings)
        self.save_settings_button.clicked.connect(self._request_save_settings)
        button_row.addWidget(self.reload_settings_button)
        button_row.addWidget(self.apply_settings_button)
        button_row.addWidget(self.save_settings_button)
        settings_layout.addLayout(button_row)

        alignment_row = QtWidgets.QHBoxLayout()
        self.copy_alignment_button = QtWidgets.QPushButton(
            "Copy Alignment", settings_box
        )
        self.paste_alignment_button = QtWidgets.QPushButton(
            "Paste Alignment", settings_box
        )
        self.copy_alignment_button.clicked.connect(self._copy_alignment)
        self.paste_alignment_button.clicked.connect(self._paste_alignment)
        alignment_row.addWidget(self.copy_alignment_button)
        alignment_row.addWidget(self.paste_alignment_button)
        settings_layout.addLayout(alignment_row)

        self.alignment_copy_hint_label = QtWidgets.QLabel("", settings_box)
        self.alignment_copy_hint_label.setWordWrap(True)
        self.alignment_copy_hint_label.setProperty("class", "mutedHint")
        settings_layout.addWidget(self.alignment_copy_hint_label)

        settings_section, self.settings_section_toggle = _build_collapsible_section(
            controls_container,
            "Layer Settings",
            settings_box,
            expanded=False,
        )
        controls_layout.addWidget(settings_section)

        move_row = QtWidgets.QHBoxLayout()
        self.move_up_button = QtWidgets.QPushButton("Move Up", controls_container)
        self.move_down_button = QtWidgets.QPushButton("Move Down", controls_container)
        self.move_up_button.clicked.connect(lambda: self._request_move(-1))
        self.move_down_button.clicked.connect(lambda: self._request_move(1))
        move_row.addWidget(self.move_up_button)
        move_row.addWidget(self.move_down_button)
        controls_layout.addLayout(move_row)
        controls_layout.addStretch(1)

        self.setWidget(container)
        self._set_opacity_enabled(False)
        self._set_reorder_enabled(False)
        self._set_translate_enabled(False)
        self._set_settings_enabled(False)
        self._install_shortcuts()
        self._update_layer_list_height()

    def _update_layer_list_height(self) -> None:
        count = int(self.layer_list.count())
        row_hint = int(self.layer_list.sizeHintForRow(0))
        if row_hint <= 0:
            row_hint = max(22, self.layer_list.fontMetrics().height() + 10)
        frame = int(self.layer_list.frameWidth() * 2)
        list_margins = int(
            self.layer_list.contentsMargins().top()
            + self.layer_list.contentsMargins().bottom()
        )
        # Keep the list compact for small layer counts while still usable.
        visible_rows = max(3, min(10, count if count > 0 else 3))
        target_height = int((row_hint * visible_rows) + frame + list_margins)
        self.layer_list.setMinimumHeight(target_height)
        self.layer_list.setMaximumHeight(target_height)

    def _set_opacity_enabled(self, enabled: bool) -> None:
        self.opacity_slider.setEnabled(bool(enabled))

    def _set_reorder_enabled(self, enabled: bool) -> None:
        enabled_flag = bool(enabled)
        self.move_up_button.setEnabled(enabled_flag)
        self.move_down_button.setEnabled(enabled_flag)

    def _set_translate_enabled(self, enabled: bool) -> None:
        enabled_flag = bool(enabled)
        self.translate_step_spin.setEnabled(enabled_flag)
        self.nudge_left_button.setEnabled(enabled_flag)
        self.nudge_right_button.setEnabled(enabled_flag)
        self.nudge_up_button.setEnabled(enabled_flag)
        self.nudge_down_button.setEnabled(enabled_flag)
        self.reset_translate_button.setEnabled(enabled_flag)
        self.scale_step_spin.setEnabled(enabled_flag)
        self.shrink_button.setEnabled(enabled_flag)
        self.expand_button.setEnabled(enabled_flag)
        self.align_left_button.setEnabled(enabled_flag)
        self.align_hcenter_button.setEnabled(enabled_flag)
        self.align_right_button.setEnabled(enabled_flag)
        self.align_top_button.setEnabled(enabled_flag)
        self.align_vcenter_button.setEnabled(enabled_flag)
        self.align_bottom_button.setEnabled(enabled_flag)
        self.interactive_resize_button.setEnabled(enabled_flag)
        if not enabled_flag and self.interactive_resize_button.isChecked():
            with QtCore.QSignalBlocker(self.interactive_resize_button):
                self.interactive_resize_button.setChecked(False)
            self.layerInteractiveResizeToggled.emit(False)

    def _set_settings_enabled(self, enabled: bool) -> None:
        enabled_flag = bool(enabled)
        self.source_path_edit.setEnabled(enabled_flag)
        self.open_source_button.setEnabled(enabled_flag)
        self.reveal_source_button.setEnabled(enabled_flag)
        self.layer_name_edit.setEnabled(enabled_flag)
        self.page_spin.setEnabled(enabled_flag)
        self.tx_spin.setEnabled(enabled_flag)
        self.ty_spin.setEnabled(enabled_flag)
        self.sx_spin.setEnabled(enabled_flag)
        self.sy_spin.setEnabled(enabled_flag)
        self.reload_settings_button.setEnabled(enabled_flag)
        self.apply_settings_button.setEnabled(enabled_flag)
        self.save_settings_button.setEnabled(enabled_flag)
        self.copy_alignment_button.setEnabled(enabled_flag)
        self.paste_alignment_button.setEnabled(
            enabled_flag and self._copied_alignment is not None
        )

    def _update_alignment_copy_hint(self) -> None:
        if self._copied_alignment is None:
            self.alignment_copy_hint_label.setText("No copied alignment.")
            return
        source = str(self._copied_alignment.get("source_name") or "Unknown")
        self.alignment_copy_hint_label.setText(f"Copied from: {source}")

    def _populate_settings_fields(self, layer: dict) -> None:
        self._settings_syncing = True
        try:
            self.source_path_edit.setText(str(layer.get("source_path") or ""))
            source_path = str(layer.get("source_path") or "").strip()
            if source_path:
                exists = Path(source_path).expanduser().exists()
                self.source_state_label.setText(
                    "Settings file found." if exists else "Settings file is missing."
                )
            else:
                self.source_state_label.setText("")
            self.layer_name_edit.setText(str(layer.get("name") or ""))
            self.page_spin.setValue(max(1, int(layer.get("page_index", 0) or 0) + 1))
            self.tx_spin.setValue(float(layer.get("tx", 0.0) or 0.0))
            self.ty_spin.setValue(float(layer.get("ty", 0.0) or 0.0))
            self.sx_spin.setValue(max(0.000001, float(layer.get("sx", 1.0) or 1.0)))
            self.sy_spin.setValue(max(0.000001, float(layer.get("sy", 1.0) or 1.0)))
        finally:
            self._settings_syncing = False

    def _settings_payload(self) -> dict:
        return {
            "name": str(self.layer_name_edit.text() or "").strip(),
            "page_index": int(self.page_spin.value()) - 1,
            "tx": float(self.tx_spin.value()),
            "ty": float(self.ty_spin.value()),
            "sx": float(self.sx_spin.value()),
            "sy": float(self.sy_spin.value()),
        }

    def _copy_alignment(self) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_settings", False)):
            return
        self._copied_alignment = {
            "source_layer_id": layer_id,
            "source_name": str(layer.get("name") or layer_id),
            "tx": float(self.tx_spin.value()),
            "ty": float(self.ty_spin.value()),
            "sx": float(self.sx_spin.value()),
            "sy": float(self.sy_spin.value()),
        }
        self._update_alignment_copy_hint()
        self._set_settings_enabled(True)

    def _paste_alignment(self) -> None:
        if self._copied_alignment is None or self._settings_syncing:
            return
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_settings", False)):
            return
        self.tx_spin.setValue(float(self._copied_alignment.get("tx", 0.0) or 0.0))
        self.ty_spin.setValue(float(self._copied_alignment.get("ty", 0.0) or 0.0))
        self.sx_spin.setValue(
            max(0.000001, float(self._copied_alignment.get("sx", 1.0) or 1.0))
        )
        self.sy_spin.setValue(
            max(0.000001, float(self._copied_alignment.get("sy", 1.0) or 1.0))
        )
        self._request_apply_settings()

    def _reload_selected_layer_settings(self) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        self._populate_settings_fields(layer)

    def _request_open_source(self) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not str(layer.get("source_path") or "").strip():
            return
        self.layerOpenSourceRequested.emit(layer_id)

    def _request_open_source_folder(self) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not str(layer.get("source_path") or "").strip():
            return
        self.layerOpenSourceFolderRequested.emit(layer_id)

    def _request_apply_settings(self) -> None:
        if self._settings_syncing:
            return
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_settings", False)):
            return
        self.layerApplySettingsRequested.emit(layer_id, self._settings_payload())

    def _request_save_settings(self) -> None:
        if self._settings_syncing:
            return
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_settings", False)):
            return
        self.layerSaveSettingsRequested.emit(layer_id, self._settings_payload())

    def _current_layer_id(self) -> str | None:
        item = self.layer_list.currentItem()
        if item is None:
            return None
        return str(item.data(QtCore.Qt.UserRole) or "") or None

    def _on_current_item_changed(self, current, _previous) -> None:
        layer_id = (
            str(current.data(QtCore.Qt.UserRole) or "") if current is not None else ""
        )
        layer = self._layer_map.get(layer_id, {})
        supports_opacity = bool(layer.get("supports_opacity", False))
        supports_translate = bool(layer.get("supports_translate", False))
        with QtCore.QSignalBlocker(self.opacity_slider):
            self.opacity_slider.setValue(
                int(round(float(layer.get("opacity", 1.0) or 1.0) * 100.0))
            )
        self._set_opacity_enabled(supports_opacity)
        self._set_reorder_enabled(bool(layer.get("supports_reorder", False)))
        self._set_translate_enabled(supports_translate)
        supports_settings = bool(layer.get("supports_settings", False))
        self._set_settings_enabled(supports_settings)
        self._populate_settings_fields(layer if supports_settings else {})
        self._update_alignment_copy_hint()
        self.details_label.setText(str(layer.get("details", "") or ""))
        if layer_id:
            self.layerSelected.emit(layer_id)

    def _on_item_changed(self, item) -> None:
        if item is None:
            return
        layer_id = str(item.data(QtCore.Qt.UserRole) or "")
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("checkable", True)):
            return
        visible = item.checkState() == QtCore.Qt.Checked
        self.layerVisibilityChanged.emit(layer_id, visible)

    def _on_opacity_changed(self, value: int) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_opacity", False)):
            return
        self.layerOpacityChanged.emit(layer_id, float(value) / 100.0)

    def _request_move(self, direction: int) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_reorder", False)):
            return
        if int(direction) not in {-1, 1}:
            return
        self.layerMoveRequested.emit(layer_id, int(direction))

    def _request_translate(self, horizontal: int, vertical: int) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_translate", False)):
            return
        step = float(self.translate_step_spin.value())
        dx = float(horizontal) * step
        dy = float(vertical) * step
        if abs(dx) < 1e-12 and abs(dy) < 1e-12:
            return
        self.layerTranslateRequested.emit(layer_id, dx, dy)

    def _request_translate_reset(self) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_translate", False)):
            return
        self.layerResetTransformRequested.emit(layer_id)

    def _request_scale_relative(self, factor: float) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_translate", False)):
            return
        normalized = max(1e-6, float(factor))
        self.layerQuickTransformRequested.emit(
            layer_id,
            {
                "action": "scale",
                "factor": float(normalized),
                "keep_center": True,
            },
        )

    def _request_shrink(self) -> None:
        step = max(0.001, float(self.scale_step_spin.value()))
        self._request_scale_relative(max(1e-6, 1.0 - (step / 100.0)))

    def _request_expand(self) -> None:
        step = max(0.001, float(self.scale_step_spin.value()))
        self._request_scale_relative(1.0 + (step / 100.0))

    def _request_align(
        self,
        *,
        horizontal: str | None = None,
        vertical: str | None = None,
    ) -> None:
        layer_id = self._current_layer_id()
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_translate", False)):
            return
        self.layerQuickTransformRequested.emit(
            layer_id,
            {
                "action": "align",
                "horizontal": str(horizontal or "").strip().lower() or None,
                "vertical": str(vertical or "").strip().lower() or None,
            },
        )

    def _install_shortcuts(self) -> None:
        shortcuts = [
            (QtGui.QKeySequence("Alt+Left"), lambda: self._request_translate(-1, 0)),
            (QtGui.QKeySequence("Alt+Right"), lambda: self._request_translate(1, 0)),
            (QtGui.QKeySequence("Alt+Up"), lambda: self._request_translate(0, -1)),
            (QtGui.QKeySequence("Alt+Down"), lambda: self._request_translate(0, 1)),
            (QtGui.QKeySequence("Alt+0"), self._request_translate_reset),
        ]
        self._shortcuts.clear()
        for sequence, callback in shortcuts:
            shortcut = QtWidgets.QShortcut(sequence, self)
            shortcut.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _on_context_menu(self, pos) -> None:
        item = self.layer_list.itemAt(pos)
        if item is None:
            return
        layer_id = str(item.data(QtCore.Qt.UserRole) or "")
        if not layer_id:
            return
        layer = self._layer_map.get(layer_id, {})
        if not bool(layer.get("supports_reorder", False)):
            return
        self.layer_list.setCurrentItem(item)
        menu = QtWidgets.QMenu(self.layer_list)
        rename_action = menu.addAction("Rename Layer")
        remove_action = menu.addAction("Remove Layer")
        menu.addSeparator()
        move_top_action = menu.addAction("Move to Top")
        move_bottom_action = menu.addAction("Move to Bottom")
        chosen = menu.exec(self.layer_list.mapToGlobal(pos))
        if chosen is rename_action:
            current_name = str(layer.get("name") or layer_id)
            new_name, ok = QtWidgets.QInputDialog.getText(
                self.layer_list,
                "Rename Layer",
                "Layer name:",
                text=current_name,
            )
            if ok:
                normalized = str(new_name or "").strip()
                if normalized and normalized != current_name:
                    self.layerRenameRequested.emit(layer_id, normalized)
            return
        if chosen is remove_action:
            self.layerRemoveRequested.emit(layer_id)
            return
        if chosen is move_top_action:
            self.layerMoveToTopRequested.emit(layer_id)
            return
        if chosen is move_bottom_action:
            self.layerMoveToBottomRequested.emit(layer_id)

    def set_layers(self, layers: list[dict]) -> None:
        self._layer_map = {
            str(layer.get("id") or ""): dict(layer or {})
            for layer in list(layers or [])
            if str(layer.get("id") or "")
        }
        selected_layer_id = self._current_layer_id()
        with QtCore.QSignalBlocker(self.layer_list):
            self.layer_list.clear()
            for layer in list(layers or []):
                layer_id = str(layer.get("id") or "")
                if not layer_id:
                    continue
                item = QtWidgets.QListWidgetItem(str(layer.get("name") or layer_id))
                item.setData(QtCore.Qt.UserRole, layer_id)
                flags = item.flags()
                if bool(layer.get("checkable", True)):
                    item.setFlags(
                        flags
                        | QtCore.Qt.ItemIsUserCheckable
                        | QtCore.Qt.ItemIsSelectable
                        | QtCore.Qt.ItemIsEnabled
                    )
                    item.setCheckState(
                        QtCore.Qt.Checked
                        if bool(layer.get("visible", True))
                        else QtCore.Qt.Unchecked
                    )
                else:
                    item.setFlags(flags & ~QtCore.Qt.ItemIsUserCheckable)
                self.layer_list.addItem(item)
            if selected_layer_id:
                for index in range(self.layer_list.count()):
                    item = self.layer_list.item(index)
                    if str(item.data(QtCore.Qt.UserRole) or "") == selected_layer_id:
                        self.layer_list.setCurrentItem(item)
                        break
            if self.layer_list.currentItem() is None and self.layer_list.count() > 0:
                self.layer_list.setCurrentRow(0)
        self._update_layer_list_height()
        if self.layer_list.currentItem() is None:
            self.details_label.setText("")
            self._set_opacity_enabled(False)
            self._set_reorder_enabled(False)
            self._set_translate_enabled(False)
            self._set_settings_enabled(False)
            self._populate_settings_fields({})
            self._update_alignment_copy_hint()
            return
        self._on_current_item_changed(self.layer_list.currentItem(), None)
