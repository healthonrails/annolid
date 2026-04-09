from __future__ import annotations

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

    def __init__(self, parent=None):
        super().__init__("Layers", parent)
        self.setObjectName("viewerLayerDock")
        self._layer_map: dict[str, dict] = {}
        self._shortcuts: list[QtWidgets.QShortcut] = []
        self._build_ui()

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.layer_list = QtWidgets.QListWidget(container)
        self.layer_list.itemChanged.connect(self._on_item_changed)
        self.layer_list.currentItemChanged.connect(self._on_current_item_changed)
        self.layer_list.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.layer_list.customContextMenuRequested.connect(self._on_context_menu)
        layout.addWidget(self.layer_list)

        self.details_label = QtWidgets.QLabel("", container)
        self.details_label.setWordWrap(True)
        layout.addWidget(self.details_label)

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, container)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.valueChanged.connect(self._on_opacity_changed)
        layout.addWidget(self.opacity_slider)

        translate_box = QtWidgets.QGroupBox("Align / Nudge", container)
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

        self.alignment_hint_label = QtWidgets.QLabel(
            "Shortcut: Alt+Arrows nudge, Alt+0 resets", translate_box
        )
        self.alignment_hint_label.setWordWrap(True)
        self.alignment_hint_label.setProperty("class", "mutedHint")
        translate_layout.addWidget(self.alignment_hint_label)

        layout.addWidget(translate_box)

        move_row = QtWidgets.QHBoxLayout()
        self.move_up_button = QtWidgets.QPushButton("Move Up", container)
        self.move_down_button = QtWidgets.QPushButton("Move Down", container)
        self.move_up_button.clicked.connect(lambda: self._request_move(-1))
        self.move_down_button.clicked.connect(lambda: self._request_move(1))
        move_row.addWidget(self.move_up_button)
        move_row.addWidget(self.move_down_button)
        layout.addLayout(move_row)

        self.setWidget(container)
        self._set_opacity_enabled(False)
        self._set_reorder_enabled(False)
        self._set_translate_enabled(False)
        self._install_shortcuts()

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
        if self.layer_list.currentItem() is None:
            self.details_label.setText("")
            self._set_opacity_enabled(False)
            self._set_reorder_enabled(False)
