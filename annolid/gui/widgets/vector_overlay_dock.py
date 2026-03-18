from __future__ import annotations

from pathlib import Path

from qtpy import QtCore, QtWidgets


class VectorOverlayDockWidget(QtWidgets.QDockWidget):
    overlayApplyRequested = QtCore.Signal(str, dict)
    overlayResetRequested = QtCore.Signal(str)
    overlayLandmarkAlignRequested = QtCore.Signal(str)
    overlayPairSelectedRequested = QtCore.Signal(str)
    overlayPairSelectionChanged = QtCore.Signal(str, str)
    overlayRemovePairRequested = QtCore.Signal(str, str)
    overlayClearPairsRequested = QtCore.Signal(str)
    overlayImportFitModeChanged = QtCore.Signal(str)
    overlayImportFitMarginChanged = QtCore.Signal(float)
    overlayRefitRequested = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__("Vector Overlays", parent)
        self.setObjectName("vectorOverlayDock")
        self._overlay_map: dict[str, dict] = {}
        self._current_overlay_id: str | None = None
        self._current_pair_id: str | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.overlay_list = QtWidgets.QListWidget(container)
        self.overlay_list.currentItemChanged.connect(self._on_current_item_changed)
        layout.addWidget(self.overlay_list)

        self.landmark_status_label = QtWidgets.QLabel("", container)
        self.landmark_status_label.setWordWrap(True)
        layout.addWidget(self.landmark_status_label)

        self.landmark_pairs_list = QtWidgets.QListWidget(container)
        self.landmark_pairs_list.currentItemChanged.connect(
            self._on_current_pair_changed
        )
        layout.addWidget(self.landmark_pairs_list)

        import_group = QtWidgets.QGroupBox("Import Fit", container)
        import_layout = QtWidgets.QFormLayout(import_group)
        import_layout.setContentsMargins(10, 10, 10, 10)
        import_layout.setLabelAlignment(QtCore.Qt.AlignLeft)
        self.import_fit_combo = QtWidgets.QComboBox(import_group)
        self.import_fit_combo.addItem("Auto", "auto")
        self.import_fit_combo.addItem("Document Bounds", "document")
        self.import_fit_combo.addItem("Shape Bounds", "shape")
        self.import_fit_combo.currentIndexChanged.connect(self._emit_import_fit_mode)
        self.import_fit_margin_spin = QtWidgets.QDoubleSpinBox(import_group)
        self.import_fit_margin_spin.setRange(0.1, 1.0)
        self.import_fit_margin_spin.setSingleStep(0.05)
        self.import_fit_margin_spin.setDecimals(2)
        self.import_fit_margin_spin.setValue(1.0)
        self.import_fit_margin_spin.valueChanged.connect(self._emit_import_fit_margin)
        import_layout.addRow("Mode", self.import_fit_combo)
        import_layout.addRow("Margin", self.import_fit_margin_spin)
        layout.addWidget(import_group)

        transform_group = QtWidgets.QGroupBox("Transform", container)
        transform_layout = QtWidgets.QFormLayout(transform_group)
        transform_layout.setContentsMargins(10, 10, 10, 10)
        transform_layout.setLabelAlignment(QtCore.Qt.AlignLeft)

        self.visible_checkbox = QtWidgets.QCheckBox("Visible", transform_group)
        transform_layout.addRow("State", self.visible_checkbox)

        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal, transform_group)
        self.opacity_slider.setRange(0, 100)
        transform_layout.addRow("Opacity", self.opacity_slider)

        self.tx_spin = self._double_spin(
            transform_group, minimum=-1_000_000.0, maximum=1_000_000.0
        )
        self.ty_spin = self._double_spin(
            transform_group, minimum=-1_000_000.0, maximum=1_000_000.0
        )
        self.sx_spin = self._double_spin(
            transform_group, minimum=0.01, maximum=1_000.0, value=1.0
        )
        self.sy_spin = self._double_spin(
            transform_group, minimum=0.01, maximum=1_000.0, value=1.0
        )
        self.rotation_spin = self._double_spin(
            transform_group, minimum=-360.0, maximum=360.0
        )
        self.z_order_spin = QtWidgets.QSpinBox(transform_group)
        self.z_order_spin.setRange(-1000, 1000)

        transform_layout.addRow("Translate X", self.tx_spin)
        transform_layout.addRow("Translate Y", self.ty_spin)
        transform_layout.addRow("Scale X", self.sx_spin)
        transform_layout.addRow("Scale Y", self.sy_spin)
        transform_layout.addRow("Rotation", self.rotation_spin)
        transform_layout.addRow("Z Order", self.z_order_spin)
        layout.addWidget(transform_group)

        actions_group = QtWidgets.QGroupBox("Actions", container)
        buttons = QtWidgets.QGridLayout(actions_group)
        buttons.setContentsMargins(10, 10, 10, 10)
        buttons.setHorizontalSpacing(8)
        buttons.setVerticalSpacing(8)
        self.apply_button = QtWidgets.QPushButton("Apply", container)
        self.reset_button = QtWidgets.QPushButton("Reset", container)
        self.refit_button = QtWidgets.QPushButton("Re-fit", container)
        self.pair_selected_button = QtWidgets.QPushButton("Pair Selected", container)
        self.remove_pair_button = QtWidgets.QPushButton("Remove Pair", container)
        self.clear_pairs_button = QtWidgets.QPushButton("Clear Pairs", container)
        self.align_landmarks_button = QtWidgets.QPushButton("Align Points", container)
        self.apply_button.clicked.connect(self._emit_apply_request)
        self.reset_button.clicked.connect(self._emit_reset_request)
        self.refit_button.clicked.connect(self._emit_refit_request)
        self.pair_selected_button.clicked.connect(self._emit_pair_selected_request)
        self.remove_pair_button.clicked.connect(self._emit_remove_pair_request)
        self.clear_pairs_button.clicked.connect(self._emit_clear_pairs_request)
        self.align_landmarks_button.clicked.connect(self._emit_landmark_align_request)
        buttons.addWidget(self.apply_button, 0, 0)
        buttons.addWidget(self.reset_button, 0, 1)
        buttons.addWidget(self.refit_button, 1, 0)
        buttons.addWidget(self.pair_selected_button, 1, 1)
        buttons.addWidget(self.remove_pair_button, 2, 0)
        buttons.addWidget(self.clear_pairs_button, 2, 1)
        buttons.addWidget(self.align_landmarks_button, 3, 0, 1, 2)
        layout.addWidget(actions_group)

        layout.addStretch(1)

        self.setWidget(container)
        self._set_controls_enabled(False)

    def _double_spin(
        self,
        parent,
        *,
        minimum: float,
        maximum: float,
        value: float = 0.0,
    ) -> QtWidgets.QDoubleSpinBox:
        spin = QtWidgets.QDoubleSpinBox(parent)
        spin.setDecimals(3)
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(0.1)
        return spin

    def _selected_overlay_id(self) -> str | None:
        item = self.overlay_list.currentItem()
        if item is None:
            return self._current_overlay_id
        return str(item.data(QtCore.Qt.UserRole) or "") or self._current_overlay_id

    def _set_controls_enabled(self, enabled: bool) -> None:
        for widget in (
            self.visible_checkbox,
            self.opacity_slider,
            self.tx_spin,
            self.ty_spin,
            self.sx_spin,
            self.sy_spin,
            self.rotation_spin,
            self.z_order_spin,
            self.apply_button,
            self.reset_button,
            self.refit_button,
            self.pair_selected_button,
            self.remove_pair_button,
            self.clear_pairs_button,
            self.align_landmarks_button,
        ):
            widget.setEnabled(bool(enabled))
        self.import_fit_combo.setEnabled(True)
        self.import_fit_margin_spin.setEnabled(True)
        self._update_pair_buttons()

    def import_fit_mode(self) -> str:
        value = str(self.import_fit_combo.currentData() or "").strip().lower()
        return value if value in {"auto", "document", "shape"} else "auto"

    def set_import_fit_mode(self, mode: str) -> None:
        target = str(mode or "").strip().lower()
        if target not in {"auto", "document", "shape"}:
            target = "auto"
        with QtCore.QSignalBlocker(self.import_fit_combo):
            for index in range(self.import_fit_combo.count()):
                if str(self.import_fit_combo.itemData(index) or "") == target:
                    self.import_fit_combo.setCurrentIndex(index)
                    return
            self.import_fit_combo.setCurrentIndex(0)

    def import_fit_margin(self) -> float:
        return max(0.1, min(1.0, float(self.import_fit_margin_spin.value())))

    def set_import_fit_margin(self, margin: float) -> None:
        value = max(0.1, min(1.0, float(margin)))
        with QtCore.QSignalBlocker(self.import_fit_margin_spin):
            self.import_fit_margin_spin.setValue(value)

    def _emit_import_fit_mode(self, _index: int) -> None:
        self.overlayImportFitModeChanged.emit(self.import_fit_mode())

    def _emit_import_fit_margin(self, value: float) -> None:
        self.overlayImportFitMarginChanged.emit(max(0.1, min(1.0, float(value))))

    def _update_pair_buttons(self) -> None:
        enabled = bool(self._selected_overlay_id())
        has_pairs = self.landmark_pairs_list.count() > 0
        has_selected_pair = self.landmark_pairs_list.currentItem() is not None
        self.remove_pair_button.setEnabled(enabled and has_pairs and has_selected_pair)
        self.clear_pairs_button.setEnabled(enabled and has_pairs)

    def _selected_pair_id(self) -> str | None:
        item = self.landmark_pairs_list.currentItem()
        if item is None:
            return self._current_pair_id
        return str(item.data(QtCore.Qt.UserRole) or "") or self._current_pair_id

    def set_overlays(self, overlays: list[dict]) -> None:
        previous_id = self._selected_overlay_id()
        self._overlay_map = {
            str(overlay.get("id") or ""): dict(overlay or {}) for overlay in overlays
        }
        with QtCore.QSignalBlocker(self.overlay_list):
            self.overlay_list.clear()
            for overlay in overlays:
                overlay_id = str((overlay or {}).get("id") or "")
                source = Path(str((overlay or {}).get("source") or overlay_id)).name
                item = QtWidgets.QListWidgetItem(source)
                item.setData(QtCore.Qt.UserRole, overlay_id)
                self.overlay_list.addItem(item)
        target_id = previous_id if previous_id in self._overlay_map else None
        app = QtWidgets.QApplication.instance()
        platform_name = ""
        try:
            platform_name = str(app.platformName() or "").lower() if app else ""
        except Exception:
            platform_name = ""
        headless_platform = platform_name in {"minimal", "offscreen"}
        if target_id:
            self._current_overlay_id = target_id
            if not headless_platform:
                for index in range(self.overlay_list.count()):
                    item = self.overlay_list.item(index)
                    if str(item.data(QtCore.Qt.UserRole) or "") == target_id:
                        self.overlay_list.setCurrentItem(item)
                        return
            self._load_overlay(self._overlay_map.get(target_id))
            return
        if self.overlay_list.count() > 0:
            first_item = self.overlay_list.item(0)
            self._current_overlay_id = str(first_item.data(QtCore.Qt.UserRole) or "")
            if headless_platform:
                self._load_overlay(self._overlay_map.get(self._current_overlay_id))
            else:
                self.overlay_list.setCurrentRow(0)
        else:
            self._current_overlay_id = None
            self._load_overlay(None)

    def _on_current_item_changed(self, current, _previous) -> None:
        overlay_id = str(current.data(QtCore.Qt.UserRole) or "") if current else None
        self._current_overlay_id = overlay_id
        self._current_pair_id = None
        self._load_overlay(self._overlay_map.get(overlay_id) if overlay_id else None)

    def _on_current_pair_changed(self, current, _previous) -> None:
        self._current_pair_id = (
            str(current.data(QtCore.Qt.UserRole) or "") if current is not None else None
        )
        self._update_pair_buttons()
        overlay_id = self._selected_overlay_id()
        if overlay_id:
            self._refresh_landmark_status(overlay_id)
        if overlay_id:
            self.overlayPairSelectionChanged.emit(
                overlay_id, str(self._current_pair_id or "")
            )

    def _refresh_landmark_status(self, overlay_id: str | None) -> None:
        overlay = self._overlay_map.get(str(overlay_id or ""))
        if not overlay:
            self.landmark_status_label.setText("")
            return
        landmark_info = dict((overlay or {}).get("landmark_summary") or {})
        matched_count = int(landmark_info.get("matched_count", 0) or 0)
        explicit_count = int(landmark_info.get("explicit_count", 0) or 0)
        auto_count = int(landmark_info.get("auto_count", 0) or 0)
        labels = list(landmark_info.get("labels") or [])
        candidate = dict(landmark_info.get("pair_candidate") or {})
        explicit_pairs = list(landmark_info.get("explicit_pairs") or [])
        status_lines = []
        if labels:
            detail = ", ".join(str(label) for label in labels[:6])
            if len(labels) > 6:
                detail += ", ..."
            status_lines.append(
                f"Matched landmarks: {matched_count} [manual {explicit_count}, label {auto_count}] ({detail})"
            )
        else:
            status_lines.append(
                f"Matched landmarks: {matched_count} [manual {explicit_count}, label {auto_count}]"
            )
        pair_lookup = {
            str(pair.get("pair_id") or ""): pair for pair in explicit_pairs if pair
        }
        selected_pair = pair_lookup.get(str(self._current_pair_id or ""))
        if selected_pair:
            status_lines.append(
                "Selected pair: "
                f"{selected_pair.get('overlay_label')} -> {selected_pair.get('image_label')}"
            )
        if candidate:
            status_lines.append(
                "Pair selected ready: "
                f"{candidate.get('overlay_label')} -> {candidate.get('image_label')}"
            )
        self.landmark_status_label.setText("\n".join(status_lines))

    def _load_overlay(self, overlay: dict | None) -> None:
        enabled = overlay is not None
        self._set_controls_enabled(enabled)
        if not enabled:
            self.landmark_status_label.setText("")
            self.landmark_pairs_list.clear()
            return
        landmark_info = dict((overlay or {}).get("landmark_summary") or {})
        matched_count = int(landmark_info.get("matched_count", 0) or 0)
        candidate = dict(landmark_info.get("pair_candidate") or {})
        explicit_pairs = list(landmark_info.get("explicit_pairs") or [])
        with QtCore.QSignalBlocker(self.landmark_pairs_list):
            self.landmark_pairs_list.clear()
            for pair in explicit_pairs:
                item = QtWidgets.QListWidgetItem(
                    f"{pair.get('overlay_label')} -> {pair.get('image_label')}"
                )
                item.setData(QtCore.Qt.UserRole, str(pair.get("pair_id") or ""))
                self.landmark_pairs_list.addItem(item)
            target_pair_id = self._current_pair_id
            if self.landmark_pairs_list.count() > 0:
                if target_pair_id:
                    for index in range(self.landmark_pairs_list.count()):
                        item = self.landmark_pairs_list.item(index)
                        if str(item.data(QtCore.Qt.UserRole) or "") == target_pair_id:
                            self.landmark_pairs_list.setCurrentItem(item)
                            break
                    else:
                        self.landmark_pairs_list.setCurrentRow(0)
                else:
                    self.landmark_pairs_list.setCurrentRow(0)
                current_item = self.landmark_pairs_list.currentItem()
                self._current_pair_id = (
                    str(current_item.data(QtCore.Qt.UserRole) or "")
                    if current_item is not None
                    else None
                )
            else:
                self._current_pair_id = None
        self.pair_selected_button.setEnabled(bool(candidate))
        self._update_pair_buttons()
        self.align_landmarks_button.setEnabled(matched_count >= 3)
        overlay_id = str((overlay or {}).get("id") or "")
        self._refresh_landmark_status(overlay_id)
        if overlay_id:
            self.overlayPairSelectionChanged.emit(
                overlay_id, str(self._current_pair_id or "")
            )
        transform = dict((overlay or {}).get("transform") or {})
        with QtCore.QSignalBlocker(self.visible_checkbox):
            self.visible_checkbox.setChecked(bool(transform.get("visible", True)))
        with QtCore.QSignalBlocker(self.opacity_slider):
            self.opacity_slider.setValue(
                int(round(float(transform.get("opacity", 0.5)) * 100.0))
            )
        for widget, key in (
            (self.tx_spin, "tx"),
            (self.ty_spin, "ty"),
            (self.sx_spin, "sx"),
            (self.sy_spin, "sy"),
            (self.rotation_spin, "rotation_deg"),
        ):
            with QtCore.QSignalBlocker(widget):
                widget.setValue(float(transform.get(key, widget.value())))
        with QtCore.QSignalBlocker(self.z_order_spin):
            self.z_order_spin.setValue(
                int(transform.get("z_order", self.z_order_spin.value()))
            )

    def _emit_apply_request(self) -> None:
        overlay_id = self._selected_overlay_id()
        if not overlay_id:
            return
        payload = {
            "visible": self.visible_checkbox.isChecked(),
            "opacity": self.opacity_slider.value() / 100.0,
            "tx": self.tx_spin.value(),
            "ty": self.ty_spin.value(),
            "sx": self.sx_spin.value(),
            "sy": self.sy_spin.value(),
            "rotation_deg": self.rotation_spin.value(),
            "z_order": self.z_order_spin.value(),
        }
        self.overlayApplyRequested.emit(overlay_id, payload)

    def _emit_reset_request(self) -> None:
        overlay_id = self._selected_overlay_id()
        if overlay_id:
            self.overlayResetRequested.emit(overlay_id)

    def _emit_refit_request(self) -> None:
        overlay_id = self._selected_overlay_id()
        if overlay_id:
            self.overlayRefitRequested.emit(overlay_id)

    def _emit_landmark_align_request(self) -> None:
        overlay_id = self._selected_overlay_id()
        if overlay_id:
            self.overlayLandmarkAlignRequested.emit(overlay_id)

    def _emit_pair_selected_request(self) -> None:
        overlay_id = self._selected_overlay_id()
        if overlay_id:
            self.overlayPairSelectedRequested.emit(overlay_id)

    def _emit_remove_pair_request(self) -> None:
        overlay_id = self._selected_overlay_id()
        item = self.landmark_pairs_list.currentItem()
        if overlay_id and item is not None:
            pair_id = str(item.data(QtCore.Qt.UserRole) or "")
            if pair_id:
                self.overlayRemovePairRequested.emit(overlay_id, pair_id)

    def _emit_clear_pairs_request(self) -> None:
        overlay_id = self._selected_overlay_id()
        if overlay_id:
            self.overlayClearPairsRequested.emit(overlay_id)

    def set_selected_pair(self, overlay_id: str, pair_id: str | None) -> None:
        if str(overlay_id or "") != str(self._selected_overlay_id() or ""):
            return
        target_pair_id = str(pair_id or "")
        self._current_pair_id = target_pair_id or None
        with QtCore.QSignalBlocker(self.landmark_pairs_list):
            matched_item = None
            for index in range(self.landmark_pairs_list.count()):
                item = self.landmark_pairs_list.item(index)
                if str(item.data(QtCore.Qt.UserRole) or "") == target_pair_id:
                    matched_item = item
                    break
            self.landmark_pairs_list.setCurrentItem(matched_item)
        self._update_pair_buttons()

    def set_current_overlay(self, overlay_id: str | None) -> None:
        target = str(overlay_id or "")
        if not target:
            return
        with QtCore.QSignalBlocker(self.overlay_list):
            matched_item = None
            for index in range(self.overlay_list.count()):
                item = self.overlay_list.item(index)
                if str(item.data(QtCore.Qt.UserRole) or "") == target:
                    matched_item = item
                    break
            if matched_item is not None:
                self.overlay_list.setCurrentItem(matched_item)
                self._current_overlay_id = target
                self._current_pair_id = None
                self._load_overlay(self._overlay_map.get(target))
