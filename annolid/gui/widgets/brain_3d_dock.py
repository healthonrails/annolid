from __future__ import annotations

from qtpy import QtCore, QtWidgets


class Brain3DSessionDockWidget(QtWidgets.QDockWidget):
    rebuildRequested = QtCore.Signal()
    regenerateRequested = QtCore.Signal()
    applyEditsRequested = QtCore.Signal()
    planeSelectionChanged = QtCore.Signal(int)
    regionStateRequested = QtCore.Signal(int, str, str)

    def __init__(self, parent=None):
        super().__init__("Brain 3D Session", parent)
        self.setObjectName("brain3dSessionDock")
        self._regions: dict[str, dict] = {}
        self._build_ui()

    def _build_ui(self) -> None:
        container = QtWidgets.QWidget(self)
        layout = QtWidgets.QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.summary_label = QtWidgets.QLabel("", container)
        self.summary_label.setWordWrap(True)
        layout.addWidget(self.summary_label)

        plane_row = QtWidgets.QHBoxLayout()
        plane_row.addWidget(QtWidgets.QLabel("Coronal Plane", container))
        self.plane_spin = QtWidgets.QSpinBox(container)
        self.plane_spin.setRange(0, 0)
        self.plane_spin.valueChanged.connect(self._on_plane_changed)
        plane_row.addWidget(self.plane_spin, stretch=1)
        layout.addLayout(plane_row)

        self.region_list = QtWidgets.QListWidget(container)
        self.region_list.currentItemChanged.connect(self._on_region_selection_changed)
        layout.addWidget(self.region_list, stretch=1)

        state_row = QtWidgets.QHBoxLayout()
        state_row.addWidget(QtWidgets.QLabel("State", container))
        self.state_combo = QtWidgets.QComboBox(container)
        self.state_combo.addItem("Present", "present")
        self.state_combo.addItem("Hidden", "hidden")
        self.state_combo.addItem("Created", "created")
        state_row.addWidget(self.state_combo, stretch=1)
        self.apply_state_button = QtWidgets.QPushButton("Apply Region State", container)
        self.apply_state_button.clicked.connect(self._emit_apply_region_state)
        state_row.addWidget(self.apply_state_button)
        layout.addLayout(state_row)

        actions_row = QtWidgets.QGridLayout()
        self.rebuild_button = QtWidgets.QPushButton("Build Model", container)
        self.rebuild_button.clicked.connect(self.rebuildRequested.emit)
        self.regenerate_button = QtWidgets.QPushButton("Regenerate Coronal", container)
        self.regenerate_button.clicked.connect(self.regenerateRequested.emit)
        self.apply_edits_button = QtWidgets.QPushButton(
            "Apply Current Edits", container
        )
        self.apply_edits_button.clicked.connect(self.applyEditsRequested.emit)
        actions_row.addWidget(self.rebuild_button, 0, 0)
        actions_row.addWidget(self.regenerate_button, 0, 1)
        actions_row.addWidget(self.apply_edits_button, 1, 0, 1, 2)
        layout.addLayout(actions_row)

        self.setWidget(container)
        self._set_enabled(False)

    def _set_enabled(self, enabled: bool) -> None:
        self.plane_spin.setEnabled(bool(enabled))
        self.region_list.setEnabled(bool(enabled))
        self.state_combo.setEnabled(bool(enabled))
        self.apply_state_button.setEnabled(bool(enabled))
        self.regenerate_button.setEnabled(bool(enabled))
        self.apply_edits_button.setEnabled(bool(enabled))

    def set_summary(
        self,
        *,
        region_count: int,
        source_page_count: int,
        plane_count: int,
    ) -> None:
        self.summary_label.setText(
            f"Regions: {int(region_count)} | Source sagittal pages: {int(source_page_count)} | Coronal planes: {int(plane_count)}"
        )
        self._set_enabled(bool(region_count > 0 and plane_count > 0))
        with QtCore.QSignalBlocker(self.plane_spin):
            self.plane_spin.setRange(0, max(0, int(plane_count) - 1))

    def set_current_plane(self, plane_index: int) -> None:
        with QtCore.QSignalBlocker(self.plane_spin):
            self.plane_spin.setValue(max(0, int(plane_index)))

    def set_regions(self, regions: list[dict]) -> None:
        self._regions = {}
        selected_region = self.selected_region_id()
        with QtCore.QSignalBlocker(self.region_list):
            self.region_list.clear()
            for region in list(regions or []):
                region_id = str(region.get("region_id", "") or "")
                if not region_id:
                    continue
                self._regions[region_id] = dict(region)
                label = str(region.get("label", "") or region_id)
                state = str(region.get("state", "present") or "present")
                source = str(region.get("source", "model") or "model")
                points = int(region.get("points_count", 0) or 0)
                item = QtWidgets.QListWidgetItem(
                    f"{label} | state={state} | source={source} | points={points}"
                )
                item.setData(QtCore.Qt.UserRole, region_id)
                self.region_list.addItem(item)
        if selected_region:
            for idx in range(self.region_list.count()):
                item = self.region_list.item(idx)
                if str(item.data(QtCore.Qt.UserRole) or "") == selected_region:
                    self.region_list.setCurrentItem(item)
                    break
        if self.region_list.count() > 0 and self.region_list.currentItem() is None:
            self.region_list.setCurrentRow(0)
        self._on_region_selection_changed()

    def selected_region_id(self) -> str:
        item = self.region_list.currentItem()
        if item is None:
            return ""
        return str(item.data(QtCore.Qt.UserRole) or "")

    def _on_plane_changed(self, value: int) -> None:
        self.planeSelectionChanged.emit(int(value))

    def _on_region_selection_changed(self, *_args) -> None:
        region_id = self.selected_region_id()
        region = dict(self._regions.get(region_id) or {})
        state = str(region.get("state", "present") or "present")
        if state not in {"present", "hidden", "created"}:
            state = "present"
        with QtCore.QSignalBlocker(self.state_combo):
            for idx in range(self.state_combo.count()):
                if str(self.state_combo.itemData(idx) or "") == state:
                    self.state_combo.setCurrentIndex(idx)
                    break
        self.apply_state_button.setEnabled(bool(region_id))

    def _emit_apply_region_state(self) -> None:
        region_id = self.selected_region_id()
        if not region_id:
            return
        state = str(self.state_combo.currentData() or "present")
        self.regionStateRequested.emit(int(self.plane_spin.value()), region_id, state)
