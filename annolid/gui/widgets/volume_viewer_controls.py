from __future__ import annotations
from typing import Any, Callable, Optional
from qtpy import QtCore, QtWidgets, QtGui


class VolumeQuickActionsWidget(QtWidgets.QGroupBox):
    """Group of quick action buttons for the volume viewer."""

    def __init__(
        self,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
        on_load_volume: Callable[[], None],
        on_reload_volume: Callable[[], None],
        on_load_points: Callable[[], None],
        on_load_mesh: Callable[[], None],
        on_save_snapshot: Callable[[], None],
        on_reset_view: Callable[[], None],
        on_show_help: Callable[[], None],
    ) -> None:
        super().__init__("Quick Actions", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        grid = QtWidgets.QGridLayout()
        grid.setSpacing(6)
        layout.addLayout(grid)

        # Style icons are usually available via the parent's style()
        style = self.style()

        self.btn_load = self._create_button(
            "Load Volume",
            style.standardIcon(QtWidgets.QStyle.SP_DirOpenIcon),
            on_load_volume,
            "Open a 3D volume, DICOM folder, OME-TIFF, or Zarr store.",
        )
        self.btn_reload = self._create_button(
            "Reload Volume",
            style.standardIcon(QtWidgets.QStyle.SP_BrowserReload),
            on_reload_volume,
            "Re-read the active volume from disk.",
        )
        self.btn_points = self._create_button(
            "Load Points",
            style.standardIcon(QtWidgets.QStyle.SP_FileIcon),
            on_load_points,
            "Load CSV/PLY/XYZ point clouds.",
        )
        self.btn_mesh = self._create_button(
            "Load Mesh",
            style.standardIcon(QtWidgets.QStyle.SP_FileDialogNewFolder),
            on_load_mesh,
            "Load STL/OBJ/PLY meshes.",
        )
        self.btn_snapshot = self._create_button(
            "Snapshot",
            style.standardIcon(QtWidgets.QStyle.SP_DialogSaveButton),
            on_save_snapshot,
            "Save the current viewport as a PNG.",
        )
        self.btn_reset = self._create_button(
            "Reset View",
            style.standardIcon(QtWidgets.QStyle.SP_BrowserReload),
            on_reset_view,
            "Reset camera to fit all visible data.",
        )
        self.btn_help = self._create_button(
            "Help",
            style.standardIcon(QtWidgets.QStyle.SP_MessageBoxQuestion),
            on_show_help,
            "Show keyboard and mouse interaction help.",
        )

        grid.addWidget(self.btn_load, 0, 0)
        grid.addWidget(self.btn_reload, 0, 1)
        grid.addWidget(self.btn_points, 1, 0)
        grid.addWidget(self.btn_mesh, 1, 1)
        grid.addWidget(self.btn_snapshot, 2, 0)
        grid.addWidget(self.btn_reset, 2, 1)
        grid.addWidget(self.btn_help, 3, 0, 1, 2)

    def _create_button(
        self, text: str, icon: QtGui.QIcon, slot: Callable[[], None], tool_tip: str
    ) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(text)
        btn.setIcon(icon)
        btn.setToolTip(tool_tip)
        btn.clicked.connect(slot)
        return btn


class VolumeDisplayControlsWidget(QtWidgets.QGroupBox):
    """Controls for volume opacity, shading, and visibility."""

    def __init__(
        self,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
        on_opacity_changed: Callable[[int], None],
        on_shading_toggled: Callable[[int], None],
        on_visibility_toggled: Callable[[int], None],
        on_intensity_changed: Callable[[int], None],
    ) -> None:
        super().__init__("Volume Display", parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # Opacity
        layout.addWidget(QtWidgets.QLabel("Opacity Scale:"))
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(0, 200)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(on_opacity_changed)
        layout.addWidget(self.opacity_slider)

        # Intensity
        layout.addWidget(QtWidgets.QLabel("Light Intensity:"))
        self.intensity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.intensity_slider.setRange(0, 200)
        self.intensity_slider.setValue(100)
        self.intensity_slider.valueChanged.connect(on_intensity_changed)
        layout.addWidget(self.intensity_slider)

        # Checkboxes
        self.check_shading = QtWidgets.QCheckBox("Enable Shading")
        self.check_shading.setChecked(True)
        self.check_shading.stateChanged.connect(on_shading_toggled)
        layout.addWidget(self.check_shading)

        self.check_visible = QtWidgets.QCheckBox("Volume Visible")
        self.check_visible.setChecked(True)
        self.check_visible.stateChanged.connect(on_visibility_toggled)
        layout.addWidget(self.check_visible)


class VolumeSliceControlsWidget(QtWidgets.QGroupBox):
    """Controls for axial/coronal/sagittal slices."""

    def __init__(
        self,
        *,
        parent: Optional[QtWidgets.QWidget] = None,
        on_plane_toggled: Callable[[int, int], None],
        on_plane_moved: Callable[[int, int], None],
    ) -> None:
        super().__init__("Slicing & Clipping", parent)
        self.on_plane_toggled = on_plane_toggled
        self.on_plane_moved = on_plane_moved

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.planes: dict[int, dict[str, Any]] = {}

        for axis, name in [(0, "Axial (Z)"), (1, "Coronal (Y)"), (2, "Sagittal (X)")]:
            plane_layout = QtWidgets.QVBoxLayout()
            plane_layout.setSpacing(2)

            check = QtWidgets.QCheckBox(name)
            check.stateChanged.connect(
                lambda state, a=axis: self.on_plane_toggled(a, state)
            )

            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 100)
            slider.setEnabled(False)
            slider.valueChanged.connect(lambda val, a=axis: self.on_plane_moved(a, val))

            label = QtWidgets.QLabel("Slice: -")

            plane_layout.addWidget(check)
            plane_layout.addWidget(slider)
            plane_layout.addWidget(label)
            layout.addLayout(plane_layout)

            self.planes[axis] = {"check": check, "slider": slider, "label": label}

    def update_plane_range(self, axis: int, max_val: int, current_val: int):
        if axis in self.planes:
            self.planes[axis]["slider"].setRange(0, max_val)
            self.planes[axis]["slider"].setValue(current_val)
            self.planes[axis]["label"].setText(f"Slice: {current_val}")

    def set_plane_enabled(self, axis: int, enabled: bool):
        if axis in self.planes:
            self.planes[axis]["slider"].setEnabled(enabled)

    def set_plane_value(self, axis: int, value: int):
        if axis in self.planes:
            self.planes[axis]["slider"].setValue(value)
            self.planes[axis]["label"].setText(f"Slice: {value}")


class VolumeControlsGroup(QtWidgets.QWidget):
    """Container for all 3D volume controls."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None, **kwargs) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        quick_action_keys = (
            "on_load_volume",
            "on_reload_volume",
            "on_load_points",
            "on_load_mesh",
            "on_save_snapshot",
            "on_reset_view",
            "on_show_help",
        )
        display_keys = (
            "on_opacity_changed",
            "on_shading_toggled",
            "on_visibility_toggled",
            "on_intensity_changed",
        )
        slice_keys = (
            "on_plane_toggled",
            "on_plane_moved",
        )

        quick_kwargs = {k: kwargs[k] for k in quick_action_keys if k in kwargs}
        display_kwargs = {k: kwargs[k] for k in display_keys if k in kwargs}
        slice_kwargs = {k: kwargs[k] for k in slice_keys if k in kwargs}

        self.quick_actions = VolumeQuickActionsWidget(parent=self, **quick_kwargs)
        self.display = VolumeDisplayControlsWidget(parent=self, **display_kwargs)
        self.slices = VolumeSliceControlsWidget(parent=self, **slice_kwargs)

        layout.addWidget(self.quick_actions)
        layout.addWidget(self.display)
        layout.addWidget(self.slices)
        layout.addStretch()


class SliceViewerControls(QtWidgets.QWidget):
    """Dedicated controls for the 2D slice viewer mode."""

    def __init__(
        self,
        parent: Optional[QtWidgets.QWidget] = None,
        on_index_changed: Optional[Callable[[int], None]] = None,
        on_axis_changed: Optional[Callable[[int], None]] = None,
        on_window_changed: Optional[Callable[[], None]] = None,
        on_gamma_changed: Optional[Callable[[int], None]] = None,
        on_auto_window: Optional[Callable[[], None]] = None,
    ) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(10)

        # Navigation
        nav_group = QtWidgets.QGroupBox("Slice Navigation")
        nav_layout = QtWidgets.QVBoxLayout(nav_group)

        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(["Axial (Z)", "Coronal (Y)", "Sagittal (X)"])
        if on_axis_changed:
            self.axis_combo.currentIndexChanged.connect(on_axis_changed)
        nav_layout.addWidget(QtWidgets.QLabel("View Axis:"))
        nav_layout.addWidget(self.axis_combo)

        self.slice_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        if on_index_changed:
            self.slice_slider.valueChanged.connect(on_index_changed)
        nav_layout.addWidget(QtWidgets.QLabel("Slice Index:"))
        nav_layout.addWidget(self.slice_slider)

        self.status_label = QtWidgets.QLabel("Slice: -/-")
        nav_layout.addWidget(self.status_label)
        layout.addWidget(nav_group)

        # Intensity
        intensity_group = QtWidgets.QGroupBox("Intensity Adjustment")
        int_layout = QtWidgets.QGridLayout(intensity_group)

        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin = QtWidgets.QDoubleSpinBox()
        for s in (self.min_spin, self.max_spin):
            s.setRange(-1e6, 1e6)
            if on_window_changed:
                s.valueChanged.connect(on_window_changed)

        int_layout.addWidget(QtWidgets.QLabel("Min:"), 0, 0)
        int_layout.addWidget(self.min_spin, 0, 1)
        int_layout.addWidget(QtWidgets.QLabel("Max:"), 1, 0)
        int_layout.addWidget(self.max_spin, 1, 1)

        self.auto_btn = QtWidgets.QPushButton("Auto Window")
        if on_auto_window:
            self.auto_btn.clicked.connect(on_auto_window)
        int_layout.addWidget(self.auto_btn, 2, 0, 1, 2)

        layout.addWidget(intensity_group)

        # Color/Gamma
        gamma_group = QtWidgets.QGroupBox("Color & Gamma")
        gamma_layout = QtWidgets.QVBoxLayout(gamma_group)

        self.gamma_label = QtWidgets.QLabel("Gamma: 1.00")
        self.gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.gamma_slider.setRange(10, 500)  # 0.1 to 5.0
        self.gamma_slider.setValue(100)
        if on_gamma_changed:
            self.gamma_slider.valueChanged.connect(on_gamma_changed)

        gamma_layout.addWidget(self.gamma_label)
        gamma_layout.addWidget(self.gamma_slider)
        layout.addWidget(gamma_group)

        layout.addStretch()
