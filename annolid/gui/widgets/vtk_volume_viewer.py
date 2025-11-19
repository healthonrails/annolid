from __future__ import annotations
from annolid.utils.logger import logger

import os
import re
import tempfile
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple
from functools import partial

import numpy as np
import pandas as pd
from PIL import Image
from qtpy import QtCore, QtWidgets, QtGui

# VTK imports (modular) — if these fail, the caller should fall back
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
    vtkTexture,
    vtkPointPicker,
    vtkColorTransferFunction,
    vtkPolyDataMapper,
    vtkActor,
    vtkProperty,
    vtkImageActor,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkCommonDataModel import (
    vtkImageData,
    vtkPolyData,
    vtkCellArray,
    vtkPiecewiseFunction,
    vtkPlane,
)
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonCore import vtkStringArray
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleUser,
)
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkIOImage import vtkPNGWriter, vtkImageReader2Factory
try:
    from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLPolyDataMapper
except Exception:  # pragma: no cover - optional renderer
    vtkOpenGLPolyDataMapper = None
try:  # Prefer GDCM when available (more robust series handling)
    from vtkmodules.vtkIOImage import vtkGDCMImageReader  # type: ignore
    _HAS_GDCM = True
except Exception:  # pragma: no cover
    _HAS_GDCM = False
try:
    from vtkmodules.vtkInteractionWidgets import vtkImagePlaneWidget
except Exception:
    try:
        from vtkmodules.vtkInteractionImage import vtkImagePlaneWidget
    except Exception:  # pragma: no cover
        vtkImagePlaneWidget = None
_HAS_PLANE_WIDGET = vtkImagePlaneWidget is not None


POINT_CLOUD_EXTS = (".ply", ".csv", ".xyz")
MESH_EXTS = (".stl", ".obj")
VOLUME_FILE_EXTS = (".tif", ".tiff", ".nii", ".nii.gz")
DICOM_EXTS = (".dcm", ".dicom", ".ima")
TIFF_SUFFIXES = (".tif", ".tiff")
OME_TIFF_SUFFIXES = (".ome.tif", ".ome.tiff")
_auto_ooc_raw = os.environ.get(
    "ANNOLID_VTK_OUT_OF_CORE_THRESHOLD_MB", "2048") or ""
try:
    AUTO_OUT_OF_CORE_MB = float(
        _auto_ooc_raw.strip()) if _auto_ooc_raw else 0.0
except Exception:
    AUTO_OUT_OF_CORE_MB = 0.0
_max_voxels_raw = os.environ.get(
    "ANNOLID_VTK_MAX_VOLUME_VOXELS", "134217728") or ""
try:
    MAX_VOLUME_VOXELS = int(float(_max_voxels_raw.strip()))
except Exception:
    MAX_VOLUME_VOXELS = 134217728
_slice_mode_bytes_raw = os.environ.get(
    "ANNOLID_VTK_SLICE_MODE_BYTES", "68719476736"
) or ""
try:
    SLICE_MODE_BYTES = float(_slice_mode_bytes_raw.strip())
except Exception:
    SLICE_MODE_BYTES = 68719476736.0

PLANE_DEFS = (
    (0, "Axial (Z)", "SetPlaneOrientationToZAxes"),
    (1, "Coronal (Y)", "SetPlaneOrientationToYAxes"),
    (2, "Sagittal (X)", "SetPlaneOrientationToXAxes"),
)
PLANE_COLORS: dict[int, tuple[float, float, float]] = {
    0: (0.9, 0.2, 0.2),
    1: (0.2, 0.9, 0.2),
    2: (0.2, 0.2, 0.9),
}


@dataclass
class _RegionSelectionEntry:
    item: QtWidgets.QListWidgetItem
    checkbox: QtWidgets.QCheckBox
    color_button: QtWidgets.QToolButton
    display_text: str


@dataclass
class _SlicePlaneControl:
    axis: int
    name: str
    slider: QtWidgets.QSlider
    checkbox: QtWidgets.QCheckBox
    label: QtWidgets.QLabel


@dataclass
class _VolumeData:
    array: Optional[np.ndarray]
    spacing: Optional[Tuple[float, float, float]]
    vmin: float
    vmax: float
    is_grayscale: bool = True
    is_out_of_core: bool = False
    backing_path: Optional[Path] = None
    vtk_image: Optional[vtkImageData] = None  # type: ignore[name-defined]
    slice_mode: bool = False
    slice_loader: Optional["_BaseSliceLoader"] = None
    slice_axis: int = 0
    volume_shape: Optional[tuple[int, int, int]] = None


class _BaseSliceLoader:
    """Abstract loader that can retrieve 2D slices on demand."""

    def total_slices(self) -> int:
        raise NotImplementedError

    def shape(self) -> tuple[int, int, int]:
        raise NotImplementedError

    def dtype(self) -> np.dtype:
        raise NotImplementedError

    def read_slice(self, axis: int, index: int) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return


class _MemmapSliceLoader(_BaseSliceLoader):
    def __init__(self, array: np.ndarray):
        self._array = array
        if self._array.ndim == 2:
            self._array = self._array[np.newaxis, ...]

    def total_slices(self) -> int:
        return int(self._array.shape[0])

    def shape(self) -> tuple[int, int, int]:
        shp = self._array.shape
        if len(shp) == 2:
            return (1, int(shp[0]), int(shp[1]))
        return (int(shp[0]), int(shp[1]), int(shp[2]))

    def dtype(self) -> np.dtype:
        return self._array.dtype

    def read_slice(self, axis: int, index: int) -> np.ndarray:
        if axis != 0:
            raise NotImplementedError(
                "Memmap loader currently supports axial slices only.")
        index = max(0, min(int(index), self.total_slices() - 1))
        return np.array(self._array[index], copy=True)

    def close(self) -> None:
        self._array = None  # type: ignore[assignment]


class _TiffSliceLoader(_BaseSliceLoader):
    def __init__(self, path: Path, shape: tuple[int, int, int], dtype: np.dtype):
        import tifffile  # local import to avoid hard dependency at module load

        self._path = str(path)
        self._tif = tifffile.TiffFile(self._path)
        self._shape = (int(shape[0]), int(shape[1]), int(shape[2]))
        self._dtype = np.dtype(dtype)

    def total_slices(self) -> int:
        return self._shape[0]

    def shape(self) -> tuple[int, int, int]:
        return self._shape

    def dtype(self) -> np.dtype:
        return self._dtype

    def read_slice(self, axis: int, index: int) -> np.ndarray:
        if axis != 0:
            raise NotImplementedError(
                "TIFF slice loader currently supports axial slices only.")
        idx = max(0, min(int(index), self.total_slices() - 1))
        return self._tif.pages[idx].asarray()

    def close(self) -> None:
        try:
            self._tif.close()
        except Exception:
            pass


class VTKVolumeViewerDialog(QtWidgets.QMainWindow):
    """
    True 3D volume renderer using VTK's GPU volume mapper.

    - Loads a TIFF stack into a 3D volume
    - Interact with mouse: rotate, zoom, pan
    - Simple UI controls for opacity scaling and shading toggle
    """

    def __init__(self, src_path: Optional[str | Path], parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("3D Volume Renderer (VTK)")
        self.resize(900, 700)
        self._source_path: Optional[Path] = None
        self._path = Path(".")
        if src_path:
            try:
                candidate = Path(src_path).expanduser()
            except Exception:
                candidate = Path(src_path)
            try:
                candidate = candidate.resolve()
            except Exception:
                pass
            self._source_path = candidate
            self._path = self._resolve_initial_source(candidate)

        # Volume state placeholders
        self._has_volume: bool = False
        self._volume_np: Optional[np.ndarray] = None
        self._vtk_img = None
        self._vmin = 0.0
        self._vmax = 1.0
        self._opacity_tf = None
        self._color_tf = None
        self._volume_shape: tuple[int, int, int] = (0, 0, 0)
        self._slice_plane_widgets: dict[int, vtkImagePlaneWidget] = {}
        self._slice_clipping_planes: dict[int, vtkPlane] = {}
        self._out_of_core_active = False
        self._out_of_core_backing_path: Optional[Path] = None
        self._out_of_core_array: Optional[np.memmap] = None
        self._slice_mode = False
        self._slice_loader: Optional[_BaseSliceLoader] = None
        self._slice_actor: Optional[vtkImageActor] = None
        self._slice_current_index = 0
        self._slice_axis = 0
        self._slice_vmin = 0.0
        self._slice_vmax = 1.0
        self._slice_gamma = 1.0
        self._slice_window_override = False
        self._slice_last_data_min = 0.0
        self._slice_last_data_max = 1.0

        central_widget = QtWidgets.QWidget(self)
        self.setCentralWidget(central_widget)
        central_layout = QtWidgets.QVBoxLayout(central_widget)
        central_layout.setContentsMargins(2, 2, 2, 2)
        central_layout.setSpacing(6)
        self.vtk_widget = QVTKRenderWindowInteractor(central_widget)
        central_layout.addWidget(self.vtk_widget, 1)

        # Controls panel (dockable)
        controls_panel = QtWidgets.QWidget()
        controls_panel.setMinimumWidth(180)
        controls_panel.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        controls_layout = QtWidgets.QVBoxLayout(controls_panel)
        controls_layout.setAlignment(QtCore.Qt.AlignTop)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)
        self._controls_dock = QtWidgets.QDockWidget("Controls", self)
        self._controls_dock.setWidget(controls_panel)
        self._controls_dock.setAllowedAreas(
            QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self._controls_dock.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetFloatable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self._controls_dock)
        self._apply_default_dock_width()

        self.volume_group = QtWidgets.QGroupBox("Volume Controls")
        volume_layout = QtWidgets.QGridLayout()
        self.volume_group.setLayout(volume_layout)

        # Blend mode
        volume_layout.addWidget(QtWidgets.QLabel("Blend:"), 0, 0)
        self.blend_combo = QtWidgets.QComboBox()
        self.blend_combo.addItems(
            ["Composite", "MIP-Max", "MIP-Min", "Additive"])
        self.blend_combo.currentIndexChanged.connect(self._update_blend_mode)
        volume_layout.addWidget(self.blend_combo, 0, 1)

        # Colormap
        volume_layout.addWidget(QtWidgets.QLabel("Colormap:"), 0, 2)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["Grayscale", "Invert Gray", "Hot"])
        self.cmap_combo.currentIndexChanged.connect(
            self._update_transfer_functions)
        volume_layout.addWidget(self.cmap_combo, 0, 3)

        # Intensity window (min/max)
        volume_layout.addWidget(QtWidgets.QLabel("Window:"), 1, 0)
        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setDecimals(3)
            spin.setKeyboardTracking(False)
        self.min_spin.valueChanged.connect(self._on_window_changed)
        self.max_spin.valueChanged.connect(self._on_window_changed)
        volume_layout.addWidget(self.min_spin, 1, 1)
        volume_layout.addWidget(self.max_spin, 1, 2)
        self.auto_window_btn = QtWidgets.QPushButton("Auto")
        self.auto_window_btn.clicked.connect(self._auto_window)
        volume_layout.addWidget(self.auto_window_btn, 1, 3)

        # Density (global opacity) and shading
        volume_layout.addWidget(QtWidgets.QLabel("Density:"), 2, 0)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(1, 100)
        self.opacity_slider.setValue(30)
        self.opacity_slider.valueChanged.connect(self._update_opacity)
        volume_layout.addWidget(self.opacity_slider, 2, 1, 1, 2)
        self.shade_checkbox = QtWidgets.QCheckBox("Shading")
        self.shade_checkbox.setChecked(True)
        self.shade_checkbox.stateChanged.connect(self._update_shading)
        volume_layout.addWidget(self.shade_checkbox, 2, 3)

        # Interpolation
        volume_layout.addWidget(QtWidgets.QLabel("Interpolation:"), 3, 0)
        self.interp_combo = QtWidgets.QComboBox()
        self.interp_combo.addItems(["Linear", "Nearest"])
        self.interp_combo.currentIndexChanged.connect(
            self._update_interpolation)
        volume_layout.addWidget(self.interp_combo, 3, 1)

        # Spacing (X, Y, Z)
        volume_layout.addWidget(QtWidgets.QLabel("Spacing X/Y/Z:"), 3, 2)
        self.spacing_x = QtWidgets.QDoubleSpinBox()
        self.spacing_y = QtWidgets.QDoubleSpinBox()
        self.spacing_z = QtWidgets.QDoubleSpinBox()
        for s in (self.spacing_x, self.spacing_y, self.spacing_z):
            s.setDecimals(3)
            s.setRange(0.001, 10000.0)
            s.setValue(1.0)
            s.valueChanged.connect(self._update_spacing)
        spacing_box = QtWidgets.QHBoxLayout()
        spacing_box.addWidget(self.spacing_x)
        spacing_box.addWidget(self.spacing_y)
        spacing_box.addWidget(self.spacing_z)
        spacing_widget = QtWidgets.QWidget()
        spacing_widget.setLayout(spacing_box)
        volume_layout.addWidget(spacing_widget, 3, 3)

        controls_layout.addWidget(self.volume_group)

        # Point cloud controls
        self.load_pc_btn = QtWidgets.QPushButton("Load Point Cloud…")
        self.load_pc_btn.clicked.connect(self._load_point_cloud_folder)
        self.clear_pc_btn = QtWidgets.QPushButton("Clear Points")
        self.clear_pc_btn.clicked.connect(self._clear_point_clouds)
        self.point_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.point_size_slider.setRange(1, 12)
        self.point_size_slider.setValue(3)
        self.point_size_slider.setToolTip("Point size")
        self.point_size_slider.valueChanged.connect(self._update_point_sizes)

        self.point_group = QtWidgets.QGroupBox("Point Cloud")
        point_layout = QtWidgets.QVBoxLayout()
        point_layout.setContentsMargins(4, 4, 4, 4)
        load_row = QtWidgets.QHBoxLayout()
        load_row.addWidget(self.load_pc_btn)
        load_row.addStretch(1)
        point_layout.addLayout(load_row)

        self.point_detail_widget = QtWidgets.QWidget()
        point_detail_layout = QtWidgets.QVBoxLayout(self.point_detail_widget)
        point_detail_layout.setContentsMargins(0, 0, 0, 0)
        point_detail_layout.setSpacing(6)
        clear_row = QtWidgets.QHBoxLayout()
        clear_row.addWidget(self.clear_pc_btn)
        clear_row.addStretch(1)
        point_detail_layout.addLayout(clear_row)
        size_row = QtWidgets.QHBoxLayout()
        size_row.addWidget(QtWidgets.QLabel("Point Size:"))
        size_row.addWidget(self.point_size_slider)
        size_row.addStretch(1)
        point_detail_layout.addLayout(size_row)

        self.region_group = QtWidgets.QGroupBox("Point Cloud Regions")
        region_layout = QtWidgets.QVBoxLayout()
        region_layout.setSpacing(4)
        region_btn_row = QtWidgets.QHBoxLayout()
        self.select_all_regions_btn = QtWidgets.QPushButton("Select All")
        self.deselect_all_regions_btn = QtWidgets.QPushButton("Deselect All")
        region_btn_row.addWidget(self.select_all_regions_btn)
        region_btn_row.addWidget(self.deselect_all_regions_btn)
        region_btn_row.addStretch(1)
        region_layout.addLayout(region_btn_row)
        self.select_all_regions_btn.clicked.connect(
            lambda: self._set_region_check_states(True))
        self.deselect_all_regions_btn.clicked.connect(
            lambda: self._set_region_check_states(False))
        self.region_search = QtWidgets.QLineEdit()
        self.region_search.setPlaceholderText("Filter regions…")
        self.region_search.textChanged.connect(self._filter_region_items)
        region_layout.addWidget(self.region_search)
        self.region_list_widget = QtWidgets.QListWidget()
        self.region_list_widget.setSelectionMode(
            QtWidgets.QAbstractItemView.NoSelection)
        self.region_list_widget.setFocusPolicy(QtCore.Qt.NoFocus)
        region_layout.addWidget(self.region_list_widget)
        self.region_group.setLayout(region_layout)
        self.region_group.setEnabled(False)
        point_detail_layout.addWidget(self.region_group)
        self.point_detail_widget.setVisible(False)
        point_layout.addWidget(self.point_detail_widget)
        self.point_group.setLayout(point_layout)
        controls_layout.addWidget(self.point_group)

        # Mesh controls
        self.load_mesh_btn = QtWidgets.QPushButton("Load Mesh…")
        self.load_mesh_btn.clicked.connect(self._load_mesh_dialog)
        self.clear_mesh_btn = QtWidgets.QPushButton("Clear Meshes")
        self.clear_mesh_btn.clicked.connect(self._clear_meshes)
        self.load_diffuse_tex_btn = QtWidgets.QPushButton(
            "Load Diffuse Texture…")
        self.load_diffuse_tex_btn.clicked.connect(self._load_diffuse_texture)
        self.load_normal_tex_btn = QtWidgets.QPushButton("Load Normal Map…")
        self.load_normal_tex_btn.clicked.connect(self._load_normal_texture)

        self.mesh_group = QtWidgets.QGroupBox("Mesh")
        mesh_layout = QtWidgets.QVBoxLayout()
        mesh_layout.setContentsMargins(4, 4, 4, 4)
        mesh_load_row = QtWidgets.QHBoxLayout()
        mesh_load_row.addWidget(self.load_mesh_btn)
        mesh_load_row.addStretch(1)
        mesh_layout.addLayout(mesh_load_row)
        self.mesh_detail_widget = QtWidgets.QWidget()
        mesh_detail_layout = QtWidgets.QVBoxLayout(self.mesh_detail_widget)
        mesh_detail_layout.setContentsMargins(0, 0, 0, 0)
        mesh_detail_layout.setSpacing(6)
        mesh_clear_row = QtWidgets.QHBoxLayout()
        mesh_clear_row.addWidget(self.clear_mesh_btn)
        mesh_clear_row.addStretch(1)
        mesh_detail_layout.addLayout(mesh_clear_row)
        texture_group = QtWidgets.QGroupBox("Textures")
        texture_layout = QtWidgets.QHBoxLayout()
        texture_layout.setContentsMargins(4, 2, 4, 2)
        texture_layout.addWidget(self.load_diffuse_tex_btn)
        texture_layout.addWidget(self.load_normal_tex_btn)
        texture_group.setLayout(texture_layout)
        mesh_detail_layout.addWidget(texture_group)
        mesh_detail_layout.addStretch(1)
        self.mesh_detail_widget.setLayout(mesh_detail_layout)
        self.mesh_detail_widget.setVisible(False)
        mesh_layout.addWidget(self.mesh_detail_widget)
        self.mesh_group.setLayout(mesh_layout)
        controls_layout.addWidget(self.mesh_group)

        self.reset_cam_btn = QtWidgets.QPushButton("Reset Camera")
        self.reset_cam_btn.clicked.connect(self._reset_camera)
        self.reload_volume_btn = QtWidgets.QPushButton("Reload Volume")
        self.reload_volume_btn.setToolTip(
            "Re-read the active volume from disk. Use this after toggling the "
            "out-of-core TIFF option."
        )
        self.reload_volume_btn.clicked.connect(self._reload_volume)
        self.snapshot_btn = QtWidgets.QPushButton("Save Snapshot…")
        self.snapshot_btn.clicked.connect(self._save_snapshot)
        self.wl_mode_checkbox = QtWidgets.QCheckBox("Window/Level Mode")
        self.wl_mode_checkbox.setToolTip(
            "Enable to adjust intensity window by left-drag; camera interaction is paused"
        )
        self.wl_mode_checkbox.stateChanged.connect(self._toggle_wl_mode)

        self.general_group = QtWidgets.QGroupBox("General")
        general_layout = QtWidgets.QVBoxLayout()
        wl_layout = QtWidgets.QHBoxLayout()
        wl_layout.addWidget(self.wl_mode_checkbox)
        wl_layout.addStretch(1)
        general_layout.addLayout(wl_layout)
        general_buttons_layout = QtWidgets.QHBoxLayout()
        general_buttons_layout.addWidget(self.reset_cam_btn)
        general_buttons_layout.addWidget(self.reload_volume_btn)
        general_buttons_layout.addWidget(self.snapshot_btn)
        general_buttons_layout.addStretch(1)
        general_layout.addLayout(general_buttons_layout)
        self.general_group.setLayout(general_layout)
        controls_layout.addWidget(self.general_group)

        self.slice_view_group = QtWidgets.QGroupBox("Slice Viewer")
        slice_view_layout = QtWidgets.QVBoxLayout()
        slice_view_layout.setContentsMargins(4, 4, 4, 4)
        slice_view_layout.setSpacing(4)
        self.slice_hint_label = QtWidgets.QLabel(
            "Large TIFF detected. Showing on-demand axial slices.")
        self.slice_hint_label.setWordWrap(True)
        slice_view_layout.addWidget(self.slice_hint_label)
        self.slice_status_label = QtWidgets.QLabel("Slice: -/-")
        slice_view_layout.addWidget(self.slice_status_label)
        contrast_row = QtWidgets.QHBoxLayout()
        contrast_row.addWidget(QtWidgets.QLabel("Window min/max:"))
        self.slice_min_spin = QtWidgets.QDoubleSpinBox()
        self.slice_max_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.slice_min_spin, self.slice_max_spin):
            spin.setDecimals(3)
            spin.setRange(-1e9, 1e9)
            spin.setKeyboardTracking(False)
        self.slice_min_spin.valueChanged.connect(
            lambda _: self._on_slice_window_changed())
        self.slice_max_spin.valueChanged.connect(
            lambda _: self._on_slice_window_changed())
        contrast_row.addWidget(self.slice_min_spin)
        contrast_row.addWidget(self.slice_max_spin)
        self.slice_auto_btn = QtWidgets.QPushButton("Auto")
        self.slice_auto_btn.clicked.connect(self._slice_auto_window)
        contrast_row.addWidget(self.slice_auto_btn)
        slice_view_layout.addLayout(contrast_row)
        gamma_row = QtWidgets.QHBoxLayout()
        self.slice_gamma_label = QtWidgets.QLabel("Gamma: 1.00")
        self.slice_gamma_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_gamma_slider.setRange(10, 400)  # 0.1 - 4.0
        self.slice_gamma_slider.setValue(100)
        self.slice_gamma_slider.valueChanged.connect(
            self._on_slice_gamma_changed)
        gamma_row.addWidget(self.slice_gamma_label)
        gamma_row.addWidget(self.slice_gamma_slider)
        slice_view_layout.addLayout(gamma_row)
        self.slice_index_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slice_index_slider.setRange(0, 0)
        self.slice_index_slider.setEnabled(False)
        self.slice_index_slider.valueChanged.connect(
            self._on_slice_index_changed)
        slice_view_layout.addWidget(self.slice_index_slider)
        self.slice_view_group.setLayout(slice_view_layout)
        self.slice_view_group.setVisible(False)
        controls_layout.addWidget(self.slice_view_group)

        self.slice_group = QtWidgets.QGroupBox("Slice Planes")
        slice_layout = QtWidgets.QVBoxLayout()
        slice_layout.setContentsMargins(4, 4, 4, 4)
        slice_layout.setSpacing(4)
        self._slice_plane_controls: dict[int, _SlicePlaneControl] = {}
        for axis, name, _ in PLANE_DEFS:
            row = QtWidgets.QHBoxLayout()
            label = QtWidgets.QLabel(f"{name}: -/-")
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setRange(0, 0)
            slider.setEnabled(False)
            slider.valueChanged.connect(
                partial(self._on_plane_slider_changed, axis))
            checkbox = QtWidgets.QCheckBox("Show")
            checkbox.setEnabled(False)
            checkbox.setToolTip(f"Show the {name} slice plane.")
            checkbox.stateChanged.connect(
                partial(self._on_plane_checkbox_changed, axis))
            row.addWidget(label)
            row.addWidget(slider, 2)
            row.addWidget(checkbox)
            slice_layout.addLayout(row)
            self._slice_plane_controls[axis] = _SlicePlaneControl(
                axis=axis,
                name=name,
                slider=slider,
                checkbox=checkbox,
                label=label,
            )
        self.slice_group.setLayout(slice_layout)
        controls_layout.addWidget(self.slice_group)
        self._configure_plane_controls()

        status_row = QtWidgets.QHBoxLayout()
        self.volume_io_label = QtWidgets.QLabel("Volume I/O: in-memory")
        status_row.addWidget(self.volume_io_label)
        self.mesh_status_label = QtWidgets.QLabel("Mesh: none loaded")
        status_row.addWidget(self.mesh_status_label)
        status_row.addStretch(1)
        controls_layout.addLayout(status_row)
        controls_layout.addStretch(1)

        # Build pipeline
        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)

        self.interactor = self.vtk_widget.GetRenderWindow().GetInteractor()
        # Camera interaction style (rotate/pan/zoom)
        self._style_trackball = vtkInteractorStyleTrackballCamera()
        self._style_inactive = vtkInteractorStyleUser()
        self.interactor.SetInteractorStyle(self._style_trackball)
        # Create a volume and add it
        self.volume = vtkVolume()
        self.mapper = vtkSmartVolumeMapper()
        self.property = vtkVolumeProperty()
        self.property.ShadeOn()
        self.property.SetInterpolationTypeToLinear()

        # Conditionally load volume if path looks like a volume source
        _loaded_volume = False
        if src_path and self._is_volume_candidate(self._path):
            try:
                _loaded_volume = self._load_volume()
            except Exception:
                _loaded_volume = False

        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.property)
        # Only add the volume actor if we actually loaded a volume
        if _loaded_volume:
            self.renderer.AddVolume(self.volume)
            self.renderer.ResetCamera()
        self.renderer.SetBackground(0.1, 0.1, 0.12)

        # Setup for interactive point picking
        self._picker = vtkPointPicker()
        self._picker.SetTolerance(0.005)  # Adjust sensitivity
        self._last_picked_id = -1
        self._last_picked_actor = None

        # Setup interactive window/level mode and key/mouse bindings
        self._wl_mode = False
        self._wl_drag = False
        self._wl_last = (0, 0)
        self._install_interaction_bindings()

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self._point_actors: list[vtkActor] = []
        self._region_actors: dict[str, vtkActor] = {}
        self._region_entries: dict[str, _RegionSelectionEntry] = {}
        self._region_colors: dict[str, QtGui.QColor] = {}
        self._mesh_actors: list[vtkActor] = []
        self._mesh_textures: dict[int, dict[str, dict[str, object]]] = {}
        self._mesh_actor_names: dict[int, str] = {}
        self._active_mesh_actor: Optional[vtkActor] = None

        # If the provided path is a point cloud, add it now
        if src_path and not _loaded_volume and self._is_point_cloud_candidate(self._path):
            try:
                ext = self._path.suffix.lower()
                if ext == '.ply':
                    self._add_point_cloud_ply(str(self._path))
                elif ext in ('.csv', '.xyz'):
                    self._add_point_cloud_csv_or_xyz(str(self._path))
                self._update_point_sizes()
                self.renderer.ResetCamera()
                self.vtk_widget.GetRenderWindow().Render()
            except Exception:
                pass
        elif src_path and not _loaded_volume and self._is_mesh_candidate(self._path):
            try:
                self._load_mesh_file(str(self._path))
                self.renderer.ResetCamera()
                self.vtk_widget.GetRenderWindow().Render()
            except Exception:
                pass

        # Enable/disable volume-related controls based on whether a volume was loaded
        self._set_volume_controls_enabled(_loaded_volume)
        self._update_mesh_status_label()

    def setModal(self, modal: bool):
        """QMainWindow cannot be modal; keep compatibility with QDialog API."""
        return

    def _apply_default_dock_width(self):
        if not hasattr(self, "_controls_dock") or not self._controls_dock:
            return
        try:
            total_width = max(1, self.width())
            preferred = max(160, min(260, int(total_width * 0.25)))
            self.resizeDocks([self._controls_dock], [
                             preferred], QtCore.Qt.Horizontal)
        except Exception:
            pass

    def _install_interaction_bindings(self):
        # Mouse + key handlers
        self.interactor.AddObserver(
            "LeftButtonPressEvent", self._vtk_on_left_press)
        self.interactor.AddObserver(
            "LeftButtonReleaseEvent", self._vtk_on_left_release)
        self.interactor.AddObserver("MouseMoveEvent", self._vtk_on_mouse_move)
        self.interactor.AddObserver("KeyPressEvent", self._vtk_on_key_press)

    def _set_volume_controls_enabled(self, enabled: bool):
        widgets = [
            self.min_spin,
            self.max_spin,
            self.auto_window_btn,
            self.shade_checkbox,
            self.opacity_slider,
            self.interp_combo,
            self.spacing_x,
            self.spacing_y,
            self.spacing_z,
            self.blend_combo,
            self.cmap_combo,
            self.wl_mode_checkbox,
        ]
        for w in widgets:
            try:
                w.setEnabled(enabled)
            except Exception:
                pass
        self.volume_group.setVisible(enabled)
        self.wl_mode_checkbox.setVisible(enabled)
        self.slice_group.setVisible(enabled)
        for axis, control in self._slice_plane_controls.items():
            available = self._plane_is_available(axis)
            control.slider.setEnabled(enabled and available)
            control.checkbox.setEnabled(enabled and available)
        if not enabled:
            self._clear_slice_clipping_planes()
        if hasattr(self, "slice_group"):
            self.slice_group.setVisible(enabled)
        if hasattr(self, "slice_view_group") and enabled and not self._slice_mode:
            self.slice_view_group.setVisible(False)

    def _reload_volume(self):
        if not self._is_volume_candidate(getattr(self, "_path", Path("."))):
            QtWidgets.QMessageBox.information(
                self,
                "Reload Volume",
                "No volume source is associated with this viewer.",
            )
            return
        try:
            self._teardown_slice_planes()
            self._teardown_slice_mode()
            self._clear_slice_clipping_planes()
            self._volume_shape = (0, 0, 0)
            self._volume_np = None
            self._vtk_img = None
            self._has_volume = False
            loaded = self._load_volume()
            if loaded:
                self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Reload Volume",
                f"Failed to reload volume:\n{exc}",
            )

    def _load_volume(self) -> bool:
        old_array = self._out_of_core_array
        old_path = self._out_of_core_backing_path
        self._out_of_core_array = None
        self._out_of_core_backing_path = None
        volume_data = self._read_volume_any()
        if volume_data.slice_mode:
            self._init_slice_mode(volume_data)
            self._cleanup_out_of_core_backing(old_array, old_path)
            return False
        volume = volume_data.array
        if volume is None:
            raise RuntimeError("Volume source returned no data.")
        spacing = volume_data.spacing
        # Convert color stack to luminance for volume rendering if needed
        if (
            not volume_data.is_grayscale
            and volume.ndim == 4
            and volume.shape[-1] in (3, 4)
        ):
            volume = np.dot(
                volume[..., :3],
                [0.299, 0.587, 0.114],
            ).astype(volume.dtype)
        if not volume.flags.c_contiguous:
            volume = np.ascontiguousarray(volume)
        z, y, x = volume.shape
        self._volume_shape = (int(z), int(y), int(x))

        vtk_img = vtkImageData()
        vtk_img.SetDimensions(int(x), int(y), int(z))
        # Apply spacing from source if available
        if spacing is not None and len(spacing) == 3:
            vtk_img.SetSpacing(float(spacing[0]), float(
                spacing[1]), float(spacing[2]))
        else:
            vtk_img.SetSpacing(1.0, 1.0, 1.0)
        vtk_img.SetOrigin(0.0, 0.0, 0.0)

        # Map numpy dtype to VTK scalars
        vtk_array = numpy_to_vtk(
            num_array=volume.reshape(-1),
            deep=not volume_data.is_out_of_core,
        )
        vtk_img.GetPointData().SetScalars(vtk_array)

        # Keep handles and stats
        self._vtk_img = vtk_img
        self._volume_np = volume
        self._out_of_core_active = volume_data.is_out_of_core
        if volume_data.is_out_of_core:
            self._out_of_core_array = volume  # keep memmap alive
            self._out_of_core_backing_path = volume_data.backing_path
        self._vmin = volume_data.vmin
        self._vmax = volume_data.vmax
        self._update_volume_io_label(self._out_of_core_active)

        # Initialize window controls
        for spin in (self.min_spin, self.max_spin):
            spin.blockSignals(True)
        self.min_spin.setRange(self._vmin, self._vmax)
        self.max_spin.setRange(self._vmin, self._vmax)
        self.min_spin.setValue(self._vmin)
        self.max_spin.setValue(self._vmax)
        for spin in (self.min_spin, self.max_spin):
            spin.blockSignals(False)

        # Create initial transfer functions
        self._opacity_tf = vtkPiecewiseFunction()
        self._color_tf = vtkColorTransferFunction()
        self._update_transfer_functions()

        self.property.SetScalarOpacity(self._opacity_tf)
        self.property.SetColor(self._color_tf)
        self.property.SetAmbient(0.1)
        self.property.SetDiffuse(0.9)
        self.property.SetSpecular(0.2)

        self.mapper.SetInputData(vtk_img)
        self._clear_slice_clipping_planes()
        self._setup_slice_plane()
        self._configure_plane_controls()
        # If spacing provided, reflect it in UI
        if spacing is not None and len(spacing) == 3:
            try:
                self.spacing_x.blockSignals(True)
                self.spacing_y.blockSignals(True)
                self.spacing_z.blockSignals(True)
                self.spacing_x.setValue(float(spacing[0]))
                self.spacing_y.setValue(float(spacing[1]))
                self.spacing_z.setValue(float(spacing[2]))
            finally:
                self.spacing_x.blockSignals(False)
                self.spacing_y.blockSignals(False)
                self.spacing_z.blockSignals(False)

        self._update_opacity()
        self._update_shading()
        self._update_blend_mode()
        self._has_volume = True
        self._cleanup_out_of_core_backing(old_array, old_path)
        return True

    def _init_slice_mode(self, volume_data: _VolumeData):
        self._teardown_slice_mode()
        if volume_data.slice_loader is None:
            raise RuntimeError(
                "Slice loader is not available for this volume.")
        self._slice_mode = True
        self._slice_loader = volume_data.slice_loader
        self._slice_axis = volume_data.slice_axis or 0
        self._slice_vmin = volume_data.vmin
        self._slice_vmax = volume_data.vmax
        self._slice_window_override = False
        if hasattr(self, "volume"):
            try:
                self.renderer.RemoveVolume(self.volume)
            except Exception:
                pass
        self._set_volume_controls_enabled(False)
        if hasattr(self, "slice_group"):
            self.slice_group.setVisible(False)
        self.slice_view_group.setVisible(True)
        blank = vtkImageData()
        blank.SetDimensions(1, 1, 1)
        blank.SetSpacing(1.0, 1.0, 1.0)
        blank.SetOrigin(0.0, 0.0, 0.0)
        vtk_blank = numpy_to_vtk(np.zeros(1, dtype=np.float32), deep=True)
        blank.GetPointData().SetScalars(vtk_blank)
        try:
            self.mapper.SetInputData(blank)
        except Exception:
            pass
        try:
            self.renderer.RemoveVolume(self.volume)
        except Exception:
            pass
        try:
            self.volume.VisibilityOff()
        except Exception:
            pass
        total = self._slice_loader.total_slices() if self._slice_loader else 0
        self.slice_index_slider.blockSignals(True)
        self.slice_index_slider.setRange(0, max(0, total - 1))
        self.slice_index_slider.setValue(0)
        self.slice_index_slider.setEnabled(total > 1)
        self.slice_index_slider.blockSignals(False)
        self._update_slice_status_label(0, total)
        self._slice_gamma = 1.0
        self.slice_gamma_slider.blockSignals(True)
        self.slice_gamma_slider.setValue(100)
        self.slice_gamma_slider.blockSignals(False)
        self._update_slice_gamma_label()
        self._configure_slice_window_controls()
        if self._slice_actor is None:
            self._slice_actor = vtkImageActor()
        try:
            self.renderer.RemoveActor(self._slice_actor)
        except Exception:
            pass
        self.renderer.AddActor(self._slice_actor)
        self._volume_shape = volume_data.volume_shape or (total, 0, 0)
        self._load_slice_image(0)
        self._update_volume_io_label(True)

    def _load_slice_image(self, index: int):
        if not self._slice_loader or self._slice_actor is None:
            return
        total = max(1, self._slice_loader.total_slices())
        sanitized = max(0, min(int(index), total - 1))
        slice_array = self._slice_loader.read_slice(
            self._slice_axis, sanitized)
        arr = np.asarray(slice_array)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            arr = self._convert_frame_to_plane(arr, arr.dtype)
        if arr.ndim != 2:
            arr = np.squeeze(arr)
            if arr.ndim != 2:
                arr = arr[..., 0]
        raw_min = float(np.min(arr))
        raw_max = float(np.max(arr))
        self._slice_last_data_min = raw_min
        self._slice_last_data_max = raw_max
        need_window_update = False
        if not self._slice_window_override:
            self._slice_vmin = raw_min
            self._slice_vmax = raw_max
            need_window_update = True
        display_min = self._slice_vmin
        display_max = self._slice_vmax
        if display_max <= display_min:
            display_max = display_min + 1e-3
            need_window_update = True
        if raw_max <= display_min:
            display_min = raw_min
            display_max = max(raw_min + 1e-3, raw_max)
            need_window_update = True
        elif raw_min >= display_max:
            display_min = raw_min
            display_max = max(raw_max, raw_min + 1e-3)
            need_window_update = True
        if need_window_update:
            self._slice_vmin = display_min
            self._slice_vmax = display_max
            self._configure_slice_window_controls()
        norm = arr.astype(np.float32)
        span = float(display_max - display_min)
        span = max(span, 1e-6)
        norm = (norm - display_min) / span
        norm = np.clip(norm, 0.0, 1.0)
        gamma = max(0.1, float(self._slice_gamma))
        if abs(gamma - 1.0) > 1e-6:
            norm = np.power(norm, gamma)
        h, w = norm.shape
        vtk_img = vtkImageData()
        vtk_img.SetDimensions(int(w), int(h), 1)
        vtk_img.SetSpacing(1.0, 1.0, 1.0)
        vtk_img.SetOrigin(0.0, 0.0, 0.0)
        vtk_array = numpy_to_vtk(norm.reshape(-1), deep=True)
        vtk_img.GetPointData().SetScalars(vtk_array)
        self._slice_actor.SetInputData(vtk_img)

        # FIX: tell VTK our intensities are in [0, 1]
        img_prop = self._slice_actor.GetProperty()
        img_prop.SetColorWindow(1.0)  # max - min of the displayed range
        img_prop.SetColorLevel(0.5)  # center of the displayed range

        self._slice_current_index = sanitized
        self._update_slice_status_label(sanitized, total)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _configure_slice_window_controls(self):
        if not hasattr(self, "slice_min_spin"):
            return
        vmin = float(self._slice_vmin)
        vmax = float(self._slice_vmax)
        if vmax <= vmin:
            vmax = vmin + 1e-3
        for spin in (self.slice_min_spin, self.slice_max_spin):
            spin.blockSignals(True)
            spin.setRange(vmin - 1e6, vmax + 1e6)
        self.slice_min_spin.setValue(vmin)
        self.slice_max_spin.setValue(vmax)
        for spin in (self.slice_min_spin, self.slice_max_spin):
            spin.blockSignals(False)
        self._slice_vmin = vmin
        self._slice_vmax = vmax

    def _on_slice_window_changed(self):
        if not self._slice_mode:
            return
        vmin = float(min(self.slice_min_spin.value(),
                         self.slice_max_spin.value()))
        vmax = float(max(self.slice_min_spin.value(),
                         self.slice_max_spin.value()))
        if vmax <= vmin:
            vmax = vmin + 1e-3
        self.slice_min_spin.blockSignals(True)
        self.slice_max_spin.blockSignals(True)
        self.slice_min_spin.setValue(vmin)
        self.slice_max_spin.setValue(vmax)
        self.slice_min_spin.blockSignals(False)
        self.slice_max_spin.blockSignals(False)
        self._slice_vmin = vmin
        self._slice_vmax = vmax
        self._slice_window_override = True
        self._load_slice_image(self._slice_current_index)

    def _slice_auto_window(self):
        if not self._slice_mode or not self._slice_loader:
            return
        idx = self._slice_current_index
        sample = self._slice_loader.read_slice(self._slice_axis, idx)
        arr = np.asarray(sample).astype(np.float32)
        if arr.size == 0:
            return
        p2 = float(np.percentile(arr, 2))
        p98 = float(np.percentile(arr, 98))
        if p98 <= p2:
            p2 = float(arr.min())
            p98 = float(arr.max())
        self._slice_vmin = p2
        self._slice_vmax = p98
        self._slice_window_override = True
        self._configure_slice_window_controls()
        self._load_slice_image(idx)

    def _on_slice_gamma_changed(self, value: int):
        self._slice_gamma = max(0.1, float(value) / 100.0)
        self._update_slice_gamma_label()
        if self._slice_mode:
            self._slice_window_override = True
            self._load_slice_image(self._slice_current_index)

    def _update_slice_gamma_label(self):
        if hasattr(self, "slice_gamma_label"):
            self.slice_gamma_label.setText(f"Gamma: {self._slice_gamma:.2f}")

    def _on_slice_index_changed(self, value: int):
        if not self._slice_mode:
            return
        self._load_slice_image(int(value))

    def _update_slice_status_label(self, index: int, total: int):
        if not hasattr(self, "slice_status_label"):
            return
        if total <= 0:
            self.slice_status_label.setText("Slice: -/-")
        else:
            self.slice_status_label.setText(f"Slice: {index + 1}/{total}")
        if total > 0:
            self.slice_hint_label.setText(
                "Large TIFF in slice mode. Adjust contrast or switch slices as needed."
            )

    def _teardown_slice_mode(self):
        self._slice_mode = False
        if self._slice_actor is not None:
            try:
                self.renderer.RemoveActor(self._slice_actor)
            except Exception:
                pass
            self._slice_actor = None
        self.slice_view_group.setVisible(False)
        self.slice_index_slider.blockSignals(True)
        self.slice_index_slider.setRange(0, 0)
        self.slice_index_slider.setValue(0)
        self.slice_index_slider.setEnabled(False)
        self.slice_index_slider.blockSignals(False)
        self._update_slice_status_label(0, 0)
        self._close_slice_loader()
        self._slice_vmin = 0.0
        self._slice_vmax = 1.0
        self._slice_gamma = 1.0
        self._update_slice_gamma_label()
        self._slice_window_override = False

    def _close_slice_loader(self):
        if self._slice_loader is None:
            return
        try:
            self._slice_loader.close()
        except Exception:
            pass
        self._slice_loader = None

    def _cleanup_out_of_core_backing(
        self,
        array: Optional[np.memmap] = None,
        path: Optional[Path] = None,
    ):
        if array is None and path is None:
            array = self._out_of_core_array
            path = self._out_of_core_backing_path
            self._out_of_core_array = None
            self._out_of_core_backing_path = None
        if array is not None:
            try:
                mmap_obj = getattr(array, "_mmap", None)
                if mmap_obj is not None:
                    mmap_obj.close()
            except Exception:
                pass
        if path:
            try:
                Path(path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except TypeError:
                try:
                    Path(path).unlink()
                except Exception:
                    pass
            except Exception:
                pass

    def _read_volume_any(self) -> _VolumeData:
        """Read a 3D volume from TIFF/NIfTI/DICOM or directory (DICOM series)."""
        path = self._path
        try:
            if path.is_dir():
                volume, spacing = self._read_dicom_series(path)
                return _VolumeData(
                    array=volume,
                    spacing=spacing,
                    vmin=float(volume.min()),
                    vmax=float(volume.max()),
                    is_grayscale=volume.ndim == 3,
                    is_out_of_core=False,
                    volume_shape=tuple(int(x) for x in volume.shape[:3]),
                )

            suffix = path.suffix.lower()
            name_lower = path.name.lower()
            if name_lower.endswith('.nii') or name_lower.endswith('.nii.gz'):
                # NIfTI via VTK reader
                try:
                    from vtkmodules.vtkIOImage import vtkNIFTIImageReader
                except Exception as exc:
                    raise RuntimeError(
                        "VTK NIFTI reader is not available in this build.") from exc
                reader = vtkNIFTIImageReader()
                reader.SetFileName(str(path))
                reader.Update()
                vtk_img = reader.GetOutput()
                vol = self._vtk_image_to_numpy(vtk_img)
                s = vtk_img.GetSpacing()
                spacing = (s[0], s[1], s[2])
                vol = self._normalize_to_float01(vol)
                return _VolumeData(
                    array=vol,
                    spacing=spacing,
                    vmin=float(vol.min()),
                    vmax=float(vol.max()),
                    is_grayscale=vol.ndim == 3,
                    is_out_of_core=False,
                    volume_shape=tuple(int(x) for x in vol.shape[:3]),
                )
            if suffix in ('.dcm', '.ima', '.dicom'):
                # Treat as a DICOM series from the containing folder
                volume, spacing = self._read_dicom_series(path.parent)
                return _VolumeData(
                    array=volume,
                    spacing=spacing,
                    vmin=float(volume.min()),
                    vmax=float(volume.max()),
                    is_grayscale=volume.ndim == 3,
                    is_out_of_core=False,
                    volume_shape=tuple(int(x) for x in volume.shape[:3]),
                )

            if self._is_tiff_candidate(path):
                preferred_out_of_core = self._should_use_out_of_core_tiff(path)
                if preferred_out_of_core:
                    return self._read_tiff_out_of_core(path)
                try:
                    return self._read_tiff_eager(path)
                except MemoryError as exc:
                    logger.warning(
                        "Standard TIFF loading failed (%s). Retrying with out-of-core caching.",
                        exc,
                    )
                    return self._read_tiff_out_of_core(path)

            raise RuntimeError(f"Unsupported volume format: {path}")
        except Exception as e:
            # Re-raise with context so upper layer shows a concise message
            raise RuntimeError(f"Failed to read volume from '{path}': {e}")

    def _is_tiff_candidate(self, path: Path) -> bool:
        name_lower = path.name.lower()
        suffix = path.suffix.lower()
        if suffix in TIFF_SUFFIXES:
            return True
        return any(name_lower.endswith(ext) for ext in OME_TIFF_SUFFIXES)

    def _should_use_out_of_core_tiff(self, path: Path) -> bool:
        size_bytes = self._safe_file_size(path)
        avail_bytes = self._available_memory_bytes()
        logger.debug(
            "VTK viewer source size=%s bytes, available RAM=%s bytes",
            size_bytes,
            avail_bytes,
        )
        if avail_bytes > 0 and size_bytes > 0 and size_bytes >= avail_bytes:
            logger.info(
                "TIFF stack (%s) is larger than available RAM; enabling out-of-core caching.",
                path,
            )
            return True
        if AUTO_OUT_OF_CORE_MB > 0 and size_bytes > 0:
            if size_bytes >= AUTO_OUT_OF_CORE_MB * 1024 * 1024:
                logger.info(
                    "TIFF stack (%s) exceeds configured threshold (%.0f MB); enabling out-of-core caching.",
                    path,
                    AUTO_OUT_OF_CORE_MB,
                )
                return True
        return False

    def _safe_file_size(self, path: Path) -> int:
        try:
            return int(path.stat().st_size)
        except Exception:
            return 0

    def _available_memory_bytes(self) -> int:
        try:
            import psutil  # type: ignore
        except Exception:
            return 0
        try:
            mem = psutil.virtual_memory()
        except Exception:
            return 0
        return int(getattr(mem, "available", 0) or 0)

    def _read_tiff_eager(self, path: Path) -> _VolumeData:
        frames: list[np.ndarray] = []
        with Image.open(str(path)) as img:
            n = max(1, int(getattr(img, "n_frames", 1) or 1))
            for i in range(n):
                img.seek(i)
                frames.append(np.array(img))
        if not frames:
            raise RuntimeError("No frames found in TIFF stack.")
        vol = np.stack(frames, axis=0)
        vol = self._normalize_to_float01(vol)
        return _VolumeData(
            array=vol,
            spacing=None,
            vmin=float(vol.min()),
            vmax=float(vol.max()),
            is_grayscale=vol.ndim == 3,
            is_out_of_core=False,
            volume_shape=tuple(int(x) for x in vol.shape[:3]),
        )

    def _read_tiff_out_of_core(self, path: Path) -> _VolumeData:
        meta = self._probe_tiff_metadata(path)
        shape = meta[0] if meta else None
        dtype = meta[1] if meta else None
        memmap_arr = self._open_tiff_memmap(path)
        if memmap_arr is not None:
            mem_shape = tuple(int(x) for x in memmap_arr.shape)
            shape = shape or mem_shape
            dtype = dtype or memmap_arr.dtype
            if shape and dtype and self._should_use_slice_mode(shape, dtype):
                loader = _MemmapSliceLoader(memmap_arr)
                logger.info(
                    "Slice mode (memmap) for TIFF stack '%s' (shape=%s, dtype=%s)",
                    path,
                    loader.shape(),
                    loader.dtype(),
                )
                return self._make_slice_volume_data(loader)
            vmin, vmax = self._dtype_value_range(memmap_arr.dtype)
            logger.info(
                "Using tifffile.memmap for TIFF stack '%s' (shape=%s, dtype=%s)",
                path,
                mem_shape,
                memmap_arr.dtype,
            )
            return _VolumeData(
                array=memmap_arr,
                spacing=None,
                vmin=vmin,
                vmax=vmax,
                is_grayscale=True,
                is_out_of_core=True,
                backing_path=None,
                volume_shape=mem_shape,
            )
        if shape and dtype and self._should_use_slice_mode(shape, dtype):
            loader = _TiffSliceLoader(path, shape, dtype)
            logger.info(
                "Slice mode (paged) for TIFF stack '%s' (shape=%s, dtype=%s)",
                path,
                shape,
                dtype,
            )
            return self._make_slice_volume_data(loader)
        with Image.open(str(path)) as img:
            n_frames = max(1, int(getattr(img, "n_frames", 1) or 1))
            img.seek(0)
            first = np.array(img)
            convert_color = first.ndim == 3 and first.shape[-1] in (3, 4)
            if convert_color:
                target_dtype = first.dtype
                first_plane = self._convert_frame_to_plane(first, target_dtype)
            else:
                target_dtype = first.dtype
                first_plane = np.asarray(first, dtype=target_dtype)
                if first_plane.ndim == 3 and first_plane.shape[-1] == 1:
                    first_plane = first_plane[..., 0]
            if first_plane.ndim != 2:
                raise RuntimeError("Only grayscale TIFF stacks are supported.")
            plane_shape = first_plane.shape
            fd, tmp_path = tempfile.mkstemp(
                prefix="annolid_vtk_", suffix=".mmap")
            os.close(fd)
            backing_path = Path(tmp_path)
            try:
                writer = np.memmap(
                    backing_path,
                    mode="w+",
                    dtype=target_dtype,
                    shape=(n_frames, plane_shape[0], plane_shape[1]),
                )
                min_val = float(np.min(first_plane))
                max_val = float(np.max(first_plane))
                writer[0] = first_plane
                for idx in range(1, n_frames):
                    img.seek(idx)
                    arr = np.array(img)
                    if convert_color:
                        arr = self._convert_frame_to_plane(arr, target_dtype)
                    else:
                        arr = np.asarray(arr, dtype=target_dtype)
                        if arr.ndim == 3 and arr.shape[-1] == 1:
                            arr = arr[..., 0]
                    writer[idx] = arr
                    min_val = min(min_val, float(np.min(arr)))
                    max_val = max(max_val, float(np.max(arr)))
                writer.flush()
            except Exception:
                try:
                    backing_path.unlink()
                except Exception:
                    pass
                raise
            finally:
                try:
                    del writer
                except UnboundLocalError:
                    pass
        reader = np.memmap(
            backing_path,
            mode="r+",
            dtype=target_dtype,
            shape=(n_frames, plane_shape[0], plane_shape[1]),
        )
        return _VolumeData(
            reader,
            None,
            float(min_val),
            float(max_val),
            is_grayscale=True,
            is_out_of_core=True,
            backing_path=backing_path,
            volume_shape=(n_frames, plane_shape[0], plane_shape[1]),
        )

    def _open_tiff_memmap(self, path: Path) -> Optional[np.ndarray]:
        try:
            import tifffile  # type: ignore
        except Exception:
            return None
        try:
            try:
                reader = tifffile.memmap(str(path), mode="r+")
            except PermissionError:
                reader = tifffile.memmap(str(path), mode="r")
        except Exception:
            return None
        if reader.ndim not in (3, 4):
            return None
        if reader.ndim == 4 and reader.shape[-1] == 1:
            reader = reader[..., 0]
        if reader.ndim != 3:
            return None
        if not reader.flags.c_contiguous:
            reader = np.ascontiguousarray(reader)
        return reader

    def _make_slice_volume_data(self, loader: _BaseSliceLoader) -> _VolumeData:
        shape = loader.shape()
        dtype = loader.dtype()
        vmin, vmax = self._dtype_value_range(dtype)
        return _VolumeData(
            array=None,
            spacing=None,
            vmin=vmin,
            vmax=vmax,
            is_grayscale=True,
            is_out_of_core=True,
            slice_mode=True,
            slice_loader=loader,
            slice_axis=0,
            volume_shape=shape,
        )

    def _probe_tiff_metadata(
        self, path: Path
    ) -> Optional[tuple[tuple[int, int, int], np.dtype]]:
        try:
            import tifffile  # type: ignore
        except Exception:
            return None
        try:
            with tifffile.TiffFile(str(path)) as tif:
                series = tif.series[0]
                shape = tuple(int(x) for x in series.shape)
                dtype = np.dtype(series.dtype)
        except Exception:
            return None
        if len(shape) == 2:
            shape = (1, shape[0], shape[1])
        elif len(shape) > 3:
            shape = shape[:3]
        return shape, dtype

    def _should_use_slice_mode(
        self, shape: tuple[int, int, int], dtype: np.dtype
    ) -> bool:
        if not shape or len(shape) < 3:
            return False
        total_voxels = int(shape[0]) * int(shape[1]) * int(shape[2])
        itemsize = max(1, np.dtype(dtype).itemsize)
        size_bytes = total_voxels * itemsize
        if MAX_VOLUME_VOXELS > 0 and total_voxels > MAX_VOLUME_VOXELS:
            return True
        if SLICE_MODE_BYTES > 0 and size_bytes >= SLICE_MODE_BYTES:
            return True
        available = self._available_memory_bytes()
        if available > 0 and size_bytes >= available * 0.8:
            return True
        return False

    def _dtype_value_range(self, dtype: np.dtype) -> tuple[float, float]:
        if np.issubdtype(dtype, np.bool_):
            return 0.0, 1.0
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return float(info.min), float(info.max)
        if np.issubdtype(dtype, np.floating):
            return 0.0, 1.0
        return 0.0, 1.0

    def _convert_frame_to_plane(self, frame: np.ndarray, dtype: np.dtype) -> np.ndarray:
        arr = np.asarray(frame)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            rgb = arr[..., :3].astype(np.float32)
            gray = np.dot(rgb, [0.299, 0.587, 0.114])
            if np.issubdtype(dtype, np.bool_):
                gray = (gray > 0.5).astype(dtype)
            elif np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
                gray = np.clip(np.round(gray), info.min,
                               info.max).astype(dtype)
            else:
                gray = gray.astype(dtype)
            return gray
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        if arr.dtype != dtype:
            arr = arr.astype(dtype)
        return arr

    def _vtk_image_to_numpy(self, vtk_img) -> np.ndarray:
        from vtkmodules.util.numpy_support import vtk_to_numpy
        dims = vtk_img.GetDimensions()  # (x, y, z)
        scalars = vtk_img.GetPointData().GetScalars()
        if scalars is None:
            raise RuntimeError("No scalar data in volume.")
        arr = vtk_to_numpy(scalars)
        # VTK stores as x-fastest; reshape and permute to (Z, Y, X)
        arr = arr.reshape(dims[2], dims[1], dims[0])
        return arr

    def _normalize_to_float01(self, vol: np.ndarray) -> np.ndarray:
        if np.issubdtype(vol.dtype, np.integer):
            vmin = float(vol.min())
            vmax = float(vol.max())
            if vmax > vmin:
                vol = (vol.astype(np.float32) - vmin) / (vmax - vmin)
            else:
                vol = np.zeros_like(vol, dtype=np.float32)
        else:
            vol = vol.astype(np.float32)
            vol = np.clip(vol, 0.0, 1.0)
        return vol

    def _read_dicom_series(self, directory: Path) -> tuple[np.ndarray, Optional[tuple[float, float, float]]]:
        # First try GDCM (if present), then fallback to VTK's basic reader
        reader = None
        if _HAS_GDCM:
            try:
                reader = vtkGDCMImageReader()  # type: ignore[name-defined]
                reader.SetDirectoryName(str(directory))
                reader.Update()
            except Exception:
                reader = None
        if reader is None:
            try:
                from vtkmodules.vtkIOImage import vtkDICOMImageReader
            except Exception as exc:
                raise RuntimeError(
                    "VTK DICOM reader is not available in this build.") from exc
            reader = vtkDICOMImageReader()
            reader.SetDirectoryName(str(directory))
            reader.Update()

        vtk_img = reader.GetOutput()
        vol = self._vtk_image_to_numpy(vtk_img)
        s = vtk_img.GetSpacing()
        spacing = (s[0], s[1], s[2])
        vol = self._normalize_to_float01(vol)
        return vol, spacing

    def _update_opacity(self):
        if not getattr(self, "_has_volume", False):
            return
        # Adjust overall opacity scaling via unit distance
        val = self.opacity_slider.value() / 100.0
        # Smaller unit distance -> denser appearance
        unit = max(0.001, 2.0 * (1.0 - val) + 0.05)
        self.property.SetScalarOpacityUnitDistance(unit)
        self.vtk_widget.GetRenderWindow().Render()

    def _update_shading(self):
        if not getattr(self, "_has_volume", False):
            return
        if self.shade_checkbox.isChecked():
            self.property.ShadeOn()
        else:
            self.property.ShadeOff()
        self.vtk_widget.GetRenderWindow().Render()

    def _configure_plane_controls(self):
        for axis, control in self._slice_plane_controls.items():
            total = self._plane_total(axis)
            available = total > 0 and axis in self._slice_plane_widgets
            control.slider.blockSignals(True)
            if available:
                control.slider.setRange(0, max(0, total - 1))
                control.slider.setValue(total // 2)
            else:
                control.slider.setRange(0, 0)
                control.slider.setValue(0)
            control.slider.blockSignals(False)
            control.slider.setEnabled(available)
            control.checkbox.blockSignals(True)
            control.checkbox.setEnabled(available)
            if not available:
                control.checkbox.setChecked(False)
                if axis in self._slice_plane_widgets:
                    self._slice_plane_widgets[axis].SetEnabled(False)
            control.checkbox.blockSignals(False)
            self._update_plane_label(axis, control.slider.value(), total)
            if available:
                self._apply_plane_slice(
                    axis, control.slider.value(), render=False, update_slider=False
                )

    def _plane_property(self, axis: int) -> vtkProperty:
        prop = vtkProperty()
        color = PLANE_COLORS.get(axis, (0.8, 0.8, 0.2))
        prop.SetColor(*color)
        prop.SetOpacity(0.65)
        prop.SetAmbient(1.0)
        prop.SetDiffuse(0.0)
        prop.SetSpecular(0.0)
        prop.SetInterpolationToFlat()
        return prop

    def _teardown_slice_planes(self):
        for widget in self._slice_plane_widgets.values():
            try:
                widget.SetEnabled(False)
            except Exception:
                pass
            try:
                widget.SetInteractor(None)
            except Exception:
                pass
        self._slice_plane_widgets.clear()

    def _setup_slice_plane(self):
        self._teardown_slice_planes()
        if not _HAS_PLANE_WIDGET or self._vtk_img is None:
            return
        for axis, _, orientation_name in PLANE_DEFS:
            try:
                widget = vtkImagePlaneWidget()
            except Exception:
                continue
            widget.SetInteractor(self.interactor)
            widget.SetInputData(self._vtk_img)
            orient_fn = getattr(widget, orientation_name, None)
            if callable(orient_fn):
                orient_fn()
            widget.TextureInterpolateOn()
            widget.DisplayTextOn()
            widget.SetEnabled(False)
            if getattr(self, "renderer", None) is not None:
                try:
                    widget.SetDefaultRenderer(self.renderer)
                except Exception:
                    pass
            try:
                widget.AlwaysOnTopOn()
            except Exception:
                pass
            widget.SetPlaneProperty(self._plane_property(axis))
            try:
                widget.SetResliceInterpolateToLinear()
            except Exception:
                pass
            self._slice_plane_widgets[axis] = widget

    def _plane_total(self, axis: int) -> int:
        if axis < len(self._volume_shape):
            return self._volume_shape[axis]
        return 0

    def _plane_is_available(self, axis: int) -> bool:
        return axis in self._slice_plane_widgets and self._plane_total(axis) > 0

    def _clear_slice_clipping_planes(self):
        if not getattr(self, "mapper", None):
            self._slice_clipping_planes.clear()
            return
        for plane in self._slice_clipping_planes.values():
            try:
                self.mapper.RemoveClippingPlane(plane)
            except Exception:
                pass
        self._slice_clipping_planes.clear()

    def _update_plane_label(self, axis: int, index: int, total: int):
        control = self._slice_plane_controls.get(axis)
        if control is None:
            return
        if total <= 0:
            control.label.setText(f"{control.name}: -/-")
        else:
            control.label.setText(f"{control.name}: {index + 1}/{total}")

    def _apply_plane_slice(
        self, axis: int, index: int, render: bool = True, update_slider: bool = True
    ):
        control = self._slice_plane_controls.get(axis)
        if control is None:
            return
        total = self._plane_total(axis)
        if total <= 0:
            return
        sanitized = max(0, min(index, total - 1))
        if update_slider:
            control.slider.blockSignals(True)
            control.slider.setValue(sanitized)
            control.slider.blockSignals(False)
        self._update_plane_label(axis, sanitized, total)
        widget = self._slice_plane_widgets.get(axis)
        if widget:
            widget.SetSliceIndex(sanitized)
            widget.SetEnabled(control.checkbox.isChecked())
        self._update_clipping_for_plane(axis, control.checkbox.isChecked())
        if render:
            self.vtk_widget.GetRenderWindow().Render()

    def _on_plane_slider_changed(self, axis: int, value: int):
        control = self._slice_plane_controls.get(axis)
        if control and not control.checkbox.isChecked():
            control.checkbox.blockSignals(True)
            control.checkbox.setChecked(True)
            control.checkbox.blockSignals(False)
            self._on_plane_checkbox_changed(axis)
            return
        self._apply_plane_slice(axis, value)

    def _on_plane_checkbox_changed(self, axis: int, *_):
        widget = self._slice_plane_widgets.get(axis)
        control = self._slice_plane_controls.get(axis)
        if widget is None or control is None:
            return
        enabled = control.checkbox.isChecked()
        widget.SetEnabled(enabled)
        if enabled:
            self._apply_plane_slice(axis, control.slider.value())
        else:
            self._update_clipping_for_plane(axis, False)
            self.vtk_widget.GetRenderWindow().Render()

    def _update_clipping_for_plane(self, axis: int, enabled: bool):
        plane = self._slice_clipping_planes.get(axis)
        if not enabled:
            if plane is not None:
                try:
                    self.mapper.RemoveClippingPlane(plane)
                except Exception:
                    pass
                self._slice_clipping_planes.pop(axis, None)
            return
        widget = self._slice_plane_widgets.get(axis)
        control = self._slice_plane_controls.get(axis)
        if widget is None or control is None:
            return
        if plane is None:
            plane = vtkPlane()
            self._slice_clipping_planes[axis] = plane
            try:
                self.mapper.AddClippingPlane(plane)
            except Exception:
                pass
        origin = widget.GetCenter()
        normal = widget.GetNormal()
        plane.SetOrigin(origin)
        plane.SetNormal(normal)

    # -------------------- Interaction helpers --------------------
    def _toggle_wl_mode(self):
        if not getattr(self, "_has_volume", False):
            try:
                self.wl_mode_checkbox.setChecked(False)
            except Exception:
                pass
            self._wl_mode = False
            return
        self._wl_mode = self.wl_mode_checkbox.isChecked()
        # Disable camera interaction while in WL mode to avoid conflicts
        if self._wl_mode:
            self.interactor.SetInteractorStyle(self._style_inactive)
        else:
            self.interactor.SetInteractorStyle(self._style_trackball)

    def _vtk_on_left_press(self, obj, evt):
        if not getattr(self, "_has_volume", False) or not self._wl_mode:
            return
        self._wl_drag = True
        self._wl_last = self.interactor.GetEventPosition()

    def _vtk_on_left_release(self, obj, evt):
        if not getattr(self, "_has_volume", False) or not self._wl_mode:
            return
        self._wl_drag = False

    def _vtk_on_mouse_move(self, obj, evt):
        """Handles mouse movement for both window/level adjustment and point picking."""
        # --- BRANCH 1: Window/Level Drag Interaction ---
        # This logic takes precedence if the user is in W/L mode and dragging.
        if self._wl_mode and self._wl_drag:
            x, y = self.interactor.GetEventPosition()
            last_x, last_y = self._wl_last
            dx = x - last_x
            dy = y - last_y
            self._wl_last = (x, y)

            # Adjust window/level based on mouse delta
            wmin = float(self.min_spin.value())
            wmax = float(self.max_spin.value())
            window = max(1e-6, wmax - wmin)
            level = (wmax + wmin) * 0.5

            # Horizontal movement adjusts window width, vertical adjusts level/center
            new_window = max(1e-6, window * (1.0 + dx * 0.01))
            # Invert dy for natural feel
            new_level = level + (-dy) * (window * 0.005)

            vmin = new_level - new_window * 0.5
            vmax = new_level + new_window * 0.5

            # Clamp the new values to the full data range
            vmin = max(self._vmin, min(vmin, self._vmax - 1e-6))
            vmax = max(vmin + 1e-6, min(vmax, self._vmax))

            # Update the UI widgets without emitting signals to avoid loops
            self.min_spin.blockSignals(True)
            self.max_spin.blockSignals(True)
            self.min_spin.setValue(vmin)
            self.max_spin.setValue(vmax)
            self.min_spin.blockSignals(False)
            self.max_spin.blockSignals(False)

            self._update_transfer_functions()
            self.vtk_widget.GetRenderWindow().Render()

            # End processing for this event; do not proceed to picking logic
            return

        # --- BRANCH 2: Point Picking and Tooltip Interaction ---
        # This runs during normal mouse movement when not dragging in W/L mode.
        # We also prevent picking if the W/L mode checkbox is checked, even if not dragging.
        if self._wl_mode:
            return

        x, y = self.interactor.GetEventPosition()
        self._picker.Pick(x, y, 0, self.renderer)

        actor = self._picker.GetActor()
        point_id = self._picker.GetPointId()

        # Check if we successfully picked a point on a valid point cloud actor
        if point_id > -1 and hasattr(self, "_point_actors") and actor in self._point_actors:
            # To prevent flickering, only update the tooltip if the picked point is new
            if point_id != self._last_picked_id or actor != self._last_picked_actor:
                self._last_picked_id = point_id
                self._last_picked_actor = actor

                polydata = actor.GetMapper().GetInput()

                # Attempt to retrieve the region label data array
                label_array_abstract = polydata.GetPointData().GetAbstractArray("RegionLabel")

                tooltip_parts = []
                # If the array exists, get the string value for the picked point
                if label_array_abstract:
                    # The array must be safely cast to a vtkStringArray to get its value
                    label_array = vtkStringArray.SafeDownCast(
                        label_array_abstract)
                    if label_array:
                        region_name = label_array.GetValue(point_id)
                        tooltip_parts.append(f"<b>Region:</b> {region_name}")

                # Always add point ID and coordinates to the tooltip
                coords = self._picker.GetPickPosition()
                tooltip_parts.append(f"<b>Point ID:</b> {point_id}")
                tooltip_parts.append(
                    f"<b>Coords:</b> ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})")

                tooltip_text = "<br>".join(tooltip_parts)

                # Show the rich-text tooltip at the current cursor position
                QtWidgets.QToolTip.showText(
                    QtGui.QCursor.pos(), tooltip_text, self.vtk_widget
                )

        else:
            # If the cursor is not over a point (or moved off a point), hide the tooltip
            if self._last_picked_id != -1:
                QtWidgets.QToolTip.hideText()
                self._last_picked_id = -1
                self._last_picked_actor = None

    def _vtk_on_key_press(self, obj, evt):
        key = self.interactor.GetKeySym().lower()
        if key == 'r':
            self._reset_camera()
        elif key == 'w':
            self.wl_mode_checkbox.setChecked(
                not self.wl_mode_checkbox.isChecked())
        elif key == 'c':
            self.shade_checkbox.setChecked(not self.shade_checkbox.isChecked())
        elif key == '+':
            self.opacity_slider.setValue(
                min(100, self.opacity_slider.value() + 5))
        elif key == '-':
            self.opacity_slider.setValue(
                max(1, self.opacity_slider.value() - 5))

    def _update_interpolation(self):
        if not getattr(self, "_has_volume", False):
            return
        if self.interp_combo.currentText() == "Nearest":
            self.property.SetInterpolationTypeToNearest()
        else:
            self.property.SetInterpolationTypeToLinear()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_spacing(self):
        if not getattr(self, "_has_volume", False):
            return
        try:
            sx = float(self.spacing_x.value())
            sy = float(self.spacing_y.value())
            sz = float(self.spacing_z.value())
            self._vtk_img.SetSpacing(sx, sy, sz)
            self.vtk_widget.GetRenderWindow().Render()
        except Exception:
            pass

    def _reset_camera(self):
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    def _save_snapshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Snapshot", str(
            self._path.with_suffix('.png')), "PNG Files (*.png)")
        if not path:
            return
        w2i = vtkWindowToImageFilter()
        w2i.SetInput(self.vtk_widget.GetRenderWindow())
        w2i.Update()
        writer = vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        QtWidgets.QToolTip.showText(
            QtGui.QCursor.pos(), f"Saved: {Path(path).name}")

    def _on_window_changed(self):
        # Ensure min <= max
        vmin = min(self.min_spin.value(), self.max_spin.value())
        vmax = max(self.min_spin.value(), self.max_spin.value())
        if vmin != self.min_spin.value():
            self.min_spin.blockSignals(True)
            self.min_spin.setValue(vmin)
            self.min_spin.blockSignals(False)
        if vmax != self.max_spin.value():
            self.max_spin.blockSignals(True)
            self.max_spin.setValue(vmax)
            self.max_spin.blockSignals(False)
        self._update_transfer_functions()
        self.vtk_widget.GetRenderWindow().Render()

    def _auto_window(self):
        if self._slice_mode:
            self._slice_auto_window()
            return
        if not getattr(self, "_has_volume", False) or self._volume_np is None:
            return
        # 2-98 percentile auto window on current volume
        vol = self._volume_np
        p2, p98 = float(np.percentile(vol, 2)), float(np.percentile(vol, 98))
        if p98 <= p2:
            p2, p98 = float(vol.min()), float(vol.max())
        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(p2)
        self.max_spin.setValue(p98)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        self._update_transfer_functions()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_transfer_functions(self):
        if self._slice_mode:
            return
        if not getattr(self, "_has_volume", False) or self._opacity_tf is None or self._color_tf is None:
            return
        # Build opacity and color TF based on window and colormap
        vmin = float(self.min_spin.value()) if hasattr(
            self, 'min_spin') else self._vmin
        vmax = float(self.max_spin.value()) if hasattr(
            self, 'max_spin') else self._vmax
        if vmax <= vmin:
            vmax = vmin + 1e-3

        # Opacity: ramp from vmin to vmax
        self._opacity_tf.RemoveAllPoints()
        self._opacity_tf.AddPoint(vmin, 0.0)
        self._opacity_tf.AddPoint((vmin + vmax) * 0.5, 0.1)
        self._opacity_tf.AddPoint(vmax, 0.9)

        # Color map
        cmap = self.cmap_combo.currentText() if hasattr(
            self, 'cmap_combo') else "Grayscale"
        self._color_tf.RemoveAllPoints()
        if cmap == "Grayscale":
            self._color_tf.AddRGBPoint(vmin, 0.0, 0.0, 0.0)
            self._color_tf.AddRGBPoint(vmax, 1.0, 1.0, 1.0)
        elif cmap == "Invert Gray":
            self._color_tf.AddRGBPoint(vmin, 1.0, 1.0, 1.0)
            self._color_tf.AddRGBPoint(vmax, 0.0, 0.0, 0.0)
        elif cmap == "Hot":
            # Rough "hot" ramp: black -> red -> yellow -> white
            self._color_tf.AddRGBPoint(vmin, 0.0, 0.0, 0.0)
            self._color_tf.AddRGBPoint(
                vmin + (vmax - vmin) * 0.33, 1.0, 0.0, 0.0)
            self._color_tf.AddRGBPoint(
                vmin + (vmax - vmin) * 0.66, 1.0, 1.0, 0.0)
            self._color_tf.AddRGBPoint(vmax, 1.0, 1.0, 1.0)

    def _update_blend_mode(self):
        mode = self.blend_combo.currentText() if hasattr(
            self, 'blend_combo') else "Composite"
        try:
            if mode == "MIP-Max":
                self.mapper.SetBlendModeToMaximumIntensity()
            elif mode == "MIP-Min":
                self.mapper.SetBlendModeToMinimumIntensity()
            elif mode == "Additive":
                self.mapper.SetBlendModeToAdditive()
            else:
                self.mapper.SetBlendModeToComposite()
        except Exception:
            # Some mappers may not support all modes; ignore
            pass
        self.vtk_widget.GetRenderWindow().Render()

    # -------------------- Point cloud support --------------------
    def _load_point_cloud_folder(self):
        start_dir = str(self._path.parent) if getattr(
            self, "_path", None) and self._path.exists() else "."
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Open Point Cloud Folder",
            start_dir,
        )
        if not folder:
            return
        directory = Path(folder)
        csv_files = sorted(directory.glob("*.csv"))
        if not csv_files:
            QtWidgets.QMessageBox.warning(
                self,
                "Point Cloud",
                "No CSV point clouds found in that directory.",
            )
            return
        self._clear_point_clouds()
        combined_bounds = None
        for csv_path in csv_files:
            try:
                bounds = self._add_point_cloud_csv_or_xyz(
                    str(csv_path), focus=False
                )
                combined_bounds = self._union_bounds(combined_bounds, bounds)
            except Exception as exc:
                logger.warning(
                    "Failed to load point cloud '%s': %s", csv_path, exc
                )
        if combined_bounds:
            self._focus_on_bounds(combined_bounds)
        self._update_point_sizes()
        self.vtk_widget.GetRenderWindow().Render()

    def _add_point_cloud_ply(self, path: str):
        try:
            from vtkmodules.vtkIOGeometry import vtkPLYReader  # lazy import
        except Exception as exc:  # pragma: no cover - optional module
            raise RuntimeError(
                "VTK PLY reader module is not available. Install a VTK build with IOGeometry support."
            ) from exc
        reader = vtkPLYReader()
        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("PLY contains no points")
        # Ask user for scale factors (default to 1, or volume spacing if available)
        scale = self._prompt_point_scale()
        if scale is not None:
            poly = self._scale_poly_points(poly, scale)
        # Ensure vertices exist without vtkVertexGlyphFilter
        poly2 = self._ensure_vertices(poly)
        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly2)
        # Use embedded PLY colors if present
        mapper.ScalarVisibilityOn()
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(self.point_size_slider.value())
        self.renderer.AddActor(actor)
        self._point_actors.append(actor)
        self._focus_on_bounds(poly2.GetBounds())
        self._update_point_controls_visibility()

    def _add_point_cloud_csv_or_xyz(self, path: str, focus: bool = True):
        import numpy as np
        pts = None
        colors = None
        intensity = None
        region_labels = None
        if path.lower().endswith(".xyz"):
            data = np.loadtxt(path)
            if data.ndim != 2 or data.shape[1] < 3:
                raise RuntimeError("XYZ must have at least 3 columns: x y z")
            pts = data[:, :3].astype(np.float32)
            if data.shape[1] >= 6:
                col = data[:, 3:6]
                if col.max() <= 1.0:
                    col = (col * 255.0)
                colors = np.clip(col, 0, 255).astype(np.uint8)
            # Ask for scale factors
            scale = self._prompt_point_scale()
            if scale is not None:
                pts *= np.array(scale, dtype=np.float32)
        else:  # CSV with mapping dialog
            from .csv_mapping_dialog import CSVPointCloudMappingDialog
            df = pd.read_csv(path)
            parsed = self._auto_parse_point_cloud_csv(df)
            if parsed is not None:
                pts, colors, intensity, region_labels = parsed
            else:
                dlg = CSVPointCloudMappingDialog(df.columns, parent=self)
                if dlg.exec_() != QtWidgets.QDialog.Accepted:
                    return
                m = dlg.mapping()

                def get(name):
                    col = m.get(name)
                    return None if col is None else df[col]
                try:
                    pts = np.stack([get('x').to_numpy(), get('y').to_numpy(), get(
                        'z').to_numpy()], axis=1).astype(np.float32)
                except Exception:
                    raise RuntimeError("Invalid X/Y/Z column mapping")
                # Apply scale from dialog
                try:
                    sx, sy, sz = float(m.get('sx', 1.0)), float(
                        m.get('sy', 1.0)), float(m.get('sz', 1.0))
                    pts *= np.array([sx, sy, sz], dtype=np.float32)
                except Exception:
                    pass

                # Build colors
                color_by = m.get('color_by')
                mode = m.get('color_mode')
                if color_by:
                    series = get('color_by')
                    if mode == 'categorical' or (series.dtype == object):
                        vals = series.astype(str).to_numpy()
                        colors = self._colors_for_categories(vals)
                    else:
                        raw = series.to_numpy(dtype=float)
                        v = raw[np.isfinite(raw)] if raw.size else raw
                        vmin = float(np.nanmin(v)) if v.size else 0.0
                        vmax = float(np.nanmax(v)) if v.size else 1.0
                        if vmax <= vmin:
                            vmax = vmin + 1e-6
                        norm = (raw - vmin) / (vmax - vmin)
                        norm = np.clip(norm, 0.0, 1.0)
                        colors = self._gradient_blue_red(norm)
                else:
                    inten_series = get('intensity')
                    if inten_series is not None:
                        inten = inten_series.to_numpy(dtype=float)
                        imax = np.nanmax(inten) if inten.size else 1.0
                        imin = np.nanmin(inten) if inten.size else 0.0
                        if imax <= imin:
                            imax = imin + 1e-6
                        g = np.clip((inten - imin) / (imax - imin), 0.0, 1.0)
                        g = (g * 255.0).astype(np.uint8)
                        colors = np.stack([g, g, g], axis=1)

                # Parse region labels if specified
                region_labels = None
                label_col = m.get('label_by')
                if label_col:
                    try:
                        # Ensure labels are strings for the tooltip
                        region_labels = get('label_by').astype(str).to_numpy()
                    except Exception as e:
                        print(
                            f"Warning: Could not parse region labels from column '{label_col}': {e}")

        if pts is None or len(pts) == 0:
            raise RuntimeError("No points parsed")

        if intensity is not None:
            intensity_values = intensity.astype(float)
        else:
            intensity_values = pts[:, 2].astype(float)

        if colors is None:
            vals = intensity_values
            finite_vals = vals[np.isfinite(vals)] if vals.size else vals
            vmin = float(np.nanmin(finite_vals)) if finite_vals.size else 0.0
            vmax = float(np.nanmax(finite_vals)) if finite_vals.size else 1.0
            if vmax <= vmin:
                vmax = vmin + 1e-6
            norm = (vals - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0.0, 1.0)
            colors = self._gradient_blue_red(norm)

        safe_pts = np.asarray(pts, dtype=np.float32)
        bounds = (
            float(np.nanmin(safe_pts[:, 0])),
            float(np.nanmax(safe_pts[:, 0])),
            float(np.nanmin(safe_pts[:, 1])),
            float(np.nanmax(safe_pts[:, 1])),
            float(np.nanmin(safe_pts[:, 2])),
            float(np.nanmax(safe_pts[:, 2])),
        )

        self._region_actors = {}
        region_default_colors: dict[str, QtGui.QColor] = {}
        region_label_list: list[str] | None = None
        if region_labels is not None and len(region_labels) == len(pts):
            region_label_list = [str(label) for label in region_labels]

        if region_label_list:
            groups: dict[str, list[int]] = OrderedDict()
            for idx, label in enumerate(region_label_list):
                groups.setdefault(label, []).append(idx)

            for label, idxs in groups.items():
                indices = np.asarray(idxs, dtype=np.intp)
                subset_pts = safe_pts[indices]
                subset_colors = colors[indices] if colors is not None else None
                actor = self._create_point_actor(
                    subset_pts, subset_colors, region_label=label)
                if actor is None:
                    continue
                self.renderer.AddActor(actor)
                self._point_actors.append(actor)
                self._region_actors[label] = actor
                region_default_colors[label] = self._infer_region_color(
                    subset_colors)
        else:
            actor = self._create_point_actor(safe_pts, colors)
            if actor is not None:
                self.renderer.AddActor(actor)
                self._point_actors.append(actor)

        self._populate_region_selection(
            list(self._region_actors.keys()), region_default_colors)
        self._update_point_controls_visibility()
        if focus:
            self._focus_on_bounds(bounds)
        return bounds

    def _ensure_vertices(self, poly: vtkPolyData) -> vtkPolyData:
        """Create a polydata with Verts so that points render as glyphs.

        Avoids requiring vtkVertexGlyphFilter which may be absent in some VTK builds.
        """
        npts = poly.GetNumberOfPoints()
        verts = vtkCellArray()
        for i in range(npts):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        out = vtkPolyData()
        out.SetPoints(poly.GetPoints())
        out.SetVerts(verts)
        try:
            out.GetPointData().ShallowCopy(poly.GetPointData())
        except Exception:
            pass
        return out

    def _create_point_actor(
        self,
        pts: np.ndarray,
        colors: Optional[np.ndarray],
        region_label: Optional[str] = None,
    ) -> Optional[vtkActor]:
        if pts is None or len(pts) == 0:
            return None
        vpoints = vtkPoints()
        vpoints.SetNumberOfPoints(len(pts))
        for i, (x, y, z) in enumerate(pts):
            vpoints.SetPoint(i, float(x), float(y), float(z))
        poly = vtkPolyData()
        poly.SetPoints(vpoints)
        poly2 = self._ensure_vertices(poly)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly2)
        if colors is not None and len(colors):
            safe_colors = np.clip(colors, 0.0, 255.0).astype(np.uint8)
            c_arr = numpy_to_vtk(safe_colors, deep=True)
            try:
                c_arr.SetNumberOfComponents(3)
            except Exception:
                pass
            c_arr.SetName("Colors")
            poly2.GetPointData().SetScalars(c_arr)
            mapper.SetColorModeToDirectScalars()
            mapper.ScalarVisibilityOn()
            mapper.SetScalarModeToUsePointData()
            mapper.SelectColorArray("Colors")

        if region_label is not None:
            label_array = vtkStringArray()
            label_array.SetName("RegionLabel")
            label_array.SetNumberOfValues(len(pts))
            for i in range(len(pts)):
                label_array.SetValue(i, region_label)
            poly2.GetPointData().AddArray(label_array)

        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(self.point_size_slider.value())
        return actor

    def _gradient_blue_red(self, norm: np.ndarray) -> np.ndarray:
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        norm = np.clip(norm, 0.0, 1.0)
        c0 = np.array([30, 70, 200], dtype=np.float32)
        c1 = np.array([200, 50, 50], dtype=np.float32)
        rgb = (c0[None, :] * (1.0 - norm[:, None]) +
               c1[None, :] * norm[:, None])
        return np.clip(rgb, 0, 255).astype(np.uint8)

    def _colors_for_categories(self, vals: np.ndarray) -> np.ndarray:
        palette = np.array([
            [230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200],
            [245, 130, 48], [145, 30, 180], [70, 240, 240], [240, 50, 230],
            [210, 245, 60], [250, 190, 190], [0, 128, 128], [230, 190, 255]
        ], dtype=np.uint8)
        uniq = {}
        out = np.zeros((len(vals), 3), dtype=np.uint8)
        next_idx = 0
        for i, v in enumerate(vals):
            if v not in uniq:
                uniq[v] = next_idx
                next_idx = (next_idx + 1) % len(palette)
            out[i] = palette[uniq[v]]
        return out

    def _prompt_point_scale(self) -> Optional[Tuple[float, float, float]]:
        """Prompt user for point scale factors (spacing) before loading.

        Defaults to (1,1,1). If a volume exists, prefill with current volume spacing.
        Returns None if user cancels.
        """
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Point Spacing / Scale")
        form = QtWidgets.QFormLayout(dlg)
        sx = QtWidgets.QDoubleSpinBox(dlg)
        sy = QtWidgets.QDoubleSpinBox(dlg)
        sz = QtWidgets.QDoubleSpinBox(dlg)
        for sb in (sx, sy, sz):
            sb.setDecimals(6)
            sb.setRange(0.000001, 1e9)
            sb.setValue(1.0)
        # Prefill from volume spacing if available
        try:
            if getattr(self, "_vtk_img", None) is not None:
                sp = self._vtk_img.GetSpacing()
                sx.setValue(float(sp[0]))
                sy.setValue(float(sp[1]))
                sz.setValue(float(sp[2]))
        except Exception:
            pass
        row = QtWidgets.QHBoxLayout()
        row.addWidget(sx)
        row.addWidget(sy)
        row.addWidget(sz)
        w = QtWidgets.QWidget(dlg)
        w.setLayout(row)
        form.addRow("Scale X/Y/Z", w)
        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addWidget(buttons)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return float(sx.value()), float(sy.value()), float(sz.value())

    def _auto_parse_point_cloud_csv(
        self, df: "pd.DataFrame"
    ) -> Optional[tuple[np.ndarray, Optional[np.ndarray], np.ndarray, None]]:
        lookup = {col.lower(): col for col in df.columns}
        lower_cols = set(lookup)
        allowed = {"x", "y", "z", "intensity", "red", "green", "blue"}
        if not lower_cols.issubset(allowed):
            return None
        if not all(name in lower_cols for name in ("x", "y", "z")):
            return None

        try:
            coords = np.stack(
                [
                    df[lookup["x"]].to_numpy(dtype=float),
                    df[lookup["y"]].to_numpy(dtype=float),
                    df[lookup["z"]].to_numpy(dtype=float),
                ],
                axis=1,
            ).astype(np.float32)
        except Exception:
            return None

        intensity = None
        if "intensity" in lookup:
            intensity = df[lookup["intensity"]].to_numpy(dtype=float)
        else:
            intensity = coords[:, 2]

        colors = None
        color_keys = ("red", "green", "blue")
        if all(key in lookup for key in color_keys):
            colors = np.stack(
                [df[lookup[key]].to_numpy(dtype=float) for key in color_keys], axis=1
            )
        return coords, colors, intensity.astype(np.float32), None

    def _scale_poly_points(self, poly: vtkPolyData, scale: Tuple[float, float, float]) -> vtkPolyData:
        """Return a copy of poly with points multiplied by scale per axis."""
        pts = poly.GetPoints()
        if pts is None:
            return poly
        n = pts.GetNumberOfPoints()
        new_pts = vtkPoints()
        new_pts.SetNumberOfPoints(n)
        sx, sy, sz = scale
        for i in range(n):
            x, y, z = pts.GetPoint(i)
            new_pts.SetPoint(i, float(x) * sx, float(y) * sy, float(z) * sz)
        out = vtkPolyData()
        out.SetPoints(new_pts)
        try:
            out.GetPointData().ShallowCopy(poly.GetPointData())
        except Exception:
            pass
        return out

    def _focus_on_bounds(self, bounds: Optional[Tuple[float, float, float, float, float, float]]) -> None:
        """Adjust the camera to frame the provided bounds."""
        if not bounds:
            return
        values = np.array(bounds, dtype=float)
        if not np.all(np.isfinite(values)):
            return
        xmin, xmax, ymin, ymax, zmin, zmax = values
        center = (
            (xmin + xmax) * 0.5,
            (ymin + ymax) * 0.5,
            (zmin + zmax) * 0.5,
        )
        radius = max(xmax - xmin, ymax - ymin, zmax - zmin)
        if not np.isfinite(radius) or radius <= 0:
            radius = 1.0
        distance = radius * 2.5
        camera = self.renderer.GetActiveCamera()
        camera.SetFocalPoint(*center)
        camera.SetPosition(center[0], center[1] -
                           distance, center[2] + distance)
        camera.SetViewUp(0.0, 0.0, 1.0)
        self.renderer.ResetCameraClippingRange()

    def _union_bounds(
        self,
        first: Optional[Tuple[float, float, float, float, float, float]],
        second: Optional[Tuple[float, float, float, float, float, float]],
    ) -> Optional[Tuple[float, float, float, float, float, float]]:
        if first is None:
            return second
        if second is None:
            return first
        return (
            min(first[0], second[0]),
            max(first[1], second[1]),
            min(first[2], second[2]),
            max(first[3], second[3]),
            min(first[4], second[4]),
            max(first[5], second[5]),
        )

    def _update_point_sizes(self):
        size = int(self.point_size_slider.value())
        for actor in getattr(self, "_point_actors", []):
            actor.GetProperty().SetPointSize(size)
        self.vtk_widget.GetRenderWindow().Render()

    def _update_point_controls_visibility(self) -> None:
        has_points = bool(getattr(self, "_point_actors", []))
        self.point_detail_widget.setVisible(has_points)
        if not has_points:
            self._clear_region_selection()

    def _update_mesh_controls_visibility(self) -> None:
        has_mesh = bool(getattr(self, "_mesh_actors", []))
        self.mesh_detail_widget.setVisible(has_mesh)

    def _clear_point_clouds(self):
        for actor in getattr(self, "_point_actors", []):
            try:
                self.renderer.RemoveActor(actor)
            except Exception:
                pass
        self._point_actors = []
        self._region_actors = {}
        self._clear_region_selection()
        self._update_point_controls_visibility()
        self.vtk_widget.GetRenderWindow().Render()

    def _clear_region_selection(self) -> None:
        if not hasattr(self, "region_list_widget"):
            return
        self.region_list_widget.blockSignals(True)
        self.region_list_widget.clear()
        self.region_list_widget.blockSignals(False)
        self.region_list_widget.setEnabled(False)
        self.region_group.setEnabled(False)
        self._region_entries.clear()
        self._region_colors.clear()
        if hasattr(self, "region_search"):
            self.region_search.blockSignals(True)
            self.region_search.clear()
            self.region_search.blockSignals(False)

    def _format_region_display(self, label: str) -> str:
        acronym, name = self._split_region_label(label)
        if acronym and name and acronym != name:
            return f"{acronym} - {name}"
        if acronym:
            return acronym
        if name:
            return name
        return str(label or "")

    def _split_region_label(self, label: str) -> tuple[str, str]:
        text = (label or "").strip()
        if not text:
            return "", ""
        paren_match = re.search(r"\(([^)]+)\)\s*$", text)
        if paren_match:
            acronym = paren_match.group(1).strip()
            name = text[:paren_match.start()].strip()
            if not name:
                name = text
            return acronym, name
        for delim in (":", "—", "-", "–"):
            if delim in text:
                left, right = [part.strip() for part in text.split(delim, 1)]
                if left and right:
                    if self._looks_like_region_acronym(left):
                        return left, right
                    if self._looks_like_region_acronym(right):
                        return right, left
        return "", text

    def _looks_like_region_acronym(self, segment: str) -> bool:
        seg = (segment or "").strip()
        if not seg:
            return False
        if seg.isupper() and len(seg) <= 8:
            return True
        upper_digit_count = sum(
            1 for c in seg if c.isupper() or c.isdigit())
        if upper_digit_count >= 2 and len(seg) <= 6:
            return True
        return False

    def _populate_region_selection(
        self,
        labels: Sequence[str],
        default_colors: Mapping[str, QtGui.QColor] | None = None,
    ) -> None:
        if not labels:
            self._clear_region_selection()
            return
        seen = set()
        ordered = []
        for label in labels:
            if label not in seen:
                seen.add(label)
                ordered.append(label)

        self.region_list_widget.blockSignals(True)
        self.region_list_widget.clear()
        self._region_entries.clear()
        self._region_colors.clear()
        default_colors = default_colors or {}
        for label in ordered:
            display_text = self._format_region_display(label)
            item = QtWidgets.QListWidgetItem()
            item.setData(QtCore.Qt.UserRole, label)
            item.setFlags(QtCore.Qt.ItemIsEnabled)
            self.region_list_widget.addItem(item)

            widget = QtWidgets.QWidget()
            widget_layout = QtWidgets.QHBoxLayout(widget)
            widget_layout.setContentsMargins(2, 1, 2, 1)
            widget_layout.setSpacing(4)
            color_button = QtWidgets.QToolButton()
            color_button.setAutoRaise(True)
            color_button.setFixedSize(20, 20)
            color_button.clicked.connect(
                lambda _, lbl=label: self._pick_region_color(lbl)
            )
            checkbox = QtWidgets.QCheckBox(display_text)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(
                lambda state, lbl=label: self._set_region_visibility(
                    lbl, state == QtCore.Qt.Checked
                )
            )
            widget_layout.addWidget(color_button)
            widget_layout.addWidget(checkbox)
            widget_layout.addStretch(1)
            self.region_list_widget.setItemWidget(item, widget)
            item.setSizeHint(widget.sizeHint())

            entry = _RegionSelectionEntry(
                item=item, checkbox=checkbox, color_button=color_button,
                display_text=display_text,
            )
            self._region_entries[label] = entry

            color = default_colors.get(label)
            if color is None:
                color = QtGui.QColor(255, 255, 255)
            self._region_colors[label] = color
            self._update_region_color_button(label, color)
        self.region_list_widget.blockSignals(False)
        self.region_group.setEnabled(True)
        self.region_list_widget.setEnabled(True)
        self._filter_region_items()

    def _filter_region_items(self, query: str | None = None) -> None:
        if query is None and hasattr(self, "region_search"):
            query = self.region_search.text()
        query = (query or "").strip().lower()
        should_filter = bool(query)
        for entry in self._region_entries.values():
            display = entry.display_text.lower()
            entry.item.setHidden(should_filter and query not in display)

    def _set_region_check_states(self, checked: bool) -> None:
        if not self._region_entries:
            return
        self.region_list_widget.blockSignals(True)
        for entry in self._region_entries.values():
            entry.checkbox.blockSignals(True)
            entry.checkbox.setChecked(checked)
            entry.checkbox.blockSignals(False)
        self.region_list_widget.blockSignals(False)
        self._refresh_region_visibilities()

    def _refresh_region_visibilities(self) -> None:
        for label, entry in self._region_entries.items():
            actor = self._region_actors.get(label)
            if actor:
                actor.SetVisibility(entry.checkbox.isChecked())
        self.vtk_widget.GetRenderWindow().Render()

    def _set_region_visibility(self, label: str, visible: bool) -> None:
        actor = self._region_actors.get(label)
        if actor:
            actor.SetVisibility(visible)
            self.vtk_widget.GetRenderWindow().Render()

    def _pick_region_color(self, label: str) -> None:
        current = self._region_colors.get(label, QtGui.QColor(255, 255, 255))
        color = QtWidgets.QColorDialog.getColor(
            current, self, "Pick region color")
        if not color.isValid():
            return
        self._region_colors[label] = color
        self._update_region_color_button(label, color)
        self._apply_region_color(label, color)

    def _update_region_color_button(self, label: str, color: QtGui.QColor) -> None:
        entry = self._region_entries.get(label)
        if entry is None:
            return
        size = QtCore.QSize(16, 16)
        pixmap = QtGui.QPixmap(size)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.setBrush(QtGui.QBrush(color))
        painter.setPen(QtGui.QPen(QtGui.QColor(60, 60, 60), 1))
        painter.drawRect(1, 1, size.width() - 3, size.height() - 3)
        painter.end()
        entry.color_button.setIcon(QtGui.QIcon(pixmap))
        entry.color_button.setIconSize(size)
        entry.color_button.setToolTip(color.name())

    def _apply_region_color(self, label: str, color: QtGui.QColor) -> None:
        actor = self._region_actors.get(label)
        if actor is None:
            return
        prop = actor.GetProperty()
        prop.SetColor(color.redF(), color.greenF(), color.blueF())
        mapper = actor.GetMapper()
        if hasattr(mapper, "ScalarVisibilityOff"):
            mapper.ScalarVisibilityOff()
        self.vtk_widget.GetRenderWindow().Render()

    def _infer_region_color(
        self, colors: Optional[np.ndarray]
    ) -> QtGui.QColor:
        fallback = QtGui.QColor(255, 255, 255)
        if colors is None or len(colors) == 0:
            return fallback
        arr = np.asarray(colors, dtype=float)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.size == 0:
            return fallback
        if arr.shape[1] < 3:
            arr = np.pad(
                arr, ((0, 0), (0, 3 - arr.shape[1])), constant_values=0.0)
        arr = arr[:, :3]
        if arr.size == 0:
            return fallback
        max_val = np.nanmax(arr)
        if max_val <= 1.0:
            arr = arr * 255.0
        arr = np.nan_to_num(arr, nan=0.0, posinf=255.0, neginf=0.0)
        mean = np.nanmean(arr, axis=0)
        rgb = np.clip(mean, 0.0, 255.0).astype(int)
        return QtGui.QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def _load_mesh_dialog(self):
        start_dir = str(self._path.parent) if getattr(
            self, "_path", None) and self._path.exists() else "."
        filters = "Meshes (*.stl *.STL *.obj *.OBJ *.ply *.PLY);;All files (*.*)"
        res = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Mesh",
            start_dir,
            filters,
        )
        path = res[0] if isinstance(res, tuple) else res
        if not path:
            return
        path = str(path)
        try:
            self._load_mesh_file(path)
            self.renderer.ResetCamera()
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Mesh Viewer", f"Failed to load: {e}")

    def _load_diffuse_texture(self):
        self._load_texture_dialog("diffuse")

    def _load_normal_texture(self):
        self._load_texture_dialog("normal")

    def _load_mesh_file(self, path: str):
        path_obj = Path(path)
        ext = path_obj.suffix.lower()
        textures: dict[str, str] = {}
        if ext == ".stl":
            poly = self._read_stl_mesh(path)
        elif ext == ".obj":
            poly = self._read_obj_mesh(path)
            textures = self._parse_obj_mtl_textures(path_obj)
        elif ext == ".ply":
            poly = self._read_ply_mesh(path)
        else:
            raise RuntimeError(f"Unsupported mesh format: {ext}")
        actor = self._add_mesh_actor(poly, path_obj.name)
        if textures.get("diffuse"):
            self._apply_texture_from_path(
                actor, "diffuse", textures["diffuse"])
        if textures.get("normal"):
            self._apply_texture_from_path(actor, "normal", textures["normal"])

    def _parse_obj_mtl_textures(self, obj_path: Path) -> dict[str, str]:
        textures: dict[str, str] = {}
        mtl_files: list[Path] = []
        try:
            with open(obj_path, "r", encoding="utf-8", errors="ignore") as obj_file:
                for line in obj_file:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#"):
                        continue
                    parts = stripped.split(maxsplit=1)
                    if parts[0].lower() != "mtllib" or len(parts) < 2:
                        continue
                    for name in parts[1].split():
                        candidate = obj_path.parent / name
                        mtl_files.append(candidate)
        except Exception:
            return textures
        for mtl_path in mtl_files:
            if not mtl_path.exists():
                continue
            try:
                with open(mtl_path, "r", encoding="utf-8", errors="ignore") as mtl_file:
                    for line in mtl_file:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        tokens = line.split(maxsplit=1)
                        if len(tokens) < 2:
                            continue
                        key = tokens[0].lower()
                        raw = tokens[1].split("#", 1)[0].strip()
                        if not raw:
                            continue
                        file_name = raw.split()[-1]
                        if not file_name:
                            continue
                        file_path = (mtl_path.parent / file_name).resolve()
                        if not file_path.exists():
                            continue
                        if key == "map_kd" and "diffuse" not in textures:
                            textures["diffuse"] = str(file_path)
                        elif key in {"map_bump", "bump", "norm"} and "normal" not in textures:
                            textures["normal"] = str(file_path)
                        if "diffuse" in textures and "normal" in textures:
                            return textures
            except Exception:
                continue
        return textures

    def _add_mesh_actor(self, poly: vtkPolyData, source_name: str) -> vtkActor:
        mapper = self._create_mesh_mapper()
        mapper.SetInputData(poly)
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetRepresentationToSurface()
        actor.GetProperty().SetColor(1.0, 1.0, 1.0)
        self.renderer.AddActor(actor)
        self._mesh_actors.append(actor)
        self._mesh_textures[id(actor)] = {}
        self._mesh_actor_names[id(actor)] = source_name
        self._active_mesh_actor = actor
        self._update_mesh_status_label()
        self._update_mesh_controls_visibility()
        return actor

    def _clear_meshes(self):
        for actor in getattr(self, "_mesh_actors", []):
            try:
                self.renderer.RemoveActor(actor)
            except Exception:
                pass
        self._mesh_actors = []
        self._mesh_textures.clear()
        self._mesh_actor_names.clear()
        self._active_mesh_actor = None
        self._update_mesh_controls_visibility()
        self.vtk_widget.GetRenderWindow().Render()
        self._update_mesh_status_label()

    def _load_texture_dialog(self, kind: str):
        actor = self._current_mesh_actor()
        if actor is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Mesh Texture",
                "Load a mesh before applying textures.",
            )
            return
        filters = "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)"
        res = QtWidgets.QFileDialog.getOpenFileName(
            self,
            f"Open {kind.capitalize()} Texture",
            str(self._path.parent) if getattr(self, "_path", None) else ".",
            filters,
        )
        path = res[0] if isinstance(res, tuple) else res
        if not path:
            return
        try:
            self._apply_texture_from_path(actor, kind, path)
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                "Mesh Texture",
                f"Failed to load {kind} texture: {exc}",
            )

    def _current_mesh_actor(self) -> Optional[vtkActor]:
        actor = self._active_mesh_actor
        if actor not in self._mesh_actors:
            actor = self._mesh_actors[-1] if self._mesh_actors else None
        if actor is not None:
            self._active_mesh_actor = actor
        return actor

    def _apply_texture_from_path(self, actor: vtkActor, kind: str, path: str):
        texture = self._create_texture_from_file(path)
        self._apply_texture_to_actor(actor, texture, kind, path)

    def _create_texture_from_file(self, path: str) -> vtkTexture:
        factory = vtkImageReader2Factory()
        reader = factory.CreateImageReader2(path)
        if reader is None:
            raise RuntimeError("Unsupported texture format")
        reader.SetFileName(path)
        reader.Update()
        image = reader.GetOutput()
        if image is None:
            raise RuntimeError("Texture image could not be read")
        texture = vtkTexture()
        texture.SetInputData(image)
        texture.InterpolateOn()
        return texture

    def closeEvent(self, event):  # type: ignore[override]
        try:
            self._teardown_slice_planes()
        except Exception:
            pass
        try:
            self._teardown_slice_mode()
        except Exception:
            pass
        self._cleanup_out_of_core_backing()
        super().closeEvent(event)

    def _apply_texture_to_actor(
        self,
        actor: vtkActor,
        texture: vtkTexture,
        kind: str,
        path: Optional[str] = None,
    ):
        actor.SetTexture(texture)
        self._mesh_textures.setdefault(id(actor), {})[kind] = {
            "texture": texture,
            "path": path,
        }
        if kind == "normal":
            mapper = actor.GetMapper()
            if hasattr(mapper, "SetNormalTexture"):
                mapper.SetNormalTexture(texture)
            if hasattr(mapper, "SetUseNormalTexture"):
                mapper.SetUseNormalTexture(True)
        self._update_mesh_status_label()

    def _create_mesh_mapper(self) -> vtkPolyDataMapper:
        if vtkOpenGLPolyDataMapper is not None:
            return vtkOpenGLPolyDataMapper()
        return vtkPolyDataMapper()

    def _update_mesh_status_label(self):
        if not hasattr(self, "mesh_status_label"):
            return
        actor = self._current_mesh_actor()
        if actor is None:
            self.mesh_status_label.setText("Mesh: none loaded")
            return
        name = self._mesh_actor_names.get(id(actor), "mesh")
        textures = self._mesh_textures.get(id(actor), {})
        parts = [f"Mesh: {name}"]
        for kind_label, key in (("Diffuse", "diffuse"), ("Normal", "normal")):
            entry = textures.get(key)
            path = entry.get("path") if isinstance(entry, dict) else None
            if path:
                parts.append(f"{kind_label}: {Path(path).name}")
        self.mesh_status_label.setText(" | ".join(parts))

    def _update_volume_io_label(self, out_of_core: bool):
        if not hasattr(self, "volume_io_label"):
            return
        if self._slice_mode:
            text = "Volume I/O: slice viewer (on-demand)"
        elif out_of_core:
            text = "Volume I/O: out-of-core (auto)"
        else:
            text = "Volume I/O: in-memory"
        self.volume_io_label.setText(text)

    def _read_stl_mesh(self, path: str) -> vtkPolyData:
        try:
            from vtkmodules.vtkIOGeometry import vtkSTLReader
        except Exception as exc:
            raise RuntimeError(
                "VTK STL reader module is not available. Install VTK with IOGeometry support."
            ) from exc
        return self._read_polydata_from_reader(vtkSTLReader, path)

    def _read_obj_mesh(self, path: str) -> vtkPolyData:
        try:
            from vtkmodules.vtkIOGeometry import vtkOBJReader
        except Exception as exc:
            raise RuntimeError(
                "VTK OBJ reader module is not available. Install VTK with IOGeometry support."
            ) from exc
        return self._read_polydata_from_reader(vtkOBJReader, path)

    def _read_ply_mesh(self, path: str) -> vtkPolyData:
        try:
            from vtkmodules.vtkIOGeometry import vtkPLYReader
        except Exception as exc:
            raise RuntimeError(
                "VTK PLY reader module is not available. Install VTK with IOGeometry support."
            ) from exc
        return self._read_polydata_from_reader(vtkPLYReader, path)

    def _read_polydata_from_reader(self, reader_cls: Callable[[], object], path: str) -> vtkPolyData:
        reader = reader_cls()
        reader.SetFileName(path)
        reader.Update()
        poly = reader.GetOutput()
        if poly is None or poly.GetNumberOfPoints() == 0:
            raise RuntimeError("Mesh contains no points")
        return poly

    # -------------------- Source type helpers --------------------
    def _resolve_initial_source(self, path: Path) -> Path:
        """If a directory was provided, auto-pick the first supported file."""
        try:
            if not path.exists():
                return path
        except Exception:
            return path
        if not path.is_dir():
            return path

        try:
            entries = sorted(path.iterdir())
        except Exception:
            return path

        # DICOM or other volume directories should remain directories
        for entry in entries:
            if entry.is_file() and entry.suffix.lower() in DICOM_EXTS:
                return path

        def _find(exts: Tuple[str, ...]) -> Optional[Path]:
            for entry in entries:
                if entry.is_file() and entry.suffix.lower() in exts:
                    return entry
            return None

        for ext_group in (POINT_CLOUD_EXTS, MESH_EXTS, VOLUME_FILE_EXTS):
            candidate = _find(ext_group)
            if candidate is not None:
                logger.info(
                    "VTK viewer: auto-selecting '%s' inside '%s'.",
                    candidate.name,
                    path,
                )
                return candidate
        return path

    def _is_volume_candidate(self, path: Path) -> bool:
        if path.is_dir():
            try:
                return any(
                    entry.is_file() and entry.suffix.lower() in DICOM_EXTS
                    for entry in path.iterdir()
                )
            except Exception:
                return False
        ext = path.suffix.lower()
        name = path.name.lower()
        return (
            ext in VOLUME_FILE_EXTS
            or ext in DICOM_EXTS
            or name.endswith('.ome.tif')
            or name.endswith('.nii')
            or name.endswith('.nii.gz')
        )

    def _is_point_cloud_candidate(self, path: Path) -> bool:
        if path.is_dir():
            try:
                return any(
                    entry.is_file() and entry.suffix.lower() in POINT_CLOUD_EXTS
                    for entry in path.iterdir()
                )
            except Exception:
                return False
        return path.suffix.lower() in POINT_CLOUD_EXTS

    def _is_mesh_candidate(self, path: Path) -> bool:
        if path.is_dir():
            try:
                return any(
                    entry.is_file() and entry.suffix.lower() in MESH_EXTS
                    for entry in path.iterdir()
                )
            except Exception:
                return False
        return path.suffix.lower() in MESH_EXTS
