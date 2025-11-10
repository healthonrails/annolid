from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from qtpy import QtCore, QtWidgets, QtGui

# VTK imports (modular) — if these fail, the caller should fall back
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkVolume,
    vtkVolumeProperty,
    vtkWindowToImageFilter,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData, vtkCellArray
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonCore import vtkStringArray
from vtkmodules.vtkRenderingCore import vtkPointPicker
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleUser,
)
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkIOImage import vtkPNGWriter
from vtkmodules.vtkRenderingCore import vtkPolyDataMapper, vtkActor
try:  # Prefer GDCM when available (more robust series handling)
    from vtkmodules.vtkIOImage import vtkGDCMImageReader  # type: ignore
    _HAS_GDCM = True
except Exception:  # pragma: no cover
    _HAS_GDCM = False


class VTKVolumeViewerDialog(QtWidgets.QDialog):
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
        try:
            self._path = Path(src_path) if src_path else Path('.')
        except Exception:
            self._path = Path('.')

        # Main layout
        layout = QtWidgets.QVBoxLayout(self)
        # QVTK widget
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout.addWidget(self.vtk_widget, 1)

        # Controls area (user-friendly)
        controls = QtWidgets.QGridLayout()

        # Blend mode
        controls.addWidget(QtWidgets.QLabel("Blend:"), 0, 0)
        self.blend_combo = QtWidgets.QComboBox()
        self.blend_combo.addItems(["Composite", "MIP-Max", "MIP-Min", "Additive"])
        self.blend_combo.currentIndexChanged.connect(self._update_blend_mode)
        controls.addWidget(self.blend_combo, 0, 1)

        # Colormap
        controls.addWidget(QtWidgets.QLabel("Colormap:"), 0, 2)
        self.cmap_combo = QtWidgets.QComboBox()
        self.cmap_combo.addItems(["Grayscale", "Invert Gray", "Hot"])
        self.cmap_combo.currentIndexChanged.connect(self._update_transfer_functions)
        controls.addWidget(self.cmap_combo, 0, 3)

        # Intensity window (min/max)
        controls.addWidget(QtWidgets.QLabel("Window:"), 1, 0)
        self.min_spin = QtWidgets.QDoubleSpinBox()
        self.max_spin = QtWidgets.QDoubleSpinBox()
        for spin in (self.min_spin, self.max_spin):
            spin.setDecimals(3)
            spin.setKeyboardTracking(False)
        self.min_spin.valueChanged.connect(self._on_window_changed)
        self.max_spin.valueChanged.connect(self._on_window_changed)
        controls.addWidget(self.min_spin, 1, 1)
        controls.addWidget(self.max_spin, 1, 2)
        self.auto_window_btn = QtWidgets.QPushButton("Auto")
        self.auto_window_btn.clicked.connect(self._auto_window)
        controls.addWidget(self.auto_window_btn, 1, 3)

        # Density (global opacity) and shading
        controls.addWidget(QtWidgets.QLabel("Density:"), 2, 0)
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(1, 100)
        self.opacity_slider.setValue(30)
        self.opacity_slider.valueChanged.connect(self._update_opacity)
        controls.addWidget(self.opacity_slider, 2, 1, 1, 2)
        self.shade_checkbox = QtWidgets.QCheckBox("Shading")
        self.shade_checkbox.setChecked(True)
        self.shade_checkbox.stateChanged.connect(self._update_shading)
        controls.addWidget(self.shade_checkbox, 2, 3)

        # Interpolation
        controls.addWidget(QtWidgets.QLabel("Interpolation:"), 3, 0)
        self.interp_combo = QtWidgets.QComboBox()
        self.interp_combo.addItems(["Linear", "Nearest"])
        self.interp_combo.currentIndexChanged.connect(self._update_interpolation)
        controls.addWidget(self.interp_combo, 3, 1)

        # Spacing (X, Y, Z)
        controls.addWidget(QtWidgets.QLabel("Spacing X/Y/Z:"), 3, 2)
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
        controls.addWidget(spacing_widget, 3, 3)

        # Window/Level drag mode toggle
        self.wl_mode_checkbox = QtWidgets.QCheckBox("Window/Level Mode")
        self.wl_mode_checkbox.setToolTip(
            "Enable to adjust intensity window by left-drag; camera interaction is paused"
        )
        self.wl_mode_checkbox.stateChanged.connect(self._toggle_wl_mode)

        # Buttons
        btns = QtWidgets.QHBoxLayout()
        self.reset_cam_btn = QtWidgets.QPushButton("Reset Camera")
        self.reset_cam_btn.clicked.connect(self._reset_camera)
        self.snapshot_btn = QtWidgets.QPushButton("Save Snapshot…")
        self.snapshot_btn.clicked.connect(self._save_snapshot)
        self.load_pc_btn = QtWidgets.QPushButton("Load Point Cloud…")
        self.load_pc_btn.clicked.connect(self._load_point_cloud_dialog)
        self.clear_pc_btn = QtWidgets.QPushButton("Clear Points")
        self.clear_pc_btn.clicked.connect(self._clear_point_clouds)
        self.point_size_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.point_size_slider.setRange(1, 12)
        self.point_size_slider.setValue(3)
        self.point_size_slider.setToolTip("Point size")
        self.point_size_slider.valueChanged.connect(self._update_point_sizes)
        btns.addWidget(self.wl_mode_checkbox)
        btns.addStretch(1)
        btns.addWidget(QtWidgets.QLabel("Point Size:"))
        btns.addWidget(self.point_size_slider)
        btns.addWidget(self.load_pc_btn)
        btns.addWidget(self.clear_pc_btn)
        btns.addWidget(self.reset_cam_btn)
        btns.addWidget(self.snapshot_btn)

        controls_box = QtWidgets.QVBoxLayout()
        controls_box.addLayout(controls)
        controls_box.addLayout(btns)
        layout.addLayout(controls_box)

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

        # Volume state/placeholders (for point-cloud only sessions these remain unset)
        self._has_volume: bool = False
        self._volume_np: Optional[np.ndarray] = None
        self._vtk_img = None
        self._vmin = 0.0
        self._vmax = 1.0
        self._opacity_tf = None
        self._color_tf = None

        # Conditionally load volume if path looks like a volume source
        _loaded_volume = False
        if src_path and self._is_volume_candidate(self._path):
            try:
                self._load_volume()
                _loaded_volume = True
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
        self._picker.SetTolerance(0.005) # Adjust sensitivity
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

        # Enable/disable volume-related controls based on whether a volume was loaded
        self._set_volume_controls_enabled(_loaded_volume)

    def _install_interaction_bindings(self):
        # Mouse + key handlers
        self.interactor.AddObserver("LeftButtonPressEvent", self._vtk_on_left_press)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self._vtk_on_left_release)
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

    def _load_volume(self):
        volume, spacing = self._read_volume_any()
        # Convert color stack to luminance for volume rendering if needed
        if volume.ndim == 4 and volume.shape[-1] in (3, 4):
            volume = np.dot(volume[..., :3], [0.299, 0.587, 0.114]).astype(volume.dtype)

        # Ensure C-contiguous
        volume = np.ascontiguousarray(volume)
        z, y, x = volume.shape

        vtk_img = vtkImageData()
        vtk_img.SetDimensions(int(x), int(y), int(z))
        # Apply spacing from source if available
        if spacing is not None and len(spacing) == 3:
            vtk_img.SetSpacing(float(spacing[0]), float(spacing[1]), float(spacing[2]))
        else:
            vtk_img.SetSpacing(1.0, 1.0, 1.0)
        vtk_img.SetOrigin(0.0, 0.0, 0.0)

        # Map numpy dtype to VTK scalars
        vtk_array = numpy_to_vtk(num_array=volume.reshape(-1), deep=True)
        vtk_img.GetPointData().SetScalars(vtk_array)

        # Keep handles and stats
        self._vtk_img = vtk_img
        self._volume_np = volume
        self._vmin = float(volume.min())
        self._vmax = float(volume.max())

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

    def _read_volume_any(self) -> tuple[np.ndarray, Optional[Tuple[float, float, float]]]:
        """Read a 3D volume from TIFF/NIfTI/DICOM or directory (DICOM series).

        Returns (volume, spacing). Volume is (Z, Y, X[, C]) float32 in [0, 1].
        Spacing is a (x, y, z) tuple if available, else None.
        """
        path = self._path
        spacing: Optional[Tuple[float, float, float]] = None

        try:
            if path.is_dir():
                # DICOM series directory: prefer GDCM if present
                return self._read_dicom_series(path)

            suffix = path.suffix.lower()
            name_lower = path.name.lower()
            if name_lower.endswith('.nii') or name_lower.endswith('.nii.gz'):
                # NIfTI via VTK reader
                try:
                    from vtkmodules.vtkIOImage import vtkNIFTIImageReader
                except Exception as exc:
                    raise RuntimeError("VTK NIFTI reader is not available in this build.") from exc
                reader = vtkNIFTIImageReader()
                reader.SetFileName(str(path))
                reader.Update()
                vtk_img = reader.GetOutput()
                vol = self._vtk_image_to_numpy(vtk_img)
                s = vtk_img.GetSpacing()
                spacing = (s[0], s[1], s[2])
                vol = self._normalize_to_float01(vol)
                return vol, spacing
            if suffix in ('.dcm', '.ima', '.dicom'):
                # Treat as a DICOM series from the containing folder
                return self._read_dicom_series(path.parent)

            # Default: try multi-page TIFF via PIL
            img = Image.open(str(path))
            n = getattr(img, "n_frames", 1)
            if not n or n <= 0:
                n = 1
            frames = []
            for i in range(n):
                img.seek(i)
                frames.append(np.array(img))
            vol = np.stack(frames, axis=0)  # (Z, Y, X[, C])
            vol = self._normalize_to_float01(vol)
            return vol, None
        except Exception as e:
            # Re-raise with context so upper layer shows a concise message
            raise RuntimeError(f"Failed to read volume from '{path}': {e}")

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
                raise RuntimeError("VTK DICOM reader is not available in this build.") from exc
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
            new_level = level + (-dy) * (window * 0.005) # Invert dy for natural feel

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
                    label_array = vtkStringArray.SafeDownCast(label_array_abstract)
                    if label_array:
                        region_name = label_array.GetValue(point_id)
                        tooltip_parts.append(f"<b>Region:</b> {region_name}")

                # Always add point ID and coordinates to the tooltip
                coords = self._picker.GetPickPosition()
                tooltip_parts.append(f"<b>Point ID:</b> {point_id}")
                tooltip_parts.append(f"<b>Coords:</b> ({coords[0]:.2f}, {coords[1]:.2f}, {coords[2]:.2f})")
                
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
            self.wl_mode_checkbox.setChecked(not self.wl_mode_checkbox.isChecked())
        elif key == 'c':
            self.shade_checkbox.setChecked(not self.shade_checkbox.isChecked())
        elif key == '+':
            self.opacity_slider.setValue(min(100, self.opacity_slider.value() + 5))
        elif key == '-':
            self.opacity_slider.setValue(max(1, self.opacity_slider.value() - 5))

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
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save Snapshot", str(self._path.with_suffix('.png')), "PNG Files (*.png)")
        if not path:
            return
        w2i = vtkWindowToImageFilter()
        w2i.SetInput(self.vtk_widget.GetRenderWindow())
        w2i.Update()
        writer = vtkPNGWriter()
        writer.SetFileName(path)
        writer.SetInputConnection(w2i.GetOutputPort())
        writer.Write()
        QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), f"Saved: {Path(path).name}")

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
        if not getattr(self, "_has_volume", False) or self._opacity_tf is None or self._color_tf is None:
            return
        # Build opacity and color TF based on window and colormap
        vmin = float(self.min_spin.value()) if hasattr(self, 'min_spin') else self._vmin
        vmax = float(self.max_spin.value()) if hasattr(self, 'max_spin') else self._vmax
        if vmax <= vmin:
            vmax = vmin + 1e-3

        # Opacity: ramp from vmin to vmax
        self._opacity_tf.RemoveAllPoints()
        self._opacity_tf.AddPoint(vmin, 0.0)
        self._opacity_tf.AddPoint((vmin + vmax) * 0.5, 0.1)
        self._opacity_tf.AddPoint(vmax, 0.9)

        # Color map
        cmap = self.cmap_combo.currentText() if hasattr(self, 'cmap_combo') else "Grayscale"
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
            self._color_tf.AddRGBPoint(vmin + (vmax - vmin) * 0.33, 1.0, 0.0, 0.0)
            self._color_tf.AddRGBPoint(vmin + (vmax - vmin) * 0.66, 1.0, 1.0, 0.0)
            self._color_tf.AddRGBPoint(vmax, 1.0, 1.0, 1.0)

    def _update_blend_mode(self):
        mode = self.blend_combo.currentText() if hasattr(self, 'blend_combo') else "Composite"
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
    def _load_point_cloud_dialog(self):
        start_dir = str(self._path.parent) if getattr(self, "_path", None) and self._path.exists() else "."
        filters = "Point clouds (*.csv *.CSV *.ply *.PLY *.xyz *.XYZ);;All files (*.*)"
        res = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open Point Cloud",
            start_dir,
            filters,
        )
        path = res[0] if isinstance(res, tuple) else res
        if not path:
            return
        path = str(path)
        ext = Path(path).suffix.lower()
        try:
            if ext == ".ply":
                self._add_point_cloud_ply(path)
            elif ext in (".csv", ".xyz"):
                self._add_point_cloud_csv_or_xyz(path)
            else:
                raise RuntimeError(f"Unsupported point cloud format: {ext}")
            self._update_point_sizes()
            self.vtk_widget.GetRenderWindow().Render()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Point Cloud", f"Failed to load: {e}")

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

    def _add_point_cloud_csv_or_xyz(self, path: str):
        import numpy as np
        pts = None
        colors = None
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
            import pandas as pd
            from .csv_mapping_dialog import CSVPointCloudMappingDialog
            df = pd.read_csv(path)
            dlg = CSVPointCloudMappingDialog(df.columns, parent=self)
            if dlg.exec_() != QtWidgets.QDialog.Accepted:
                return
            m = dlg.mapping()
            def get(name):
                col = m.get(name)
                return None if col is None else df[col]
            try:
                pts = np.stack([get('x').to_numpy(), get('y').to_numpy(), get('z').to_numpy()], axis=1).astype(np.float32)
            except Exception:
                raise RuntimeError("Invalid X/Y/Z column mapping")
            # Apply scale from dialog
            try:
                sx, sy, sz = float(m.get('sx', 1.0)), float(m.get('sy', 1.0)), float(m.get('sz', 1.0))
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
                    print(f"Warning: Could not parse region labels from column '{label_col}': {e}")

        if pts is None or len(pts) == 0:
            raise RuntimeError("No points parsed")

        # Build VTK polydata with vertices
        vpoints = vtkPoints()
        vpoints.SetNumberOfPoints(int(pts.shape[0]))
        for i, (x, y, z) in enumerate(pts):
            vpoints.SetPoint(i, float(x), float(y), float(z))
        poly = vtkPolyData()
        poly.SetPoints(vpoints)
        poly2 = self._ensure_vertices(poly)

        mapper = vtkPolyDataMapper()
        mapper.SetInputData(poly2)
        if colors is not None:
            from vtkmodules.util.numpy_support import numpy_to_vtk
            c_arr = numpy_to_vtk(colors, deep=True)
            # Ensure 3 components (RGB)
            try:
                c_arr.SetNumberOfComponents(3)
            except Exception:
                pass
            c_arr.SetName("Colors")
            poly2.GetPointData().SetScalars(c_arr)
            mapper.SetColorModeToDefault()
            mapper.ScalarVisibilityOn()

        # Add region label data if available
        if region_labels is not None and len(region_labels) == len(pts):
            label_array = vtkStringArray()
            label_array.SetName("RegionLabel")
            label_array.SetNumberOfValues(len(region_labels))
            for i, label in enumerate(region_labels):
                label_array.SetValue(i, label)
            poly2.GetPointData().AddArray(label_array)
            
        actor = vtkActor()
        actor.SetMapper(mapper)
        actor.GetProperty().SetPointSize(self.point_size_slider.value())
        self.renderer.AddActor(actor)
        self._point_actors.append(actor)

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

    def _gradient_blue_red(self, norm: np.ndarray) -> np.ndarray:
        norm = np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0)
        norm = np.clip(norm, 0.0, 1.0)
        c0 = np.array([30, 70, 200], dtype=np.float32)
        c1 = np.array([200, 50, 50], dtype=np.float32)
        rgb = (c0[None, :] * (1.0 - norm[:, None]) + c1[None, :] * norm[:, None])
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
        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel, parent=dlg)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        form.addWidget(buttons)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return None
        return float(sx.value()), float(sy.value()), float(sz.value())

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

    def _update_point_sizes(self):
        size = int(self.point_size_slider.value())
        for actor in getattr(self, "_point_actors", []):
            actor.GetProperty().SetPointSize(size)
        self.vtk_widget.GetRenderWindow().Render()

    def _clear_point_clouds(self):
        for actor in getattr(self, "_point_actors", []):
            try:
                self.renderer.RemoveActor(actor)
            except Exception:
                pass
        self._point_actors = []
        self.vtk_widget.GetRenderWindow().Render()

    # -------------------- Source type helpers --------------------
    def _is_volume_candidate(self, path: Path) -> bool:
        if path.is_dir():
            return True
        ext = path.suffix.lower()
        name = path.name.lower()
        return (
            ext in ('.tif', '.tiff', '.dcm', '.dicom', '.ima')
            or name.endswith('.ome.tif')
            or name.endswith('.nii')
            or name.endswith('.nii.gz')
        )

    def _is_point_cloud_candidate(self, path: Path) -> bool:
        ext = path.suffix.lower()
        return ext in ('.ply', '.csv', '.xyz')
