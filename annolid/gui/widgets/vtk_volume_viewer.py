from __future__ import annotations

from pathlib import Path
from typing import Optional

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
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonCore import vtkCommand
from vtkmodules.vtkCommonDataModel import vtkPiecewiseFunction
from vtkmodules.vtkRenderingCore import vtkColorTransferFunction
from vtkmodules.vtkInteractionStyle import (
    vtkInteractorStyleTrackballCamera,
    vtkInteractorStyleUser,
)
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkIOImage import vtkPNGWriter


class VTKVolumeViewerDialog(QtWidgets.QDialog):
    """
    True 3D volume renderer using VTK's GPU volume mapper.

    - Loads a TIFF stack into a 3D volume
    - Interact with mouse: rotate, zoom, pan
    - Simple UI controls for opacity scaling and shading toggle
    """

    def __init__(self, tiff_path: str | Path, parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(parent)
        self.setWindowTitle("3D Volume Renderer (VTK)")
        self.resize(900, 700)
        self._path = Path(tiff_path)

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
        btns.addWidget(self.wl_mode_checkbox)
        btns.addStretch(1)
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

        self._load_volume()

        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.property)
        self.renderer.AddVolume(self.volume)
        self.renderer.ResetCamera()
        self.renderer.SetBackground(0.1, 0.1, 0.12)

        # Setup interactive window/level mode and key/mouse bindings
        self._wl_mode = False
        self._wl_drag = False
        self._wl_last = (0, 0)
        self._install_interaction_bindings()

        self.vtk_widget.Initialize()
        self.vtk_widget.Start()

    def _install_interaction_bindings(self):
        # Mouse + key handlers
        self.interactor.AddObserver("LeftButtonPressEvent", self._vtk_on_left_press)
        self.interactor.AddObserver("LeftButtonReleaseEvent", self._vtk_on_left_release)
        self.interactor.AddObserver("MouseMoveEvent", self._vtk_on_mouse_move)
        self.interactor.AddObserver("KeyPressEvent", self._vtk_on_key_press)

    def _load_volume(self):
        volume = self._read_tiff_stack()
        # Convert color stack to luminance for volume rendering if needed
        if volume.ndim == 4 and volume.shape[-1] in (3, 4):
            volume = np.dot(volume[..., :3], [0.299, 0.587, 0.114]).astype(volume.dtype)

        # Ensure C-contiguous
        volume = np.ascontiguousarray(volume)
        z, y, x = volume.shape

        vtk_img = vtkImageData()
        vtk_img.SetDimensions(int(x), int(y), int(z))
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
        self._update_opacity()
        self._update_shading()
        self._update_blend_mode()

    def _read_tiff_stack(self) -> np.ndarray:
        img = Image.open(str(self._path))
        n = getattr(img, "n_frames", 1)
        if not n or n <= 0:
            n = 1
        frames = []
        for i in range(n):
            img.seek(i)
            frames.append(np.array(img))
        vol = np.stack(frames, axis=0)  # (Z, Y, X[, C])

        # Normalize volume to float32 in [0, 1] to avoid huge GL transfer
        # function textures on large integer ranges (prevents VTK warnings
        # about unsupported 1D texture sizes and ensures stable rendering).
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

    def _update_opacity(self):
        # Adjust overall opacity scaling via unit distance
        val = self.opacity_slider.value() / 100.0
        # Smaller unit distance -> denser appearance
        unit = max(0.001, 2.0 * (1.0 - val) + 0.05)
        self.property.SetScalarOpacityUnitDistance(unit)
        self.vtk_widget.GetRenderWindow().Render()

    def _update_shading(self):
        if self.shade_checkbox.isChecked():
            self.property.ShadeOn()
        else:
            self.property.ShadeOff()
        self.vtk_widget.GetRenderWindow().Render()

    # -------------------- Interaction helpers --------------------
    def _toggle_wl_mode(self):
        self._wl_mode = self.wl_mode_checkbox.isChecked()
        # Disable camera interaction while in WL mode to avoid conflicts
        if self._wl_mode:
            self.interactor.SetInteractorStyle(self._style_inactive)
        else:
            self.interactor.SetInteractorStyle(self._style_trackball)

    def _vtk_on_left_press(self, obj, evt):
        if not self._wl_mode:
            return
        self._wl_drag = True
        self._wl_last = self.interactor.GetEventPosition()

    def _vtk_on_left_release(self, obj, evt):
        if not self._wl_mode:
            return
        self._wl_drag = False

    def _vtk_on_mouse_move(self, obj, evt):
        if not (self._wl_mode and self._wl_drag):
            return
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

        # Horizontal expands window, vertical shifts level (invert sign for natural feel)
        new_window = max(1e-6, window * (1.0 + dx * 0.01))
        new_level = level + (-dy) * (window * 0.005)

        vmin = new_level - new_window * 0.5
        vmax = new_level + new_window * 0.5

        # Clamp to data range
        vmin = max(self._vmin, min(vmin, self._vmax - 1e-6))
        vmax = max(vmin + 1e-6, min(vmax, self._vmax))

        self.min_spin.blockSignals(True)
        self.max_spin.blockSignals(True)
        self.min_spin.setValue(vmin)
        self.max_spin.setValue(vmax)
        self.min_spin.blockSignals(False)
        self.max_spin.blockSignals(False)
        self._update_transfer_functions()
        self.vtk_widget.GetRenderWindow().Render()

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
        if self.interp_combo.currentText() == "Nearest":
            self.property.SetInterpolationTypeToNearest()
        else:
            self.property.SetInterpolationTypeToLinear()
        self.vtk_widget.GetRenderWindow().Render()

    def _update_spacing(self):
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
