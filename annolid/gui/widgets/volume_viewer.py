from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from qtpy import QtCore, QtGui, QtWidgets
import qimage2ndarray


class VolumeViewerDialog(QtWidgets.QDialog):
    """
    Minimal 3D stack viewer without external dependencies.

    Features:
    - Z-slice browsing via slider
    - Optional MIP across Z when the full stack is loaded into memory
    - Simple contrast window (min/max) and invert toggle

    This is a built-in viewer with no external 3D dependencies.
    """

    def __init__(
        self, tiff_path: str | Path, parent: Optional[QtWidgets.QWidget] = None
    ):
        super().__init__(parent)
        self.setWindowTitle("3D Stack Viewer")
        self.resize(900, 700)

        self._path = Path(tiff_path)
        self._pil: Optional[Image.Image] = None
        self._n_frames: int = 0
        self._volume: Optional[np.ndarray] = None  # (Z, Y, X) or (Z, Y, X, C)
        self._dtype = None

        # UI
        self._image_label = QtWidgets.QLabel()
        self._image_label.setAlignment(QtCore.Qt.AlignCenter)
        self._image_label.setBackgroundRole(QtGui.QPalette.Base)
        self._image_label.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )
        self._image_label.setMinimumSize(200, 200)

        self._z_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self._z_slider.valueChanged.connect(self._on_controls_changed)
        self._z_label = QtWidgets.QLabel("Z: -/-")

        self._mip_checkbox = QtWidgets.QCheckBox("MIP (Z)")
        self._mip_checkbox.setToolTip(
            "Maximum Intensity Projection across Z (requires volume in memory)"
        )
        self._mip_checkbox.stateChanged.connect(self._on_controls_changed)

        self._invert_checkbox = QtWidgets.QCheckBox("Invert")
        self._invert_checkbox.stateChanged.connect(self._on_controls_changed)

        self._auto_btn = QtWidgets.QPushButton("Auto Contrast")
        self._auto_btn.clicked.connect(self._auto_contrast)

        self._min_spin = QtWidgets.QDoubleSpinBox()
        self._min_spin.setPrefix("Min ")
        self._min_spin.setDecimals(2)
        self._min_spin.valueChanged.connect(self._on_controls_changed)

        self._max_spin = QtWidgets.QDoubleSpinBox()
        self._max_spin.setPrefix("Max ")
        self._max_spin.setDecimals(2)
        self._max_spin.valueChanged.connect(self._on_controls_changed)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(self._z_label)
        controls.addWidget(self._z_slider, 1)
        controls.addSpacing(8)
        controls.addWidget(self._mip_checkbox)
        controls.addSpacing(8)
        controls.addWidget(self._invert_checkbox)
        controls.addSpacing(8)
        controls.addWidget(self._auto_btn)
        controls.addSpacing(8)
        controls.addWidget(self._min_spin)
        controls.addWidget(self._max_spin)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(self._image_label, 1)
        layout.addLayout(controls)

        self._load_stack_header()
        self._maybe_load_volume()
        self._configure_controls()
        self._update_view()

    # ------------------------- Data loading -------------------------
    def _load_stack_header(self):
        self._pil = Image.open(str(self._path))
        n = getattr(self._pil, "n_frames", 1)
        self._n_frames = int(n) if n and n > 0 else 1
        # Probe dtype
        self._pil.seek(0)
        probe = np.array(self._pil)
        self._dtype = probe.dtype

    def _estimate_bytes(self) -> int:
        assert self._pil is not None
        w, h = self._pil.size
        # Assume channels if present in first frame
        self._pil.seek(0)
        arr = np.array(self._pil)
        c = 1 if arr.ndim == 2 else arr.shape[-1]
        itemsize = arr.dtype.itemsize
        return int(self._n_frames) * int(h) * int(w) * int(c) * int(itemsize)

    def _maybe_load_volume(self, threshold_mb: int = 512):
        """Load full volume into memory if estimated size <= threshold_mb."""
        est_bytes = self._estimate_bytes()
        if est_bytes <= threshold_mb * 1024 * 1024:
            frames = []
            for i in range(self._n_frames):
                self._pil.seek(i)
                frames.append(np.array(self._pil))
            self._volume = np.stack(frames, axis=0)
        else:
            self._volume = None

    # ------------------------- UI helpers ---------------------------
    def _configure_controls(self):
        self._z_slider.setRange(0, max(0, self._n_frames - 1))
        self._z_slider.setValue(0)

        # Contrast bounds based on dtype
        if np.issubdtype(self._dtype, np.integer):
            info = np.iinfo(self._dtype)
            lo, hi = float(info.min), float(info.max)
        else:
            lo, hi = 0.0, 1.0

        self._min_spin.setRange(lo, hi)
        self._max_spin.setRange(lo, hi)
        # Sensible defaults
        self._min_spin.setValue(lo)
        self._max_spin.setValue(hi)

        # Enable MIP only if volume is in memory
        self._mip_checkbox.setEnabled(self._volume is not None)
        if self._volume is None:
            self._mip_checkbox.setToolTip(
                "MIP is disabled for large stacks not fully loaded into memory."
            )

    def _auto_contrast(self):
        # Compute robust limits from the current view target
        arr = self._current_data_for_stats()
        if arr is None:
            return
        # Percentile stretch
        vmin = float(np.percentile(arr, 2))
        vmax = float(np.percentile(arr, 98))
        if vmax <= vmin:
            vmin, vmax = float(arr.min()), float(arr.max())
        self._min_spin.blockSignals(True)
        self._max_spin.blockSignals(True)
        self._min_spin.setValue(vmin)
        self._max_spin.setValue(vmax)
        self._min_spin.blockSignals(False)
        self._max_spin.blockSignals(False)
        self._update_view()

    def _current_data_for_stats(self) -> Optional[np.ndarray]:
        if self._mip_checkbox.isChecked() and self._volume is not None:
            if self._volume.ndim == 4:
                # Convert to luminance for stats
                gray = np.dot(self._volume[..., :3], [0.299, 0.587, 0.114])
                return gray.max(axis=0)
            return self._volume.max(axis=0)
        # Current slice
        z = self._z_slider.value()
        if self._volume is not None:
            arr = self._volume[z]
        else:
            assert self._pil is not None
            self._pil.seek(z)
            arr = np.array(self._pil)
        if arr.ndim == 3 and arr.shape[-1] in (3, 4):
            # Luminance for stats
            arr = np.dot(arr[..., :3], [0.299, 0.587, 0.114])
        return arr

    def _on_controls_changed(self, *_):
        self._update_view()

    # ------------------------- Rendering ----------------------------
    def _apply_window(
        self, arr: np.ndarray, lo: float, hi: float, invert: bool
    ) -> np.ndarray:
        if arr.dtype.kind in ("i", "u", "f"):
            arrf = arr.astype(np.float32)
            denom = max(hi - lo, 1e-6)
            norm = np.clip((arrf - lo) / denom, 0.0, 1.0)
            if invert:
                norm = 1.0 - norm
            out = (norm * 255.0).astype(np.uint8)
            return out
        # Non-numeric: fallback
        return arr.astype(np.uint8) if arr.dtype != np.uint8 else arr

    def _render_image(self, arr: np.ndarray) -> QtGui.QImage:
        lo = self._min_spin.value()
        hi = self._max_spin.value()
        invert = self._invert_checkbox.isChecked()

        if arr.ndim == 2:
            img8 = self._apply_window(arr, lo, hi, invert)
        elif arr.ndim == 3 and arr.shape[-1] in (3, 4):
            # Apply window per channel for non-uint8 data
            if arr.dtype != np.uint8:
                chans = []
                for c in range(arr.shape[-1]):
                    chans.append(self._apply_window(arr[..., c], lo, hi, invert))
                img8 = np.stack(chans, axis=-1)
            else:
                img8 = arr
                if invert:
                    img8 = 255 - img8
        else:
            # Unexpected shape: try to squeeze or fallback to grayscale
            img8 = self._apply_window(np.squeeze(arr), lo, hi, invert)

        qimage = qimage2ndarray.array2qimage(img8)
        return qimage

    def _update_view(self):
        # Update Z label
        z = self._z_slider.value()
        self._z_label.setText(f"Z: {z + 1}/{self._n_frames}")

        # Select data
        if self._mip_checkbox.isChecked() and self._volume is not None:
            arr = self._volume.max(axis=0)
        else:
            if self._volume is not None:
                arr = self._volume[z]
            else:
                assert self._pil is not None
                self._pil.seek(z)
                arr = np.array(self._pil)

        # Render and fit into label
        qimage = self._render_image(arr)
        pixmap = QtGui.QPixmap.fromImage(qimage)
        scaled = pixmap.scaled(
            self._image_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self._image_label.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:  # noqa: N802
        super().resizeEvent(event)
        self._update_view()
