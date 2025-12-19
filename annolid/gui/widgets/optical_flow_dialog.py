from __future__ import annotations

from typing import Optional, Tuple, Union

from qtpy import QtCore, QtWidgets


class FlowOptionsDialog(QtWidgets.QDialog):
    """Simple dialog to pick optical-flow backend, visualization, and overlay options."""

    def __init__(
        self,
        parent=None,
        *,
        default_backend: str = "farneback",
        default_raft_model: str = "small",
        default_viz: str = "quiver",
        default_opacity: Union[int, float] = 70,
        default_quiver_step: int = 16,
        default_quiver_gain: Union[int, float] = 1.0,
        default_stable_hsv: bool = True,
        default_pyr_scale: Union[int, float] = 0.5,
        default_levels: int = 1,
        default_winsize: int = 1,
        default_iterations: int = 3,
        default_poly_n: int = 3,
        default_poly_sigma: Union[int, float] = 1.1,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optical Flow Settings")

        backend_label = QtWidgets.QLabel("Flow backend:")
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(
            ["farneback (default)", "farneback (torch)", "raft (torchvision)"]
        )
        backend_val = str(default_backend).lower(
        ) if default_backend is not None else "farneback"
        if "raft" in backend_val:
            backend_idx = 2
        elif "torch" in backend_val:
            backend_idx = 1
        else:
            backend_idx = 0
        self.backend_combo.setCurrentIndex(backend_idx)

        raft_label = QtWidgets.QLabel("RAFT model:")
        self.raft_combo = QtWidgets.QComboBox()
        self.raft_combo.addItems(["small", "large"])
        raft_val = str(default_raft_model).lower(
        ) if default_raft_model is not None else "small"
        self.raft_combo.setCurrentIndex(1 if raft_val == "large" else 0)

        viz_label = QtWidgets.QLabel("Overlay visualization:")
        self.viz_combo = QtWidgets.QComboBox()
        self.viz_combo.addItems(["quiver", "hsv"])
        viz_val = str(default_viz).lower(
        ) if default_viz is not None else "quiver"
        viz_idx = 1 if viz_val == "hsv" else 0
        self.viz_combo.setCurrentIndex(viz_idx)

        opacity_label = QtWidgets.QLabel("Overlay opacity:")
        self.opacity_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.opacity_slider.setRange(0, 100)
        try:
            opacity_default = int(round(float(default_opacity)))
        except Exception:
            opacity_default = 70
        self.opacity_slider.setValue(max(0, min(100, opacity_default)))
        self.opacity_value_label = QtWidgets.QLabel(
            f"{self.opacity_slider.value()}%")
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_value_label.setText(f"{int(v)}%")
        )

        self.stable_hsv_checkbox = QtWidgets.QCheckBox(
            "Stable HSV magnitude across frames"
        )
        self.stable_hsv_checkbox.setChecked(bool(default_stable_hsv))

        # Farneback params
        self.farneback_group = QtWidgets.QGroupBox("Farneback parameters")
        farneback_form = QtWidgets.QFormLayout(self.farneback_group)

        self.pyr_scale_spin = QtWidgets.QDoubleSpinBox()
        self.pyr_scale_spin.setRange(0.05, 0.99)
        self.pyr_scale_spin.setSingleStep(0.05)
        self.pyr_scale_spin.setValue(float(default_pyr_scale or 0.5))

        self.levels_spin = QtWidgets.QSpinBox()
        self.levels_spin.setRange(1, 10)
        self.levels_spin.setValue(int(default_levels or 1))

        self.winsize_spin = QtWidgets.QSpinBox()
        self.winsize_spin.setRange(1, 128)
        self.winsize_spin.setSingleStep(2)  # prefer odd sizes
        self.winsize_spin.setValue(max(1, int(default_winsize or 1)))

        self.iterations_spin = QtWidgets.QSpinBox()
        self.iterations_spin.setRange(1, 50)
        self.iterations_spin.setValue(int(default_iterations or 3))

        self.poly_n_spin = QtWidgets.QSpinBox()
        self.poly_n_spin.setRange(3, 15)
        self.poly_n_spin.setSingleStep(2)  # prefer odd sizes
        self.poly_n_spin.setValue(max(3, int(default_poly_n or 3)))

        self.poly_sigma_spin = QtWidgets.QDoubleSpinBox()
        self.poly_sigma_spin.setRange(0.5, 5.0)
        self.poly_sigma_spin.setSingleStep(0.05)
        self.poly_sigma_spin.setValue(float(default_poly_sigma or 1.1))

        farneback_form.addRow(QtWidgets.QLabel(
            "Pyramid scale:"), self.pyr_scale_spin)
        farneback_form.addRow(QtWidgets.QLabel("Levels:"), self.levels_spin)
        farneback_form.addRow(QtWidgets.QLabel(
            "Window size:"), self.winsize_spin)
        farneback_form.addRow(QtWidgets.QLabel(
            "Iterations:"), self.iterations_spin)
        farneback_form.addRow(QtWidgets.QLabel("Poly N:"), self.poly_n_spin)
        farneback_form.addRow(QtWidgets.QLabel(
            "Poly Sigma:"), self.poly_sigma_spin)

        self.quiver_group = QtWidgets.QGroupBox("Quiver options")
        quiver_form = QtWidgets.QFormLayout(self.quiver_group)
        self.quiver_step_spin = QtWidgets.QSpinBox()
        self.quiver_step_spin.setRange(4, 64)
        self.quiver_step_spin.setSingleStep(2)
        self.quiver_step_spin.setValue(int(default_quiver_step or 16))
        self.quiver_gain_spin = QtWidgets.QDoubleSpinBox()
        self.quiver_gain_spin.setRange(0.1, 10.0)
        self.quiver_gain_spin.setSingleStep(0.1)
        self.quiver_gain_spin.setValue(float(default_quiver_gain or 1.0))
        quiver_form.addRow(QtWidgets.QLabel(
            "Step (density):"), self.quiver_step_spin)
        quiver_form.addRow(QtWidgets.QLabel(
            "Gain (arrow length):"), self.quiver_gain_spin)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow(backend_label, self.backend_combo)
        form_layout.addRow(raft_label, self.raft_combo)
        form_layout.addRow(viz_label, self.viz_combo)
        opacity_layout = QtWidgets.QHBoxLayout()
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_value_label)
        form_layout.addRow(opacity_label, opacity_layout)
        form_layout.addRow(self.stable_hsv_checkbox)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form_layout)
        layout.addWidget(self.farneback_group)
        layout.addWidget(self.quiver_group)
        layout.addWidget(buttons)
        self.setLayout(layout)

        self.viz_combo.currentIndexChanged.connect(self._update_enabled_states)
        self.backend_combo.currentIndexChanged.connect(
            self._update_enabled_states)
        self._update_enabled_states()

    def _update_enabled_states(self) -> None:
        is_quiver = self.viz_combo.currentText().strip().lower() == "quiver"
        self.quiver_group.setEnabled(is_quiver)
        self.stable_hsv_checkbox.setEnabled(not is_quiver)
        is_raft = "raft" in self.backend_combo.currentText().strip().lower()
        self.raft_combo.setEnabled(is_raft)
        self.farneback_group.setEnabled(not is_raft)

    def _backend_value(self) -> str:
        text = self.backend_combo.currentText().strip().lower()
        if "raft" in text:
            return "raft"
        if "torch" in text:
            return "farneback_torch"
        return "farneback"

    def values(self) -> Optional[Tuple[str, str, str, int, int, float, bool, float, int, int, int, int, float]]:
        if self.result() != QtWidgets.QDialog.Accepted:
            return None
        backend = self._backend_value()
        raft_model = self.raft_combo.currentText().strip().lower()
        viz = self.viz_combo.currentText().strip().lower()
        opacity = int(self.opacity_slider.value())
        quiver_step = int(self.quiver_step_spin.value())
        quiver_gain = float(self.quiver_gain_spin.value())
        stable_hsv = bool(self.stable_hsv_checkbox.isChecked())
        pyr_scale = float(self.pyr_scale_spin.value())
        levels = int(self.levels_spin.value())
        winsize = int(self.winsize_spin.value())
        iterations = int(self.iterations_spin.value())
        poly_n = int(self.poly_n_spin.value())
        poly_sigma = float(self.poly_sigma_spin.value())
        return (
            backend,
            raft_model,
            viz,
            opacity,
            quiver_step,
            quiver_gain,
            stable_hsv,
            pyr_scale,
            levels,
            winsize,
            iterations,
            poly_n,
            poly_sigma,
        )
