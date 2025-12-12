from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Union

from qtpy import QtCore, QtWidgets


class FlowOptionsDialog(QtWidgets.QDialog):
    """Simple dialog to pick optical-flow backend, visualization, and NDJSON path."""

    def __init__(
        self,
        parent=None,
        *,
        default_backend: str = "farneback",
        default_raft_model: str = "small",
        default_viz: str = "quiver",
        default_ndjson: Optional[str] = None,
        default_opacity: Union[int, float] = 70,
        default_quiver_step: int = 16,
        default_quiver_gain: Union[int, float] = 1.0,
        default_stable_hsv: bool = True,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Optical Flow Options")

        backend_label = QtWidgets.QLabel("Flow backend:")
        self.backend_combo = QtWidgets.QComboBox()
        self.backend_combo.addItems(["farneback", "raft"])
        backend_val = str(default_backend).lower(
        ) if default_backend is not None else "farneback"
        backend_idx = 1 if backend_val == "raft" else 0
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

        ndjson_label = QtWidgets.QLabel("Save NDJSON to:")
        self.ndjson_edit = QtWidgets.QLineEdit()
        if default_ndjson:
            self.ndjson_edit.setText(default_ndjson)
        browse_btn = QtWidgets.QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_ndjson)

        form_layout = QtWidgets.QFormLayout()
        form_layout.addRow(backend_label, self.backend_combo)
        form_layout.addRow(raft_label, self.raft_combo)
        form_layout.addRow(viz_label, self.viz_combo)
        opacity_layout = QtWidgets.QHBoxLayout()
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_value_label)
        form_layout.addRow(opacity_label, opacity_layout)
        form_layout.addRow(self.stable_hsv_checkbox)

        ndjson_layout = QtWidgets.QHBoxLayout()
        ndjson_layout.addWidget(self.ndjson_edit)
        ndjson_layout.addWidget(browse_btn)
        form_layout.addRow(ndjson_label, ndjson_layout)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QtWidgets.QVBoxLayout()
        layout.addLayout(form_layout)
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
        is_raft = self.backend_combo.currentText().strip().lower() == "raft"
        self.raft_combo.setEnabled(is_raft)

    def _browse_ndjson(self) -> None:
        current = self.ndjson_edit.text().strip()
        start_path = current or str(Path.home() / "flow.ndjson")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            self.tr("Save flow NDJSON"),
            start_path,
            self.tr("NDJSON Files (*.ndjson)"),
        )
        if path:
            self.ndjson_edit.setText(path)

    def values(self) -> Optional[Tuple[str, str, str, str, int, int, float, bool]]:
        if self.result() != QtWidgets.QDialog.Accepted:
            return None
        backend = self.backend_combo.currentText().strip().lower()
        raft_model = self.raft_combo.currentText().strip().lower()
        viz = self.viz_combo.currentText().strip().lower()
        ndjson_path = self.ndjson_edit.text().strip()
        opacity = int(self.opacity_slider.value())
        quiver_step = int(self.quiver_step_spin.value())
        quiver_gain = float(self.quiver_gain_spin.value())
        stable_hsv = bool(self.stable_hsv_checkbox.isChecked())
        return backend, raft_model, viz, ndjson_path, opacity, quiver_step, quiver_gain, stable_hsv
