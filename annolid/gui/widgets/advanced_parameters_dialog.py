from typing import Dict, Optional

from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from annolid.tracking.configuration import CutieDinoTrackerConfig


class AdvancedParametersDialog(QDialog):
    """Dialog exposing advanced segmentation and tracker controls."""

    def __init__(
        self,
        parent=None,
        tracker_config: Optional[CutieDinoTrackerConfig] = None,
        sam3_runtime: Optional[Dict[str, object]] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Parameters")

        self._tracker_config = tracker_config or CutieDinoTrackerConfig()
        self._tracker_settings: Dict[str, object] = {}
        sam3_runtime = sam3_runtime or {}

        self.epsilon_value = 2.0
        self.t_max_value = 5
        self.automatic_pause_enabled = False
        self.cpu_only_enabled = False
        self.save_video_with_color_mask = False
        self.auto_recovery_missing_instances = False
        self.compute_optical_flow = True
        self.optical_flow_backend = "farneback"
        self.optical_flow_backend = "farneback"
        self.sam3_score_threshold_detection = float(
            sam3_runtime.get("score_threshold_detection") or 0.35
        )
        self.sam3_new_det_thresh = float(
            sam3_runtime.get("new_det_thresh") or 0.25
        )
        self.sam3_propagation_direction = str(
            sam3_runtime.get("propagation_direction") or "both"
        )
        self.sam3_max_frame_num_to_track = sam3_runtime.get(
            "max_frame_num_to_track", None
        )
        self.sam3_device_override = (
            "" if sam3_runtime.get("device") is None else str(
                sam3_runtime.get("device"))
        )
        self.sam3_sliding_window_size = int(
            sam3_runtime.get("sliding_window_size") or 5
        )
        self.sam3_sliding_window_stride = sam3_runtime.get(
            "sliding_window_stride", None
        )
        # Agent seeding defaults
        self.sam3_agent_det_thresh = float(
            sam3_runtime.get(
                "agent_det_thresh") or self.sam3_score_threshold_detection
        )
        self.sam3_agent_window_size = int(
            sam3_runtime.get(
                "agent_window_size") or self.sam3_sliding_window_size
        )
        self.sam3_agent_stride = sam3_runtime.get("agent_stride", None)
        self.sam3_agent_output_dir = str(
            sam3_runtime.get("agent_output_dir") or "sam3_agent_out"
        )

        layout = QVBoxLayout()
        layout.setSpacing(8)

        layout.addLayout(self._build_polygon_controls())
        layout.addWidget(self._build_tracker_group())
        layout.addWidget(self._build_sam3_group())

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _build_polygon_controls(self) -> QFormLayout:
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)

        epsilon_label = QLabel(
            "Epsilon Value Selection for Polygon Approximation (default 2.0)"
        )
        epsilon_label.setWordWrap(True)
        self.epsilon_spinbox = QDoubleSpinBox()
        self.epsilon_spinbox.setRange(0.0, 10.0)
        self.epsilon_spinbox.setSingleStep(0.1)
        self.epsilon_spinbox.setValue(self.epsilon_value)

        t_max_label = QLabel("T_max Value Selection (5 to 20)")
        self.t_max_spinbox = QSpinBox()
        self.t_max_spinbox.setRange(5, 20)
        self.t_max_spinbox.setSingleStep(1)
        self.t_max_spinbox.setValue(self.t_max_value)

        self.automatic_pause_checkbox = QCheckBox(
            "Automatic Pause on Error Detection")
        self.automatic_pause_checkbox.setChecked(self.automatic_pause_enabled)

        self.cpu_only_checkbox = QCheckBox("Use CPU Only")
        self.cpu_only_checkbox.setChecked(self.cpu_only_enabled)

        self.save_video_with_color_mask_checkbox = QCheckBox(
            "Save Video with Color Mask"
        )
        self.save_video_with_color_mask_checkbox.setChecked(
            self.save_video_with_color_mask
        )

        self.optical_flow_backend_label = QLabel("Optical Flow Backend")
        self.optical_flow_backend_combo = QComboBox()
        self.optical_flow_backend_combo.addItems(
            ["farneback (default)", "raft (torchvision)"])
        self.optical_flow_backend_combo.setCurrentIndex(
            1 if self.optical_flow_backend == "raft" else 0)

        self.auto_recovery_missing_instances_checkbox = QCheckBox(
            "Auto Recovery of Missing Instances"
        )
        self.auto_recovery_missing_instances_checkbox.setChecked(
            self.auto_recovery_missing_instances
        )

        self.compute_optical_flow_checkbox = QCheckBox(
            "Compute Motion Index based on optical flow over instance mask"
        )
        self.compute_optical_flow_checkbox.setChecked(
            self.compute_optical_flow)

        form.addRow(epsilon_label)
        form.addRow(self.epsilon_spinbox)
        form.addRow(t_max_label)
        form.addRow(self.t_max_spinbox)
        form.addRow(self.automatic_pause_checkbox)
        form.addRow(self.cpu_only_checkbox)
        form.addRow(self.save_video_with_color_mask_checkbox)
        form.addRow(self.optical_flow_backend_label,
                    self.optical_flow_backend_combo)
        form.addRow(self.auto_recovery_missing_instances_checkbox)
        form.addRow(self.compute_optical_flow_checkbox)
        return form

    def _build_tracker_group(self) -> QGroupBox:
        tracker_group = QGroupBox("Cutie + DINO tracker")
        tracker_form = QFormLayout()
        tracker_form.setHorizontalSpacing(12)
        tracker_form.setVerticalSpacing(6)

        self.mask_enforce_checkbox = QCheckBox(
            "Clamp keypoints to instance mask")
        self.mask_enforce_checkbox.setChecked(
            bool(self._tracker_config.mask_enforce_position)
        )

        self.mask_enforce_radius_spinbox = QSpinBox()
        self.mask_enforce_radius_spinbox.setRange(1, 256)
        self.mask_enforce_radius_spinbox.setSingleStep(1)
        self.mask_enforce_radius_spinbox.setValue(
            int(self._tracker_config.mask_enforce_snap_radius)
        )

        self.mask_enforce_reject_checkbox = QCheckBox(
            "Reject updates outside the mask"
        )
        self.mask_enforce_reject_checkbox.setChecked(
            bool(self._tracker_config.mask_enforce_reject_outside)
        )

        self.motion_search_tighten_spinbox = QDoubleSpinBox()
        self.motion_search_tighten_spinbox.setRange(0.1, 2.0)
        self.motion_search_tighten_spinbox.setSingleStep(0.05)
        self.motion_search_tighten_spinbox.setValue(
            float(self._tracker_config.motion_search_tighten)
        )

        self.motion_search_gain_spinbox = QDoubleSpinBox()
        self.motion_search_gain_spinbox.setRange(0.0, 3.0)
        self.motion_search_gain_spinbox.setSingleStep(0.05)
        self.motion_search_gain_spinbox.setValue(
            float(self._tracker_config.motion_search_gain)
        )

        self.motion_search_flow_gain_spinbox = QDoubleSpinBox()
        self.motion_search_flow_gain_spinbox.setRange(0.0, 3.0)
        self.motion_search_flow_gain_spinbox.setSingleStep(0.05)
        self.motion_search_flow_gain_spinbox.setValue(
            float(self._tracker_config.motion_search_flow_gain)
        )

        self.motion_search_min_radius_spinbox = QDoubleSpinBox()
        self.motion_search_min_radius_spinbox.setRange(0.5, 32.0)
        self.motion_search_min_radius_spinbox.setSingleStep(0.5)
        self.motion_search_min_radius_spinbox.setValue(
            float(self._tracker_config.motion_search_min_radius)
        )

        self.motion_search_max_radius_spinbox = QDoubleSpinBox()
        self.motion_search_max_radius_spinbox.setRange(1.0, 64.0)
        self.motion_search_max_radius_spinbox.setSingleStep(0.5)
        self.motion_search_max_radius_spinbox.setValue(
            float(self._tracker_config.motion_search_max_radius)
        )

        self.motion_search_miss_boost_spinbox = QDoubleSpinBox()
        self.motion_search_miss_boost_spinbox.setRange(0.0, 4.0)
        self.motion_search_miss_boost_spinbox.setSingleStep(0.1)
        self.motion_search_miss_boost_spinbox.setValue(
            float(self._tracker_config.motion_search_miss_boost)
        )

        self.motion_prior_penalty_weight_spinbox = QDoubleSpinBox()
        self.motion_prior_penalty_weight_spinbox.setRange(0.0, 2.0)
        self.motion_prior_penalty_weight_spinbox.setSingleStep(0.05)
        self.motion_prior_penalty_weight_spinbox.setValue(
            float(self._tracker_config.motion_prior_penalty_weight)
        )

        self.motion_prior_soft_radius_spinbox = QDoubleSpinBox()
        self.motion_prior_soft_radius_spinbox.setRange(1.0, 256.0)
        self.motion_prior_soft_radius_spinbox.setSingleStep(1.0)
        self.motion_prior_soft_radius_spinbox.setValue(
            float(self._tracker_config.motion_prior_soft_radius_px)
        )

        self.motion_prior_radius_factor_spinbox = QDoubleSpinBox()
        self.motion_prior_radius_factor_spinbox.setRange(1.0, 4.0)
        self.motion_prior_radius_factor_spinbox.setSingleStep(0.1)
        self.motion_prior_radius_factor_spinbox.setValue(
            float(self._tracker_config.motion_prior_radius_factor)
        )

        self.motion_prior_miss_relief_spinbox = QDoubleSpinBox()
        self.motion_prior_miss_relief_spinbox.setRange(0.0, 4.0)
        self.motion_prior_miss_relief_spinbox.setSingleStep(0.1)
        self.motion_prior_miss_relief_spinbox.setValue(
            float(self._tracker_config.motion_prior_miss_relief)
        )

        self.motion_prior_flow_relief_spinbox = QDoubleSpinBox()
        self.motion_prior_flow_relief_spinbox.setRange(0.0, 2.0)
        self.motion_prior_flow_relief_spinbox.setSingleStep(0.05)
        self.motion_prior_flow_relief_spinbox.setValue(
            float(self._tracker_config.motion_prior_flow_relief)
        )

        tracker_form.addRow(self.mask_enforce_checkbox)
        tracker_form.addRow("Mask snap radius (px)",
                            self.mask_enforce_radius_spinbox)
        tracker_form.addRow(self.mask_enforce_reject_checkbox)
        tracker_form.addRow(
            "Search tighten", self.motion_search_tighten_spinbox)
        tracker_form.addRow("Velocity gain", self.motion_search_gain_spinbox)
        tracker_form.addRow("Flow gain", self.motion_search_flow_gain_spinbox)
        tracker_form.addRow(
            "Min radius", self.motion_search_min_radius_spinbox)
        tracker_form.addRow(
            "Max radius", self.motion_search_max_radius_spinbox)
        tracker_form.addRow(
            "Miss boost", self.motion_search_miss_boost_spinbox)
        tracker_form.addRow("Motion prior weight",
                            self.motion_prior_penalty_weight_spinbox)
        tracker_form.addRow("Motion prior soft radius",
                            self.motion_prior_soft_radius_spinbox)
        tracker_form.addRow("Motion prior radius factor",
                            self.motion_prior_radius_factor_spinbox)
        tracker_form.addRow("Motion prior miss relief",
                            self.motion_prior_miss_relief_spinbox)
        tracker_form.addRow(
            "Flow relief", self.motion_prior_flow_relief_spinbox)

        tracker_group.setLayout(tracker_form)
        return tracker_group

    def _build_sam3_group(self) -> QGroupBox:
        """Controls specific to SAM3 tracking and agent seeding."""
        sam3_group = QGroupBox("SAM3 tracking + agent")
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)

        self.sam3_score_thresh_spinbox = QDoubleSpinBox()
        self.sam3_score_thresh_spinbox.setRange(0.0, 1.0)
        self.sam3_score_thresh_spinbox.setSingleStep(0.01)
        self.sam3_score_thresh_spinbox.setValue(
            self.sam3_score_threshold_detection)

        self.sam3_new_det_thresh_spinbox = QDoubleSpinBox()
        self.sam3_new_det_thresh_spinbox.setRange(0.0, 1.0)
        self.sam3_new_det_thresh_spinbox.setSingleStep(0.01)
        self.sam3_new_det_thresh_spinbox.setValue(self.sam3_new_det_thresh)

        self.sam3_direction_combo = QComboBox()
        self.sam3_direction_combo.addItems(["forward", "backward", "both"])
        idx = self.sam3_direction_combo.findText(
            self.sam3_propagation_direction)
        if idx >= 0:
            self.sam3_direction_combo.setCurrentIndex(idx)

        self.sam3_max_frames_spinbox = QSpinBox()
        self.sam3_max_frames_spinbox.setRange(0, 100000)
        self.sam3_max_frames_spinbox.setSpecialValueText("auto")
        self.sam3_max_frames_spinbox.setValue(
            0 if self.sam3_max_frame_num_to_track in (
                None, "") else int(self.sam3_max_frame_num_to_track)
        )

        self.sam3_device_lineedit = QLineEdit(self.sam3_device_override)
        self.sam3_device_lineedit.setPlaceholderText("auto (cuda/mps/cpu)")
        self.sam3_device_lineedit.setToolTip(
            "Optional device override for SAM3 tracker (e.g., cuda, mps, cpu). Leave empty for auto."
        )

        self.sam3_window_size_spinbox = QSpinBox()
        self.sam3_window_size_spinbox.setRange(1, 1000)
        self.sam3_window_size_spinbox.setValue(self.sam3_sliding_window_size)
        self.sam3_window_size_spinbox.setToolTip(
            "Sliding-window size for low-RAM text propagation."
        )

        self.sam3_window_stride_spinbox = QSpinBox()
        self.sam3_window_stride_spinbox.setRange(0, 1000)
        self.sam3_window_stride_spinbox.setSpecialValueText("auto")
        self.sam3_window_stride_spinbox.setValue(
            0 if self.sam3_sliding_window_stride in (
                None, "") else int(self.sam3_sliding_window_stride)
        )
        self.sam3_window_stride_spinbox.setToolTip(
            "Stride between windows; 0 uses window size."
        )

        # Agent seeding controls
        self.sam3_agent_det_thresh_spinbox = QDoubleSpinBox()
        self.sam3_agent_det_thresh_spinbox.setRange(0.0, 1.0)
        self.sam3_agent_det_thresh_spinbox.setSingleStep(0.01)
        self.sam3_agent_det_thresh_spinbox.setValue(self.sam3_agent_det_thresh)
        self.sam3_agent_det_thresh_spinbox.setToolTip(
            "Filter agent-generated masks below this score before seeding tracker."
        )

        self.sam3_agent_window_size_spinbox = QSpinBox()
        self.sam3_agent_window_size_spinbox.setRange(1, 1000)
        self.sam3_agent_window_size_spinbox.setValue(
            self.sam3_agent_window_size)
        self.sam3_agent_window_size_spinbox.setToolTip(
            "Frames per agent window; first frame of each window is corrected by the agent."
        )

        self.sam3_agent_stride_spinbox = QSpinBox()
        self.sam3_agent_stride_spinbox.setRange(0, 1000)
        self.sam3_agent_stride_spinbox.setSpecialValueText("auto")
        self.sam3_agent_stride_spinbox.setValue(
            0 if self.sam3_agent_stride in (
                None, "") else int(self.sam3_agent_stride)
        )
        self.sam3_agent_stride_spinbox.setToolTip(
            "Stride between agent windows; 0 uses agent window size."
        )

        self.sam3_agent_output_dir_lineedit = QLineEdit(
            self.sam3_agent_output_dir)
        self.sam3_agent_output_dir_lineedit.setPlaceholderText(
            "sam3_agent_out")
        self.sam3_agent_output_dir_lineedit.setToolTip(
            "Directory for agent JSON/PNG outputs."
        )

        form.addRow(
            "Score threshold (SAM3_SCORE_THRESH)", self.sam3_score_thresh_spinbox
        )
        form.addRow(
            "New detection threshold (SAM3_NEW_DET_THRESH)",
            self.sam3_new_det_thresh_spinbox,
        )
        form.addRow("Propagation direction", self.sam3_direction_combo)
        form.addRow("Max frames to track (0=auto)",
                    self.sam3_max_frames_spinbox)
        form.addRow("Device override", self.sam3_device_lineedit)
        form.addRow("Sliding window size", self.sam3_window_size_spinbox)
        form.addRow("Sliding window stride (0=auto)",
                    self.sam3_window_stride_spinbox)
        form.addRow("Agent det threshold", self.sam3_agent_det_thresh_spinbox)
        form.addRow("Agent window size", self.sam3_agent_window_size_spinbox)
        form.addRow("Agent window stride (0=auto)",
                    self.sam3_agent_stride_spinbox)
        form.addRow("Agent output dir", self.sam3_agent_output_dir_lineedit)

        sam3_group.setLayout(form)
        return sam3_group

    def accept(self) -> None:  # pragma: no cover - dialog delegate
        self._collect_values()
        super().accept()

    def _collect_values(self) -> None:
        self.epsilon_value = self.epsilon_spinbox.value()
        self.t_max_value = self.t_max_spinbox.value()
        self.automatic_pause_enabled = self.automatic_pause_checkbox.isChecked()
        self.cpu_only_enabled = self.cpu_only_checkbox.isChecked()
        self.save_video_with_color_mask = (
            self.save_video_with_color_mask_checkbox.isChecked()
        )
        self.auto_recovery_missing_instances = (
            self.auto_recovery_missing_instances_checkbox.isChecked()
        )
        self.compute_optical_flow = self.compute_optical_flow_checkbox.isChecked()
        self.optical_flow_backend = self.get_optical_flow_backend()

        snap_radius = self.mask_enforce_radius_spinbox.value()
        self._tracker_settings = {
            "mask_enforce_position": self.mask_enforce_checkbox.isChecked(),
            "mask_enforce_search_radius": snap_radius,
            "mask_enforce_snap_radius": snap_radius,
            "mask_enforce_reject_outside": self.mask_enforce_reject_checkbox.isChecked(),
            "motion_search_tighten": self.motion_search_tighten_spinbox.value(),
            "motion_search_gain": self.motion_search_gain_spinbox.value(),
            "motion_search_flow_gain": self.motion_search_flow_gain_spinbox.value(),
            "motion_search_min_radius": self.motion_search_min_radius_spinbox.value(),
            "motion_search_max_radius": self.motion_search_max_radius_spinbox.value(),
            "motion_search_miss_boost": self.motion_search_miss_boost_spinbox.value(),
            "motion_prior_penalty_weight": self.motion_prior_penalty_weight_spinbox.value(),
            "motion_prior_soft_radius_px": self.motion_prior_soft_radius_spinbox.value(),
            "motion_prior_radius_factor": self.motion_prior_radius_factor_spinbox.value(),
            "motion_prior_miss_relief": self.motion_prior_miss_relief_spinbox.value(),
            "motion_prior_flow_relief": self.motion_prior_flow_relief_spinbox.value(),
        }
        self.sam3_score_threshold_detection = self.sam3_score_thresh_spinbox.value()
        self.sam3_new_det_thresh = self.sam3_new_det_thresh_spinbox.value()
        self.sam3_propagation_direction = str(
            self.sam3_direction_combo.currentText()
        )
        self.sam3_max_frame_num_to_track = (
            None
            if self.sam3_max_frames_spinbox.value() == 0
            else int(self.sam3_max_frames_spinbox.value())
        )
        self.sam3_device_override = self.sam3_device_lineedit.text().strip()
        if self.sam3_device_override == "":
            self.sam3_device_override = None
        self.sam3_sliding_window_size = int(
            self.sam3_window_size_spinbox.value())
        self.sam3_sliding_window_stride = (
            None
            if self.sam3_window_stride_spinbox.value() == 0
            else int(self.sam3_window_stride_spinbox.value())
        )
        self.sam3_agent_det_thresh = self.sam3_agent_det_thresh_spinbox.value()
        self.sam3_agent_window_size = int(
            self.sam3_agent_window_size_spinbox.value()
        )
        self.sam3_agent_stride = (
            None
            if self.sam3_agent_stride_spinbox.value() == 0
            else int(self.sam3_agent_stride_spinbox.value())
        )
        self.sam3_agent_output_dir = (
            self.sam3_agent_output_dir_lineedit.text().strip()
            or "sam3_agent_out"
        )

    def get_epsilon_value(self) -> float:
        return self.epsilon_value

    def get_t_max_value(self) -> int:
        return int(self.t_max_value)

    def is_automatic_pause_enabled(self) -> bool:
        return self.automatic_pause_enabled

    def is_cpu_only_enabled(self) -> bool:
        return self.cpu_only_enabled

    def is_save_video_with_color_mask_enabled(self) -> bool:
        return self.save_video_with_color_mask

    def is_auto_recovery_missing_instances_enabled(self) -> bool:
        return self.auto_recovery_missing_instances

    def is_compute_optiocal_flow_enabled(self) -> bool:
        return self.compute_optical_flow

    def get_optical_flow_backend(self) -> str:
        text = self.optical_flow_backend_combo.currentText().lower()
        return "raft" if "raft" in text else "farneback"

    def get_tracker_settings(self) -> Dict[str, object]:
        return dict(self._tracker_settings)

    def get_sam3_thresholds(self) -> Dict[str, float]:
        return {
            "score_threshold_detection": float(self.sam3_score_threshold_detection),
            "new_det_thresh": float(self.sam3_new_det_thresh),
        }

    def get_sam3_runtime_settings(self) -> Dict[str, object]:
        return {
            "propagation_direction": self.sam3_propagation_direction,
            "max_frame_num_to_track": self.sam3_max_frame_num_to_track,
            "device": self.sam3_device_override,
            "sliding_window_size": self.sam3_sliding_window_size,
            "sliding_window_stride": self.sam3_sliding_window_stride,
            "agent_det_thresh": self.sam3_agent_det_thresh,
            "agent_window_size": self.sam3_agent_window_size,
            "agent_stride": self.sam3_agent_stride,
            "agent_output_dir": self.sam3_agent_output_dir,
        }

    def get_optical_flow_backend(self) -> str:
        text = self.optical_flow_backend_combo.currentText().lower()
        return "raft" if "raft" in text else "farneback"
