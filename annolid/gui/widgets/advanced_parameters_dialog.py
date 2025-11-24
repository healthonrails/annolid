from typing import Dict, Optional

from qtpy.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QLabel,
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
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Parameters")

        self._tracker_config = tracker_config or CutieDinoTrackerConfig()
        self._tracker_settings: Dict[str, object] = {}

        self.epsilon_value = 2.0
        self.t_max_value = 5
        self.automatic_pause_enabled = False
        self.cpu_only_enabled = False
        self.save_video_with_color_mask = False
        self.auto_recovery_missing_instances = False
        self.compute_optical_flow = True
        self.sam3_score_threshold_detection = 0.05
        self.sam3_new_det_thresh = 0.06

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
        """Controls specific to SAM3 detector thresholds."""
        sam3_group = QGroupBox("SAM3 detection thresholds")
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

        form.addRow(
            "Score threshold (SAM3_SCORE_THRESH)", self.sam3_score_thresh_spinbox
        )
        form.addRow(
            "New detection threshold (SAM3_NEW_DET_THRESH)",
            self.sam3_new_det_thresh_spinbox,
        )

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

    def get_tracker_settings(self) -> Dict[str, object]:
        return dict(self._tracker_settings)

    def get_sam3_thresholds(self) -> Dict[str, float]:
        return {
            "score_threshold_detection": float(self.sam3_score_threshold_detection),
            "new_det_thresh": float(self.sam3_new_det_thresh),
        }
