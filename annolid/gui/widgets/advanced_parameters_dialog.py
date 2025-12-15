from typing import Dict, Optional

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QTabWidget,
    QWidget,
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
        self.setMinimumWidth(520)

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
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        intro_label = QLabel(
            "Tune segmentation, tracking, and SAM3 agent behavior. "
            "Tooltips and helper text explain how each option affects the workflow."
        )
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet(
            "color: palette(window-text); font-weight: 500;")

        tabs = QTabWidget()
        tabs.setTabBarAutoHide(False)
        tabs.addTab(self._build_segmentation_tab(), "Segmentation")
        tabs.addTab(self._build_tracker_tab(), "Tracker")
        tabs.addTab(self._build_sam3_tab(), "SAM3 + Agent")

        layout.addWidget(intro_label)
        layout.addWidget(tabs)

        button_box = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _build_segmentation_tab(self) -> QWidget:
        tab = QWidget()
        group_layout = QVBoxLayout(tab)
        group_layout.setContentsMargins(12, 10, 12, 12)
        group_layout.setSpacing(10)

        helper = QLabel(
            "Quick knobs for polygon smoothing, runtime safety nets, and optical flow defaults."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: palette(window-text); font-size: 11px;")
        group_layout.addWidget(helper)

        form = self._make_form_layout()

        self.epsilon_spinbox = QDoubleSpinBox()
        self.epsilon_spinbox.setRange(0.0, 10.0)
        self.epsilon_spinbox.setSingleStep(0.1)
        self.epsilon_spinbox.setValue(self.epsilon_value)
        self.epsilon_spinbox.setToolTip(
            "Controls polygon simplification. Higher values create smoother, simpler masks."
        )

        self.t_max_spinbox = QSpinBox()
        self.t_max_spinbox.setRange(5, 20)
        self.t_max_spinbox.setSingleStep(1)
        self.t_max_spinbox.setValue(self.t_max_value)
        self.t_max_spinbox.setToolTip(
            "Upper bound used during polygon approximation (default 5–20)."
        )

        self.automatic_pause_checkbox = QCheckBox(
            "Automatic Pause on Error Detection")
        self.automatic_pause_checkbox.setChecked(self.automatic_pause_enabled)
        self.automatic_pause_checkbox.setToolTip(
            "Pause processing automatically if the app detects unexpected errors."
        )

        self.cpu_only_checkbox = QCheckBox("Use CPU Only")
        self.cpu_only_checkbox.setChecked(self.cpu_only_enabled)
        self.cpu_only_checkbox.setToolTip(
            "Force CPU-only processing. Use when GPUs are unavailable or unstable."
        )

        self.save_video_with_color_mask_checkbox = QCheckBox(
            "Save Video with Color Mask"
        )
        self.save_video_with_color_mask_checkbox.setChecked(
            self.save_video_with_color_mask
        )
        self.save_video_with_color_mask_checkbox.setToolTip(
            "When enabled, the exported video includes the overlay mask colors."
        )

        self.optical_flow_backend_combo = QComboBox()
        self.optical_flow_backend_combo.addItems(
            ["farneback (default)", "raft (torchvision)"])
        self.optical_flow_backend_combo.setCurrentIndex(
            1 if self.optical_flow_backend == "raft" else 0)
        self.optical_flow_backend_combo.setToolTip(
            "RAFT offers higher quality flow (requires TorchVision); Farneback is lighter."
        )

        self.auto_recovery_missing_instances_checkbox = QCheckBox(
            "Auto Recovery of Missing Instances"
        )
        self.auto_recovery_missing_instances_checkbox.setChecked(
            self.auto_recovery_missing_instances
        )
        self.auto_recovery_missing_instances_checkbox.setToolTip(
            "Try to reintroduce instances that disappear unexpectedly."
        )

        self.compute_optical_flow_checkbox = QCheckBox(
            "Compute Motion Index based on optical flow over instance mask"
        )
        self.compute_optical_flow_checkbox.setChecked(
            self.compute_optical_flow)
        self.compute_optical_flow_checkbox.setToolTip(
            "Toggle motion index calculation using optical flow across instance masks."
        )

        form.addRow(
            "Polygon epsilon",
            self._wrap_with_hint(
                self.epsilon_spinbox,
                "Higher values simplify shapes more aggressively (default 2.0).",
            ),
        )
        form.addRow(
            "T_max (frames)",
            self._wrap_with_hint(
                self.t_max_spinbox,
                "Caps the segment smoothing window. Keep between 5–20 for stable results.",
            ),
        )
        form.addRow(
            self._wrap_checkbox(
                self.automatic_pause_checkbox,
                "Pause automatically when the system sees an unexpected condition.",
            )
        )
        form.addRow(
            self._wrap_checkbox(
                self.cpu_only_checkbox,
                "Good for laptops/servers without stable GPU support.",
            )
        )
        form.addRow(
            self._wrap_checkbox(
                self.save_video_with_color_mask_checkbox,
                "Export videos with colored overlays baked in.",
            )
        )
        form.addRow(
            "Optical flow backend",
            self._wrap_with_hint(
                self.optical_flow_backend_combo,
                "Use RAFT for highest fidelity, or Farneback for speed and compatibility.",
            ),
        )
        form.addRow(
            self._wrap_checkbox(
                self.auto_recovery_missing_instances_checkbox,
                "Try to bring back instances that momentarily drop out.",
            )
        )
        form.addRow(
            self._wrap_checkbox(
                self.compute_optical_flow_checkbox,
                "If disabled, motion index will not be computed from optical flow.",
            )
        )

        group_layout.addLayout(form)
        group_layout.addStretch(1)
        return tab

    def _build_tracker_tab(self) -> QWidget:
        tab = QWidget()
        tracker_layout = QVBoxLayout(tab)
        tracker_layout.setContentsMargins(12, 10, 12, 12)
        tracker_layout.setSpacing(10)

        helper = QLabel(
            "Tweak how keypoints stay within masks and how motion search behaves frame-to-frame."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: palette(window-text); font-size: 11px;")
        tracker_layout.addWidget(helper)

        tracker_form = self._make_form_layout()

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

        tracker_form.addRow(
            self._wrap_checkbox(
                self.mask_enforce_checkbox,
                "Keeps keypoints snapped to the instance mask to reduce drift.",
            )
        )
        tracker_form.addRow(
            "Mask snap radius (px)",
            self._wrap_with_hint(
                self.mask_enforce_radius_spinbox,
                "Pixels around the mask boundary used when clamping positions.",
            ),
        )
        tracker_form.addRow(
            self._wrap_checkbox(
                self.mask_enforce_reject_checkbox,
                "Discard updates that fall outside of the mask.",
            )
        )
        tracker_form.addRow(
            "Search tighten",
            self._wrap_with_hint(
                self.motion_search_tighten_spinbox,
                "Scale down the search area for more stable motion matching.",
            ),
        )
        tracker_form.addRow(
            "Velocity gain",
            self._wrap_with_hint(
                self.motion_search_gain_spinbox,
                "Influence from previous velocity. Increase to follow faster motion.",
            ),
        )
        tracker_form.addRow(
            "Flow gain",
            self._wrap_with_hint(
                self.motion_search_flow_gain_spinbox,
                "Weight applied to optical-flow based motion.",
            ),
        )
        tracker_form.addRow(
            "Min radius",
            self._wrap_with_hint(
                self.motion_search_min_radius_spinbox,
                "Lower bound for the search radius in pixels.",
            ),
        )
        tracker_form.addRow(
            "Max radius",
            self._wrap_with_hint(
                self.motion_search_max_radius_spinbox,
                "Upper bound for the search radius in pixels.",
            ),
        )
        tracker_form.addRow(
            "Miss boost",
            self._wrap_with_hint(
                self.motion_search_miss_boost_spinbox,
                "Expand the search radius after misses to help recover tracks.",
            ),
        )
        tracker_form.addRow(
            "Motion prior weight",
            self._wrap_with_hint(
                self.motion_prior_penalty_weight_spinbox,
                "Penalty applied to deviations from the motion prior.",
            ),
        )
        tracker_form.addRow(
            "Motion prior soft radius",
            self._wrap_with_hint(
                self.motion_prior_soft_radius_spinbox,
                "Soft radius (px) used before harsher penalties kick in.",
            ),
        )
        tracker_form.addRow(
            "Motion prior radius factor",
            self._wrap_with_hint(
                self.motion_prior_radius_factor_spinbox,
                "Multiplier applied to the prior radius during tracking.",
            ),
        )
        tracker_form.addRow(
            "Motion prior miss relief",
            self._wrap_with_hint(
                self.motion_prior_miss_relief_spinbox,
                "Reduces penalties after misses to allow recovery.",
            ),
        )
        tracker_form.addRow(
            "Flow relief",
            self._wrap_with_hint(
                self.motion_prior_flow_relief_spinbox,
                "Penalty relief driven by optical flow confidence.",
            ),
        )

        tracker_layout.addLayout(tracker_form)
        tracker_layout.addStretch(1)
        return tab

    def _build_sam3_tab(self) -> QWidget:
        """Controls specific to SAM3 tracking and agent seeding."""
        tab = QWidget()
        group_layout = QVBoxLayout(tab)
        group_layout.setContentsMargins(12, 10, 12, 12)
        group_layout.setSpacing(10)

        helper = QLabel(
            "Refine how SAM3 propagates masks and how the agent seeds new corrections."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: palette(window-text); font-size: 11px;")
        group_layout.addWidget(helper)

        form = self._make_form_layout()

        self.sam3_score_thresh_spinbox = QDoubleSpinBox()
        self.sam3_score_thresh_spinbox.setRange(0.0, 1.0)
        self.sam3_score_thresh_spinbox.setSingleStep(0.01)
        self.sam3_score_thresh_spinbox.setValue(
            self.sam3_score_threshold_detection)
        self.sam3_score_thresh_spinbox.setToolTip(
            "Score threshold for SAM3 detections. Lower to accept more masks."
        )

        self.sam3_new_det_thresh_spinbox = QDoubleSpinBox()
        self.sam3_new_det_thresh_spinbox.setRange(0.0, 1.0)
        self.sam3_new_det_thresh_spinbox.setSingleStep(0.01)
        self.sam3_new_det_thresh_spinbox.setValue(self.sam3_new_det_thresh)
        self.sam3_new_det_thresh_spinbox.setToolTip(
            "Threshold for adding new detections during propagation."
        )

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
            "Score threshold (SAM3_SCORE_THRESH)",
            self._wrap_with_hint(
                self.sam3_score_thresh_spinbox,
                "Main detection threshold used by SAM3.",
            ),
        )
        form.addRow(
            "New detection threshold (SAM3_NEW_DET_THRESH)",
            self._wrap_with_hint(
                self.sam3_new_det_thresh_spinbox,
                "Controls how easily new detections are added mid-sequence.",
            ),
        )
        form.addRow(
            "Propagation direction",
            self._wrap_with_hint(
                self.sam3_direction_combo,
                "Run SAM3 forward, backward, or both directions for better coverage.",
            ),
        )
        form.addRow(
            "Max frames to track (0=auto)",
            self._wrap_with_hint(
                self.sam3_max_frames_spinbox,
                "Limit how many frames SAM3 will attempt to track. Leave on auto unless troubleshooting.",
            ),
        )
        form.addRow(
            "Device override",
            self._wrap_with_hint(
                self.sam3_device_lineedit,
                "Force a compute device (cuda, mps, cpu). Empty uses the best available.",
            ),
        )
        form.addRow(
            "Sliding window size",
            self._wrap_with_hint(
                self.sam3_window_size_spinbox,
                "Frames per propagation window. Smaller values reduce memory needs.",
            ),
        )
        form.addRow(
            "Sliding window stride (0=auto)",
            self._wrap_with_hint(
                self.sam3_window_stride_spinbox,
                "Stride between windows; 0 matches the window size.",
            ),
        )
        form.addRow(
            "Agent det threshold",
            self._wrap_with_hint(
                self.sam3_agent_det_thresh_spinbox,
                "Filter out low-confidence masks generated by the agent.",
            ),
        )
        form.addRow(
            "Agent window size",
            self._wrap_with_hint(
                self.sam3_agent_window_size_spinbox,
                "Frames per agent window; first frame gets corrections.",
            ),
        )
        form.addRow(
            "Agent window stride (0=auto)",
            self._wrap_with_hint(
                self.sam3_agent_stride_spinbox,
                "Stride between agent windows; 0 matches the window size.",
            ),
        )
        form.addRow(
            "Agent output dir",
            self._wrap_with_hint(
                self.sam3_agent_output_dir_lineedit,
                "Where agent JSON/PNG outputs are written.",
            ),
        )

        group_layout.addLayout(form)
        group_layout.addStretch(1)
        return tab

    def _make_form_layout(self) -> QFormLayout:
        form = QFormLayout()
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        form.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        return form

    def _wrap_with_hint(self, control: QWidget, hint: str) -> QWidget:
        """Stack a control with small helper text to keep the form tidy."""
        wrapper = QWidget()
        column = QVBoxLayout(wrapper)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(4 if hint else 0)
        column.addWidget(control)

        if hint:
            hint_label = QLabel(hint)
            hint_label.setWordWrap(True)
            hint_label.setStyleSheet(
                "color: palette(window-text); font-size: 11px;")
            column.addWidget(hint_label)

        return wrapper

    def _wrap_checkbox(self, checkbox: QCheckBox, hint: str) -> QWidget:
        return self._wrap_with_hint(checkbox, hint)

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
