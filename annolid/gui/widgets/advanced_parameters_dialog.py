from pathlib import Path
from typing import Dict, Optional

from qtpy.QtCore import Qt
from qtpy.QtGui import QGuiApplication
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QWidget,
    QVBoxLayout,
)

from annolid.features.dino_models import (
    check_dino_model_runtime_support,
    supports_matryoshka_truncation,
)
from annolid.gui.models_registry import describe_cutie_checkpoint_status_fast
from annolid.gui.qt_compat import configure_compact_combo
from annolid.tracking.configuration import CutieDinoTrackerConfig
from annolid.gui.widgets.dino_training_ui_shared import (
    configure_dino_model_combo,
    selected_dino_model_identifier,
)
from annolid.motion.optical_flow import (
    OPTICAL_FLOW_BACKEND_LABELS,
    normalize_optical_flow_backend,
    optical_flow_backend_combo_index,
)


class AdvancedParametersDialog(QDialog):
    """Dialog exposing advanced segmentation and tracker controls."""

    _CUSTOM_TRACKER_PRESET = "<custom>"
    _CUTIE_CUSTOM_SENTINEL = "__custom__"
    _SAM31_DEFAULT_MAX_OBJECTS = 16
    _SAM31_DEFAULT_MULTIPLEX_COUNT = 16
    _SAM31_MAX_MULTIPLEX_COUNT = 16

    @staticmethod
    def _coerce_positive_int(
        value: object,
        *,
        default: int,
        maximum: Optional[int] = None,
    ) -> int:
        try:
            parsed = int(value)
        except Exception:
            parsed = int(default)
        parsed = max(1, parsed)
        if maximum is not None:
            parsed = min(parsed, int(maximum))
        return int(parsed)

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
        self.auto_recovery_missing_instances = True
        self.compute_optical_flow = True
        self.follow_prediction_progress = True
        self.optical_flow_backend = "farneback"
        self.cutie_weights_path: Optional[str] = getattr(
            self._tracker_config, "cutie_weights_path", None
        )
        # VideoMT ONNX runtime controls
        self.videomt_mask_threshold = 0.5
        self.videomt_logit_threshold = -2.0
        self.videomt_seed_iou_threshold = 0.01
        self.videomt_window = 8
        self.videomt_input_height = 0
        self.videomt_input_width = 0
        self.sam3_score_threshold_detection = float(
            sam3_runtime.get("score_threshold_detection") or 0.35
        )
        self.sam3_new_det_thresh = float(sam3_runtime.get("new_det_thresh") or 0.25)
        self.sam3_propagation_direction = str(
            sam3_runtime.get("propagation_direction") or "both"
        )
        self.sam3_max_frame_num_to_track = sam3_runtime.get(
            "max_frame_num_to_track", None
        )
        self.sam3_device_override = (
            ""
            if sam3_runtime.get("device") is None
            else str(sam3_runtime.get("device"))
        )
        self.sam3_sliding_window_size = int(
            sam3_runtime.get("sliding_window_size") or 5
        )
        self.sam3_sliding_window_stride = sam3_runtime.get(
            "sliding_window_stride", None
        )
        # Agent seeding defaults
        self.sam3_agent_det_thresh = float(
            sam3_runtime.get("agent_det_thresh") or self.sam3_score_threshold_detection
        )
        self.sam3_agent_window_size = int(
            sam3_runtime.get("agent_window_size") or self.sam3_sliding_window_size
        )
        self.sam3_agent_stride = sam3_runtime.get("agent_stride", None)
        configured_agent_output_dir = sam3_runtime.get("agent_output_dir")
        self.sam3_agent_output_dir = (
            ""
            if configured_agent_output_dir in (None, "")
            else str(configured_agent_output_dir)
        )
        raw_compile = sam3_runtime.get("compile_model")
        self.sam3_compile_model = (
            bool(raw_compile) if raw_compile is not None else False
        )
        raw_offload = sam3_runtime.get("offload_video_to_cpu")
        self.sam3_offload_video_to_cpu = (
            True if raw_offload is None else bool(raw_offload)
        )
        raw_explicit_reseed = sam3_runtime.get("use_explicit_window_reseed")
        self.sam3_use_explicit_window_reseed = (
            True if raw_explicit_reseed is None else bool(raw_explicit_reseed)
        )
        raw_private_state = sam3_runtime.get("allow_private_state_mutation")
        self.sam3_allow_private_state_mutation = (
            False if raw_private_state is None else bool(raw_private_state)
        )
        raw_boundary_mask_match_iou_threshold = sam3_runtime.get(
            "boundary_mask_match_iou_threshold"
        )
        self.sam3_boundary_mask_match_iou_threshold = (
            0.2
            if raw_boundary_mask_match_iou_threshold is None
            else float(raw_boundary_mask_match_iou_threshold)
        )
        self.sam3_max_num_objects = self._coerce_positive_int(
            sam3_runtime.get("max_num_objects"),
            default=self._SAM31_DEFAULT_MAX_OBJECTS,
            maximum=512,
        )
        self.sam3_multiplex_count = self._coerce_positive_int(
            sam3_runtime.get("multiplex_count"),
            default=self._SAM31_DEFAULT_MULTIPLEX_COUNT,
            maximum=self._SAM31_MAX_MULTIPLEX_COUNT,
        )

        layout = QVBoxLayout()
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(14)

        intro_label = QLabel(
            "Tune segmentation, tracking, and SAM3 agent behavior. "
            "Tooltips and helper text explain how each option affects the workflow."
        )
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet("color: palette(window-text); font-weight: 500;")

        tabs = QTabWidget()
        tabs.setTabBarAutoHide(False)
        tabs.addTab(
            self._wrap_scrollable(self._build_segmentation_tab()), "Segmentation"
        )
        tabs.addTab(self._wrap_scrollable(self._build_tracker_tab()), "Tracker")
        tabs.addTab(self._wrap_scrollable(self._build_sam3_tab()), "SAM3 + Agent")

        layout.addWidget(intro_label)
        layout.addWidget(tabs)

        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)
        self._fit_initial_geometry()

    def load_window_state(self, window, optical_flow_manager=None) -> None:
        """Seed widgets from the current window state before showing the dialog."""
        self.epsilon_value = float(getattr(window, "epsilon_for_polygon", 2.0))
        self.t_max_value = int(getattr(window, "t_max_value", 5))
        self.automatic_pause_enabled = bool(
            getattr(window, "automatic_pause_enabled", False)
        )
        self.cpu_only_enabled = bool(getattr(window, "use_cpu_only", False))
        self.save_video_with_color_mask = bool(
            getattr(window, "save_video_with_color_mask", False)
        )
        self.auto_recovery_missing_instances = bool(
            getattr(window, "auto_recovery_missing_instances", True)
        )
        self.compute_optical_flow = bool(
            getattr(
                optical_flow_manager
                if optical_flow_manager is not None
                else getattr(window, "optical_flow_manager", None),
                "compute_optical_flow",
                True,
            )
        )
        self.follow_prediction_progress = bool(
            getattr(window, "_follow_prediction_progress", True)
        )
        self.optical_flow_backend = str(
            getattr(
                optical_flow_manager
                if optical_flow_manager is not None
                else getattr(window, "optical_flow_manager", None),
                "optical_flow_backend",
                "farneback",
            )
        )

        backend_idx = optical_flow_backend_combo_index(self.optical_flow_backend)

        self.epsilon_spinbox.setValue(self.epsilon_value)
        self.t_max_spinbox.setValue(self.t_max_value)
        self.automatic_pause_checkbox.setChecked(self.automatic_pause_enabled)
        self.cpu_only_checkbox.setChecked(self.cpu_only_enabled)
        self.save_video_with_color_mask_checkbox.setChecked(
            self.save_video_with_color_mask
        )
        self.auto_recovery_missing_instances_checkbox.setChecked(
            self.auto_recovery_missing_instances
        )
        self.compute_optical_flow_checkbox.setChecked(self.compute_optical_flow)
        self.optical_flow_backend_combo.setCurrentIndex(backend_idx)
        self.follow_prediction_progress_checkbox.setChecked(
            self.follow_prediction_progress
        )
        self.videomt_mask_threshold_spinbox.setValue(
            float(getattr(window, "videomt_mask_threshold", 0.5))
        )
        self.videomt_logit_threshold_spinbox.setValue(
            float(getattr(window, "videomt_logit_threshold", -2.0))
        )
        self.videomt_seed_iou_threshold_spinbox.setValue(
            float(getattr(window, "videomt_seed_iou_threshold", 0.01))
        )
        self.videomt_window_spinbox.setValue(int(getattr(window, "videomt_window", 8)))
        self.videomt_input_height_spinbox.setValue(
            int(getattr(window, "videomt_input_height", 0))
        )
        self.videomt_input_width_spinbox.setValue(
            int(getattr(window, "videomt_input_width", 0))
        )
        raw_cutie_override = getattr(window, "cutie_weights_override", None)
        if raw_cutie_override is None and hasattr(window, "settings"):
            try:
                raw_cutie_override = window.settings.value("cutie/weights_override", "")
            except Exception:
                raw_cutie_override = ""
        if not str(raw_cutie_override or "").strip():
            raw_cutie_override = getattr(
                self._tracker_config, "cutie_weights_path", None
            )
        self._set_cutie_checkpoint_selection(raw_cutie_override)

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

        self.cutie_variant_combo = QComboBox()
        self.cutie_variant_combo.addItem(
            "Follow Toolbar Selection (default: Original Cutie)",
            "",
        )
        self.cutie_variant_combo.addItem(
            "Original Base Mega — Upstream Dense (134 MB, default)",
            "cutie-base-mega.orig.pth",
        )
        self.cutie_variant_combo.addItem(
            "Improved Compact — SAM3 + DINOv3 (30 MB, 1.5x–2.7x faster)",
            "cutie-sam3-bodyparts.pth",
        )
        self.cutie_variant_combo.addItem(
            "Custom Cutie Checkpoint (.pth)…",
            self._CUTIE_CUSTOM_SENTINEL,
        )
        configure_compact_combo(
            self.cutie_variant_combo,
            min_chars=22,
            popup_min_width=430,
        )
        self.cutie_variant_combo.setToolTip(
            "Select which Cutie VOS checkpoint is used for Cutie propagation and "
            "Cutie + DINOv3 Keypoint Segmentation."
        )

        self.cutie_custom_row = QWidget()
        cutie_custom_layout = QHBoxLayout(self.cutie_custom_row)
        cutie_custom_layout.setContentsMargins(0, 0, 0, 0)
        cutie_custom_layout.setSpacing(6)
        self.cutie_custom_path_edit = QLineEdit()
        self.cutie_custom_path_edit.setPlaceholderText(
            "Path to custom Cutie .pth checkpoint"
        )
        self.cutie_browse_button = QPushButton("Browse…")
        self.cutie_browse_button.clicked.connect(self._browse_cutie_checkpoint)
        cutie_custom_layout.addWidget(self.cutie_custom_path_edit, 1)
        cutie_custom_layout.addWidget(self.cutie_browse_button)
        self.cutie_custom_row.setVisible(False)

        self.cutie_checkpoint_status_label = QLabel("")
        self.cutie_checkpoint_status_label.setWordWrap(True)
        self.cutie_checkpoint_status_label.setStyleSheet(
            "color: palette(window-text); font-size: 11px;"
        )

        cutie_selector_box = QWidget()
        cutie_selector_col = QVBoxLayout(cutie_selector_box)
        cutie_selector_col.setContentsMargins(0, 0, 0, 0)
        cutie_selector_col.setSpacing(4)
        cutie_selector_col.addWidget(self.cutie_variant_combo)
        cutie_selector_col.addWidget(self.cutie_custom_row)
        cutie_selector_col.addWidget(self.cutie_checkpoint_status_label)

        self.cutie_variant_combo.currentIndexChanged.connect(
            self._on_cutie_variant_changed
        )
        self.cutie_custom_path_edit.textChanged.connect(
            self._update_cutie_checkpoint_status
        )
        self._set_cutie_checkpoint_selection(self.cutie_weights_path)

        form.addRow("Cutie VOS checkpoint", cutie_selector_box)

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

        self.automatic_pause_checkbox = QCheckBox("Automatic Pause on Error Detection")
        self.automatic_pause_checkbox.setChecked(self.automatic_pause_enabled)
        self.automatic_pause_checkbox.setToolTip(
            "Pause CUTIE tracking when missing instances are detected so you can review and correct the frame before continuing."
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
        self.optical_flow_backend_combo.addItems(list(OPTICAL_FLOW_BACKEND_LABELS))
        backend_idx = optical_flow_backend_combo_index(self.optical_flow_backend)
        self.optical_flow_backend_combo.setCurrentIndex(backend_idx)
        self.optical_flow_backend_combo.setToolTip(
            "Choose OpenCV Farneback, Torch Farneback, or RAFT (TorchVision)."
        )

        self.auto_recovery_missing_instances_checkbox = QCheckBox(
            "Auto Recovery of Missing Instances"
        )
        self.auto_recovery_missing_instances_checkbox.setChecked(
            self.auto_recovery_missing_instances
        )
        self.auto_recovery_missing_instances_checkbox.setToolTip(
            "Try to reintroduce instances that disappear unexpectedly using SAM "
            "bbox recovery, then the most recent instance mask."
        )

        self.compute_optical_flow_checkbox = QCheckBox(
            "Compute Motion Index based on optical flow over instance mask"
        )
        self.compute_optical_flow_checkbox.setChecked(self.compute_optical_flow)
        self.compute_optical_flow_checkbox.setToolTip(
            "Toggle motion index calculation using optical flow across instance masks."
        )

        self.follow_prediction_progress_checkbox = QCheckBox(
            "Follow Prediction Progress on Timeline"
        )
        self.follow_prediction_progress_checkbox.setChecked(
            self.follow_prediction_progress
        )
        self.follow_prediction_progress_checkbox.setToolTip(
            "Auto-jump the playhead to the latest predicted frame while prediction is running."
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
                "Use Torch Farneback for verification or RAFT for highest fidelity.",
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
        form.addRow(
            self._wrap_checkbox(
                self.follow_prediction_progress_checkbox,
                "If enabled, timeline view follows prediction progress automatically.",
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

        self.tracker_preset_combo = QComboBox()
        presets = list(CutieDinoTrackerConfig.available_presets())
        self.tracker_preset_combo.addItem(self._CUSTOM_TRACKER_PRESET)
        for preset in presets:
            self.tracker_preset_combo.addItem(preset)
        current_preset = getattr(self._tracker_config, "tracker_preset", None)
        if current_preset and current_preset in presets:
            self.tracker_preset_combo.setCurrentText(str(current_preset))
        else:
            self.tracker_preset_combo.setCurrentText(self._CUSTOM_TRACKER_PRESET)

        self.dino_model_combo = QComboBox()
        configure_dino_model_combo(
            self.dino_model_combo,
            default_identifier=str(
                getattr(
                    self._tracker_config,
                    "patch_model_name",
                    getattr(self._tracker_config, "dinov3_model_name", ""),
                )
            ),
            editable=True,
        )
        self.dino_model_combo.setToolTip(
            "Choose a DINO, EmbeddingGemma 2, SigLIP 2, or on-device MobileNet backbone, or paste a Hugging Face model id."
        )
        self.dino_model_status_label = QLabel("")
        self.dino_model_status_label.setWordWrap(True)
        self.dino_model_status_label.setStyleSheet(
            "color: palette(window-text); font-size: 11px;"
        )

        self.embedding_matryoshka_dim_combo = QComboBox()
        self.embedding_matryoshka_dim_combo.addItem("Full (model default)", None)
        for mdim in (768, 512, 256, 128):
            self.embedding_matryoshka_dim_combo.addItem(f"{mdim} dims", mdim)
        configured_mdim = getattr(
            self._tracker_config, "embedding_matryoshka_dim", None
        )
        if configured_mdim is not None:
            matched_idx = self.embedding_matryoshka_dim_combo.findData(
                int(configured_mdim)
            )
            if matched_idx >= 0:
                self.embedding_matryoshka_dim_combo.setCurrentIndex(matched_idx)

        self.region_descriptor_radius_spinbox = QSpinBox()
        self.region_descriptor_radius_spinbox.setRange(0, 4)
        self.region_descriptor_radius_spinbox.setSingleStep(1)
        self.region_descriptor_radius_spinbox.setValue(
            int(getattr(self._tracker_config, "region_descriptor_radius", 0))
        )

        self.region_descriptor_sigma_spinbox = QDoubleSpinBox()
        self.region_descriptor_sigma_spinbox.setRange(0.2, 4.0)
        self.region_descriptor_sigma_spinbox.setSingleStep(0.05)
        self.region_descriptor_sigma_spinbox.setValue(
            float(getattr(self._tracker_config, "region_descriptor_sigma", 0.85))
        )

        self.region_subpatch_sampling_checkbox = QCheckBox(
            "Sub-patch continuous Gaussian sampling"
        )
        self.region_subpatch_sampling_checkbox.setChecked(
            bool(getattr(self._tracker_config, "region_subpatch_sampling", False))
        )

        self.pixel_region_refine_checkbox = QCheckBox(
            "Drift-free Gaussian pixel template refinement"
        )
        self.pixel_region_refine_checkbox.setChecked(
            bool(getattr(self._tracker_config, "pixel_region_refine_enabled", False))
        )

        self.mask_enforce_checkbox = QCheckBox("Clamp keypoints to instance mask")
        self.mask_enforce_checkbox.setChecked(
            bool(self._tracker_config.mask_enforce_position)
        )

        self.keypoint_refine_radius_spinbox = QSpinBox()
        self.keypoint_refine_radius_spinbox.setRange(0, 12)
        self.keypoint_refine_radius_spinbox.setSingleStep(1)
        self.keypoint_refine_radius_spinbox.setValue(
            int(getattr(self._tracker_config, "keypoint_refine_radius", 0))
        )

        self.keypoint_refine_sigma_spinbox = QDoubleSpinBox()
        self.keypoint_refine_sigma_spinbox.setRange(0.1, 8.0)
        self.keypoint_refine_sigma_spinbox.setSingleStep(0.05)
        self.keypoint_refine_sigma_spinbox.setValue(
            float(getattr(self._tracker_config, "keypoint_refine_sigma", 1.25))
        )

        self.keypoint_refine_temperature_spinbox = QDoubleSpinBox()
        self.keypoint_refine_temperature_spinbox.setRange(0.05, 2.0)
        self.keypoint_refine_temperature_spinbox.setSingleStep(0.05)
        self.keypoint_refine_temperature_spinbox.setValue(
            float(getattr(self._tracker_config, "keypoint_refine_temperature", 0.35))
        )

        self.dinov3_positional_debias_checkbox = QCheckBox("DINOv3 positional debias")
        self.dinov3_positional_debias_checkbox.setChecked(
            bool(getattr(self._tracker_config, "dinov3_positional_debias", False))
        )

        self.dinov3_positional_debias_components_spinbox = QSpinBox()
        self.dinov3_positional_debias_components_spinbox.setRange(1, 10)
        self.dinov3_positional_debias_components_spinbox.setSingleStep(1)
        self.dinov3_positional_debias_components_spinbox.setValue(
            int(getattr(self._tracker_config, "dinov3_positional_debias_components", 6))
        )

        self.dinov3_positional_debias_strength_spinbox = QDoubleSpinBox()
        self.dinov3_positional_debias_strength_spinbox.setRange(0.0, 1.0)
        self.dinov3_positional_debias_strength_spinbox.setSingleStep(0.05)
        self.dinov3_positional_debias_strength_spinbox.setValue(
            float(
                getattr(self._tracker_config, "dinov3_positional_debias_strength", 0.0)
            )
        )

        self.dinov3_backward_consistency_checkbox = QCheckBox(
            "DINOv3 backward consistency"
        )
        self.dinov3_backward_consistency_checkbox.setChecked(
            bool(getattr(self._tracker_config, "dinov3_backward_consistency", False))
        )

        self.dinov3_backward_consistency_weight_spinbox = QDoubleSpinBox()
        self.dinov3_backward_consistency_weight_spinbox.setRange(0.0, 1.0)
        self.dinov3_backward_consistency_weight_spinbox.setSingleStep(0.05)
        self.dinov3_backward_consistency_weight_spinbox.setValue(
            float(
                getattr(self._tracker_config, "dinov3_backward_consistency_weight", 0.0)
            )
        )

        self.keypoint_cluster_refine_checkbox = QCheckBox("Coherent peak refinement")
        self.keypoint_cluster_refine_checkbox.setChecked(
            bool(getattr(self._tracker_config, "keypoint_cluster_refine", False))
        )

        self.keypoint_cluster_refine_radius_spinbox = QSpinBox()
        self.keypoint_cluster_refine_radius_spinbox.setRange(0, 8)
        self.keypoint_cluster_refine_radius_spinbox.setSingleStep(1)
        self.keypoint_cluster_refine_radius_spinbox.setValue(
            int(getattr(self._tracker_config, "keypoint_cluster_refine_radius", 1))
        )

        self.keypoint_cluster_refine_drop_spinbox = QDoubleSpinBox()
        self.keypoint_cluster_refine_drop_spinbox.setRange(0.0, 1.0)
        self.keypoint_cluster_refine_drop_spinbox.setSingleStep(0.01)
        self.keypoint_cluster_refine_drop_spinbox.setValue(
            float(getattr(self._tracker_config, "keypoint_cluster_refine_drop", 0.1))
        )

        self.mask_enforce_radius_spinbox = QSpinBox()
        self.mask_enforce_radius_spinbox.setRange(1, 256)
        self.mask_enforce_radius_spinbox.setSingleStep(1)
        self.mask_enforce_radius_spinbox.setValue(
            int(self._tracker_config.mask_enforce_snap_radius)
        )

        self.mask_enforce_reject_checkbox = QCheckBox("Reject updates outside the mask")
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

        configure_compact_combo(
            self.tracker_preset_combo,
            min_chars=20,
            popup_min_width=380,
        )
        configure_compact_combo(
            self.dino_model_combo,
            min_chars=22,
            popup_min_width=440,
        )
        configure_compact_combo(
            self.embedding_matryoshka_dim_combo,
            min_chars=18,
            popup_min_width=320,
        )
        tracker_form.addRow(
            "Tracker preset",
            self._wrap_with_hint(
                self.tracker_preset_combo,
                "Choose a tuned preset (e.g. rodents at 30fps with occlusions) or keep custom values.",
            ),
        )
        dino_model_wrapper = self._wrap_with_hint(
            self.dino_model_combo,
            "Supports DINOv3/v2, Google EmbeddingGemma 2, SigLIP 2, and fast on-device MobileNetV4/V3 pyramids.",
        )
        dino_model_layout = dino_model_wrapper.layout()
        if dino_model_layout is not None:
            dino_model_layout.addWidget(self.dino_model_status_label)
        tracker_form.addRow("DINO feature model", dino_model_wrapper)
        tracker_form.addRow(
            "Embedding Matryoshka dim",
            self._wrap_with_hint(
                self.embedding_matryoshka_dim_combo,
                "Optional prefix dimension (e.g. 256/128 for EmbeddingGemma 2) to cut matching memory and latency.",
            ),
        )
        tracker_form.addRow(
            "Gaussian region radius (patches)",
            self._wrap_with_hint(
                self.region_descriptor_radius_spinbox,
                "0 disables. Aggregates a mask-aware Gaussian patch region around each keypoint without background bleed.",
            ),
        )
        tracker_form.addRow(
            "Gaussian region sigma",
            self._wrap_with_hint(
                self.region_descriptor_sigma_spinbox,
                "Spatial Gaussian spread (patch units) for local region descriptors and sub-patch sampling.",
            ),
        )
        tracker_form.addRow(
            self._wrap_checkbox(
                self.region_subpatch_sampling_checkbox,
                "Sample descriptors continuously at sub-patch coordinates instead of snapping to discrete patch cells.",
            )
        )
        tracker_form.addRow(
            self._wrap_checkbox(
                self.pixel_region_refine_checkbox,
                "Anchor a Gaussian-weighted local pixel/gradient ZNCC template at manual seeds to prevent multi-frame LK drift.",
            )
        )
        tracker_form.addRow(
            self._wrap_checkbox(
                self.mask_enforce_checkbox,
                "Keeps keypoints snapped to the instance mask to reduce drift.",
            )
        )
        tracker_form.addRow(
            "Keypoint refine radius (patches)",
            self._wrap_with_hint(
                self.keypoint_refine_radius_spinbox,
                "0 disables. When enabled, reports the Gaussian-weighted centroid of nearby candidates for smoother sub-patch motion.",
            ),
        )
        tracker_form.addRow(
            "Keypoint refine sigma",
            self._wrap_with_hint(
                self.keypoint_refine_sigma_spinbox,
                "Spatial Gaussian spread in patch units (typical 1.0–2.0).",
            ),
        )
        tracker_form.addRow(
            "Keypoint refine temperature",
            self._wrap_with_hint(
                self.keypoint_refine_temperature_spinbox,
                "Softmax temperature over candidate scores (lower = sharper peak).",
            ),
        )
        tracker_form.addRow(
            self._wrap_checkbox(
                self.dinov3_positional_debias_checkbox,
                "Suppress coordinate-correlated DINO feature responses before matching.",
            )
        )
        tracker_form.addRow(
            "Debias components",
            self._wrap_with_hint(
                self.dinov3_positional_debias_components_spinbox,
                "Number of low-dimensional positional terms to remove.",
            ),
        )
        tracker_form.addRow(
            "Debias strength",
            self._wrap_with_hint(
                self.dinov3_positional_debias_strength_spinbox,
                "0 disables projection; 1 removes the full estimated positional component.",
            ),
        )
        tracker_form.addRow(
            self._wrap_checkbox(
                self.dinov3_backward_consistency_checkbox,
                "Reward candidates that map back to the previous keypoint patch.",
            )
        )
        tracker_form.addRow(
            "Backward weight",
            self._wrap_with_hint(
                self.dinov3_backward_consistency_weight_spinbox,
                "Score weight for backward nearest-neighbor agreement.",
            ),
        )
        tracker_form.addRow(
            self._wrap_checkbox(
                self.keypoint_cluster_refine_checkbox,
                "Aggregate the connected local similarity peak around the selected seed.",
            )
        )
        tracker_form.addRow(
            "Peak radius",
            self._wrap_with_hint(
                self.keypoint_cluster_refine_radius_spinbox,
                "Patch radius for connected peak aggregation.",
            ),
        )
        tracker_form.addRow(
            "Peak drop",
            self._wrap_with_hint(
                self.keypoint_cluster_refine_drop_spinbox,
                "Maximum score drop from the seed patch kept in the connected peak.",
            ),
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
        self.videomt_mask_threshold_spinbox = QDoubleSpinBox()
        self.videomt_mask_threshold_spinbox.setRange(0.01, 0.99)
        self.videomt_mask_threshold_spinbox.setSingleStep(0.01)
        self.videomt_mask_threshold_spinbox.setValue(self.videomt_mask_threshold)

        self.videomt_logit_threshold_spinbox = QDoubleSpinBox()
        self.videomt_logit_threshold_spinbox.setRange(-20.0, 20.0)
        self.videomt_logit_threshold_spinbox.setSingleStep(0.1)
        self.videomt_logit_threshold_spinbox.setValue(self.videomt_logit_threshold)

        self.videomt_seed_iou_threshold_spinbox = QDoubleSpinBox()
        self.videomt_seed_iou_threshold_spinbox.setRange(0.0, 1.0)
        self.videomt_seed_iou_threshold_spinbox.setSingleStep(0.01)
        self.videomt_seed_iou_threshold_spinbox.setValue(
            self.videomt_seed_iou_threshold
        )

        self.videomt_window_spinbox = QSpinBox()
        self.videomt_window_spinbox.setRange(1, 256)
        self.videomt_window_spinbox.setValue(int(self.videomt_window))

        self.videomt_input_height_spinbox = QSpinBox()
        self.videomt_input_height_spinbox.setRange(0, 4096)
        self.videomt_input_height_spinbox.setSpecialValueText("auto")
        self.videomt_input_height_spinbox.setValue(int(self.videomt_input_height))

        self.videomt_input_width_spinbox = QSpinBox()
        self.videomt_input_width_spinbox.setRange(0, 4096)
        self.videomt_input_width_spinbox.setSpecialValueText("auto")
        self.videomt_input_width_spinbox.setValue(int(self.videomt_input_width))

        tracker_form.addRow(
            "VideoMT mask threshold",
            self._wrap_with_hint(
                self.videomt_mask_threshold_spinbox,
                "Binary mask threshold for VideoMT ONNX predictions (default 0.50).",
            ),
        )
        tracker_form.addRow(
            "VideoMT logit threshold",
            self._wrap_with_hint(
                self.videomt_logit_threshold_spinbox,
                "Minimum query foreground-vs-background score to accept (default -2.0).",
            ),
        )
        tracker_form.addRow(
            "VideoMT seed IoU threshold",
            self._wrap_with_hint(
                self.videomt_seed_iou_threshold_spinbox,
                "Minimum IoU to map model queries to your seed labels (default 0.01).",
            ),
        )
        tracker_form.addRow(
            "VideoMT window (5D models)",
            self._wrap_with_hint(
                self.videomt_window_spinbox,
                "Temporal clip length for [1,3,T,H,W] VideoMT models when T is dynamic.",
            ),
        )
        tracker_form.addRow(
            "VideoMT input height (0=auto)",
            self._wrap_with_hint(
                self.videomt_input_height_spinbox,
                "Force resize height before inference. Keep auto unless required by the model.",
            ),
        )
        tracker_form.addRow(
            "VideoMT input width (0=auto)",
            self._wrap_with_hint(
                self.videomt_input_width_spinbox,
                "Force resize width before inference. Keep auto unless required by the model.",
            ),
        )

        tracker_layout.addLayout(tracker_form)
        tracker_layout.addStretch(1)

        self.tracker_preset_combo.currentTextChanged.connect(self._apply_tracker_preset)
        self.dino_model_combo.currentIndexChanged.connect(
            self._update_dino_model_runtime_status
        )
        if self.dino_model_combo.isEditable():
            self.dino_model_combo.editTextChanged.connect(
                self._update_dino_model_runtime_status
            )
        self._update_dino_model_runtime_status()
        return tab

    def _update_dino_model_runtime_status(self, *_args) -> None:
        selected_model = selected_dino_model_identifier(self.dino_model_combo)
        supported, message = check_dino_model_runtime_support(selected_model)
        self.dino_model_status_label.setText(message)
        if supported:
            self.dino_model_status_label.setStyleSheet(
                "color: palette(window-text); font-size: 11px;"
            )
        else:
            self.dino_model_status_label.setStyleSheet(
                "color: #b45309; font-size: 11px; font-weight: 500;"
            )
        if supports_matryoshka_truncation(selected_model):
            self.embedding_matryoshka_dim_combo.setToolTip(
                "Native Matryoshka prefix truncation (768/512/256/128) supported by this model."
            )
        else:
            self.embedding_matryoshka_dim_combo.setToolTip(
                "Selected backbone is non-Matryoshka; tracker keeps full channels by default "
                "(or uses deterministic orthonormal compression when strict MRL is disabled)."
            )

    def _apply_tracker_preset(self, preset: str) -> None:
        if preset == self._CUSTOM_TRACKER_PRESET:
            return
        try:
            cfg = CutieDinoTrackerConfig.from_preset(str(preset))
        except ValueError:
            return
        self.region_descriptor_radius_spinbox.setValue(
            int(cfg.region_descriptor_radius)
        )
        self.region_descriptor_sigma_spinbox.setValue(
            float(cfg.region_descriptor_sigma)
        )
        self.region_subpatch_sampling_checkbox.setChecked(
            bool(cfg.region_subpatch_sampling)
        )
        self.pixel_region_refine_checkbox.setChecked(
            bool(cfg.pixel_region_refine_enabled)
        )
        self.keypoint_refine_radius_spinbox.setValue(int(cfg.keypoint_refine_radius))
        self.keypoint_refine_sigma_spinbox.setValue(float(cfg.keypoint_refine_sigma))
        self.keypoint_refine_temperature_spinbox.setValue(
            float(cfg.keypoint_refine_temperature)
        )
        self.dinov3_positional_debias_checkbox.setChecked(
            bool(cfg.dinov3_positional_debias)
        )
        self.dinov3_positional_debias_components_spinbox.setValue(
            int(cfg.dinov3_positional_debias_components)
        )
        self.dinov3_positional_debias_strength_spinbox.setValue(
            float(cfg.dinov3_positional_debias_strength)
        )
        self.dinov3_backward_consistency_checkbox.setChecked(
            bool(cfg.dinov3_backward_consistency)
        )
        self.dinov3_backward_consistency_weight_spinbox.setValue(
            float(cfg.dinov3_backward_consistency_weight)
        )
        self.keypoint_cluster_refine_checkbox.setChecked(
            bool(cfg.keypoint_cluster_refine)
        )
        self.keypoint_cluster_refine_radius_spinbox.setValue(
            int(cfg.keypoint_cluster_refine_radius)
        )
        self.keypoint_cluster_refine_drop_spinbox.setValue(
            float(cfg.keypoint_cluster_refine_drop)
        )
        self.mask_enforce_checkbox.setChecked(bool(cfg.mask_enforce_position))
        self.mask_enforce_radius_spinbox.setValue(int(cfg.mask_enforce_snap_radius))
        self.mask_enforce_reject_checkbox.setChecked(
            bool(cfg.mask_enforce_reject_outside)
        )
        self.motion_search_tighten_spinbox.setValue(float(cfg.motion_search_tighten))
        self.motion_search_gain_spinbox.setValue(float(cfg.motion_search_gain))
        self.motion_search_flow_gain_spinbox.setValue(
            float(cfg.motion_search_flow_gain)
        )
        self.motion_search_min_radius_spinbox.setValue(
            float(cfg.motion_search_min_radius)
        )
        self.motion_search_max_radius_spinbox.setValue(
            float(cfg.motion_search_max_radius)
        )
        self.motion_search_miss_boost_spinbox.setValue(
            float(cfg.motion_search_miss_boost)
        )
        self.motion_prior_penalty_weight_spinbox.setValue(
            float(cfg.motion_prior_penalty_weight)
        )
        self.motion_prior_soft_radius_spinbox.setValue(
            float(cfg.motion_prior_soft_radius_px)
        )
        self.motion_prior_radius_factor_spinbox.setValue(
            float(cfg.motion_prior_radius_factor)
        )
        self.motion_prior_miss_relief_spinbox.setValue(
            float(cfg.motion_prior_miss_relief)
        )
        self.motion_prior_flow_relief_spinbox.setValue(
            float(cfg.motion_prior_flow_relief)
        )

    def _wrap_scrollable(self, widget: QWidget) -> QScrollArea:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area.setWidget(widget)
        return scroll_area

    def _fit_initial_geometry(self) -> None:
        screen = QGuiApplication.primaryScreen()
        if screen is None:
            self.resize(660, 640)
            return

        available = screen.availableGeometry()
        max_width = max(1, available.width() - 80)
        max_height = max(1, available.height() - 80)
        width = min(680, max(560, int(available.width() * 0.46)), max_width)
        height = min(700, max(540, int(available.height() * 0.76)), max_height)
        self.resize(width, height)

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
        self.sam3_score_thresh_spinbox.setValue(self.sam3_score_threshold_detection)
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

        self.sam3_max_objects_spinbox = QSpinBox()
        self.sam3_max_objects_spinbox.setRange(1, 512)
        self.sam3_max_objects_spinbox.setValue(int(self.sam3_max_num_objects))
        self.sam3_max_objects_spinbox.setToolTip(
            "Maximum total objects SAM3.1 may track across multiplex buckets."
        )

        self.sam3_multiplex_count_spinbox = QSpinBox()
        self.sam3_multiplex_count_spinbox.setRange(1, self._SAM31_MAX_MULTIPLEX_COUNT)
        self.sam3_multiplex_count_spinbox.setValue(int(self.sam3_multiplex_count))
        self.sam3_multiplex_count_spinbox.setToolTip(
            "SAM3.1 object bucket size. Use 16 to process the full multiplex bucket in one pass."
        )

        self.sam3_direction_combo = QComboBox()
        self.sam3_direction_combo.addItems(["forward", "backward", "both"])
        idx = self.sam3_direction_combo.findText(self.sam3_propagation_direction)
        if idx >= 0:
            self.sam3_direction_combo.setCurrentIndex(idx)

        self.sam3_max_frames_spinbox = QSpinBox()
        self.sam3_max_frames_spinbox.setRange(0, 100000)
        self.sam3_max_frames_spinbox.setSpecialValueText("auto")
        self.sam3_max_frames_spinbox.setValue(
            0
            if self.sam3_max_frame_num_to_track in (None, "")
            else int(self.sam3_max_frame_num_to_track)
        )

        self.sam3_device_lineedit = QLineEdit(self.sam3_device_override)
        self.sam3_device_lineedit.setPlaceholderText("auto (cuda/mps/cpu)")
        self.sam3_device_lineedit.setToolTip(
            "Optional device override for SAM3 tracker (e.g., cuda, mps, cpu). Leave empty for auto."
        )

        self.sam3_compile_checkbox = QCheckBox("Enable torch.compile (CUDA only)")
        self.sam3_compile_checkbox.setChecked(bool(self.sam3_compile_model))
        self.sam3_compile_checkbox.setToolTip(
            "Uses torch.compile to speed up SAM3 on NVIDIA GPUs. First run may be slower due to compilation."
        )

        self.sam3_offload_checkbox = QCheckBox("Offload video to CPU (lower VRAM)")
        self.sam3_offload_checkbox.setChecked(bool(self.sam3_offload_video_to_cpu))
        self.sam3_offload_checkbox.setToolTip(
            "Keep frame storage on CPU to reduce VRAM usage. Disable on powerful GPUs to reduce CPU↔GPU transfers."
        )
        self.sam3_explicit_reseed_checkbox = QCheckBox(
            "Use explicit window-boundary reseeding"
        )
        self.sam3_explicit_reseed_checkbox.setChecked(
            bool(self.sam3_use_explicit_window_reseed)
        )
        self.sam3_explicit_reseed_checkbox.setToolTip(
            "Replay explicit carry prompts at each window boundary to improve long-video continuity without relying on private predictor state."
        )

        self.sam3_boundary_mask_match_iou_spinbox = QDoubleSpinBox()
        self.sam3_boundary_mask_match_iou_spinbox.setRange(0.0, 1.0)
        self.sam3_boundary_mask_match_iou_spinbox.setSingleStep(0.01)
        self.sam3_boundary_mask_match_iou_spinbox.setValue(
            float(self.sam3_boundary_mask_match_iou_threshold)
        )
        self.sam3_boundary_mask_match_iou_spinbox.setToolTip(
            "IoU threshold used to match carried boundary masks to the previous window's last-frame shapes."
        )

        self.sam3_private_state_checkbox = QCheckBox(
            "Allow private SAM3 state mutation (legacy compatibility)"
        )
        self.sam3_private_state_checkbox.setChecked(
            bool(self.sam3_allow_private_state_mutation)
        )
        self.sam3_private_state_checkbox.setToolTip(
            "Enables mutation of private predictor internals for legacy behavior. Keep disabled unless debugging compatibility."
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
            0
            if self.sam3_sliding_window_stride in (None, "")
            else int(self.sam3_sliding_window_stride)
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
        self.sam3_agent_window_size_spinbox.setValue(self.sam3_agent_window_size)
        self.sam3_agent_window_size_spinbox.setToolTip(
            "Frames per agent window; first frame of each window is corrected by the agent."
        )

        self.sam3_agent_stride_spinbox = QSpinBox()
        self.sam3_agent_stride_spinbox.setRange(0, 1000)
        self.sam3_agent_stride_spinbox.setSpecialValueText("auto")
        self.sam3_agent_stride_spinbox.setValue(
            0 if self.sam3_agent_stride in (None, "") else int(self.sam3_agent_stride)
        )
        self.sam3_agent_stride_spinbox.setToolTip(
            "Stride between agent windows; 0 uses agent window size."
        )

        self.sam3_agent_output_dir_lineedit = QLineEdit(self.sam3_agent_output_dir)
        self.sam3_agent_output_dir_lineedit.setPlaceholderText(
            "Disabled by default (set a folder to save agent JSON/PNG)"
        )
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
            "Max tracked objects",
            self._wrap_with_hint(
                self.sam3_max_objects_spinbox,
                "Total object capacity for SAM3.1 video tracking. Increase for crowded videos; memory use grows with capacity.",
            ),
        )
        form.addRow(
            "Multiplex bucket size",
            self._wrap_with_hint(
                self.sam3_multiplex_count_spinbox,
                "SAM3.1 can track up to 16 objects together per bucket. The default 16 uses the released model's full bucket.",
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
            "Performance",
            self._wrap_checkbox(
                self.sam3_compile_checkbox,
                "Enabling compile often yields a 1.2–2× speedup on NVIDIA GPUs after warmup.",
            ),
        )
        form.addRow(
            "",
            self._wrap_checkbox(
                self.sam3_offload_checkbox,
                "Leave on for low-VRAM GPUs; disable if you have enough VRAM and want higher throughput.",
            ),
        )
        form.addRow(
            "Window handoff",
            self._wrap_checkbox(
                self.sam3_explicit_reseed_checkbox,
                "Recommended: explicit carry prompts at window boundaries reduce brittle cross-window drift.",
            ),
        )
        form.addRow(
            "Boundary mask match IoU",
            self._wrap_with_hint(
                self.sam3_boundary_mask_match_iou_spinbox,
                "Match carried masks to the previous window's last-frame shapes by IoU. Higher values are stricter; 0.20 is a reasonable default.",
            ),
        )
        form.addRow(
            "",
            self._wrap_checkbox(
                self.sam3_private_state_checkbox,
                "Legacy fallback path that touches private SAM3 internals. Prefer leaving this disabled.",
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
                "Optional. Leave empty to avoid saving agent JSON/PNG outputs.",
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
            hint_label.setStyleSheet("color: palette(window-text); font-size: 11px;")
            column.addWidget(hint_label)

        return wrapper

    def _wrap_checkbox(self, checkbox: QCheckBox, hint: str) -> QWidget:
        return self._wrap_with_hint(checkbox, hint)

    def _set_cutie_checkpoint_selection(self, raw_spec: Optional[object]) -> None:
        text = str(raw_spec or "").strip()
        if text.lower() == "cutie.pt":
            text = ""
        self.cutie_variant_combo.blockSignals(True)
        self.cutie_custom_path_edit.blockSignals(True)
        try:
            if not text:
                self.cutie_variant_combo.setCurrentIndex(0)
                self.cutie_custom_path_edit.setText("")
                self.cutie_custom_row.setVisible(False)
            elif text == "cutie-sam3-bodyparts.pth":
                self.cutie_variant_combo.setCurrentIndex(
                    self.cutie_variant_combo.findData("cutie-sam3-bodyparts.pth")
                )
                self.cutie_custom_path_edit.setText("")
                self.cutie_custom_row.setVisible(False)
            elif text in ("cutie-base-mega.orig.pth", "cutie-base-mega.pth"):
                self.cutie_variant_combo.setCurrentIndex(
                    self.cutie_variant_combo.findData("cutie-base-mega.orig.pth")
                )
                self.cutie_custom_path_edit.setText("")
                self.cutie_custom_row.setVisible(False)
            else:
                self.cutie_variant_combo.setCurrentIndex(
                    self.cutie_variant_combo.findData(self._CUTIE_CUSTOM_SENTINEL)
                )
                self.cutie_custom_path_edit.setText(text)
                self.cutie_custom_row.setVisible(True)
        finally:
            self.cutie_custom_path_edit.blockSignals(False)
            self.cutie_variant_combo.blockSignals(False)
        self._update_cutie_checkpoint_status()

    def _current_cutie_spec_from_widgets(self) -> Optional[str]:
        data = str(self.cutie_variant_combo.currentData() or "").strip()
        if not data:
            return None
        if data == self._CUTIE_CUSTOM_SENTINEL:
            custom = self.cutie_custom_path_edit.text().strip()
            return custom or None
        return data

    def _on_cutie_variant_changed(self, *_args) -> None:
        is_custom = (
            self.cutie_variant_combo.currentData() == self._CUTIE_CUSTOM_SENTINEL
        )
        self.cutie_custom_row.setVisible(is_custom)
        self._update_cutie_checkpoint_status()

    def _browse_cutie_checkpoint(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select Cutie VOS Checkpoint",
            "",
            "Cutie Checkpoints (*.pth *.pt);;All Files (*)",
        )
        if selected:
            self.cutie_custom_path_edit.setText(
                str(Path(selected).expanduser().resolve())
            )

    def _update_cutie_checkpoint_status(self, *_args) -> None:
        self.cutie_weights_path = self._current_cutie_spec_from_widgets()
        if isinstance(getattr(self, "_tracker_settings", None), dict):
            self._tracker_settings["cutie_weights_path"] = self.cutie_weights_path
        data = str(self.cutie_variant_combo.currentData() or "").strip()
        if data == self._CUTIE_CUSTOM_SENTINEL:
            custom_text = self.cutie_custom_path_edit.text().strip()
            if not custom_text:
                self.cutie_checkpoint_status_label.setText(
                    "Choose a custom .pth Cutie checkpoint file."
                )
                self.cutie_checkpoint_status_label.setStyleSheet(
                    "color: #b45309; font-size: 11px; font-weight: 500;"
                )
                return
            ok, msg = describe_cutie_checkpoint_status_fast(custom_text)
        elif not data:
            ok, msg = describe_cutie_checkpoint_status_fast(None)
            msg = f"Follows toolbar Cutie model — default: {msg}"
        else:
            ok, msg = describe_cutie_checkpoint_status_fast(data)

        self.cutie_checkpoint_status_label.setText(msg)
        if ok:
            self.cutie_checkpoint_status_label.setStyleSheet(
                "color: palette(window-text); font-size: 11px;"
            )
        else:
            self.cutie_checkpoint_status_label.setStyleSheet(
                "color: #b45309; font-size: 11px; font-weight: 500;"
            )

    def accept(self) -> None:  # pragma: no cover - dialog delegate
        if self.cutie_variant_combo.currentData() == self._CUTIE_CUSTOM_SENTINEL:
            custom_text = self.cutie_custom_path_edit.text().strip()
            ok, msg = (
                describe_cutie_checkpoint_status_fast(custom_text)
                if custom_text
                else (False, "Please select a custom Cutie .pth checkpoint file.")
            )
            if not ok:
                QMessageBox.warning(self, "Cutie Checkpoint Not Found", msg)
                return
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
        self.follow_prediction_progress = (
            self.follow_prediction_progress_checkbox.isChecked()
        )
        self.optical_flow_backend = self.get_optical_flow_backend()
        self.cutie_weights_path = self._current_cutie_spec_from_widgets()

        snap_radius = self.mask_enforce_radius_spinbox.value()
        preset = None
        if hasattr(self, "tracker_preset_combo"):
            selected = str(self.tracker_preset_combo.currentText()).strip()
            if selected and selected != self._CUSTOM_TRACKER_PRESET:
                preset = selected
        selected_dino_model = selected_dino_model_identifier(self.dino_model_combo)
        selected_mdim = self.embedding_matryoshka_dim_combo.currentData()
        self._tracker_settings = {
            "tracker_preset": preset,
            "cutie_weights_path": self.cutie_weights_path,
            "patch_model_name": selected_dino_model,
            "dinov3_model_name": selected_dino_model,
            "embedding_matryoshka_dim": (
                int(selected_mdim) if selected_mdim is not None else None
            ),
            "region_descriptor_radius": self.region_descriptor_radius_spinbox.value(),
            "region_descriptor_sigma": self.region_descriptor_sigma_spinbox.value(),
            "region_subpatch_sampling": self.region_subpatch_sampling_checkbox.isChecked(),
            "pixel_region_refine_enabled": self.pixel_region_refine_checkbox.isChecked(),
            "keypoint_refine_radius": self.keypoint_refine_radius_spinbox.value(),
            "keypoint_refine_sigma": self.keypoint_refine_sigma_spinbox.value(),
            "keypoint_refine_temperature": self.keypoint_refine_temperature_spinbox.value(),
            "dinov3_positional_debias": self.dinov3_positional_debias_checkbox.isChecked(),
            "dinov3_positional_debias_components": self.dinov3_positional_debias_components_spinbox.value(),
            "dinov3_positional_debias_strength": self.dinov3_positional_debias_strength_spinbox.value(),
            "dinov3_backward_consistency": self.dinov3_backward_consistency_checkbox.isChecked(),
            "dinov3_backward_consistency_weight": self.dinov3_backward_consistency_weight_spinbox.value(),
            "keypoint_cluster_refine": self.keypoint_cluster_refine_checkbox.isChecked(),
            "keypoint_cluster_refine_radius": self.keypoint_cluster_refine_radius_spinbox.value(),
            "keypoint_cluster_refine_drop": self.keypoint_cluster_refine_drop_spinbox.value(),
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
        self.videomt_mask_threshold = self.videomt_mask_threshold_spinbox.value()
        self.videomt_logit_threshold = self.videomt_logit_threshold_spinbox.value()
        self.videomt_seed_iou_threshold = (
            self.videomt_seed_iou_threshold_spinbox.value()
        )
        self.videomt_window = int(self.videomt_window_spinbox.value())
        self.videomt_input_height = int(self.videomt_input_height_spinbox.value())
        self.videomt_input_width = int(self.videomt_input_width_spinbox.value())
        self.sam3_score_threshold_detection = self.sam3_score_thresh_spinbox.value()
        self.sam3_new_det_thresh = self.sam3_new_det_thresh_spinbox.value()
        self.sam3_propagation_direction = str(self.sam3_direction_combo.currentText())
        self.sam3_max_frame_num_to_track = (
            None
            if self.sam3_max_frames_spinbox.value() == 0
            else int(self.sam3_max_frames_spinbox.value())
        )
        self.sam3_device_override = self.sam3_device_lineedit.text().strip()
        if self.sam3_device_override == "":
            self.sam3_device_override = None
        self.sam3_sliding_window_size = int(self.sam3_window_size_spinbox.value())
        self.sam3_sliding_window_stride = (
            None
            if self.sam3_window_stride_spinbox.value() == 0
            else int(self.sam3_window_stride_spinbox.value())
        )
        self.sam3_max_num_objects = int(self.sam3_max_objects_spinbox.value())
        self.sam3_multiplex_count = int(self.sam3_multiplex_count_spinbox.value())
        self.sam3_agent_det_thresh = self.sam3_agent_det_thresh_spinbox.value()
        self.sam3_agent_window_size = int(self.sam3_agent_window_size_spinbox.value())
        self.sam3_agent_stride = (
            None
            if self.sam3_agent_stride_spinbox.value() == 0
            else int(self.sam3_agent_stride_spinbox.value())
        )
        self.sam3_agent_output_dir = (
            self.sam3_agent_output_dir_lineedit.text().strip() or None
        )
        self.sam3_compile_model = self.sam3_compile_checkbox.isChecked()
        self.sam3_offload_video_to_cpu = self.sam3_offload_checkbox.isChecked()
        self.sam3_use_explicit_window_reseed = (
            self.sam3_explicit_reseed_checkbox.isChecked()
        )
        self.sam3_boundary_mask_match_iou_threshold = (
            self.sam3_boundary_mask_match_iou_spinbox.value()
        )
        self.sam3_allow_private_state_mutation = (
            self.sam3_private_state_checkbox.isChecked()
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

    def is_follow_prediction_progress_enabled(self) -> bool:
        return self.follow_prediction_progress

    def get_optical_flow_backend(self) -> str:
        return normalize_optical_flow_backend(
            self.optical_flow_backend_combo.currentText()
        )

    def get_cutie_weights_path(self) -> Optional[str]:
        return self._current_cutie_spec_from_widgets()

    def get_tracker_settings(self) -> Dict[str, object]:
        return dict(self._tracker_settings)

    def get_videomt_settings(self) -> Dict[str, object]:
        return {
            "videomt_mask_threshold": float(self.videomt_mask_threshold),
            "videomt_logit_threshold": float(self.videomt_logit_threshold),
            "videomt_seed_iou_threshold": float(self.videomt_seed_iou_threshold),
            "videomt_window": int(self.videomt_window),
            "videomt_input_height": int(self.videomt_input_height),
            "videomt_input_width": int(self.videomt_input_width),
        }

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
            "max_num_objects": int(self.sam3_max_num_objects),
            "multiplex_count": int(self.sam3_multiplex_count),
            "compile_model": bool(self.sam3_compile_model),
            "offload_video_to_cpu": bool(self.sam3_offload_video_to_cpu),
            "use_explicit_window_reseed": bool(self.sam3_use_explicit_window_reseed),
            "boundary_mask_match_iou_threshold": float(
                self.sam3_boundary_mask_match_iou_threshold
            ),
            "allow_private_state_mutation": bool(
                self.sam3_allow_private_state_mutation
            ),
            "agent_det_thresh": self.sam3_agent_det_thresh,
            "agent_window_size": self.sam3_agent_window_size,
            "agent_stride": self.sam3_agent_stride,
            "agent_output_dir": self.sam3_agent_output_dir,
        }
