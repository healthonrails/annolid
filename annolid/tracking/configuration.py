"""Typed configuration for the Cutie + DINO tracker pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from annolid.features.dino_models import (
    DEFAULT_DINO_FEATURE_MODEL_ID,
    resolve_dino_model_from_runtime,
    set_dino_model_on_runtime,
)

ProgressHook = Callable[[int, int], None]
ErrorHook = Callable[[Exception], None]

TRACKER_PRESETS: Dict[str, Dict[str, object]] = {
    "rodent_30fps_occlusions": {
        "use_cutie_tracking": True,
        "restrict_to_initial_mask": False,
        "mask_descriptor_weight": 0.15,
        "cutie_mem_every": 4,
        "cutie_frames_to_propagate": 60,
        "cutie_max_mem_frames": 7,
        "mask_dilation_iterations": 3,
        "mask_dilation_kernel": 3,
        "mask_similarity_bonus": 0.25,
        "max_mask_fallback_frames": 10,
        "velocity_smoothing": 0.25,
        "mask_enforce_position": True,
        "mask_enforce_search_radius": 16,
        "mask_enforce_snap_radius": 16,
        "mask_enforce_reject_outside": True,
        "motion_search_tighten": 0.8,
        "motion_search_gain": 0.9,
        "motion_search_flow_gain": None,
        "motion_search_min_radius": 2.5,
        "motion_search_max_radius": 12.0,
        "motion_search_miss_boost": 1.5,
        "motion_prior_penalty_weight": 0.25,
        "motion_prior_soft_radius_px": 20.0,
        "motion_prior_radius_factor": 1.8,
        "motion_prior_miss_relief": 1.25,
        "motion_prior_flow_relief": 0.15,
        "structural_consistency_weight": 0.55,
        "appearance_bundle_radius": 4,
        "appearance_bundle_size": 60,
        "appearance_bundle_weight": 0.65,
        "baseline_similarity_weight": 0.2,
        "context_radius": 1,
        "context_radius_large": 2,
        "context_large_weight": 0.4,
        "context_weight": 0.4,
        "part_shared_weight": 0.2,
        "part_shared_momentum": 0.12,
        "max_candidate_tracks": 12,
        "candidate_prune_ratio": 0.45,
        "candidate_prune_min": 64,
        "support_probe_count": 12,
        "support_probe_sigma": 1.25,
        "support_probe_radius": 4,
        "support_probe_weight": 0.45,
        "support_probe_mask_only": True,
        "support_probe_mask_bonus": 0.05,
        "dinov3_positional_debias": True,
        "dinov3_positional_debias_components": 6,
        "dinov3_positional_debias_strength": 0.35,
        "dinov3_backward_consistency": True,
        "dinov3_backward_consistency_weight": 0.15,
        "dinov3_backward_consistency_tolerance": 1,
        "keypoint_cluster_refine": True,
        "keypoint_cluster_refine_radius": 2,
        "keypoint_cluster_refine_drop": 0.12,
        "keypoint_refine_radius": 1,
        "keypoint_refine_sigma": 1.25,
        "keypoint_refine_temperature": 0.3,
        "pixel_refine_enabled": True,
        "pixel_refine_weight": 0.85,
        "pixel_refine_window": 15,
        "pixel_refine_max_error": 18.0,
        "pixel_refine_max_jump_px": 24.0,
    },
    "fly_70fps_keypoints": {
        "use_cutie_tracking": True,
        "restrict_to_initial_mask": False,
        "mask_descriptor_weight": 0.0,
        "mask_similarity_bonus": 0.08,
        "max_mask_fallback_frames": 4,
        "velocity_smoothing": 0.45,
        "mask_enforce_position": True,
        "mask_enforce_search_radius": 8,
        "mask_enforce_snap_radius": 8,
        "mask_enforce_reject_outside": True,
        "motion_search_tighten": 0.65,
        "motion_search_gain": 1.4,
        "motion_search_flow_gain": 1.4,
        "motion_search_min_radius": 1.0,
        "motion_search_max_radius": 6.0,
        "motion_search_miss_boost": 1.0,
        "motion_prior_penalty_weight": 0.2,
        "motion_prior_soft_radius_px": 8.0,
        "motion_prior_radius_factor": 1.6,
        "motion_prior_miss_relief": 1.0,
        "motion_prior_flow_relief": 0.25,
        "structural_consistency_weight": 0.25,
        "appearance_bundle_radius": 2,
        "appearance_bundle_size": 24,
        "appearance_bundle_weight": 0.45,
        "baseline_similarity_weight": 0.15,
        "context_radius": 1,
        "context_radius_large": 1,
        "context_large_weight": 0.0,
        "context_weight": 0.25,
        "part_shared_weight": 0.1,
        "part_shared_momentum": 0.08,
        "max_candidate_tracks": 8,
        "candidate_prune_ratio": 0.65,
        "candidate_prune_min": 32,
        "support_probe_count": 6,
        "support_probe_sigma": 0.9,
        "support_probe_radius": 2,
        "support_probe_weight": 0.2,
        "support_probe_mask_only": False,
        "support_probe_mask_bonus": 0.0,
        "dinov3_positional_debias": True,
        "dinov3_positional_debias_components": 6,
        "dinov3_positional_debias_strength": 0.5,
        "dinov3_backward_consistency": True,
        "dinov3_backward_consistency_weight": 0.2,
        "dinov3_backward_consistency_tolerance": 1,
        "keypoint_cluster_refine": True,
        "keypoint_cluster_refine_radius": 1,
        "keypoint_cluster_refine_drop": 0.08,
        "keypoint_refine_radius": 1,
        "keypoint_refine_sigma": 0.8,
        "keypoint_refine_temperature": 0.2,
        "pixel_refine_enabled": True,
        "pixel_refine_weight": 0.95,
        "pixel_refine_window": 11,
        "pixel_refine_max_error": 12.0,
        "pixel_refine_max_jump_px": 18.0,
    },
}


@dataclass(slots=True)
class CutieDinoTrackerConfig:
    """Runtime options shared by CLI and GUI entry points."""

    tracker_preset: Optional[str] = None
    use_cutie_tracking: bool = True
    restrict_to_initial_mask: bool = False
    mask_descriptor_weight: float = 0.0
    cutie_mem_every: int = 5
    cutie_frames_to_propagate: int = 30
    cutie_device: Optional[str] = None
    cutie_max_mem_frames: int = 5
    mask_dilation_iterations: int = 2
    mask_dilation_kernel: int = 2
    mask_similarity_bonus: float = 0.12
    max_mask_fallback_frames: int = 5
    velocity_smoothing: float = 0.3
    mask_enforce_position: bool = True
    mask_enforce_search_radius: int = 12
    mask_enforce_snap_radius: Optional[int] = None
    mask_enforce_reject_outside: bool = True
    motion_search_tighten: float = 0.85
    motion_search_gain: float = 0.9
    motion_search_min_radius: float = 2.5
    motion_search_max_radius: float = 12.0
    motion_search_miss_boost: float = 1.0
    motion_search_flow_gain: Optional[float] = None
    motion_prior_penalty_weight: float = 0.3
    motion_prior_soft_radius_px: float = 20.0
    motion_prior_radius_factor: float = 1.5
    motion_prior_miss_relief: float = 1.0
    motion_prior_flow_relief: float = 0.0
    structural_consistency_weight: float = 0.4
    appearance_bundle_radius: int = 4
    appearance_bundle_size: int = 50
    appearance_bundle_weight: float = 0.75
    baseline_similarity_weight: float = 0.3
    context_radius: int = 1
    context_radius_large: int = 2
    context_large_weight: float = 0.35
    context_weight: float = 0.3
    part_shared_weight: float = 0.15
    part_shared_momentum: float = 0.1
    symmetry_pairs: Tuple[Tuple[str, str], ...] = ()
    symmetry_penalty: float = 0.0
    max_candidate_tracks: int = 8
    candidate_prune_ratio: float = 0.5
    candidate_prune_min: int = 48
    support_probe_count: int = 8
    support_probe_sigma: float = 1.25
    support_probe_radius: int = 4
    support_probe_weight: float = 0.35
    support_probe_mask_only: bool = True
    support_probe_mask_bonus: float = 0.05
    dinov3_positional_debias: bool = True
    dinov3_positional_debias_components: int = 6
    dinov3_positional_debias_strength: float = 0.35
    dinov3_backward_consistency: bool = True
    dinov3_backward_consistency_weight: float = 0.15
    dinov3_backward_consistency_tolerance: int = 1
    keypoint_cluster_refine: bool = True
    keypoint_cluster_refine_radius: int = 1
    keypoint_cluster_refine_drop: float = 0.1
    keypoint_refine_radius: int = 0
    keypoint_refine_sigma: float = 1.25
    keypoint_refine_temperature: float = 0.35
    pixel_refine_enabled: bool = True
    pixel_refine_weight: float = 0.85
    pixel_refine_window: int = 15
    pixel_refine_max_error: float = 18.0
    pixel_refine_max_jump_px: float = 24.0
    kpseg_apply_mode: str = "never"  # never|auto|always
    kpseg_min_reliable_frames: int = 5
    kpseg_reliable_ratio: float = 0.6
    kpseg_disable_patience: int = 5
    kpseg_update_tracker_state: bool = True
    kpseg_min_score: float = 0.25
    kpseg_blend_alpha: float = 0.7
    kpseg_use_mask_gate: bool = True
    kpseg_fallback_to_track: bool = True
    kpseg_max_jump_px: float = 0.0
    kpseg_fallback_mode: str = "per_keypoint"  # per_keypoint|instance
    kpseg_fallback_ratio: float = 0.5
    kpseg_smoothing: str = "none"  # none|ema|one_euro|kalman
    kpseg_smoothing_alpha: float = 0.7
    kpseg_smoothing_min_score: float = 0.25
    kpseg_smoothing_fps: Optional[float] = None
    kpseg_one_euro_min_cutoff: float = 1.0
    kpseg_one_euro_beta: float = 0.0
    kpseg_one_euro_d_cutoff: float = 1.0
    kpseg_kalman_process_noise: float = 1e-2
    kpseg_kalman_measurement_noise: float = 1e-1
    kpseg_temporal_fusion_enable: bool = False
    kpseg_temporal_fusion_alpha: float = 0.25
    kpseg_temporal_fusion_low_conf_threshold: float = 0.35
    kpseg_temporal_fusion_max_instances: int = 64
    # Multi-animal tracking configuration (Phase 6)
    tracking_enable: bool = False
    tracking_matcher_algorithm: str = "greedy"  # greedy|hungarian
    tracking_instance_match_distance_px: float = 150.0
    tracking_instance_match_min_similarity: float = 0.3
    tracking_instance_dropout_timeout_frames: int = 30
    tracking_enforce_skeletal_constraints: bool = True
    tracking_constraint_limb_std_multiplier: float = 3.0
    tracking_constraint_max_velocity_px: float = 200.0
    tracking_centroid_weight: float = 1.0
    tracking_pose_weight: float = 0.5
    tracking_size_weight: float = 0.3
    # Temporal smoothing configuration (Phase 7)
    tracking_smoother_enable: bool = False
    tracking_smoother_mode: str = "one_euro"  # ema|one_euro|kalman
    tracking_smoother_fps: Optional[float] = None
    tracking_smoother_ema_alpha: float = 0.7
    tracking_smoother_one_euro_min_cutoff: float = 1.0
    tracking_smoother_one_euro_beta: float = 0.0
    tracking_smoother_kalman_process_noise: float = 1e-2
    tracking_smoother_kalman_measurement_noise: float = 1e-1
    progress_hook: Optional[ProgressHook] = None
    error_hook: Optional[ErrorHook] = None
    analytics_hook: Optional[Callable[[dict], None]] = None
    persist_labelme_json: bool = False
    patch_model_name: str = DEFAULT_DINO_FEATURE_MODEL_ID
    dinov3_model_name: str = DEFAULT_DINO_FEATURE_MODEL_ID

    @classmethod
    def available_presets(cls) -> Tuple[str, ...]:
        return tuple(sorted(TRACKER_PRESETS.keys()))

    @classmethod
    def from_preset(cls, preset: str, **overrides: object) -> "CutieDinoTrackerConfig":
        cfg = cls(tracker_preset=str(preset))
        for key, value in overrides.items():
            if key in cls.__dataclass_fields__:
                setattr(cfg, key, value)
        cfg.normalize()
        return cfg

    def normalize(self) -> None:
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.tracker_preset:
            self._apply_preset_defaults(str(self.tracker_preset))

        set_dino_model_on_runtime(self, resolve_dino_model_from_runtime(self))

        if self.mask_enforce_snap_radius is None:
            self.mask_enforce_snap_radius = int(self.mask_enforce_search_radius)
        else:
            self.mask_enforce_search_radius = int(self.mask_enforce_snap_radius)

        if self.motion_search_flow_gain is None:
            self.motion_search_flow_gain = float(self.motion_search_gain)
        else:
            self.motion_search_flow_gain = float(self.motion_search_flow_gain)

        self.mask_enforce_search_radius = max(1, int(self.mask_enforce_search_radius))
        self.mask_enforce_snap_radius = max(1, int(self.mask_enforce_snap_radius))
        self.mask_enforce_reject_outside = bool(self.mask_enforce_reject_outside)
        self.motion_prior_flow_relief = max(0.0, float(self.motion_prior_flow_relief))

        self.keypoint_refine_radius = max(0, int(self.keypoint_refine_radius))
        self.keypoint_refine_sigma = max(1e-4, float(self.keypoint_refine_sigma))
        self.keypoint_refine_temperature = max(
            1e-4, float(self.keypoint_refine_temperature)
        )
        self.dinov3_positional_debias = bool(self.dinov3_positional_debias)
        self.dinov3_positional_debias_components = max(
            1, int(self.dinov3_positional_debias_components)
        )
        self.dinov3_positional_debias_strength = float(
            min(1.0, max(0.0, float(self.dinov3_positional_debias_strength)))
        )
        if not self.dinov3_positional_debias:
            self.dinov3_positional_debias_strength = 0.0
        self.dinov3_backward_consistency = bool(self.dinov3_backward_consistency)
        self.dinov3_backward_consistency_weight = float(
            min(1.0, max(0.0, float(self.dinov3_backward_consistency_weight)))
        )
        self.dinov3_backward_consistency_tolerance = max(
            0, int(self.dinov3_backward_consistency_tolerance)
        )
        if not self.dinov3_backward_consistency:
            self.dinov3_backward_consistency_weight = 0.0
        self.keypoint_cluster_refine = bool(self.keypoint_cluster_refine)
        self.keypoint_cluster_refine_radius = max(
            0, int(self.keypoint_cluster_refine_radius)
        )
        self.keypoint_cluster_refine_drop = max(
            0.0, float(self.keypoint_cluster_refine_drop)
        )
        self.pixel_refine_enabled = bool(self.pixel_refine_enabled)
        self.pixel_refine_weight = float(
            min(1.0, max(0.0, float(self.pixel_refine_weight)))
        )
        self.pixel_refine_window = max(3, int(self.pixel_refine_window))
        if self.pixel_refine_window % 2 == 0:
            self.pixel_refine_window += 1
        self.pixel_refine_max_error = max(0.0, float(self.pixel_refine_max_error))
        self.pixel_refine_max_jump_px = max(0.0, float(self.pixel_refine_max_jump_px))
        mode = str(self.kpseg_apply_mode or "never").strip().lower()
        if mode not in ("never", "auto", "always"):
            mode = "never"
        self.kpseg_apply_mode = mode
        self.kpseg_min_reliable_frames = max(1, int(self.kpseg_min_reliable_frames))
        self.kpseg_reliable_ratio = float(
            min(1.0, max(0.0, float(self.kpseg_reliable_ratio)))
        )
        self.kpseg_disable_patience = max(1, int(self.kpseg_disable_patience))
        self.kpseg_update_tracker_state = bool(self.kpseg_update_tracker_state)
        self.kpseg_min_score = max(0.0, float(self.kpseg_min_score))
        self.kpseg_blend_alpha = float(
            min(1.0, max(0.0, float(self.kpseg_blend_alpha)))
        )
        self.kpseg_use_mask_gate = bool(self.kpseg_use_mask_gate)
        self.kpseg_fallback_to_track = bool(self.kpseg_fallback_to_track)
        self.kpseg_max_jump_px = max(0.0, float(self.kpseg_max_jump_px))
        fallback_mode = str(self.kpseg_fallback_mode or "per_keypoint").strip().lower()
        if fallback_mode not in ("per_keypoint", "instance"):
            fallback_mode = "per_keypoint"
        self.kpseg_fallback_mode = fallback_mode
        self.kpseg_fallback_ratio = float(
            min(1.0, max(0.0, float(self.kpseg_fallback_ratio)))
        )
        smoothing = str(self.kpseg_smoothing or "none").strip().lower()
        if smoothing not in ("none", "ema", "one_euro", "kalman"):
            smoothing = "none"
        self.kpseg_smoothing = smoothing
        self.kpseg_smoothing_alpha = float(
            min(1.0, max(0.0, float(self.kpseg_smoothing_alpha)))
        )
        self.kpseg_smoothing_min_score = max(0.0, float(self.kpseg_smoothing_min_score))
        if self.kpseg_smoothing_fps is not None:
            self.kpseg_smoothing_fps = max(1e-3, float(self.kpseg_smoothing_fps))
        self.kpseg_one_euro_min_cutoff = max(
            1e-6, float(self.kpseg_one_euro_min_cutoff)
        )
        self.kpseg_one_euro_d_cutoff = max(1e-6, float(self.kpseg_one_euro_d_cutoff))
        self.kpseg_kalman_process_noise = max(
            1e-8, float(self.kpseg_kalman_process_noise)
        )
        self.kpseg_kalman_measurement_noise = max(
            1e-8, float(self.kpseg_kalman_measurement_noise)
        )
        self.kpseg_temporal_fusion_enable = bool(self.kpseg_temporal_fusion_enable)
        self.kpseg_temporal_fusion_alpha = float(
            min(0.95, max(0.0, float(self.kpseg_temporal_fusion_alpha)))
        )
        self.kpseg_temporal_fusion_low_conf_threshold = float(
            min(1.0, max(0.0, float(self.kpseg_temporal_fusion_low_conf_threshold)))
        )
        self.kpseg_temporal_fusion_max_instances = max(
            1, int(self.kpseg_temporal_fusion_max_instances)
        )
        self.part_shared_weight = max(0.0, min(1.0, float(self.part_shared_weight)))
        self.part_shared_momentum = max(0.0, min(1.0, float(self.part_shared_momentum)))
        self.context_radius = max(0, int(self.context_radius))
        self.context_radius_large = max(
            self.context_radius, int(self.context_radius_large)
        )
        self.context_large_weight = max(0.0, min(1.0, float(self.context_large_weight)))
        self.context_weight = max(0.0, min(1.0, float(self.context_weight)))
        self.candidate_prune_ratio = max(
            0.0, min(1.0, float(self.candidate_prune_ratio))
        )
        self.candidate_prune_min = max(0, int(self.candidate_prune_min))

        # Validate multi-animal tracking parameters
        self.tracking_enable = bool(self.tracking_enable)
        matcher = str(self.tracking_matcher_algorithm or "greedy").strip().lower()
        if matcher not in ("greedy", "hungarian"):
            matcher = "greedy"
        self.tracking_matcher_algorithm = matcher
        self.tracking_instance_match_distance_px = max(
            1.0, float(self.tracking_instance_match_distance_px)
        )
        self.tracking_instance_match_min_similarity = float(
            min(1.0, max(0.0, float(self.tracking_instance_match_min_similarity)))
        )
        self.tracking_instance_dropout_timeout_frames = max(
            1, int(self.tracking_instance_dropout_timeout_frames)
        )
        self.tracking_enforce_skeletal_constraints = bool(
            self.tracking_enforce_skeletal_constraints
        )
        self.tracking_constraint_limb_std_multiplier = max(
            1.0, float(self.tracking_constraint_limb_std_multiplier)
        )
        self.tracking_constraint_max_velocity_px = max(
            1.0, float(self.tracking_constraint_max_velocity_px)
        )
        self.tracking_centroid_weight = max(0.0, float(self.tracking_centroid_weight))
        self.tracking_pose_weight = max(0.0, float(self.tracking_pose_weight))
        self.tracking_size_weight = max(0.0, float(self.tracking_size_weight))

        # Validate temporal smoothing parameters
        self.tracking_smoother_enable = bool(self.tracking_smoother_enable)
        smoother_mode = str(self.tracking_smoother_mode or "one_euro").strip().lower()
        if smoother_mode not in ("ema", "one_euro", "kalman"):
            smoother_mode = "one_euro"
        self.tracking_smoother_mode = smoother_mode
        if self.tracking_smoother_fps is not None:
            self.tracking_smoother_fps = max(1e-3, float(self.tracking_smoother_fps))
        self.tracking_smoother_ema_alpha = float(
            min(1.0, max(0.0, float(self.tracking_smoother_ema_alpha)))
        )
        self.tracking_smoother_one_euro_min_cutoff = max(
            1e-6, float(self.tracking_smoother_one_euro_min_cutoff)
        )
        self.tracking_smoother_one_euro_beta = max(
            0.0, float(self.tracking_smoother_one_euro_beta)
        )
        self.tracking_smoother_kalman_process_noise = max(
            1e-8, float(self.tracking_smoother_kalman_process_noise)
        )
        self.tracking_smoother_kalman_measurement_noise = max(
            1e-8, float(self.tracking_smoother_kalman_measurement_noise)
        )

    def _apply_preset_defaults(self, preset: str) -> None:
        preset_values = TRACKER_PRESETS.get(preset)
        if preset_values is None:
            raise ValueError(
                f"Unknown tracker preset '{preset}'. "
                f"Available presets: {', '.join(sorted(TRACKER_PRESETS)) or '<none>'}"
            )

        for key, value in preset_values.items():
            field = self.__dataclass_fields__.get(key)
            if field is None:
                continue
            current_value = getattr(self, key)
            default_value = field.default
            if current_value == default_value:
                setattr(self, key, value)
