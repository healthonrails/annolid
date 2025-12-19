"""Typed configuration for the Cutie + DINO tracker pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

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
        "keypoint_refine_radius": 1,
        "keypoint_refine_sigma": 1.25,
        "keypoint_refine_temperature": 0.3,
    }
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
    keypoint_refine_radius: int = 0
    keypoint_refine_sigma: float = 1.25
    keypoint_refine_temperature: float = 0.35
    progress_hook: Optional[ProgressHook] = None
    error_hook: Optional[ErrorHook] = None
    analytics_hook: Optional[Callable[[dict], None]] = None
    persist_labelme_json: bool = False

    @classmethod
    def available_presets(cls) -> Tuple[str, ...]:
        return tuple(sorted(TRACKER_PRESETS.keys()))

    @classmethod
    def from_preset(
        cls, preset: str, **overrides: object
    ) -> "CutieDinoTrackerConfig":
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

        if self.mask_enforce_snap_radius is None:
            self.mask_enforce_snap_radius = int(
                self.mask_enforce_search_radius)
        else:
            self.mask_enforce_search_radius = int(
                self.mask_enforce_snap_radius)

        if self.motion_search_flow_gain is None:
            self.motion_search_flow_gain = float(self.motion_search_gain)
        else:
            self.motion_search_flow_gain = float(self.motion_search_flow_gain)

        self.mask_enforce_search_radius = max(
            1, int(self.mask_enforce_search_radius))
        self.mask_enforce_snap_radius = max(
            1, int(self.mask_enforce_snap_radius))
        self.mask_enforce_reject_outside = bool(
            self.mask_enforce_reject_outside)
        self.motion_prior_flow_relief = max(
            0.0, float(self.motion_prior_flow_relief))

        self.keypoint_refine_radius = max(0, int(self.keypoint_refine_radius))
        self.keypoint_refine_sigma = max(
            1e-4, float(self.keypoint_refine_sigma))
        self.keypoint_refine_temperature = max(
            1e-4, float(self.keypoint_refine_temperature)
        )
        self.part_shared_weight = max(
            0.0, min(1.0, float(self.part_shared_weight)))
        self.part_shared_momentum = max(
            0.0, min(1.0, float(self.part_shared_momentum)))
        self.context_radius = max(0, int(self.context_radius))
        self.context_radius_large = max(
            self.context_radius, int(self.context_radius_large))
        self.context_large_weight = max(
            0.0, min(1.0, float(self.context_large_weight)))
        self.context_weight = max(
            0.0, min(1.0, float(self.context_weight)))
        self.candidate_prune_ratio = max(
            0.0, min(1.0, float(self.candidate_prune_ratio)))
        self.candidate_prune_min = max(0, int(self.candidate_prune_min))

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
