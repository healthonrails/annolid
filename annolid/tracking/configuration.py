"""Typed configuration for the Cutie + DINO tracker pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

ProgressHook = Callable[[int, int], None]
ErrorHook = Callable[[Exception], None]


@dataclass(slots=True)
class CutieDinoTrackerConfig:
    """Runtime options shared by CLI and GUI entry points."""

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
    motion_search_gain: float = 0.6
    motion_search_min_radius: float = 1.0
    motion_search_max_radius: float = 8.0
    motion_search_miss_boost: float = 1.0
    motion_search_flow_gain: Optional[float] = None
    motion_prior_penalty_weight: float = 0.3
    motion_prior_soft_radius_px: float = 12.0
    motion_prior_radius_factor: float = 1.5
    motion_prior_miss_relief: float = 0.75
    motion_prior_flow_relief: float = 0.0
    structural_consistency_weight: float = 0.4
    appearance_bundle_radius: int = 3
    appearance_bundle_size: int = 30
    appearance_bundle_weight: float = 0.99
    baseline_similarity_weight: float = 0.65
    symmetry_pairs: Tuple[Tuple[str, str], ...] = (('leftear', 'rightear'),)
    symmetry_penalty: float = 0.0
    max_candidate_tracks: int = 8
    support_probe_count: int = 8
    support_probe_sigma: float = 1.25
    support_probe_radius: int = 4
    support_probe_weight: float = 0.35
    support_probe_mask_only: bool = True
    support_probe_mask_bonus: float = 0.05
    progress_hook: Optional[ProgressHook] = None
    error_hook: Optional[ErrorHook] = None
    analytics_hook: Optional[Callable[[dict], None]] = None

    def __post_init__(self) -> None:
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
