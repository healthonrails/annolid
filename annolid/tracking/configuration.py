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
    structural_consistency_weight: float = 0.99
    appearance_bundle_radius: int = 3
    appearance_bundle_size: int = 30
    appearance_bundle_weight: float = 0.99
    baseline_similarity_weight: float = 0.65
    symmetry_pairs: Tuple[Tuple[str, str], ...] = (('leftear', 'rightear'),)
    symmetry_penalty: float = 0.4
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
