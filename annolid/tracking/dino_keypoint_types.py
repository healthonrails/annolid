"""Core data structures and configuration options for Cutie + DINO keypoint tracking."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from annolid.tracking.configuration import CutieDinoTrackerConfig

__all__ = [
    "BodyPrior",
    "Candidate",
    "FrameMotionEstimate",
    "KeypointTrack",
    "MotionPrior",
    "PixelFlowEstimate",
    "PixelRefineOptions",
    "RefineRegion",
    "SupportProbe",
]


@dataclass
class SupportProbe:
    offset_rc: Tuple[int, int]
    descriptor: torch.Tensor
    weight: float
    in_mask: bool = False


@dataclass
class KeypointTrack:
    """Internal state for a single tracked keypoint."""

    key: str
    storage_label: str
    instance_label: str
    display_label: str
    patch_rc: Tuple[int, int]
    descriptor: torch.Tensor
    reference_descriptor: torch.Tensor
    context_descriptor: Optional[torch.Tensor] = None
    manual_codebook: Optional[torch.Tensor] = None
    velocity: Tuple[float, float] = (0.0, 0.0)
    misses: int = 0
    struct_refs: Dict[str, float] = field(default_factory=dict)
    last_position: Tuple[float, float] = (0.0, 0.0)
    quality: float = 1.0
    appearance_codebook: Optional[torch.Tensor] = None
    baseline_similarity: float = 1.0
    symmetry_partner: Optional[str] = None
    symmetry_axis: Optional[Tuple[float, float]] = None
    symmetry_midpoint: Optional[Tuple[float, float]] = None
    symmetry_sign: float = 0.0
    support_probes: List[SupportProbe] = field(default_factory=list)
    body_coords: Optional[Tuple[float, float, float]] = None
    pixel_template: Optional[Any] = None
    seed_position: Optional[Tuple[float, float]] = None
    template_heading_rad: float = 0.0
    last_template_angle_rad: float = 0.0


@dataclass
class Candidate:
    rc: Tuple[int, int]
    xy: Tuple[float, float]
    score: float
    similarity: float
    descriptor: torch.Tensor


@dataclass
class RefineRegion:
    logits: np.ndarray
    r_min: int
    c_min: int
    valid_mask: Optional[np.ndarray] = None


@dataclass
class MotionPrior:
    predicted_xy: Tuple[float, float]
    predicted_rc: Tuple[int, int]
    base_xy: Tuple[float, float]
    flow_vec: Optional[Tuple[float, float]]
    flow_speed: float
    velocity_speed: float
    radius: int
    radius_px: float
    confidence: float


@dataclass
class PixelFlowEstimate:
    xy: Tuple[float, float]
    error: float


@dataclass(frozen=True)
class FrameMotionEstimate:
    """Current-frame point estimate and the observed displacement behind it."""

    predicted_xy: Tuple[float, float]
    flow_vec: Optional[Tuple[float, float]]
    pixel_flow: Optional[PixelFlowEstimate]


@dataclass(frozen=True)
class PixelRefineOptions:
    enabled: bool
    weight: float
    window: int
    max_error: float
    max_jump_px: float

    @classmethod
    def from_runtime(
        cls, runtime_config: CutieDinoTrackerConfig
    ) -> "PixelRefineOptions":
        window = max(3, int(getattr(runtime_config, "pixel_refine_window", 15)))
        if window % 2 == 0:
            window += 1
        return cls(
            enabled=bool(getattr(runtime_config, "pixel_refine_enabled", True)),
            weight=float(
                np.clip(
                    getattr(runtime_config, "pixel_refine_weight", 0.85),
                    0.0,
                    1.0,
                )
            ),
            window=int(window),
            max_error=max(
                0.0, float(getattr(runtime_config, "pixel_refine_max_error", 18.0))
            ),
            max_jump_px=max(
                0.0,
                float(getattr(runtime_config, "pixel_refine_max_jump_px", 24.0)),
            ),
        )

    def error_confidence(self, error: float) -> float:
        """Map LK error to confidence; a zero limit disables error gating."""
        error = float(error)
        if not math.isfinite(error):
            return 0.0
        if self.max_error <= 0.0:
            return 1.0
        return float(np.clip(1.0 - (max(0.0, error) / self.max_error), 0.0, 1.0))


@dataclass
class BodyPrior:
    centroid_rc: Tuple[float, float]
    axis_rc: Tuple[float, float]
    perp_rc: Tuple[float, float]
    sigma_axis: float
    sigma_perp: float
    thickness_map: np.ndarray
    sigma_thickness: float

    def coords(self, rc: Tuple[float, float]) -> Tuple[float, float]:
        dr = float(rc[0] - self.centroid_rc[0])
        dc = float(rc[1] - self.centroid_rc[1])
        t = dr * float(self.axis_rc[0]) + dc * float(self.axis_rc[1])
        p = dr * float(self.perp_rc[0]) + dc * float(self.perp_rc[1])
        return (t, p)

    def signature(self, rc: Tuple[float, float]) -> Tuple[float, float, float]:
        t, p = self.coords(rc)
        rr = int(round(float(rc[0])))
        cc = int(round(float(rc[1])))
        if (
            rr < 0
            or cc < 0
            or rr >= int(self.thickness_map.shape[0])
            or cc >= int(self.thickness_map.shape[1])
        ):
            thickness = 0.0
        else:
            thickness = float(self.thickness_map[rr, cc])
        return (t, p, thickness)
