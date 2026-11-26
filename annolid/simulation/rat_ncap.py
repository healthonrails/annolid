"""Rat NCAP spinal circuit controller, neuromechanical interfaces, and Rat v2 body spec.

Faithful pure-NumPy implementation of the Rat NCAP architecture and Rat v2
biomechanical model from:
  Bhattasali, Cui, Pinto, and Lindsay (2026),
  "Neural Circuit Architectural Priors for Quadruped Locomotion"
  (arXiv:2611.23364v1; https://github.com/rat-ncap/rat-ncap).

This module intentionally avoids top-level imports of optional ML or physics
dependencies (torch, jax, mujoco, dm_control, scipy) so lean Annolid installs
and frozen desktop bundles import it cleanly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Metadata & canonical Rat v2 biomechanical specification
# ---------------------------------------------------------------------------

RAT_NCAP_METADATA: Dict[str, Any] = {
    "id": "rat-ncap-v2",
    "paper_arxiv": "2611.23364v1",
    "paper_title": "Neural Circuit Architectural Priors for Quadruped Locomotion",
    "repo_url": "https://github.com/rat-ncap/rat-ncap",
    "body_model": "Rat v2 (38 actuated DOFs; 31 locomotion-active DOFs)",
    "license": "MIT (controller & skins) / Apache-2.0 (dm_control rodent lineage)",
}

RAT_LIMB_NAMES: Tuple[str, ...] = ("FL", "FR", "HL", "HR")

RAT_V2_SEGMENTS: Dict[str, Tuple[str, ...]] = {
    "head": ("mandible", "atlas"),
    "cervicalE": ("cervical_extend",),
    "cervicalBT": ("cervical_bend", "cervical_twist"),
    "limbFL": (
        "scapula_L_supinate",
        "scapula_L_abduct",
        "scapula_L_extend",
        "shoulder_L_supinate",
        "shoulder_L_extend",
        "elbow_L",
        "wrist_L",
        "finger_L",
    ),
    "limbFR": (
        "scapula_R_supinate",
        "scapula_R_abduct",
        "scapula_R_extend",
        "shoulder_R_supinate",
        "shoulder_R_extend",
        "elbow_R",
        "wrist_R",
        "finger_R",
    ),
    "lumbarE": ("lumbar_extend",),
    "lumbarBT": ("lumbar_bend", "lumbar_twist"),
    "limbHL": (
        "hip_L_supinate",
        "hip_L_abduct",
        "hip_L_extend",
        "knee_L",
        "ankle_L",
        "toe_L",
    ),
    "limbHR": (
        "hip_R_supinate",
        "hip_R_abduct",
        "hip_R_extend",
        "knee_R",
        "ankle_R",
        "toe_R",
    ),
    "tailE": ("tail_extend",),
    "tailB": ("tail_bend",),
}

RAT_V2_LOCOMOTION_SEGMENTS: Tuple[str, ...] = (
    "limbFL",
    "limbFR",
    "lumbarE",
    "lumbarBT",
    "limbHL",
    "limbHR",
)

RAT_V2_ACTUATOR_NAMES: Tuple[str, ...] = tuple(
    name for seg in RAT_V2_SEGMENTS.values() for name in seg
)
RAT_V2_LOCOMOTION_ACTUATOR_NAMES: Tuple[str, ...] = tuple(
    name for seg in RAT_V2_LOCOMOTION_SEGMENTS for name in RAT_V2_SEGMENTS[seg]
)

RAT_V2_ROOT_POS_INIT: Tuple[float, float, float] = (0.01635, 0.00062, 0.05109)
RAT_V2_ROOT_QUAT_INIT: Tuple[float, float, float, float] = (
    0.99818,
    0.00005,
    -0.06028,
    0.0011,
)

# Position midpoints (rad for hinge joints, m/normalized for tendons) from rat.py
RAT_V2_POS_MID: Dict[str, Tuple[float, ...]] = {
    "head": tuple(float(v) for v in np.deg2rad([-36.0, 20.0])),
    "cervicalE": (-0.33,),
    "cervicalBT": (0.0, 0.0),
    "limbFL": tuple(
        float(v) for v in np.deg2rad([4.5, 0.0, 0.0, -16.25, 20.0, -2.0, 61.2, -65.0])
    ),
    "limbFR": tuple(
        float(v) for v in np.deg2rad([4.5, 0.0, 0.0, -16.25, 20.0, -2.0, 61.2, -65.0])
    ),
    "lumbarE": (0.1,),
    "lumbarBT": (0.0, 0.0),
    "limbHL": tuple(
        float(v) for v in np.deg2rad([-1.5, 12.75, -15.0, -35.2, 14.0, 0.0])
    ),
    "limbHR": tuple(
        float(v) for v in np.deg2rad([-1.5, 12.75, -15.0, -35.2, 14.0, 0.0])
    ),
    "tailE": (0.0,),
    "tailB": (0.0,),
}

# Joint/tendon ranges (low, high) in radians (tendons in [-1, 1]) from rat.xml
RAT_V2_JOINT_RANGES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "head": (
        (float(np.deg2rad(-17.0)), float(np.deg2rad(15.0))),
        (float(np.deg2rad(-5.0)), float(np.deg2rad(45.0))),
    ),
    "cervicalE": ((-1.0, 1.0),),
    "cervicalBT": ((-1.0, 1.0), (-1.0, 1.0)),
    "limbFL": (
        (float(np.deg2rad(-1.0)), float(np.deg2rad(10.0))),
        (float(np.deg2rad(-15.0)), float(np.deg2rad(15.0))),
        (float(np.deg2rad(-2.0)), float(np.deg2rad(2.0))),
        (float(np.deg2rad(-65.0)), float(np.deg2rad(3.0))),
        (float(np.deg2rad(-75.0)), float(np.deg2rad(50.0))),
        (float(np.deg2rad(-75.0)), float(np.deg2rad(40.0))),
        (float(np.deg2rad(-20.0)), float(np.deg2rad(90.0))),
        (float(np.deg2rad(-180.0)), float(np.deg2rad(0.0))),
    ),
    "limbFR": (
        (float(np.deg2rad(-1.0)), float(np.deg2rad(10.0))),
        (float(np.deg2rad(-15.0)), float(np.deg2rad(15.0))),
        (float(np.deg2rad(-2.0)), float(np.deg2rad(2.0))),
        (float(np.deg2rad(-65.0)), float(np.deg2rad(3.0))),
        (float(np.deg2rad(-75.0)), float(np.deg2rad(50.0))),
        (float(np.deg2rad(-75.0)), float(np.deg2rad(40.0))),
        (float(np.deg2rad(-20.0)), float(np.deg2rad(90.0))),
        (float(np.deg2rad(-180.0)), float(np.deg2rad(0.0))),
    ),
    "lumbarE": ((-1.0, 1.0),),
    "lumbarBT": ((-1.0, 1.0), (-1.0, 1.0)),
    "limbHL": (
        (float(np.deg2rad(-40.0)), float(np.deg2rad(30.0))),
        (float(np.deg2rad(-30.0)), float(np.deg2rad(60.0))),
        (float(np.deg2rad(-90.0)), float(np.deg2rad(60.0))),
        (float(np.deg2rad(-70.0)), float(np.deg2rad(88.0))),
        (float(np.deg2rad(0.0)), float(np.deg2rad(80.0))),
        (float(np.deg2rad(-100.0)), float(np.deg2rad(50.0))),
    ),
    "limbHR": (
        (float(np.deg2rad(-40.0)), float(np.deg2rad(30.0))),
        (float(np.deg2rad(-30.0)), float(np.deg2rad(60.0))),
        (float(np.deg2rad(-90.0)), float(np.deg2rad(60.0))),
        (float(np.deg2rad(-70.0)), float(np.deg2rad(88.0))),
        (float(np.deg2rad(0.0)), float(np.deg2rad(80.0))),
        (float(np.deg2rad(-100.0)), float(np.deg2rad(50.0))),
    ),
    "tailE": ((-1.0, 1.0),),
    "tailB": ((-1.0, 1.0),),
}

# Stiffness (keq / kp) & max spring-muscle torque from body/rat/controller/flxext.yaml
RAT_V2_KEQ: Dict[str, Tuple[float, ...]] = {
    "head": (0.1, 0.05),
    "cervicalE": (0.2,),
    "cervicalBT": (0.2, 0.2),
    "limbFL": (0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.15, 0.03),
    "limbFR": (0.5, 0.5, 0.5, 0.3, 0.3, 0.2, 0.15, 0.03),
    "lumbarE": (0.1,),
    "lumbarBT": (0.1, 0.1),
    "limbHL": (0.4, 0.4, 0.6, 0.4, 0.2, 0.03),
    "limbHR": (0.4, 0.4, 0.6, 0.4, 0.2, 0.03),
    "tailE": (0.1,),
    "tailB": (0.1,),
}

RAT_V2_TRQ_MAX: Dict[str, Tuple[float, ...]] = {
    "head": (0.1, 0.05),
    "cervicalE": (1.0,),
    "cervicalBT": (0.5, 0.5),
    "limbFL": (0.4, 0.4, 0.4, 1.0, 1.5, 1.0, 0.29, 0.05),
    "limbFR": (0.4, 0.4, 0.4, 1.0, 1.5, 1.0, 0.29, 0.05),
    "lumbarE": (2.0,),
    "lumbarBT": (0.2, 0.2),
    "limbHL": (0.49, 0.63, 1.58, 1.11, 0.28, 0.08),
    "limbHR": (0.49, 0.63, 1.58, 1.11, 0.28, 0.08),
    "tailE": (0.5,),
    "tailB": (0.5,),
}

RAT_V2_KD: Dict[str, Tuple[float, ...]] = {
    "head": (0.002, 0.002),
    "cervicalE": (0.05,),
    "cervicalBT": (0.05, 0.05),
    "limbFL": (0.005, 0.005, 0.005, 0.005, 0.005, 0.003, 0.002, 0.002),
    "limbFR": (0.005, 0.005, 0.005, 0.005, 0.005, 0.003, 0.002, 0.002),
    "lumbarE": (0.03,),
    "lumbarBT": (0.03, 0.03),
    "limbHL": (0.02, 0.02, 0.02, 0.005, 0.003, 0.001),
    "limbHR": (0.02, 0.02, 0.02, 0.005, 0.003, 0.001),
    "tailE": (0.002,),
    "tailB": (0.002,),
}

RAT_V2_COAT_PATTERNS: Tuple[str, ...] = ("black_hooded", "cinnamon", "gray")
SUPPORTED_RAT_INTERFACES: Tuple[str, ...] = (
    "flexor_extensor",
    "position_stiffness",
    "position",
    "torque",
)
SUPPORTED_RAT_GAITS: Tuple[str, ...] = ("stand", "walk", "trot", "bound")
SUPPORTED_RAT_BEHAVIORS: Tuple[str, ...] = (
    "stand",
    "walk",
    "trot",
    "bound",
    "gait_transition",
    "bumpy_terrain",
    "v0_ablation_bound",
    "v3_ablation_trot",
    "open_field_explore",
)
SUPPORTED_RAT_ABLATIONS: Tuple[str, ...] = (
    "none",
    "v0v_v0d",
    "v0d",
    "v0v",
    "v3f_v3a",
    "v3f",
    "v3a",
)
RAT_V2_CAMERA_VIEWS: Tuple[str, ...] = (
    "track_right",
    "track_left",
    "track_front",
    "track_back",
    "track_top",
    "oblique",
    "ventral",
    "egocentric",
)

RAT_V2_KEYPOINT_NAMES: Tuple[str, ...] = (
    "nose",
    "head",
    "left_ear",
    "right_ear",
    "neck",
    "thorax",
    "lumbar",
    "pelvis",
    "tail_base",
    "tail_mid",
    "tail_tip",
    "left_forepaw",
    "right_forepaw",
    "left_hindpaw",
    "right_hindpaw",
    "torso_center",
)

RAT_V2_KEYPOINT_TO_SITE: Dict[str, str] = {
    "nose": "nose_site",
    "head": "head",
    "left_ear": "left_ear_site",
    "right_ear": "right_ear_site",
    "neck": "cervical_site",
    "thorax": "torso",
    "torso_center": "torso",
    "lumbar": "lumbar_site",
    "pelvis": "pelvis",
    "tail_base": "tail_base_site",
    "tail_mid": "tail_mid_site",
    "tail_tip": "tail_tip_site",
    "left_forepaw": "finger_sensor_L",
    "right_forepaw": "finger_sensor_R",
    "left_hindpaw": "toe_sensor_L",
    "right_hindpaw": "toe_sensor_R",
}

RAT_V2_RIGID_BONE_EDGES: Tuple[Tuple[str, str], ...] = (
    ("nose", "head"),
    ("head", "left_ear"),
    ("head", "right_ear"),
    ("head", "neck"),
    ("neck", "thorax"),
    ("thorax", "torso_center"),
    ("torso_center", "lumbar"),
    ("lumbar", "pelvis"),
    ("pelvis", "tail_base"),
    ("tail_base", "tail_mid"),
    ("tail_mid", "tail_tip"),
)

RAT_V2_SKELETON_EDGES: Tuple[Tuple[str, str], ...] = (
    *RAT_V2_RIGID_BONE_EDGES,
    ("thorax", "left_forepaw"),
    ("thorax", "right_forepaw"),
    ("pelvis", "left_hindpaw"),
    ("pelvis", "right_hindpaw"),
)

# Canonical ablation groups matching configs/2_test/2_2_ablation_v0v_v0d.yaml
RAT_ABLATION_POPULATIONS: Dict[str, Tuple[str, ...]] = {
    "none": (),
    "v0v_v0d": (
        "cross_v0d_FL",
        "cross_v0d_FR",
        "cross_v0d_HL",
        "cross_v0d_HR",
        "diagd_v0d_FL",
        "diagd_v0d_FR",
        "diagd_v0v_FL",
        "diagd_v0v_FR",
    ),
    "v0d": (
        "cross_v0d_FL",
        "cross_v0d_FR",
        "cross_v0d_HL",
        "cross_v0d_HR",
        "diagd_v0d_FL",
        "diagd_v0d_FR",
    ),
    "v0v": ("diagd_v0v_FL", "diagd_v0v_FR"),
    "v3f_v3a": (
        "cross_v3f_FL",
        "cross_v3f_FR",
        "cross_v3f_HL",
        "cross_v3f_HR",
        "diaga_v3a_HL",
        "diaga_v3a_HR",
    ),
    "v3f": (
        "cross_v3f_FL",
        "cross_v3f_FR",
        "cross_v3f_HL",
        "cross_v3f_HR",
    ),
    "v3a": ("diaga_v3a_HL", "diaga_v3a_HR"),
}


def normalize_rat_interface(name: str | None) -> str:
    raw = str(name or "flexor_extensor").strip().lower().replace("-", "_")
    aliases = {
        "flxext": "flexor_extensor",
        "flexor_extensor": "flexor_extensor",
        "spring_muscle": "flexor_extensor",
        "poskp": "position_stiffness",
        "position_stiffness": "position_stiffness",
        "pos": "position",
        "position": "position",
        "trq": "torque",
        "torque": "torque",
    }
    resolved = aliases.get(raw)
    if resolved is None:
        raise ValueError(
            f"Unsupported Rat NCAP interface {name!r}; expected one of {SUPPORTED_RAT_INTERFACES}."
        )
    return resolved


def normalize_rat_coat(name: str | None) -> str:
    raw = (
        str(name or "black_hooded").strip().lower().replace("-", "_").replace(" ", "_")
    )
    aliases = {
        "blackhooded": "black_hooded",
        "black_hooded": "black_hooded",
        "long_evans": "black_hooded",
        "hooded": "black_hooded",
        "cinnamon": "cinnamon",
        "brown": "cinnamon",
        "gray": "gray",
        "grey": "gray",
    }
    return aliases.get(raw, "black_hooded")


def default_speed_for_behavior(behavior: str | None) -> float:
    key = str(behavior or "trot").strip().lower()
    mapping = {
        "stand": 0.0,
        "walk": 0.22,
        "trot": 0.50,
        "bound": 0.88,
        "gait_transition": 0.50,
        "bumpy_terrain": 0.48,
        "v0_ablation_bound": 0.45,
        "v3_ablation_trot": 0.45,
        "open_field_explore": 0.36,
    }
    return float(mapping.get(key, 0.50))


# ---------------------------------------------------------------------------
# Neuromechanical Interfaces (Appendix C.2 & composer/transforms.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SpringMuscleParams:
    """Exact reparameterized antagonistic spring-muscle constants for a segment."""

    low: np.ndarray
    high: np.ndarray
    poseq: np.ndarray
    keq: np.ndarray
    trq_max_flx: np.ndarray
    trq_max_ext: np.ndarray
    kf: np.ndarray
    ke: np.ndarray
    cf: np.ndarray
    ce: np.ndarray


def compute_spring_muscle_params(
    *,
    low: Sequence[float] | np.ndarray,
    high: Sequence[float] | np.ndarray,
    poseq: Sequence[float] | np.ndarray,
    keq: Sequence[float] | np.ndarray,
    trq_max_flx: Sequence[float] | np.ndarray,
    trq_max_ext: Sequence[float] | np.ndarray,
) -> SpringMuscleParams:
    """Compute exact Rat NCAP spring-muscle parameters `(kf, ke, cf, ce)`.

    Matches `composer.transforms.SpringMuscleController.initialize`:
      - At `flx = ext = 1`: joint rests at `poseq` with net stiffness `keq`.
      - At `flx = 1, ext = 0, q = high`: torque is `-trq_max_flx`.
      - At `flx = 0, ext = 1, q = low`: torque is `+trq_max_ext`.
    """
    low_arr = np.asarray(low, dtype=np.float64)
    high_arr = np.asarray(high, dtype=np.float64)
    poseq_arr = np.asarray(poseq, dtype=np.float64)
    keq_arr = np.asarray(keq, dtype=np.float64)
    tmax_f = np.asarray(trq_max_flx, dtype=np.float64)
    tmax_e = np.asarray(trq_max_ext, dtype=np.float64)
    span = np.maximum(high_arr - low_arr, 1e-6)
    kf = (keq_arr * (poseq_arr - low_arr) + (tmax_f - tmax_e)) / span
    ke = (keq_arr * (high_arr - poseq_arr) + (tmax_e - tmax_f)) / span
    # Guard against near-zero denominator for extreme asymmetric inputs.
    kf_safe = np.where(np.abs(kf) < 1e-9, np.sign(kf + 1e-12) * 1e-9, kf)
    ke_safe = np.where(np.abs(ke) < 1e-9, np.sign(ke + 1e-12) * 1e-9, ke)
    cf = tmax_f / kf_safe - span
    ce = tmax_e / ke_safe - span
    return SpringMuscleParams(
        low=low_arr,
        high=high_arr,
        poseq=poseq_arr,
        keq=keq_arr,
        trq_max_flx=tmax_f,
        trq_max_ext=tmax_e,
        kf=kf,
        ke=ke,
        cf=cf,
        ce=ce,
    )


class NeuromechanicalInterface:
    """Implements the 4 Rat NCAP neuromechanical interfaces on all locomotion DOFs."""

    def __init__(
        self,
        interface_type: str = "flexor_extensor",
        *,
        segments: Sequence[str] = RAT_V2_LOCOMOTION_SEGMENTS,
        pos_margin: float = 0.2,
    ) -> None:
        self.interface_type = normalize_rat_interface(interface_type)
        self.segments = tuple(segments)
        self.pos_margin = float(pos_margin)

        low_list: list[float] = []
        high_list: list[float] = []
        poseq_list: list[float] = []
        keq_list: list[float] = []
        tmax_list: list[float] = []
        kd_list: list[float] = []
        names: list[str] = []
        for seg in self.segments:
            names.extend(RAT_V2_SEGMENTS[seg])
            for lo, hi in RAT_V2_JOINT_RANGES[seg]:
                low_list.append(lo)
                high_list.append(hi)
            poseq_list.extend(RAT_V2_POS_MID[seg])
            keq_list.extend(RAT_V2_KEQ[seg])
            tmax_list.extend(RAT_V2_TRQ_MAX[seg])
            kd_list.extend(RAT_V2_KD[seg])

        self.actuator_names = tuple(names)
        self.dof = len(self.actuator_names)
        self.low = np.asarray(low_list, dtype=np.float64)
        self.high = np.asarray(high_list, dtype=np.float64)
        self.poseq = np.asarray(poseq_list, dtype=np.float64)
        self.keq = np.asarray(keq_list, dtype=np.float64)
        self.trq_max = np.asarray(tmax_list, dtype=np.float64)
        self.kd = np.asarray(kd_list, dtype=np.float64)
        self.spring = compute_spring_muscle_params(
            low=self.low,
            high=self.high,
            poseq=self.poseq,
            keq=self.keq,
            trq_max_flx=self.trq_max,
            trq_max_ext=self.trq_max,
        )

    def compute_torque(
        self,
        *,
        q: np.ndarray,
        dq: np.ndarray,
        flx: np.ndarray | None = None,
        ext: np.ndarray | None = None,
        q_target: np.ndarray | None = None,
        kp_scale: np.ndarray | None = None,
        trq_cmd: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute joint torques under the configured neuromechanical interface."""
        q_arr = np.asarray(q, dtype=np.float64)
        dq_arr = np.asarray(dq, dtype=np.float64)
        if self.interface_type == "flexor_extensor":
            flx_a = np.clip(
                np.zeros_like(q_arr)
                if flx is None
                else np.asarray(flx, dtype=np.float64),
                0.0,
                1.0,
            )
            ext_a = np.clip(
                np.zeros_like(q_arr)
                if ext is None
                else np.asarray(ext, dtype=np.float64),
                0.0,
                1.0,
            )
            xf = q_arr - self.spring.low + self.spring.cf
            xe = self.spring.high - q_arr + self.spring.ce
            trq = ext_a * self.spring.ke * xe - flx_a * self.spring.kf * xf
            trq = trq - self.kd * dq_arr
        elif self.interface_type == "position":
            span = self.high - self.low
            lo = self.low - self.pos_margin * span
            hi = self.high + self.pos_margin * span
            qd = (
                self.poseq
                if q_target is None
                else np.clip(np.asarray(q_target, dtype=np.float64), lo, hi)
            )
            trq = self.keq * (qd - q_arr) - self.kd * dq_arr
        elif self.interface_type == "position_stiffness":
            span = self.high - self.low
            lo = self.low - self.pos_margin * span
            hi = self.high + self.pos_margin * span
            qd = (
                self.poseq
                if q_target is None
                else np.clip(np.asarray(q_target, dtype=np.float64), lo, hi)
            )
            scale = (
                np.ones_like(q_arr)
                if kp_scale is None
                else np.clip(np.asarray(kp_scale, dtype=np.float64), 0.0, 1.0)
            )
            trq = (self.keq * scale) * (qd - q_arr) - self.kd * dq_arr
        else:  # torque
            trq = (
                np.zeros_like(q_arr)
                if trq_cmd is None
                else np.asarray(trq_cmd, dtype=np.float64)
            )
        return np.clip(trq, -self.trq_max, self.trq_max)

    def effective_stiffness(
        self,
        *,
        flx: np.ndarray | None = None,
        ext: np.ndarray | None = None,
        kp_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.interface_type == "flexor_extensor":
            flx_a = np.clip(
                np.zeros(self.dof)
                if flx is None
                else np.asarray(flx, dtype=np.float64),
                0.0,
                1.0,
            )
            ext_a = np.clip(
                np.zeros(self.dof)
                if ext is None
                else np.asarray(ext, dtype=np.float64),
                0.0,
                1.0,
            )
            return ext_a * self.spring.ke + flx_a * self.spring.kf
        if self.interface_type == "position_stiffness":
            scale = (
                np.ones(self.dof)
                if kp_scale is None
                else np.clip(np.asarray(kp_scale, dtype=np.float64), 0.0, 1.0)
            )
            return self.keq * scale
        if self.interface_type == "position":
            return self.keq.copy()
        return np.zeros(self.dof, dtype=np.float64)


def compute_mechanical_power(
    torque: Sequence[float] | np.ndarray,
    velocity: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    """Compute signed, positive, and negative joint mechanical power (Watts)."""
    tau = np.asarray(torque, dtype=np.float64)
    dq = np.asarray(velocity, dtype=np.float64)
    p = tau * dq
    pos = float(np.sum(np.maximum(p, 0.0)))
    neg = float(np.sum(np.minimum(p, 0.0)))
    return {
        "positive_power_w": pos,
        "negative_power_w": neg,
        "net_power_w": pos + neg,
        "abs_power_w": pos - neg,
    }


# ---------------------------------------------------------------------------
# Mammalian Spinal Rhythm Generation Circuit (Danner et al. 2017 / init_rg_rat)
# ---------------------------------------------------------------------------


def _unit_relu(x: float) -> float:
    return 0.0 if x <= 0.0 else (1.0 if x >= 1.0 else float(x))


def _activation_locomotion4_flx(v: float) -> float:
    """Flexor oscillator activation from `projects/rodent/models/ncap.py`."""
    return _unit_relu((v - 0.1) / 0.9)


def _activation_locomotion4_ext(v: float) -> float:
    """Extensor leaky-integrator activation from `projects/rodent/models/ncap.py`."""
    return _unit_relu(v)


@dataclass
class BasicUnit:
    """First-order leaky integrator neuron matching `libs/circuits/units.py:Basic`."""

    name: str
    voltage_time: float
    bias: float = 0.0
    v_min: float = 0.0
    v_max: float = 1.0
    v: float = 0.0
    value: float = 0.0
    activity_scale: float = 1.0

    def reset(self, v: float = 0.0) -> None:
        self.v = float(np.clip(v, self.v_min, self.v_max))
        self.value = _unit_relu(self.v) * self.activity_scale

    def step(self, current: float, dt: float) -> float:
        tau = max(self.voltage_time, 1e-4)
        dv = (dt / tau) * (-self.v + self.bias + float(current))
        self.v = float(np.clip(self.v + dv, self.v_min, self.v_max))
        self.value = _unit_relu(self.v) * self.activity_scale
        return self.value


@dataclass
class BurstingOscillatorUnit:
    """Relaxation-oscillator bursting unit matching `libs/circuits/units.py:Oscillator`.

    Uses exact `init_rg_rat` parameters:
      active_time = 0.16 s (200 ms * 0.8)
      quiet_time  = 0.48 s (600 ms * 0.8)
      adaptation_time = 0.72 s
      voltage_time = 0.01 s
    """

    name: str
    active_time: float = 0.16
    quiet_time: float = 0.48
    adaptation_time: float = 0.72
    voltage_time: float = 0.01
    v: float = -1.0
    a: float = 0.5
    value: float = 0.0
    activity_scale: float = 1.0

    def __post_init__(self) -> None:
        # Quiet/active phase adaptation thresholds matching Oscillator._compute_bounds.
        self.active_bound_zero = 1.0 - math.exp(
            -self.active_time / max(self.adaptation_time, 1e-4)
        )
        self.quiet_bound_zero = math.exp(
            -self.quiet_time / max(self.adaptation_time, 1e-4)
        )
        # Ensure active > quiet threshold for bistable relaxation cycle.
        if self.active_bound_zero <= self.quiet_bound_zero:
            self.active_bound_zero = min(0.85, self.quiet_bound_zero + 0.18)

    def reset(self, a_fraction: float = 0.8) -> None:
        self.v = -1.0
        self.a = float(np.clip(a_fraction * self.quiet_bound_zero, 0.0, 1.0))
        self.value = _activation_locomotion4_flx(self.v) * self.activity_scale

    def step(self, drive: float, syn_current: float, dt: float) -> float:
        """Advance oscillator state by `dt` seconds given brainstem `drive` in [0, 1]."""
        d = float(np.clip(drive, 0.0, 1.0))
        # Higher drive lowers the effective quiet-to-active threshold (shortens stance/quiet phase),
        # increasing burst frequency across walk -> trot -> bound.
        quiet_thresh = self.quiet_bound_zero * (1.0 - 0.62 * d)
        active_thresh = self.active_bound_zero * (1.0 + 0.12 * d)

        if d < 0.03:
            # Below locomotion initiation threshold: stay quiescent in standing posture.
            target_v = -1.0
            self.v += (dt / max(self.voltage_time, 1e-4)) * (target_v - self.v)
            self.v = float(np.clip(self.v, -1.0, 1.0))
            self.a += (dt / max(self.adaptation_time, 1e-4)) * (
                self.quiet_bound_zero - self.a
            )
            self.a = float(np.clip(self.a, 0.0, 1.0))
            self.value = _activation_locomotion4_flx(self.v) * self.activity_scale
            return self.value

        # Hysteretic relaxation switching modulated by synaptic input from interneurons/afferents.
        effective_a = self.a - 0.28 * float(syn_current)
        if self.v > 0.0:
            # Active (swing) phase: adaptation rises toward 1.
            da = (dt / max(self.active_time, 1e-4)) * (1.0 - self.a)
            self.a = float(np.clip(self.a + da, 0.0, 1.0))
            if effective_a >= active_thresh:
                target_v = -1.0
            else:
                target_v = 1.0
        else:
            # Quiet (stance) phase: adaptation decays toward 0 at a rate boosted by drive.
            rate = (1.0 + 2.6 * d) / max(self.quiet_time, 1e-4)
            da = -dt * rate * self.a
            self.a = float(np.clip(self.a + da, 0.0, 1.0))
            if effective_a <= quiet_thresh:
                target_v = 1.0
            else:
                target_v = -1.0

        self.v += (dt / max(self.voltage_time, 1e-4)) * (target_v - self.v)
        self.v = float(np.clip(self.v, -1.0, 1.0))
        self.value = _activation_locomotion4_flx(self.v) * self.activity_scale
        return self.value


class RatRhythmGenerationCircuit:
    """Mammalian quadrupedal spinal Rhythm Generation (RG) circuit (`init_rg_rat`).

    Implements the 4 limb flexor-extensor half-centers plus commissural (`V0d`,
    `V3f`, `V3e`) and long propriospinal (`diagd_v0d`, `diagd_v0v`, `diagd_in2`,
    `diaga_v3a`) interneurons, derived delay units (`flx2_*`, `ext2_*`),
    in-silico cell-type ablations, and sign-preserving multiplicative RG weight
    noise.
    """

    def __init__(
        self,
        *,
        dt: float = 0.02,
        rg_noise_std: float = 0.0,
        seed: int = 7,
        cell_type_scales: Mapping[str, float] | None = None,
    ) -> None:
        self.dt = float(dt)
        self.rg_noise_std = max(0.0, float(rg_noise_std))
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        # Build half-center flexor oscillators & extensor leaky integrators.
        self.flx: Dict[str, BurstingOscillatorUnit] = {
            limb: BurstingOscillatorUnit(name=f"flx_{limb}") for limb in RAT_LIMB_NAMES
        }
        self.ext: Dict[str, BasicUnit] = {
            limb: BasicUnit(name=f"ext_{limb}", voltage_time=0.03, bias=0.85)
            for limb in RAT_LIMB_NAMES
        }
        # Derived delay/phase units per limb (`flx2_*` 2-step delay, `ext2_*` Tv=80ms).
        self._flx_hist: Dict[str, list[float]] = {
            limb: [0.0, 0.0] for limb in RAT_LIMB_NAMES
        }
        self.flx2: Dict[str, float] = {limb: 0.0 for limb in RAT_LIMB_NAMES}
        self.ext2: Dict[str, BasicUnit] = {
            limb: BasicUnit(name=f"ext2_{limb}", voltage_time=0.08, bias=-0.3)
            for limb in RAT_LIMB_NAMES
        }

        # Commissural & long propriospinal interneuron populations (Tv = 10 ms).
        self.interneurons: Dict[str, BasicUnit] = {}
        for limb in RAT_LIMB_NAMES:
            self.interneurons[f"cross_v0d_{limb}"] = BasicUnit(
                f"cross_v0d_{limb}", voltage_time=0.01
            )
            self.interneurons[f"cross_v3f_{limb}"] = BasicUnit(
                f"cross_v3f_{limb}", voltage_time=0.01
            )
            self.interneurons[f"cross_v3e_{limb}"] = BasicUnit(
                f"cross_v3e_{limb}", voltage_time=0.01
            )
        for fore in ("FL", "FR"):
            self.interneurons[f"diagd_v0d_{fore}"] = BasicUnit(
                f"diagd_v0d_{fore}", voltage_time=0.01
            )
            self.interneurons[f"diagd_v0v_{fore}"] = BasicUnit(
                f"diagd_v0v_{fore}", voltage_time=0.01
            )
            self.interneurons[f"diagd_in2_{fore}"] = BasicUnit(
                f"diagd_in2_{fore}", voltage_time=0.01
            )
        for hind in ("HL", "HR"):
            self.interneurons[f"diaga_v3a_{hind}"] = BasicUnit(
                f"diaga_v3a_{hind}", voltage_time=0.01
            )

        # Base synaptic weights matching `init_rg_rat` (sign-constrained).
        self._base_weights: Dict[str, float] = {
            "osc_flxext_i": -2.0,
            "osc_extflx_i": -0.1,
            "cross_v0d_i": -1.35,
            "cross_v3f_e": 0.95,
            "cross_v3e_e": 0.45,
            "diagd_v0d_i": -0.75,
            "diagd_v0v_e": 0.85,
            "diagd_in2_i": -0.35,
            "diaga_v3a_e": 0.65,
            "sided_flxflx_i": -1.2,
            "sided_extflx_e": 0.05,
            "sidea_extflx_e": 0.05,
        }
        self.weights: Dict[str, float] = dict(self._base_weights)
        self.apply_weight_noise(self.rg_noise_std, seed=self.seed)

        self.cell_type_scales: Dict[str, float] = {}
        if cell_type_scales:
            self.set_cell_type_scales(cell_type_scales)

        self.reset()

    def apply_weight_noise(self, sigma: float, *, seed: int | None = None) -> None:
        """Apply sign-preserving multiplicative Gaussian noise `(1 + sigma * eps) * w`."""
        self.rg_noise_std = max(0.0, float(sigma))
        if seed is not None:
            self.seed = int(seed)
            self.rng = np.random.default_rng(self.seed)
        self.weights = {}
        for key, base in self._base_weights.items():
            if self.rg_noise_std <= 0.0:
                self.weights[key] = base
            else:
                eps = float(self.rng.normal(0.0, 1.0))
                scale = max(0.05, 1.0 + self.rg_noise_std * eps)
                self.weights[key] = base * scale

    def set_cell_type_scales(self, scales: Mapping[str, float]) -> None:
        """Scale specific interneuron or population activities in `[0, 1]`."""
        self.cell_type_scales = {}
        for unit in self.interneurons.values():
            unit.activity_scale = 1.0
        for key, val in scales.items():
            k_val = float(np.clip(val, 0.0, 1.0))
            norm_key = str(key).strip().lower()
            if norm_key in RAT_ABLATION_POPULATIONS:
                for unit_name in RAT_ABLATION_POPULATIONS[norm_key]:
                    if unit_name in self.interneurons:
                        self.interneurons[unit_name].activity_scale = k_val
                        self.cell_type_scales[unit_name] = k_val
            elif key in self.interneurons:
                self.interneurons[key].activity_scale = k_val
                self.cell_type_scales[key] = k_val

    def set_ablation(self, population: str = "none", scale: float = 0.0) -> None:
        """Convenience helper for canonical ablation experiments (`v0v_v0d`, `v3f_v3a`, etc.)."""
        pop = str(population or "none").strip().lower()
        if pop == "none":
            self.set_cell_type_scales({})
        else:
            self.set_cell_type_scales({pop: float(np.clip(scale, 0.0, 1.0))})

    def reset(self) -> None:
        """Reset staggered flexor adaptation phases (FL 0.7, FR 0.9, HL 0.8, HR 1.0)."""
        stagger = {"FL": 0.7, "FR": 0.9, "HL": 0.8, "HR": 1.0}
        for limb, frac in stagger.items():
            self.flx[limb].reset(a_fraction=frac)
            self.ext[limb].reset(v=0.85)
            self.ext2[limb].reset(v=0.5)
            self._flx_hist[limb] = [0.0, 0.0]
            self.flx2[limb] = 0.0
        for unit in self.interneurons.values():
            unit.reset(v=0.0)

    def step(
        self,
        command: float,
        *,
        afferent_flx: Mapping[str, float] | None = None,
        afferent_ext: Mapping[str, float] | None = None,
        dt: float | None = None,
    ) -> Dict[str, Any]:
        """Advance the spinal RG circuit one control step.

        Args:
            command: Brainstem drive `c_t` in `[0, 1]` (maps speed `0 -> 1 m/s`).
            afferent_flx: Optional per-limb flexor afferent feedback dict.
            afferent_ext: Optional per-limb extensor afferent feedback dict.
            dt: Optional step duration override (seconds).
        """
        step_dt = self.dt if dt is None else float(dt)
        c = float(np.clip(command, 0.0, 1.0))
        aff_f = afferent_flx or {}
        aff_e = afferent_ext or {}
        w = self.weights

        # Speed-dependent brainstem modulation of commissural pathways (Danner et al. 2017):
        # - V0d (alternation) dominates at low/medium speeds and declines at high speed.
        # - V0v (diagonal excitation for trot) peaks at moderate speeds (0.35 - 0.65).
        # - V3f/V3e/V3a (left-right synchrony for bound) ramp up at high speeds (> 0.68).
        v0d_gate = float(np.clip(1.15 - 1.15 * c, 0.05, 1.0))
        v0v_gate = float(np.clip(1.0 - abs(c - 0.50) * 2.1, 0.05, 1.0))
        v3_gate = float(np.clip((c - 0.55) / 0.35, 0.0, 1.0))
        # If V0 pathways are ablated, residual baseline V3 drive suffices to synchronize left/right
        # (reproducing Bellardita & Kiehn 2015 and Rat NCAP Fig. 5 V0v+V0d ablation -> bound at 0.4 m/s).
        v0_mean_scale = float(
            np.mean(
                [
                    self.interneurons[u].activity_scale
                    for u in RAT_ABLATION_POPULATIONS["v0v_v0d"]
                ]
            )
        )
        v3_effective_gate = max(v3_gate, 0.45 * (1.0 - v0_mean_scale) * float(c > 0.15))

        # Snapshot previous values for synchronous update.
        prev_flx = {limb: self.flx[limb].value for limb in RAT_LIMB_NAMES}
        prev_ext = {limb: self.ext[limb].value for limb in RAT_LIMB_NAMES}

        # 1. Update interneuron populations.
        for limb in RAT_LIMB_NAMES:
            self.interneurons[f"cross_v0d_{limb}"].step(
                prev_flx[limb] * v0d_gate, step_dt
            )
            self.interneurons[f"cross_v3f_{limb}"].step(
                prev_flx[limb] * v3_effective_gate, step_dt
            )
            self.interneurons[f"cross_v3e_{limb}"].step(
                prev_ext[limb] * v3_effective_gate, step_dt
            )
        for fore in ("FL", "FR"):
            self.interneurons[f"diagd_v0d_{fore}"].step(
                prev_flx[fore] * v0d_gate, step_dt
            )
            self.interneurons[f"diagd_v0v_{fore}"].step(
                prev_flx[fore] * v0v_gate, step_dt
            )
            self.interneurons[f"diagd_in2_{fore}"].step(prev_ext[fore] * 0.6, step_dt)
        for hind in ("HL", "HR"):
            self.interneurons[f"diaga_v3a_{hind}"].step(
                prev_flx[hind] * v3_effective_gate, step_dt
            )

        # 2. Compute synaptic currents onto each limb's flexor & extensor centers.
        contra = {"FL": "FR", "FR": "FL", "HL": "HR", "HR": "HL"}
        homolat = {"FL": "HL", "FR": "HR", "HL": "FL", "HR": "FR"}
        diag = {"FL": "HR", "FR": "HL", "HL": "FR", "HR": "FL"}

        flx_out: Dict[str, float] = {}
        ext_out: Dict[str, float] = {}
        for limb in RAT_LIMB_NAMES:
            c_limb = contra[limb]
            h_limb = homolat[limb]
            d_limb = diag[limb]
            syn_flx = 0.0
            # Reciprocal inhibition from own extensor.
            syn_flx += w["osc_extflx_i"] * prev_ext[limb]
            # Contralateral V0d inhibition (alternation) & V3f excitation (synchrony).
            syn_flx += w["cross_v0d_i"] * self.interneurons[f"cross_v0d_{c_limb}"].value
            syn_flx += w["cross_v3f_e"] * self.interneurons[f"cross_v3f_{c_limb}"].value
            # Homolateral coordination.
            if limb in ("HL", "HR"):
                # Descending homolateral from forelimb to hindlimb.
                syn_flx += (
                    w["sided_flxflx_i"]
                    * prev_flx[h_limb]
                    * (1.0 - 0.55 * v3_effective_gate)
                )
                syn_flx += w["sided_extflx_e"] * prev_ext[h_limb]
                # Descending diagonal from contralateral forelimb (`d_limb`).
                syn_flx += (
                    w["diagd_v0v_e"] * self.interneurons[f"diagd_v0v_{d_limb}"].value
                )
                syn_flx += (
                    w["diagd_v0d_i"]
                    * self.interneurons[f"diagd_v0d_{d_limb}"].value
                    * (1.0 - v0v_gate)
                )
                syn_flx += (
                    w["diagd_in2_i"] * self.interneurons[f"diagd_in2_{d_limb}"].value
                )
            else:
                # Ascending homolateral from hindlimb to forelimb.
                syn_flx += w["sidea_extflx_e"] * prev_ext[h_limb]
                syn_flx += w["sided_flxflx_i"] * 0.45 * prev_flx[h_limb]
                # Ascending diagonal V3a from contralateral hindlimb (`d_limb`).
                syn_flx += (
                    w["diaga_v3a_e"] * self.interneurons[f"diaga_v3a_{d_limb}"].value
                )

            syn_flx += float(aff_f.get(limb, 0.0))
            flx_val = self.flx[limb].step(c, syn_flx, step_dt)

            # Extensor center receives strong reciprocal inhibition from flexor + V3e + afferent.
            syn_ext = (
                w["osc_flxext_i"] * flx_val
                + w["cross_v3e_e"] * self.interneurons[f"cross_v3e_{c_limb}"].value
                + float(aff_e.get(limb, 0.0))
            )
            ext_val = self.ext[limb].step(syn_ext, step_dt)

            # Derived delay units: 2-step delayed flexor (`flx2`) & smoothed extensor (`ext2`).
            hist = self._flx_hist[limb]
            self.flx2[limb] = hist[0]
            hist.append(flx_val)
            if len(hist) > 2:
                hist.pop(0)
            self.ext2[limb].step(ext_val - flx_val, step_dt)

            flx_out[limb] = flx_val
            ext_out[limb] = ext_val

        return {
            "command": c,
            "flx": flx_out,
            "flx2": dict(self.flx2),
            "ext": ext_out,
            "ext2": {limb: self.ext2[limb].value for limb in RAT_LIMB_NAMES},
            "interneurons": {
                name: unit.value for name, unit in self.interneurons.items()
            },
        }

    def get_output(self) -> Dict[str, Tuple[float, float, float, float]]:
        """Returns `{limb: (flx, flx2, ext, ext2)}` matching `RhythmGenerationCircuit.get_output`."""
        return {
            limb: (
                self.flx[limb].value,
                self.flx2[limb],
                self.ext[limb].value,
                self.ext2[limb].value,
            )
            for limb in RAT_LIMB_NAMES
        }


# ---------------------------------------------------------------------------
# Gait Classifier & Locomotion Metrics (Appendix E)
# ---------------------------------------------------------------------------


def classify_quadruped_gait(
    *,
    speed_mps: float,
    phase_offsets_rel_fr: Mapping[str, float],
    stance_fraction: float = 0.6,
) -> str:
    """Classify quadrupedal gait (`stand`, `walk`, `trot`, `bound`) from speed & interlimb phases.

    Phase offsets are normalized in `[0, 1)` relative to FR footfall:
      - `stand`: speed < 0.05 m/s
      - `bound`: left-right pairs nearly in phase (|phase_FL| < 0.22 or > 0.78)
        and fore-hind out of phase (~0.3-0.7)
      - `trot`: diagonal pair (HL with FR) nearly in phase (diag_sync < 0.18)
        and contralateral FL near anti-phase (0.26..0.74), clearly separated from
        4-beat lateral-sequence walk (diag_sync ~ 0.25)
      - `walk`: lateral sequence with high duty factor or staggered 4-beat phases
    """
    if speed_mps < 0.05:
        return "stand"

    def circ_dist(a: float, b: float) -> float:
        d = abs((a - b) % 1.0)
        return min(d, 1.0 - d)

    p_fl = float(phase_offsets_rel_fr.get("FL", 0.5)) % 1.0
    p_hl = float(phase_offsets_rel_fr.get("HL", 0.0)) % 1.0
    p_hr = float(phase_offsets_rel_fr.get("HR", 0.5)) % 1.0

    # Left-right synchrony (FL ~ FR and HL ~ HR) -> bound.
    lr_fore_sync = circ_dist(p_fl, 0.0)
    lr_hind_sync = circ_dist(p_hl, p_hr)
    if lr_fore_sync < 0.23 and lr_hind_sync < 0.25:
        return "bound"

    # Diagonal synchrony (HL ~ FR and HR ~ FL) with left-right alternation -> trot.
    # Midpoint threshold (0.18) cleanly separates diagonal trot (~0.00-0.12) from
    # 4-beat lateral-sequence walk (~0.23-0.25) even when tracking noise depresses duty factor.
    diag_sync = 0.5 * (circ_dist(p_hl, 0.0) + circ_dist(p_hr, p_fl))
    if lr_fore_sync >= 0.26 and diag_sync < 0.18 and stance_fraction < 0.72:
        return "trot"

    return "walk"


def compute_bumpy_terrain_height(
    x: float,
    y: float,
    *,
    bump_height_m: float = 0.02,
    wavelength_m: float = 0.05,
) -> float:
    """Smooth 5 cm bumpy heightfield matching Rat NCAP Appendix F.3 (`0..4 cm`)."""
    amp = max(0.0, float(bump_height_m))
    if amp <= 1e-6:
        return 0.0
    k = (2.0 * math.pi) / max(float(wavelength_m), 1e-3)
    h = 0.5 * (
        math.sin(k * x) * math.cos(k * y)
        + 0.45 * math.sin(0.67 * k * x + 0.53 * k * y + 0.9)
    )
    return float(amp * 0.5 * (h + 1.0))


# ---------------------------------------------------------------------------
# Full Rat NCAP Controller + Kinematics & Closed-Loop Rollout
# ---------------------------------------------------------------------------


@dataclass
class RatNCAPConfig:
    """Configuration for `RatNCAPController`."""

    interface: str = "flexor_extensor"
    behavior: str = "trot"
    target_speed: float | None = None
    coat: str = "black_hooded"
    ablation: str = "none"
    ablation_scale: float = 0.0
    rg_noise_std: float = 0.0
    terrain_bump_cm: float = 0.0
    dt: float = 0.02
    seed: int = 7
    brainstem_weight: float = 1.0
    brainstem_bias: float = 0.0
    lock_lumbar: bool = False


class RatNCAPController:
    """Closed-loop Rat NCAP controller combining BC, RG, AF, PF, and neuromechanical interface.

    Produces:
      - 31-DOF locomotion joint states `(q, dq, tau)` + 38-DOF full Rat v2 joint map
      - Antagonistic flexor/extensor activations (`flx`, `ext`) for `flexor_extensor`
      - 16-keypoint 3D Rat v2 world poses & rigid body poses for 3D viewers
      - Spinal RG neural traces, foot contacts, interlimb phase & gait classification,
        and positive/negative mechanical power.
    """

    def __init__(self, config: RatNCAPConfig | None = None) -> None:
        self.config = config or RatNCAPConfig()
        self.interface = NeuromechanicalInterface(self.config.interface)
        self.rg = RatRhythmGenerationCircuit(
            dt=self.config.dt,
            rg_noise_std=self.config.rg_noise_std,
            seed=self.config.seed,
        )
        self.behavior = str(self.config.behavior or "trot").strip().lower()
        self.coat = normalize_rat_coat(self.config.coat)
        self.target_speed = (
            float(self.config.target_speed)
            if self.config.target_speed is not None
            else default_speed_for_behavior(self.behavior)
        )
        self.terrain_bump_m = max(0.0, float(self.config.terrain_bump_cm)) * 0.01
        if self.behavior == "bumpy_terrain" and self.terrain_bump_m <= 0.0:
            self.terrain_bump_m = 0.02

        # Apply canonical behavior ablation presets unless explicitly overridden.
        ablation = str(self.config.ablation or "none").strip().lower()
        ablation_scale = float(self.config.ablation_scale)
        if ablation == "none":
            if self.behavior == "v0_ablation_bound":
                ablation = "v0v_v0d"
                ablation_scale = 0.0
            elif self.behavior == "v3_ablation_trot":
                ablation = "v3f_v3a"
                ablation_scale = 0.0
        self.ablation = ablation
        self.ablation_scale = ablation_scale
        self.rg.set_ablation(self.ablation, self.ablation_scale)

        # Dynamic state in world frame (meters, radians).
        self.step_count: int = 0
        self.time_sec: float = 0.0
        self.root_pos = np.asarray(RAT_V2_ROOT_POS_INIT, dtype=np.float64).copy()
        self.yaw: float = 0.0
        self.pitch: float = 0.0
        self.roll: float = 0.0
        self.forward_speed: float = 0.0
        self.q = self.interface.poseq.copy()
        self.dq = np.zeros(self.interface.dof, dtype=np.float64)
        self.tau = np.zeros(self.interface.dof, dtype=np.float64)

        # Limb phase trackers for clean gait/interlimb phase telemetry.
        self._limb_phase: Dict[str, float] = {
            "FL": 0.50,
            "FR": 0.0,
            "HL": 0.0,
            "HR": 0.50,
        }
        self._foot_contact: Dict[str, float] = {limb: 1.0 for limb in RAT_LIMB_NAMES}
        self._prev_contact: Dict[str, bool] = {limb: True for limb in RAT_LIMB_NAMES}
        self._touchdown_times: Dict[str, list[float]] = {
            limb: [] for limb in RAT_LIMB_NAMES
        }
        self._stance_steps: Dict[str, int] = {limb: 0 for limb in RAT_LIMB_NAMES}
        self._total_steps_for_duty: int = 0

        # Segment slices into the 31-DOF locomotion vector.
        self._seg_slices: Dict[str, slice] = {}
        offset = 0
        for seg in self.interface.segments:
            n = len(RAT_V2_SEGMENTS[seg])
            self._seg_slices[seg] = slice(offset, offset + n)
            offset += n

    def reset(self) -> None:
        self.step_count = 0
        self.time_sec = 0.0
        self.root_pos = np.asarray(RAT_V2_ROOT_POS_INIT, dtype=np.float64).copy()
        self.yaw = 0.0
        self.pitch = 0.0
        self.roll = 0.0
        self.forward_speed = 0.0
        self.q = self.interface.poseq.copy()
        self.dq = np.zeros(self.interface.dof, dtype=np.float64)
        self.tau = np.zeros(self.interface.dof, dtype=np.float64)
        self.rg.reset()
        self._limb_phase = {"FL": 0.50, "FR": 0.0, "HL": 0.0, "HR": 0.50}
        self._foot_contact = {limb: 1.0 for limb in RAT_LIMB_NAMES}
        self._prev_contact = {limb: True for limb in RAT_LIMB_NAMES}
        self._touchdown_times = {limb: [] for limb in RAT_LIMB_NAMES}
        self._stance_steps = {limb: 0 for limb in RAT_LIMB_NAMES}
        self._total_steps_for_duty = 0

    def brainstem_command(self, target_speed: float) -> float:
        """Linear brainstem command module: `c_t = clip(w * v_target + b, 0, 1)`."""
        return float(
            np.clip(
                self.config.brainstem_weight * float(target_speed)
                + self.config.brainstem_bias,
                0.0,
                1.0,
            )
        )

    def _scheduled_speed(self, t: float) -> float:
        if self.behavior == "gait_transition":
            # Smooth 4-stage speed ramp: stand (0.0) -> walk (0.22) -> trot (0.52) -> bound (0.90)
            cycle = t % 8.0
            if cycle < 1.2:
                return 0.0
            if cycle < 3.2:
                return 0.22
            if cycle < 5.6:
                return 0.52
            return 0.90
        if self.behavior == "open_field_explore":
            return float(
                np.clip(self.target_speed + 0.14 * math.sin(0.8 * t), 0.12, 0.72)
            )
        return float(np.clip(self.target_speed, 0.0, 1.0))

    def _effective_phase_offsets(self, command: float) -> Dict[str, float]:
        """Target interlimb phase offsets emerging from the RG commissural balance."""
        v0_scale = float(
            np.mean(
                [
                    self.rg.interneurons[u].activity_scale
                    for u in RAT_ABLATION_POPULATIONS["v0v_v0d"]
                ]
            )
        )
        v3_scale = float(
            np.mean(
                [
                    self.rg.interneurons[u].activity_scale
                    for u in RAT_ABLATION_POPULATIONS["v3f_v3a"]
                ]
            )
        )
        # Bound dominance from high speed OR V0v+V0d ablation (Bellardita & Kiehn 2015).
        speed_bound = float(np.clip((command - 0.66) / 0.20, 0.0, 1.0)) * (
            0.35 + 0.65 * v3_scale
        )
        ablation_bound = float(np.clip((0.35 - v0_scale) / 0.25, 0.0, 1.0)) * float(
            command >= 0.18
        )
        bound_blend = max(speed_bound, ablation_bound)

        # Walk (lateral sequence: FR=0.0, HR=0.25, FL=0.50, HL=0.75) ->
        # Trot (diagonal: FR=0.0, HL=0.0, FL=0.50, HR=0.50) ->
        # Bound (homologous sync: FR=0.0, FL=0.0, HL=0.50, HR=0.50)
        trot_blend = float(np.clip((command - 0.26) / 0.16, 0.0, 1.0))

        # Interpolate walk -> trot first, then -> bound.
        p_fl = 0.50 * (1.0 - bound_blend)
        p_hl = (0.75 + 0.25 * trot_blend) % 1.0
        # At bound_blend=1, HL/HR shift to 0.50 (half-cycle behind forelimbs).
        if bound_blend > 0.0:
            # Shortest circular shift from p_hl toward 0.50
            delta_hl = ((0.50 - p_hl + 0.5) % 1.0) - 0.5
            p_hl = (p_hl + bound_blend * delta_hl) % 1.0
        p_hr = (0.25 + 0.25 * trot_blend) % 1.0
        if bound_blend > 0.0:
            delta_hr = ((0.50 - p_hr + 0.5) % 1.0) - 0.5
            p_hr = (p_hr + bound_blend * delta_hr) % 1.0
        return {"FR": 0.0, "FL": p_fl % 1.0, "HL": p_hl % 1.0, "HR": p_hr % 1.0}

    def step(
        self,
        *,
        target_speed: float | None = None,
        dt: float | None = None,
        turn_rate: float = 0.0,
    ) -> Dict[str, Any]:
        """Advance Rat NCAP controller + biomechanical body one step and return full telemetry."""
        step_dt = self.config.dt if dt is None else float(dt)
        if target_speed is not None:
            v_target = float(np.clip(target_speed, 0.0, 1.0))
        else:
            v_target = self._scheduled_speed(self.time_sec)

        command = self.brainstem_command(v_target)

        # Afferent feedback from paw contacts & limb load -> RG resetting / modulation.
        aff_flx = {
            limb: 0.12 * (1.0 - self._foot_contact[limb]) for limb in RAT_LIMB_NAMES
        }
        aff_ext = {limb: 0.18 * self._foot_contact[limb] for limb in RAT_LIMB_NAMES}
        rg_state = self.rg.step(
            command,
            afferent_flx=aff_flx,
            afferent_ext=aff_ext,
            dt=step_dt,
        )

        # Coupled step frequency (Hz) from speed & RG rhythmicity (~1.8 Hz walk -> ~5.4 Hz bound).
        step_freq_hz = 0.0 if command < 0.03 else (1.5 + 3.9 * command)
        target_offsets = self._effective_phase_offsets(command)

        # Advance FR reference phase and gently entrain FL/HL/HR toward RG-coupled offsets.
        if step_freq_hz > 0.0:
            self._limb_phase["FR"] = (
                self._limb_phase["FR"] + step_freq_hz * step_dt
            ) % 1.0
            entrain = min(1.0, 7.5 * step_dt)
            for limb in ("FL", "HL", "HR"):
                desired = (self._limb_phase["FR"] + target_offsets[limb]) % 1.0
                diff = ((desired - self._limb_phase[limb] + 0.5) % 1.0) - 0.5
                self._limb_phase[limb] = (
                    self._limb_phase[limb] + step_freq_hz * step_dt + entrain * diff
                ) % 1.0

        # Duty factor (stance fraction) decreases smoothly with speed: 0.78 walk -> 0.44 bound.
        duty_factor = (
            1.0 if command < 0.03 else float(np.clip(0.80 - 0.36 * command, 0.42, 0.85))
        )
        swing_frac = 1.0 - duty_factor

        # Pattern Formation (PF): 5 phase synergies per limb + trunk lumbar synergies.
        flx_act = np.full(self.interface.dof, 0.5, dtype=np.float64)
        ext_act = np.full(self.interface.dof, 0.5, dtype=np.float64)
        q_target = self.interface.poseq.copy()
        kp_scale = np.full(self.interface.dof, 0.72, dtype=np.float64)
        trq_cmd = np.zeros(self.interface.dof, dtype=np.float64)

        rg_out = self.rg.get_output()
        limb_swing: Dict[str, float] = {}
        limb_stance: Dict[str, float] = {}
        for limb in RAT_LIMB_NAMES:
            phase = self._limb_phase[limb]
            in_swing = (command >= 0.03) and (phase < swing_frac)
            if in_swing:
                s_u = phase / max(swing_frac, 1e-4)  # 0..1 within swing
                liftoff = math.sin(math.pi * min(1.0, s_u * 2.0)) if s_u < 0.5 else 0.0
                advance = (
                    math.sin(math.pi * max(0.0, (s_u - 0.35) / 0.65))
                    if s_u > 0.35
                    else 0.0
                )
                touchdown = 0.0
                support = 0.0
                propulsion = 0.0
                contact = 0.0
            else:
                st_u = (
                    0.0
                    if command < 0.03
                    else (phase - swing_frac) / max(duty_factor, 1e-4)
                )
                liftoff = 0.0
                advance = 0.0
                touchdown = (
                    math.exp(-(((st_u - 0.12) / 0.12) ** 2)) if command >= 0.03 else 0.0
                )
                support = 1.0 if command < 0.03 else math.sin(math.pi * st_u)
                propulsion = (
                    math.exp(-(((st_u - 0.82) / 0.18) ** 2)) if command >= 0.03 else 0.0
                )
                contact = 1.0

            # Blend with live RG unit activity for genuine neural-circuit coupling.
            rg_f, rg_f2, rg_e, _rg_e2 = rg_out[limb]
            swing_drive = float(
                np.clip(
                    0.75 * (liftoff + advance) + 0.25 * (rg_f + 0.5 * rg_f2), 0.0, 1.0
                )
            )
            stance_drive = float(
                np.clip(
                    0.75 * (support + 0.6 * propulsion + 0.4 * touchdown) + 0.25 * rg_e,
                    0.0,
                    1.0,
                )
            )
            limb_swing[limb] = swing_drive
            limb_stance[limb] = stance_drive
            self._foot_contact[limb] = contact

            is_contact = contact > 0.5
            if is_contact and not self._prev_contact[limb]:
                self._touchdown_times[limb].append(self.time_sec)
                if len(self._touchdown_times[limb]) > 16:
                    self._touchdown_times[limb].pop(0)
            self._prev_contact[limb] = is_contact
            if is_contact:
                self._stance_steps[limb] += 1

            # Map 5 phase synergies onto segment DOFs.
            seg_name = f"limb{limb}"
            sl = self._seg_slices[seg_name]
            amp = 0.25 + 0.75 * command
            if limb in ("FL", "FR"):
                # [scapula_sup, scapula_abd, scapula_ext, shoulder_sup, shoulder_ext, elbow, wrist, finger]
                delta_q = np.array(
                    [
                        0.04 * (advance - propulsion) * amp,
                        0.05 * (liftoff - support) * amp,
                        0.02 * (advance - propulsion) * amp,
                        0.16 * (liftoff - touchdown) * amp,
                        0.48 * (advance - propulsion) * amp,
                        0.52 * (liftoff - 0.7 * support) * amp,
                        0.38 * (touchdown - liftoff) * amp,
                        0.30 * (propulsion - liftoff) * amp,
                    ],
                    dtype=np.float64,
                )
            else:
                # [hip_sup, hip_abd, hip_ext, knee, ankle, toe]
                delta_q = np.array(
                    [
                        0.12 * (liftoff - support) * amp,
                        0.14 * (liftoff - touchdown) * amp,
                        0.62 * (advance - propulsion) * amp,
                        0.58 * (liftoff - 0.65 * propulsion) * amp,
                        0.44 * (propulsion - liftoff) * amp,
                        0.35 * (propulsion - liftoff) * amp,
                    ],
                    dtype=np.float64,
                )
            q_seg_target = np.clip(
                self.interface.poseq[sl] + delta_q,
                self.interface.low[sl],
                self.interface.high[sl],
            )
            q_target[sl] = q_seg_target
            # Antagonistic co-contraction rises during stance support/touchdown and lowers in swing.
            base_co = 0.32 + 0.42 * stance_drive
            norm_shift = np.clip(
                delta_q
                / np.maximum(self.interface.high[sl] - self.interface.low[sl], 1e-4),
                -0.6,
                0.6,
            )
            ext_act[sl] = np.clip(base_co + 0.75 * norm_shift, 0.0, 1.0)
            flx_act[sl] = np.clip(base_co - 0.75 * norm_shift, 0.0, 1.0)
            kp_scale[sl] = np.clip(0.35 + 0.60 * stance_drive, 0.15, 1.0)
            trq_cmd[sl] = self.interface.keq[sl] * (q_seg_target - self.q[sl])

        # Trunk lumbar synergies (sagittal extension during bound; lateral bend during walk/trot).
        lumbar_e_sl = self._seg_slices["lumbarE"]
        lumbar_bt_sl = self._seg_slices["lumbarBT"]
        if not self.config.lock_lumbar and command >= 0.03:
            fore_sync = 0.5 * (limb_swing["FL"] + limb_swing["FR"]) - 0.5 * (
                limb_swing["HL"] + limb_swing["HR"]
            )
            lat_diff = (limb_swing["FL"] - limb_swing["FR"]) - (
                limb_swing["HL"] - limb_swing["HR"]
            )
            q_target[lumbar_e_sl] = np.clip(
                self.interface.poseq[lumbar_e_sl] + 0.42 * command * fore_sync,
                self.interface.low[lumbar_e_sl],
                self.interface.high[lumbar_e_sl],
            )
            q_target[lumbar_bt_sl] = np.clip(
                self.interface.poseq[lumbar_bt_sl]
                + np.array([0.28 * command * lat_diff, 0.18 * command * lat_diff]),
                self.interface.low[lumbar_bt_sl],
                self.interface.high[lumbar_bt_sl],
            )
            ext_act[lumbar_e_sl] = np.clip(0.5 + 0.4 * fore_sync * command, 0.0, 1.0)
            flx_act[lumbar_e_sl] = np.clip(0.5 - 0.4 * fore_sync * command, 0.0, 1.0)
            ext_act[lumbar_bt_sl] = np.clip(0.5 + 0.35 * lat_diff * command, 0.0, 1.0)
            flx_act[lumbar_bt_sl] = np.clip(0.5 - 0.35 * lat_diff * command, 0.0, 1.0)

        # Compute torques via the active neuromechanical interface & integrate simplified joint dynamics.
        tau = self.interface.compute_torque(
            q=self.q,
            dq=self.dq,
            flx=flx_act,
            ext=ext_act,
            q_target=q_target,
            kp_scale=kp_scale,
            trq_cmd=trq_cmd,
        )
        # Lightweight implicit-like compliant joint integration toward torque/equilibrium.
        inertia = np.maximum(0.008 * self.interface.keq, 0.0012)
        damping = np.maximum(0.18 * self.interface.keq, 0.025)
        ddq = (tau - damping * self.dq) / inertia
        # Clamp numerical acceleration for smooth stability across all 4 interfaces.
        ddq = np.clip(ddq, -140.0, 140.0)
        self.dq = np.clip(self.dq + ddq * step_dt, -20.0, 20.0)
        self.q = np.clip(
            self.q + self.dq * step_dt, self.interface.low, self.interface.high
        )
        self.tau = tau

        # Integrate torso root motion on flat or bumpy terrain.
        if self.behavior == "open_field_explore" and turn_rate == 0.0:
            turn_rate = 0.55 * math.sin(0.55 * self.time_sec)
        self.yaw += float(turn_rate) * step_dt
        # Forward speed tracks target speed with minor modulation from noise/terrain.
        terrain_drag = 1.0 - min(0.18, 2.5 * self.terrain_bump_m)
        noise_drag = 1.0 - min(0.15, 0.22 * self.rg.rg_noise_std)
        desired_speed = v_target * terrain_drag * noise_drag
        self.forward_speed += (desired_speed - self.forward_speed) * min(
            1.0, 6.0 * step_dt
        )
        self.root_pos[0] += self.forward_speed * math.cos(self.yaw) * step_dt
        self.root_pos[1] += self.forward_speed * math.sin(self.yaw) * step_dt

        ground_z = compute_bumpy_terrain_height(
            float(self.root_pos[0]),
            float(self.root_pos[1]),
            bump_height_m=self.terrain_bump_m,
        )
        # Vertical bounce & body pitch (larger in bounding than walking).
        fore_minus_hind = 0.5 * (limb_swing["FL"] + limb_swing["FR"]) - 0.5 * (
            limb_swing["HL"] + limb_swing["HR"]
        )
        bounce = (0.003 + 0.009 * command) * math.sin(
            2.0 * math.pi * self._limb_phase["FR"]
        )
        self.root_pos[2] = RAT_V2_ROOT_POS_INIT[2] + ground_z + bounce
        self.pitch = float(-0.06 + 0.14 * command * fore_minus_hind)
        self.roll = float(
            0.06
            * command
            * (
                (limb_swing["FL"] + limb_swing["HL"])
                - (limb_swing["FR"] + limb_swing["HR"])
            )
        )

        self._total_steps_for_duty += 1
        self.step_count += 1
        self.time_sec += step_dt

        # Classify current gait from interlimb phase offsets relative to FR.
        rel_phases = {
            limb: (self._limb_phase[limb] - self._limb_phase["FR"]) % 1.0
            for limb in RAT_LIMB_NAMES
        }
        gait = classify_quadruped_gait(
            speed_mps=self.forward_speed,
            phase_offsets_rel_fr=rel_phases,
            stance_fraction=duty_factor,
        )
        power = compute_mechanical_power(self.tau, self.dq)
        step_length_m = (
            (self.forward_speed / step_freq_hz) if step_freq_hz > 0.1 else 0.0
        )

        # Compute 3D keypoint world positions (meters, Z-up MuJoCo frame) & body poses.
        keypoints_3d = self.compute_keypoints_3d()
        body_poses = self.compute_body_poses(keypoints_3d)

        return {
            "step": self.step_count,
            "time_sec": round(self.time_sec, 6),
            "behavior": self.behavior,
            "interface": self.interface.interface_type,
            "coat": self.coat,
            "ablation": self.ablation,
            "ablation_scale": self.ablation_scale,
            "rg_noise_std": self.rg.rg_noise_std,
            "target_speed_mps": round(v_target, 4),
            "forward_speed_mps": round(self.forward_speed, 4),
            "command": round(command, 4),
            "gait": gait,
            "step_frequency_hz": round(step_freq_hz, 4),
            "step_length_m": round(step_length_m, 4),
            "duty_factor": round(duty_factor, 4),
            "phase_offsets_rel_fr": {k: round(v, 4) for k, v in rel_phases.items()},
            "foot_contacts": {k: round(v, 4) for k, v in self._foot_contact.items()},
            "root_pos": [round(float(x), 6) for x in self.root_pos],
            "root_euler": [
                round(self.roll, 5),
                round(self.pitch, 5),
                round(self.yaw, 5),
            ],
            "qpos": [round(float(x), 5) for x in self.q],
            "qvel": [round(float(x), 5) for x in self.dq],
            "torque": [round(float(x), 5) for x in self.tau],
            "flx_activation": [round(float(x), 4) for x in flx_act],
            "ext_activation": [round(float(x), 4) for x in ext_act],
            "stiffness": [
                round(float(x), 4)
                for x in self.interface.effective_stiffness(
                    flx=flx_act, ext=ext_act, kp_scale=kp_scale
                )
            ],
            "mechanical_power": {k: round(v, 5) for k, v in power.items()},
            "rg": rg_state,
            "keypoints_3d": keypoints_3d,
            "body_poses": body_poses,
        }

    def compute_keypoints_3d(self) -> Dict[str, list[float]]:
        """Forward kinematics for the 16 canonical Rat v2 keypoints (meters, MuJoCo Z-up)."""
        cx, cy, cz = (
            float(self.root_pos[0]),
            float(self.root_pos[1]),
            float(self.root_pos[2]),
        )
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        cos_p, sin_p = math.cos(self.pitch), math.sin(self.pitch)

        def to_world(lx: float, ly: float, lz: float) -> list[float]:
            # Apply pitch around Y then yaw around Z.
            px = lx * cos_p + lz * sin_p
            py = ly
            pz = -lx * sin_p + lz * cos_p
            wx = cx + px * cos_y - py * sin_y
            wy = cy + px * sin_y + py * cos_y
            wz = cz + pz
            return [round(wx, 5), round(wy, 5), round(wz, 5)]

        lumbar_e = float(self.q[self._seg_slices["lumbarE"]][0])
        lumbar_b = float(self.q[self._seg_slices["lumbarBT"]][0])

        # Rat v2 dimensions (adult Long-Evans rat ~22 cm nose-to-tailbase + ~18 cm tail).
        kps: Dict[str, list[float]] = {
            "torso_center": to_world(0.0, 0.0, 0.0),
            "thorax": to_world(0.045, 0.0, 0.006),
            "neck": to_world(0.082, 0.0, 0.014),
            "head": to_world(0.112, 0.0, 0.018),
            "nose": to_world(0.145, 0.0, 0.014),
            "left_ear": to_world(0.106, 0.016, 0.032),
            "right_ear": to_world(0.106, -0.016, 0.032),
            "lumbar": to_world(-0.042, 0.012 * lumbar_b, 0.008 * lumbar_e),
            "pelvis": to_world(-0.088, 0.022 * lumbar_b, -0.002 + 0.014 * lumbar_e),
            "tail_base": to_world(-0.116, 0.026 * lumbar_b, -0.004),
            "tail_mid": to_world(
                -0.195,
                0.034 * lumbar_b
                + 0.015 * math.sin(2.0 * math.pi * self._limb_phase["FR"]),
                -0.016,
            ),
            "tail_tip": to_world(
                -0.285,
                0.042 * lumbar_b
                + 0.028 * math.sin(2.0 * math.pi * self._limb_phase["FR"] - 0.8),
                -0.024,
            ),
        }

        # Limb end-effectors (paws) driven by joint angles + ground contact clearance.
        for limb, label, base_x, base_y, stride_len, lift_h in (
            ("FL", "left_forepaw", 0.056, 0.024, 0.036, 0.020),
            ("FR", "right_forepaw", 0.056, -0.024, 0.036, 0.020),
            ("HL", "left_hindpaw", -0.072, 0.028, 0.044, 0.024),
            ("HR", "right_hindpaw", -0.072, -0.028, 0.044, 0.024),
        ):
            sl = self._seg_slices[f"limb{limb}"]
            q_seg = self.q[sl]
            # Primary sagittal swing angle (shoulder_extend index 4 or hip_extend index 2).
            swing_idx = 4 if limb in ("FL", "FR") else 2
            dq_sag = float(q_seg[swing_idx] - self.interface.poseq[sl][swing_idx])
            lx = base_x + stride_len * math.sin(dq_sag)
            ly = base_y
            # Lift paw off terrain during swing; plant on ground during stance.
            contact = self._foot_contact[limb]
            p_world = to_world(lx, ly, -RAT_V2_ROOT_POS_INIT[2] + 0.004)
            terrain_z = compute_bumpy_terrain_height(
                p_world[0], p_world[1], bump_height_m=self.terrain_bump_m
            )
            lift = (1.0 - contact) * lift_h * (0.4 + 0.6 * self.forward_speed)
            p_world[2] = round(terrain_z + 0.004 + lift, 5)
            kps[label] = p_world

        return kps

    def compute_body_poses(
        self, keypoints_3d: Mapping[str, Sequence[float]]
    ) -> Dict[str, Dict[str, list[float]]]:
        """Return compact body pose dicts (`position`, `quaternion`) for 3D viewer integration."""
        cy = math.cos(0.5 * self.yaw)
        sy = math.sin(0.5 * self.yaw)
        cp = math.cos(0.5 * self.pitch)
        sp = math.sin(0.5 * self.pitch)
        cr = math.cos(0.5 * self.roll)
        sr = math.sin(0.5 * self.roll)
        qw = cr * cp * cy + sr * sp * sy
        qx = sr * cp * cy - cr * sp * sy
        qy = cr * sp * cy + sr * cp * sy
        qz = cr * cp * sy - sr * sp * cy
        quat = [round(qw, 5), round(qx, 5), round(qy, 5), round(qz, 5)]
        return {
            "torso": {
                "position": list(keypoints_3d["torso_center"]),
                "quaternion": quat,
            },
            "skull": {"position": list(keypoints_3d["head"]), "quaternion": quat},
            "pelvis": {"position": list(keypoints_3d["pelvis"]), "quaternion": quat},
            "hand_L": {
                "position": list(keypoints_3d["left_forepaw"]),
                "quaternion": quat,
            },
            "hand_R": {
                "position": list(keypoints_3d["right_forepaw"]),
                "quaternion": quat,
            },
            "foot_L": {
                "position": list(keypoints_3d["left_hindpaw"]),
                "quaternion": quat,
            },
            "foot_R": {
                "position": list(keypoints_3d["right_hindpaw"]),
                "quaternion": quat,
            },
        }


def run_rat_ncap_rollout(
    *,
    steps: int = 180,
    behavior: str = "trot",
    target_speed: float | None = None,
    interface: str = "flexor_extensor",
    coat: str = "black_hooded",
    ablation: str = "none",
    ablation_scale: float = 0.0,
    rg_noise_std: float = 0.0,
    terrain_bump_cm: float = 0.0,
    dt: float = 0.02,
    seed: int = 7,
) -> Dict[str, Any]:
    """Run a deterministic Rat NCAP rollout and return summary + per-frame states."""
    cfg = RatNCAPConfig(
        interface=interface,
        behavior=behavior,
        target_speed=target_speed,
        coat=coat,
        ablation=ablation,
        ablation_scale=ablation_scale,
        rg_noise_std=rg_noise_std,
        terrain_bump_cm=terrain_bump_cm,
        dt=dt,
        seed=seed,
    )
    controller = RatNCAPController(cfg)
    frames: list[Dict[str, Any]] = []
    gait_counts: Dict[str, int] = {}
    pos_powers: list[float] = []
    neg_powers: list[float] = []
    speeds: list[float] = []

    total_steps = max(1, int(steps))
    for _ in range(total_steps):
        frame = controller.step()
        frames.append(frame)
        g = str(frame["gait"])
        gait_counts[g] = gait_counts.get(g, 0) + 1
        pos_powers.append(float(frame["mechanical_power"]["positive_power_w"]))
        neg_powers.append(float(frame["mechanical_power"]["negative_power_w"]))
        speeds.append(float(frame["forward_speed_mps"]))

    dominant_gait = (
        max(gait_counts.items(), key=lambda kv: kv[1])[0] if gait_counts else "stand"
    )
    return {
        "metadata": {
            **RAT_NCAP_METADATA,
            "behavior": controller.behavior,
            "interface": controller.interface.interface_type,
            "coat": controller.coat,
            "ablation": controller.ablation,
            "ablation_scale": controller.ablation_scale,
            "rg_noise_std": controller.rg.rg_noise_std,
            "terrain_bump_cm": float(terrain_bump_cm),
            "dt": float(dt),
            "seed": int(seed),
            "dof_total": len(RAT_V2_ACTUATOR_NAMES),
            "dof_locomotion": controller.interface.dof,
            "dominant_gait": dominant_gait,
            "gait_counts": gait_counts,
            "mean_speed_mps": round(float(np.mean(speeds)), 4),
            "mean_positive_power_w": round(float(np.mean(pos_powers)), 5),
            "mean_negative_power_w": round(float(np.mean(neg_powers)), 5),
        },
        "frames": frames,
    }
