"""Real-to-Sim 3D Biomechanical Inverse Kinematics, Spring-Muscle Inverse Dynamics,
and Latent Spinal Circuit System Identification for Rat v2 / Rat NCAP.

Pure-NumPy implementation (zero heavy optional dependencies) bridging markerless
2D/3D video pose tracking with Annolid's Rat v2 musculoskeletal model and mammalian
spinal Rhythm Generation (RG) circuit (`annolid.simulation.rat_ncap`).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np

from annolid.simulation.rat_ncap import (
    RAT_LIMB_NAMES,
    RAT_V2_ROOT_POS_INIT,
    RAT_V2_SEGMENTS,
    NeuromechanicalInterface,
    classify_quadruped_gait,
    compute_mechanical_power,
)
from annolid.simulation.types import Pose3DFrame

# Canonical Rat v2 local trunk landmark coordinates (meters, MuJoCo Z-up frame)
# matching `RatNCAPController.compute_keypoints_3d`.
RAT_V2_CANONICAL_TRUNK_LOCAL: Dict[str, Tuple[float, float, float]] = {
    "torso_center": (0.0, 0.0, 0.0),
    "thorax": (0.045, 0.0, 0.006),
    "neck": (0.082, 0.0, 0.014),
    "head": (0.112, 0.0, 0.018),
    "nose": (0.145, 0.0, 0.014),
    "left_ear": (0.106, 0.016, 0.032),
    "right_ear": (0.106, -0.016, 0.032),
    "lumbar": (-0.042, 0.0, 0.0),
    "pelvis": (-0.088, 0.0, -0.002),
}

# Mapping from site aliases (MuJoCo Rat v2, DANNCE/Rat7M, Anipose, LocoMouse, SLEAP)
# back to canonical Rat v2 keypoint names.
SITE_TO_RAT_V2_KEYPOINT: Dict[str, str] = {
    "nose_site": "nose",
    "nose": "nose",
    "snout": "nose",
    "HeadF": "nose",
    "head_front": "nose",
    "head": "head",
    "head_site": "head",
    "HeadB": "head",
    "head_back": "head",
    "left_ear_site": "left_ear",
    "left_ear": "left_ear",
    "HeadL": "left_ear",
    "ear_L": "left_ear",
    "right_ear_site": "right_ear",
    "right_ear": "right_ear",
    "HeadR": "right_ear",
    "ear_R": "right_ear",
    "cervical_site": "neck",
    "neck": "neck",
    "torso": "thorax",
    "thorax": "thorax",
    "SpineF": "thorax",
    "spine_front": "thorax",
    "torso_center": "torso_center",
    "SpineM": "torso_center",
    "spine_mid": "torso_center",
    "body_center": "torso_center",
    "lumbar_site": "lumbar",
    "lumbar": "lumbar",
    "SpineL": "lumbar",
    "spine_low": "lumbar",
    "pelvis": "pelvis",
    "Pelvis": "pelvis",
    "hip_center": "pelvis",
    "tail_base_site": "tail_base",
    "tail_base": "tail_base",
    "TailBase": "tail_base",
    "tailbase": "tail_base",
    "tail_mid_site": "tail_mid",
    "tail_mid": "tail_mid",
    "TailMid": "tail_mid",
    "tailmid": "tail_mid",
    "tail_tip_site": "tail_tip",
    "tail_tip": "tail_tip",
    "TailTip": "tail_tip",
    "tailtip": "tail_tip",
    "finger_sensor_L": "left_forepaw",
    "left_forepaw": "left_forepaw",
    "hand_L": "left_forepaw",
    "HandL": "left_forepaw",
    "ForepawL": "left_forepaw",
    "FL": "left_forepaw",
    "left_front": "left_forepaw",
    "front_left": "left_forepaw",
    "finger_sensor_R": "right_forepaw",
    "right_forepaw": "right_forepaw",
    "hand_R": "right_forepaw",
    "HandR": "right_forepaw",
    "ForepawR": "right_forepaw",
    "FR": "right_forepaw",
    "right_front": "right_forepaw",
    "front_right": "right_forepaw",
    "toe_sensor_L": "left_hindpaw",
    "left_hindpaw": "left_hindpaw",
    "foot_L": "left_hindpaw",
    "FootL": "left_hindpaw",
    "HindpawL": "left_hindpaw",
    "HL": "left_hindpaw",
    "left_hind": "left_hindpaw",
    "hind_left": "left_hindpaw",
    "toe_sensor_R": "right_hindpaw",
    "right_hindpaw": "right_hindpaw",
    "foot_R": "right_hindpaw",
    "FootR": "right_hindpaw",
    "HindpawR": "right_hindpaw",
    "HR": "right_hindpaw",
    "right_hind": "right_hindpaw",
    "hind_right": "right_hindpaw",
}

_SITE_TO_RAT_V2_LOWER: Dict[str, str] = {
    k.lower(): v for k, v in SITE_TO_RAT_V2_KEYPOINT.items()
}

_SCALE_ANCHOR_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("nose", "pelvis"),
    ("head", "pelvis"),
    ("thorax", "pelvis"),
    ("nose", "thorax"),
    ("nose", "torso_center"),
    ("head", "torso_center"),
    ("thorax", "torso_center"),
    ("torso_center", "pelvis"),
    ("left_ear", "right_ear"),
)

_PAW_SPECS: Tuple[Tuple[str, str, float, float, float, float], ...] = (
    ("FL", "left_forepaw", 0.056, 0.024, 0.036, 0.020),
    ("FR", "right_forepaw", 0.056, -0.024, 0.036, 0.020),
    ("HL", "left_hindpaw", -0.072, 0.028, 0.044, 0.024),
    ("HR", "right_hindpaw", -0.072, -0.028, 0.044, 0.024),
)


def normalize_rat_keypoint_dict(
    points: Mapping[str, Sequence[float]],
    *,
    default_z: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Normalize keypoint or site names (Rat v2, DANNCE/Rat7M, Anipose, Mouse) into canonical 3D vectors."""
    out: Dict[str, np.ndarray] = {}
    for raw_name, coords in points.items():
        if not isinstance(coords, (list, tuple, np.ndarray)) or len(coords) < 2:
            continue
        raw_str = str(raw_name).strip()
        canonical = SITE_TO_RAT_V2_KEYPOINT.get(
            raw_str, _SITE_TO_RAT_V2_LOWER.get(raw_str.lower(), raw_str)
        )
        z = float(coords[2]) if len(coords) >= 3 else float(default_z)
        vec = np.array([float(coords[0]), float(coords[1]), z], dtype=np.float64)
        if np.all(np.isfinite(vec)):
            if canonical not in out or raw_str == canonical:
                out[canonical] = vec
    return out


def estimate_rodent_body_scale(
    frames_3d: Sequence[Pose3DFrame | Mapping[str, Sequence[float]]],
) -> Dict[str, Any]:
    """Estimate isotropic body-scale factor and length units relative to canonical Rat v2.

    Supports seamless ingestion of millimeter-scaled DANNCE/Rat7M captures (`unit_to_meters=0.001`)
    and cross-species rodents (e.g. adult 25 g *Mus musculus* with `body_scale_rel_rat ~ 0.35`)
    while snapping canonical Rat v2 inputs (`[0.80, 1.25]`) to exact `1.0`.
    """
    ratios: list[float] = []
    step = max(1, len(frames_3d) // 40)
    for item in frames_3d[::step]:
        pts = item.points if isinstance(item, Pose3DFrame) else item
        nk = normalize_rat_keypoint_dict(pts)
        for a, b in _SCALE_ANCHOR_PAIRS:
            if a in nk and b in nk:
                d_obs = float(np.linalg.norm(nk[a] - nk[b]))
                d_ref = float(
                    np.linalg.norm(
                        np.asarray(RAT_V2_CANONICAL_TRUNK_LOCAL[a], dtype=np.float64)
                        - np.asarray(RAT_V2_CANONICAL_TRUNK_LOCAL[b], dtype=np.float64)
                    )
                )
                if d_obs > 1e-6 and d_ref > 1e-6:
                    ratios.append(d_obs / d_ref)
    if not ratios:
        return {
            "raw_scale_ratio": 1.0,
            "detected_unit": "m",
            "unit_to_meters": 1.0,
            "body_scale_rel_rat": 1.0,
            "coord_to_rat_scale": 1.0,
        }
    raw_ratio = float(np.median(ratios))
    # Detect millimeter exports (e.g. DANNCE / Rat7M / Anipose in mm: raw_ratio ~ 250..1400)
    if raw_ratio > 50.0:
        unit = "mm"
        unit_to_m = 0.001
        body_scale = raw_ratio * 0.001
    elif raw_ratio > 10.0:
        unit = "cm"
        unit_to_m = 0.01
        body_scale = raw_ratio * 0.01
    else:
        unit = "m"
        unit_to_m = 1.0
        body_scale = raw_ratio

    # Keep canonical Rat v2 strictly unscaled inside [0.80, 1.25] for exact backward compatibility
    if 0.80 <= body_scale <= 1.25:
        eff_body_scale = 1.0
    else:
        eff_body_scale = float(np.clip(body_scale, 0.15, 4.0))

    return {
        "raw_scale_ratio": round(raw_ratio, 5),
        "detected_unit": unit,
        "unit_to_meters": unit_to_m,
        "body_scale_rel_rat": round(eff_body_scale, 4),
        "coord_to_rat_scale": float(unit_to_m / eff_body_scale),
    }


def _interpolate_missing_keypoints(
    norm_frames: list[Dict[str, np.ndarray]],
) -> list[Dict[str, np.ndarray]]:
    """Linearly bridge transient missing keypoints in real markerless 3D tracks."""
    n = len(norm_frames)
    if n < 2:
        return norm_frames
    all_keys: set[str] = set()
    for fr in norm_frames:
        all_keys.update(fr.keys())
    filled = [{k: v.copy() for k, v in fr.items()} for fr in norm_frames]
    for key in sorted(all_keys):
        valid_idx = [i for i in range(n) if key in norm_frames[i]]
        if not valid_idx or len(valid_idx) == n:
            continue
        xs = np.asarray(valid_idx, dtype=np.float64)
        ys = np.asarray([norm_frames[i][key] for i in valid_idx], dtype=np.float64)
        all_t = np.arange(n, dtype=np.float64)
        interp_cols = np.column_stack(
            [np.interp(all_t, xs, ys[:, dim]) for dim in range(3)]
        )
        for i in range(n):
            if key not in filled[i]:
                filled[i][key] = interp_cols[i]
    return filled


def _rotation_matrix_from_euler_xyz(
    roll: float, pitch: float, yaw: float
) -> np.ndarray:
    """Match RatNCAPController world transform: pitch around Y, roll around X, yaw around Z."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    return rz @ ry @ rx


def _euler_from_rotation_matrix(rot: np.ndarray) -> Tuple[float, float, float]:
    """Extract (roll, pitch, yaw) from R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r20 = float(np.clip(rot[2, 0], -1.0, 1.0))
    pitch = math.asin(-r20)
    cos_p = math.cos(pitch)
    if abs(cos_p) > 1e-6:
        roll = math.atan2(float(rot[2, 1]), float(rot[2, 2]))
        yaw = math.atan2(float(rot[1, 0]), float(rot[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(-float(rot[0, 1]), float(rot[1, 1]))
    return roll, pitch, yaw


def estimate_rat_v2_root_pose(
    keypoints_3d: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, float, float, float]:
    """Estimate 6-DOF Rat v2 trunk pose `(root_pos, roll, pitch, yaw)` via Kabsch/SVD."""
    # Prefer rigid anterior/cranial landmarks for clean 6-DOF trunk alignment
    # (excludes articulated lumbar/pelvis deflection so trunk pose is unbiased and
    # multi-landmark averaging suppresses single-point common-mode noise).
    trunk_keys = [
        k
        for k in (
            "torso_center",
            "thorax",
            "neck",
            "head",
            "nose",
            "left_ear",
            "right_ear",
        )
        if k in keypoints_3d
    ]
    if len(trunk_keys) < 3 and "pelvis" in keypoints_3d:
        trunk_keys.append("pelvis")

    if len(trunk_keys) >= 3:
        local_pts = np.asarray(
            [RAT_V2_CANONICAL_TRUNK_LOCAL[k] for k in trunk_keys], dtype=np.float64
        )
        world_pts = np.asarray([keypoints_3d[k] for k in trunk_keys], dtype=np.float64)
        # Check non-collinearity (rank >= 2)
        local_centered = local_pts - np.mean(local_pts, axis=0)
        world_centered = world_pts - np.mean(world_pts, axis=0)
        if np.linalg.matrix_rank(local_centered) >= 2:
            cov = local_centered.T @ world_centered
            u, _, vt = np.linalg.svd(cov)
            rot = vt.T @ u.T
            if float(np.linalg.det(rot)) < 0.0:
                vt[-1, :] *= -1.0
                rot = vt.T @ u.T
            roll, pitch, yaw = _euler_from_rotation_matrix(rot)
            # Kabsch optimal translation averages all trunk landmarks (1/sqrt(K) noise reduction)
            root_pos = np.mean(world_pts, axis=0) - rot @ np.mean(local_pts, axis=0)
            return root_pos, roll, pitch, yaw

    # Fallback when fewer than 3 trunk points exist
    if "torso_center" in keypoints_3d:
        root_pos = keypoints_3d["torso_center"].copy()
    elif "thorax" in keypoints_3d:
        root_pos = keypoints_3d["thorax"].copy()
    elif keypoints_3d:
        root_pos = np.mean(np.asarray(list(keypoints_3d.values())), axis=0)
    else:
        root_pos = np.asarray(RAT_V2_ROOT_POS_INIT, dtype=np.float64).copy()

    ant = keypoints_3d.get("head")
    if ant is None:
        ant = keypoints_3d.get("nose")
    if ant is None:
        ant = keypoints_3d.get("thorax")
    post = keypoints_3d.get("pelvis")
    if post is None:
        post = keypoints_3d.get("torso_center")
    yaw = 0.0
    pitch = 0.0
    if ant is not None and post is not None:
        diff = ant - post
        horiz = math.hypot(float(diff[0]), float(diff[1]))
        if horiz > 1e-6:
            yaw = math.atan2(float(diff[1]), float(diff[0]))
            pitch = -math.atan2(float(diff[2]), horiz)
    return root_pos, 0.0, pitch, yaw


def forward_kinematics_rat_v2(
    *,
    root_pos: Sequence[float] | np.ndarray,
    root_euler: Sequence[float],
    qpos: Sequence[float] | np.ndarray,
    interface: NeuromechanicalInterface | None = None,
    ground_z: float = 0.0,
) -> Dict[str, np.ndarray]:
    """Forward kinematics mapping Rat v2 `(root_pos, root_euler, qpos)` -> 16 3D keypoints."""
    iface = interface or NeuromechanicalInterface("flexor_extensor")
    q = np.asarray(qpos, dtype=np.float64).reshape(-1)
    root = np.asarray(root_pos, dtype=np.float64).reshape(3)
    roll, pitch, yaw = float(root_euler[0]), float(root_euler[1]), float(root_euler[2])
    rot = _rotation_matrix_from_euler_xyz(roll, pitch, yaw)

    seg_slices: Dict[str, slice] = {}
    off = 0
    for seg in iface.segments:
        n = len(RAT_V2_SEGMENTS[seg])
        seg_slices[seg] = slice(off, off + n)
        off += n

    def to_world(lx: float, ly: float, lz: float) -> np.ndarray:
        return root + rot @ np.array([lx, ly, lz], dtype=np.float64)

    lumbar_e = float(q[seg_slices["lumbarE"]][0])
    lumbar_b = float(q[seg_slices["lumbarBT"]][0])

    kps: Dict[str, np.ndarray] = {
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
        "tail_mid": to_world(-0.195, 0.034 * lumbar_b, -0.016),
        "tail_tip": to_world(-0.285, 0.042 * lumbar_b, -0.024),
    }

    for limb, label, base_x, base_y, stride_len, lift_h in _PAW_SPECS:
        sl = seg_slices[f"limb{limb}"]
        q_seg = q[sl]
        poseq_seg = iface.poseq[sl]
        if limb in ("FL", "FR"):
            dq_sag = float(q_seg[4] - poseq_seg[4])
            dq_elbow = float(q_seg[5] - poseq_seg[5])
            lift = max(0.0, dq_elbow / 0.52) * lift_h
        else:
            dq_sag = float(q_seg[2] - poseq_seg[2])
            dq_knee = float(q_seg[3] - poseq_seg[3])
            lift = max(0.0, dq_knee / 0.58) * lift_h
        lx = base_x + stride_len * math.sin(dq_sag)
        ly = base_y
        p_world = to_world(lx, ly, -RAT_V2_ROOT_POS_INIT[2] + 0.004)
        p_world[2] = float(ground_z) + 0.004 + min(lift, lift_h * 1.4)
        kps[label] = p_world
    return kps


def solve_rat_v2_ik_frame(
    keypoints_3d: Mapping[str, Sequence[float]],
    *,
    interface: NeuromechanicalInterface | None = None,
    prev_qpos: np.ndarray | None = None,
    temporal_regularization: float = 0.12,
    ground_z: float | None = None,
) -> Dict[str, Any]:
    """Solve anatomically constrained 31-DOF Rat v2 inverse kinematics for a single frame.

    Recovers 6-DOF trunk pose via rigid Kabsch alignment, lumbar extension/lateral bend
    from posterior spine curvature, and 4-limb sagittal/flexion/abduction joint angles
    strictly within `RAT_V2_JOINT_RANGES`.
    """
    iface = interface or NeuromechanicalInterface("flexor_extensor")
    norm_kps = normalize_rat_keypoint_dict(keypoints_3d)
    root_pos, roll, pitch, yaw = estimate_rat_v2_root_pose(norm_kps)
    rot = _rotation_matrix_from_euler_xyz(roll, pitch, yaw)
    rot_inv = rot.T

    def to_local(world_pt: np.ndarray) -> np.ndarray:
        return rot_inv @ (world_pt - root_pos)

    seg_slices: Dict[str, slice] = {}
    off = 0
    for seg in iface.segments:
        n = len(RAT_V2_SEGMENTS[seg])
        seg_slices[seg] = slice(off, off + n)
        off += n

    q = iface.poseq.copy()

    # Estimate ground level from lowest observed paw if not explicitly provided
    paw_zs = [
        float(norm_kps[label][2]) for _, label, *_ in _PAW_SPECS if label in norm_kps
    ]
    est_ground_z = (
        float(ground_z)
        if ground_z is not None
        else (
            min(paw_zs) - 0.004
            if paw_zs
            else float(root_pos[2] - RAT_V2_ROOT_POS_INIT[2])
        )
    )

    # 1. Solve lumbar extension & lateral bending from lumbar / pelvis offsets
    le_vals: list[float] = []
    lb_vals: list[float] = []
    if "lumbar" in norm_kps:
        loc_l = to_local(norm_kps["lumbar"])
        le_vals.append(float(loc_l[2]) / 0.008)
        lb_vals.append(float(loc_l[1]) / 0.012)
    if "pelvis" in norm_kps:
        loc_p = to_local(norm_kps["pelvis"])
        le_vals.append(float(loc_p[2] + 0.002) / 0.014)
        lb_vals.append(float(loc_p[1]) / 0.022)
    if le_vals:
        sl_le = seg_slices["lumbarE"]
        q[sl_le] = np.clip(float(np.mean(le_vals)), iface.low[sl_le], iface.high[sl_le])
    if lb_vals:
        sl_lbt = seg_slices["lumbarBT"]
        lb_mean = float(np.mean(lb_vals))
        q[sl_lbt] = np.clip(
            np.array([lb_mean, 0.64 * lb_mean], dtype=np.float64),
            iface.low[sl_lbt],
            iface.high[sl_lbt],
        )

    # 2. Solve limb DOFs from paw trajectories in body-local frame + ground clearance
    paw_clearance: Dict[str, float] = {}
    for limb, label, base_x, base_y, stride_len, lift_h in _PAW_SPECS:
        sl = seg_slices[f"limb{limb}"]
        if label not in norm_kps:
            paw_clearance[limb] = 0.0
            continue
        p_world = norm_kps[label]
        p_local = to_local(p_world)
        dx = float(p_local[0]) - base_x
        dy = float(p_local[1]) - base_y
        clearance = max(0.0, float(p_world[2]) - (est_ground_z + 0.004))
        paw_clearance[limb] = clearance

        sin_sag = float(np.clip(dx / max(stride_len, 1e-4), -0.98, 0.98))
        dq_sag = math.asin(sin_sag)
        lift_norm = float(np.clip(clearance / max(lift_h, 1e-4), 0.0, 1.4))
        stance_norm = float(np.clip(1.0 - lift_norm, 0.0, 1.0))
        abd_angle = float(np.clip(dy / 0.055, -0.35, 0.35))

        if limb in ("FL", "FR"):
            # [scapula_sup, scapula_abd, scapula_ext, shoulder_sup, shoulder_ext, elbow, wrist, finger]
            dq_vec = np.array(
                [
                    0.08 * dq_sag,
                    abd_angle * 0.25 + 0.05 * (lift_norm - 0.5 * stance_norm),
                    0.04 * dq_sag,
                    0.16 * (lift_norm - 0.35 * stance_norm),
                    dq_sag,
                    0.52 * lift_norm - 0.36 * stance_norm,
                    0.35 * (stance_norm - lift_norm),
                    0.28 * (-sin_sag - 0.5 * lift_norm),
                ],
                dtype=np.float64,
            )
        else:
            # [hip_sup, hip_abd, hip_ext, knee, ankle, toe]
            dq_vec = np.array(
                [
                    0.12 * (lift_norm - 0.5 * stance_norm),
                    abd_angle * 0.45 + 0.14 * lift_norm,
                    dq_sag,
                    0.58 * lift_norm - 0.34 * stance_norm,
                    0.42 * (-sin_sag - 0.6 * lift_norm),
                    0.34 * (-sin_sag - 0.5 * lift_norm),
                ],
                dtype=np.float64,
            )
        q[sl] = np.clip(iface.poseq[sl] + dq_vec, iface.low[sl], iface.high[sl])

    # Optional temporal continuity regularization
    if prev_qpos is not None:
        reg = float(np.clip(temporal_regularization, 0.0, 0.8))
        if reg > 0.0:
            q = np.clip(
                (1.0 - reg) * q + reg * np.asarray(prev_qpos, dtype=np.float64),
                iface.low,
                iface.high,
            )

    # Forward kinematics verification & RMSE
    recon = forward_kinematics_rat_v2(
        root_pos=root_pos,
        root_euler=(roll, pitch, yaw),
        qpos=q,
        interface=iface,
        ground_z=est_ground_z,
    )
    sq_errs: list[float] = []
    for k, obs_pt in norm_kps.items():
        if k in recon:
            diff = recon[k] - obs_pt
            sq_errs.append(float(np.dot(diff, diff)))
    rmse_m = math.sqrt(float(np.mean(sq_errs))) if sq_errs else 0.0

    return {
        "root_pos": [round(float(x), 6) for x in root_pos],
        "root_euler": [
            round(float(roll), 5),
            round(float(pitch), 5),
            round(float(yaw), 5),
        ],
        "qpos": q,
        "reconstructed_keypoints_3d": {
            k: [round(float(v[0]), 5), round(float(v[1]), 5), round(float(v[2]), 5)]
            for k, v in recon.items()
        },
        "ik_rmse_m": round(rmse_m, 6),
        "paw_clearance_m": {k: round(v, 5) for k, v in paw_clearance.items()},
        "ground_z_m": round(est_ground_z, 5),
        "joint_limit_violations": int(
            np.sum((q < iface.low - 1e-9) | (q > iface.high + 1e-9))
        ),
    }


def invert_spring_muscle_activations(
    *,
    q_seq: np.ndarray,
    dt: float,
    interface: NeuromechanicalInterface | None = None,
    co_contraction_baseline: float = 0.45,
) -> Dict[str, Any]:
    """Exact closed-form inverse dynamics & antagonistic spring-muscle activation inversion.

    Given joint trajectories `q_seq (T, 31)`, computes finite-difference `(qvel, qacc)`,
    required joint torques `tau_req = I * ddq + d * dq`, and solves the exact KKT
    minimum-deviation antagonistic `(flx, ext)` activations satisfying:
        `ext * ke * xe - flx * kf * xf - kd * dq = tau_req`,   `flx, ext in [0, 1]`.
    """
    iface = interface or NeuromechanicalInterface("flexor_extensor")
    q_arr = np.asarray(q_seq, dtype=np.float64)
    if q_arr.ndim == 1:
        q_arr = q_arr.reshape(1, -1)
    n_frames, dof = q_arr.shape
    step_dt = max(1e-4, float(dt))

    if n_frames >= 3:
        dq = np.gradient(q_arr, step_dt, axis=0)
        ddq = np.gradient(dq, step_dt, axis=0)
    elif n_frames == 2:
        vel = (q_arr[1] - q_arr[0]) / step_dt
        dq = np.vstack([vel, vel])
        ddq = np.zeros_like(q_arr)
    else:
        dq = np.zeros_like(q_arr)
        ddq = np.zeros_like(q_arr)

    dq = np.clip(dq, -20.0, 20.0)
    ddq = np.clip(ddq, -140.0, 140.0)

    inertia = np.maximum(0.008 * iface.keq, 0.0012)
    damping = np.maximum(0.18 * iface.keq, 0.025)
    tau_req = np.clip(
        inertia[None, :] * ddq + damping[None, :] * dq,
        -iface.trq_max[None, :],
        iface.trq_max[None, :],
    )

    flx_seq = np.zeros((n_frames, dof), dtype=np.float64)
    ext_seq = np.zeros((n_frames, dof), dtype=np.float64)
    tau_recon = np.zeros((n_frames, dof), dtype=np.float64)
    stiff_seq = np.zeros((n_frames, dof), dtype=np.float64)
    powers: list[Dict[str, float]] = []

    c0 = float(np.clip(co_contraction_baseline, 0.05, 0.95))
    ke = iface.spring.ke
    kf = iface.spring.kf
    ce = iface.spring.ce
    cf = iface.spring.cf

    for t in range(n_frames):
        qt = q_arr[t]
        dqt = dq[t]
        # Spring torque needed before viscous damping term `-kd * dq`
        y = tau_req[t] + iface.kd * dqt
        xf = qt - iface.low + cf
        xe = iface.high - qt + ce
        a_coef = ke * xe  # >= 0 coefficient on ext
        b_coef = kf * xf  # >= 0 coefficient on flx
        denom = np.maximum(a_coef * a_coef + b_coef * b_coef, 1e-12)

        # Exact unconstrained KKT projection of (c0, c0) onto A*ext - B*flx = y
        lam = (y - (a_coef - b_coef) * c0) / denom
        e_opt = c0 + lam * a_coef
        f_opt = c0 - lam * b_coef

        # Box-constraint projection onto [0, 1]^2 preserving exact torque whenever feasible
        e_clamped = np.clip(e_opt, 0.0, 1.0)
        f_clamped = np.clip(f_opt, 0.0, 1.0)
        for j in range(dof):
            aj = float(a_coef[j])
            bj = float(b_coef[j])
            yj = float(y[j])
            if e_opt[j] < 0.0 or e_opt[j] > 1.0:
                ej = float(e_clamped[j])
                fj = (
                    float(np.clip((aj * ej - yj) / bj, 0.0, 1.0))
                    if abs(bj) > 1e-9
                    else c0
                )
                e_clamped[j] = ej
                f_clamped[j] = fj
            elif f_opt[j] < 0.0 or f_opt[j] > 1.0:
                fj = float(f_clamped[j])
                ej = (
                    float(np.clip((yj + bj * fj) / aj, 0.0, 1.0))
                    if abs(aj) > 1e-9
                    else c0
                )
                e_clamped[j] = ej
                f_clamped[j] = fj

        flx_seq[t] = f_clamped
        ext_seq[t] = e_clamped
        tau_rec_t = iface.compute_torque(q=qt, dq=dqt, flx=f_clamped, ext=e_clamped)
        tau_recon[t] = tau_rec_t
        stiff_seq[t] = iface.effective_stiffness(flx=f_clamped, ext=e_clamped)
        powers.append(compute_mechanical_power(tau_rec_t, dqt))

    residual_rmse = float(np.sqrt(np.mean(np.square(tau_recon - tau_req))))
    return {
        "qvel": dq,
        "qacc": ddq,
        "tau_required": tau_req,
        "tau_reconstructed": tau_recon,
        "flx_activation": flx_seq,
        "ext_activation": ext_seq,
        "effective_stiffness": stiff_seq,
        "mechanical_power": powers,
        "torque_residual_rmse_nm": round(residual_rmse, 6),
        "mean_positive_power_w": round(
            float(np.mean([p["positive_power_w"] for p in powers])), 5
        ),
        "mean_negative_power_w": round(
            float(np.mean([p["negative_power_w"] for p in powers])), 5
        ),
    }


def _circular_phase_offset(
    sig_a: np.ndarray,
    sig_ref: np.ndarray,
    *,
    dt: float = 0.02,
    center_freq_hz: float | None = None,
) -> Tuple[float, float]:
    """Estimate normalized interlimb phase offset in `[0, 1)` and phase-locking value (PLV).

    Uses frequency-band-filtered discrete Hilbert analytic phase via FFT (zero SciPy
    dependency) so high-frequency 3D tracking jitter is rejected cleanly.
    """
    a = np.asarray(sig_a, dtype=np.float64) - float(np.mean(sig_a))
    r = np.asarray(sig_ref, dtype=np.float64) - float(np.mean(sig_ref))
    n = min(a.size, r.size)
    if n < 8 or float(np.std(a)) < 1e-6 or float(np.std(r)) < 1e-6:
        return 0.5, 0.0

    freqs = np.fft.fftfreq(n, d=max(1e-4, float(dt)))
    nyquist_lim = 0.48 / max(1e-4, float(dt))
    # Band-pass positive frequencies around the locomotor cadence (supporting both rat 1-6 Hz & mouse 2-12 Hz)
    if center_freq_hz is not None and center_freq_hz > 0.5:
        lo_f = max(0.5, 0.55 * float(center_freq_hz))
        hi_f = min(nyquist_lim, 1.65 * float(center_freq_hz))
    else:
        lo_f = 0.6
        hi_f = min(14.0, nyquist_lim)

    def _analytic_bandpass(x: np.ndarray) -> np.ndarray:
        fft_x = np.fft.fft(x)
        h = np.zeros(x.size, dtype=np.float64)
        mask = (freqs >= lo_f) & (freqs <= hi_f)
        h[mask] = 2.0
        if not np.any(mask):
            # Fallback to full positive spectrum
            if x.size % 2 == 0:
                h[1 : x.size // 2] = 2.0
            else:
                h[1 : (x.size + 1) // 2] = 2.0
        return np.fft.ifft(fft_x * h)

    za = _analytic_bandpass(a[:n])
    zr = _analytic_bandpass(r[:n])
    margin = min(n // 8, 8)
    if n - 2 * margin >= 6:
        za = za[margin : n - margin]
        zr = zr[margin : n - margin]
    # Amplitude-weighted circular mean suppresses low-SNR silent frames
    cross = za * np.conj(zr)
    mag = np.abs(cross)
    total_mag = float(np.sum(mag))
    if total_mag < 1e-12:
        return 0.5, 0.0
    phasor = np.sum(cross) / total_mag
    plv = float(np.clip(np.abs(phasor), 0.0, 1.0))
    offset = float((np.angle(phasor) / (2.0 * math.pi)) % 1.0)
    return offset, plv


def infer_spinal_circuit_phenotype(
    frames_3d: Sequence[Pose3DFrame | Mapping[str, Sequence[float]]],
    *,
    dt: float = 0.02,
    ik_frames: Sequence[Dict[str, Any]] | None = None,
    auto_scale: bool = True,
) -> Dict[str, Any]:
    """Infer latent brainstem command, Hildebrand gait phase, and V0/V3 spinal circuit phenotype.

    Operates directly on 3D keypoint sequences (or solved IK frames) to classify
    locomotor gaits and identify commissural interneuron deficits (e.g. Bellardita & Kiehn
    2015 `V0v+V0d` ablation where bounding emerges at low/moderate speeds). Supports
    automatic Sim(3) body-scale & millimeter-unit normalization with Froude dynamic
    similarity (`v_eq = v / sqrt(body_scale)`) for cross-species rodent transfer (Rat & Mouse).
    """
    if not frames_3d:
        return {
            "dominant_gait": "stand",
            "phenotype_class": "intact_stand",
            "confidence": 0.0,
            "mean_speed_mps": 0.0,
            "froude_equiv_speed_mps": 0.0,
            "detected_body_scale": 1.0,
            "detected_unit": "m",
            "step_frequency_hz": 0.0,
            "duty_factor": 1.0,
            "phase_offsets_rel_fr": {"FR": 0.0, "FL": 0.5, "HL": 0.0, "HR": 0.5},
            "phase_locking_value": 0.0,
            "estimated_brainstem_command": 0.0,
            "v0_integrity_index": 1.0,
            "v3_integrity_index": 1.0,
            "lumbar_excursion_rad": 0.0,
        }

    step_dt = max(1e-4, float(dt))
    iface = NeuromechanicalInterface("flexor_extensor")
    scale_info = (
        estimate_rodent_body_scale(frames_3d)
        if auto_scale
        else {
            "detected_unit": "m",
            "unit_to_meters": 1.0,
            "body_scale_rel_rat": 1.0,
            "coord_to_rat_scale": 1.0,
        }
    )
    coord_scale = float(scale_info["coord_to_rat_scale"])
    body_scale = float(scale_info["body_scale_rel_rat"])

    solved = list(ik_frames) if ik_frames is not None else []
    if not solved:
        all_paw_zs: list[float] = []
        raw_norm_frames: list[Dict[str, np.ndarray]] = []
        for item in frames_3d:
            pts = item.points if isinstance(item, Pose3DFrame) else item
            nk = normalize_rat_keypoint_dict(pts)
            if abs(coord_scale - 1.0) > 1e-6:
                nk = {k: v * coord_scale for k, v in nk.items()}
            raw_norm_frames.append(nk)
        norm_frames = _interpolate_missing_keypoints(raw_norm_frames)
        for nk in norm_frames:
            for _, lbl, *_ in _PAW_SPECS:
                if lbl in nk:
                    all_paw_zs.append(float(nk[lbl][2]))
        seq_ground_z = (
            float(np.percentile(all_paw_zs, 15)) - 0.004 if all_paw_zs else None
        )
        prev_q: np.ndarray | None = None
        for nk in norm_frames:
            res = solve_rat_v2_ik_frame(
                nk, interface=iface, prev_qpos=prev_q, ground_z=seq_ground_z
            )
            prev_q = res["qpos"]
            solved.append(res)

    n = len(solved)
    roots = np.asarray([f["root_pos"] for f in solved], dtype=np.float64)
    q_seq = np.asarray([f["qpos"] for f in solved], dtype=np.float64)

    # 1. Estimate horizontal speed over a stride-scale displacement window (~0.20 s)
    #    to eliminate positive Rayleigh bias from frame-to-frame differentiated 3D noise.
    #    `roots` are in rat-normalized meters; multiply by `body_scale` for true physical m/s
    #    and apply Froude dynamic similarity (`v_phys / sqrt(body_scale)`) for gait-regime bands.
    if n >= 2:
        win = min(max(1, int(round(0.20 / step_dt))), max(1, n - 1))
        disp_win = roots[win:, :2] - roots[:-win, :2]
        inst_speeds_rat_geom = np.linalg.norm(disp_win, axis=1) / (float(win) * step_dt)
        inst_speeds_phys = inst_speeds_rat_geom * body_scale
        mean_speed = float(np.median(inst_speeds_phys))
        speed_std = float(np.std(inst_speeds_phys))
        froude_speed = mean_speed / math.sqrt(max(body_scale, 0.15))
        froude_speed_std = speed_std / math.sqrt(max(body_scale, 0.15))
    else:
        mean_speed = 0.0
        speed_std = 0.0
        froude_speed = 0.0
        froude_speed_std = 0.0

    # 2. Extract sagittal limb-swing signals (shoulder_extend / hip_extend) and paw contacts
    seg_slices: Dict[str, slice] = {}
    off = 0
    for seg in iface.segments:
        seg_len = len(RAT_V2_SEGMENTS[seg])
        seg_slices[seg] = slice(off, off + seg_len)
        off += seg_len

    limb_signals: Dict[str, np.ndarray] = {}
    limb_clearances: Dict[str, np.ndarray] = {}
    for limb in RAT_LIMB_NAMES:
        sl = seg_slices[f"limb{limb}"]
        sag_idx = 4 if limb in ("FL", "FR") else 2
        limb_signals[limb] = q_seq[:, sl.start + sag_idx]
        limb_clearances[limb] = np.asarray(
            [float(f["paw_clearance_m"].get(limb, 0.0)) for f in solved],
            dtype=np.float64,
        )

    # 3. Dominant locomotor step frequency via combined vertical-lift & sagittal power spectrum
    step_freq_hz = 0.0
    combined_limb_signals: Dict[str, np.ndarray] = {}
    for limb in RAT_LIMB_NAMES:
        sag = limb_signals[limb] - float(np.mean(limb_signals[limb]))
        clr = limb_clearances[limb] - float(np.mean(limb_clearances[limb]))
        sag_std = float(np.std(sag))
        clr_std = float(np.std(clr))
        # Vertical paw lift (~15-24 mm) has very high SNR even under 5+ mm 3D tracking noise
        if clr_std > 0.0015:
            combined_limb_signals[limb] = clr / clr_std
        elif sag_std > 1e-5:
            combined_limb_signals[limb] = sag / sag_std
        else:
            combined_limb_signals[limb] = sag

    if n >= 12 and froude_speed >= 0.05:
        freqs = np.fft.rfftfreq(n, d=step_dt)
        psd = np.zeros_like(freqs)
        for limb in RAT_LIMB_NAMES:
            sig = combined_limb_signals[limb]
            psd += np.abs(np.fft.rfft(sig)) ** 2
        max_cadence = min(14.0, 0.48 / step_dt)
        band = (freqs >= 0.8) & (freqs <= max_cadence)
        if np.any(band):
            band_freqs = freqs[band]
            band_psd = psd[band]
            peak_idx = int(np.argmax(band_psd))
            if float(band_psd[peak_idx]) > 1e-8:
                step_freq_hz = float(band_freqs[peak_idx])

    # 4. Duty factor from edge-preserving 3-tap median-filtered paw clearance above sequence floor
    duty_vals: list[float] = []
    for limb in RAT_LIMB_NAMES:
        clr = limb_clearances[limb]
        if clr.size >= 3:
            padded = np.pad(clr, (1, 1), mode="edge")
            clr_s = np.median(
                np.vstack([padded[:-2], padded[1:-1], padded[2:]]), axis=0
            )
        else:
            clr_s = clr
        floor_c = float(np.percentile(clr_s, 15)) if clr_s.size > 0 else 0.0
        rel_clr = np.maximum(0.0, clr_s - floor_c)
        peak_clr = float(np.percentile(rel_clr, 92)) if rel_clr.size > 0 else 0.0
        if peak_clr > 0.003:
            thresh = max(0.0025, 0.35 * peak_clr)
            duty_vals.append(float(np.mean(rel_clr <= thresh)))
    if froude_speed < 0.05:
        duty_factor = 1.0
    elif duty_vals:
        duty_factor = float(np.clip(np.mean(duty_vals), 0.35, 0.92))
    else:
        duty_factor = float(np.clip(0.80 - 0.36 * min(froude_speed, 1.0), 0.42, 0.85))

    # 5. Interlimb phase offsets relative to FR & phase-locking stability
    phase_offsets: Dict[str, float] = {"FR": 0.0}
    plvs: list[float] = []
    for limb in ("FL", "HL", "HR"):
        ref_sig = combined_limb_signals["FR"]
        cmp_sig = combined_limb_signals[limb]
        offset, plv = _circular_phase_offset(
            cmp_sig,
            ref_sig,
            dt=step_dt,
            center_freq_hz=step_freq_hz if step_freq_hz > 0.5 else None,
        )
        phase_offsets[limb] = offset
        plvs.append(plv)
    mean_plv = float(np.mean(plvs)) if plvs else 0.0

    # 6. Classify dominant gait & mechanistic spinal phenotype (Froude-normalized speed bands)
    gait = classify_quadruped_gait(
        speed_mps=froude_speed,
        phase_offsets_rel_fr=phase_offsets,
        stance_fraction=duty_factor,
    )

    def _circ_dist(a: float, b: float) -> float:
        d = abs((a - b) % 1.0)
        return min(d, 1.0 - d)

    lr_fore_sync = _circ_dist(phase_offsets["FL"], 0.0)
    lr_hind_sync = _circ_dist(phase_offsets["HL"], phase_offsets["HR"])
    lr_sync_score = 1.0 - 2.0 * min(0.5, 0.5 * (lr_fore_sync + lr_hind_sync))

    est_command = float(np.clip(froude_speed, 0.0, 1.0))

    if 0.16 <= froude_speed <= 0.66:
        v0_integrity = float(np.clip(1.0 - lr_sync_score, 0.0, 1.0))
        v3_integrity = 1.0
    elif froude_speed > 0.66:
        v0_integrity = 1.0
        v3_integrity = float(np.clip(lr_sync_score, 0.0, 1.0))
    else:
        v0_integrity = 1.0
        v3_integrity = 1.0

    lumbar_sl = seg_slices["lumbarE"]
    lumbar_exc = float(np.ptp(q_seq[:, lumbar_sl.start])) if q_seq.shape[0] > 1 else 0.0

    if froude_speed < 0.05:
        phenotype = "intact_stand"
        conf = 0.95
    elif froude_speed_std > 0.18 and n >= 40:
        phenotype = "gait_transition"
        conf = float(np.clip(0.65 + 0.3 * mean_plv, 0.0, 1.0))
    elif mean_plv < 0.16 and froude_speed >= 0.12:
        phenotype = "ataxic_low_coordination"
        conf = float(np.clip(1.0 - mean_plv, 0.4, 0.95))
    elif gait == "bound" and 0.16 <= froude_speed <= 0.66:
        phenotype = "v0_commissural_deficit_phenotype"
        conf = float(
            np.clip(0.55 + 0.45 * lr_sync_score * max(mean_plv, 0.4), 0.5, 0.99)
        )
    elif gait in ("walk", "trot") and froude_speed >= 0.72:
        phenotype = "v3_synchrony_deficit_phenotype"
        conf = float(
            np.clip(0.55 + 0.45 * (1.0 - lr_sync_score) * max(mean_plv, 0.4), 0.5, 0.99)
        )
    else:
        phenotype = f"intact_{gait}"
        conf = float(np.clip(0.60 + 0.38 * mean_plv, 0.5, 0.99))

    return {
        "dominant_gait": gait,
        "phenotype_class": phenotype,
        "confidence": round(conf, 4),
        "mean_speed_mps": round(mean_speed, 4),
        "froude_equiv_speed_mps": round(froude_speed, 4),
        "detected_body_scale": round(body_scale, 4),
        "detected_unit": str(scale_info["detected_unit"]),
        "speed_std_mps": round(speed_std, 4),
        "step_frequency_hz": round(step_freq_hz, 4),
        "duty_factor": round(duty_factor, 4),
        "phase_offsets_rel_fr": {k: round(v, 4) for k, v in phase_offsets.items()},
        "phase_locking_value": round(mean_plv, 4),
        "lr_synchrony_index": round(lr_sync_score, 4),
        "estimated_brainstem_command": round(est_command, 4),
        "v0_integrity_index": round(v0_integrity, 4),
        "v3_integrity_index": round(v3_integrity, 4),
        "lumbar_excursion_rad": round(lumbar_exc, 4),
    }


def solve_rat_v2_ik_sequence(
    frames_3d: Sequence[Pose3DFrame | Mapping[str, Sequence[float]]],
    *,
    dt: float = 0.02,
    interface_type: str = "flexor_extensor",
    temporal_regularization: float = 0.12,
    co_contraction_baseline: float = 0.45,
    auto_scale: bool = True,
) -> Dict[str, Any]:
    """Full Real-to-Sim Rat v2 pipeline: 3D keypoints -> 31-DOF IK -> Spring-Muscle ID -> Spinal Inference."""
    iface = NeuromechanicalInterface(interface_type)
    ik_frames: list[Dict[str, Any]] = []
    prev_q: np.ndarray | None = None

    scale_info = (
        estimate_rodent_body_scale(frames_3d)
        if auto_scale and frames_3d
        else {
            "detected_unit": "m",
            "unit_to_meters": 1.0,
            "body_scale_rel_rat": 1.0,
            "coord_to_rat_scale": 1.0,
        }
    )
    coord_scale = float(scale_info["coord_to_rat_scale"])

    all_paw_zs: list[float] = []
    raw_norm_seq: list[Dict[str, np.ndarray]] = []
    for item in frames_3d:
        pts = item.points if isinstance(item, Pose3DFrame) else item
        nk = normalize_rat_keypoint_dict(pts)
        if abs(coord_scale - 1.0) > 1e-6:
            nk = {k: v * coord_scale for k, v in nk.items()}
        raw_norm_seq.append(nk)
    norm_seq = _interpolate_missing_keypoints(raw_norm_seq)
    for nk in norm_seq:
        for _, lbl, *_ in _PAW_SPECS:
            if lbl in nk:
                all_paw_zs.append(float(nk[lbl][2]))
    seq_ground_z = float(np.percentile(all_paw_zs, 15)) - 0.004 if all_paw_zs else None

    for nk in norm_seq:
        res = solve_rat_v2_ik_frame(
            nk,
            interface=iface,
            prev_qpos=prev_q,
            temporal_regularization=temporal_regularization,
            ground_z=seq_ground_z,
        )
        prev_q = res["qpos"]
        ik_frames.append(res)

    if not ik_frames:
        return {
            "frames": [],
            "inverse_dynamics": {},
            "spinal_circuit_inference": infer_spinal_circuit_phenotype([], dt=dt),
            "summary": {"frames": 0, "mean_ik_rmse_m": 0.0},
        }

    q_seq = np.asarray([f["qpos"] for f in ik_frames], dtype=np.float64)
    id_res = invert_spring_muscle_activations(
        q_seq=q_seq,
        dt=dt,
        interface=iface,
        co_contraction_baseline=co_contraction_baseline,
    )
    spinal = infer_spinal_circuit_phenotype(
        frames_3d, dt=dt, ik_frames=ik_frames, auto_scale=auto_scale
    )

    enriched_frames: list[Dict[str, Any]] = []
    for idx, f in enumerate(ik_frames):
        enriched_frames.append(
            {
                "root_pos": f["root_pos"],
                "root_euler": f["root_euler"],
                "qpos": [round(float(v), 5) for v in f["qpos"]],
                "qvel": [round(float(v), 5) for v in id_res["qvel"][idx]],
                "torque": [
                    round(float(v), 5) for v in id_res["tau_reconstructed"][idx]
                ],
                "flx_activation": [
                    round(float(v), 4) for v in id_res["flx_activation"][idx]
                ],
                "ext_activation": [
                    round(float(v), 4) for v in id_res["ext_activation"][idx]
                ],
                "stiffness": [
                    round(float(v), 4) for v in id_res["effective_stiffness"][idx]
                ],
                "mechanical_power": {
                    k: round(v, 5) for k, v in id_res["mechanical_power"][idx].items()
                },
                "reconstructed_keypoints_3d": f["reconstructed_keypoints_3d"],
                "ik_rmse_m": f["ik_rmse_m"],
                "paw_clearance_m": f["paw_clearance_m"],
                "joint_limit_violations": f["joint_limit_violations"],
            }
        )

    mean_rmse = float(np.mean([f["ik_rmse_m"] for f in ik_frames]))
    total_violations = int(sum(f["joint_limit_violations"] for f in ik_frames))
    return {
        "frames": enriched_frames,
        "inverse_dynamics": {
            "torque_residual_rmse_nm": id_res["torque_residual_rmse_nm"],
            "mean_positive_power_w": id_res["mean_positive_power_w"],
            "mean_negative_power_w": id_res["mean_negative_power_w"],
        },
        "spinal_circuit_inference": spinal,
        "summary": {
            "frames": len(enriched_frames),
            "interface": iface.interface_type,
            "mean_ik_rmse_m": round(mean_rmse, 6),
            "joint_limit_violations": total_violations,
        },
    }
