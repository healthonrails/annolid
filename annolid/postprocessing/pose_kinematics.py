"""Pose-specific calibrated kinematics: per-keypoint speeds, bone lengths, joint angles, egocentric pose, and dyadic keypoints.

Complements whole-animal centroid kinematics in ``kinematics_stats.py`` with the
skeleton-level biomechanical metrics researchers expect from SLEAP, DeepLabCut,
and ``movement``.
"""

from __future__ import annotations

from collections import defaultdict
import csv
from dataclasses import dataclass, field
import io
from itertools import combinations
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from annolid.annotation.pose_interop.converters import read_pose_dataset
from annolid.annotation.pose_interop.model import PoseDataset, PoseInstance
from annolid.annotation.pose_schema import PoseSchema, find_pose_schema_in_dir
from annolid.postprocessing.spatial_calibration import (
    PlanarCalibration,
    resolve_planar_calibration,
)
from annolid.utils.atomic_io import atomic_write_json, atomic_write_text

__all__ = [
    "KeypointKinematicsRow",
    "BoneLengthRow",
    "JointAngleRow",
    "BodyHeadingRow",
    "DyadicKeypointRow",
    "PoseKinematicsReport",
    "infer_skeleton_joint_triplets",
    "analyze_pose_kinematics",
    "format_pose_kinematics_table",
    "write_pose_kinematics_bundle",
]


def _round_opt(val: Optional[float], ndigits: int = 4) -> Optional[float]:
    if val is None or not math.isfinite(float(val)):
        return None
    return round(float(val), ndigits)


@dataclass(frozen=True)
class KeypointKinematicsRow:
    """Calibrated speed, distance, and acceleration summary for one (instance, keypoint)."""

    instance: str
    keypoint: str
    observed_frames: int
    coverage_ratio: float
    total_distance_px: float
    total_distance_cm: Optional[float]
    mean_speed_px_s: float
    median_speed_px_s: float
    p95_speed_px_s: float
    max_speed_px_s: float
    mean_speed_cm_s: Optional[float]
    max_speed_cm_s: Optional[float]
    mean_accel_px_s2: float
    max_accel_px_s2: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "keypoint": self.keypoint,
            "observed_frames": int(self.observed_frames),
            "coverage_ratio": round(float(self.coverage_ratio), 6),
            "total_distance_px": round(float(self.total_distance_px), 4),
            "total_distance_cm": _round_opt(self.total_distance_cm),
            "mean_speed_px_s": round(float(self.mean_speed_px_s), 4),
            "median_speed_px_s": round(float(self.median_speed_px_s), 4),
            "p95_speed_px_s": round(float(self.p95_speed_px_s), 4),
            "max_speed_px_s": round(float(self.max_speed_px_s), 4),
            "mean_speed_cm_s": _round_opt(self.mean_speed_cm_s),
            "max_speed_cm_s": _round_opt(self.max_speed_cm_s),
            "mean_accel_px_s2": round(float(self.mean_accel_px_s2), 4),
            "max_accel_px_s2": round(float(self.max_accel_px_s2), 4),
        }


@dataclass(frozen=True)
class BoneLengthRow:
    """Calibrated skeleton edge/bone length distribution for one (instance, edge)."""

    instance: str
    kp_a: str
    kp_b: str
    count: int
    mean_px: float
    median_px: float
    std_px: float
    min_px: float
    max_px: float
    mean_cm: Optional[float]
    median_cm: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "kp_a": self.kp_a,
            "kp_b": self.kp_b,
            "count": int(self.count),
            "mean_px": round(float(self.mean_px), 4),
            "median_px": round(float(self.median_px), 4),
            "std_px": round(float(self.std_px), 4),
            "min_px": round(float(self.min_px), 4),
            "max_px": round(float(self.max_px), 4),
            "mean_cm": _round_opt(self.mean_cm),
            "median_cm": _round_opt(self.median_cm),
        }


@dataclass(frozen=True)
class JointAngleRow:
    """3-point joint angle distribution (``kp_a - vertex - kp_b``) in degrees."""

    instance: str
    kp_a: str
    vertex: str
    kp_b: str
    count: int
    mean_deg: float
    median_deg: float
    std_deg: float
    min_deg: float
    max_deg: float
    mean_angular_velocity_deg_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "joint": f"{self.kp_a}-{self.vertex}-{self.kp_b}",
            "kp_a": self.kp_a,
            "vertex": self.vertex,
            "kp_b": self.kp_b,
            "count": int(self.count),
            "mean_deg": round(float(self.mean_deg), 4),
            "median_deg": round(float(self.median_deg), 4),
            "std_deg": round(float(self.std_deg), 4),
            "min_deg": round(float(self.min_deg), 4),
            "max_deg": round(float(self.max_deg), 4),
            "mean_angular_velocity_deg_s": round(
                float(self.mean_angular_velocity_deg_s), 4
            ),
        }


@dataclass(frozen=True)
class BodyHeadingRow:
    """Body orientation heading and rotational speed summary for one instance."""

    instance: str
    posterior_kp: str
    anterior_kp: str
    observed_frames: int
    mean_heading_deg: float
    resultant_length: float
    mean_angular_velocity_deg_s: float
    max_angular_velocity_deg_s: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance": self.instance,
            "posterior_kp": self.posterior_kp,
            "anterior_kp": self.anterior_kp,
            "observed_frames": int(self.observed_frames),
            "mean_heading_deg": round(float(self.mean_heading_deg), 4),
            "resultant_length": round(float(self.resultant_length), 6),
            "mean_angular_velocity_deg_s": round(
                float(self.mean_angular_velocity_deg_s), 4
            ),
            "max_angular_velocity_deg_s": round(
                float(self.max_angular_velocity_deg_s), 4
            ),
        }


@dataclass(frozen=True)
class DyadicKeypointRow:
    """Inter-animal keypoint-pair proximity and directed bearing summary."""

    instance_a: str
    kp_a: str
    instance_b: str
    kp_b: str
    shared_frames: int
    min_distance_px: float
    mean_distance_px: float
    median_distance_px: float
    min_distance_cm: Optional[float]
    mean_distance_cm: Optional[float]
    close_contact_frames: int
    close_contact_seconds: float
    mean_bearing_a_to_b_deg: Optional[float]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "instance_a": self.instance_a,
            "kp_a": self.kp_a,
            "instance_b": self.instance_b,
            "kp_b": self.kp_b,
            "shared_frames": int(self.shared_frames),
            "min_distance_px": round(float(self.min_distance_px), 4),
            "mean_distance_px": round(float(self.mean_distance_px), 4),
            "median_distance_px": round(float(self.median_distance_px), 4),
            "min_distance_cm": _round_opt(self.min_distance_cm),
            "mean_distance_cm": _round_opt(self.mean_distance_cm),
            "close_contact_frames": int(self.close_contact_frames),
            "close_contact_seconds": round(float(self.close_contact_seconds), 4),
            "mean_bearing_a_to_b_deg": _round_opt(self.mean_bearing_a_to_b_deg),
        }


@dataclass
class PoseKinematicsReport:
    """Complete pose kinematics summary plus framewise angle and egocentric tables."""

    source_path: str
    fps: float
    px_per_cm: Optional[float]
    has_calibration: bool
    frame_min: Optional[int]
    frame_max: Optional[int]
    frames_analyzed: int
    keypoints: Tuple[str, ...]
    instances: Tuple[str, ...]
    keypoint_rows: Tuple[KeypointKinematicsRow, ...]
    bone_rows: Tuple[BoneLengthRow, ...]
    joint_angle_rows: Tuple[JointAngleRow, ...]
    heading_rows: Tuple[BodyHeadingRow, ...]
    dyadic_rows: Tuple[DyadicKeypointRow, ...]
    joint_angles_timeseries: List[Dict[str, Any]] = field(
        default_factory=list, repr=False
    )
    egocentric_rows: List[Dict[str, Any]] = field(default_factory=list, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_path": self.source_path,
            "fps": float(self.fps),
            "px_per_cm": _round_opt(self.px_per_cm),
            "has_calibration": bool(self.has_calibration),
            "frame_min": self.frame_min,
            "frame_max": self.frame_max,
            "frames_analyzed": int(self.frames_analyzed),
            "keypoints": list(self.keypoints),
            "instances": list(self.instances),
            "keypoint_kinematics": [r.to_dict() for r in self.keypoint_rows],
            "bone_lengths": [r.to_dict() for r in self.bone_rows],
            "joint_angles": [r.to_dict() for r in self.joint_angle_rows],
            "body_headings": [r.to_dict() for r in self.heading_rows],
            "dyadic_keypoints": [r.to_dict() for r in self.dyadic_rows],
            "egocentric_row_count": len(self.egocentric_rows),
        }


def infer_skeleton_joint_triplets(
    edges: Sequence[Tuple[str, str]],
    *,
    max_triplets: int = 24,
) -> List[Tuple[str, str, str]]:
    """Derive canonical 3-point joint triplets ``(a, vertex, b)`` from skeleton edges."""
    adj: Dict[str, List[str]] = defaultdict(list)
    for a, b in edges:
        if a == b:
            continue
        if b not in adj[a]:
            adj[a].append(b)
        if a not in adj[b]:
            adj[b].append(a)
    triplets: List[Tuple[str, str, str]] = []
    for vertex in sorted(adj.keys()):
        neighbors = adj[vertex]
        if len(neighbors) < 2:
            continue
        for n1, n2 in combinations(neighbors, 2):
            triplets.append((n1, vertex, n2))
            if len(triplets) >= max_triplets:
                return triplets
    return triplets


def _infer_heading_pair(keypoints: Sequence[str]) -> Optional[Tuple[str, str]]:
    """Select an anatomical ``(posterior, anterior)`` keypoint pair for body heading."""
    lower_map = {kp.lower(): kp for kp in keypoints}
    preferred_pairs = (
        ("tail_base", "nose"),
        ("tailbase", "nose"),
        ("root_tail", "nose"),
        ("centroid", "nose"),
        ("neck", "nose"),
        ("spine", "nose"),
        ("mid_back", "nose"),
        ("center_back", "nose"),
        ("abdomen", "head"),
        ("thorax", "head"),
        ("tail", "head"),
        ("tail_tip", "head"),
        ("spine_3", "head"),
        ("back", "nose"),
    )
    for post, ant in preferred_pairs:
        if post in lower_map and ant in lower_map:
            return (lower_map[post], lower_map[ant])
    if len(keypoints) >= 2:
        return (keypoints[-1], keypoints[0])
    return None


def _infer_egocentric_anchor(keypoints: Sequence[str]) -> Optional[str]:
    lower_map = {kp.lower(): kp for kp in keypoints}
    for cand in (
        "centroid",
        "neck",
        "thorax",
        "mid_back",
        "center_back",
        "spine",
        "spine_2",
        "back",
        "tail_base",
        "tailbase",
        "dorsal_fin",
    ):
        if cand in lower_map:
            return lower_map[cand]
    return keypoints[0] if keypoints else None


def _infer_dyadic_kp_pairs(keypoints: Sequence[str]) -> List[Tuple[str, str]]:
    lower_map = {kp.lower(): kp for kp in keypoints}
    pairs: List[Tuple[str, str]] = []
    candidates = (
        ("nose", "nose"),
        ("nose", "tail_base"),
        ("nose", "tailbase"),
        ("nose", "centroid"),
        ("head", "head"),
        ("head", "abdomen"),
        ("head", "tail"),
    )
    for a, b in candidates:
        if a in lower_map and b in lower_map:
            pair = (lower_map[a], lower_map[b])
            if pair not in pairs:
                pairs.append(pair)
    if not pairs and keypoints:
        pairs.append((keypoints[0], keypoints[0]))
    return pairs


def _angle_deg_3pt(
    ax: float, ay: float, vx: float, vy: float, bx: float, by: float
) -> Optional[float]:
    u_x, u_y = ax - vx, ay - vy
    v_x, v_y = bx - vx, by - vy
    nu = math.hypot(u_x, u_y)
    nv = math.hypot(v_x, v_y)
    if nu < 1e-6 or nv < 1e-6:
        return None
    cos_val = max(-1.0, min(1.0, (u_x * v_x + u_y * v_y) / (nu * nv)))
    return float(math.degrees(math.acos(cos_val)))


def _wrap_angle_deg(angle_deg: float) -> float:
    return float((angle_deg + 180.0) % 360.0 - 180.0)


def _to_cm_point(
    x: float,
    y: float,
    *,
    px_per_cm: Optional[float],
    planar_cal: Optional[PlanarCalibration],
) -> Optional[Tuple[float, float]]:
    if planar_cal is not None:
        return planar_cal.pixel_to_world(x, y)
    if px_per_cm is not None and px_per_cm > 0.0:
        return (x / float(px_per_cm), y / float(px_per_cm))
    return None


def analyze_pose_kinematics(
    source: str | Path | PoseDataset,
    *,
    source_format: str = "auto",
    pose_schema: Optional[PoseSchema | str | Path] = None,
    fps: float = 30.0,
    px_per_cm: Optional[float] = None,
    calibration: Optional[str | Path | PlanarCalibration] = None,
    min_confidence: Optional[float] = None,
    joint_triplets: Optional[Sequence[Tuple[str, str, str]]] = None,
    heading_pair: Optional[Tuple[str, str]] = None,
    egocentric_anchor: Optional[str] = None,
    pair_keypoints: Optional[Sequence[Tuple[str, str]]] = None,
    close_contact_threshold_px: float = 35.0,
    close_contact_threshold_cm: Optional[float] = None,
) -> PoseKinematicsReport:
    """Compute calibrated keypoint kinematics, bone lengths, joint angles, egocentric pose, and dyadic pairs."""
    eff_fps = max(1e-3, float(fps))
    if isinstance(source, PoseDataset):
        dataset = source
        src_str = "<PoseDataset>"
        src_path_obj: Optional[Path] = None
    else:
        src_path_obj = Path(source).expanduser().resolve()
        src_str = str(src_path_obj)
        dataset = read_pose_dataset(
            src_path_obj, source_format=source_format, pose_schema=pose_schema
        )

    if isinstance(pose_schema, PoseSchema):
        schema = pose_schema
    elif pose_schema is not None:
        schema = PoseSchema.load(pose_schema)
    elif src_path_obj is not None:
        found = find_pose_schema_in_dir(src_path_obj, search_parent=True)
        schema = (
            PoseSchema.load(found)
            if found is not None
            else dataset.to_pose_schema(preset="auto")
        )
    else:
        schema = dataset.to_pose_schema(preset="auto")

    planar_cal: Optional[PlanarCalibration] = None
    if isinstance(calibration, PlanarCalibration):
        planar_cal = calibration
    elif calibration is not None:
        planar_cal = resolve_planar_calibration(calibration)

    eff_ppc = (
        float(px_per_cm)
        if px_per_cm is not None
        and math.isfinite(float(px_per_cm))
        and float(px_per_cm) > 0.0
        else None
    )
    has_cal = planar_cal is not None or eff_ppc is not None

    keypoints = list(schema.keypoints or dataset.resolved_keypoints())
    edges = list(schema.base_edges())
    triplets = (
        list(joint_triplets)
        if joint_triplets is not None
        else infer_skeleton_joint_triplets(edges)
    )
    eff_heading_pair = heading_pair or _infer_heading_pair(keypoints)
    eff_anchor = egocentric_anchor or _infer_egocentric_anchor(keypoints)
    eff_dyadic_pairs = (
        list(pair_keypoints)
        if pair_keypoints is not None
        else _infer_dyadic_kp_pairs(keypoints)
    )

    sorted_frames = sorted(
        dataset.frames, key=lambda f: (int(f.video_index), int(f.frame_index))
    )
    if not sorted_frames:
        return PoseKinematicsReport(
            source_path=src_str,
            fps=eff_fps,
            px_per_cm=eff_ppc,
            has_calibration=has_cal,
            frame_min=None,
            frame_max=None,
            frames_analyzed=0,
            keypoints=tuple(keypoints),
            instances=(),
            keypoint_rows=(),
            bone_rows=(),
            joint_angle_rows=(),
            heading_rows=(),
            dyadic_rows=(),
        )

    frame_min = int(min(f.frame_index for f in sorted_frames))
    frame_max = int(max(f.frame_index for f in sorted_frames))
    total_span = max(1, frame_max - frame_min + 1)

    # Extract clean valid points per (frame, instance, keypoint).
    def _valid_pt(inst: PoseInstance, kp: str) -> Optional[Tuple[float, float]]:
        pt = inst.keypoints.get(kp)
        if pt is None or not pt.is_valid or int(pt.visibility) <= 0:
            return None
        if (
            min_confidence is not None
            and pt.score is not None
            and math.isfinite(float(pt.score))
            and float(pt.score) < float(min_confidence)
        ):
            return None
        return (float(pt.x), float(pt.y))

    inst_set: List[str] = []
    frame_inst_kps: Dict[int, Dict[str, Dict[str, Tuple[float, float]]]] = defaultdict(
        dict
    )
    kp_tracks: Dict[Tuple[str, str], List[Tuple[int, float, float]]] = defaultdict(list)

    for frame in sorted_frames:
        fidx = int(frame.frame_index)
        for i_pos, inst in enumerate(frame.instances):
            inst_name = (
                str(inst.instance_name or f"instance_{i_pos}").strip()
                or f"instance_{i_pos}"
            )
            if inst_name not in inst_set:
                inst_set.append(inst_name)
            kp_dict: Dict[str, Tuple[float, float]] = {}
            for kp in keypoints:
                xy = _valid_pt(inst, kp)
                if xy is not None:
                    kp_dict[kp] = xy
                    kp_tracks[(inst_name, kp)].append((fidx, xy[0], xy[1]))
            # Include any extra keypoints present on the instance.
            for kp in inst.keypoints:
                if kp not in kp_dict:
                    xy = _valid_pt(inst, kp)
                    if xy is not None:
                        kp_dict[kp] = xy
                        kp_tracks[(inst_name, kp)].append((fidx, xy[0], xy[1]))
            if kp_dict:
                frame_inst_kps[fidx][inst_name] = kp_dict

    # 1. Per-keypoint calibrated speeds & accelerations.
    keypoint_rows: List[KeypointKinematicsRow] = []
    for inst_name in inst_set:
        for kp in keypoints:
            pts = sorted(kp_tracks.get((inst_name, kp), []), key=lambda t: t[0])
            if not pts:
                continue
            dist_px = 0.0
            dist_cm = 0.0 if has_cal else None
            speeds_px: List[float] = []
            speeds_cm: List[float] = []
            step_times: List[float] = []

            for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
                dt_frames = f1 - f0
                if dt_frames <= 0:
                    continue
                dt_s = float(dt_frames) / eff_fps
                d_px = math.hypot(x1 - x0, y1 - y0)
                dist_px += d_px
                speeds_px.append(d_px / dt_s)
                step_times.append(dt_s)
                if has_cal:
                    p0_cm = _to_cm_point(
                        x0, y0, px_per_cm=eff_ppc, planar_cal=planar_cal
                    )
                    p1_cm = _to_cm_point(
                        x1, y1, px_per_cm=eff_ppc, planar_cal=planar_cal
                    )
                    if p0_cm is not None and p1_cm is not None:
                        d_cm = math.hypot(p1_cm[0] - p0_cm[0], p1_cm[1] - p0_cm[1])
                        dist_cm = float(dist_cm or 0.0) + d_cm
                        speeds_cm.append(d_cm / dt_s)

            accels_px: List[float] = []
            for i in range(1, len(speeds_px)):
                dt_mid = max(1e-6, 0.5 * (step_times[i - 1] + step_times[i]))
                accels_px.append(abs(speeds_px[i] - speeds_px[i - 1]) / dt_mid)

            sp_arr = (
                np.asarray(speeds_px, dtype=np.float64) if speeds_px else np.zeros(1)
            )
            ac_arr = (
                np.asarray(accels_px, dtype=np.float64) if accels_px else np.zeros(1)
            )
            keypoint_rows.append(
                KeypointKinematicsRow(
                    instance=inst_name,
                    keypoint=kp,
                    observed_frames=len(pts),
                    coverage_ratio=float(len(pts) / float(total_span)),
                    total_distance_px=dist_px,
                    total_distance_cm=dist_cm,
                    mean_speed_px_s=float(np.mean(sp_arr)) if speeds_px else 0.0,
                    median_speed_px_s=float(np.median(sp_arr)) if speeds_px else 0.0,
                    p95_speed_px_s=float(np.percentile(sp_arr, 95))
                    if speeds_px
                    else 0.0,
                    max_speed_px_s=float(np.max(sp_arr)) if speeds_px else 0.0,
                    mean_speed_cm_s=float(np.mean(speeds_cm)) if speeds_cm else None,
                    max_speed_cm_s=float(np.max(speeds_cm)) if speeds_cm else None,
                    mean_accel_px_s2=float(np.mean(ac_arr)) if accels_px else 0.0,
                    max_accel_px_s2=float(np.max(ac_arr)) if accels_px else 0.0,
                )
            )

    # 2. Per-bone calibrated lengths.
    bone_rows: List[BoneLengthRow] = []
    for inst_name in inst_set:
        for kp_a, kp_b in edges:
            vals_px: List[float] = []
            vals_cm: List[float] = []
            for fidx in sorted(frame_inst_kps):
                kmap = frame_inst_kps[fidx].get(inst_name)
                if not kmap or kp_a not in kmap or kp_b not in kmap:
                    continue
                ax, ay = kmap[kp_a]
                bx, by = kmap[kp_b]
                vals_px.append(math.hypot(bx - ax, by - ay))
                if has_cal:
                    pa_cm = _to_cm_point(
                        ax, ay, px_per_cm=eff_ppc, planar_cal=planar_cal
                    )
                    pb_cm = _to_cm_point(
                        bx, by, px_per_cm=eff_ppc, planar_cal=planar_cal
                    )
                    if pa_cm is not None and pb_cm is not None:
                        vals_cm.append(
                            math.hypot(pb_cm[0] - pa_cm[0], pb_cm[1] - pa_cm[1])
                        )
            if not vals_px:
                continue
            arr = np.asarray(vals_px, dtype=np.float64)
            bone_rows.append(
                BoneLengthRow(
                    instance=inst_name,
                    kp_a=kp_a,
                    kp_b=kp_b,
                    count=len(vals_px),
                    mean_px=float(np.mean(arr)),
                    median_px=float(np.median(arr)),
                    std_px=float(np.std(arr)),
                    min_px=float(np.min(arr)),
                    max_px=float(np.max(arr)),
                    mean_cm=float(np.mean(vals_cm)) if vals_cm else None,
                    median_cm=float(np.median(vals_cm)) if vals_cm else None,
                )
            )

    # 3. 3-point joint angles & framewise time series.
    joint_angle_rows: List[JointAngleRow] = []
    joint_ts: List[Dict[str, Any]] = []
    for inst_name in inst_set:
        for kp_a, vertex, kp_b in triplets:
            series: List[Tuple[int, float]] = []
            for fidx in sorted(frame_inst_kps):
                kmap = frame_inst_kps[fidx].get(inst_name)
                if (
                    not kmap
                    or kp_a not in kmap
                    or vertex not in kmap
                    or kp_b not in kmap
                ):
                    continue
                ang = _angle_deg_3pt(
                    kmap[kp_a][0],
                    kmap[kp_a][1],
                    kmap[vertex][0],
                    kmap[vertex][1],
                    kmap[kp_b][0],
                    kmap[kp_b][1],
                )
                if ang is not None:
                    series.append((fidx, ang))
                    joint_ts.append(
                        {
                            "frame_index": fidx,
                            "time_s": round(fidx / eff_fps, 6),
                            "instance": inst_name,
                            "joint": f"{kp_a}-{vertex}-{kp_b}",
                            "angle_deg": round(ang, 4),
                        }
                    )
            if not series:
                continue
            vals = np.asarray([a for _, a in series], dtype=np.float64)
            ang_vels = [
                abs(a1 - a0) / (float(f1 - f0) / eff_fps)
                for (f0, a0), (f1, a1) in zip(series, series[1:])
                if f1 > f0
            ]
            joint_angle_rows.append(
                JointAngleRow(
                    instance=inst_name,
                    kp_a=kp_a,
                    vertex=vertex,
                    kp_b=kp_b,
                    count=len(series),
                    mean_deg=float(np.mean(vals)),
                    median_deg=float(np.median(vals)),
                    std_deg=float(np.std(vals)),
                    min_deg=float(np.min(vals)),
                    max_deg=float(np.max(vals)),
                    mean_angular_velocity_deg_s=(
                        float(np.mean(ang_vels)) if ang_vels else 0.0
                    ),
                )
            )

    # 4. Body heading & egocentric body-aligned coordinates.
    heading_rows: List[BodyHeadingRow] = []
    heading_by_frame_inst: Dict[Tuple[int, str], float] = {}
    egocentric_rows: List[Dict[str, Any]] = []

    for inst_name in inst_set:
        h_series: List[Tuple[int, float]] = []
        if eff_heading_pair is not None:
            post_kp, ant_kp = eff_heading_pair
            for fidx in sorted(frame_inst_kps):
                kmap = frame_inst_kps[fidx].get(inst_name)
                if not kmap or post_kp not in kmap or ant_kp not in kmap:
                    continue
                dx = kmap[ant_kp][0] - kmap[post_kp][0]
                dy = kmap[ant_kp][1] - kmap[post_kp][1]
                if math.hypot(dx, dy) < 1e-6:
                    continue
                theta_rad = math.atan2(dy, dx)
                heading_by_frame_inst[(fidx, inst_name)] = theta_rad
                h_series.append((fidx, theta_rad))

                # Compute egocentric body-aligned coordinates for all keypoints on this frame.
                if eff_anchor and eff_anchor in kmap:
                    ox, oy = kmap[eff_anchor]
                else:
                    ox = sum(p[0] for p in kmap.values()) / float(len(kmap))
                    oy = sum(p[1] for p in kmap.values()) / float(len(kmap))
                cos_t = math.cos(-theta_rad)
                sin_t = math.sin(-theta_rad)
                ego_rec: Dict[str, Any] = {
                    "frame_index": fidx,
                    "time_s": round(fidx / eff_fps, 6),
                    "instance": inst_name,
                    "heading_deg": round(math.degrees(theta_rad), 4),
                    "anchor_x": round(ox, 4),
                    "anchor_y": round(oy, 4),
                }
                for kp in keypoints:
                    if kp in kmap:
                        rx = kmap[kp][0] - ox
                        ry = kmap[kp][1] - oy
                        ex = rx * cos_t - ry * sin_t
                        ey = rx * sin_t + ry * cos_t
                        ego_rec[f"{kp}_ego_x"] = round(ex, 4)
                        ego_rec[f"{kp}_ego_y"] = round(ey, 4)
                    else:
                        ego_rec[f"{kp}_ego_x"] = ""
                        ego_rec[f"{kp}_ego_y"] = ""
                egocentric_rows.append(ego_rec)

            if h_series:
                angles_rad = np.asarray([a for _, a in h_series], dtype=np.float64)
                mean_cos = float(np.mean(np.cos(angles_rad)))
                mean_sin = float(np.mean(np.sin(angles_rad)))
                mean_h_deg = math.degrees(math.atan2(mean_sin, mean_cos))
                resultant = math.hypot(mean_cos, mean_sin)
                ang_vels = [
                    abs(_wrap_angle_deg(math.degrees(a1 - a0)))
                    / (float(f1 - f0) / eff_fps)
                    for (f0, a0), (f1, a1) in zip(h_series, h_series[1:])
                    if f1 > f0
                ]
                heading_rows.append(
                    BodyHeadingRow(
                        instance=inst_name,
                        posterior_kp=post_kp,
                        anterior_kp=ant_kp,
                        observed_frames=len(h_series),
                        mean_heading_deg=mean_h_deg,
                        resultant_length=resultant,
                        mean_angular_velocity_deg_s=(
                            float(np.mean(ang_vels)) if ang_vels else 0.0
                        ),
                        max_angular_velocity_deg_s=(
                            float(np.max(ang_vels)) if ang_vels else 0.0
                        ),
                    )
                )

    # 5. Inter-animal dyadic keypoint-pair distances & directed bearings.
    dyadic_rows: List[DyadicKeypointRow] = []
    if len(inst_set) >= 2 and eff_dyadic_pairs:
        for inst_a, inst_b in combinations(inst_set, 2):
            for kp_a, kp_b in eff_dyadic_pairs:
                dists_px: List[float] = []
                dists_cm: List[float] = []
                bearings: List[float] = []
                close_count = 0
                for fidx in sorted(frame_inst_kps):
                    ka = frame_inst_kps[fidx].get(inst_a)
                    kb = frame_inst_kps[fidx].get(inst_b)
                    if not ka or not kb or kp_a not in ka or kp_b not in kb:
                        continue
                    ax, ay = ka[kp_a]
                    bx, by = kb[kp_b]
                    d_px = math.hypot(bx - ax, by - ay)
                    dists_px.append(d_px)
                    d_cm: Optional[float] = None
                    if has_cal:
                        pa_cm = _to_cm_point(
                            ax, ay, px_per_cm=eff_ppc, planar_cal=planar_cal
                        )
                        pb_cm = _to_cm_point(
                            bx, by, px_per_cm=eff_ppc, planar_cal=planar_cal
                        )
                        if pa_cm is not None and pb_cm is not None:
                            d_cm = math.hypot(pb_cm[0] - pa_cm[0], pb_cm[1] - pa_cm[1])
                            dists_cm.append(d_cm)
                    if close_contact_threshold_cm is not None and d_cm is not None:
                        if d_cm <= float(close_contact_threshold_cm):
                            close_count += 1
                    elif d_px <= float(close_contact_threshold_px):
                        close_count += 1

                    h_a = heading_by_frame_inst.get((fidx, inst_a))
                    if h_a is not None and d_px > 1e-6:
                        vec_ang = math.atan2(by - ay, bx - ax)
                        bearings.append(
                            abs(_wrap_angle_deg(math.degrees(vec_ang - h_a)))
                        )

                if not dists_px:
                    continue
                arr = np.asarray(dists_px, dtype=np.float64)
                dyadic_rows.append(
                    DyadicKeypointRow(
                        instance_a=inst_a,
                        kp_a=kp_a,
                        instance_b=inst_b,
                        kp_b=kp_b,
                        shared_frames=len(dists_px),
                        min_distance_px=float(np.min(arr)),
                        mean_distance_px=float(np.mean(arr)),
                        median_distance_px=float(np.median(arr)),
                        min_distance_cm=float(np.min(dists_cm)) if dists_cm else None,
                        mean_distance_cm=float(np.mean(dists_cm)) if dists_cm else None,
                        close_contact_frames=close_count,
                        close_contact_seconds=float(close_count) / eff_fps,
                        mean_bearing_a_to_b_deg=(
                            float(np.mean(bearings)) if bearings else None
                        ),
                    )
                )

    return PoseKinematicsReport(
        source_path=src_str,
        fps=eff_fps,
        px_per_cm=eff_ppc,
        has_calibration=has_cal,
        frame_min=frame_min,
        frame_max=frame_max,
        frames_analyzed=len(sorted_frames),
        keypoints=tuple(keypoints),
        instances=tuple(inst_set),
        keypoint_rows=tuple(keypoint_rows),
        bone_rows=tuple(bone_rows),
        joint_angle_rows=tuple(joint_angle_rows),
        heading_rows=tuple(heading_rows),
        dyadic_rows=tuple(dyadic_rows),
        joint_angles_timeseries=joint_ts,
        egocentric_rows=egocentric_rows,
    )


def format_pose_kinematics_table(report: PoseKinematicsReport) -> str:
    """Format a compact CLI summary table for ``PoseKinematicsReport``."""
    lines = [
        "=== Pose Kinematics Summary ===",
        f"Frames analyzed: {report.frames_analyzed} (range {report.frame_min}..{report.frame_max}, fps={report.fps:.2f})",
        f"Instances      : {', '.join(report.instances) if report.instances else 'none'}",
    ]
    if report.keypoint_rows:
        lines.append("")
        lines.append(
            f"{'Instance':<14} {'Keypoint':<16} {'Frames':>6} {'MeanSpd(px/s)':>14} {'Dist(px)':>10}"
        )
        lines.append("-" * 64)
        for r in report.keypoint_rows[:30]:
            lines.append(
                f"{r.instance:<14} {r.keypoint:<16} {r.observed_frames:>6} "
                f"{r.mean_speed_px_s:>14.2f} {r.total_distance_px:>10.1f}"
            )
    if report.bone_rows:
        lines.append("")
        lines.append(f"{'Instance':<14} {'Bone':<24} {'Median(px)':>10} {'Std(px)':>9}")
        lines.append("-" * 60)
        for b in report.bone_rows[:20]:
            lines.append(
                f"{b.instance:<14} {f'{b.kp_a}-{b.kp_b}':<24} {b.median_px:>10.2f} {b.std_px:>9.2f}"
            )
    if report.joint_angle_rows:
        lines.append("")
        lines.append(
            f"{'Instance':<14} {'Joint':<28} {'Mean(deg)':>10} {'AngVel(deg/s)':>14}"
        )
        lines.append("-" * 68)
        for j in report.joint_angle_rows[:20]:
            lines.append(
                f"{j.instance:<14} {f'{j.kp_a}-{j.vertex}-{j.kp_b}':<28} "
                f"{j.mean_deg:>10.1f} {j.mean_angular_velocity_deg_s:>14.1f}"
            )
    return "\n".join(lines)


def _write_csv_dicts(
    path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]
) -> str:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(fieldnames), lineterminator="\n")
    writer.writeheader()
    for r in rows:
        writer.writerow({k: r.get(k, "") for k in fieldnames})
    atomic_write_text(path, buf.getvalue(), encoding="utf-8")
    return str(path)


def write_pose_kinematics_bundle(
    report: PoseKinematicsReport,
    output_dir: str | Path,
) -> Dict[str, str]:
    """Write JSON summary and CSV tables (keypoints, bones, angles, dyadic, egocentric)."""
    out_root = Path(output_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    written: Dict[str, str] = {}

    summary_path = out_root / "pose_kinematics_summary.json"
    atomic_write_json(summary_path, report.to_dict(), indent=2)
    written["summary_json"] = str(summary_path)

    written["keypoint_kinematics_csv"] = _write_csv_dicts(
        out_root / "keypoint_kinematics.csv",
        [r.to_dict() for r in report.keypoint_rows],
        [
            "instance",
            "keypoint",
            "observed_frames",
            "coverage_ratio",
            "total_distance_px",
            "total_distance_cm",
            "mean_speed_px_s",
            "median_speed_px_s",
            "p95_speed_px_s",
            "max_speed_px_s",
            "mean_speed_cm_s",
            "max_speed_cm_s",
            "mean_accel_px_s2",
            "max_accel_px_s2",
        ],
    )
    written["bone_lengths_csv"] = _write_csv_dicts(
        out_root / "bone_lengths.csv",
        [r.to_dict() for r in report.bone_rows],
        [
            "instance",
            "kp_a",
            "kp_b",
            "count",
            "mean_px",
            "median_px",
            "std_px",
            "min_px",
            "max_px",
            "mean_cm",
            "median_cm",
        ],
    )
    written["joint_angles_csv"] = _write_csv_dicts(
        out_root / "joint_angles.csv",
        [r.to_dict() for r in report.joint_angle_rows],
        [
            "instance",
            "joint",
            "kp_a",
            "vertex",
            "kp_b",
            "count",
            "mean_deg",
            "median_deg",
            "std_deg",
            "min_deg",
            "max_deg",
            "mean_angular_velocity_deg_s",
        ],
    )
    if report.joint_angles_timeseries:
        written["joint_angles_timeseries_csv"] = _write_csv_dicts(
            out_root / "joint_angles_timeseries.csv",
            report.joint_angles_timeseries,
            ["frame_index", "time_s", "instance", "joint", "angle_deg"],
        )
    if report.dyadic_rows:
        written["dyadic_keypoints_csv"] = _write_csv_dicts(
            out_root / "dyadic_keypoints.csv",
            [r.to_dict() for r in report.dyadic_rows],
            [
                "instance_a",
                "kp_a",
                "instance_b",
                "kp_b",
                "shared_frames",
                "min_distance_px",
                "mean_distance_px",
                "median_distance_px",
                "min_distance_cm",
                "mean_distance_cm",
                "close_contact_frames",
                "close_contact_seconds",
                "mean_bearing_a_to_b_deg",
            ],
        )
    if report.egocentric_rows:
        ego_cols = list(report.egocentric_rows[0].keys())
        written["egocentric_pose_csv"] = _write_csv_dicts(
            out_root / "egocentric_pose.csv",
            report.egocentric_rows,
            ego_cols,
        )
    return written
