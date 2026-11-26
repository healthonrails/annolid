from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
from PIL import Image

from annolid.simulation.types import Pose2DFrame, Pose3DFrame


@dataclass(frozen=True)
class CameraCalibration:
    """Pinhole camera intrinsics, world-to-camera extrinsics, and radial/tangential distortion.

    Projection convention:
        X_cam = R @ X_world + t
        u = fx * (X_cam[0] / X_cam[2]) + cx
        v = fy * (X_cam[1] / X_cam[2]) + cy
    """

    name: str
    fx: float
    fy: float
    cx: float
    cy: float
    rotation: np.ndarray  # (3, 3) world-to-camera rotation R
    translation: np.ndarray  # (3,) world-to-camera translation t
    distortion: Tuple[float, ...] = ()  # (k1, k2, p1, p2[, k3])
    image_size: Tuple[int, int] | None = None  # (width, height)

    def __post_init__(self) -> None:
        if float(self.fx) <= 0.0 or float(self.fy) <= 0.0:
            raise ValueError("Camera focal lengths fx and fy must be positive.")
        rot = np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)
        trans = np.asarray(self.translation, dtype=np.float64).reshape(3)
        object.__setattr__(self, "rotation", rot)
        object.__setattr__(self, "translation", trans)
        object.__setattr__(
            self, "distortion", tuple(float(v) for v in (self.distortion or ()))
        )

    @property
    def intrinsic_matrix(self) -> np.ndarray:
        return np.array(
            [
                [float(self.fx), 0.0, float(self.cx)],
                [0.0, float(self.fy), float(self.cy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    def projection_matrix(self) -> np.ndarray:
        """Return 3x4 pinhole projection matrix P = K [R | t]."""
        rt = np.hstack([self.rotation, self.translation.reshape(3, 1)])
        return self.intrinsic_matrix @ rt

    @property
    def camera_center_world(self) -> np.ndarray:
        """World coordinates of camera optical center C = -R^T t."""
        return -(self.rotation.T @ self.translation)

    def project(
        self,
        point_3d: Sequence[float] | np.ndarray,
        *,
        apply_distortion: bool = True,
    ) -> Tuple[float, float, float]:
        """Project a 3D world point to `(u_px, v_px, z_cam)`."""
        pt = np.asarray(point_3d, dtype=np.float64).reshape(3)
        cam = self.rotation @ pt + self.translation
        z = float(cam[2])
        z_safe = z if abs(z) > 1e-9 else (1e-9 if z >= 0.0 else -1e-9)
        x = float(cam[0]) / z_safe
        y = float(cam[1]) / z_safe
        if apply_distortion and self.distortion:
            x, y = _distort_normalized(x, y, self.distortion)
        u = float(self.fx) * x + float(self.cx)
        v = float(self.fy) * y + float(self.cy)
        return u, v, z

    def undistort_point(
        self, u_px: float, v_px: float, *, iterations: int = 8
    ) -> Tuple[float, float]:
        """Invert Brown-Conrady distortion back to undistorted pixel coordinates."""
        if not self.distortion:
            return float(u_px), float(v_px)
        xd = (float(u_px) - float(self.cx)) / float(self.fx)
        yd = (float(v_px) - float(self.cy)) / float(self.fy)
        x, y = xd, yd
        for _ in range(max(1, int(iterations))):
            fx_d, fy_d = _distort_normalized(x, y, self.distortion)
            err_x = fx_d - xd
            err_y = fy_d - yd
            if err_x * err_x + err_y * err_y < 1e-18:
                break
            x -= err_x
            y -= err_y
        return float(self.fx) * x + float(self.cx), float(self.fy) * y + float(self.cy)

    @classmethod
    def from_look_at(
        cls,
        *,
        name: str,
        position: Sequence[float],
        target: Sequence[float],
        up: Sequence[float] = (0.0, 0.0, 1.0),
        fx: float = 800.0,
        fy: float = 800.0,
        cx: float = 320.0,
        cy: float = 240.0,
        distortion: Sequence[float] = (),
        image_size: Tuple[int, int] | None = None,
    ) -> CameraCalibration:
        """Construct a right-handed OpenCV-convention (+Z forward, +X right, +Y down) camera."""
        pos = np.asarray(position, dtype=np.float64).reshape(3)
        tgt = np.asarray(target, dtype=np.float64).reshape(3)
        up_vec = np.asarray(up, dtype=np.float64).reshape(3)

        z_axis = tgt - pos
        z_norm = float(np.linalg.norm(z_axis))
        if z_norm < 1e-9:
            raise ValueError("Camera position and target must be distinct.")
        z_axis = z_axis / z_norm

        x_axis = np.cross(z_axis, up_vec)
        x_norm = float(np.linalg.norm(x_axis))
        if x_norm < 1e-9:
            # Handle top-down camera aligned with up axis
            alt_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            if abs(float(np.dot(z_axis, alt_up))) > 0.99:
                alt_up = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            x_axis = np.cross(z_axis, alt_up)
            x_norm = float(np.linalg.norm(x_axis))
        x_axis = x_axis / x_norm
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / max(float(np.linalg.norm(y_axis)), 1e-12)

        rot = np.vstack([x_axis, y_axis, z_axis])
        trans = -(rot @ pos)
        return cls(
            name=name,
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            rotation=rot,
            translation=trans,
            distortion=tuple(float(v) for v in distortion),
            image_size=image_size,
        )

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, Any], *, default_name: str = "cam0"
    ) -> CameraCalibration:
        intr = dict(payload.get("intrinsics") or payload)
        extr = dict(payload.get("extrinsics") or payload)
        rot = extr.get("rotation")
        if rot is None:
            rot = extr.get("R")
        if rot is None:
            rot = np.eye(3, dtype=np.float64)
        trans = extr.get("translation")
        if trans is None:
            trans = extr.get("t")
        if trans is None:
            trans = np.zeros(3, dtype=np.float64)
        img_sz = payload.get("image_size")
        parsed_sz = (
            (int(img_sz[0]), int(img_sz[1]))
            if isinstance(img_sz, (list, tuple)) and len(img_sz) >= 2
            else None
        )
        return cls(
            name=str(payload.get("name") or default_name),
            fx=float(intr["fx"]),
            fy=float(intr["fy"]),
            cx=float(intr.get("cx", 0.0)),
            cy=float(intr.get("cy", 0.0)),
            rotation=np.asarray(rot, dtype=np.float64).reshape(3, 3),
            translation=np.asarray(trans, dtype=np.float64).reshape(3),
            distortion=tuple(float(v) for v in (payload.get("distortion") or ())),
            image_size=parsed_sz,
        )


def _distort_normalized(
    x: float, y: float, distortion: Sequence[float]
) -> Tuple[float, float]:
    k1 = float(distortion[0]) if len(distortion) > 0 else 0.0
    k2 = float(distortion[1]) if len(distortion) > 1 else 0.0
    p1 = float(distortion[2]) if len(distortion) > 2 else 0.0
    p2 = float(distortion[3]) if len(distortion) > 3 else 0.0
    k3 = float(distortion[4]) if len(distortion) > 4 else 0.0
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2
    radial = 1.0 + k1 * r2 + k2 * r4 + k3 * r6
    x_d = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    y_d = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    return x_d, y_d


def load_depth_records(path: str | Path) -> Dict[int, Dict[str, Any]]:
    depth_path = Path(path).expanduser()
    records: Dict[int, Dict[str, Any]] = {}
    for line in depth_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        frame_index = int(payload.get("frame_index") or 0)
        records[frame_index] = payload
    return records


def lift_pose_frames_with_depth(
    pose_frames: Sequence[Pose2DFrame],
    *,
    depth_records: Mapping[int, Dict[str, Any]],
    coordinate_system: Mapping[str, Any] | None = None,
    sample_mode: str = "nearest",
    patch_radius: int = 1,
) -> list[Pose3DFrame]:
    """Lift 2D pose frames into 3D using depth sidecars, camera intrinsics, and optional extrinsics.

    Args:
        pose_frames: Input 2D pose observations.
        depth_records: Mapping of `frame_index -> depth record dict`.
        coordinate_system: Optional dict containing `camera_intrinsics` (`fx, fy, cx, cy`),
            optional `camera_extrinsics` (`R`/`rotation`, `t`/`translation` or `camera_to_world`),
            and optional `depth_sample_mode` (`nearest`, `bilinear`, `median_patch`).
        sample_mode: Depth sampling strategy (`nearest`, `bilinear`, `median_patch`).
        patch_radius: Half-width in pixels when `sample_mode == "median_patch"`.
    """
    coordinate_system = dict(coordinate_system or {})
    intrinsics = dict(coordinate_system.get("camera_intrinsics") or {})
    extrinsics = dict(coordinate_system.get("camera_extrinsics") or {})
    resolved_mode = (
        str(coordinate_system.get("depth_sample_mode") or sample_mode or "nearest")
        .strip()
        .lower()
    )
    resolved_radius = int(coordinate_system.get("depth_patch_radius", patch_radius))

    lifted: list[Pose3DFrame] = []
    for frame in pose_frames:
        depth_record = depth_records.get(frame.frame_index)
        if depth_record is None:
            raise KeyError(f"Missing depth record for frame_index={frame.frame_index}")
        depth_map = _decode_depth_map(depth_record)
        points_3d: Dict[str, Tuple[float, float, float]] = {}
        for label, point in frame.points.items():
            x_px, y_px = point
            z_val = _sample_depth(
                depth_map,
                x_px=x_px,
                y_px=y_px,
                mode=resolved_mode,
                patch_radius=resolved_radius,
            )
            cam_pt = _pixel_to_3d(
                x_px=x_px,
                y_px=y_px,
                z_val=z_val,
                intrinsics=intrinsics,
            )
            points_3d[label] = _apply_camera_extrinsics(cam_pt, extrinsics)
        lifted.append(
            Pose3DFrame(
                frame_index=frame.frame_index,
                video_name=frame.video_name,
                timestamp_sec=frame.timestamp_sec,
                points=points_3d,
                scores=dict(frame.scores),
                source_record=dict(frame.source_record),
            )
        )
    return lifted


def _solve_dlt_weighted(
    proj_matrices: Sequence[np.ndarray],
    pixels: Sequence[Tuple[float, float]],
    weights: Sequence[float],
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for proj, (u, v), w in zip(proj_matrices, pixels, weights):
        sw = math.sqrt(max(float(w), 1e-6))
        rows.append(sw * (u * proj[2, :] - proj[0, :]))
        rows.append(sw * (v * proj[2, :] - proj[1, :]))
    design = np.asarray(rows, dtype=np.float64)
    _, _, vh = np.linalg.svd(design, full_matrices=False)
    homog = vh[-1, :]
    if abs(float(homog[3])) < 1e-12:
        return homog[:3].copy()
    return (homog[:3] / homog[3]).astype(np.float64)


def _refine_point_gauss_newton(
    x_init: np.ndarray,
    calibs: Sequence[CameraCalibration],
    pixels: Sequence[Tuple[float, float]],
    weights: Sequence[float],
    *,
    max_iters: int = 10,
    huber_delta_px: float = 4.0,
) -> np.ndarray:
    x = np.asarray(x_init, dtype=np.float64).copy()
    lam = 1e-4
    for _ in range(max(1, int(max_iters))):
        jtj = np.zeros((3, 3), dtype=np.float64)
        jtr = np.zeros(3, dtype=np.float64)
        for calib, (u_obs, v_obs), w in zip(calibs, pixels, weights):
            cam = calib.rotation @ x + calib.translation
            z = float(cam[2])
            if abs(z) < 1e-6:
                continue
            inv_z = 1.0 / z
            inv_z2 = inv_z * inv_z
            u_pred = calib.fx * float(cam[0]) * inv_z + calib.cx
            v_pred = calib.fy * float(cam[1]) * inv_z + calib.cy
            ru = u_pred - u_obs
            rv = v_pred - v_obs
            err = math.hypot(ru, rv)
            huber = 1.0 if err <= huber_delta_px else (huber_delta_px / max(err, 1e-9))
            eff_w = max(float(w), 1e-6) * huber

            r0, r1, r2 = calib.rotation[0], calib.rotation[1], calib.rotation[2]
            ju = calib.fx * (r0 * inv_z - float(cam[0]) * inv_z2 * r2)
            jv = calib.fy * (r1 * inv_z - float(cam[1]) * inv_z2 * r2)

            jtj += eff_w * (np.outer(ju, ju) + np.outer(jv, jv))
            jtr += eff_w * (ju * ru + jv * rv)
        jtj_damped = jtj + lam * np.eye(3, dtype=np.float64)
        try:
            delta = -np.linalg.solve(jtj_damped, jtr)
        except np.linalg.LinAlgError:
            break
        x = x + delta
        if float(np.linalg.norm(delta)) < 1e-8:
            break
    return x


def _reprojection_errors(
    point_3d: np.ndarray,
    calibs: Sequence[CameraCalibration],
    pixels: Sequence[Tuple[float, float]],
) -> list[float]:
    errs: list[float] = []
    for calib, (u_obs, v_obs) in zip(calibs, pixels):
        u_pred, v_pred, _ = calib.project(point_3d, apply_distortion=False)
        errs.append(float(math.hypot(u_pred - u_obs, v_pred - v_obs)))
    return errs


def triangulate_point_multiview(
    observations: Sequence[Tuple[CameraCalibration, Tuple[float, float], float]],
    *,
    min_score: float = 0.0,
    refine_gauss_newton: bool = True,
    max_gn_iters: int = 10,
    outlier_reproj_thresh_px: float | None = 12.0,
) -> Tuple[Tuple[float, float, float], Dict[str, Any]]:
    """Triangulate a single 3D world point from >=2 calibrated camera observations.

    Uses confidence-weighted DLT initialization, optional leave-one-out outlier view
    rejection when >=3 views are available, and Huber-weighted Gauss-Newton refinement.
    """
    filtered = [
        (calib, calib.undistort_point(float(uv[0]), float(uv[1])), float(score))
        for calib, uv, score in observations
        if float(score) >= float(min_score)
    ]
    if len(filtered) < 2:
        raise ValueError(
            f"At least 2 camera views with score >= {min_score} are required for triangulation; got {len(filtered)}."
        )

    calibs = [item[0] for item in filtered]
    pixels = [item[1] for item in filtered]
    scores = [max(item[2], 1e-3) for item in filtered]
    proj_mats = [c.projection_matrix() for c in calibs]

    def _fit_subset(indices: Sequence[int]) -> Tuple[np.ndarray, list[float]]:
        sub_c = [calibs[i] for i in indices]
        sub_p = [pixels[i] for i in indices]
        sub_w = [scores[i] for i in indices]
        sub_m = [proj_mats[i] for i in indices]
        pt = _solve_dlt_weighted(sub_m, sub_p, sub_w)
        if refine_gauss_newton:
            pt = _refine_point_gauss_newton(
                pt, sub_c, sub_p, sub_w, max_iters=max_gn_iters
            )
        return pt, _reprojection_errors(pt, sub_c, sub_p)

    active_indices = list(range(len(filtered)))
    rejected_views: list[str] = []
    pt, errs = _fit_subset(active_indices)

    # Iteratively prune single-view outliers when >=3 cameras remain
    if outlier_reproj_thresh_px is not None and outlier_reproj_thresh_px > 0.0:
        while len(active_indices) >= 3 and max(errs) > float(outlier_reproj_thresh_px):
            best_drop: int | None = None
            best_max_err = max(errs)
            best_fit: Tuple[np.ndarray, list[float]] | None = None
            for pos in range(len(active_indices)):
                cand_idx = [idx for j, idx in enumerate(active_indices) if j != pos]
                cand_pt, cand_errs = _fit_subset(cand_idx)
                cand_max = max(cand_errs)
                if cand_max < best_max_err:
                    best_max_err = cand_max
                    best_drop = pos
                    best_fit = (cand_pt, cand_errs)
            if best_drop is None or best_fit is None:
                break
            dropped_global = active_indices.pop(best_drop)
            rejected_views.append(calibs[dropped_global].name)
            pt, errs = best_fit

    rmse_px = float(math.sqrt(float(np.mean(np.square(errs))))) if errs else 0.0
    diag = {
        "reprojection_rmse_px": round(rmse_px, 4),
        "max_reprojection_px": round(float(max(errs)) if errs else 0.0, 4),
        "used_views": [calibs[i].name for i in active_indices],
        "rejected_views": rejected_views,
        "mean_score": round(float(np.mean([scores[i] for i in active_indices])), 4),
    }
    return (float(pt[0]), float(pt[1]), float(pt[2])), diag


def triangulate_multiview_pose_frames(
    views_by_camera: Mapping[str, Sequence[Pose2DFrame]],
    calibrations: Mapping[str, CameraCalibration],
    *,
    min_score: float = 0.0,
    refine_gauss_newton: bool = True,
    outlier_reproj_thresh_px: float | None = 12.0,
) -> Tuple[list[Pose3DFrame], Dict[str, Any]]:
    """Triangulate synchronized multi-camera `Pose2DFrame` streams into 3D `Pose3DFrame`s."""
    missing_calibs = [cam for cam in views_by_camera if cam not in calibrations]
    if missing_calibs:
        raise KeyError(f"Missing camera calibrations for views: {missing_calibs}")

    frames_by_idx: Dict[int, Dict[str, Pose2DFrame]] = {}
    for cam_name, cam_frames in views_by_camera.items():
        for frame in cam_frames:
            frames_by_idx.setdefault(int(frame.frame_index), {})[cam_name] = frame

    lifted: list[Pose3DFrame] = []
    all_reproj: list[float] = []
    rejected_total = 0
    triangulated_points = 0

    for frame_idx in sorted(frames_by_idx):
        cam_map = frames_by_idx[frame_idx]
        first_frame = next(iter(cam_map.values()))
        all_labels: list[str] = []
        seen_labels: set[str] = set()
        for f in cam_map.values():
            for lbl in f.points:
                if lbl not in seen_labels:
                    seen_labels.add(lbl)
                    all_labels.append(lbl)

        points_3d: Dict[str, Tuple[float, float, float]] = {}
        scores_3d: Dict[str, float] = {}
        per_point_diag: Dict[str, Dict[str, Any]] = {}

        for label in all_labels:
            obs: list[Tuple[CameraCalibration, Tuple[float, float], float]] = []
            for cam_name, frame in cam_map.items():
                if label in frame.points:
                    score = float(frame.scores.get(label, 1.0))
                    if score >= float(min_score):
                        obs.append((calibrations[cam_name], frame.points[label], score))
            if len(obs) < 2:
                continue
            pt_3d, diag = triangulate_point_multiview(
                obs,
                min_score=min_score,
                refine_gauss_newton=refine_gauss_newton,
                outlier_reproj_thresh_px=outlier_reproj_thresh_px,
            )
            points_3d[label] = pt_3d
            scores_3d[label] = float(diag["mean_score"])
            per_point_diag[label] = diag
            all_reproj.append(float(diag["reprojection_rmse_px"]))
            rejected_total += len(diag["rejected_views"])
            triangulated_points += 1

        src = dict(first_frame.source_record)
        src["_triangulation"] = {
            "cameras": sorted(cam_map.keys()),
            "per_keypoint": per_point_diag,
            "mean_reprojection_rmse_px": round(
                float(
                    np.mean(
                        [d["reprojection_rmse_px"] for d in per_point_diag.values()]
                    )
                ),
                4,
            )
            if per_point_diag
            else 0.0,
        }
        lifted.append(
            Pose3DFrame(
                frame_index=frame_idx,
                video_name=first_frame.video_name,
                timestamp_sec=first_frame.timestamp_sec,
                points=points_3d,
                scores=scores_3d,
                source_record=src,
            )
        )

    summary = {
        "frames": len(lifted),
        "cameras": sorted(views_by_camera.keys()),
        "triangulated_points": triangulated_points,
        "rejected_outlier_views": rejected_total,
        "mean_reprojection_rmse_px": round(float(np.mean(all_reproj)), 4)
        if all_reproj
        else 0.0,
        "p95_reprojection_rmse_px": round(float(np.percentile(all_reproj, 95)), 4)
        if all_reproj
        else 0.0,
    }
    return lifted, summary


def _compute_bone_length_cv(
    frames: Sequence[Pose3DFrame],
    edges: Sequence[Tuple[str, str]],
) -> Tuple[float, Dict[str, float]]:
    cvs: list[float] = []
    medians: Dict[str, float] = {}
    for a, b in edges:
        lengths = [
            float(
                np.linalg.norm(
                    np.asarray(f.points[a], dtype=np.float64)
                    - np.asarray(f.points[b], dtype=np.float64)
                )
            )
            for f in frames
            if a in f.points and b in f.points
        ]
        if len(lengths) >= 2:
            arr = np.asarray(lengths, dtype=np.float64)
            med = float(np.median(arr))
            medians[f"{a}--{b}"] = med
            if med > 1e-9:
                cvs.append(float(np.std(arr) / med))
        elif len(lengths) == 1:
            medians[f"{a}--{b}"] = lengths[0]
    mean_cv = float(np.mean(cvs)) if cvs else 0.0
    return mean_cv, medians


def enforce_anatomical_skeleton_constraints(
    frames_3d: Sequence[Pose3DFrame],
    skeleton_edges: Sequence[Tuple[str, str]],
    *,
    reference_lengths: Mapping[Tuple[str, str], float] | None = None,
    iterations: int = 8,
    stiffness: float = 0.85,
    temporal_smooth_weight: float = 0.15,
    confidence_power: float = 1.0,
) -> Tuple[list[Pose3DFrame], Dict[str, Any]]:
    """Project noisy 3D pose trajectories onto anatomically consistent bone-length manifolds.

    Uses confidence-weighted Position-Based Dynamics (PBD) bone projection paired with
    optional temporal velocity regularization so high-confidence landmarks anchor the
    skeleton while noisy or occluded endpoints satisfy anatomical bone lengths.
    """
    if not frames_3d:
        return [], {
            "bone_length_cv_before": 0.0,
            "bone_length_cv_after": 0.0,
            "mean_displacement": 0.0,
            "estimated_bone_lengths": {},
        }

    edges = [(str(a), str(b)) for a, b in skeleton_edges]
    cv_before, _ = _compute_bone_length_cv(frames_3d, edges)

    # Estimate canonical bone lengths from confidence-weighted median if not supplied
    target_lengths: Dict[Tuple[str, str], float] = {}
    for a, b in edges:
        if reference_lengths is not None:
            if (a, b) in reference_lengths:
                target_lengths[(a, b)] = float(reference_lengths[(a, b)])
                continue
            if (b, a) in reference_lengths:
                target_lengths[(a, b)] = float(reference_lengths[(b, a)])
                continue
        vals: list[float] = []
        wts: list[float] = []
        for f in frames_3d:
            if a in f.points and b in f.points:
                pa = np.asarray(f.points[a], dtype=np.float64)
                pb = np.asarray(f.points[b], dtype=np.float64)
                d = float(np.linalg.norm(pb - pa))
                if d > 1e-9:
                    vals.append(d)
                    wa = max(float(f.scores.get(a, 1.0)), 1e-3)
                    wb = max(float(f.scores.get(b, 1.0)), 1e-3)
                    wts.append(math.sqrt(wa * wb))
        if vals:
            target_lengths[(a, b)] = _weighted_median(vals, wts)

    # Mutable per-frame point arrays
    orig_pts = [
        {k: np.asarray(v, dtype=np.float64).copy() for k, v in f.points.items()}
        for f in frames_3d
    ]
    curr_pts = [{k: v.copy() for k, v in frame_map.items()} for frame_map in orig_pts]
    n_frames = len(curr_pts)
    stiff = float(np.clip(stiffness, 0.05, 1.0))
    t_weight = float(np.clip(temporal_smooth_weight, 0.0, 0.5))

    for _ in range(max(1, int(iterations))):
        # 1. Anatomical bone-length projection per frame
        for f_idx in range(n_frames):
            pts = curr_pts[f_idx]
            scores = frames_3d[f_idx].scores
            for a, b in edges:
                if (a, b) not in target_lengths or a not in pts or b not in pts:
                    continue
                target_len = target_lengths[(a, b)]
                diff = pts[b] - pts[a]
                dist = float(np.linalg.norm(diff))
                if dist < 1e-9:
                    continue
                err = dist - target_len
                direction = diff / dist
                wa = max(float(scores.get(a, 1.0)), 0.05) ** float(confidence_power)
                wb = max(float(scores.get(b, 1.0)), 0.05) ** float(confidence_power)
                # Inverse-confidence compliance: low-confidence point moves more
                inv_a = 1.0 / wa
                inv_b = 1.0 / wb
                denom = inv_a + inv_b
                corr = stiff * err * direction
                pts[a] = pts[a] + (inv_a / denom) * corr
                pts[b] = pts[b] - (inv_b / denom) * corr

            # Mild anchor pull toward original observation proportional to confidence
            for label, p_orig in orig_pts[f_idx].items():
                conf = float(np.clip(scores.get(label, 1.0), 0.05, 1.0))
                anchor = 0.08 * conf
                pts[label] = (1.0 - anchor) * pts[label] + anchor * p_orig

        # 2. Temporal midpoint smoothing (Laplacian acceleration penalty)
        if t_weight > 0.0 and n_frames >= 3:
            smoothed = [{k: v.copy() for k, v in m.items()} for m in curr_pts]
            for f_idx in range(1, n_frames - 1):
                for label in curr_pts[f_idx]:
                    if label in curr_pts[f_idx - 1] and label in curr_pts[f_idx + 1]:
                        midpoint = 0.5 * (
                            curr_pts[f_idx - 1][label] + curr_pts[f_idx + 1][label]
                        )
                        smoothed[f_idx][label] = (1.0 - t_weight) * curr_pts[f_idx][
                            label
                        ] + t_weight * midpoint
            curr_pts = smoothed

    # Final hard bone projection pass (zero temporal blur) for exact anatomical manifold
    for f_idx in range(n_frames):
        pts = curr_pts[f_idx]
        scores = frames_3d[f_idx].scores
        for _ in range(3):
            for a, b in edges:
                if (a, b) not in target_lengths or a not in pts or b not in pts:
                    continue
                target_len = target_lengths[(a, b)]
                diff = pts[b] - pts[a]
                dist = float(np.linalg.norm(diff))
                if dist < 1e-9:
                    continue
                err = dist - target_len
                direction = diff / dist
                wa = max(float(scores.get(a, 1.0)), 0.05) ** float(confidence_power)
                wb = max(float(scores.get(b, 1.0)), 0.05) ** float(confidence_power)
                inv_a = 1.0 / wa
                inv_b = 1.0 / wb
                denom = inv_a + inv_b
                pts[a] = pts[a] + (inv_a / denom) * err * direction
                pts[b] = pts[b] - (inv_b / denom) * err * direction

    displacements: list[float] = []
    out_frames: list[Pose3DFrame] = []
    for f_idx, frame in enumerate(frames_3d):
        pts_dict: Dict[str, Tuple[float, float, float]] = {}
        for label, vec in curr_pts[f_idx].items():
            pts_dict[label] = (float(vec[0]), float(vec[1]), float(vec[2]))
            displacements.append(float(np.linalg.norm(vec - orig_pts[f_idx][label])))
        out_frames.append(
            Pose3DFrame(
                frame_index=frame.frame_index,
                video_name=frame.video_name,
                timestamp_sec=frame.timestamp_sec,
                points=pts_dict,
                scores=dict(frame.scores),
                source_record=dict(frame.source_record),
            )
        )

    cv_after, _ = _compute_bone_length_cv(out_frames, edges)
    diagnostics = {
        "bone_length_cv_before": round(cv_before, 6),
        "bone_length_cv_after": round(cv_after, 6),
        "mean_displacement": round(
            float(np.mean(displacements)) if displacements else 0.0, 6
        ),
        "estimated_bone_lengths": {
            f"{a}--{b}": round(length, 6) for (a, b), length in target_lengths.items()
        },
    }
    return out_frames, diagnostics


def _weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    order = np.argsort(np.asarray(values, dtype=np.float64))
    v_sorted = np.asarray(values, dtype=np.float64)[order]
    w_sorted = np.asarray(weights, dtype=np.float64)[order]
    cum = np.cumsum(w_sorted)
    half = 0.5 * float(cum[-1])
    idx = int(np.searchsorted(cum, half))
    idx = min(max(idx, 0), len(v_sorted) - 1)
    return float(v_sorted[idx])


def _decode_depth_map(record: Mapping[str, Any]) -> np.ndarray:
    depth_map = dict((record.get("otherData") or {}).get("depth_map") or {})
    encoded = str(depth_map.get("image_data") or "")
    if not encoded:
        raise ValueError("Depth record is missing otherData.depth_map.image_data")
    scale = dict(depth_map.get("scale") or {})
    d_min = float(scale.get("min", 0.0))
    d_max = float(scale.get("max", 1.0))
    payload = base64.b64decode(encoded)
    with Image.open(io.BytesIO(payload)) as image:
        quantized = np.array(image, dtype=np.float32)
    if quantized.size == 0:
        raise ValueError("Decoded depth map is empty")
    depth = (quantized / 65535.0) * (d_max - d_min) + d_min
    return depth


def _sample_depth(
    depth_map: np.ndarray,
    *,
    x_px: float,
    y_px: float,
    mode: str = "nearest",
    patch_radius: int = 1,
) -> float:
    height, width = depth_map.shape[:2]
    xf = float(np.clip(float(x_px), 0.0, float(max(width - 1, 0))))
    yf = float(np.clip(float(y_px), 0.0, float(max(height - 1, 0))))
    norm_mode = str(mode or "nearest").strip().lower()

    if norm_mode == "bilinear":
        x0 = int(math.floor(xf))
        y0 = int(math.floor(yf))
        x1 = min(x0 + 1, max(width - 1, 0))
        y1 = min(y0 + 1, max(height - 1, 0))
        wx = xf - float(x0)
        wy = yf - float(y0)
        v00 = float(depth_map[y0, x0])
        v10 = float(depth_map[y0, x1])
        v01 = float(depth_map[y1, x0])
        v11 = float(depth_map[y1, x1])
        top = (1.0 - wx) * v00 + wx * v10
        bot = (1.0 - wx) * v01 + wx * v11
        return float((1.0 - wy) * top + wy * bot)

    if norm_mode in ("median_patch", "median"):
        r = max(1, int(patch_radius))
        cx = int(round(xf))
        cy = int(round(yf))
        y_lo = max(0, cy - r)
        y_hi = min(height, cy + r + 1)
        x_lo = max(0, cx - r)
        x_hi = min(width, cx + r + 1)
        patch = np.asarray(depth_map[y_lo:y_hi, x_lo:x_hi], dtype=np.float64).reshape(
            -1
        )
        valid = patch[np.isfinite(patch)]
        if valid.size == 0:
            return float(depth_map[min(cy, height - 1), min(cx, width - 1)])
        return float(np.median(valid))

    x_idx = min(max(int(round(xf)), 0), max(width - 1, 0))
    y_idx = min(max(int(round(yf)), 0), max(height - 1, 0))
    return float(depth_map[y_idx, x_idx])


def _pixel_to_3d(
    *,
    x_px: float,
    y_px: float,
    z_val: float,
    intrinsics: Mapping[str, Any],
) -> tuple[float, float, float]:
    fx = _as_positive_float(intrinsics.get("fx"))
    fy = _as_positive_float(intrinsics.get("fy"))
    cx = _as_float(intrinsics.get("cx"), default=0.0)
    cy = _as_float(intrinsics.get("cy"), default=0.0)
    if fx is None or fy is None:
        return float(x_px), float(y_px), float(z_val)
    x_3d = (float(x_px) - cx) * float(z_val) / fx
    y_3d = (float(y_px) - cy) * float(z_val) / fy
    return float(x_3d), float(y_3d), float(z_val)


def _apply_camera_extrinsics(
    point_cam: Tuple[float, float, float],
    extrinsics: Mapping[str, Any],
) -> Tuple[float, float, float]:
    if not extrinsics:
        return point_cam
    pt = np.asarray(point_cam, dtype=np.float64).reshape(3)
    c2w = extrinsics.get("camera_to_world")
    if c2w is not None:
        mat = np.asarray(c2w, dtype=np.float64).reshape(4, 4)
        world = mat[:3, :3] @ pt + mat[:3, 3]
        return float(world[0]), float(world[1]), float(world[2])
    rot_raw = extrinsics.get("rotation")
    if rot_raw is None:
        rot_raw = extrinsics.get("R")
    trans_raw = extrinsics.get("translation")
    if trans_raw is None:
        trans_raw = extrinsics.get("t")
    if rot_raw is None and trans_raw is None:
        return point_cam
    rot = (
        np.asarray(rot_raw, dtype=np.float64).reshape(3, 3)
        if rot_raw is not None
        else np.eye(3, dtype=np.float64)
    )
    trans = (
        np.asarray(trans_raw, dtype=np.float64).reshape(3)
        if trans_raw is not None
        else np.zeros(3, dtype=np.float64)
    )
    # World-to-camera convention X_cam = R @ X_world + t  =>  X_world = R^T @ (X_cam - t)
    world = rot.T @ (pt - trans)
    return float(world[0]), float(world[1]), float(world[2])


def _as_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _as_positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return parsed
