#!/usr/bin/env python3
"""Reproducible cross-species digital-animal & neural-control benchmark suite
and camera-ready publication figure generator for Annolid ("From Pixels to Circuits").

Executes 4 quantitative benchmark suites directly against Annolid's production
`annolid.simulation` modules plus independent MuJoCo 3.5.0 multi-body cross-validation
and statistical hypothesis testing (bootstrap 95% CIs, paired Wilcoxon signed-rank,
Mann-Whitney U, McNemar exact tests) and exports:
  - `paper/pixels_to_circuits/benchmark_results.json`
  - `paper/pixels_to_circuits/figures/fig1_architecture.{pdf,png}`
  - `paper/pixels_to_circuits/figures/fig2_multiview_3d_lifting.{pdf,png}`
  - `paper/pixels_to_circuits/figures/fig3_rat_ncap_biomechanics.{pdf,png}`
  - `paper/pixels_to_circuits/figures/fig4_real2sim_inversion.{pdf,png}`
  - `paper/pixels_to_circuits/figures/fig5_drosophila_connectome.{pdf,png}`
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
from pathlib import Path
import time
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

os.environ.setdefault("MPLBACKEND", "Agg")

import imageio.v2 as imageio
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from scipy import stats as scipy_stats

from annolid.simulation import (
    RAT_V2_RIGID_BONE_EDGES,
    RAT_V2_SKELETON_EDGES,
    CameraCalibration,
    FlyBrainController,
    Pose2DFrame,
    Pose3DFrame,
    RatNCAPConfig,
    RatNCAPController,
    SparseSynapticWeights,
    build_canonical_fly_connectome,
    build_plastic_fly_connectome,
    build_spiking_fly_connectome,
    enforce_anatomical_skeleton_constraints,
    infer_spinal_circuit_phenotype,
    invert_spring_muscle_activations,
    lift_pose_frames_with_depth,
    run_rat_ncap_rollout,
    solve_rat_v2_ik_sequence,
    triangulate_multiview_pose_frames,
)
from annolid.simulation.biomechanics import normalize_rat_keypoint_dict
from annolid.simulation.rat_ncap import classify_quadruped_gait

REPO_ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = REPO_ROOT / "paper" / "pixels_to_circuits"
FIG_DIR = PAPER_DIR / "figures"


# ---------------------------------------------------------------------------
# Statistical Rigor Utilities (Bootstrap 95% CIs, Wilcoxon, McNemar)
# ---------------------------------------------------------------------------


def _bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 2026,
    alpha: float = 0.05,
) -> Tuple[float, float, float]:
    """Return `(mean, ci_low, ci_high)` via percentile bootstrap."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return 0.0, 0.0, 0.0
    mean_v = float(np.mean(arr))
    if arr.size == 1 or float(np.std(arr)) < 1e-12:
        return mean_v, mean_v, mean_v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boot_means = np.mean(arr[idx], axis=1)
    lo = float(np.percentile(boot_means, 100.0 * alpha / 2.0))
    hi = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    return mean_v, lo, hi


def _paired_wilcoxon_p(a: Sequence[float], b: Sequence[float]) -> float:
    """Two-sided paired Wilcoxon signed-rank p-value (handles zero-diff ties safely)."""
    arr_a = np.asarray(a, dtype=np.float64)
    arr_b = np.asarray(b, dtype=np.float64)
    diff = arr_a - arr_b
    if np.all(np.abs(diff) < 1e-12):
        return 1.0
    _, pval = scipy_stats.wilcoxon(arr_a, arr_b, zero_method="wilcox")
    return float(pval)


def _mcnemar_exact_p(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> float:
    """Exact two-sided McNemar binomial p-value for paired binary classification accuracy."""
    a_arr = np.asarray(correct_a, dtype=bool)
    b_arr = np.asarray(correct_b, dtype=bool)
    n01 = int(np.sum((~a_arr) & b_arr))
    n10 = int(np.sum(a_arr & (~b_arr)))
    total = n01 + n10
    if total == 0:
        return 1.0
    res = scipy_stats.binomtest(min(n01, n10), total, 0.5, alternative="two-sided")
    return float(res.pvalue)


def _walking_spec() -> SimpleNamespace:
    legs = ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")
    names = [f"adhere_claw_{leg}" for leg in legs]
    names.extend(["head_abduct", "head_twist", "head", "abdomen_abduct", "abdomen"])
    for leg in legs:
        names.extend(
            [
                f"coxa_abduct_{leg}",
                f"coxa_twist_{leg}",
                f"coxa_{leg}",
                f"femur_twist_{leg}",
                f"femur_{leg}",
                f"tibia_{leg}",
                f"tarsus_{leg}",
                f"tarsus2_{leg}",
            ]
        )
    minimum = np.concatenate(
        [
            np.zeros(6, dtype=np.float64),
            np.full(len(names) - 6, -1.5, dtype=np.float64),
        ]
    )
    maximum = np.concatenate(
        [
            np.ones(6, dtype=np.float64),
            np.full(len(names) - 6, 2.0, dtype=np.float64),
        ]
    )
    return SimpleNamespace(
        name="\t".join(names),
        shape=(len(names),),
        dtype=np.float64,
        minimum=minimum,
        maximum=maximum,
    )


def _flight_spec() -> SimpleNamespace:
    names = (
        "head_abduct",
        "head_twist",
        "head",
        "wing_yaw_left",
        "wing_roll_left",
        "wing_pitch_left",
        "wing_yaw_right",
        "wing_roll_right",
        "wing_pitch_right",
        "abdomen_abduct",
        "abdomen",
        "user_0",
    )
    return SimpleNamespace(
        name="\t".join(names),
        shape=(len(names),),
        dtype=np.float64,
        minimum=np.full(len(names), -1.0, dtype=np.float64),
        maximum=np.full(len(names), 1.0, dtype=np.float64),
    )


def _make_rig_cameras(n_cams: int) -> Dict[str, CameraCalibration]:
    cams: Dict[str, CameraCalibration] = {}
    for idx in range(n_cams):
        ang = (2.0 * math.pi * idx) / float(n_cams) + 0.35
        pos = [0.72 * math.cos(ang), 0.72 * math.sin(ang), 0.48]
        cams[f"cam{idx}"] = CameraCalibration.from_look_at(
            name=f"cam{idx}",
            position=pos,
            target=[0.18, 0.0, 0.05],
            fx=820.0,
            fy=820.0,
            cx=320.0,
            cy=240.0,
            distortion=(0.08, -0.02, 0.001, -0.001),
            image_size=(640, 480),
        )
    return cams


def _per_frame_mpjpe_mm(
    pred_frames: Sequence[Pose3DFrame], gt_frames: Sequence[Dict[str, List[float]]]
) -> List[float]:
    frame_errs: List[float] = []
    for pf, gt in zip(pred_frames, gt_frames):
        errs = [
            float(
                np.linalg.norm(
                    np.asarray(pf.points[lbl], dtype=np.float64)
                    - np.asarray(p_gt, dtype=np.float64)
                )
            )
            * 1000.0
            for lbl, p_gt in gt.items()
            if lbl in pf.points
        ]
        if errs:
            frame_errs.append(float(np.mean(errs)))
    return frame_errs


def _mpjpe_mm(
    pred_frames: Sequence[Pose3DFrame], gt_frames: Sequence[Dict[str, List[float]]]
) -> float:
    vals = _per_frame_mpjpe_mm(pred_frames, gt_frames)
    return float(np.mean(vals)) if vals else 0.0


def _encode_depth_record(
    frame_index: int, depth: np.ndarray, d_min: float = 0.0, d_max: float = 2.0
) -> Dict[str, Any]:
    norm = np.clip((depth - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)
    quant = np.round(norm * 65535.0).astype(np.uint16)
    buf = io.BytesIO()
    imageio.imwrite(buf, quant, format="png")
    return {
        "frame_index": frame_index,
        "otherData": {
            "depth_map": {
                "image_data": base64.b64encode(buf.getvalue()).decode("ascii"),
                "scale": {"min": d_min, "max": d_max},
                "dtype": "uint16",
            }
        },
    }


# ---------------------------------------------------------------------------
# Experiment 1: Multi-View 3D Lifting, Temporal Ablation, Occlusion & Depth
# ---------------------------------------------------------------------------


def run_experiment_1_multiview_lifting() -> Dict[str, Any]:
    print(
        "[Exp 1] Running multi-view triangulation, PBD temporal ablation, occlusion & depth..."
    )
    rollout = run_rat_ncap_rollout(
        steps=60,
        behavior="trot",
        target_speed=0.50,
        interface="flexor_extensor",
        seed=7,
    )
    gt_frames = [f["keypoints_3d"] for f in rollout["frames"]]

    noise_levels_px = [0.0, 1.0, 2.0, 4.0, 6.0, 8.0]
    cam_counts = [2, 3, 4]
    sweep_results: List[Dict[str, Any]] = []

    for n_cams in cam_counts:
        cams = _make_rig_cameras(n_cams)
        for sigma_px in noise_levels_px:
            rng = np.random.default_rng(1000 + n_cams * 100 + int(sigma_px * 10))
            views: Dict[str, List[Pose2DFrame]] = {c: [] for c in cams}
            for f_idx, gt in enumerate(gt_frames):
                for cname, cam in cams.items():
                    pts2d: Dict[str, Tuple[float, float]] = {}
                    scores2d: Dict[str, float] = {}
                    for lbl, p3 in gt.items():
                        u, v, _ = cam.project(p3, apply_distortion=True)
                        noise = (
                            rng.normal(0.0, sigma_px, size=2)
                            if sigma_px > 0
                            else np.zeros(2)
                        )
                        pts2d[lbl] = (float(u + noise[0]), float(v + noise[1]))
                        scores2d[lbl] = float(
                            np.clip(0.95 - 0.02 * sigma_px, 0.5, 0.98)
                        )
                    views[cname].append(
                        Pose2DFrame(
                            frame_index=f_idx,
                            image_height=480,
                            image_width=640,
                            video_name=f"{cname}.mp4",
                            timestamp_sec=f_idx * 0.02,
                            points=pts2d,
                            scores=scores2d,
                        )
                    )

            lifted_dlt, _ = triangulate_multiview_pose_frames(
                views, cams, refine_gauss_newton=False, outlier_reproj_thresh_px=None
            )
            lifted_gn, tri_summary = triangulate_multiview_pose_frames(
                views, cams, refine_gauss_newton=True, outlier_reproj_thresh_px=None
            )
            # Ablate temporal smoothing: lambda_t = 0.0 (spatial-only rigid PBD) vs lambda_t = 0.15
            con_rigid_spatial, diag_spatial = enforce_anatomical_skeleton_constraints(
                lifted_gn,
                RAT_V2_RIGID_BONE_EDGES,
                iterations=8,
                temporal_smooth_weight=0.0,
            )
            con_rigid, diag_rigid = enforce_anatomical_skeleton_constraints(
                lifted_gn,
                RAT_V2_RIGID_BONE_EDGES,
                iterations=8,
                temporal_smooth_weight=0.15,
            )
            con_all, diag_all = enforce_anatomical_skeleton_constraints(
                lifted_gn,
                RAT_V2_SKELETON_EDGES,
                iterations=8,
                temporal_smooth_weight=0.15,
            )

            err_gn_f = _per_frame_mpjpe_mm(lifted_gn, gt_frames)
            err_rigid_f = _per_frame_mpjpe_mm(con_rigid, gt_frames)
            err_spatial_f = _per_frame_mpjpe_mm(con_rigid_spatial, gt_frames)
            m_rigid, lo_r, hi_r = _bootstrap_ci(err_rigid_f)
            m_gn, lo_g, hi_g = _bootstrap_ci(err_gn_f)
            p_wilcox = _paired_wilcoxon_p(err_rigid_f, err_gn_f)

            sweep_results.append(
                {
                    "n_cams": n_cams,
                    "sigma_px": sigma_px,
                    "mpjpe_dlt_mm": round(_mpjpe_mm(lifted_dlt, gt_frames), 3),
                    "mpjpe_gn_mm": round(m_gn, 3),
                    "mpjpe_gn_ci95": [round(lo_g, 3), round(hi_g, 3)],
                    "mpjpe_rigid_spatial_mm": round(float(np.mean(err_spatial_f)), 3),
                    "mpjpe_rigid_mm": round(m_rigid, 3),
                    "mpjpe_rigid_ci95": [round(lo_r, 3), round(hi_r, 3)],
                    "mpjpe_all_edges_mm": round(_mpjpe_mm(con_all, gt_frames), 3),
                    "wilcoxon_p_rigid_vs_gn": float(f"{p_wilcox:.3e}"),
                    "reproj_rmse_px": tri_summary["mean_reprojection_rmse_px"],
                    "bone_cv_before_pct": round(
                        diag_rigid["bone_length_cv_before"] * 100.0, 3
                    ),
                    "bone_cv_spatial_after_pct": round(
                        diag_spatial["bone_length_cv_after"] * 100.0, 3
                    ),
                    "bone_cv_after_pct": round(
                        diag_rigid["bone_length_cv_after"] * 100.0, 3
                    ),
                    "all_edges_cv_after_pct": round(
                        diag_all["bone_length_cv_after"] * 100.0, 3
                    ),
                }
            )

    # 1B: Severe single-view outlier stress test (20% keypoints corrupted by 65 px in 1 camera)
    outlier_results: List[Dict[str, Any]] = []
    for n_cams in (3, 4):
        cams = _make_rig_cameras(n_cams)
        rng = np.random.default_rng(7700 + n_cams)
        views = {c: [] for c in cams}
        cam_names = list(cams.keys())
        injected_outliers = 0
        for f_idx, gt in enumerate(gt_frames):
            corrupt_map: Dict[str, str] = {}
            for lbl in gt:
                if rng.random() < 0.20:
                    corrupt_map[lbl] = cam_names[int(rng.integers(0, n_cams))]
                    injected_outliers += 1
            for cname, cam in cams.items():
                pts2d = {}
                scores2d = {}
                for lbl, p3 in gt.items():
                    u, v, _ = cam.project(p3, apply_distortion=True)
                    noise = rng.normal(0.0, 1.5, size=2)
                    if corrupt_map.get(lbl) == cname:
                        ang = float(rng.uniform(0.0, math.tau))
                        noise += np.array([65.0 * math.cos(ang), 65.0 * math.sin(ang)])
                    pts2d[lbl] = (float(u + noise[0]), float(v + noise[1]))
                    scores2d[lbl] = 0.90
                views[cname].append(
                    Pose2DFrame(
                        frame_index=f_idx,
                        image_height=480,
                        image_width=640,
                        video_name=f"{cname}.mp4",
                        timestamp_sec=f_idx * 0.02,
                        points=pts2d,
                        scores=scores2d,
                    )
                )

        raw_dlt, raw_sum = triangulate_multiview_pose_frames(
            views, cams, refine_gauss_newton=False, outlier_reproj_thresh_px=None
        )
        rob_gn, rob_sum = triangulate_multiview_pose_frames(
            views, cams, refine_gauss_newton=True, outlier_reproj_thresh_px=10.0
        )
        rob_rigid, rob_skel = enforce_anatomical_skeleton_constraints(
            rob_gn, RAT_V2_RIGID_BONE_EDGES, iterations=8, temporal_smooth_weight=0.15
        )
        err_raw_f = _per_frame_mpjpe_mm(raw_dlt, gt_frames)
        err_rob_f = _per_frame_mpjpe_mm(rob_rigid, gt_frames)
        m_rob, lo_rob, hi_rob = _bootstrap_ci(err_rob_f)
        outlier_results.append(
            {
                "n_cams": n_cams,
                "injected_outliers": injected_outliers,
                "rejected_outliers": rob_sum["rejected_outlier_views"],
                "mpjpe_unprotected_dlt_mm": round(float(np.mean(err_raw_f)), 3),
                "mpjpe_robust_gn_mm": round(_mpjpe_mm(rob_gn, gt_frames), 3),
                "mpjpe_robust_gn_rigid_mm": round(m_rob, 3),
                "mpjpe_robust_ci95": [round(lo_rob, 3), round(hi_rob, 3)],
                "wilcoxon_p_vs_dlt": float(
                    f"{_paired_wilcoxon_p(err_rob_f, err_raw_f):.3e}"
                ),
                "reproj_unprotected_px": raw_sum["mean_reprojection_rmse_px"],
                "reproj_robust_px": rob_sum["mean_reprojection_rmse_px"],
                "bone_cv_before_pct": round(
                    rob_skel["bone_length_cv_before"] * 100.0, 3
                ),
                "bone_cv_after_pct": round(rob_skel["bone_length_cv_after"] * 100.0, 3),
            }
        )

    # 1B-2: Partial-visibility camera-dropout occlusion benchmark (30% single-view keypoint dropout @ sigma=4px)
    occlusion_results: List[Dict[str, Any]] = []
    for n_cams in (3, 4):
        cams = _make_rig_cameras(n_cams)
        rng = np.random.default_rng(8800 + n_cams)
        views = {c: [] for c in cams}
        cam_names = list(cams.keys())
        dropped_obs = 0
        for f_idx, gt in enumerate(gt_frames):
            drop_map: Dict[str, str] = {}
            for lbl in gt:
                if rng.random() < 0.30:
                    drop_map[lbl] = cam_names[int(rng.integers(0, n_cams))]
                    dropped_obs += 1
            for cname, cam in cams.items():
                pts2d = {}
                scores2d = {}
                for lbl, p3 in gt.items():
                    if drop_map.get(lbl) == cname:
                        continue  # Occluded in this camera view
                    u, v, _ = cam.project(p3, apply_distortion=True)
                    noise = rng.normal(0.0, 4.0, size=2)
                    pts2d[lbl] = (float(u + noise[0]), float(v + noise[1]))
                    scores2d[lbl] = 0.88
                views[cname].append(
                    Pose2DFrame(
                        frame_index=f_idx,
                        image_height=480,
                        image_width=640,
                        video_name=f"{cname}.mp4",
                        timestamp_sec=f_idx * 0.02,
                        points=pts2d,
                        scores=scores2d,
                    )
                )
        occ_gn, occ_sum = triangulate_multiview_pose_frames(
            views, cams, refine_gauss_newton=True, outlier_reproj_thresh_px=12.0
        )
        occ_rigid, occ_skel = enforce_anatomical_skeleton_constraints(
            occ_gn, RAT_V2_RIGID_BONE_EDGES, iterations=8, temporal_smooth_weight=0.15
        )
        e_gn = _per_frame_mpjpe_mm(occ_gn, gt_frames)
        e_rig = _per_frame_mpjpe_mm(occ_rigid, gt_frames)
        m_rig, lo_rig, hi_rig = _bootstrap_ci(e_rig)
        occlusion_results.append(
            {
                "n_cams": n_cams,
                "dropout_rate": 0.30,
                "sigma_px": 4.0,
                "dropped_observations": dropped_obs,
                "reconstructed_points": occ_sum["triangulated_points"],
                "mpjpe_gn_mm": round(float(np.mean(e_gn)), 3),
                "mpjpe_rigid_mm": round(m_rig, 3),
                "mpjpe_rigid_ci95": [round(lo_rig, 3), round(hi_rig, 3)],
                "wilcoxon_p_rigid_vs_gn": float(
                    f"{_paired_wilcoxon_p(e_rig, e_gn):.3e}"
                ),
                "bone_cv_before_pct": round(
                    occ_skel["bone_length_cv_before"] * 100.0, 3
                ),
                "bone_cv_after_pct": round(occ_skel["bone_length_cv_after"] * 100.0, 3),
            }
        )

    # 1C: Monocular depth-map lifting sampling mode ablation (clean vs 8% salt-and-pepper spikes)
    rng_d = np.random.default_rng(31415)
    cam_top = CameraCalibration.from_look_at(
        name="top_depth",
        position=[0.18, 0.0, 0.90],
        target=[0.18, 0.0, 0.05],
        up=(0.0, 1.0, 0.0),
        fx=640.0,
        fy=640.0,
        cx=160.0,
        cy=120.0,
        image_size=(320, 240),
    )
    depth_modes = ["nearest", "bilinear", "median_patch"]
    depth_bench: Dict[str, Dict[str, Any]] = {}
    h_img, w_img = 240, 320
    xs_grid, ys_grid = np.meshgrid(np.arange(w_img), np.arange(h_img))

    for condition in ("clean_subpixel", "spiked_8pct"):
        pose_2d_list: List[Pose2DFrame] = []
        depth_recs: Dict[int, Dict[str, Any]] = {}
        gt_sub: List[Dict[str, List[float]]] = []
        for f_idx, gt in enumerate(gt_frames[:30]):
            gt_sub.append(gt)
            base_z = 0.85 + 0.0004 * (xs_grid - 160.0) - 0.0003 * (ys_grid - 120.0)
            pts2d = {}
            for lbl, p3 in gt.items():
                u, v, z_cam = cam_top.project(p3, apply_distortion=False)
                u_c = float(np.clip(u, 4.0, w_img - 5.0))
                v_c = float(np.clip(v, 4.0, h_img - 5.0))
                pts2d[lbl] = (u_c, v_c)
                ix, iy = int(round(u_c)), int(round(v_c))
                y0, y1 = max(0, iy - 3), min(h_img, iy + 4)
                x0, x1 = max(0, ix - 3), min(w_img, ix + 4)
                patch_x = xs_grid[y0:y1, x0:x1] - u_c
                patch_y = ys_grid[y0:y1, x0:x1] - v_c
                base_z[y0:y1, x0:x1] = z_cam + 0.008 * patch_x + 0.005 * patch_y

            if condition == "spiked_8pct":
                spike_mask = rng_d.random((h_img, w_img)) < 0.08
                base_z = np.where(
                    spike_mask,
                    rng_d.uniform(0.15, 1.85, size=(h_img, w_img)),
                    base_z,
                )
            depth_recs[f_idx] = _encode_depth_record(
                f_idx, base_z, d_min=0.0, d_max=2.0
            )
            pose_2d_list.append(
                Pose2DFrame(
                    frame_index=f_idx,
                    image_height=h_img,
                    image_width=w_img,
                    video_name="depth_top.mp4",
                    timestamp_sec=f_idx * 0.02,
                    points=pts2d,
                )
            )

        coord_sys = {
            "camera_intrinsics": {
                "fx": cam_top.fx,
                "fy": cam_top.fy,
                "cx": cam_top.cx,
                "cy": cam_top.cy,
            },
            "camera_extrinsics": {
                "R": cam_top.rotation.tolist(),
                "t": cam_top.translation.tolist(),
            },
        }
        per_mode_errs: Dict[str, List[float]] = {}
        cond_out: Dict[str, Any] = {}
        for mode in depth_modes:
            lifted_d = lift_pose_frames_with_depth(
                pose_2d_list,
                depth_records=depth_recs,
                coordinate_system=coord_sys,
                sample_mode=mode,
                patch_radius=1,
            )
            f_errs = _per_frame_mpjpe_mm(lifted_d, gt_sub)
            per_mode_errs[mode] = f_errs
            m_v, lo_v, hi_v = _bootstrap_ci(f_errs)
            cond_out[f"{mode}_mpjpe_mm"] = round(m_v, 3)
            cond_out[f"{mode}_ci95"] = [round(lo_v, 3), round(hi_v, 3)]
        cond_out["wilcoxon_p_bilinear_vs_nearest"] = float(
            f"{_paired_wilcoxon_p(per_mode_errs['bilinear'], per_mode_errs['nearest']):.3e}"
        )
        cond_out["wilcoxon_p_median_vs_nearest"] = float(
            f"{_paired_wilcoxon_p(per_mode_errs['median_patch'], per_mode_errs['nearest']):.3e}"
        )
        depth_bench[condition] = cond_out

    return {
        "multiview_sweep": sweep_results,
        "outlier_stress": outlier_results,
        "occlusion_dropout": occlusion_results,
        "depth_sampling_ablation": depth_bench,
    }


# ---------------------------------------------------------------------------
# Experiment 2: Mammalian Rat v2 Neuromechanical Interfaces & Spinal Phase Map
# ---------------------------------------------------------------------------


def run_experiment_2_rat_ncap() -> Dict[str, Any]:
    print(
        "[Exp 2] Running Rat NCAP speed x ablation phase diagram & 4-interface sweep..."
    )
    speeds = [round(v, 2) for v in np.linspace(0.0, 0.96, 17)]
    v0_scales = [1.0, 0.75, 0.50, 0.25, 0.0]
    gait_code = {"stand": 0, "walk": 1, "trot": 2, "bound": 3}

    phase_grid: List[Dict[str, Any]] = []
    for k_v0 in v0_scales:
        for spd in speeds:
            r = run_rat_ncap_rollout(
                steps=140,
                behavior="trot",
                target_speed=spd,
                interface="flexor_extensor",
                ablation="v0v_v0d" if k_v0 < 1.0 else "none",
                ablation_scale=k_v0,
                seed=7,
            )
            tail = r["frames"][-40:]
            gait = tail[-1]["gait"]
            p_fl = float(tail[-1]["phase_offsets_rel_fr"]["FL"])
            lr_sync = 1.0 - 2.0 * min(abs(p_fl), abs(1.0 - p_fl))
            phase_grid.append(
                {
                    "v0_scale": k_v0,
                    "target_speed_mps": spd,
                    "gait": gait,
                    "gait_code": gait_code[gait],
                    "step_frequency_hz": round(
                        float(np.mean([f["step_frequency_hz"] for f in tail])), 3
                    ),
                    "duty_factor": round(
                        float(np.mean([f["duty_factor"] for f in tail])), 3
                    ),
                    "lr_synchrony": round(lr_sync, 4),
                    "mean_positive_power_w": r["metadata"]["mean_positive_power_w"],
                }
            )

    # V3 ablation comparison across speeds
    v3_sweep: List[Dict[str, Any]] = []
    for k_v3 in (1.0, 0.0):
        for spd in (0.22, 0.45, 0.60, 0.80, 0.92):
            r = run_rat_ncap_rollout(
                steps=140,
                behavior="trot",
                target_speed=spd,
                ablation="v3f_v3a" if k_v3 < 1.0 else "none",
                ablation_scale=k_v3,
                seed=7,
            )
            tail = r["frames"][-1]
            v3_sweep.append(
                {
                    "v3_scale": k_v3,
                    "target_speed_mps": spd,
                    "gait": tail["gait"],
                    "phase_FL": tail["phase_offsets_rel_fr"]["FL"],
                    "phase_HL": tail["phase_offsets_rel_fr"]["HL"],
                    "step_frequency_hz": tail["step_frequency_hz"],
                }
            )

    # Four neuromechanical interfaces with joint excursion & RMS velocity/torque context
    interfaces = ("flexor_extensor", "position_stiffness", "position", "torque")
    interface_table: List[Dict[str, Any]] = []
    for iface in interfaces:
        for beh, spd, bump in (
            ("walk", 0.22, 0.0),
            ("trot", 0.50, 0.0),
            ("bound", 0.88, 0.0),
            ("bumpy_terrain", 0.48, 3.0),
        ):
            r = run_rat_ncap_rollout(
                steps=150,
                behavior=beh,
                target_speed=spd,
                interface=iface,
                terrain_bump_cm=bump,
                seed=7,
            )
            tail = r["frames"][-60:]
            qs = np.asarray([f["qpos"] for f in tail], dtype=np.float64)
            dqs = np.asarray([f["qvel"] for f in tail], dtype=np.float64)
            taus = np.asarray([f["torque"] for f in tail], dtype=np.float64)
            stiff_mean = float(np.mean([np.mean(f["stiffness"]) for f in tail]))
            peak_tau = float(np.max(np.abs(taus)))
            rms_tau = float(np.sqrt(np.mean(np.square(taus))))
            rms_dq = float(np.sqrt(np.mean(np.square(dqs))))
            mean_ptp_deg = float(np.rad2deg(np.mean(np.ptp(qs, axis=0))))
            pos_w = float(
                np.mean([f["mechanical_power"]["positive_power_w"] for f in tail])
            )
            neg_w = float(
                np.mean([f["mechanical_power"]["negative_power_w"] for f in tail])
            )
            abs_w = float(np.mean([f["mechanical_power"]["abs_power_w"] for f in tail]))
            mean_v = float(np.mean([f["forward_speed_mps"] for f in tail]))
            cot = abs_w / max(0.35 * 9.81 * max(mean_v, 0.05), 1e-6)
            interface_table.append(
                {
                    "interface": iface,
                    "behavior": beh,
                    "dominant_gait": r["metadata"]["dominant_gait"],
                    "mean_speed_mps": round(mean_v, 4),
                    "mean_positive_power_w": round(pos_w, 5),
                    "mean_negative_power_w": round(neg_w, 5),
                    "mean_abs_power_w": round(abs_w, 5),
                    "cost_of_transport": round(cot, 4),
                    "mean_stiffness_nm_rad": round(stiff_mean, 4),
                    "peak_torque_nm": round(peak_tau, 4),
                    "rms_torque_nm": round(rms_tau, 4),
                    "rms_joint_vel_rad_s": round(rms_dq, 4),
                    "mean_joint_excursion_deg": round(mean_ptp_deg, 2),
                }
            )

    # RG synaptic weight noise + bumpy terrain robustness (10 seeds)
    rg_noise_sweep: List[Dict[str, Any]] = []
    for sigma in (0.0, 0.10, 0.20, 0.30, 0.40):
        spds_out, pos_out, trot_ok = [], [], 0
        for s in range(10):
            r = run_rat_ncap_rollout(
                steps=120,
                behavior="trot",
                target_speed=0.50,
                rg_noise_std=sigma,
                seed=100 + s,
            )
            spds_out.append(r["metadata"]["mean_speed_mps"])
            pos_out.append(r["metadata"]["mean_positive_power_w"])
            if r["frames"][-1]["gait"] == "trot":
                trot_ok += 1
        m_s, lo_s, hi_s = _bootstrap_ci(spds_out)
        rg_noise_sweep.append(
            {
                "rg_noise_std": sigma,
                "mean_speed_mps": round(m_s, 4),
                "speed_ci95": [round(lo_s, 4), round(hi_s, 4)],
                "speed_retention_pct": round(m_s / 0.483 * 100.0, 2),
                "mean_positive_power_w": round(float(np.mean(pos_out)), 5),
                "trot_preserved_pct": round(100.0 * trot_ok / 10.0, 1),
            }
        )

    return {
        "speeds_mps": speeds,
        "v0_scales": v0_scales,
        "phase_grid": phase_grid,
        "v3_sweep": v3_sweep,
        "interface_table": interface_table,
        "rg_noise_sweep": rg_noise_sweep,
    }


# ---------------------------------------------------------------------------
# Experiment 3: Real-to-Sim Rat v2 IK, 6-Class Blind Phenotype & Independent MuJoCo
# ---------------------------------------------------------------------------


_INDEPENDENT_RAT_MJCF = """
<mujoco model="independent_articulated_rat">
  <compiler angle="radian"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <default>
    <joint damping="0.003" armature="0.0002"/>
    <geom friction="0.9 0.01 0.001" solref="0.015 1.0"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="40 40 0.05" pos="0 0 0"/>
    <body name="torso" pos="0 0 0.056">
      <freejoint name="root"/>
      <geom name="torso_geom" type="capsule" fromto="-0.042 0 0 0.045 0 0.006" size="0.018" mass="0.16" contype="0" conaffinity="0"/>
      <site name="torso_center" pos="0 0 0"/>
      <site name="thorax" pos="0.045 0 0.006"/>
      <site name="neck" pos="0.082 0 0.014"/>
      <body name="head_body" pos="0.082 0 0.014">
        <geom name="head_geom" type="capsule" fromto="0 0 0 0.060 0 0.0" size="0.013" mass="0.045" contype="0" conaffinity="0"/>
        <site name="head" pos="0.030 0 0.004"/>
        <site name="nose" pos="0.063 0 0.0"/>
        <site name="left_ear" pos="0.024 0.016 0.018"/>
        <site name="right_ear" pos="0.024 -0.016 0.018"/>
      </body>
      <body name="lumbar_body" pos="-0.042 0 0">
        <joint name="lumbar_pitch" type="hinge" axis="0 1 0" range="-0.45 0.45"/>
        <joint name="lumbar_yaw" type="hinge" axis="0 0 1" range="-0.45 0.45"/>
        <geom name="lumbar_geom" type="capsule" fromto="0 0 0 -0.046 0 -0.002" size="0.017" mass="0.075" contype="0" conaffinity="0"/>
        <site name="lumbar" pos="0 0 0"/>
        <site name="pelvis" pos="-0.046 0 -0.002"/>
        <site name="tail_base" pos="-0.074 0 -0.004"/>
        <site name="tail_mid" pos="-0.153 0 -0.016"/>
        <site name="tail_tip" pos="-0.243 0 -0.024"/>
        <body name="hl_thigh" pos="-0.030 0.028 -0.004">
          <joint name="hl_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.028" size="0.006" mass="0.012" contype="0" conaffinity="0"/>
          <body name="hl_shin" pos="0 0 -0.028">
            <joint name="hl_knee" type="hinge" axis="0 1 0" range="-1.3 1.3"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.024" size="0.005" mass="0.007" contype="0" conaffinity="0"/>
            <geom name="hl_foot" type="sphere" pos="0 0 -0.024" size="0.005" mass="0.003"/>
            <site name="left_hindpaw" pos="0 0 -0.024"/>
          </body>
        </body>
        <body name="hr_thigh" pos="-0.030 -0.028 -0.004">
          <joint name="hr_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.028" size="0.006" mass="0.012" contype="0" conaffinity="0"/>
          <body name="hr_shin" pos="0 0 -0.028">
            <joint name="hr_knee" type="hinge" axis="0 1 0" range="-1.3 1.3"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.024" size="0.005" mass="0.007" contype="0" conaffinity="0"/>
            <geom name="hr_foot" type="sphere" pos="0 0 -0.024" size="0.005" mass="0.003"/>
            <site name="right_hindpaw" pos="0 0 -0.024"/>
          </body>
        </body>
      </body>
      <body name="fl_upper" pos="0.056 0.024 -0.004">
        <joint name="fl_shoulder" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.026" size="0.006" mass="0.010" contype="0" conaffinity="0"/>
        <body name="fl_lower" pos="0 0 -0.026">
          <joint name="fl_elbow" type="hinge" axis="0 1 0" range="-1.3 1.3"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.024" size="0.005" mass="0.006" contype="0" conaffinity="0"/>
          <geom name="fl_foot" type="sphere" pos="0 0 -0.024" size="0.005" mass="0.002"/>
          <site name="left_forepaw" pos="0 0 -0.024"/>
        </body>
      </body>
      <body name="fr_upper" pos="0.056 -0.024 -0.004">
        <joint name="fr_shoulder" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.026" size="0.006" mass="0.010" contype="0" conaffinity="0"/>
        <body name="fr_lower" pos="0 0 -0.026">
          <joint name="fr_elbow" type="hinge" axis="0 1 0" range="-1.3 1.3"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.024" size="0.005" mass="0.006" contype="0" conaffinity="0"/>
          <geom name="fr_foot" type="sphere" pos="0 0 -0.024" size="0.005" mass="0.002"/>
          <site name="right_forepaw" pos="0 0 -0.024"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_lp" joint="lumbar_pitch" kp="0.4" kv="0.03"/>
    <position name="a_ly" joint="lumbar_yaw" kp="0.4" kv="0.03"/>
    <position name="a_hl_h" joint="hl_hip" kp="0.9" kv="0.015"/>
    <position name="a_hl_k" joint="hl_knee" kp="0.8" kv="0.014"/>
    <position name="a_hr_h" joint="hr_hip" kp="0.9" kv="0.015"/>
    <position name="a_hr_k" joint="hr_knee" kp="0.8" kv="0.014"/>
    <position name="a_fl_s" joint="fl_shoulder" kp="0.8" kv="0.014"/>
    <position name="a_fl_e" joint="fl_elbow" kp="0.8" kv="0.014"/>
    <position name="a_fr_s" joint="fr_shoulder" kp="0.8" kv="0.014"/>
    <position name="a_fr_e" joint="fr_elbow" kp="0.8" kv="0.014"/>
  </actuator>
</mujoco>
"""


def _run_independent_mujoco_cross_validation() -> Dict[str, Any]:
    """Independent MuJoCo 3.5.0 rigid-body quadruped cross-validation (breaks inverse-crime)."""
    print(
        "[Exp 3C] Running independent MuJoCo 3.5.0 multi-body quadruped cross-validation..."
    )
    model = mujoco.MjModel.from_xml_string(_INDEPENDENT_RAT_MJCF)
    data = mujoco.MjData(model)
    site_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
    ]
    foot_geom_ids = {
        lbl: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, nm)
        for lbl, nm in (
            ("HL", "hl_foot"),
            ("HR", "hr_foot"),
            ("FL", "fl_foot"),
            ("FR", "fr_foot"),
        )
    }
    regimes = [
        (
            "intact_walk",
            1.9,
            {"FR": 0.0, "HL": 0.75, "FL": 0.50, "HR": 0.25},
            0.16,
            0.74,
        ),
        (
            "intact_trot",
            3.4,
            {"FR": 0.0, "HL": 0.00, "FL": 0.50, "HR": 0.50},
            0.48,
            0.60,
        ),
        (
            "v0_commissural_deficit_phenotype",
            3.4,
            {"FR": 0.0, "HL": 0.50, "FL": 0.00, "HR": 0.50},
            0.48,
            0.55,
        ),
        (
            "intact_bound",
            4.8,
            {"FR": 0.0, "HL": 0.50, "FL": 0.00, "HR": 0.50},
            0.88,
            0.46,
        ),
        (
            "v3_synchrony_deficit_phenotype",
            4.8,
            {"FR": 0.0, "HL": 0.00, "FL": 0.50, "HR": 0.50},
            0.88,
            0.52,
        ),
    ]

    per_regime_summary: List[Dict[str, Any]] = []
    mj_sequences: List[Tuple[str, List[Dict[str, List[float]]]]] = []
    total_mass = float(np.sum(model.body_mass))

    for gt_cls, freq_hz, phase_offs, speed_bias, duty in regimes:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.056
        data.qvel[0] = speed_bias
        kps_seq: List[Dict[str, List[float]]] = []
        mj_hip_q: List[List[float]] = []
        mj_hip_tau_inv: List[List[float]] = []
        mj_hip_tau_smooth: List[List[float]] = []
        mj_foot_con: List[List[int]] = []

        for f in range(130):
            t = f * 0.02
            for s in range(10):
                ts = t + s * 0.002
                phi = {
                    limb: (freq_hz * ts + ph0) % 1.0 for limb, ph0 in phase_offs.items()
                }

                def _leg_cmd(u: float) -> Tuple[float, float]:
                    sw_dur = 1.0 - duty
                    if u < sw_dur:
                        su = u / max(sw_dur, 1e-4)
                        lift = math.sin(math.pi * su)
                        hip = 0.52 * math.cos(math.pi * su)
                        knee = 1.15 * lift
                    else:
                        st_u = (u - sw_dur) / max(duty, 1e-4)
                        hip = -0.52 * math.cos(math.pi * st_u)
                        knee = -0.05
                    return hip, knee

                hl_h, hl_k = _leg_cmd(phi["HL"])
                hr_h, hr_k = _leg_cmd(phi["HR"])
                fl_h, fl_k = _leg_cmd(phi["FL"])
                fr_h, fr_k = _leg_cmd(phi["FR"])
                data.ctrl[:] = [
                    0.14
                    * (
                        math.sin(2.0 * math.pi * phi["FL"])
                        - math.sin(2.0 * math.pi * phi["HL"])
                    ),
                    0.09
                    * (
                        math.sin(2.0 * math.pi * phi["FL"])
                        - math.sin(2.0 * math.pi * phi["FR"])
                    ),
                    hl_h,
                    hl_k,
                    hr_h,
                    hr_k,
                    fl_h,
                    fl_k,
                    fr_h,
                    fr_k,
                ]
                data.xfrc_applied[1, 0] = 28.0 * (speed_bias - data.qvel[0])
                data.xfrc_applied[1, 1] = -12.0 * data.qpos[1] - 2.0 * data.qvel[1]
                data.xfrc_applied[1, 2] = (
                    0.70 * total_mass * 9.81
                    + 180.0 * (0.056 - data.qpos[2])
                    - 5.0 * data.qvel[2]
                )
                data.xfrc_applied[1, 3] = -2.5 * data.qpos[4] - 0.12 * data.qvel[3]
                data.xfrc_applied[1, 4] = -2.5 * data.qpos[5] - 0.12 * data.qvel[4]
                data.xfrc_applied[1, 5] = -2.5 * data.qpos[6] - 0.12 * data.qvel[5]
                mujoco.mj_step(model, data)

            mujoco.mj_inverse(model, data)
            tau_no_con = data.qfrc_inverse + data.qfrc_constraint
            kps_seq.append(
                {
                    nm: data.site_xpos[i].copy().tolist()
                    for i, nm in enumerate(site_names)
                }
            )
            mj_hip_q.append(
                [
                    float(data.qpos[9]),
                    float(data.qpos[11]),
                    float(data.qpos[13]),
                    float(data.qpos[15]),
                ]
            )
            mj_hip_tau_inv.append(
                [
                    float(data.qfrc_inverse[8]),
                    float(data.qfrc_inverse[10]),
                    float(data.qfrc_inverse[12]),
                    float(data.qfrc_inverse[14]),
                ]
            )
            mj_hip_tau_smooth.append(
                [
                    float(tau_no_con[8]),
                    float(tau_no_con[10]),
                    float(tau_no_con[12]),
                    float(tau_no_con[14]),
                ]
            )
            active_geoms = {int(data.contact[c].geom1) for c in range(data.ncon)} | {
                int(data.contact[c].geom2) for c in range(data.ncon)
            }
            mj_foot_con.append(
                [
                    1 if foot_geom_ids[lb] in active_geoms else 0
                    for lb in ("HL", "HR", "FL", "FR")
                ]
            )

        seq = kps_seq[25:]
        mj_sequences.append((gt_cls, seq))
        res = solve_rat_v2_ik_sequence(seq, dt=0.02)
        inf = res["spinal_circuit_inference"]
        q_ik = np.asarray([fr["qpos"] for fr in res["frames"]], dtype=np.float64)
        q_mj = np.asarray(mj_hip_q[25:], dtype=np.float64)
        ik_corrs = [
            abs(float(np.corrcoef(q_ik[:, 21], q_mj[:, 0])[0, 1])),
            abs(float(np.corrcoef(q_ik[:, 27], q_mj[:, 1])[0, 1])),
            abs(float(np.corrcoef(q_ik[:, 4], q_mj[:, 2])[0, 1])),
            abs(float(np.corrcoef(q_ik[:, 12], q_mj[:, 3])[0, 1])),
        ]
        tau_ik = np.asarray([fr["torque"] for fr in res["frames"]], dtype=np.float64)
        tau_ik_l = np.column_stack(
            [tau_ik[:, 21], tau_ik[:, 27], tau_ik[:, 4], tau_ik[:, 12]]
        )
        tau_inv = np.asarray(mj_hip_tau_inv[25:], dtype=np.float64)
        tau_sm = np.asarray(mj_hip_tau_smooth[25:], dtype=np.float64)
        fcon = np.asarray(mj_foot_con[25:], dtype=int)

        r_no_grf = float(
            np.mean(
                [
                    abs(float(np.corrcoef(tau_ik_l[:, c], tau_sm[:, c])[0, 1]))
                    for c in range(4)
                ]
            )
        )
        r_full_mj = float(
            np.mean(
                [
                    abs(float(np.corrcoef(tau_ik_l[:, c], tau_inv[:, c])[0, 1]))
                    for c in range(4)
                ]
            )
        )
        rms_sw = (
            float(np.sqrt(np.mean(np.square(tau_inv[fcon == 0]))))
            if np.any(fcon == 0)
            else 0.0
        )
        rms_st = (
            float(np.sqrt(np.mean(np.square(tau_inv[fcon == 1]))))
            if np.any(fcon == 1)
            else 0.0
        )
        rms_ik = float(np.sqrt(np.mean(np.square(tau_ik_l))))
        paw_z_ptp_mm = float(np.ptp([fr["right_forepaw"][2] for fr in seq]) * 1000.0)
        paw_x_ptp_mm = float(
            np.ptp([fr["right_forepaw"][0] - fr["torso_center"][0] for fr in seq])
            * 1000.0
        )
        per_regime_summary.append(
            {
                "ground_truth_phenotype": gt_cls,
                "predicted_phenotype": inf["phenotype_class"],
                "matched": inf["phenotype_class"] == gt_cls,
                "mean_speed_mps": inf["mean_speed_mps"],
                "paw_lift_mm": round(paw_z_ptp_mm, 1),
                "paw_stride_mm": round(paw_x_ptp_mm, 1),
                "mujoco_contact_duty": round(float(np.mean(fcon)), 3),
                "sagittal_ik_corr": round(float(np.mean(ik_corrs)), 4),
                "joint_limit_violations": res["summary"]["joint_limit_violations"],
                "torque_corr_no_grf": round(r_no_grf, 4),
                "torque_corr_full_mujoco": round(r_full_mj, 4),
                "mujoco_swing_torque_rms_nm": round(rms_sw, 4),
                "mujoco_stance_torque_rms_nm": round(rms_st, 4),
                "decoupled_kkt_torque_rms_nm": round(rms_ik, 4),
                "kkt_to_mujoco_ratio": round(
                    rms_ik / max(0.5 * (rms_sw + rms_st), 1e-6), 2
                ),
            }
        )

    # Multi-seed noisy out-of-model MuJoCo blind phenotype transfer (5 regimes x 5 seeds = 25 trials per noise level)
    noisy_transfer: List[Dict[str, Any]] = []
    for sigma_mm in (0.0, 2.0, 4.0):
        rng_m = np.random.default_rng(4400 + int(sigma_mm * 10))
        hits = []
        for gt_cls, seq in mj_sequences:
            for _ in range(5):
                if sigma_mm > 0.0:
                    noisy = [
                        {
                            k: (
                                np.asarray(v, dtype=np.float64)
                                + rng_m.normal(0.0, sigma_mm * 0.001, size=3)
                            ).tolist()
                            for k, v in frame.items()
                        }
                        for frame in seq
                    ]
                else:
                    noisy = seq
                pred = infer_spinal_circuit_phenotype(noisy, dt=0.02)["phenotype_class"]
                hits.append(1.0 if pred == gt_cls else 0.0)
        acc, lo, hi = _bootstrap_ci([h * 100.0 for h in hits])
        noisy_transfer.append(
            {
                "sigma_mm": sigma_mm,
                "n_trials": len(hits),
                "accuracy_pct": round(acc, 2),
                "ci95": [round(lo, 2), round(hi, 2)],
            }
        )

    return {
        "mujoco_version": str(mujoco.__version__),
        "total_body_mass_kg": round(float(np.sum(model.body_mass)), 4),
        "per_regime": per_regime_summary,
        "noisy_blind_transfer": noisy_transfer,
    }


_INDEPENDENT_MOUSE_MJCF = """
<mujoco model="independent_mus_musculus_25g">
  <compiler angle="radian"/>
  <option timestep="0.001" gravity="0 0 -9.81" integrator="implicitfast"/>
  <default>
    <joint damping="0.0002" armature="0.00002"/>
    <geom friction="0.9 0.01 0.001" solref="0.012 1.0"/>
  </default>
  <worldbody>
    <geom name="floor" type="plane" size="20 20 0.05" pos="0 0 0"/>
    <body name="torso" pos="0 0 0.0196">
      <freejoint name="root"/>
      <geom type="capsule" fromto="-0.0147 0 0 0.0158 0 0.0021" size="0.0063" mass="0.0115" contype="0" conaffinity="0"/>
      <site name="SpineM" pos="0 0 0"/>
      <site name="SpineF" pos="0.0158 0 0.0021"/>
      <site name="neck" pos="0.0287 0 0.0049"/>
      <body name="head_b" pos="0.0287 0 0.0049">
        <geom type="capsule" fromto="0 0 0 0.021 0 0" size="0.0045" mass="0.0032" contype="0" conaffinity="0"/>
        <site name="HeadB" pos="0.0105 0 0.0014"/>
        <site name="HeadF" pos="0.0221 0 0.0"/>
        <site name="HeadL" pos="0.0084 0.0056 0.0063"/>
        <site name="HeadR" pos="0.0084 -0.0056 0.0063"/>
      </body>
      <body name="lumbar_b" pos="-0.0147 0 0">
        <joint name="lp" type="hinge" axis="0 1 0" range="-0.45 0.45"/>
        <joint name="ly" type="hinge" axis="0 0 1" range="-0.45 0.45"/>
        <geom type="capsule" fromto="0 0 0 -0.0161 0 -0.0007" size="0.006" mass="0.0053" contype="0" conaffinity="0"/>
        <site name="SpineL" pos="0 0 0"/>
        <site name="Pelvis" pos="-0.0161 0 -0.0007"/>
        <site name="TailBase" pos="-0.0259 0 -0.0014"/>
        <body name="hl_t" pos="-0.0105 0.0098 -0.0014">
          <joint name="hl_h" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.0098" size="0.0021" mass="0.0008" contype="0" conaffinity="0"/>
          <body name="hl_s" pos="0 0 -0.0098">
            <joint name="hl_k" type="hinge" axis="0 1 0" range="-1.3 1.3"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.0084" size="0.0018" mass="0.0005" contype="0" conaffinity="0"/>
            <geom name="hl_f" type="sphere" pos="0 0 -0.0084" size="0.0018" mass="0.0002"/>
            <site name="FootL" pos="0 0 -0.0084"/>
          </body>
        </body>
        <body name="hr_t" pos="-0.0105 -0.0098 -0.0014">
          <joint name="hr_h" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.0098" size="0.0021" mass="0.0008" contype="0" conaffinity="0"/>
          <body name="hr_s" pos="0 0 -0.0098">
            <joint name="hr_k" type="hinge" axis="0 1 0" range="-1.3 1.3"/>
            <geom type="capsule" fromto="0 0 0 0 0 -0.0084" size="0.0018" mass="0.0005" contype="0" conaffinity="0"/>
            <geom name="hr_f" type="sphere" pos="0 0 -0.0084" size="0.0018" mass="0.0002"/>
            <site name="FootR" pos="0 0 -0.0084"/>
          </body>
        </body>
      </body>
      <body name="fl_u" pos="0.0196 0.0084 -0.0014">
        <joint name="fl_s" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.0091" size="0.0021" mass="0.0007" contype="0" conaffinity="0"/>
        <body name="fl_l" pos="0 0 -0.0091">
          <joint name="fl_e" type="hinge" axis="0 1 0" range="-1.3 1.3"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.0084" size="0.0018" mass="0.0004" contype="0" conaffinity="0"/>
          <geom name="fl_f" type="sphere" pos="0 0 -0.0084" size="0.0018" mass="0.0002"/>
          <site name="HandL" pos="0 0 -0.0084"/>
        </body>
      </body>
      <body name="fr_u" pos="0.0196 -0.0084 -0.0014">
        <joint name="fr_s" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -0.0091" size="0.0021" mass="0.0007" contype="0" conaffinity="0"/>
        <body name="fr_l" pos="0 0 -0.0091">
          <joint name="fr_e" type="hinge" axis="0 1 0" range="-1.3 1.3"/>
          <geom type="capsule" fromto="0 0 0 0 0 -0.0084" size="0.0018" mass="0.0004" contype="0" conaffinity="0"/>
          <geom name="fr_f" type="sphere" pos="0 0 -0.0084" size="0.0018" mass="0.0002"/>
          <site name="HandR" pos="0 0 -0.0084"/>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="a_lp" joint="lp" kp="0.015" kv="0.0008"/>
    <position name="a_ly" joint="ly" kp="0.015" kv="0.0008"/>
    <position name="a_hl_h" joint="hl_h" kp="0.025" kv="0.0004"/>
    <position name="a_hl_k" joint="hl_k" kp="0.022" kv="0.00035"/>
    <position name="a_hr_h" joint="hr_h" kp="0.025" kv="0.0004"/>
    <position name="a_hr_k" joint="hr_k" kp="0.022" kv="0.00035"/>
    <position name="a_fl_s" joint="fl_s" kp="0.022" kv="0.00035"/>
    <position name="a_fl_e" joint="fl_e" kp="0.022" kv="0.00035"/>
    <position name="a_fr_s" joint="fr_s" kp="0.022" kv="0.00035"/>
    <position name="a_fr_e" joint="fr_e" kp="0.022" kv="0.00035"/>
  </actuator>
</mujoco>
"""


def _run_cross_species_mouse_mujoco_validation() -> Dict[str, Any]:
    """Cross-species 25 g Mus musculus MuJoCo 3.5.0 validation (Bellardita & Kiehn 2015 cadences, DANNCE mm aliases, 8% paw occlusion)."""
    print(
        "[Exp 3D] Running cross-species 25g Mus musculus MuJoCo 3.5.0 + DANNCE mm validation..."
    )
    model = mujoco.MjModel.from_xml_string(_INDEPENDENT_MOUSE_MJCF)
    data = mujoco.MjData(model)
    mass = float(np.sum(model.body_mass))
    snames = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        for i in range(model.nsite)
    ]
    mouse_regimes = [
        (
            "intact_walk",
            2.7,
            {"FR": 0.0, "HL": 0.75, "FL": 0.50, "HR": 0.25},
            0.09,
            0.74,
        ),
        (
            "intact_trot",
            5.9,
            {"FR": 0.0, "HL": 0.00, "FL": 0.50, "HR": 0.50},
            0.28,
            0.58,
        ),
        (
            "v0_commissural_deficit_phenotype",
            5.9,
            {"FR": 0.0, "HL": 0.50, "FL": 0.00, "HR": 0.50},
            0.28,
            0.52,
        ),
        (
            "intact_bound",
            9.8,
            {"FR": 0.0, "HL": 0.50, "FL": 0.00, "HR": 0.50},
            0.52,
            0.46,
        ),
        (
            "v3_synchrony_deficit_phenotype",
            9.8,
            {"FR": 0.0, "HL": 0.00, "FL": 0.50, "HR": 0.50},
            0.52,
            0.50,
        ),
    ]
    per_regime: List[Dict[str, Any]] = []
    clean_mm_seqs: List[Tuple[str, List[Dict[str, List[float]]]]] = []
    for gt_cls, f_hz, offs, spd, duty in mouse_regimes:
        mujoco.mj_resetData(model, data)
        data.qpos[2] = 0.0196
        data.qvel[0] = spd
        seq_mm: List[Dict[str, List[float]]] = []
        mj_q: List[List[float]] = []
        for f in range(180):
            t = f * 0.01
            for s in range(10):
                ts = t + s * 0.001
                phi = {lb: (f_hz * ts + ph0) % 1.0 for lb, ph0 in offs.items()}

                def _lc(u: float) -> Tuple[float, float]:
                    sw = 1.0 - duty
                    if u < sw:
                        su = u / max(sw, 1e-4)
                        return 0.52 * math.cos(math.pi * su), 1.15 * math.sin(
                            math.pi * su
                        )
                    st = (u - sw) / max(duty, 1e-4)
                    return -0.52 * math.cos(math.pi * st), -0.05

                hl_h, hl_k = _lc(phi["HL"])
                hr_h, hr_k = _lc(phi["HR"])
                fl_h, fl_k = _lc(phi["FL"])
                fr_h, fr_k = _lc(phi["FR"])
                data.ctrl[:] = [
                    0.12
                    * (
                        math.sin(2.0 * math.pi * phi["FL"])
                        - math.sin(2.0 * math.pi * phi["HL"])
                    ),
                    0.08
                    * (
                        math.sin(2.0 * math.pi * phi["FL"])
                        - math.sin(2.0 * math.pi * phi["FR"])
                    ),
                    hl_h,
                    hl_k,
                    hr_h,
                    hr_k,
                    fl_h,
                    fl_k,
                    fr_h,
                    fr_k,
                ]
                data.xfrc_applied[1, 0] = 1.2 * (spd - data.qvel[0])
                data.xfrc_applied[1, 1] = -0.5 * data.qpos[1] - 0.08 * data.qvel[1]
                data.xfrc_applied[1, 2] = (
                    0.72 * mass * 9.81
                    + 8.0 * (0.0196 - data.qpos[2])
                    - 0.25 * data.qvel[2]
                )
                data.xfrc_applied[1, 3:6] = (
                    -0.008 * data.qpos[4:7] - 0.0006 * data.qvel[3:6]
                )
                mujoco.mj_step(model, data)
            if f >= 30:
                seq_mm.append(
                    {
                        nm: (data.site_xpos[i] * 1000.0).tolist()
                        for i, nm in enumerate(snames)
                    }
                )
                mj_q.append(
                    [
                        float(data.qpos[9]),
                        float(data.qpos[11]),
                        float(data.qpos[13]),
                        float(data.qpos[15]),
                    ]
                )
        clean_mm_seqs.append((gt_cls, seq_mm))
        res = solve_rat_v2_ik_sequence(seq_mm, dt=0.01)
        inf = res["spinal_circuit_inference"]
        q_ik = np.asarray([fr["qpos"] for fr in res["frames"]], dtype=np.float64)
        q_mj = np.asarray(mj_q, dtype=np.float64)
        ik_r = float(
            np.mean(
                [
                    abs(float(np.corrcoef(q_ik[:, 21], q_mj[:, 0])[0, 1])),
                    abs(float(np.corrcoef(q_ik[:, 27], q_mj[:, 1])[0, 1])),
                    abs(float(np.corrcoef(q_ik[:, 4], q_mj[:, 2])[0, 1])),
                    abs(float(np.corrcoef(q_ik[:, 12], q_mj[:, 3])[0, 1])),
                ]
            )
        )
        per_regime.append(
            {
                "ground_truth_phenotype": gt_cls,
                "predicted_phenotype": inf["phenotype_class"],
                "matched": inf["phenotype_class"] == gt_cls,
                "detected_unit": inf["detected_unit"],
                "detected_body_scale": inf["detected_body_scale"],
                "mean_speed_mps": inf["mean_speed_mps"],
                "froude_equiv_speed_mps": inf["froude_equiv_speed_mps"],
                "step_frequency_hz": inf["step_frequency_hz"],
                "sagittal_ik_corr": round(ik_r, 4),
                "joint_limit_violations": res["summary"]["joint_limit_violations"],
            }
        )

    # Noisy + 8% paw occlusion DANNCE mm transfer across 25 trials per noise level (0, 0.7, 1.4 mm — proportional to 0, 2, 4 mm on rat!)
    transfer_rows: List[Dict[str, Any]] = []
    for sig_mm in (0.0, 0.7, 1.4):
        rng_m = np.random.default_rng(6600 + int(sig_mm * 100))
        hits = []
        for gt_cls, seq_mm in clean_mm_seqs:
            for _ in range(5):
                corrupted = []
                for frame in seq_mm:
                    fr_out = {}
                    for nm, pt in frame.items():
                        if (
                            sig_mm > 0.0
                            and rng_m.random() < 0.08
                            and nm
                            in (
                                "HandL",
                                "HandR",
                                "FootL",
                                "FootR",
                            )
                        ):
                            continue
                        noise = (
                            rng_m.normal(0.0, sig_mm, size=3)
                            if sig_mm > 0.0
                            else np.zeros(3)
                        )
                        fr_out[nm] = (np.asarray(pt, dtype=np.float64) + noise).tolist()
                    corrupted.append(fr_out)
                pred = infer_spinal_circuit_phenotype(corrupted, dt=0.01)[
                    "phenotype_class"
                ]
                hits.append(100.0 if pred == gt_cls else 0.0)
        acc, lo, hi = _bootstrap_ci(hits)
        transfer_rows.append(
            {
                "sigma_mm": sig_mm,
                "rat_equiv_sigma_mm": round(sig_mm / 0.35, 1),
                "paw_dropout_pct": 8.0 if sig_mm > 0.0 else 0.0,
                "n_trials": len(hits),
                "accuracy_pct": round(acc, 2),
                "ci95": [round(lo, 2), round(hi, 2)],
            }
        )

    return {
        "species": "Mus musculus (25g MuJoCo 3.5.0 + DANNCE mm aliases)",
        "total_body_mass_g": round(mass * 1000.0, 1),
        "per_regime": per_regime,
        "noisy_occluded_transfer": transfer_rows,
    }


def _naive_single_anchor_phenotype(
    frames_kps: Sequence[Dict[str, List[float]]], dt: float = 0.02
) -> str:
    """Ablation baseline: single-point torso_center anchor + raw un-bandpassed sagittal phase (all 6 classes)."""
    roots = []
    sigs: Dict[str, List[float]] = {k: [] for k in ("FL", "FR", "HL", "HR")}
    for kps in frames_kps:
        nk = normalize_rat_keypoint_dict(kps)
        root = nk["torso_center"]
        roots.append(root)
        diff = nk["head"] - nk["pelvis"]
        yaw = math.atan2(float(diff[1]), float(diff[0]))
        cy, sy = math.cos(-yaw), math.sin(-yaw)
        for limb, lbl, bx in (
            ("FL", "left_forepaw", 0.056),
            ("FR", "right_forepaw", 0.056),
            ("HL", "left_hindpaw", -0.072),
            ("HR", "right_hindpaw", -0.072),
        ):
            d = nk[lbl] - root
            lx = d[0] * cy - d[1] * sy - bx
            sigs[limb].append(float(lx))
    roots_arr = np.asarray(roots, dtype=np.float64)
    spd = float(
        np.median(np.linalg.norm(np.diff(roots_arr[:, :2], axis=0), axis=1) / dt)
    )
    if spd < 0.05:
        return "intact_stand"

    def _raw_phase(a_in: List[float], r_in: List[float]) -> float:
        a = np.asarray(a_in) - np.mean(a_in)
        r = np.asarray(r_in) - np.mean(r_in)
        fa, fr = np.fft.fft(a), np.fft.fft(r)
        h = np.zeros(a.size)
        h[0] = 1.0
        h[1 : a.size // 2] = 2.0
        za, zr = np.fft.ifft(fa * h), np.fft.ifft(fr * h)
        ph = np.mean(np.exp(1j * (np.angle(za) - np.angle(zr))))
        return float((np.angle(ph) / (2.0 * math.pi)) % 1.0)

    p_fl = _raw_phase(sigs["FL"], sigs["FR"])
    p_hl = _raw_phase(sigs["HL"], sigs["FR"])
    p_hr = _raw_phase(sigs["HR"], sigs["FR"])

    duty = float(np.clip(0.80 - 0.36 * min(spd, 1.0), 0.42, 0.85))
    gait = classify_quadruped_gait(
        speed_mps=spd,
        phase_offsets_rel_fr={"FR": 0.0, "FL": p_fl, "HL": p_hl, "HR": p_hr},
        stance_fraction=duty,
    )
    if gait == "bound" and 0.16 <= spd <= 0.66:
        return "v0_commissural_deficit_phenotype"
    if gait in ("walk", "trot") and spd >= 0.72:
        return "v3_synchrony_deficit_phenotype"
    return f"intact_{gait}"


def _class_f1(y_true: Sequence[str], y_pred: Sequence[str], target_class: str) -> float:
    tp = sum(
        1 for yt, yp in zip(y_true, y_pred) if yt == target_class and yp == target_class
    )
    fp = sum(
        1 for yt, yp in zip(y_true, y_pred) if yt != target_class and yp == target_class
    )
    fn = sum(
        1 for yt, yp in zip(y_true, y_pred) if yt == target_class and yp != target_class
    )
    return (2.0 * tp) / max(2.0 * tp + fp + fn, 1.0)


def run_experiment_3_real2sim_inversion() -> Dict[str, Any]:
    print(
        "[Exp 3] Running Real-to-Sim 3D IK, KKT Spring-Muscle ID & 180-trial 6-class blind ID..."
    )
    # 3A: IK + KKT Spring-Muscle ID accuracy across observation noise
    r_ref = run_rat_ncap_rollout(
        steps=140,
        behavior="trot",
        target_speed=0.50,
        interface="flexor_extensor",
        seed=7,
    )
    frames_ref = r_ref["frames"][20:]
    q_gt = np.asarray([f["qpos"] for f in frames_ref], dtype=np.float64)
    id_gt_exact = invert_spring_muscle_activations(q_seq=q_gt, dt=0.02)

    ik_noise_sweep: List[Dict[str, Any]] = []
    for sigma_mm in (0.0, 1.0, 2.0, 4.0, 6.0, 8.0):
        rng = np.random.default_rng(5000 + int(sigma_mm * 10))
        noisy_seq = [
            {
                k: (
                    np.asarray(v, dtype=np.float64)
                    + (
                        rng.normal(0.0, sigma_mm * 0.001, size=3)
                        if sigma_mm > 0
                        else 0.0
                    )
                ).tolist()
                for k, v in f["keypoints_3d"].items()
            }
            for f in frames_ref
        ]
        res = solve_rat_v2_ik_sequence(
            noisy_seq, dt=0.02, interface_type="flexor_extensor"
        )
        q_hat = np.asarray([f["qpos"] for f in res["frames"]], dtype=np.float64)
        sag_indices = (4, 12, 21, 27)
        corrs = [
            float(np.corrcoef(q_gt[:, idx], q_hat[:, idx])[0, 1]) for idx in sag_indices
        ]
        sag_rmse_deg = float(
            np.rad2deg(
                np.sqrt(
                    np.mean(np.square(q_gt[:, sag_indices] - q_hat[:, sag_indices]))
                )
            )
        )
        ik_noise_sweep.append(
            {
                "sigma_mm": sigma_mm,
                "ik_rmse_mm": round(res["summary"]["mean_ik_rmse_m"] * 1000.0, 3),
                "sagittal_joint_corr": round(float(np.mean(corrs)), 4),
                "sagittal_joint_rmse_deg": round(sag_rmse_deg, 3),
                "torque_residual_rmse_nm": res["inverse_dynamics"][
                    "torque_residual_rmse_nm"
                ],
                "joint_limit_violations": res["summary"]["joint_limit_violations"],
                "predicted_gait": res["spinal_circuit_inference"]["dominant_gait"],
                "v0_integrity_index": res["spinal_circuit_inference"][
                    "v0_integrity_index"
                ],
            }
        )

    # 3B: 180-trial 6-class blind phenotype classification (including BOTH V0 & V3 speed-matched deficits!)
    classes = [
        "intact_stand",
        "intact_walk",
        "intact_trot",
        "intact_bound",
        "v0_commissural_deficit_phenotype",
        "v3_synchrony_deficit_phenotype",
    ]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    trial_specs: List[Tuple[str, List[Dict[str, List[float]]]]] = []
    rng_tr = np.random.default_rng(2026)
    for cls_name in classes:
        for t_idx in range(30):
            seed = 200 + t_idx
            if cls_name == "intact_stand":
                spd = float(rng_tr.uniform(0.0, 0.02))
                r = run_rat_ncap_rollout(
                    steps=120, behavior="stand", target_speed=spd, seed=seed
                )
            elif cls_name == "intact_walk":
                spd = float(rng_tr.uniform(0.10, 0.20))
                r = run_rat_ncap_rollout(
                    steps=130, behavior="walk", target_speed=spd, seed=seed
                )
            elif cls_name == "intact_trot":
                spd = float(rng_tr.uniform(0.36, 0.58))
                r = run_rat_ncap_rollout(
                    steps=130, behavior="trot", target_speed=spd, seed=seed
                )
            elif cls_name == "intact_bound":
                spd = float(rng_tr.uniform(0.84, 0.96))
                r = run_rat_ncap_rollout(
                    steps=130, behavior="bound", target_speed=spd, seed=seed
                )
            elif cls_name == "v0_commissural_deficit_phenotype":
                spd = float(rng_tr.uniform(0.32, 0.58))
                r = run_rat_ncap_rollout(
                    steps=130,
                    behavior="trot",
                    target_speed=spd,
                    ablation="v0v_v0d",
                    ablation_scale=0.0,
                    seed=seed,
                )
            else:  # v3_synchrony_deficit_phenotype OVERLAPPING intact_bound speeds!
                spd = float(rng_tr.uniform(0.84, 0.96))
                r = run_rat_ncap_rollout(
                    steps=130,
                    behavior="trot",
                    target_speed=spd,
                    ablation="v3f_v3a",
                    ablation_scale=0.0,
                    seed=seed,
                )
            trial_specs.append(
                (cls_name, [f["keypoints_3d"] for f in r["frames"][30:]])
            )

    noise_eval_curve: List[Dict[str, Any]] = []
    confusion_at_4mm = np.zeros((len(classes), len(classes)), dtype=int)
    v0_scores_intact_4mm: List[float] = []
    v0_scores_ablated_4mm: List[float] = []
    v3_scores_intact_4mm: List[float] = []
    v3_scores_ablated_4mm: List[float] = []

    for sigma_mm in (0.0, 2.0, 4.0, 6.0, 8.0, 10.0):
        rng_n = np.random.default_rng(9000 + int(sigma_mm * 10))
        y_true: List[str] = []
        y_ours: List[str] = []
        y_naive: List[str] = []

        for gt_cls, kps_clean in trial_specs:
            if sigma_mm > 0.0:
                noisy = [
                    {
                        k: (
                            np.asarray(v, dtype=np.float64)
                            + rng_n.normal(0.0, sigma_mm * 0.001, size=3)
                        ).tolist()
                        for k, v in frame.items()
                    }
                    for frame in kps_clean
                ]
            else:
                noisy = kps_clean

            inf = infer_spinal_circuit_phenotype(noisy, dt=0.02)
            pred_ours = str(inf["phenotype_class"])
            pred_naive = _naive_single_anchor_phenotype(noisy, dt=0.02)
            y_true.append(gt_cls)
            y_ours.append(pred_ours)
            y_naive.append(pred_naive)

            if sigma_mm == 4.0:
                if pred_ours in class_to_idx:
                    confusion_at_4mm[class_to_idx[gt_cls], class_to_idx[pred_ours]] += 1
                if gt_cls == "intact_trot":
                    v0_scores_intact_4mm.append(float(inf["v0_integrity_index"]))
                elif gt_cls == "v0_commissural_deficit_phenotype":
                    v0_scores_ablated_4mm.append(float(inf["v0_integrity_index"]))
                elif gt_cls == "intact_bound":
                    v3_scores_intact_4mm.append(float(inf["v3_integrity_index"]))
                elif gt_cls == "v3_synchrony_deficit_phenotype":
                    v3_scores_ablated_4mm.append(float(inf["v3_integrity_index"]))

        corr_ours = [yt == yp for yt, yp in zip(y_true, y_ours)]
        corr_naive = [yt == yp for yt, yp in zip(y_true, y_naive)]
        acc_o, lo_o, hi_o = _bootstrap_ci([100.0 if c else 0.0 for c in corr_ours])
        acc_n, lo_n, hi_n = _bootstrap_ci([100.0 if c else 0.0 for c in corr_naive])
        p_mcnemar = _mcnemar_exact_p(corr_ours, corr_naive)
        per_cls_f1 = {c: _class_f1(y_true, y_ours, c) for c in classes}
        macro_f1 = float(np.mean(list(per_cls_f1.values())))
        walk_recall = (
            sum(1 for yt, yp in zip(y_true, y_ours) if yt == "intact_walk" and yp == yt)
            / 30.0
        )

        noise_eval_curve.append(
            {
                "sigma_mm": sigma_mm,
                "n_trials": len(trial_specs),
                "ours_accuracy_pct": round(acc_o, 2),
                "ours_ci95": [round(lo_o, 2), round(hi_o, 2)],
                "naive_accuracy_pct": round(acc_n, 2),
                "naive_ci95": [round(lo_n, 2), round(hi_n, 2)],
                "mcnemar_p_ours_vs_naive": float(f"{p_mcnemar:.3e}"),
                "macro_f1": round(macro_f1, 4),
                "walk_recall_pct": round(walk_recall * 100.0, 2),
                "v0_deficit_f1": round(
                    per_cls_f1["v0_commissural_deficit_phenotype"], 4
                ),
                "v3_deficit_f1": round(per_cls_f1["v3_synchrony_deficit_phenotype"], 4),
            }
        )

    _, p_v0_mw = scipy_stats.mannwhitneyu(
        v0_scores_intact_4mm, v0_scores_ablated_4mm, alternative="greater"
    )
    _, p_v3_mw = scipy_stats.mannwhitneyu(
        v3_scores_intact_4mm, v3_scores_ablated_4mm, alternative="greater"
    )

    # 3C: Independent Rat MuJoCo 3.5.0 Multi-Body Cross-Validation
    mujoco_xval = _run_independent_mujoco_cross_validation()
    # 3D: Cross-Species 25g Mouse MuJoCo 3.5.0 + DANNCE mm Schema + Froude Similarity Validation
    mouse_mujoco_xval = _run_cross_species_mouse_mujoco_validation()

    return {
        "exact_kkt_torque_residual_nm": id_gt_exact["torque_residual_rmse_nm"],
        "ik_noise_sweep": ik_noise_sweep,
        "phenotype_classes": classes,
        "blind_phenotype_noise_curve": noise_eval_curve,
        "confusion_matrix_at_4mm": confusion_at_4mm.tolist(),
        "v0_integrity_intact_trot_4mm_mean": round(
            float(np.mean(v0_scores_intact_4mm)), 4
        ),
        "v0_integrity_ablated_4mm_mean": round(
            float(np.mean(v0_scores_ablated_4mm)), 4
        ),
        "v0_integrity_mannwhitney_p": float(f"{float(p_v0_mw):.3e}"),
        "v3_integrity_intact_bound_4mm_mean": round(
            float(np.mean(v3_scores_intact_4mm)), 4
        ),
        "v3_integrity_ablated_4mm_mean": round(
            float(np.mean(v3_scores_ablated_4mm)), 4
        ),
        "v3_integrity_mannwhitney_p": float(f"{float(p_v3_mw):.3e}"),
        "mujoco_cross_validation": mujoco_xval,
        "mouse_mujoco_cross_species": mouse_mujoco_xval,
    }


# ---------------------------------------------------------------------------
# Experiment 4: Drosophila Connectome, Motor Outputs, Spiking Plasticity & Scaling
# ---------------------------------------------------------------------------


def _run_mb_conditioning_protocol(
    walk_spec: SimpleNamespace,
    *,
    dynamics: str,
) -> List[Dict[str, Any]]:
    is_spk = dynamics == "spiking"
    mb_graph = build_plastic_fly_connectome(spiking=is_spk, seed=7)
    ctrl_mb = FlyBrainController(
        walk_spec,
        behavior="walk_imitation",
        connectome=mb_graph,
        dynamics=dynamics,
        seed=7,
        enable_plasticity=True,
        suppress_behavior_drive=True,
    )
    learning_trace: List[Dict[str, Any]] = []
    step_idx = 0

    def _record_probe(phase_label: str) -> None:
        saved_v = ctrl_mb._voltage.copy()
        saved_r = ctrl_mb._rates.copy()
        saved_g = ctrl_mb._syn_conductance.copy()
        saved_ref = ctrl_mb._refractory_rem.copy()
        saved_elig = dict(ctrl_mb._plastic_eligibility)
        for _ in range(38):
            ctrl_mb.step({"walker/touch": np.ones(6), "odor_a": 1.0}, dt=0.004)
        val_a = ctrl_mb.rate("DNp09_L") - ctrl_mb.rate("MDN")
        ctrl_mb._voltage = saved_v.copy()
        ctrl_mb._rates = saved_r.copy()
        ctrl_mb._syn_conductance = saved_g.copy()
        ctrl_mb._refractory_rem = saved_ref.copy()
        for _ in range(38):
            ctrl_mb.step({"walker/touch": np.ones(6), "odor_b": 1.0}, dt=0.004)
        val_b = ctrl_mb.rate("DNp09_L") - ctrl_mb.rate("MDN")
        ctrl_mb._voltage = saved_v
        ctrl_mb._rates = saved_r
        ctrl_mb._syn_conductance = saved_g
        ctrl_mb._refractory_rem = saved_ref
        ctrl_mb._plastic_eligibility = saved_elig
        w = ctrl_mb.plasticity_weights()
        learning_trace.append(
            {
                "trial": step_idx,
                "phase": phase_label,
                "w_a_approach": w["odor_a_approach"],
                "w_a_avoid": w["odor_a_avoid"],
                "w_b_approach": w["odor_b_approach"],
                "w_b_avoid": w["odor_b_avoid"],
                "odor_a_motor_valence": round(float(val_a), 4),
                "odor_b_motor_valence": round(float(val_b), 4),
            }
        )

    _record_probe("baseline")
    for _ in range(8):
        step_idx += 1
        for _ in range(25):
            ctrl_mb.step(
                {"walker/touch": np.ones(6), "odor_a": 1.0, "reward_da": 1.0}, dt=0.004
            )
        for _ in range(75):
            ctrl_mb.step({"walker/touch": np.ones(6)}, dt=0.004)
        for _ in range(25):
            ctrl_mb.step(
                {"walker/touch": np.ones(6), "odor_b": 1.0, "punish_da": 1.0}, dt=0.004
            )
        for _ in range(75):
            ctrl_mb.step({"walker/touch": np.ones(6)}, dt=0.004)
        _record_probe("acquisition")

    for _ in range(12):
        step_idx += 1
        for _ in range(25):
            ctrl_mb.step(
                {"walker/touch": np.ones(6), "odor_a": 1.0, "punish_da": 1.0}, dt=0.004
            )
        for _ in range(75):
            ctrl_mb.step({"walker/touch": np.ones(6)}, dt=0.004)
        for _ in range(25):
            ctrl_mb.step(
                {"walker/touch": np.ones(6), "odor_b": 1.0, "reward_da": 1.0}, dt=0.004
            )
        for _ in range(75):
            ctrl_mb.step({"walker/touch": np.ones(6)}, dt=0.004)
        _record_probe("reversal")

    return learning_trace


def run_experiment_4_drosophila_connectome() -> Dict[str, Any]:
    print(
        "[Exp 4] Running Drosophila connectome perturbations, motor readouts, LIF plasticity & MaleCNS scaling..."
    )
    walk_spec = _walking_spec()
    flight_spec = _flight_spec()
    walk_act_idx = {n: i for i, n in enumerate(walk_spec.name.split("\t"))}
    flight_act_idx = {n: i for i, n in enumerate(flight_spec.name.split("\t"))}
    obs_grounded = {"walker/touch": np.ones(6, dtype=np.float64)}
    obs_air = {
        "walker/touch": np.zeros(6, dtype=np.float64),
        "walker/world_zaxis": np.array([0.65, 0.0, 0.76]),
        "walker/gyro": np.zeros(3),
    }

    # 4A: Rate vs 1,470-neuron LIF Spiking selectivity + closed-loop motor verification
    behaviors = (
        ("walk_imitation", walk_spec, obs_grounded, "dnp09_forward_walk"),
        ("backward_walk", walk_spec, obs_grounded, "moonwalker_backward"),
        ("grooming", walk_spec, obs_grounded, "antennal_grooming"),
        ("escape_takeoff", walk_spec, obs_grounded, "giant_fiber_escape"),
        ("flight_imitation", flight_spec, obs_air, "flight_tectulum"),
    )
    dn_keys = ("DNp09", "MDN", "Pair1", "aDN1", "DNp01", "DNg02")
    selectivity_rows: List[Dict[str, Any]] = []
    motor_verification: Dict[str, Dict[str, Any]] = {}

    for dyn in ("rate", "spiking"):
        graph = (
            build_spiking_fly_connectome(seed=7)
            if dyn == "spiking"
            else build_canonical_fly_connectome()
        )
        for beh, spec, obs, expected in behaviors:
            ctrl = FlyBrainController(
                spec, behavior=beh, connectome=graph, dynamics=dyn, seed=7
            )
            n_steps = 110 if beh == "escape_takeoff" else 60
            dt = 0.0005 if beh == "flight_imitation" else 0.002
            actions = []
            for _ in range(n_steps):
                actions.append(ctrl.step(obs, dt=dt))
            diag = ctrl.diagnostics()
            row = {
                "dynamics": dyn,
                "behavior": beh,
                "expected_circuit": expected,
                "dominant_circuit": diag["dominant_circuit"],
                "matched": diag["dominant_circuit"] == expected,
                "active_nodes": diag["active_nodes"],
                "synaptic_edges": diag["synaptic_edges"],
                "dn_rates": {k: diag["descending_neurons"][k] for k in dn_keys},
            }
            if "spiking" in diag:
                row["mean_firing_rate_hz"] = diag["spiking"]["mean_firing_rate_hz"]
            selectivity_rows.append(row)

            # Closed-loop motor-output verification in 1,470-neuron LIF Spiking mode
            if dyn == "spiking":
                act_arr = np.asarray(actions[-20:], dtype=np.float64)
                if beh == "walk_imitation":
                    # Tripod anti-phase check between contralateral T1_left and T1_right
                    dphi_deg = abs(
                        math.degrees(
                            math.atan2(
                                math.sin(ctrl._leg_phases[0] - ctrl._leg_phases[1]),
                                math.cos(ctrl._leg_phases[0] - ctrl._leg_phases[1]),
                            )
                        )
                    )
                    motor_verification[beh] = {
                        "tripod_contralateral_phase_deg": round(dphi_deg, 1),
                        "t1_claw_duty": round(
                            float(
                                np.mean(act_arr[:, walk_act_idx["adhere_claw_T1_left"]])
                            ),
                            3,
                        ),
                        "direction_sign": 1.0,
                    }
                elif beh == "backward_walk":
                    motor_verification[beh] = {
                        "vnc_reverse_rate": round(float(ctrl.rate("VNC_Reverse")), 3),
                        "direction_sign": -1.0
                        if ctrl.rate("VNC_Reverse") > 0.45
                        else 1.0,
                    }
                elif beh == "grooming":
                    motor_verification[beh] = {
                        "t1_claw_adhesion": round(
                            float(
                                np.mean(act_arr[:, walk_act_idx["adhere_claw_T1_left"]])
                            ),
                            3,
                        ),
                        "t2_t3_claw_adhesion": round(
                            float(
                                np.mean(
                                    [
                                        act_arr[:, walk_act_idx["adhere_claw_T2_left"]],
                                        act_arr[:, walk_act_idx["adhere_claw_T3_left"]],
                                    ]
                                )
                            ),
                            3,
                        ),
                        "t1_coxa_elevation": round(
                            float(np.mean(act_arr[:, walk_act_idx["coxa_T1_left"]])), 3
                        ),
                    }
                elif beh == "escape_takeoff":
                    last_a = actions[-1]
                    all_claws = float(
                        np.mean(
                            [
                                last_a[walk_act_idx[f"adhere_claw_{lg}"]]
                                for lg in (
                                    "T1_left",
                                    "T1_right",
                                    "T2_left",
                                    "T2_right",
                                    "T3_left",
                                    "T3_right",
                                )
                            ]
                        )
                    )
                    motor_verification[beh] = {
                        "jump_all_claw_adhesion": round(all_claws, 3),
                        "t2_femur_jump_thrust": round(
                            float(last_a[walk_act_idx["femur_T2_left"]]), 3
                        ),
                    }
                elif beh == "flight_imitation":
                    motor_verification[beh] = {
                        "wbpg_freq_trim_user0": round(
                            float(np.mean(act_arr[:, flight_act_idx["user_0"]])), 3
                        ),
                        "wing_pitch_stabilizer": round(
                            float(
                                np.mean(act_arr[:, flight_act_idx["wing_pitch_left"]])
                            ),
                            3,
                        ),
                    }

    # 4B: Optogenetic Chrimson stimulation & Kir2.1 genetic silencing causal table (LIF Spiking)
    spiking_graph = build_spiking_fly_connectome(seed=7)
    causal_tests: List[Dict[str, Any]] = []

    for label, base_beh, stim_target, stim_cur, check_dn, expected_circuit in (
        (
            "Opto MDN (Moonwalk)",
            "walk_imitation",
            "MDN",
            2.4,
            "MDN",
            "moonwalker_backward",
        ),
        (
            "Opto Pair1 (Arrest)",
            "walk_imitation",
            "Pair1",
            2.4,
            "Pair1",
            "pair1_quiet_stance",
        ),
        (
            "Opto aDN1 (Groom)",
            "walk_imitation",
            "aDN1",
            2.4,
            "aDN1",
            "antennal_grooming",
        ),
        (
            "Opto DNp01 (Escape)",
            "walk_imitation",
            "DNp01",
            2.4,
            "DNp01",
            "giant_fiber_escape",
        ),
    ):
        ctrl_b = FlyBrainController(
            walk_spec,
            behavior=base_beh,
            connectome=spiking_graph,
            dynamics="spiking",
            seed=7,
        )
        ctrl_s = FlyBrainController(
            walk_spec,
            behavior=base_beh,
            connectome=spiking_graph,
            dynamics="spiking",
            seed=7,
        )
        for _ in range(25):
            ctrl_b.step(obs_grounded, dt=0.002)
            ctrl_s.step(obs_grounded, dt=0.002)
        ctrl_s.stimulate_cell_type(stim_target, stim_cur, duration_sec=0.14)
        for _ in range(55):
            ctrl_b.step(obs_grounded, dt=0.002)
            ctrl_s.step(obs_grounded, dt=0.002)
        db, ds = ctrl_b.diagnostics(), ctrl_s.diagnostics()
        causal_tests.append(
            {
                "intervention": label,
                "kind": "optogenetics",
                "baseline_circuit": db["dominant_circuit"],
                "perturbed_circuit": ds["dominant_circuit"],
                "expected_circuit": expected_circuit,
                "target_dn": check_dn,
                "dn_before": db["descending_neurons"][check_dn],
                "dn_after": ds["descending_neurons"][check_dn],
                "dnp09_before": db["descending_neurons"]["DNp09"],
                "dnp09_after": ds["descending_neurons"]["DNp09"],
            }
        )

    for label, base_beh, ablate_target, check_dn, extra_opto, touch_val in (
        ("Silence Pair1 (Groom)", "grooming", "Pair1", "Pair1", None, 1.0),
        (
            "Silence VNC_MDN_Inh (Walk+MDN)",
            "walk_imitation",
            "VNC_MDN_Inh",
            "DNp09",
            ("MDN", 2.4),
            1.0,
        ),
        ("Silence DNp01 (Escape)", "escape_takeoff", "DNp01", "DNp01", None, 1.0),
        (
            "Silence AN_Contact (Fraenkel)",
            "walk_imitation",
            "AN_Contact",
            "DNg02",
            ("Flight_Command", 1.15),
            4.0,
        ),
    ):
        obs = {"walker/touch": np.full(6, touch_val, dtype=np.float64)}
        ctrl_b = FlyBrainController(
            walk_spec,
            behavior=base_beh,
            connectome=spiking_graph,
            dynamics="spiking",
            seed=7,
        )
        ctrl_a = FlyBrainController(
            walk_spec,
            behavior=base_beh,
            connectome=spiking_graph,
            dynamics="spiking",
            seed=7,
        )
        ctrl_a.set_ablation(ablate_target, scale=0.0)
        if extra_opto is not None:
            ctrl_b.stimulate_cell_type(extra_opto[0], extra_opto[1], duration_sec=0.25)
            ctrl_a.stimulate_cell_type(extra_opto[0], extra_opto[1], duration_sec=0.25)
        steps_run = 110 if base_beh == "escape_takeoff" else 60
        for _ in range(steps_run):
            ctrl_b.step(obs, dt=0.002)
            ctrl_a.step(obs, dt=0.002)
        db, da = ctrl_b.diagnostics(), ctrl_a.diagnostics()
        causal_tests.append(
            {
                "intervention": label,
                "kind": "silencing",
                "baseline_circuit": db["dominant_circuit"],
                "perturbed_circuit": da["dominant_circuit"],
                "target_dn": check_dn,
                "dn_before": db["descending_neurons"][check_dn],
                "dn_after": da["descending_neurons"][check_dn],
                "dnp09_before": db["descending_neurons"]["DNp09"],
                "dnp09_after": da["descending_neurons"]["DNp09"],
            }
        )

    # 4C: Dale's-law synaptic noise robustness across Rate vs LIF Spiking (15 seeds x 3 behaviors = 45 trials)
    noise_robustness: List[Dict[str, Any]] = []
    for sigma in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        row_n: Dict[str, Any] = {"sigma": sigma, "n_trials": 45}
        hits_by_dyn: Dict[str, List[bool]] = {}
        for dyn in ("rate", "spiking"):
            base_g = (
                build_spiking_fly_connectome(seed=7)
                if dyn == "spiking"
                else build_canonical_fly_connectome()
            )
            hits: List[bool] = []
            for seed in range(15):
                for beh, expected in (
                    ("walk_imitation", "dnp09_forward_walk"),
                    ("backward_walk", "moonwalker_backward"),
                    ("grooming", "antennal_grooming"),
                ):
                    c = FlyBrainController(
                        walk_spec,
                        behavior=beh,
                        connectome=base_g,
                        dynamics=dyn,
                        seed=50 + seed,
                        synaptic_noise_std=sigma,
                    )
                    for _ in range(45):
                        c.step(obs_grounded, dt=0.002)
                    hits.append(c.dominant_circuit() == expected)
            hits_by_dyn[dyn] = hits
            acc, lo, hi = _bootstrap_ci([100.0 if h else 0.0 for h in hits])
            row_n[f"{dyn}_accuracy_pct"] = round(acc, 2)
            row_n[f"{dyn}_ci95"] = [round(lo, 2), round(hi, 2)]
        row_n["mcnemar_p_spiking_vs_rate"] = float(
            f"{_mcnemar_exact_p(hits_by_dyn['spiking'], hits_by_dyn['rate']):.3e}"
        )
        noise_robustness.append(row_n)

    # 4D: Three-Factor PAM/PPL1 Mushroom Body Acquisition + Reversal in BOTH Rate AND 1,534-Neuron LIF Spiking!
    learning_trace_rate = _run_mb_conditioning_protocol(walk_spec, dynamics="rate")
    learning_trace_spiking = _run_mb_conditioning_protocol(
        walk_spec, dynamics="spiking"
    )

    # 4D-2: High-Dimensional 2,000-KC APL-Sparsified Mushroom Body (10 Odors + Noisy Gen + Selective Reversal in 1,534-N LIF!)
    mb_2k_graph = build_plastic_fly_connectome(spiking=True, seed=7)
    ctrl_2k = FlyBrainController(
        walk_spec,
        behavior="walk_imitation",
        connectome=mb_2k_graph,
        dynamics="spiking",
        seed=7,
        enable_plasticity=True,
        suppress_behavior_drive=True,
        sparse_mb_kcs=2000,
        sparse_mb_pns=50,
    )
    rng_2k = np.random.default_rng(42)
    odors_10 = [
        rng_2k.exponential(1.0, size=50) * (rng_2k.random(50) < 0.35) for _ in range(10)
    ]
    t0_2k = time.perf_counter()
    for _ in range(7):
        for oi, pat in enumerate(odors_10):
            da_key = "reward_da" if oi < 5 else "punish_da"
            for _ in range(25):
                ctrl_2k.step(
                    {"walker/touch": np.ones(6), "odor_pattern": pat, da_key: 1.0},
                    dt=0.004,
                )
            for _ in range(60):
                ctrl_2k.step({"walker/touch": np.ones(6)}, dt=0.004)
    train_2k_sec = time.perf_counter() - t0_2k

    def _probe_2k(pat: np.ndarray) -> float:
        sv = ctrl_2k._voltage.copy()
        sr = ctrl_2k._rates.copy()
        sg = ctrl_2k._syn_conductance.copy()
        sref = ctrl_2k._refractory_rem.copy()
        sel = ctrl_2k.sparse_mb.eligibility.copy()
        for _ in range(38):
            ctrl_2k.step({"walker/touch": np.ones(6), "odor_pattern": pat}, dt=0.004)
        val = float(ctrl_2k.rate("DNp09_L") - ctrl_2k.rate("MDN"))
        ctrl_2k._voltage, ctrl_2k._rates = sv, sr
        ctrl_2k._syn_conductance, ctrl_2k._refractory_rem = sg, sref
        ctrl_2k.sparse_mb.eligibility = sel
        return round(val, 4)

    vals_acq_10 = [_probe_2k(p) for p in odors_10]
    acq_10_hits = sum(
        (vals_acq_10[i] > 0.1 if i < 5 else vals_acq_10[i] < -0.1) for i in range(10)
    )
    noisy_10_hits = 0
    for i, pat in enumerate(odors_10):
        v_n = _probe_2k(np.maximum(0.0, pat + rng_2k.normal(0.0, 0.20, size=50)))
        if (v_n > 0.1) if i < 5 else (v_n < -0.1):
            noisy_10_hits += 1
    # Selective single-odor reversal: swap Odor #0 ONLY (CS+ -> CS-) without touching Odors #1..9
    for _ in range(10):
        for _ in range(25):
            ctrl_2k.step(
                {
                    "walker/touch": np.ones(6),
                    "odor_pattern": odors_10[0],
                    "punish_da": 1.0,
                },
                dt=0.004,
            )
        for _ in range(60):
            ctrl_2k.step({"walker/touch": np.ones(6)}, dt=0.004)
    vals_rev_10 = [_probe_2k(p) for p in odors_10]
    others_9_preserved = sum(
        (vals_rev_10[i] > 0.1 if i < 5 else vals_rev_10[i] < -0.1) for i in range(1, 10)
    )
    sparse_mb_2k_bench = {
        **ctrl_2k.sparse_mb.diagnostics(),
        "lif_connectome_neurons": ctrl_2k._num_nodes,
        "train_time_sec": round(train_2k_sec, 2),
        "n_odors": 10,
        "acquisition_accuracy_pct": round(100.0 * acq_10_hits / 10.0, 1),
        "noisy_generalization_accuracy_pct": round(100.0 * noisy_10_hits / 10.0, 1),
        "odor0_before_reversal": vals_acq_10[0],
        "odor0_after_reversal": vals_rev_10[0],
        "other_9_odors_preserved": others_9_preserved,
        "acquisition_valences": vals_acq_10,
        "post_reversal_valences": vals_rev_10,
    }

    # 4E: CPU Throughput & Full LIF Step Whole-CNS MaleCNS Scaling (across realistic edge densities!)
    throughput_rows: List[Dict[str, Any]] = []

    rat_ctrl = RatNCAPController(RatNCAPConfig(behavior="trot", target_speed=0.5))
    t0 = time.perf_counter()
    n_rat = 500
    for _ in range(n_rat):
        rat_ctrl.step()
    dt_rat = time.perf_counter() - t0
    throughput_rows.append(
        {
            "workload": "Rat NCAP Rollout (31 DOF)",
            "nodes": 22,
            "steps_per_sec": round(n_rat / max(dt_rat, 1e-6), 1),
            "ms_per_step": round(1000.0 * dt_rat / n_rat, 4),
            "realtime_factor": round((n_rat * 0.02) / max(dt_rat, 1e-6), 1),
        }
    )

    sample_kps = [rat_ctrl.step()["keypoints_3d"] for _ in range(200)]
    t0 = time.perf_counter()
    solve_rat_v2_ik_sequence(sample_kps, dt=0.02)
    dt_ik = time.perf_counter() - t0
    throughput_rows.append(
        {
            "workload": "Rat v2 IK + KKT Spring-Muscle ID",
            "nodes": 31,
            "steps_per_sec": round(len(sample_kps) / max(dt_ik, 1e-6), 1),
            "ms_per_step": round(1000.0 * dt_ik / len(sample_kps), 4),
            "realtime_factor": round((len(sample_kps) * 0.02) / max(dt_ik, 1e-6), 1),
        }
    )

    fb_rate = FlyBrainController(
        walk_spec, behavior="walk_imitation", dynamics="rate", seed=7
    )
    t0 = time.perf_counter()
    n_fb = 400
    for _ in range(n_fb):
        fb_rate.step(obs_grounded, dt=0.002)
    dt_fbr = time.perf_counter() - t0
    throughput_rows.append(
        {
            "workload": f"FlyBrain Rate ({fb_rate._num_nodes} pops)",
            "nodes": fb_rate._num_nodes,
            "steps_per_sec": round(n_fb / max(dt_fbr, 1e-6), 1),
            "ms_per_step": round(1000.0 * dt_fbr / n_fb, 4),
            "realtime_factor": round((n_fb * 0.002) / max(dt_fbr, 1e-6), 2),
        }
    )

    fb_spk = FlyBrainController(
        walk_spec,
        behavior="walk_imitation",
        connectome=spiking_graph,
        dynamics="spiking",
        seed=7,
    )
    t0 = time.perf_counter()
    n_fbs = 250
    for _ in range(n_fbs):
        fb_spk.step(obs_grounded, dt=0.002)
    dt_fbs = time.perf_counter() - t0
    throughput_rows.append(
        {
            "workload": f"FlyBrain LIF Spiking ({fb_spk._num_nodes} N)",
            "nodes": fb_spk._num_nodes,
            "steps_per_sec": round(n_fbs / max(dt_fbs, 1e-6), 1),
            "ms_per_step": round(1000.0 * dt_fbs / n_fbs, 4),
            "realtime_factor": round((n_fbs * 0.002) / max(dt_fbs, 1e-6), 2),
        }
    )

    # Whole-CNS scaling: BOTH spike-propagation-only AND complete LIF substep across scales & realistic degrees
    csr_scaling: List[Dict[str, Any]] = []
    rng_s = np.random.default_rng(777)
    scaling_configs = (
        (1000, 12),
        (10000, 12),
        (50000, 12),
        (166700, 12),
        (166700, 50),
        (166700, 100),
    )
    for n_neurons, avg_deg in scaling_configs:
        n_edges = n_neurons * avg_deg
        pre_idx = rng_s.integers(0, n_neurons, size=n_edges, dtype=np.int32)
        post_idx = rng_s.integers(0, n_neurons, size=n_edges, dtype=np.int32)
        vals = rng_s.uniform(-0.2, 0.2, size=n_edges).astype(np.float64)
        sw = SparseSynapticWeights((n_neurons, n_neurons), post_idx, pre_idx, vals)
        active_spikes = rng_s.choice(
            n_neurons, size=max(10, n_neurons // 100), replace=False
        ).astype(np.int32)
        # 1) Propagation-only
        _ = sw.propagate_spikes(active_spikes)
        n_iters = 10
        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = sw.propagate_spikes(active_spikes)
        dt_prop_ms = (time.perf_counter() - t0) / n_iters * 1000.0

        # 2) Complete LIF substep (voltage + refractory + spike detect + conductance + CSR propagation)
        v = rng_s.uniform(-53.5, -46.0, size=n_neurons)
        g = np.zeros(n_neurons, dtype=np.float64)
        ref = np.zeros(n_neurons, dtype=np.float64)
        rates = np.zeros(n_neurons, dtype=np.float64)
        tau = np.full(n_neurons, 0.015, dtype=np.float64)
        drive = rng_s.uniform(0.0, 6.0, size=n_neurons)
        dt_sub = 0.0002
        syn_decay = math.exp(-dt_sub / 0.005)
        rate_decay = math.exp(-dt_sub / 0.020)
        t0 = time.perf_counter()
        for _ in range(n_iters):
            g *= syn_decay
            ref = np.maximum(0.0, ref - dt_sub)
            active = ref <= 0.0
            dv = ((-52.0 - v) + g + drive) / tau
            v = np.where(active, np.clip(v + dt_sub * dv, -75.0, -35.0), -52.0)
            spiked = np.flatnonzero(v >= -45.0)
            if spiked.size == 0:
                spiked = active_spikes
            rates *= rate_decay
            v[spiked] = -52.0
            ref[spiked] = 0.0022
            rates[spiked] = np.minimum(1.0, rates[spiked] + 0.55)
            g = np.clip(g + sw.propagate_spikes(spiked) * 28.0, -35.0, 45.0)
        dt_full_ms = (time.perf_counter() - t0) / n_iters * 1000.0

        csr_mb = (
            (sw._csr.data.nbytes + sw._csr.indices.nbytes + sw._csr.indptr.nbytes)
            / (1024.0 * 1024.0)
            if sw._csr is not None
            else (n_edges * 16) / (1024.0 * 1024.0)
        )
        state_mb = (
            v.nbytes + g.nbytes + ref.nbytes + rates.nbytes + tau.nbytes + drive.nbytes
        ) / (1024.0 * 1024.0)
        # 3) Biologically structured 78-Neuropil Modular + Lognormal MaleCNS topology (at 166.7k scale)
        mod_prop_ms = None
        mod_full_ms = None
        if n_neurons == 166700:
            n_mod = 78
            mod_size = n_neurons // n_mod
            pre_m = np.repeat(np.arange(n_neurons, dtype=np.int32), avg_deg)
            mods = np.minimum(pre_m // mod_size, n_mod - 1)
            is_local = rng_s.random(n_edges) < 0.80
            local_post = np.clip(
                mods * mod_size + rng_s.integers(0, mod_size, size=n_edges),
                0,
                n_neurons - 1,
            ).astype(np.int32)
            global_post = rng_s.integers(0, n_neurons, size=n_edges, dtype=np.int32)
            post_m = np.where(is_local, local_post, global_post)
            signs = np.where(rng_s.random(n_edges) < 0.78, 1.0, -1.0)
            vals_m = signs * np.minimum(1.0, rng_s.lognormal(-2.2, 0.8, size=n_edges))
            sw_mod = SparseSynapticWeights(
                (n_neurons, n_neurons), post_m, pre_m, vals_m
            )
            spk_loc = rng_s.choice(
                np.arange(0, 2 * mod_size, dtype=np.int32),
                size=active_spikes.size,
                replace=False,
            )
            _ = sw_mod.propagate_spikes(spk_loc)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                _ = sw_mod.propagate_spikes(spk_loc)
            mod_prop_ms = round((time.perf_counter() - t0) / n_iters * 1000.0, 3)

            vm = rng_s.uniform(-53.5, -46.0, size=n_neurons)
            gm = np.zeros(n_neurons, dtype=np.float64)
            refm = np.zeros(n_neurons, dtype=np.float64)
            rm = np.zeros(n_neurons, dtype=np.float64)
            t0 = time.perf_counter()
            for _ in range(n_iters):
                gm *= syn_decay
                refm = np.maximum(0.0, refm - dt_sub)
                act_m = refm <= 0.0
                dvm = ((-52.0 - vm) + gm + drive) / tau
                vm = np.where(act_m, np.clip(vm + dt_sub * dvm, -75.0, -35.0), -52.0)
                sp_m = np.flatnonzero(vm >= -45.0)
                if sp_m.size == 0:
                    sp_m = spk_loc
                rm *= rate_decay
                vm[sp_m] = -52.0
                refm[sp_m] = 0.0022
                rm[sp_m] = np.minimum(1.0, rm[sp_m] + 0.55)
                gm = np.clip(gm + sw_mod.propagate_spikes(sp_m) * 28.0, -35.0, 45.0)
            mod_full_ms = round((time.perf_counter() - t0) / n_iters * 1000.0, 3)

        csr_scaling.append(
            {
                "neurons": n_neurons,
                "mean_out_degree": avg_deg,
                "directed_edges": n_edges,
                "active_spikes_per_step": int(active_spikes.size),
                "ms_per_propagation": round(dt_prop_ms, 3),
                "ms_per_full_lif_substep": round(dt_full_ms, 3),
                "modular78_ms_per_propagation": mod_prop_ms,
                "modular78_ms_per_full_lif_substep": mod_full_ms,
                "csr_ram_mb": round(csr_mb, 2),
                "state_ram_mb": round(state_mb, 2),
                "total_ram_mb": round(csr_mb + state_mb, 2),
            }
        )

    return {
        "selectivity": selectivity_rows,
        "motor_output_verification": motor_verification,
        "causal_perturbations": causal_tests,
        "noise_robustness": noise_robustness,
        "mushroom_body_learning_rate": learning_trace_rate,
        "mushroom_body_learning_spiking": learning_trace_spiking,
        "sparse_mb_2000kc_lif": sparse_mb_2k_bench,
        "throughput": throughput_rows,
        "csr_whole_cns_scaling": csr_scaling,
    }


# ---------------------------------------------------------------------------
# Camera-Ready Publication Figure Generation (5 Multi-Panel Vector PDF + PNG)
# ---------------------------------------------------------------------------


def _setup_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.5,
            "axes.titlesize": 9.3,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.4,
            "xtick.labelsize": 7.7,
            "ytick.labelsize": 7.7,
            "legend.fontsize": 7.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def _save_fig(fig: plt.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_fig1_architecture(exp1: Dict[str, Any]) -> None:
    s4_6 = next(
        r for r in exp1["multiview_sweep"] if r["n_cams"] == 4 and r["sigma_px"] == 6.0
    )
    cv_red = 100.0 * (
        1.0 - s4_6["bone_cv_after_pct"] / max(s4_6["bone_cv_before_pct"], 1e-6)
    )

    fig, ax = plt.subplots(figsize=(10.5, 4.1))
    ax.set_xlim(0, 10.5)
    ax.set_ylim(0, 4.1)
    ax.axis("off")

    def box(x, y, w, h, title, body, fc, ec, title_color="#0f172a"):
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.14",
            facecolor=fc,
            edgecolor=ec,
            linewidth=1.5,
        )
        ax.add_patch(p)
        ax.text(
            x + w / 2,
            y + h - 0.24,
            title,
            ha="center",
            va="top",
            fontsize=8.6,
            fontweight="bold",
            color=title_color,
        )
        ax.text(
            x + w / 2,
            y + h - 0.50,
            body,
            ha="center",
            va="top",
            fontsize=7.25,
            color="#1e293b",
            linespacing=1.32,
        )

    def arrow(x0, y0, x1, y1, color="#334155", label="", rad=0.0, dy=0.12):
        arr = FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>,head_length=6,head_width=3.6",
            linewidth=1.6,
            color=color,
        )
        ax.add_patch(arr)
        if label:
            mx, my = 0.5 * (x0 + x1), 0.5 * (y0 + y1) + dy
            ax.text(
                mx,
                my,
                label,
                ha="center",
                va="center",
                fontsize=7.1,
                fontweight="bold",
                color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85),
            )

    ax.text(
        5.25,
        3.95,
        "Annolid Bidirectional Neuro-Biomechanical Digital Twin Architecture (Pure-NumPy Core + Optional MuJoCo)",
        ha="center",
        va="top",
        fontsize=9.9,
        fontweight="bold",
        color="#0f172a",
    )

    box(
        0.15,
        1.55,
        2.10,
        2.05,
        "1. Markerless Ethology",
        "• Single / Multi-view video\n• LabelMe / NDJSON poses\n• Depth sidecars\n• Chirality-free parts\n• Temporal smoothing\n  (EMA / One Euro / Kalman)",
        "#eff6ff",
        "#2563eb",
        "#1d4ed8",
    )
    box(
        2.65,
        1.55,
        2.25,
        2.05,
        "2. 3D Anatomical Lifting",
        f"• Undistort + Weighted DLT\n• Huber Gauss-Newton\n• Leave-1-out outlier filter\n• Bilinear / Median depth\n• Rigid-bone PBD manifold\n  (Bone-Length CV ↓ {cv_red:.1f}%)",
        "#f0fdf4",
        "#16a34a",
        "#15803d",
    )
    box(
        5.30,
        1.55,
        2.30,
        2.05,
        "3. 3D Biomechanics Twin",
        "• Kabsch 6-DOF trunk fit\n• 31-DOF Rat v2 IK\n• Exact KKT Spring-Muscle\n  ID + MuJoCo cross-check\n• FlyBody 3D MuJoCo /\n  Three.js 41-bone GLB rig",
        "#fff7ed",
        "#ea580c",
        "#c2410c",
    )
    box(
        7.98,
        1.55,
        2.37,
        2.05,
        "4. Mechanistic Circuits",
        "• Mammalian Rat NCAP CPG\n  (V0d/V0v/V3f/V3a + RG)\n• Drosophila MaleCNS/FlyWire\n  LIF Spiking (166.7k CSR)\n• Opto / Kir2.1 lesions\n• LIF PAM/PPL1 Plasticity",
        "#faf5ff",
        "#7c3aed",
        "#6d28d9",
    )

    arrow(2.27, 2.65, 2.63, 2.65, "#1d4ed8")
    arrow(4.92, 2.65, 5.28, 2.65, "#15803d")
    arrow(7.62, 2.65, 7.96, 2.65, "#c2410c")

    box(
        0.15,
        0.10,
        4.85,
        1.08,
        "Sim-to-Real Synthetic Supervision & Photometry",
        "WebGL 3D ID-pass masks + 16-pt Rat / 13-pt Mouse / 11-pt Fly\nReinhard Lab matching + arena compositing → Zero-label YOLO/VOS",
        "#f8fafc",
        "#475569",
        "#334155",
    )
    box(
        5.42,
        0.10,
        4.93,
        1.08,
        "Real-to-Sim Causal Circuit ID & Perturbation",
        "Blind 6-class V0/V3 spinal phenotype detection (Rat NCAP + MuJoCo)\nIn-silico Chrimson optogenetics, Kir2.1 silencing & Dale's-law noise",
        "#fef2f2",
        "#dc2626",
        "#b91c1c",
    )

    arrow(9.15, 1.53, 9.15, 1.20, "#b91c1c", label="Neural Telemetry", dy=0.02)
    arrow(6.45, 1.53, 6.45, 1.20, "#b91c1c", label="IK/ID Inversion", dy=0.02)
    arrow(5.40, 0.64, 5.02, 0.64, "#475569")
    arrow(1.20, 1.20, 1.20, 1.53, "#334155", label="Synthetic Dataset", dy=0.02)

    _save_fig(fig, "fig1_architecture")


def plot_fig2_multiview_lifting(exp1: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))
    sweep = exp1["multiview_sweep"]

    # Panel A: 4-view MPJPE vs 2D pixel noise including spatial-only PBD (lambda_t=0) vs temporal (0.15)
    ax = axes[0]
    s4 = [r for r in sweep if r["n_cams"] == 4]
    sigmas = [r["sigma_px"] for r in s4]
    ax.plot(
        sigmas,
        [r["mpjpe_gn_mm"] for r in s4],
        "s-",
        color="#2563eb",
        lw=1.7,
        label="4-View DLT + Huber GN",
    )
    ax.plot(
        sigmas,
        [r["mpjpe_rigid_spatial_mm"] for r in s4],
        "d--",
        color="#0d9488",
        lw=1.6,
        label="+ Rigid PBD ($\\lambda_t=0$)",
    )
    rig_m = [r["mpjpe_rigid_mm"] for r in s4]
    rig_lo = [r["mpjpe_rigid_ci95"][0] for r in s4]
    rig_hi = [r["mpjpe_rigid_ci95"][1] for r in s4]
    ax.plot(
        sigmas,
        rig_m,
        "^-",
        color="#16a34a",
        lw=2.0,
        label="+ Rigid PBD ($\\lambda_t=0.15$)",
    )
    ax.fill_between(sigmas, rig_lo, rig_hi, color="#16a34a", alpha=0.16)
    ax.plot(
        sigmas,
        [r["mpjpe_all_edges_mm"] for r in s4],
        "x:",
        color="#dc2626",
        lw=1.5,
        label="+ Naive All-Edge PBD",
    )
    ax.set_xlabel("2D Keypoint Pixel Noise $\\sigma$ (px)")
    ax.set_ylabel("3D MPJPE (mm) $\\downarrow$")
    ax.set_title("(a) Multi-View 3D Accuracy & PBD Ablation")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="upper left")

    # Panel B: Outlier rejection + 30% camera-dropout occlusion
    ax = axes[1]
    out = exp1["outlier_stress"]
    occ = exp1["occlusion_dropout"]
    groups = [
        "3-Cam\n20% Outliers",
        "4-Cam\n20% Outliers",
        "3-Cam\n30% Dropout",
        "4-Cam\n30% Dropout",
    ]
    baseline_vals = [
        out[0]["mpjpe_unprotected_dlt_mm"],
        out[1]["mpjpe_unprotected_dlt_mm"],
        occ[0]["mpjpe_gn_mm"],
        occ[1]["mpjpe_gn_mm"],
    ]
    ours_vals = [
        out[0]["mpjpe_robust_gn_rigid_mm"],
        out[1]["mpjpe_robust_gn_rigid_mm"],
        occ[0]["mpjpe_rigid_mm"],
        occ[1]["mpjpe_rigid_mm"],
    ]
    x = np.arange(len(groups))
    w = 0.35
    b1 = ax.bar(
        x - w / 2, baseline_vals, w, color="#ef4444", label="Unprotected / Raw GN"
    )
    b2 = ax.bar(
        x + w / 2, ours_vals, w, color="#16a34a", label="Outlier-Reject + Rigid PBD"
    )
    for bars in (b1, b2):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.18,
                f"{h:.2f}",
                ha="center",
                va="bottom",
                fontsize=6.8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=7.2)
    ax.set_ylabel("3D MPJPE (mm) $\\downarrow$")
    ax.set_title("(b) Outliers (65 px) & 30% View Occlusion")
    ax.set_ylim(0, max(baseline_vals) * 1.24)
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="upper right")

    # Panel C: Monocular Depth Lifting Modes
    ax = axes[2]
    db = exp1["depth_sampling_ablation"]
    modes = ["nearest", "bilinear", "median_patch"]
    labels = ["Nearest\n(Legacy)", "Bilinear\n(Subpixel)", "Median 3×3\n(Robust)"]
    x = np.arange(len(modes))
    w = 0.34
    clean_vals = [db["clean_subpixel"][f"{m}_mpjpe_mm"] for m in modes]
    spike_vals = [db["spiked_8pct"][f"{m}_mpjpe_mm"] for m in modes]
    b_c = ax.bar(x - w / 2, clean_vals, w, color="#3b82f6", label="Clean Depth")
    b_s = ax.bar(x + w / 2, spike_vals, w, color="#f97316", label="8% Sensor Spikes")
    for bars in (b_c, b_s):
        for bar in bars:
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + 0.5,
                f"{h:.1f}",
                ha="center",
                va="bottom",
                fontsize=7.0,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("3D Depth-Lifted MPJPE (mm) $\\downarrow$")
    ax.set_title("(c) Monocular Depth Sampling Modes")
    ax.set_ylim(0, max(spike_vals) * 1.24)
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="upper right")

    fig.tight_layout(w_pad=1.5)
    _save_fig(fig, "fig2_multiview_3d_lifting")


def plot_fig3_rat_ncap_biomechanics(exp2: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.4))

    ax = axes[0]
    speeds = exp2["speeds_mps"]
    v0_scales = exp2["v0_scales"]
    mat = np.zeros((len(v0_scales), len(speeds)), dtype=int)
    for item in exp2["phase_grid"]:
        i = v0_scales.index(item["v0_scale"])
        j = speeds.index(item["target_speed_mps"])
        mat[i, j] = int(item["gait_code"])

    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(["#cbd5e1", "#38bdf8", "#22c55e", "#f97316"])
    ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0, vmax=3, origin="upper")
    ax.set_yticks(np.arange(len(v0_scales)))
    ax.set_yticklabels([f"{s:.2f}" for s in v0_scales])
    xt_idx = [0, 4, 8, 12, 16]
    ax.set_xticks(xt_idx)
    ax.set_xticklabels([f"{speeds[k]:.2f}" for k in xt_idx])
    ax.set_xlabel("Brainstem Speed Command (m/s)")
    ax.set_ylabel("$V0_v + V0_d$ Activity Scale ($k$)")
    ax.set_title("(a) Spinal Circuit Gait Phase Diagram")
    ax.grid(False)
    patches = [
        mpatches.Patch(color="#cbd5e1", label="Stand"),
        mpatches.Patch(color="#38bdf8", label="Walk"),
        mpatches.Patch(color="#22c55e", label="Trot"),
        mpatches.Patch(color="#f97316", label="Bound"),
    ]
    ax.legend(
        handles=patches,
        loc="lower left",
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.92,
    )

    # Panel B: Positive Mechanical Power & Joint Excursion across 4 Neuromechanical Interfaces
    ax = axes[1]
    itab = exp2["interface_table"]
    ifaces = ["flexor_extensor", "position_stiffness", "position", "torque"]
    iface_lbls = [
        "Flex-Ext\nSpring",
        "Pos+Stiff\nPD",
        "Fixed\nPos PD",
        "Direct\nTorque",
    ]
    behs = ["walk", "trot", "bound", "bumpy_terrain"]
    beh_colors = {
        "walk": "#38bdf8",
        "trot": "#22c55e",
        "bound": "#f97316",
        "bumpy_terrain": "#8b5cf6",
    }
    beh_lbls = {
        "walk": "Walk (0.22)",
        "trot": "Trot (0.50)",
        "bound": "Bound (0.88)",
        "bumpy_terrain": "Bumpy (3cm)",
    }
    x = np.arange(len(ifaces))
    w = 0.19
    for b_i, beh in enumerate(behs):
        vals = [
            next(
                r["mean_positive_power_w"]
                for r in itab
                if r["interface"] == iface and r["behavior"] == beh
            )
            for iface in ifaces
        ]
        ax.bar(x + (b_i - 1.5) * w, vals, w, color=beh_colors[beh], label=beh_lbls[beh])
    ax.set_xticks(x)
    ax.set_xticklabels(iface_lbls)
    ax.set_ylabel("Mean Positive Mechanical Power (W)")
    ax.set_title("(b) Neuromechanical Interface Energetics")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left", ncol=2)

    # Panel C: Bellardita-Kiehn V0 Shift
    ax = axes[2]
    intact = [r for r in exp2["phase_grid"] if r["v0_scale"] == 1.0]
    ablated = [r for r in exp2["phase_grid"] if r["v0_scale"] == 0.0]
    sp = [r["target_speed_mps"] for r in intact]
    ax.plot(
        sp,
        [r["lr_synchrony"] for r in intact],
        "o-",
        color="#16a34a",
        lw=2.0,
        label="Intact ($k_{V0}=1.0$)",
    )
    ax.plot(
        sp,
        [r["lr_synchrony"] for r in ablated],
        "s--",
        color="#dc2626",
        lw=2.0,
        label="$V0_v+V0_d$ Ablated ($k_{V0}=0.0$)",
    )
    ax.axvspan(0.18, 0.66, color="#fef3c7", alpha=0.45, label="Diagnostic Trot Regime")
    ax.axhline(0.55, color="#64748b", ls=":", lw=1.1)
    ax.set_xlabel("Brainstem Speed Command (m/s)")
    ax.set_ylabel("Left-Right Limb Synchrony Index")
    ax.set_ylim(-0.08, 1.08)
    ax.set_title("(c) Bellardita–Kiehn $V0$ Phenotype Shift")
    ax.legend(frameon=True, facecolor="white", framealpha=0.92, loc="lower right")

    fig.tight_layout(w_pad=1.5)
    _save_fig(fig, "fig3_rat_ncap_biomechanics")


def plot_fig4_real2sim_inversion(exp3: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.55))

    # Panel A: 6-Class Blind Phenotype Accuracy (180 Trials) + 95% CIs + V0/V3 F1 + Independent MuJoCo points
    ax = axes[0]
    curve = exp3["blind_phenotype_noise_curve"]
    ns = [r["sigma_mm"] for r in curve]
    o_m = [r["ours_accuracy_pct"] for r in curve]
    o_lo = [r["ours_ci95"][0] for r in curve]
    o_hi = [r["ours_ci95"][1] for r in curve]
    n_m = [r["naive_accuracy_pct"] for r in curve]
    ax.plot(ns, o_m, "o-", color="#16a34a", lw=2.1, label="Ours 6-Class Acc. (95% CI)")
    ax.fill_between(ns, o_lo, o_hi, color="#16a34a", alpha=0.16)
    ax.plot(ns, n_m, "x--", color="#dc2626", lw=1.7, label="Single-Anchor Baseline")
    ax.plot(
        ns,
        [r["v0_deficit_f1"] * 100.0 for r in curve],
        "^:",
        color="#7c3aed",
        lw=1.6,
        label="$V0$-Deficit F1 (%)",
    )
    ax.plot(
        ns,
        [r["v3_deficit_f1"] * 100.0 for r in curve],
        "s:",
        color="#0284c7",
        lw=1.6,
        label="$V3$-Deficit F1 (%)",
    )
    mj_tr = exp3["mujoco_cross_validation"]["noisy_blind_transfer"]
    ax.scatter(
        [r["sigma_mm"] for r in mj_tr],
        [r["accuracy_pct"] for r in mj_tr],
        marker="*",
        s=74,
        color="#f59e0b",
        edgecolor="#78350f",
        zorder=6,
        label="Indep. Rat MuJoCo",
    )
    ms_tr = exp3["mouse_mujoco_cross_species"]["noisy_occluded_transfer"]
    ax.scatter(
        [r["rat_equiv_sigma_mm"] for r in ms_tr],
        [r["accuracy_pct"] for r in ms_tr],
        marker="D",
        s=36,
        color="#ec4899",
        edgecolor="#831843",
        zorder=7,
        label="25g Mouse MuJoCo (DANNCE mm)",
    )
    ax.set_xlabel("3D Keypoint Noise $\\sigma_{\\mathrm{3D}}$ (mm)")
    ax.set_ylabel("Accuracy / F1 (%)")
    ax.set_ylim(10, 106)
    ax.set_title("(a) 6-Class Blind Spinal ID (180 Trials)")
    ax.legend(
        frameon=True, facecolor="white", framealpha=0.92, loc="lower left", fontsize=6.7
    )

    # Panel B: 6x6 Confusion Matrix at sigma = 4 mm (180 blind trials)
    ax = axes[1]
    cm = np.asarray(exp3["confusion_matrix_at_4mm"], dtype=int)
    short_lbls = ["Stand", "Walk", "Trot", "Bound", "$V0$-Def", "$V3$-Def"]
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=30)
    ax.grid(False)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            color = "white" if val >= 16 else "#0f172a"
            ax.text(
                j,
                i,
                str(val),
                ha="center",
                va="center",
                color=color,
                fontsize=7.6,
                fontweight="bold",
            )
    ax.set_xticks(range(len(short_lbls)))
    ax.set_yticks(range(len(short_lbls)))
    ax.set_xticklabels(short_lbls, rotation=24, ha="right", fontsize=7.3)
    ax.set_yticklabels(short_lbls, fontsize=7.3)
    ax.set_xlabel("Predicted Phenotype")
    ax.set_ylabel("Ground-Truth Phenotype")
    ax.set_title("(b) 6×6 Blind Confusion ($\\sigma=4$ mm)")

    # Panel C: Independent MuJoCo 3.5.0 Multi-Body Cross-Validation (IK r & Lagrangian ID audit)
    ax = axes[2]
    mj_reg = exp3["mujoco_cross_validation"]["per_regime"]
    reg_lbls = ["Walk", "Trot", "$V0$-Def", "Bound", "$V3$-Def"]
    x = np.arange(len(reg_lbls))
    w = 0.27
    sw_rms = [r["mujoco_swing_torque_rms_nm"] for r in mj_reg]
    st_rms = [r["mujoco_stance_torque_rms_nm"] for r in mj_reg]
    ik_rms = [r["decoupled_kkt_torque_rms_nm"] for r in mj_reg]
    ax.bar(x - w, sw_rms, w, color="#3b82f6", label="MuJoCo Swing")
    ax.bar(x, st_rms, w, color="#f97316", label="MuJoCo Stance")
    ax.bar(x + w, ik_rms, w, color="#16a34a", label="Decoupled KKT")
    ax2 = ax.twinx()
    ik_r = [r["sagittal_ik_corr"] for r in mj_reg]
    ax2.plot(x, ik_r, "ko-", lw=1.6, markersize=4.5, label="MuJoCo IK $r$")
    ax2.set_ylim(0.45, 1.08)
    ax2.set_ylabel("Out-of-Model IK Corr. $r$", color="#0f172a")
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(reg_lbls)
    ax.set_ylabel("Hip/Shoulder Torque RMS (Nm)")
    ax.set_ylim(0, max(ik_rms) * 1.48)
    ax.set_title("(c) Indep. MuJoCo 3.5.0 Physics Audit")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        ncol=2,
        frameon=True,
        facecolor="white",
        framealpha=0.94,
        fontsize=6.3,
    )

    fig.tight_layout(w_pad=1.6)
    _save_fig(fig, "fig4_real2sim_inversion")


def plot_fig5_drosophila_connectome(exp4: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.5))

    # Panel A: LIF Spiking (1,470 neurons) Descending Neuron Selectivity Matrix
    ax = axes[0]
    spk_rows = [r for r in exp4["selectivity"] if r["dynamics"] == "spiking"]
    dn_cols = ["DNp09", "MDN", "Pair1", "aDN1", "DNp01", "DNg02"]
    beh_lbls = ["Walk", "Backward", "Groom", "Escape", "Flight"]
    mat = np.array([[r["dn_rates"][k] for k in dn_cols] for r in spk_rows], dtype=float)
    im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0.0, vmax=1.0)
    ax.grid(False)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color=("black" if v > 0.58 else "white"),
                fontsize=7.0,
            )
    ax.set_xticks(range(len(dn_cols)))
    ax.set_xticklabels(dn_cols, rotation=25, ha="right")
    ax.set_yticks(range(len(beh_lbls)))
    ax.set_yticklabels(beh_lbls)
    n_spk_nodes = int(spk_rows[0]["active_nodes"]) if spk_rows else 1470
    ax.set_title(f"(a) {n_spk_nodes:,}-Neuron LIF Descending Selectivity")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Norm. Rate")

    # Panel B: Three-Factor PAM/PPL1 Mushroom Body Acquisition & Reversal in BOTH Rate AND 1,534-N Spiking LIF!
    ax = axes[1]
    tr_r = exp4["mushroom_body_learning_rate"]
    tr_s = exp4["mushroom_body_learning_spiking"]
    trials = [r["trial"] for r in tr_s]
    ax.plot(
        trials,
        [r["odor_a_motor_valence"] for r in tr_s],
        "o-",
        color="#2563eb",
        lw=2.1,
        markersize=4.2,
        label="LIF 1,534 N: Odor A (Rew→Shk)",
    )
    ax.plot(
        trials,
        [r["odor_b_motor_valence"] for r in tr_s],
        "s-",
        color="#dc2626",
        lw=2.1,
        markersize=4.2,
        label="LIF 1,534 N: Odor B (Shk→Rew)",
    )
    ax.plot(
        trials,
        [r["odor_a_motor_valence"] for r in tr_r],
        "--",
        color="#93c5fd",
        lw=1.4,
        label="Rate 61 pop: Odor A",
    )
    ax.plot(
        trials,
        [r["odor_b_motor_valence"] for r in tr_r],
        "--",
        color="#fca5a5",
        lw=1.4,
        label="Rate 61 pop: Odor B",
    )
    ax.axvline(8.5, color="#334155", ls="--", lw=1.2)
    ax.axhline(0.0, color="#64748b", ls=":", lw=1.0)
    ax.text(
        4.2,
        -1.02,
        "Acquisition",
        ha="center",
        fontsize=7.3,
        fontweight="bold",
        color="#15803d",
    )
    ax.text(
        14.5,
        -1.02,
        "Reversal",
        ha="center",
        fontsize=7.3,
        fontweight="bold",
        color="#b91c1c",
    )
    ax.set_xlabel("Conditioning Trial")
    ax.set_ylabel("Motor Valence ($r_{\\mathrm{DNp09}} - r_{\\mathrm{MDN}}$)")
    ax.set_ylim(-1.14, 1.44)
    ax.set_title("(b) LIF Spiking & Rate MB Reversal")
    ax.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.95,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=2,
        fontsize=6.1,
        columnspacing=0.8,
        handlelength=1.6,
    )

    # Panel C: Full LIF Step Whole-CNS Scaling (Erdos-Renyi vs 78-Neuropil Modular MaleCNS Cache Locality)
    ax = axes[2]
    sc_deg12 = [r for r in exp4["csr_whole_cns_scaling"] if r["mean_out_degree"] == 12]
    sc_166k = [r for r in exp4["csr_whole_cns_scaling"] if r["neurons"] == 166700]
    labels = [
        "10k N\n(0.12M)",
        "50k N\n(0.6M)",
        "166.7k\n(2.0M)",
        "166.7k\n(8.3M)",
        "166.7k\n(16.7M)",
    ]
    rows = [sc_deg12[1], sc_deg12[2], sc_166k[0], sc_166k[1], sc_166k[2]]
    x = np.arange(len(rows))
    w = 0.34
    full_er_ms = [r["ms_per_full_lif_substep"] for r in rows]
    full_mod_ms = [
        (
            r["modular78_ms_per_full_lif_substep"]
            if r["modular78_ms_per_full_lif_substep"] is not None
            else r["ms_per_full_lif_substep"]
        )
        for r in rows
    ]
    ram_mb = [r["total_ram_mb"] for r in rows]
    ax.bar(x - w / 2, full_er_ms, w, color="#a78bfa", label="Full LIF (Random ER)")
    b2 = ax.bar(
        x + w / 2, full_mod_ms, w, color="#6d28d9", label="Full LIF (78-Neuropil Mod.)"
    )
    for idx_b, bar in enumerate(b2):
        h = bar.get_height()
        x_off = 0.08 if idx_b < 3 else 0.0
        ax.text(
            bar.get_x() + bar.get_width() / 2 + x_off,
            h + 0.45,
            f"{h:.1f}",
            ha="center",
            va="bottom",
            fontsize=6.7,
            color="#4c1d95",
            fontweight="bold",
        )
    ax2 = ax.twinx()
    ax2.plot(
        x,
        ram_mb,
        "s--",
        color="#0d9488",
        lw=1.8,
        markersize=4.5,
        label="Total RAM (MB)",
    )
    ax2.set_ylim(-15, max(ram_mb) * 1.18)
    ax2.set_ylabel("Total RAM (CSR + State, MB)", color="#0d9488")
    ax2.spines["top"].set_visible(False)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.0)
    ax.set_ylabel("Full LIF Substep Time (ms)")
    ax.set_ylim(0, max(full_er_ms) * 1.32)
    ax.set_title("(c) MaleCNS Modular vs. ER Scaling")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(
        h1 + h2,
        l1 + l2,
        loc="upper left",
        frameon=True,
        facecolor="white",
        framealpha=0.92,
        fontsize=6.7,
    )

    fig.tight_layout(w_pad=1.6)
    _save_fig(fig, "fig5_drosophila_connectome")


def main() -> None:
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _setup_pub_style()

    t_start = time.perf_counter()
    exp1 = run_experiment_1_multiview_lifting()
    exp2 = run_experiment_2_rat_ncap()
    exp3 = run_experiment_3_real2sim_inversion()
    exp4 = run_experiment_4_drosophila_connectome()

    bundle = {
        "metadata": {
            "framework": "Annolid Simulation & Digital Twin Suite",
            "total_benchmark_sec": round(time.perf_counter() - t_start, 2),
        },
        "exp1_multiview_lifting": exp1,
        "exp2_rat_ncap": exp2,
        "exp3_real2sim_inversion": exp3,
        "exp4_drosophila_connectome": exp4,
    }
    out_json = PAPER_DIR / "benchmark_results.json"
    out_json.write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
    print(
        f"[Done] Saved benchmark bundle to {out_json} in {bundle['metadata']['total_benchmark_sec']}s"
    )

    print("[Plot] Generating 5 publication-grade figures (PDF + 300 DPI PNG)...")
    plot_fig1_architecture(exp1)
    plot_fig2_multiview_lifting(exp1)
    plot_fig3_rat_ncap_biomechanics(exp2)
    plot_fig4_real2sim_inversion(exp3)
    plot_fig5_drosophila_connectome(exp4)
    print(f"[Plot] Figures saved in {FIG_DIR}")


if __name__ == "__main__":
    main()
