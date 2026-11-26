from __future__ import annotations

import base64
import io
import math
from types import SimpleNamespace

import imageio.v2 as imageio
import numpy as np
import pytest

from annolid.simulation import (
    RAT_V2_RIGID_BONE_EDGES,
    CameraCalibration,
    FlyBrainController,
    Pose2DFrame,
    Pose3DFrame,
    build_plastic_fly_connectome,
    build_simulation_adapter,
    enforce_anatomical_skeleton_constraints,
    estimate_rodent_body_scale,
    infer_spinal_circuit_phenotype,
    invert_spring_muscle_activations,
    lift_pose_frames_with_depth,
    run_rat_ncap_rollout,
    solve_rat_v2_ik_frame,
    solve_rat_v2_ik_sequence,
    triangulate_multiview_pose_frames,
    triangulate_point_multiview,
)


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
        [np.zeros(6, dtype=np.float64), np.full(len(names) - 6, -1.5, dtype=np.float64)]
    )
    maximum = np.concatenate(
        [np.ones(6, dtype=np.float64), np.full(len(names) - 6, 2.0, dtype=np.float64)]
    )
    return SimpleNamespace(
        name="\t".join(names),
        shape=(len(names),),
        dtype=np.float64,
        minimum=minimum,
        maximum=maximum,
    )


def _depth_record(
    depth_array: np.ndarray, *, d_min: float = 0.0, d_max: float = 4.0
) -> dict:
    norm = np.clip((depth_array - d_min) / max(d_max - d_min, 1e-6), 0.0, 1.0)
    quantized = np.round(norm * 65535.0).astype(np.uint16)
    buf = io.BytesIO()
    imageio.imwrite(buf, quantized, format="png")
    return {
        "frame_index": 0,
        "otherData": {
            "depth_map": {
                "image_data": base64.b64encode(buf.getvalue()).decode("ascii"),
                "scale": {"min": d_min, "max": d_max},
                "dtype": "uint16",
            }
        },
    }


def _make_rig_cameras(n_cams: int = 4) -> dict[str, CameraCalibration]:
    cams: dict[str, CameraCalibration] = {}
    for idx in range(n_cams):
        ang = (2.0 * math.pi * idx) / float(n_cams) + 0.35
        pos = [0.65 * math.cos(ang), 0.65 * math.sin(ang), 0.45]
        cams[f"cam{idx}"] = CameraCalibration.from_look_at(
            name=f"cam{idx}",
            position=pos,
            target=[0.1, 0.0, 0.05],
            fx=820.0,
            fy=820.0,
            cx=320.0,
            cy=240.0,
            distortion=(0.08, -0.02, 0.001, -0.001),
            image_size=(640, 480),
        )
    return cams


def test_depth_lifting_bilinear_median_patch_and_camera_extrinsics() -> None:
    # Smooth ramp depth map with a single-pixel spike at (2, 2)
    ys, xs = np.mgrid[0:5, 0:5]
    depth = 1.0 + 0.25 * xs + 0.1 * ys
    depth_spiked = depth.copy()
    depth_spiked[2, 2] = 3.95  # corrupted sensor spike

    frame = Pose2DFrame(
        frame_index=0,
        image_height=5,
        image_width=5,
        video_name="depth_demo.mp4",
        points={"nose": (2.0, 2.0), "subpix": (1.5, 1.5)},
    )

    lifted_nearest = lift_pose_frames_with_depth(
        [frame],
        depth_records={0: _depth_record(depth_spiked)},
        coordinate_system={
            "camera_intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 0.0, "cy": 0.0}
        },
        sample_mode="nearest",
    )
    lifted_median = lift_pose_frames_with_depth(
        [frame],
        depth_records={0: _depth_record(depth_spiked)},
        coordinate_system={
            "camera_intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 0.0, "cy": 0.0}
        },
        sample_mode="median_patch",
        patch_radius=1,
    )
    # Nearest hits the corrupted spike (~3.95), median_patch recovers neighborhood median (~1.80)
    assert lifted_nearest[0].points["nose"][2] > 3.5
    assert abs(lifted_median[0].points["nose"][2] - 1.80) < 0.02

    # Bilinear interpolation at (1.5, 1.5) matches exact analytical plane value
    lifted_bilin = lift_pose_frames_with_depth(
        [frame],
        depth_records={0: _depth_record(depth)},
        coordinate_system={
            "camera_intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 0.0, "cy": 0.0},
            "camera_extrinsics": {
                "R": np.eye(3).tolist(),
                "t": [0.0, 0.0, -1.0],  # camera shifted +1 in world Z
            },
        },
        sample_mode="bilinear",
    )
    expected_z_cam = 1.0 + 0.25 * 1.5 + 0.1 * 1.5
    assert lifted_bilin[0].points["subpix"][2] == pytest.approx(
        expected_z_cam + 1.0, abs=0.01
    )


def test_multiview_triangulation_with_distortion_and_outlier_rejection() -> None:
    cams = _make_rig_cameras(4)
    gt_point = np.array([0.12, -0.04, 0.06], dtype=np.float64)

    obs = []
    for name, cam in cams.items():
        u, v, z = cam.project(gt_point, apply_distortion=True)
        assert z > 0.0
        obs.append((cam, (u, v), 0.95))

    # Clean 4-view triangulation recovers exact 3D point under lens distortion
    pt_clean, diag_clean = triangulate_point_multiview(obs)
    assert np.linalg.norm(np.asarray(pt_clean) - gt_point) < 1e-4
    assert diag_clean["reprojection_rmse_px"] < 0.05
    assert diag_clean["rejected_views"] == []

    # Corrupt cam2 with a severe 70 px tracking outlier -> leave-one-out rejects cam2
    obs_corrupted = list(obs)
    c2, (u2, v2), s2 = obs_corrupted[2]
    obs_corrupted[2] = (c2, (u2 + 70.0, v2 - 55.0), s2)

    pt_robust, diag_robust = triangulate_point_multiview(
        obs_corrupted, outlier_reproj_thresh_px=8.0
    )
    assert c2.name in diag_robust["rejected_views"]
    assert np.linalg.norm(np.asarray(pt_robust) - gt_point) < 2e-3
    assert diag_robust["reprojection_rmse_px"] < 0.5


def test_multiview_stream_triangulation_and_anatomical_skeleton_constraints() -> None:
    cams = _make_rig_cameras(4)
    rollout = run_rat_ncap_rollout(steps=30, behavior="trot", target_speed=0.50, seed=5)
    rng = np.random.default_rng(42)

    views: dict[str, list[Pose2DFrame]] = {name: [] for name in cams}
    gt_frames: list[dict[str, list[float]]] = []
    for idx, frame in enumerate(rollout["frames"]):
        kps = frame["keypoints_3d"]
        gt_frames.append(kps)
        for cam_name, cam in cams.items():
            pts_2d: dict[str, tuple[float, float]] = {}
            scores_2d: dict[str, float] = {}
            for lbl, p3 in kps.items():
                u, v, _ = cam.project(p3, apply_distortion=True)
                noise = rng.normal(0.0, 2.5, size=2)
                pts_2d[lbl] = (float(u + noise[0]), float(v + noise[1]))
                scores_2d[lbl] = 0.9
            views[cam_name].append(
                Pose2DFrame(
                    frame_index=idx,
                    image_height=480,
                    image_width=640,
                    video_name=f"{cam_name}.mp4",
                    timestamp_sec=float(frame["time_sec"]),
                    points=pts_2d,
                    scores=scores_2d,
                )
            )

    lifted, tri_summary = triangulate_multiview_pose_frames(views, cams)
    assert tri_summary["frames"] == 30
    assert tri_summary["mean_reprojection_rmse_px"] < 4.5

    constrained, skel_diag = enforce_anatomical_skeleton_constraints(
        lifted, RAT_V2_RIGID_BONE_EDGES, iterations=8, temporal_smooth_weight=0.15
    )
    assert len(constrained) == 30
    # Anatomical projection dramatically tightens bone-length CV on rigid axial/cranial/tail bones
    assert skel_diag["bone_length_cv_after"] < skel_diag["bone_length_cv_before"] * 0.20

    # Compare 3D MPJPE before vs after anatomical constraint projection
    def _mpjpe(frames_list: list[Pose3DFrame]) -> float:
        errs = []
        for f, gt in zip(frames_list, gt_frames):
            for lbl, p_gt in gt.items():
                if lbl in f.points:
                    errs.append(
                        float(
                            np.linalg.norm(np.asarray(f.points[lbl]) - np.asarray(p_gt))
                        )
                    )
        return float(np.mean(errs))

    err_raw = _mpjpe(lifted)
    err_con = _mpjpe(constrained)
    assert err_raw < 0.015  # < 15 mm from 2.5 px noise
    assert err_con < err_raw


def test_rat_v2_ik_spring_muscle_id_and_blind_spinal_phenotype_identification() -> None:
    # 1. Exact KKT spring-muscle inverse dynamics residual check
    rollout_trot = run_rat_ncap_rollout(
        steps=120,
        behavior="trot",
        target_speed=0.48,
        interface="flexor_extensor",
        seed=7,
    )
    q_gt = np.asarray([f["qpos"] for f in rollout_trot["frames"]], dtype=np.float64)
    id_res = invert_spring_muscle_activations(q_seq=q_gt, dt=0.02)
    assert id_res["torque_residual_rmse_nm"] < 1e-4
    assert np.all(id_res["flx_activation"] >= 0.0) and np.all(
        id_res["flx_activation"] <= 1.0
    )
    assert np.all(id_res["ext_activation"] >= 0.0) and np.all(
        id_res["ext_activation"] <= 1.0
    )

    # 2. Single-frame IK respects joint ranges and reconstructs keypoints accurately
    ik_single = solve_rat_v2_ik_frame(rollout_trot["frames"][40]["keypoints_3d"])
    assert ik_single["joint_limit_violations"] == 0
    assert ik_single["ik_rmse_m"] < 0.012

    # 3. Blind spinal phenotype classification distinguishes intact trot vs V0-ablated bound at identical 0.45 m/s!
    rollout_intact = run_rat_ncap_rollout(
        steps=140, behavior="trot", target_speed=0.45, ablation="none", seed=11
    )
    rollout_v0 = run_rat_ncap_rollout(
        steps=140,
        behavior="trot",
        target_speed=0.45,
        ablation="v0v_v0d",
        ablation_scale=0.0,
        seed=11,
    )
    # Add realistic 3 mm 3D tracking noise to keypoints before blind inference
    rng = np.random.default_rng(123)

    def _noisy_kps(
        rollout: dict, sigma_m: float = 0.003
    ) -> list[dict[str, list[float]]]:
        seq = []
        for f in rollout["frames"][30:]:
            seq.append(
                {
                    k: (
                        np.asarray(v, dtype=np.float64)
                        + rng.normal(0.0, sigma_m, size=3)
                    ).tolist()
                    for k, v in f["keypoints_3d"].items()
                }
            )
        return seq

    inf_intact = infer_spinal_circuit_phenotype(_noisy_kps(rollout_intact), dt=0.02)
    inf_v0 = infer_spinal_circuit_phenotype(_noisy_kps(rollout_v0), dt=0.02)
    assert inf_intact["dominant_gait"] == "trot"
    assert inf_intact["phenotype_class"] == "intact_trot"
    assert inf_intact["v0_integrity_index"] > 0.65

    assert inf_v0["dominant_gait"] == "bound"
    assert inf_v0["phenotype_class"] == "v0_commissural_deficit_phenotype"
    assert inf_v0["v0_integrity_index"] < 0.35

    # 3b. Walk recall under 4 mm noise (lateral-sequence phase stays distinct from trot)
    rollout_walk = run_rat_ncap_rollout(
        steps=140, behavior="walk", target_speed=0.16, ablation="none", seed=17
    )
    inf_walk = infer_spinal_circuit_phenotype(
        _noisy_kps(rollout_walk, sigma_m=0.004), dt=0.02
    )
    assert inf_walk["dominant_gait"] == "walk"
    assert inf_walk["phenotype_class"] == "intact_walk"

    # 3c. Blind V3 synchrony-deficit identification vs speed-matched intact bound (0.90 m/s)
    rollout_bound = run_rat_ncap_rollout(
        steps=140, behavior="bound", target_speed=0.90, ablation="none", seed=19
    )
    rollout_v3 = run_rat_ncap_rollout(
        steps=140,
        behavior="trot",
        target_speed=0.90,
        ablation="v3f_v3a",
        ablation_scale=0.0,
        seed=19,
    )
    inf_bound = infer_spinal_circuit_phenotype(
        _noisy_kps(rollout_bound, sigma_m=0.003), dt=0.02
    )
    inf_v3 = infer_spinal_circuit_phenotype(
        _noisy_kps(rollout_v3, sigma_m=0.003), dt=0.02
    )
    assert inf_bound["phenotype_class"] == "intact_bound"
    assert inf_bound["v3_integrity_index"] > 0.65
    assert inf_v3["phenotype_class"] == "v3_synchrony_deficit_phenotype"
    assert inf_v3["v3_integrity_index"] < 0.45

    # 4. RatNCAPAdapter.fit_3d populates full 31-DOF qpos, muscle activations, and spinal metadata
    adapter = build_simulation_adapter("rat_ncap")
    adapter.configure({"fps": 50.0, "interface": "flexor_extensor"})
    adapter.initialize()
    obs_3d = [
        Pose3DFrame(
            frame_index=idx,
            video_name="rat_v0.mp4",
            timestamp_sec=idx * 0.02,
            points={k: (v[0], v[1], v[2]) for k, v in pts.items()},
        )
        for idx, pts in enumerate(_noisy_kps(rollout_v0, sigma_m=0.001))
    ]
    fit_res = adapter.fit_3d(obs_3d)
    assert len(fit_res.frames) == len(obs_3d)
    assert len(fit_res.frames[0].state["qpos"]) == 31
    assert len(fit_res.frames[0].state["flx_activation"]) == 31
    assert fit_res.metadata["spinal_circuit_inference"]["phenotype_class"] == (
        "v0_commissural_deficit_phenotype"
    )
    seq_res = solve_rat_v2_ik_sequence(obs_3d[:20], dt=0.02)
    assert seq_res["summary"]["joint_limit_violations"] == 0

    # 5. Cross-species Sim(3) auto-scaling, DANNCE mm aliases, intermittent dropout & Froude similarity
    # Scale an intact V0-ablated trot rollout down to 0.35x adult mouse size & convert to millimeters with DANNCE names!
    dannce_alias = {
        "nose": "HeadF",
        "head": "HeadB",
        "left_ear": "HeadL",
        "right_ear": "HeadR",
        "thorax": "SpineF",
        "torso_center": "SpineM",
        "lumbar": "SpineL",
        "pelvis": "Pelvis",
        "tail_base": "TailBase",
        "left_forepaw": "HandL",
        "right_forepaw": "HandR",
        "left_hindpaw": "FootL",
        "right_hindpaw": "FootR",
    }
    mouse_mm_seq = []
    # Under Froude similarity, mouse time scales by sqrt(0.35) -> dt_mouse = 0.02 * sqrt(0.35)
    dt_mouse = 0.02 * math.sqrt(0.35)
    for f_idx, frame in enumerate(rollout_v0["frames"][30:]):
        kps_mm = {}
        for k_rat, k_dan in dannce_alias.items():
            if k_rat in frame["keypoints_3d"]:
                # Drop left_forepaw on every 10th frame to test transient occlusion interpolation
                if k_rat == "left_forepaw" and f_idx % 10 == 5:
                    continue
                kps_mm[k_dan] = [
                    float(v) * 0.35 * 1000.0 for v in frame["keypoints_3d"][k_rat]
                ]
        mouse_mm_seq.append(kps_mm)
    sc = estimate_rodent_body_scale(mouse_mm_seq)
    assert sc["detected_unit"] == "mm"
    assert 0.30 <= sc["body_scale_rel_rat"] <= 0.40
    inf_mouse_mm = infer_spinal_circuit_phenotype(mouse_mm_seq, dt=dt_mouse)
    assert inf_mouse_mm["detected_unit"] == "mm"
    assert inf_mouse_mm["phenotype_class"] == "v0_commissural_deficit_phenotype"
    assert inf_mouse_mm["v0_integrity_index"] < 0.35


def test_fly_brain_optogenetics_silencing_synaptic_noise_and_mb_plasticity() -> None:
    spec = _walking_spec()
    obs_grounded = {"walker/touch": np.ones(6, dtype=np.float64)}

    # 1. Optogenetic Chrimson stimulation of MDN overrides forward walk into backward moonwalk
    ctrl = FlyBrainController(
        spec, behavior="walk_imitation", dynamics="spiking", seed=7
    )
    for _ in range(30):
        ctrl.step(obs_grounded, dt=0.002)
    assert ctrl.dominant_circuit() == "dnp09_forward_walk"

    ctrl.stimulate_cell_type("MDN", 2.2, duration_sec=0.12)
    for _ in range(45):
        ctrl.step(obs_grounded, dt=0.002)
    assert ctrl.dominant_circuit() == "moonwalker_backward"
    assert ctrl.rate("MDN") > 0.45

    # 2. Genetic Kir2.1 silencing of Pair1 prevents grooming locomotor arrest
    ctrl_groom = FlyBrainController(spec, behavior="grooming", dynamics="rate", seed=7)
    ctrl_groom.set_ablation("Pair1", scale=0.0)
    for _ in range(40):
        ctrl_groom.step(obs_grounded, dt=0.002)
    assert ctrl_groom.rate("Pair1") == 0.0
    assert ctrl_groom.rate("aDN1") > 0.40
    assert "Pair1" in ctrl_groom.diagnostics()["ablations"]

    # 3. Dale's-law sign-preserving synaptic noise preserves sign & determinism
    ctrl_n1 = FlyBrainController(
        spec,
        behavior="walk_imitation",
        dynamics="rate",
        seed=19,
        synaptic_noise_std=0.25,
    )
    ctrl_n2 = FlyBrainController(
        spec,
        behavior="walk_imitation",
        dynamics="rate",
        seed=19,
        synaptic_noise_std=0.25,
    )
    w_base = ctrl_n1._base_weights.to_dense()
    w_noisy = ctrl_n1._weights.to_dense()
    np.testing.assert_allclose(w_noisy, ctrl_n2._weights.to_dense())
    assert np.all(np.sign(w_base) == np.sign(w_noisy))

    # 4. Three-factor PAM/PPL1 Mushroom Body conditioning & reversal learning
    mb_graph = build_plastic_fly_connectome(spiking=False, seed=7)
    ctrl_mb = FlyBrainController(
        spec,
        behavior="walk_imitation",
        connectome=mb_graph,
        dynamics="rate",
        seed=7,
        enable_plasticity=True,
        suppress_behavior_drive=True,
    )
    # Pair Odor A with PAM reward, brief inter-trial interval, then Odor B with PPL1 punishment
    for _ in range(120):
        ctrl_mb.step(
            {"walker/touch": np.ones(6), "odor_a": 1.0, "reward_da": 1.0}, dt=0.004
        )
    for _ in range(90):
        ctrl_mb.step({"walker/touch": np.ones(6)}, dt=0.004)
    for _ in range(120):
        ctrl_mb.step(
            {"walker/touch": np.ones(6), "odor_b": 1.0, "punish_da": 1.0}, dt=0.004
        )

    w_acq = ctrl_mb.plasticity_weights()
    assert w_acq["odor_a_approach"] > 1.35
    assert w_acq["odor_a_avoid"] < 0.70
    assert w_acq["odor_b_avoid"] > 1.35
    assert w_acq["odor_b_approach"] < 0.70

    # Test Odor A alone -> drives approach/forward walking (DNp09 > MDN)
    for _ in range(45):
        ctrl_mb.step({"walker/touch": np.ones(6), "odor_a": 1.0}, dt=0.004)
    assert ctrl_mb.rate("MBON_Approach") > ctrl_mb.rate("MBON_Avoid")
    assert ctrl_mb.rate("DNp09_L") > ctrl_mb.rate("MDN")

    # Test Odor B alone -> drives avoidance/retreat (MDN > DNp09)
    for _ in range(55):
        ctrl_mb.step({"walker/touch": np.ones(6), "odor_b": 1.0}, dt=0.004)
    assert ctrl_mb.rate("MBON_Avoid") > ctrl_mb.rate("MBON_Approach")
    assert ctrl_mb.rate("MDN") > ctrl_mb.rate("DNp09_L")

    # 5. Three-factor PAM/PPL1 conditioning & reversal in 1,534-neuron biophysical LIF Spiking mode
    mb_spk_graph = build_plastic_fly_connectome(spiking=True, seed=7)
    ctrl_mb_spk = FlyBrainController(
        spec,
        behavior="walk_imitation",
        connectome=mb_spk_graph,
        dynamics="spiking",
        seed=7,
        enable_plasticity=True,
        suppress_behavior_drive=True,
    )
    assert ctrl_mb_spk._num_nodes > 1500
    for _ in range(6):
        for _ in range(25):
            ctrl_mb_spk.step(
                {"walker/touch": np.ones(6), "odor_a": 1.0, "reward_da": 1.0}, dt=0.004
            )
        for _ in range(65):
            ctrl_mb_spk.step({"walker/touch": np.ones(6)}, dt=0.004)
        for _ in range(25):
            ctrl_mb_spk.step(
                {"walker/touch": np.ones(6), "odor_b": 1.0, "punish_da": 1.0}, dt=0.004
            )
        for _ in range(65):
            ctrl_mb_spk.step({"walker/touch": np.ones(6)}, dt=0.004)
    for _ in range(40):
        ctrl_mb_spk.step({"walker/touch": np.ones(6), "odor_a": 1.0}, dt=0.004)
    assert ctrl_mb_spk.rate("DNp09_L") > ctrl_mb_spk.rate("MDN")
    for _ in range(45):
        ctrl_mb_spk.step({"walker/touch": np.ones(6)}, dt=0.004)
    for _ in range(40):
        ctrl_mb_spk.step({"walker/touch": np.ones(6), "odor_b": 1.0}, dt=0.004)
    assert ctrl_mb_spk.rate("MDN") > ctrl_mb_spk.rate("DNp09_L")

    # 6. Closed-loop motor-output verification (grooming T1 lift + T2/T3 claw adhesion; Giant Fiber jump claw release)
    ctrl_g_out = FlyBrainController(
        spec, behavior="grooming", dynamics="spiking", seed=7
    )
    act_g = np.zeros(spec.shape, dtype=np.float64)
    for _ in range(55):
        act_g = ctrl_g_out.step(obs_grounded, dt=0.002)
    act_names = spec.name.split("\t")
    idx_map = {n: i for i, n in enumerate(act_names)}
    assert act_g[idx_map["adhere_claw_T1_left"]] == 0.0
    assert act_g[idx_map["adhere_claw_T2_left"]] > 0.8
    assert act_g[idx_map["coxa_T1_left"]] > 0.4

    ctrl_esc_out = FlyBrainController(
        spec, behavior="escape_takeoff", dynamics="spiking", seed=7
    )
    act_e = np.zeros(spec.shape, dtype=np.float64)
    for _ in range(110):
        act_e = ctrl_esc_out.step(obs_grounded, dt=0.002)
    assert all(
        act_e[idx_map[f"adhere_claw_{leg}"]] == 0.0
        for leg in ("T1_left", "T1_right", "T2_left", "T2_right", "T3_left", "T3_right")
    )
    assert act_e[idx_map["femur_T2_left"]] > 1.0

    # 7. High-dimensional 2,000-KC APL-sparsified Mushroom Body in 1,534-N LIF Spiking mode
    ctrl_2k = FlyBrainController(
        spec,
        behavior="walk_imitation",
        connectome=mb_spk_graph,
        dynamics="spiking",
        seed=7,
        enable_plasticity=True,
        suppress_behavior_drive=True,
        sparse_mb_kcs=2000,
        sparse_mb_pns=50,
    )
    assert ctrl_2k.sparse_mb is not None
    rng_o = np.random.default_rng(101)
    odors = [
        rng_o.exponential(1.0, size=50) * (rng_o.random(50) < 0.35) for _ in range(4)
    ]
    for _ in range(5):
        for oi, pat in enumerate(odors):
            da_key = "reward_da" if oi < 2 else "punish_da"
            for _ in range(24):
                ctrl_2k.step(
                    {"walker/touch": np.ones(6), "odor_pattern": pat, da_key: 1.0},
                    dt=0.004,
                )
            for _ in range(55):
                ctrl_2k.step({"walker/touch": np.ones(6)}, dt=0.004)
    # Check 5% KC sparsity & clean multi-odor discrimination in LIF spiking mode
    vals = []
    for pat in odors:
        for _ in range(38):
            ctrl_2k.step({"walker/touch": np.ones(6), "odor_pattern": pat}, dt=0.004)
        vals.append(ctrl_2k.rate("DNp09_L") - ctrl_2k.rate("MDN"))
        for _ in range(40):
            ctrl_2k.step({"walker/touch": np.ones(6)}, dt=0.004)
    mb_diag = ctrl_2k.sparse_mb.diagnostics()
    assert mb_diag["n_kcs"] == 2000
    assert mb_diag["plastic_synapses"] == 4000
    assert vals[0] > 0.2 and vals[1] > 0.2
    assert vals[2] < -0.2 and vals[3] < -0.2
