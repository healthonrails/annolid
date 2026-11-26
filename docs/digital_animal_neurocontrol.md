# Digital Animal Behavior, 3D Biomechanical Twins & Neural Circuit Control

Annolid provides a unified, zero-heavy-dependency pure-NumPy stack (`annolid.simulation`) bridging **markerless 2D/3D pose tracking**, **anatomically constrained 3D biomechanics**, and **mechanistic neural circuit controllers** across invertebrate (*Drosophila*) and mammalian (rat/mouse) model organisms.

For the full methods, benchmarks, and publication manuscript, see [`paper/pixels_to_circuits/pixels_to_circuits.md`](https://github.com/healthonrails/annolid/blob/main/paper/pixels_to_circuits/pixels_to_circuits.md).

---

## 1. Multi-View 3D Triangulation, Depth Lifting & Rigid-Bone Constraints

`annolid.simulation.lifting` supports calibrated multi-camera triangulation, robust monocular depth lifting, and anatomical skeleton projection without requiring SciPy, PyTorch, or MuJoCo:

```python
from annolid.simulation import (
    RAT_V2_RIGID_BONE_EDGES,
    CameraCalibration,
    enforce_anatomical_skeleton_constraints,
    lift_pose_frames_with_depth,
    triangulate_multiview_pose_frames,
)

# 1. Calibrate cameras with intrinsics, extrinsics, and Brown-Conrady distortion
cam0 = CameraCalibration.from_look_at(
    name="cam0",
    position=[0.7, 0.0, 0.5],
    target=[0.0, 0.0, 0.05],
    fx=820.0,
    fy=820.0,
    cx=320.0,
    cy=240.0,
    distortion=(0.08, -0.02, 0.001, -0.001),
)

# 2. Confidence-weighted DLT + leave-one-out outlier rejection + Huber Gauss-Newton
frames_3d, tri_summary = triangulate_multiview_pose_frames(
    views_by_camera,
    calibrations,
    outlier_reproj_thresh_px=10.0,
)

# 3. Rigid-bone PBD manifold projection (keeps articulated paw chains unconstrained)
clean_3d, skel_diag = enforce_anatomical_skeleton_constraints(
    frames_3d,
    RAT_V2_RIGID_BONE_EDGES,
    iterations=8,
    temporal_smooth_weight=0.15,
)
```

For monocular depth sidecars via CLI (`annolid-run predict simulation_runner`), choose between `--depth-sample-mode nearest`, `bilinear` (subpixel smooth depth), and `median_patch` (spike-resistant 3×3 neighborhood median):

```bash
annolid-run predict simulation_runner \
  --backend rat_ncap \
  --input pose.ndjson \
  --depth-ndjson depth.ndjson \
  --depth-sample-mode median_patch \
  --mapping annolid/configs/rat_ncap_template.yaml \
  --out-ndjson rat_fit.ndjson
```

---

## 2. Real-to-Sim Rat v2 31-DOF IK, Exact KKT Spring-Muscle ID & Spinal Inference

`annolid.simulation.biomechanics` (automatically invoked by `RatNCAPAdapter.fit_2d()` and `fit_3d()`) converts observed 3D keypoint sequences into physically bounded joint angles, antagonistic flexor/extensor activations, mechanical power, and blind spinal circuit phenotype diagnostics:

```python
from annolid.simulation import (
    infer_spinal_circuit_phenotype,
    solve_rat_v2_ik_sequence,
)

result = solve_rat_v2_ik_sequence(
    clean_3d,
    dt=0.02,
    interface_type="flexor_extensor",
)
print(result["summary"]["mean_ik_rmse_m"])
print(result["inverse_dynamics"]["torque_residual_rmse_nm"])
print(result["spinal_circuit_inference"]["phenotype_class"])
print(result["spinal_circuit_inference"]["v0_integrity_index"])
```

Key outputs:

- **31-DOF `qpos` & `qvel`** strictly within `RAT_V2_JOINT_RANGES` (0 joint-limit violations),
- **Exact KKT antagonistic `(flx_activation, ext_activation)`**, joint `torque`, `stiffness`, and positive/negative `mechanical_power`,
- **Latent spinal circuit metrics**: `dominant_gait`, `step_frequency_hz`, `duty_factor`, `phase_offsets_rel_fr`, `phase_locking_value`, `estimated_brainstem_command`, `v0_integrity_index`, `v3_integrity_index`, and automated `phenotype_class` (`intact_stand`, `intact_walk`, `intact_trot`, `intact_bound`, `v0_commissural_deficit_phenotype`, `v3_synchrony_deficit_phenotype`, `gait_transition`).

---

## 3. *Drosophila* LIF Connectome Optogenetics, Silencing & Mushroom Body Plasticity

`FlyBrainController` (`annolid.simulation.fly_brain`) supports in-silico causal circuit perturbations and online three-factor dopaminergic associative learning across both compact rate circuits and expanded multi-neuron LIF spiking connectomes:

```python
from annolid.simulation import (
    FlyBrainController,
    build_plastic_fly_connectome,
    build_spiking_fly_connectome,
)

# 1. Optogenetic Chrimson stimulation & Kir2.1 silencing in 1,470-neuron LIF spiking mode
ctrl = FlyBrainController(
    action_spec,
    behavior="walk_imitation",
    connectome=build_spiking_fly_connectome(seed=7),
    dynamics="spiking",
    synaptic_noise_std=0.2,  # Dale's-law sign-preserving multiplicative noise
)
ctrl.stimulate_cell_type("MDN", current=2.4, duration_sec=0.12)
ctrl.set_ablation("VNC_MDN_Inh", scale=0.0)  # full Kir2.1 silencing

# 2. Three-factor PAM/PPL1 dopaminergic Mushroom Body conditioning
mb_ctrl = FlyBrainController(
    action_spec,
    connectome=build_plastic_fly_connectome(spiking=False, seed=7),
    dynamics="rate",
    enable_plasticity=True,
    suppress_behavior_drive=True,
)
mb_ctrl.step({"walker/touch": [1]*6, "odor_a": 1.0, "reward_da": 1.0}, dt=0.004)
print(mb_ctrl.plasticity_weights())
```

---

## Related Documentation

- [Simulation and FlyBody](simulation_flybody.md)
- [Rat NCAP 3D Simulation](simulation_rat_ncap.md)
- [Two-Mouse 3D Simulation](two_mice_simulation.md)
- [Sim-to-Real Studio (3D Simulation → Colab)](sim2real_colab_workflow.md)
