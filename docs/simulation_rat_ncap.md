# Rat 3D Model, Rat NCAP Spinal Controls & Simulation

Annolid includes built-in support for the **Rat v2** biomechanical body model and the **Rat NCAP** (Neural Circuit Architectural Priors) spinal cord controller from:

> Bhattasali, Cui, Pinto, and Lindsay (2026), *"Neural Circuit Architectural Priors for Quadruped Locomotion"* ([arXiv:2611.23364v1](https://arxiv.org/html/2611.23364v1), [github.com/rat-ncap/rat-ncap](https://github.com/rat-ncap/rat-ncap)).

The implementation runs **zero-dependency out of the box** (pure NumPy in Python and standalone Three.js ES modules in the 3D viewer) while optionally discovering a local `rat-ncap` checkout and `mujoco` installation for MuJoCo XML workflows.

---

## 1. Interactive 3D Rat Simulation (GUI & Three.js)

Open from Annolid's main menu:

- **Tools / View → 3D Examples → Rat NCAP 3D Simulation** (`rat_ncap_html`) — launches the interactive 3D Long-Evans Rat v2 biomechanical rig with live spinal circuit controls.
- **Tools / View → 3D Examples → Rat NCAP Simulation Example** (`rat_ncap_simulation_json`) — opens a pre-generated 3D gait-transition rollout (`stand → walk → trot → bound`) rendered with the 41-bone skinned Long-Evans Rat v2 model, timeline scrubbing, coat switcher, follow camera, and toggleable keypoints/skeleton/trails in the Three.js simulation viewer (the same skinned model also renders saved CLI/Python `rat_ncap` rollouts opened via **Open / Run Simulation in 3D Viewer…**).
- **Tools / View → 3D Examples → Rat NCAP Status…** — inspects built-in Rat NCAP readiness and any optional local `rat-ncap` checkout (`ANNOLID_RAT_NCAP_PATH` or `~/Downloads/rat-ncap`).

### Bundled 3D Rat v2 Skinned Model & License Attribution
- Annolid bundles a converted glTF 2.0 skinned mesh (`annolid/gui/assets/threejs/rat_v2_skin.glb`, 14,754 vertices, 19,608 faces, 41 bones, 55 hinge joints, 8 tendons, and 16 bone-attached anatomical keypoints) together with the three Long-Evans coat textures (`rat_blackhooded.jpg`, `rat_cinnamon.jpg`, `rat_gray.jpg`).
- The rat skin geometry and coat textures are derived from `rat-ncap/rat-ncap` (`rat.skn` and `rat_*.png`) and redistributed under the `rat-ncap` repository's **MIT License**.
- The Three.js viewer (`rat_ncap_model.js` / `rat_ncap_scene.js`) drives all 41 bones via MuJoCo-faithful forward kinematics and renders GPU-skinned anatomical segmentation masks for Sim-to-Real dataset generation.

### In-Viewer Controls
- **Brainstem Speed Command** (`0.00 – 1.00 m/s`) and **Auto Gait Ramp** (`stand → walk → trot → bound`)
- **4 Neuromechanical Interfaces**:
  - `flexor_extensor` (antagonistic linear spring muscles with exact stiffness/equilibrium parameter matching)
  - `position_stiffness` (PD with dynamic stiffness scaling)
  - `position` (fixed-gain PD)
  - `torque` (direct torque)
- **3 Long-Evans Coat Skins**: `black_hooded`, `cinnamon`, `gray`
- **Arenas & Bumpy Terrain**: `treadmill`, `bumpy_terrain` (0–4 cm bump amplitude), `open_field`, `circular_open_field`, `elevated_plus_maze`
- **In-Silico Spinal Perturbations**:
  - `V0v + V0d` commissural activity scale (`1.0 → 0.0`, reproducing the trot-to-bound transition upon V0 ablation)
  - `V3f + V3a` synchrony activity scale (`1.0 → 0.0`, preserving trot)
  - Sign-preserving multiplicative RG synaptic weight noise (`σ ∈ [0, 0.5]`)
  - Lumbar spine lock, muscle effort highlight, and 3D keypoint markers
- **Live Spinal Oscilloscope & Hildebrand Footfall Diagram**: real-time traces of flexor oscillators (`FL/FR/HL/HR`), `V0d`/`V3f` interneurons, stance/swing contact bars, step frequency, step length, duty factor, and positive/negative mechanical power.

---

## 2. Python API & CLI Rollouts

### Python API
```python
from annolid.simulation import (
    RatNCAPConfig,
    RatNCAPController,
    run_rat_ncap_rollout,
)

# Run a closed-loop gait transition or ablation rollout
rollout = run_rat_ncap_rollout(
    steps=180,
    behavior="trot",
    interface="flexor_extensor",
    coat="black_hooded",
    ablation="v0v_v0d",
    ablation_scale=0.0,
)
print(rollout["metadata"]["dominant_gait"])  # "bound"
```

### CLI (`annolid-run` & `rat_ncap_live`)
```bash
# Probe Rat NCAP runtime status
python -m annolid.simulation.rat_ncap_live --probe --json

# Generate a live 3D viewer payload
python -m annolid.simulation.rat_ncap_live \
  --behavior gait_transition \
  --interface flexor_extensor \
  --coat black_hooded \
  --steps 180 \
  --out rat_gait_transition.json

# Run via simulation_runner CLI plugin
annolid-run predict simulation_runner \
  --backend rat_ncap \
  --rollout-steps 180 \
  --behavior trot \
  --interface flexor_extensor \
  --out-ndjson rat_trot.ndjson

# Generate a Rat v2 keypoint-to-site mapping template
annolid-run predict simulation_runner \
  --backend rat_ncap \
  --template-keypoints nose,head,thorax,pelvis,left_forepaw,right_forepaw,left_hindpaw,right_hindpaw \
  --write-mapping-template rat_mapping.yaml
```

---

## 3. Sim-to-Real Synthetic Rat Datasets

Rat NCAP 3D is integrated into **Annolid Sim-to-Real Studio** (`species="rat"`):

```bash
# Analyze real rat footage and infer matching Rat NCAP parameters
annolid-run sim2real analyze --video rat_open_field.mp4 --project ./rat_s2r --species rat

# Generate a YOLO segmentation/pose/detection dataset from Rat NCAP 3D
annolid-run sim2real generate \
  --project ./rat_s2r \
  --species rat \
  --task yolo_pose \
  --rat-coat black_hooded \
  --rat-interface flexor_extensor \
  --scenarios trot walk gait_transition \
  --frames-per-scenario 24
```

Keypoints exported for `yolo_pose` follow the 16-point **Rat v2** skeleton (`nose`, `head`, `left_ear`, `right_ear`, `neck`, `thorax`, `torso_center`, `lumbar`, `pelvis`, `tail_base`, `tail_mid`, `tail_tip`, `left_forepaw`, `right_forepaw`, `left_hindpaw`, `right_hindpaw`).
