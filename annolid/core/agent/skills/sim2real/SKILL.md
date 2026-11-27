---
name: sim2real
description: Run Annolid Sim-to-Real Studio workflows to analyze real lab videos, generate synthetic 3D Mouse/FlyBody datasets, train YOLO models, and close the sim-to-real gap iteratively.
metadata: '{"annolid":{"always":false,"tags":["sim2real","sim-to-real","synthetic","synthetic data","mujoco","mouse 3d","flybody","domain randomization","yolo_seg","yolo_pose","yolo_detect","real gap"]}}'
---

# Sim-to-Real Studio Workflow

Use this skill when the user asks to generate synthetic training data from 3D animal simulators (Mouse 3D or FlyBody 3D), analyze real lab videos for domain matching, train models on synthetic datasets, or run iterative sim-to-real evaluation loops.

## Preferred Tool: `sim2real_workflow`

Use the typed `sim2real_workflow` tool (CLI equivalent: `annolid-run sim2real ...`).

### End-to-End Lifecycle Actions

1. **Check project state**:
   - `sim2real_workflow(action="status", project_dir="...")`
2. **Analyze real example videos** (extracts arena, camera view, lighting, and species domain cues):
   - `sim2real_workflow(action="analyze", project_dir="...", example_videos=["/path/to/clip1.mp4"], species="mouse"|"fly")`
3. **Generate synthetic dataset**:
   - `sim2real_workflow(action="generate", project_dir="...", species="mouse"|"fly", task="yolo_seg"|"yolo_pose"|"yolo_detect", export_level="instances"|"parts", arena?, camera_view?, lighting?, coat_pair?, scenarios?, frames_per_scenario?)`
4. **Iterate (generate -> train -> evaluate loop)**:
   - `sim2real_workflow(action="iterate", project_dir="...", iterations=2, train_backend="colab"|"local", colab_backend="auto", accelerator="L4", epochs=50, imgsz=640, keep_runtime=true)`
5. **Evaluate trained weights on real clips**:
   - `sim2real_workflow(action="evaluate", project_dir="...", example_videos=[...], weights="...", conf=0.25)`
6. **Apply recommended domain-randomization adjustments**:
   - `sim2real_workflow(action="apply_recommendation", project_dir="...")`
7. **Build summary report**:
   - `sim2real_workflow(action="report", project_dir="...")`

## Key Parameters & Provenance Rules

- **Species**: Mouse 3D (`mouse`) or FlyBody 3D (`fly`).
- **Tasks**: `yolo_seg`, `yolo_pose`, `yolo_detect`.
- **Export level**: `instances` (whole-animal masks/boxes) vs `parts` (anatomical parts/keypoints).
- **Fallback provenance (`allow_fallback`)**: Never silently pass `allow_fallback=true` without telling the user! If MuJoCo / 3D renderer dependencies are missing and fallback rendering is used, disclose clearly in the summary that outputs are fallback-rendered rather than true 3D simulator renders.
- Keep `project_dir` inside the writable workspace or allowed roots, and report exact generated dataset paths, checkpoint paths, evaluation metrics, and report files.
