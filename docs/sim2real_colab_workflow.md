# Sim-to-Real Studio: Example Videos → 3D Mouse/Fly Simulation → Colab Iteration

Annolid's **Sim-to-Real Studio** (`annolid-run sim2real` and **AI & Models → Sim-to-Real Studio (3D Sim → Colab)…**) lets you bootstrap and iteratively improve segmentation, pose, and detection models starting directly from real example videos—with zero or minimal manual labels.

## End-to-End Workflow

1. **Add real example videos (or an image folder)** and run **Analyze Environment**:
   - Extracts resolution, FPS, median arena background (`reference_background.png`), and sample montage (`example_samples.png`).
   - Infers species (`mouse` vs `fly`, or overrides with `--species`), illumination/IR profile (`default`, `ir_night`, `calms_gray`, `dim_warm`), arena geometry (`open_field`, `circular_open_field`, `home_cage`, `light_dark_box`, mazes), camera perspective, coat appearance, wire-bar occluders, and optical blur/noise/vignette statistics.
2. **Generate an environment-matched 3D synthetic dataset**:
   - **Mouse 3D (`threejs_two_mice_3d`)**: drives Annolid's bundled Three.js Two-Mice 3D scene via Playwright Chromium with 10 neuroscience arenas, 6 camera perspectives, 4 lighting presets, 5 coat pairs, wire-bar occluders, headstages, exact WebGL 3D ID-pass segmentation masks (`instances` or `head`/`torso`/`limbs`/`tail` parts), and surface-occlusion-aware 13-keypoint pose labels.
   - **Fly 3D (`flybody_mujoco`)**: drives Annolid's FlyBody MuJoCo/dm_control simulation (automatically discovering `.venv311` or your FlyBody runtime) across `walk_imitation`, `forage_walk`, `grooming`, `backward_walk`, `escape_takeoff`, `flight_imitation`, and `template_task`, rendering pixel-exact geom-ID body-part masks (`head`, `thorax`, `abdomen`, `wings`, `legs`) and 11 projected 2D anatomical keypoints.
   - Applies label-preserving **photometric domain matching** (Reinhard Lab color/contrast transfer, IR/grayscale matching, optical blur, sensor noise, radial vignette), optional **real arena background compositing**, configurable **domain randomization**, and optional **sim + real LabelMe frame mixing**.
   - Exports portable Ultralytics YOLO datasets (`yolo_seg`, `yolo_pose`, `yolo_detect` with relative `data.yaml`) and LabelMe JSONs.
3. **Train / fine-tune on Google Colab (or locally)**:
   - Uses your signed-in personal Google Colab account (`annolid-run colab auth login --method personal`), reusing a single warm Colab GPU session across multi-iteration loops and cleanly stopping it afterward unless `--keep-runtime` is set.
   - Automatically warm-starts Iteration 2+ from the previous iteration's `best.pt`.
4. **Evaluate on real example videos & iterate**:
   - Extracts compact evaluation clips from your example videos, runs `yolo_labelme` inference on Colab/locally, parses training `results.csv` (`mAP@50`, losses), and computes real-video metrics (detection coverage, mean/P10 confidence, expected subject-count accuracy, frame-to-frame temporal IoU stability, plus optional supervised mIoU/Precision/Recall/F1 if you provide labeled validation frames).
   - Generates deterministic **Diagnosis & Next-Iteration Recommendations** that adjust scenarios, real-background compositing, photometry strength, domain randomization, frame counts, `epochs`, `imgsz`, and checkpoint warm-starting.
5. **Write reproducible reports**:
   - Produces `reports/sim2real_report.md`, `reports/sim2real_report.json`, and self-contained `reports/sim2real_report.html` with inline SVG metric trend charts and visual contact sheets comparing real example frames, matched 3D synthetic frames, and predicted overlays.

## Quick Start (CLI)

### 1. Analyze your example video(s)

```bash
annolid-run sim2real analyze \
  --video /path/to/lab_recording.mp4 \
  --project ./sim2real_mouse_project \
  --species auto \
  --task yolo_seg \
  --export-level instances
```

### 2. Generate an environment-matched synthetic dataset (optional standalone step)

```bash
# Mouse 3D (Three.js + Playwright Chromium)
annolid-run sim2real generate \
  --project ./sim2real_mouse_project \
  --frames-per-scenario 24

# Fly 3D (FlyBody MuJoCo via .venv311 / installed FlyBody runtime)
annolid-run sim2real generate \
  --project ./sim2real_fly_project \
  --species fly \
  --scenarios walk_imitation forage_walk grooming \
  --composite-bg \
  --frames-per-scenario 24
```

### 3. Run multi-iteration closed-loop training & evaluation on Google Colab

Ensure you are signed in to your personal Google Colab account first:

```bash
annolid-run colab auth login --method personal
```

Then launch a 2-iteration closed loop on a Colab T4 GPU:

```bash
annolid-run sim2real iterate \
  --project ./sim2real_mouse_project \
  --iterations 2 \
  --train-backend colab \
  --gpu T4 \
  --epochs 20 \
  --imgsz 640
```

### 4. Inspect status or regenerate reports

```bash
annolid-run sim2real status --project ./sim2real_mouse_project
annolid-run sim2real report --project ./sim2real_mouse_project
```

## GUI Usage

1. Open **AI & Models → Sim-to-Real Studio (3D Sim → Colab)…** (or click **Sim-to-Real Studio…** in the Google Colab Studio header bar).
2. Add your example videos in **1. Example Videos & Match** and click **Analyze Environment & Match 3D Simulation**.
3. Review or override simulation controls in **2. 3D Sim Dataset** (Mouse 3D or FlyBody 3D, photometry matching, background compositing, optional real LabelMe frames).
4. Run 1 iteration or a multi-iteration loop in **3. Train & Iterate** while monitoring live training loss/mAP curves and logs.
5. Compare iterations and apply advisor suggestions in **4. Iterations & Advisor**, then open the visual HTML/Markdown report in **5. Report**.

## Notes on Renderers & Provenance

- **Mouse 3D** requires Playwright Chromium (`playwright install chromium`).
- **Fly 3D** requires FlyBody + MuJoCo (`docs/simulation_flybody.md` or `.venv311`).
- Neither engine silently substitutes a fallback renderer. If you want to test pipelines in an environment without Chromium or MuJoCo, pass `--allow-fallback` (or check the fallback option in the GUI); the manifest and reports explicitly record `procedural_mouse_fallback` or `procedural_fly_fallback`.
