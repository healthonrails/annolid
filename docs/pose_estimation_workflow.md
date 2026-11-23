# Pose Estimation Workflow (SLEAP & DeepLabCut Parity)

Annolid provides an end-to-end pose estimation workflow spanning visual-diversity frame extraction, rapid multi-animal keypoint labeling, model training, DeepLabCut/SLEAP-grade evaluation, offline trajectory filtering and gap interpolation, active-learning outlier refinement, calibrated pose kinematics, and labeled skeleton video export.

For bidirectional format conversion with DeepLabCut, Lightning Pose, SLEAP (`.slp`, `.pkg.slp`, `.analysis.h5`), SimBA, `movement`, and COCO Keypoints, see [DeepLabCut, SLEAP & Pose Interoperability](pose_interoperability.md).

---

## 1. Define a Skeleton (`PoseSchema`)

Choose a built-in species preset in **Project Wizard → Pose Schema** or **Settings → Pose & Keypoint Manager**:

- **Rodents**: `Mouse (5 keypoints)`, `Mouse CalMS21 / MARS (7 keypoints)`, `Mouse (9 keypoints)`, `Mouse (11 keypoints)`, `Mouse MaBE2022 (12 keypoints)`, `Rat (8 keypoints)`
- **Insects & Fish**: `Fly (5 keypoints)`, `Fly Full-Body (12 keypoints)`, `Fish (5 keypoints)`, `Zebrafish Spine (5 keypoints)`
- **Human**: `Human COCO (17 keypoints)`

Skeleton definitions store canonical keypoint ordering, anatomical edges, left/right `symmetry_pairs`, `flip_idx`, optional custom 2D preview layouts, and multi-animal instance prefixes in `pose_schema.json` / `.yaml`. In the **Keypoint Sequencer** dock, drag any node in the skeleton preview to arrange a custom 2D anatomical layout (optionally over a loaded **Ref Image…**), or click two nodes to toggle an edge.

---

## 2. Extract Diverse Training Frames (`kmeans` & `uniform`)

Instead of sampling only random or high-motion frames, use visual-diversity **k-means clustering** (equivalent to DeepLabCut's default `algo='kmeans'` and SLEAP's image-clustering suggestions) or **uniform** spacing:

- **GUI**: Open **Video Tools → Extract frames** (`Ctrl+Shift+E`), select **`kmeans`** (recommended for pose diversity) or **`uniform`**, and set the frame count.
- **CLI**:

```bash
# Visually diverse frames via deterministic k-means++ clustering
annolid-run extract-frames trial.mp4 --algo kmeans --num-frames 50 -o ./frames_kmeans

# Evenly spaced frames across a subclip
annolid-run extract-frames trial.mp4 --algo uniform --num-frames 40 \
  --start-seconds 10 --end-seconds 120 -o ./frames_uniform
```

---

## 3. Annotate & Proofread Keypoints in the GUI

- **Keypoint Sequencer** (`Ctrl+Shift+K` or **View → Docks → Keypoint Sequencer**):
  - Lock keypoints in your desired click order, then place points sequentially.
  - Use **Skip (Absent)** to advance past out-of-frame landmarks without placing a dummy point, and **Prev KP** to step backward.
  - Enable **Auto-advance to next instance on wrap** in multi-animal scenes to step automatically from `mouse_1` to `mouse_2` after completing a skeleton.
- **Visibility, Whole-Skeleton Selection & Rigid Transforms**:
  - **Shift+Click** places an occluded keypoint (`v=1`, rendered with dashed bones and hollow circles).
  - Right-click any selected keypoint on the canvas (or in the Labels panel) to **Select Whole Pose Instance** (selects all keypoints of that animal so dragging moves the entire skeleton together), **Rotate Pose Left/Right (±15°)** / **Scale Pose Up/Down (±10%)** around the skeleton centroid, **Toggle Keypoint Visibility**, **Mark Keypoint Visible**, or **Mark Keypoint Occluded** (`Ctrl+Alt+I` / `V` / `O`).
- **Live Canvas Pose Controls**:
  - Open **Analysis → Pose Estimation Workbench…** (`Ctrl+Shift+P`) to switch live canvas coloring between **By Instance (Animal)**, **By Bodypart (DLC)**, and **By Confidence (Score)**, adjust the live **Canvas p-cutoff** threshold, and toggle **Show keypoint labels on canvas**.

---

## 4. Evaluate Pose Models (`eval-pose`)

Compare any ground-truth and predicted pose datasets across LabelMe directories, AnnotationStore sidecars, SLEAP `.slp` / `.analysis.h5`, DeepLabCut `.csv` / `.h5` / training projects, COCO Keypoints JSON, or Annolid tracking CSVs.

Metrics computed with multi-animal Hungarian matching:

- **DeepLabCut metrics**: Pixel **RMSE** and **MAE** across all keypoints and restricted to `score >= pcutoff`.
- **SLEAP localization metrics**: Median (`p50`), `p75`, `p90`, `p95`, and max Euclidean error (px).
- **PCK**: Fixed-pixel PCK (`2, 4, 8, 16 px`) and instance-scale-normalized PCK (`5%, 10%, 20%`).
- **COCO / SLEAP OKS**: `mean_oks`, `mAP` (`0.50:0.05:0.95`), `AP50`, `AP75`, and `AR` for arbitrary animal keypoint counts.
- **Detection & Symmetry**: Keypoint/instance precision, recall, F1, and multi-animal **Left/Right symmetry swap rate**.
- **Per-Keypoint & Worst-Frame Tables**: Ranked highest-error frames with double-click canvas seeking in **Pose Estimation Workbench → 1. Evaluate Pose**.

```bash
annolid-run eval-pose ./gt_labelme ./pred_labelme \
  --pose-schema ./pose_schema.json \
  --pcutoff 0.6 \
  --output-dir ./pose_eval_report
```

Writes `pose_eval_report.json`, `pose_eval_per_keypoint.csv`, `pose_eval_worst_frames.csv`, and a self-contained `pose_eval_report.html` with inline SVG error charts and PCK CDF curves.

---

## 5. Active-Learning Outlier Refinement (`extract-outliers`)

Automatically mine the highest-risk frames from model predictions (combining low keypoint confidence, per-keypoint velocity jumps, anatomical bone stretch/collapse, and missing keypoints) spaced by `--min-frame-gap`:

```bash
annolid-run extract-outliers ./pred_labelme \
  --video trial.mp4 \
  --pose-schema ./pose_schema.json \
  --num-frames 20 \
  --min-frame-gap 15 \
  --min-confidence 0.6 \
  -o ./refinement_batch
```

Annolid extracts the matching PNG frames, writes pre-populated editable LabelMe JSONs (plus `pose_schema.json` and `outlier_frames_manifest.json`), and lets you open `./refinement_batch` directly in the GUI (or via **Pose Estimation Workbench → 3. Active Learning Outliers**), drag only the mispredicted points, and merge the corrected frames back into your training set.

---

## 6. Filter, Interpolate & Repair Identities (`filter-pose`, `swap-pose`)

### Offline Pose Filtering & Bounded Gap Interpolation

Clean noisy predictions using p-cutoff masking, velocity-jump and anatomical bone-length outlier rejection, bounded `linear`/`cubic`/`nearest` gap interpolation, and zero-phase `savgol`, `median`, `one_euro`, or `moving_average` smoothing. When filtering a LabelMe directory, all non-point shapes (animal polygons, masks, and zones) are preserved intact.

```bash
annolid-run filter-pose ./pred_labelme \
  --pose-schema ./pose_schema.json \
  --min-confidence 0.5 \
  --max-jump-px 45 \
  --enforce-bone-bounds \
  --interpolate linear --max-gap 5 \
  --method savgol --window 5 \
  -o ./pred_filtered
```

### Multi-Animal Pose Identity Swap Repair

Swap two animals (updating both pose keypoints and parent polygons/bboxes consistently) from frame `t` forward — via **Pose Estimation Workbench → 5. Labeled Video & Swap IDs**, the canvas right-click menu (**Swap Pose Instances Forward…**), or CLI:

```bash
# Dry-run preview first
annolid-run swap-pose ./pred_labelme \
  --start-frame 420 --instance-a mouse_1 --instance-b mouse_2 --dry-run

# Apply swap from frame 420 to end of video
annolid-run swap-pose ./pred_labelme \
  --start-frame 420 --instance-a mouse_1 --instance-b mouse_2
```

Automatic repair via `annolid-run repair-tracks` (temporal continuity or Identity Governor) also keeps child pose keypoints synchronized when parent animal identities are corrected, and `annolid-run track-qc` (and **Analysis & Reporting Studio → Track QC & Gaps**) checks stretched bones, collapsed bones, and per-keypoint velocity spikes.

---

## 7. Calibrated Pose Kinematics (`pose-kinematics`)

Compute skeleton-level kinematics with optional orthographic (`--px-per-cm`) or planar homography (`--calibration`) calibration:

- Per-keypoint speed, acceleration, distance, and coverage (`keypoint_kinematics.csv`)
- Per-bone length distributions (`bone_lengths.csv`)
- Auto-derived 3-point joint angles and angular velocities (`joint_angles.csv`, `joint_angles_timeseries.csv`)
- Body heading and **egocentric body-aligned coordinates** (`egocentric_pose.csv`)
- Inter-animal keypoint-pair distances, close contact, and directed bearings (`dyadic_keypoints.csv`)

```bash
annolid-run pose-kinematics ./pred_filtered \
  --pose-schema ./pose_schema.json \
  --fps 30 --px-per-cm 12.5 \
  --output-dir ./pose_kinematics_out
```

---

## 8. Render Labeled Skeleton Videos (`render-pose-video`)

Render deterministic, full-resolution skeleton overlay videos from any LabelMe, AnnotationStore, SLEAP, DeepLabCut, or Tracking CSV source (with optional raw-vs-filtered comparison markers):

```bash
annolid-run render-pose-video trial.mp4 ./pred_filtered \
  -o ./trial_labeled_pose.mp4 \
  --pose-schema ./pose_schema.json \
  --comparison ./pred_labelme \
  --color-mode instance \
  --min-confidence 0.5 \
  --trail-length 15 \
  --draw-labels
```

Supported `--color-mode` options: `instance` (per-animal high-contrast palette), `bodypart` (DLC-style per-keypoint palette), and `confidence` (red → amber → green score heatmap).
