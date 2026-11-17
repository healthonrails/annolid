# Instance Body Part Segmentation

Instance Body Part Segmentation decomposes already-tracked whole-animal polygons (from Cutie, SAM2/SAM3, YOLO, or manual annotation) into non-overlapping anatomical sub-regions such as `head`, `torso`, `tail`, limbs, or wings.

Unlike full-frame segmentation trackers—which operate on the entire image and treat every polygon as a competing top-level track—Instance Body Part Segmentation operates strictly inside each tracked parent instance's padded bounding-box ROI and enforces hierarchical containment:

- **Zero cross-animal bleed**: pixels outside a parent animal's mask are hard-masked out, even during close social contact or huddling.
- **Merged-parent occlusion recovery**: when two huddling animals temporarily collapse into a single bloated parent mask (or carry distinct multi-animal keypoints inside one blob), edge-weighted geodesic + Lab + area-balanced splitting decomposes the parent blob and freezes EMA prototype drift.
- **Disjoint intra-instance partition**: sibling body parts inside the same parent instance never overlap.
- **360° circular & coiled posture invariance**: every instance crop is projected into a pose-normalized canonical body frame `(u, v, thickness)` combining continuous 2D directional dipoles (keypoints, prototype activations, temporal pose, and intrinsic appearance contrast for `aspect_ratio < 1.15` circular postures) with an **intrinsic masked geodesic spine** (`tortuosity > 1.18`) for C-shaped, U-shaped, and coiled animals.

---

## Modes and Feature Backends

### Segmentation Modes (`--mode`)

| Mode | Inputs Required | Best For |
| --- | --- | --- |
| `auto` (default) | Tracked instance polygons (+ optional seed parts or keypoints) | Automatically selects `seeded` when manual body-part polygons exist, `keypoint_guided` when keypoints exist inside the instance, or `unsupervised` otherwise. |
| `seeded` | Manual body-part polygons drawn inside at least one instance on 1+ seed frames | Custom anatomical regions (`snout`, `ear`, `head`, `torso`, `tail`), partial-part labeling (e.g. `head` only), and multi-animal sharing from a single seeded animal. |
| `keypoint_guided` | Tracked whole-instance polygons + keypoints (from DINO-KPSEG, YOLO-Pose, TAPNext, CoWTracker, or manual points) | Converting sparse landmarks and optional `PoseSchema` skeleton bones into dense anatomical part masks without manual part polygons. |
| `unsupervised` | Tracked whole-instance polygons only | Fast axial/morphological decomposition (`anterior`/`posterior` for 2 parts; `head`/`torso`/`tail` for 3 parts) across elongated, circular, and curled postures with zero part or keypoint supervision. |

### Feature Backends (`--backbone`)

| Backend | Dependencies | Throughput & Tradeoffs |
| --- | --- | --- |
| `fast` (default) | OpenCV + NumPy | Runs on CPU in ~1–3 ms per animal crop using illumination-normalized multi-scale CIE Lab, gradient magnitude, and local detail features plus O(1) edge-preserving guided filtering. Recommended default for long videos. |
| `dino` | PyTorch + `transformers` | Extracts dense DINOv3 ViT patch features on each `crop_size × crop_size` instance ROI with foreground mean debiasing. Best semantic invariance across coat patterns or lighting shifts. |
| `hybrid` | PyTorch + `transformers` + OpenCV | Concatenates L2-normalized DINOv3 semantic features with high-resolution Lab/gradient boundary cues. Highest boundary sharpness + semantic stability on GPU/MPS. |

---

## GUI Workflow

1. **Track whole animals first** (for example with `Cutie`, `SAM3`, or `YOLO`) so your video's results folder contains tracked whole-instance polygons (`mouse_1`, `mouse_2`, ...).
2. *(Optional, for `seeded` mode)* On one or more seed frames:
   - Draw body-part polygons (such as `head`, `torso`, `tail`) inside an animal's polygon and save the frame.
   - If your animals share appearance/morphology, seeding **one** animal is sufficient—prototypes are automatically shared across unseeded instances when `share_seed_across_instances` is enabled (the default).
   - If your seeded parts cover less than 75% of the parent mask (for example, labeling only `head`), the remaining body is automatically tracked as an internal unassigned background region and omitted from output shapes so `head` never expands to fill the whole body.
3. In the top toolbar model selector, choose **`Instance Body Part Segmentation`**.
4. Click **Predict** to run body-part segmentation across the video frame range.

---

## CLI Usage

Run body-part segmentation from the terminal with `annolid-run segment-body-parts`:

```bash
# Auto mode with fast CPU crop features and tracking CSV export
annolid-run segment-body-parts /path/to/social_assay.mp4 \
  --mode auto \
  --backbone fast \
  --export-csv

# Keypoint-guided mode with fine limb/wing granularity and a PoseSchema YAML/JSON
annolid-run segment-body-parts /path/to/fly_courtship.mp4 \
  --annotations /path/to/fly_courtship \
  --mode keypoint_guided \
  --keypoint-granularity fine \
  --pose-schema /path/to/pose_schema.yaml \
  --export-csv

# Seeded mode with hybrid DINOv3 + Lab/gradient features and JSON report
annolid-run segment-body-parts /path/to/session.mp4 \
  --mode seeded \
  --backbone hybrid \
  --dino-model facebook/dinov3-vits16-pretrain-lvd1689m \
  --qualify-labels \
  --out /path/to/session/body_part_report.json
```

Key CLI flags:

- `--mode {auto,seeded,keypoint_guided,unsupervised}`: Segmentation strategy (default: `auto`).
- `--backbone {fast,dino,hybrid}`: Per-crop feature extractor (default: `fast`).
- `--crop-size INT`: Working resolution of the longest ROI side per instance (default: `224`).
- `--num-parts INT` / `--parts a,b,c`: Part count and names for `unsupervised` mode (default: `3` -> `head,torso,tail`).
- `--keypoint-granularity {coarse,fine,per_keypoint}`: Automatic lexical grouping of keypoint names when no explicit `--keypoint-part-map` is supplied (default: `coarse`).
- `--partition-full-instance` / `--partial-parts`: Force full-instance partitioning or preserve unseeded body regions.
- `--no-split-merged-parents` / `--no-recover-merged-instances`: Opt out of automatic geodesic splitting or missing-partner recovery when huddling animals merge into one bloated parent mask.
- `--qualify-labels`: Write shape `label` as `<instance>:<part>` (e.g. `mouse_1:head`) instead of bare `<part>` with `group_id` + `parent_instance` metadata.
- `--export-csv`: Export `<results_dir>_tracking.csv` after segmentation completes.

---

## Python API

```python
from annolid.segmentation import (
    BodyPartSegmentationConfig,
    InstanceBodyPartVideoProcessor,
)

config = BodyPartSegmentationConfig(
    mode="auto",
    feature_backend="fast",
    crop_size=224,
    guided_radius=4,
    split_merged_parent_masks=True,
)

processor = InstanceBodyPartVideoProcessor(
    "/path/to/video.mp4",
    results_folder="/path/to/video",
    body_part_config=config,
)
summary_message = processor.process_video_frames(start_frame=0)
processor.cleanup()
print(summary_message, processor.last_summary)
```

For single-frame NumPy arrays in custom pipelines, use `InstanceBodyPartSegmenter.fit_seed_frame(...)` and `InstanceBodyPartSegmenter.segment_frame(...)` directly.

---

## Direct Cutie VOS Propagation with Compact SAM3+DINOv3 Distilled Weights

When propagating seeded whole-instance and/or anatomical body-part masks directly with Cutie's temporal memory engine (`InferenceCore`), Annolid ships a directly compressed and Dual-Teacher (`SAM 3.1` + `DINOv3 ViT-L/16`) distilled Cutie checkpoint while keeping `CUTIE.load_weights` backward-compatible with legacy dense checkpoints:

- **Default & SAM3 Body-Parts Cutie (`Cutie` / `Cutie (SAM3 Body Parts)`)**: loads `annolid/segmentation/cutie_vos/weights/cutie-base-mega.pth` (mirrored at `cutie-sam3-bodyparts.pth`), combining algebraic 48-layer `Conv+BN` folding, HOSVD Tucker-2 spatial-channel factorization, structured `FFN` pruning, and mixed-precision `FP16`/`FP32` serialization (`35.02M -> 15.76M` parameters, `-55.0%`; `133.94 MB -> 30.26 MB` on disk, `4.43x` smaller; `1.64x` faster cold load; `1.48x` faster CPU propagation and up to `2.72x` faster on Apple Silicon `mps`) while improving held-out body-part `J&F` from `0.7798` to `0.8127` (`+0.0329` macro-average across 5 suites).
- **Fresh-Install & Frozen-Bundle Auto-Bootstrap**: On fresh clones, pip wheels, or frozen desktop apps (`Annolid.app`, which stays strictly read-only by routing downloads/caches to `~/.annolid/workspace/downloads`), selecting either `Cutie` or `Cutie (SAM3 Body Parts)` automatically resolves in order:
  1. Local `weights/` or `~/.annolid/workspace/downloads/` cache,
  2. Explicit `ANNOLID_CUTIE_UPGRADED_WEIGHTS_URL` + `ANNOLID_CUTIE_UPGRADED_WEIGHTS_SHA256` override,
  3. Canonical GitHub Release asset (`SAM3_BODYPARTS_CUTIE_RELEASE_URL` verified against `SAM3_BODYPARTS_CUTIE_SHA256`),
  4. Seamless fallback to upstream `cutie-base-mega.pth` (MD5-verified) + automatic `< 0.5s` `Conv+BN` folding & HOSVD Tucker-2 `30.26 MB` compression.
- **Immutable Upstream Backup**: `annolid/segmentation/cutie_vos/weights/cutie-base-mega.orig.pth` preserves the untouched upstream Cutie Base Mega checkpoint (`MD5 a6071de6136982e396851903ab4c083a`) for reproducible head-to-head comparison or legacy fallback.

### Selecting Custom Cutie Weights

- **GUI**: Select **`Cutie`** or **`Cutie (SAM3 Body Parts)`** in the top toolbar AI model selector.
- **Python API**:

```python
from annolid.segmentation.cutie_vos.engine import CutieEngine
from annolid.segmentation.cutie_vos.video_processor import CutieVideoProcessor
from annolid.tracking.configuration import CutieDinoTrackerConfig

engine = CutieEngine(model_weights_path="cutie-sam3-bodyparts.pth")
processor = CutieVideoProcessor(
    "/path/to/video.mp4",
    model_weights_path="cutie-sam3-bodyparts.pth",
)
tracker_cfg = CutieDinoTrackerConfig(
    cutie_weights_path="cutie-sam3-bodyparts.pth",
)
```

### Reproducing Direct Weight Compression, Distillation, and Benchmarks

```bash
# Back up cutie-base-mega.orig.pth, compress via Conv+BN folding + HOSVD Tucker-2,
# and distill SAM 3.1 + DINOv3 ViT-L/16 into cutie-base-mega.pth & cutie-sam3-bodyparts.pth
.venv/bin/python scripts/upgrade_cutie_weights_with_sam3.py

# Evaluate cutie-base-mega.orig.pth vs cutie-base-mega.pth on 9 held-out suites (accuracy + FPS)
.venv/bin/python scripts/evaluate_cutie_bodyparts_benchmark.py
```

---

## Saved Data & Downstream Compatibility Contract

To protect LabelMe compatibility, Qt flag editors, and whole-animal post-processing tools, body-part polygons follow an explicit metadata contract (`annolid.utils.body_part_shapes`):

1. **Boolean-only `flags`**:
   - `shape["flags"]` stores `{"is_body_part": true}` (strictly `bool`, preventing non-boolean Qt checkbox errors).
   - Recovered parent polygons from split huddling blobs carry `{"recovered_merged_parent": true}`.
2. **Hierarchical linkage on shape root / `other_data`**:
   - `semantic_type`: `"body_part"`
   - `parent_instance` & `instance_label`: parent animal identity (e.g. `"mouse_1"`)
   - `part_label`: canonical anatomical part name (e.g. `"head"`)
   - `group_id`: inherited from the parent animal's `group_id` when present
   - `score`: mean part assignment probability in `[0, 1]`
   - `area_ratio`: fraction of parent mask area occupied by the part
3. **Downstream tool behavior**:
   - **`AnnotationAdapter` (`annolid.tracking`)**: Attaches body parts to `InstanceState.body_parts` rather than creating extra top-level animals, and round-trips them cleanly on save.
   - **Tracking CSV (`labelme2csv`)**: Exports whole animals under `<instance>` (e.g. `mouse_1`) and child body parts under `<instance>:<part>` (e.g. `mouse_1:head`) so trajectories, areas, and centroids are queryable without colliding with whole-animal rows.
   - **Track QC (`annolid-run track-qc`)**: Excludes child body parts from whole-animal count and overlap false positives, while running dedicated intra-instance checks:
     - `body_part_outside_instance` (warning when >15% of a body-part polygon spills outside its parent instance polygon),
     - `body_part_overlap` (warning when sibling parts of the same parent overlap with IoU >= 0.10),
     - `body_part_missing_parent` (warning when a body-part polygon references a `parent_instance` that is absent on that frame).
   - **Identity Repair & Governor (`temporal_identity_repair`, `identity_governor`)**: Ignores child body-part shapes when computing whole-animal trajectories and swap repairs.
