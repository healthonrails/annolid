# DINOv3 Keypoint Segmentation (DinoKPSEG)

Annolid includes an experimental keypoint-centric segmentation model that:

- Extracts **frozen DINOv3 dense features** (ViT patch grid).
- Trains a **configurable head** (`conv`, `multitask`, or unified attention `relational`).
- Uses **Gaussian keypoint heatmaps** (or circular masks) as supervision.
- Runs inference via the same prediction pipeline used for YOLO, saving results as LabelMe JSON.

## Training

This trainer consumes either:

- A standard YOLO pose dataset (`data.yaml` with `kpt_shape` and labels in `labels/*.txt`), or
- A native LabelMe dataset (per-image `*.json` next to images, with point/polygon shapes), or
- A COCO keypoints dataset (`images` + `annotations/*.json`).

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --epochs 50
```

### Native LabelMe training

Create a small spec YAML that points to a directory (or JSONL index) of LabelMe JSON files:

```yaml
format: labelme
path: /path/to/dataset_root
train: annotations/train   # dir of LabelMe JSONs (images resolved via imagePath/sidecars)
val: annotations/val
kpt_shape: [4, 3]          # K keypoints, dims (2 or 3)
keypoint_names: [nose, leftear, rightear, tailbase]
```

Then train with:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/labelme_spec.yaml \
  --data-format labelme \
  --epochs 50
```

### Native COCO keypoints training

Create a COCO spec YAML:

```yaml
format: coco
path: /path/to/dataset_root
image_root: .                  # optional; defaults to path
train: annotations/train.json
val: annotations/val.json
# Optional overrides:
# kpt_shape: [27, 3]
# keypoint_names: [nose, left_ear, ...]
```

Then train with:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/coco_spec.yaml \
  --data-format coco \
  --epochs 50
```

Annolid will stage the COCO annotations into YOLO-pose labels automatically inside
the run directory and train with the same DinoKPSEG pipeline.

### One-command aggressive schedule launch config

Generate a ready-to-run training config YAML (useful after dataset split/prep):

```bash
python -m annolid.segmentation.dino_kpseg.dataset_tools train-config \
  --data /path/to/data_split.yaml \
  --output /path/to/configs \
  --schedule-profile aggressive_s
```

You can also audit and precompute directly from COCO specs:

```bash
python -m annolid.segmentation.dino_kpseg.dataset_tools audit \
  --data /path/to/coco_spec.yaml \
  --data-format coco

python -m annolid.segmentation.dino_kpseg.dataset_tools precompute \
  --data /path/to/coco_spec.yaml \
  --data-format coco \
  --model-name facebook/dinov3-vits16-pretrain-lvd1689m \
  --layers -2,-1 \
  --feature-merge concat
```

This writes a config like `/path/to/configs/train_aggressive_s.yaml`. Launch training with:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --config /path/to/configs/train_aggressive_s.yaml
```

CLI flags always override config values, for example:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --config /path/to/configs/train_aggressive_s.yaml \
  --batch 16 \
  --epochs 80
```

LabelMe conventions:

- Keypoints are `shape_type: point` with `label` matching an entry in `keypoint_names`.
- Instances are grouped via `group_id` (all shapes with the same `group_id` belong together).
- Polygons (`shape_type: polygon`) are optional; when present they are used to compute per-instance crops in `--instance-mode per_instance`.

Defaults include Gaussian heatmaps, Dice loss, and a coordinate regression loss. Override as needed:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --mask-type gaussian \
  --heatmap-sigma 3 \
  --dice-loss-weight 0.5 \
  --coord-loss-weight 0.25 \
  --coord-loss-type smooth_l1
```

To train per instance (avoid multi-animal mask unions), use bounding-box crops:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --instance-mode per_instance \
  --bbox-scale 1.25
```

To fuse multiple DINO layers, pass a comma-separated list (features are concatenated):

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --layers -2,-1
```

You can also control how multi-layer features are merged:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --layers -3,-2,-1 \
  --feature-merge mean
```

Optional trainable channel alignment before the keypoint head:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --layers -3,-2,-1 \
  --feature-merge concat \
  --feature-align-dim auto
```

### Single-stage Multitask Head (Step 3)

Enable a shared single-stage multitask head (keypoints + objectness + box + instance-mask branches):

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --head-type multitask \
  --obj-loss-weight 0.25 \
  --box-loss-weight 0.25 \
  --inst-loss-weight 0.25
```

Notes:
- Keypoint supervision and decoding stay unchanged.
- Auxiliary branches are additive; set weights to `0` to disable them.

### Training Schedule Hardening (Step 4)

Recommended stability knobs:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --schedule-profile aggressive_s \
  --ema \
  --ema-decay 0.9995 \
  --multitask-aux-warmup-epochs 10
```

What this adds:
- EMA model used for validation/checkpoint export.
- Auxiliary multitask losses warmed up over early epochs.
- Non-finite loss batches are skipped defensively.

### Runtime Safety Mode

For long runs on shared desktops/laptops, enable conservative runtime caps:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --auto-safe-mode
```

This mode applies safer limits to batch size, image short-side, workers, and torch CPU thread usage, and disables expensive TensorBoard graph/projector exports.

Manual runtime controls are also available:
- `--workers`
- `--max-cpu-threads`
- `--max-interop-threads`

In the GUI (`Train Models` -> `DINO KPSEG` -> `Advanced`), use:
- `Auto safe mode (recommended for GUI responsiveness)`
- `DataLoader workers`
- `Max CPU threads`
- `Max interop threads`

Equivalent `annolid-run` training flags are also supported (for example
`--schedule-profile`, `--batch`, `--cos-lr`, `--warmup-epochs`,
`--best-metric`, and augmentation epoch windows).

### Unified Attention Head

DinoKPSEG now uses a single canonical attention architecture (`relational`) for
attention-based training. It keeps attention queries over keypoints and decodes masks
from query embeddings and refined spatial features via
`einsum(mask_embed, spatial_features)`.

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --head-type relational \
  --relational-heads 4 \
  --relational-layers 2
```

Optional symmetric-pair regularizers (requires `flip_idx` or inferable keypoint names):

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --lr-pair-loss-weight 0.05 \
  --lr-pair-margin-px 8
```

Note:
- For YOLO pose datasets with very high left/right ambiguity, DinoKPSEG now auto-disables default LR regularizer weights to avoid destabilizing training. Explicit non-default values still override this behavior.
- For high-ambiguity datasets, DinoKPSEG also auto-reduces geometry-heavy augmentation (disables hflip, and for ambiguous small datasets disables rotation/translate/scale) to reduce swap drift.
- For very small training sets (`<=64` images), DinoKPSEG applies a conservative small-data profile when defaults are used: `head_type` becomes `conv`, `feature_align_dim` becomes `auto`, `lr` becomes `1e-4`, and `early_stop_patience` is capped at `12`.
- For small/high-ambiguity datasets with orientation anchors (for example `nose` + `tailbase`), DinoKPSEG can auto-canonicalize left/right supervision to reduce persistent identity inversion. Override with `--lr-canonicalize` or `--no-lr-canonicalize`.

Optional left/right side-consistency (uses asymmetric anchors like `nose`/`tailbase` to define an axis):

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --lr-side-loss-weight 0.10 \
  --lr-side-loss-margin 0.0
```

Outputs:

- A new run directory under `ANNOLID_RUNS_ROOT` (or `~/annolid_logs/runs`) such as:
  - `~/annolid_logs/runs/dino_kpseg/train/20260101_120000/weights/best.pt`
  - `~/annolid_logs/runs/dino_kpseg/train/20260101_120000/weights/last.pt`
  - `~/annolid_logs/runs/dino_kpseg/train/20260101_120000/args.yaml`
  - `~/annolid_logs/runs/dino_kpseg/train/20260101_120000/results.csv`

To control where runs are written:

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --runs-root /path/to/annolid_logs/runs \
  --run-name experiment_01 \
  --epochs 50
```

## Inference (CLI)

```bash
annolid-run dino_kpseg predict \
  --weights /path/to/weights/best.pt \
  --image /path/to/frame.png \
  --tta-hflip \
  --tta-merge max \
  --min-keypoint-score 0.20 \
  --out /path/to/prediction.json
```

Notes:
- `--tta-hflip` enables horizontal flip test-time augmentation.
- `--tta-merge` controls TTA fusion: `mean` (default) or `max`.
- `--min-keypoint-score` drops low-confidence keypoints from output JSON.

## Inference (GUI)

In **Inference Wizard** with model type **DINO KPSEG**, the Configure page now exposes:
- `Enable horizontal flip TTA`
- `TTA merge` (`mean` or `max`)
- `Min keypoint score`

These settings are persisted in `QSettings` and are also applied in the standard
GUI prediction workflow (not only wizard runs).

## Evaluation

Run evaluation on the train/val split and report mean pixel error, PCK, and left/right swap rate:

```bash
python -m annolid.segmentation.dino_kpseg.eval \
  --data /path/to/YOLO_dataset/data.yaml \
  --weights /path/to/dino_kpseg/weights/best.pt \
  --split val \
  --thresholds 4,8,16
```

For LabelMe datasets, pass the spec YAML and set `--data-format labelme`:

```bash
python -m annolid.segmentation.dino_kpseg.eval \
  --data /path/to/labelme_spec.yaml \
  --data-format labelme \
  --weights /path/to/dino_kpseg/weights/best.pt \
  --split val
```

For COCO keypoints datasets, pass the COCO spec YAML and set `--data-format coco`.
Annolid will stage a temporary YOLO-pose view internally before evaluation.

## LabelMe to COCO (GUI)

Use this when your annotations are in LabelMe JSON format and you want a COCO
train/valid dataset for training or interoperability.

1. Open Annolid.
2. Go to **Convert -> LabelMe -> COCO**.
3. In **Annotation dir**, choose the folder containing LabelMe `*.json` files.
4. (Optional) Set **Output dir**. If left empty, Annolid uses
   `<annotation_dir>_coco_dataset`.
5. (Optional) Set **Labels file** (`labels.txt`). If empty, labels are auto-detected
   from shape labels in JSON files.
6. Set **Train split** (for example `70%`).
7. Choose **Output mode**:
   - `Segmentation`: polygon-oriented COCO export (default, backward compatible).
   - `Keypoints`: strict COCO pose-style export (`keypoints` + `num_keypoints`).
8. Click **OK** to export.

What the dialog shows:

- Number of discovered JSON files.
- Number of valid JSON+image pairs.
- Estimated train/valid counts based on the selected split.

What Annolid writes:

- `train/annotations.json`
- `train/JPEGImages/*.jpg`
- `valid/annotations.json`
- `valid/JPEGImages/*.jpg`
- `annotations_train.json` and `annotations_valid.json` (root-level convenience copies)
- `data.yaml`

In **Keypoints** mode, each COCO annotation represents one instance and includes:

- `keypoints` (flattened `[x, y, v]` list),
- `num_keypoints`,
- category-level `keypoints` names in `categories[*].keypoints`.

Notes:

- The exporter expects image files to be resolvable from each LabelMe JSON.
- Shapes are exported as COCO segmentations; supported shape types include
  polygon, rectangle, circle, and point.
- If no valid JSON/image pairs are found, export is blocked and the dialog shows
  an actionable warning.

## COCO to LabelMe (GUI)

Use this when you have COCO keypoint annotations and want editable LabelMe JSON files.

1. Open Annolid.
2. Go to **Convert → COCO to LabelMe**.
3. Select your COCO annotations directory (for example `annotations/`).
4. Optionally select an image directory (you can cancel to use paths from the COCO file).
5. Select an output directory for the generated LabelMe dataset.

What the converter writes:

- One image file plus one sidecar LabelMe JSON per COCO image (saved together).
- `polygon` shapes from COCO polygon segmentations (when available).
- `rectangle` shapes from COCO bounding boxes (fallback when no polygons exist).
- `point` shapes from COCO keypoints.

Notes:

- Keypoints with non-visible flags (`v <= 0`) are skipped.
- `group_id` is set from COCO annotation id, so polygon/box/keypoints from the same object stay linked.
- If your COCO JSON is under `annotations/` and image files are under sibling `images/`,
  Annolid auto-resolves those paths (you can still set images dir explicitly).
- `imagePath` in output JSON is written as the local image filename next to the JSON sidecar.

## Running Inference (GUI)

1. Train the model (or point Annolid to a DinoKPSEG checkpoint).
2. In Annolid, select **AI Model → DINOv3 Keypoint Segmentation**.
3. Click **Pred** to run prediction.

Optional instance-aware inference: draw rectangle prompts on the canvas (one per animal).
Annolid will run DinoKPSEG per box and keep keypoints grouped by instance.

The predictor saves:

- Point shapes for each keypoint.
- A small polygon “circle mask” per keypoint (for quick visual segmentation).
