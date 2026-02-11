# DINOv3 Keypoint Segmentation (DinoKPSEG)

Annolid includes an experimental keypoint-centric segmentation model that:

- Extracts **frozen DINOv3 dense features** (ViT patch grid).
- Trains a **small convolutional head** to predict per-keypoint masks.
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

### Relational (Attention) Head

For better left/right consistency on symmetric keypoints (e.g., ears), you can enable the attention head.
When `kpt_names` are available, DinoKPSEG will automatically treat asymmetric keypoints (e.g. `nose`, `head`, `tailbase`)
as orientation anchors and inject them into other keypoints via cross-attention.

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --head-type attn \
  --attn-heads 4 \
  --attn-layers 2
```

Optional symmetric-pair regularizers (requires `flip_idx` or inferable keypoint names):

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --lr-pair-loss-weight 0.05 \
  --lr-pair-margin-px 8
```

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
