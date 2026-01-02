# DINOv3 Keypoint Segmentation (DinoKPSEG)

Annolid includes an experimental keypoint-centric segmentation model that:

- Extracts **frozen DINOv3 dense features** (ViT patch grid).
- Trains a **small convolutional head** to predict per-keypoint masks.
- Uses **small circular masks around each keypoint** as supervision.
- Runs inference via the same prediction pipeline used for YOLO, saving results as LabelMe JSON.

## Training

This trainer consumes a standard YOLO pose dataset (`data.yaml` with `kpt_shape` and labels in `labels/*.txt`).

```bash
python -m annolid.segmentation.dino_kpseg.train \
  --data /path/to/YOLO_dataset/data.yaml \
  --epochs 50
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

## Running Inference (GUI)

1. Train the model (or point Annolid to a DinoKPSEG checkpoint).
2. In Annolid, select **AI Model → DINOv3 Keypoint Segmentation**.
3. Click **Pred** to run prediction.

The predictor saves:

- Point shapes for each keypoint.
- A small polygon “circle mask” per keypoint (for quick visual segmentation).
