---
name: polygon-classifier-training
description: Prepare polygon/tracking feature CSVs and train temporal (TCN) or ConvNet polygon behavior classifiers with video-grouped splits and reproducible reports.
metadata: '{"annolid":{"always":false,"tags":["polygon classifier","tcn","convnet","behavior classifier","train behavior","polygon features","temporal convolutional","annolid-run train","polygon_classifier","workbench","held-out video"]}}'
---

# Polygon Classifier Training

Use this skill when the user wants to train, tune, or evaluate a supervised behavior classifier from Annolid polygon annotations (LabelMe `.json` directories or `*_annotations.ndjson`) or `*_tracking.csv` files paired with behavior timestamp CSVs.

## Supported Inputs

- **Paired video folders + behavior CSVs** (`--train-video <ANN_DIR> <LABEL_CSV>`, `--test-video <ANN_DIR> <LABEL_CSV>`):
  - `<ANN_DIR>` can contain per-frame LabelMe `.json` files, an `*_annotations.ndjson` store, or both (Annolid handles dual annotation storage automatically).
  - `<LABEL_CSV>` holds behavior start/stop or per-frame labels (e.g., BORIS/Annolid timestamps CSV).
- **Pre-exported polygon feature CSVs** (`--train-csv`, `--test-csv`).
- **GUI alternative**: Polygon Classifier Workbench dialog in Annolid GUI (or Agent Capabilities -> "Polygon Classifier" preset).

## Canonical CLI Workflow (`annolid_run` / `annolid-run`)

Inspect exact flags first if unsure:

```bash
annolid-run train polygon_classifier --help
```

Train with explicit video-grouped train/test splits (never split frames of the same video across train and test!):

```bash
annolid-run train polygon_classifier \
  --train-video /path/to/train_video_1_ann /path/to/train_video_1_timestamps.csv \
  --train-video /path/to/train_video_2_ann /path/to/train_video_2_timestamps.csv \
  --test-video /path/to/heldout_video_ann /path/to/heldout_video_timestamps.csv \
  --model-type tcn \
  --num-points 16 \
  --window-size 15 \
  --num-epochs 50 \
  --batch-size 32 \
  --learning-rate 1e-3 \
  --output-dir /path/to/runs/polygon_clf \
  --run-name mouse_tcn_v1
```

Key options:

- `--model-type {tcn,convnet}`: prefer `tcn` for temporal bout dynamics; `convnet` for lightweight single-window baselines.
- `--num-points`: resampled polygon vertex count per instance (keep identical between train and inference).
- `--include-unlabeled`: include background/unlabeled frames explicitly when calibrating precision vs recall.
- `--device cpu|cuda|mps|auto`: honor explicit `cpu` requests when debugging or on CPU-only machines.
- `--csv-output-dir`: persist extracted feature CSVs for auditability and reuse.
- `--run-config`: optional YAML/JSON config for reproducible sweeps.
- **Remote GPU training**: escalate to `google_colab(action="train", model="polygon_classifier", ...)` or `model="tcn_behavior"` when local GPU compute is insufficient.

## Scientific & Data-Integrity Guardrails

1. **Prevent Data Leakage**: Split train and validation/test sets by `source_group` (recording/video) **before** any class oversampling or window augmentation.
2. **Mask Padded Steps**: Ensure TCN sequence padding uses ignore index `-100` so padded frames never distort loss or metrics.
3. **Require Mutating Consent**: Pass `allow_mutation=true` when invoking `annolid_run` or `annolid_train_start`.

## Output Contract & Next Step

Always report:

- `checkpoint_path` (saved model weights),
- `metrics_path` (`accuracy`, `macro_f1`, `weighted_f1`, per-class precision/recall/F1, confusion matrix),
- held-out predictions CSV and Markdown report paths,
- train vs held-out recording split summary (confirming zero video overlap).

Follow up with the `behavior-evaluation` skill to run formal QC, held-out evaluation, and error analysis.
