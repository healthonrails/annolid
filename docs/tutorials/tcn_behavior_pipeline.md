# TCN Behavior Classification From Pose Features

Annolid includes a DAART-style temporal convolutional network (TCN) pipeline for
frame-level behavior classification from pose or feature CSV files, following the
animal action segmentation benchmark described by Blau et al. (2024). It is
useful when you already have per-frame keypoints, geometric features, or
polygon-derived features plus sparse one-hot behavior labels.

The pipeline supports:

- DLC-style marker CSVs (`input_type: markers`)
- Generic per-frame feature CSVs (`input_type: features`)
- Optional position-velocity features (`add_velocity: true`)
- Sparse labels with background ignored during training
- Held-out-session evaluation with macro F1 and per-class F1

## Config

Create a YAML file such as `tcn_behavior.yaml`:

```yaml
labels:
  - background
  - still
  - walk
  - groom

sessions:
  - id: session_001
    features: /path/to/session_001_labeled.csv
    labels: /path/to/session_001_labels.csv
    split: train
  - id: session_002
    features: /path/to/session_002_labeled.csv
    labels: /path/to/session_002_labels.csv
    split: test

feature:
  input_type: features
  add_velocity: true
  zscore: true

model:
  hidden_dim: 32
  num_blocks: 2
  kernel_size: 9
  dropout: 0.1

training:
  epochs: 500
  batch_size: 8
  sequence_length: 1000
  learning_rate: 0.0001
  device: auto
```

For DLC marker CSVs, set:

```yaml
feature:
  input_type: markers
  add_velocity: false
```

The label CSV should be one-hot with one row per frame. The first column may be a
frame index named `frame`, `frames`, `frame_index`, or `index`. Generic feature
CSVs may also include one of those frame-index columns; it will be dropped before
training. The first listed label is treated as background by convention.

## Polygon Classifier Workbench

Users who start from Annolid polygon annotations do not need to hand-convert
polygon CSVs into TCN feature and label files. In the GUI, open
**Tools -> Polygon Classifier Workbench**, generate or select a polygon feature
CSV, then choose **TCN** in the training tab's model selector. Annolid preserves
the polygon CSV format, builds temporary dense per-video TCN inputs inside the
run directory, and saves a `polygon_tcn_classifier_best.pt` checkpoint that can
be used from the same inference tab.

For a step-by-step guide to collecting compatible training and test videos with
Annolid polygon tracking, see
[Collect Videos for Polygon TCN Training and Testing](polygon_tcn_video_collection.md).

## Train

```bash
annolid-run train tcn_behavior \
  --config /path/to/tcn_behavior.yaml \
  --output-dir /path/to/run_dir
```

Outputs:

- `best_model.pt`: model weights plus label names and normalization stats
- `metrics.json`: training history and held-out test metrics when test sessions are present

## Predict

```bash
annolid-run predict tcn_behavior \
  --config /path/to/tcn_behavior.yaml \
  --checkpoint-path /path/to/run_dir/best_model.pt \
  --output-csv /path/to/predictions.csv \
  --smoothing-window 51
```

The prediction CSV contains one row per frame:

```text
session_id,frame,predicted_index,predicted_label
```

When labels are present for the prediction split, you can also write metrics:

```bash
annolid-run predict tcn_behavior \
  --config /path/to/tcn_behavior.yaml \
  --checkpoint-path /path/to/run_dir/best_model.pt \
  --output-csv /path/to/predictions.csv \
  --metrics-json /path/to/test_metrics.json \
  --smoothing-window 51
```

Use `--smoothing-window 1` to disable temporal smoothing. Larger odd-valued
windows average neighboring class probabilities before the final label choice;
choose the window on validation data rather than on the final test set.

## Citation

Blau, A., Schaffer, E. S., Mishra, N., Miska, N. J., The International Brain
Laboratory, Paninski, L., & Whiteway, M. R. (2024). A study of animal action
segmentation algorithms across supervised, unsupervised, and semi-supervised
learning paradigms. arXiv:2407.16727.
[DOI](https://doi.org/10.48550/arXiv.2407.16727)

The TCN setup in this tutorial follows the supervised TCN baseline discussed in
the paper's methods and supplementary material:
[paper PDF, supplementary methods](https://arxiv.org/pdf/2407.16727#page=33.07)
