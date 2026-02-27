---
name: dataset-create
description: Build Annolid dataset indexes/specs and generate YOLO-ready datasets from labeled data sources.
metadata: '{"annolid":{"requires":{"bins":[]}}}'
---

# Dataset Create

Use this skill when the user asks to create or prepare a training dataset.

## Preferred workflow

1. Confirm source layout and output location.
2. Build/update an Annolid index JSONL from LabelMe pairs.
3. Optionally write a LabelMe spec with train/val/test splits.
4. Convert index -> YOLO dataset when requested.
5. Report exact output paths and counts.

## Commands

### 1) Index LabelMe pairs (+ optional spec)

```bash
python -m annolid.engine.cli collect-labels \
  --source /path/to/source_a \
  --source /path/to/source_b \
  --dataset-root /path/to/dataset_root \
  --index-file logs/label_index/annolid_dataset.jsonl \
  --recursive \
  --write-spec \
  --spec-path /path/to/dataset_root/labelme_spec.yaml \
  --val-size 0.1 \
  --test-size 0.1 \
  --infer-flip-idx
```

### 2) Convert index JSONL -> YOLO dataset

```bash
python -m annolid.engine.cli index-to-yolo \
  --index-file /path/to/dataset_root/logs/label_index/annolid_dataset.jsonl \
  --output-dir /path/to/output \
  --dataset-name YOLO_dataset \
  --val-size 0.1 \
  --test-size 0.1 \
  --task pose
```

### 3) Import DeepLabCut training data into LabelMe + optional indexing

```bash
python -m annolid.engine.cli import-deeplabcut-training-data \
  --source-dir /path/to/deeplabcut_project \
  --labeled-data labeled-data \
  --instance-label mouse \
  --write-pose-schema
```

## Completion checklist

Before finishing, verify:

- Index file exists (`annolid_dataset.jsonl`).
- If YOLO conversion was requested: `images/`, `labels/`, and `data.yaml` exist.
- If spec was requested: YAML has `kpt_shape` (pose datasets) and split entries.

In the final response, always include output file paths.
