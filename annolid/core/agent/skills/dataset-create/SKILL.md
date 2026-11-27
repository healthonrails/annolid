---
name: dataset-create
description: Inspect dataset folders, build Annolid label indexes and specs, import DeepLabCut or SLEAP data, and export YOLO-ready training datasets.
metadata: '{"annolid":{"always":false,"tags":["dataset","labelme","yolo","coco","deeplabcut","dlc","sleap","index","spec","splits","prepare"]}}'
---

# Dataset Create & Prepare

Use this skill when the user asks to inspect, index, split, import, or prepare a training dataset for YOLO, DinoKPSEG, Mask R-CNN, or Keypoint R-CNN.

## Preferred Typed Tools (First Choice)

Prefer Annolid's typed dataset tools over raw shell commands:

1. **`annolid_dataset_inspect(dataset_folder)`** (read-only)
   - Detects existing specs (`labelme_spec.yaml`, `coco_spec.yaml`, `data.yaml`), raw LabelMe pairs, shape counts, inferred splits, and external formats (`deeplabcut`, `sleap`, `coco_instances`, `coco_keypoints`).
   - Returns `ready_for_training`, `recommended_models`, and concrete `next_actions`.

2. **`annolid_dataset_prepare(dataset_folder, mode, allow_mutation=true, ...)`**
   - Supported `mode` values:
     - `labelme_spec`: build `labelme_spec.yaml` + train/val/test splits (`val_size`, `test_size`, `seed`, `keypoint_names`, `kpt_dims`, `include_empty`).
     - `yolo_from_labelme`: index LabelMe pairs and export YOLO `images/`, `labels/`, `data.yaml` (`task`: `"auto"|"pose"|"segmentation"`, `dataset_name`, `output_dir`, `val_size`, `test_size`).
     - `coco_spec`: infer and write `coco_spec.yaml`.
     - `coco_to_yolo`: materialize COCO pose/seg/detect dataset into YOLO format.
     - `deeplabcut_import`: import DeepLabCut `labeled-data` CSV/H5 into LabelMe + optional `pose_schema.json` and Annolid index (`instance_label`, `write_pose_schema`, `pose_schema_preset`, `write_index`).
     - `sleap_import`: import `.slp`/`.pkg.slp`/`.analysis.h5` into LabelMe + PoseSchema + optional index.
   - Always pass `allow_mutation=true` only when the user intends to write/prepare dataset files.

## CLI Equivalents (via `annolid_run` or Terminal)

When multi-source collection or pose export is needed beyond `annolid_dataset_prepare`:

```bash
# Collect multiple LabelMe sources into an index JSONL + spec YAML
annolid-run collect-labels \
  --source /path/to/source_a --source /path/to/source_b \
  --dataset-root /path/to/dataset_root \
  --index-file logs/label_index/annolid_dataset.jsonl \
  --recursive --write-spec \
  --spec-path /path/to/dataset_root/labelme_spec.yaml \
  --val-size 0.1 --test-size 0.1 --infer-flip-idx

# Convert an index JSONL to a YOLO dataset
annolid-run index-to-yolo \
  --index-file /path/to/dataset_root/logs/label_index/annolid_dataset.jsonl \
  --output-dir /path/to/output --dataset-name YOLO_dataset \
  --val-size 0.1 --test-size 0.1 --task pose

# Import / export pose formats (SLEAP, DeepLabCut, Lightning Pose)
annolid-run import-pose /path/to/labels.pkg.slp --output-dir /path/to/sleap_labelme --write-index
annolid-run export-pose /path/to/labelme_dir --format deeplabcut-csv --output /path/to/exported_dlc.csv
```

## Completion Checklist

- Report exact output paths (`data.yaml`, `labelme_spec.yaml`, `coco_spec.yaml`, index JSONL, `pose_schema.json`) and image/label split counts.
- Verify `kpt_shape` is present for pose datasets before handing off to `annolid_train_start`.
