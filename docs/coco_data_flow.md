# COCO Data Flow

Annolid now uses one shared COCO dataset layer across YOLO, DinoKPSEG, conversion tooling, and Mask R-CNN metadata readers. This page explains what inputs are accepted, how Annolid interprets them, and what each model workflow expects.

## Accepted Inputs

Annolid can start from either of these COCO inputs:

- a COCO spec YAML
- a COCO annotations directory
- a COCO dataset root that contains an `annotations/` folder
- a COCO dataset root with split-local annotations such as `train/annotations.json` and `val/annotations.json`

Annolid treats a dataset as COCO when either is true:

- `format` or `type` is `coco`, `coco_pose`, or `coco_keypoints`
- one of `train`, `val`, or `test` points to a `.json` annotations file

## Accepted Annotation Filenames

When you point Annolid at an annotations directory, it will look for these common COCO filenames:

- `train.json`, `val.json`, `test.json`
- `instances_train.json`, `instances_val.json`, `instances_test.json`
- `person_keypoints_train.json`, `person_keypoints_val.json`, `person_keypoints_test.json`

This means you can usually select the COCO `annotations/` directory directly instead of writing a YAML first.
Annolid also accepts a dataset root and will auto-discover `annotations/` when possible.

Annolid also accepts split-local COCO layouts such as:

- `train/annotations.json`
- `val/annotations.json`
- `test/annotations.json`

In that layout, images can live next to each split-local JSON file, and Annolid will resolve them during COCO to YOLO staging.

## Task Detection

Annolid infers whether a COCO dataset is pose or detection:

- `pose`: keypoint information is present in the YAML or in COCO categories/annotations
- `detect`: no keypoint information is present

This inference is shared, so different model entry points no longer make different decisions about the same COCO dataset.

## Model Support

### YOLO workflows

YOLO training can start from:

- COCO pose
- COCO detection
- a plain YOLO `data.yaml` that lives next to COCO annotations
- staged YOLO datasets

If the source is COCO, Annolid materializes a temporary YOLO-style dataset with:

- `images/`
- `labels/`
- `data.yaml`

If the selected directory is a dataset root (not the annotations folder), Annolid will auto-locate COCO annotations before staging.

If you launch a YOLO pose model from a plain `data.yaml` and that YAML does not include `kpt_shape`, Annolid now checks for nearby COCO keypoint annotations and upgrades the dataset to a staged YOLO pose dataset automatically.

### DinoKPSEG workflows

DinoKPSEG COCO workflows are pose-only.

If you pass a COCO detection spec to a DinoKPSEG flow, Annolid now fails early with a clear error instead of getting farther into training and failing in a less obvious place.

### Mask R-CNN workflows

Mask R-CNN still uses torchvision-specific dataset wrappers, but category names, keypoint metadata, and category-id mapping now come from the same shared COCO utilities used elsewhere in Annolid.

## Validation and Auto-Splitting

- If a staged YOLO dataset needs both `train` and `val`, Annolid validates that those splits are present before launching training.
- If a COCO spec provides `train` but no `val`, Annolid may auto-split validation during COCO to YOLO staging when the workflow allows it.
- Pose-only consumers explicitly enforce `expected_task="pose"`.
- If a YOLO-facing YAML is missing `names`/`nc`, Annolid attempts to infer class names from nearby COCO annotation JSON categories before falling back to defaults.
- If a YOLO pose launch still cannot resolve keypoint metadata, Annolid now stops before invoking Ultralytics and reports that the selected dataset is not pose-compatible.

## Recommended COCO YAMLs

COCO pose spec:

```yaml
format: coco
path: /path/to/dataset_root
image_root: images
train: annotations/person_keypoints_train.json
val: annotations/person_keypoints_val.json
```

COCO detection spec:

```yaml
format: coco
path: /path/to/dataset_root
image_root: images
train: annotations/instances_train.json
val: annotations/instances_val.json
```

## Typical Workflows

### GUI YOLO training from COCO

1. Open `File -> Train models`.
2. Choose `YOLO`.
3. Select either:
   - a COCO spec YAML, or
   - a COCO annotations directory, or
   - a COCO dataset root containing `annotations/`, or
   - a plain `data.yaml` beside a COCO keypoints dataset
4. Start training as usual.

Annolid will stage the COCO dataset into a YOLO-compatible temporary dataset before invoking Ultralytics. For pose models, this includes auto-detecting nearby COCO keypoints annotations when a plain `data.yaml` is missing `kpt_shape`.

### CLI and tooling

Shared COCO staging is also used by:

- `annolid-run` model workflows that consume the shared dataset layer
- the dataset conversion helper script in `annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py`

## Troubleshooting

If COCO training fails, check these first:

- `path` points to the dataset root, not just the annotations directory
- `image_root` matches where images actually live
- `train` and `val` annotation JSONs exist
- pose datasets include keypoints in the COCO categories or annotations
- when using a YOLO pose model, prefer `person_keypoints_*.json` over detection-only `instances_*.json`
- DinoKPSEG inputs are COCO pose, not COCO detection
- if your YAML has no `names`/`nc`, ensure COCO `categories` contain valid names for class inference

## Implementation Note

The shared COCO implementation lives in `annolid/datasets/coco.py`.

For backward compatibility, `annolid/yolo/dataset_prep.py` remains as a thin re-export shim.
