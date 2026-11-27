---
name: dataset-convert
description: Convert datasets between COCO, LabelMe, YOLO pose/seg/detect, DeepLabCut, and SLEAP formats using Annolid-native tools and deterministic scripts.
metadata: '{"annolid":{"always":false,"tags":["dataset","convert","labelme","coco","yolo","pose","keypoints","deeplabcut","sleap","format conversion"]}}'
---

# Dataset Convert

Use this skill when the user asks to convert annotation datasets between LabelMe, COCO, YOLO, DeepLabCut, or SLEAP formats.

## Prefer Typed Annolid Tools First

When running inside Annolid Bot, prefer typed tools before raw scripts:

1. Inspect first (read-only): `annolid_dataset_inspect(dataset_dir="...")`
2. Prepare/convert (mutating — set `allow_mutation=true`):
   - `annolid_dataset_prepare(mode="yolo_from_labelme", source_dir="...", output_dir="...", task="seg"|"detect"|"pose", allow_mutation=true)`
   - `annolid_dataset_prepare(mode="coco_to_yolo", source_dir="...", output_dir="...", allow_mutation=true)`
   - `annolid_dataset_prepare(mode="deeplabcut_import", source_dir="...", output_dir="...", allow_mutation=true)`
   - `annolid_dataset_prepare(mode="sleap_import", source_dir="...", output_dir="...", allow_mutation=true)`
3. Pose CLI converters via `annolid_run` (or shell):
   - `annolid-run import-pose ...` / `annolid-run export-pose ...`

## Deterministic Bundled Converter Script

For standalone/scripted conversions between LabelMe, COCO, and YOLO pose, use the bundled converter script:

`python annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py ...`

### Supported script modes

1. `labelme-to-coco`: LabelMe directory -> COCO train/valid JSON (`--mode instance|keypoints`).
2. `coco-to-labelme`: COCO JSON (or annotations dir) -> LabelMe sidecar JSON next to images.
3. `coco-spec-to-yolo`: COCO pose spec YAML -> YOLO pose dataset (`data.yaml`, `images/`, `labels/`).

### Commands

#### LabelMe -> COCO keypoints

```bash
python annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py \
  labelme-to-coco \
  --input-dir /path/to/labelme_dataset \
  --output-dir /path/to/coco_out \
  --mode keypoints \
  --train-valid-split 0.8
```

#### COCO annotations dir -> LabelMe dataset

```bash
python annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py \
  coco-to-labelme \
  --annotations-dir /path/to/coco/annotations \
  --output-dir /path/to/labelme_out \
  --images-dir /path/to/coco/images
```

#### COCO pose spec -> YOLO pose dataset

```bash
python annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py \
  coco-spec-to-yolo \
  --spec-yaml /path/to/coco_pose_spec.yaml \
  --output-dir /path/to/yolo_pose_dataset
```

## Notes

- `labelme-to-coco` in `keypoints` mode requires `pycocotools`.
- For `labelme-to-coco` in `instance` mode without a labels file, pass `--labels-file ""`.
- For large datasets, prefer `--link-mode hardlink` (default) to avoid image copying.
- Always return output paths and split/image/annotation summary counts after conversion.
