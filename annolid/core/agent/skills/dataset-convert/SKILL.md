---
name: dataset-convert
description: Convert datasets between COCO, LabelMe, and YOLO pose formats using Annolid-native converters.
metadata: '{"annolid":{"requires":{"bins":[]}}}'
---

# Dataset Convert

Use this skill when the user asks to convert dataset formats.

Prefer the bundled converter script for deterministic runs:

`python annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py ...`

## Supported conversions

1. `labelme-to-coco`: LabelMe directory -> COCO train/valid JSON.
2. `coco-to-labelme`: COCO JSON (or annotations dir) -> LabelMe sidecar JSON next to images.
3. `coco-spec-to-yolo`: COCO pose spec YAML -> YOLO pose dataset (`data.yaml`, `images/`, `labels/`).

## Commands

### LabelMe -> COCO keypoints

```bash
python annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py \
  labelme-to-coco \
  --input-dir /path/to/labelme_dataset \
  --output-dir /path/to/coco_out \
  --mode keypoints \
  --train-valid-split 0.8
```

### COCO annotations dir -> LabelMe dataset

```bash
python annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py \
  coco-to-labelme \
  --annotations-dir /path/to/coco/annotations \
  --output-dir /path/to/labelme_out \
  --images-dir /path/to/coco/images
```

### COCO pose spec -> YOLO pose dataset

```bash
python annolid/core/agent/skills/dataset-convert/scripts/convert_dataset.py \
  coco-spec-to-yolo \
  --spec-yaml /path/to/coco_pose_spec.yaml \
  --output-dir /path/to/yolo_pose_dataset
```

## Notes

- `labelme-to-coco` keypoints mode requires `pycocotools`.
- For large datasets, prefer `--link-mode hardlink` (default) to avoid image copying.
- Always return output paths and summary counts after conversion.
