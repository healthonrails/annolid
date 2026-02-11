# DinoKPSEG Tiny COCO Fixture

Small COCO keypoints fixture for testing DinoKPSEG COCO support.

- `images/`: 3 tiny synthetic PNG images
- `annotations/train.json`: 2 training images
- `annotations/val.json`: 1 validation image
- `coco_spec.yaml`: Annolid COCO spec for `--data`

Example:

```bash
annolid-run train dino_kpseg --data tests/fixtures/dino_kpseg_coco_tiny/coco_spec.yaml --data-format coco
```
