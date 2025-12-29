# Model training

You do **not** always need to train a model to use Annolid effectively. For many videos, you can label a single frame and track with a video-object-segmentation backend (e.g., Cutie / EfficientTAM-style trackers), then review and correct.

Train a model when you need:
- higher speed for long recordings,
- better generalisation to your camera/arena/species,
- a domain-specific detector/segmenter (e.g., custom objects, special backgrounds),
- a pose model with named keypoints.

## Option A (recommended): train YOLO segmentation/pose
Annolid integrates Ultralytics YOLO for segmentation and pose.

### 1) Create a YOLO dataset from your Annolid labels
1. Label frames in Annolid (polygons/keypoints saved as LabelMe JSON).
2. Convert your LabelMe JSON folder to YOLO format:
   - GUI: *File → Convert Labelme to YOLO format*
   - CLI: `python -m annolid.main --labelme2yolo /path/to/json_folder --val_size 0.1 --test_size 0.1`

This produces a dataset folder containing images, label files, and a `data.yaml`.

### 2) Train from the Annolid GUI
Use *File → Train models* and select **YOLO**. Choose:
- a base model (or a custom YOLO export),
- epochs and image size,
- an output directory for runs/checkpoints.

After training, you can select the resulting weights in the AI model selector (or “Browse Custom YOLO…”).

## Option B: Mask R-CNN via Detectron2 (optional)
If you specifically need Detectron2-based Mask R-CNN training/inference:
- Prefer the Colab notebook (*File → Open in Colab*) for a working GPU environment.
- Or follow the Detectron2 installation guide linked from [Install options](how_to_install.md).

## COCO export
If you need COCO format for interoperability with other toolchains:
- GUI: *File → COCO format*

```{note}
Annolid can export multiple formats (COCO/YOLO/CSV). Pick the one that matches your training/inference stack.
```
