# Technical overview

Annolid supports multiple “paths” depending on your goal (manual annotation, semi-automatic tracking, or training a custom model), but most workflows follow the same loop:

1. **Prepare videos** (optional downsampling/cropping for speed).
2. **Label** a starting frame (or a small set of frames) with polygons and/or keypoints.
3. **Run AI** for segmentation/tracking/keypoints (choose a backend that fits your task).
4. **Review + correct** and re-run from the first error if needed.
5. **Export** (LabelMe JSON, CSV summaries, COCO/YOLO datasets).
6. **Analyze** (place preference, motion/freezing metrics, event time budgets, reports).

![Overview of Annolid workflow](../../docs/imgs/annolid_workflow.png)

## Supported workflows (today)
### 1) “Track from one frame” (no training required)
For many videos, you can get high-quality tracking without training:
- Create polygons (manually or with AI polygons).
- Use a video-object-segmentation tracker (e.g., Cutie / EfficientTAM-style backends) to propagate masks through the video.
- Export results to CSV for analysis.

### 2) Prompt-assisted labeling (fewer clicks)
To speed up annotation:
- **Point prompt**: click an object → SAM-family model returns a mask → Annolid converts it to a polygon.
- **Text prompt**: type labels (e.g., `mouse, feeder`) → Grounding DINO proposes boxes → SAM refines masks → polygons.

### 3) Train domain-specific models (optional)
When you need a model tuned to your camera, arena, or species:
- Convert LabelMe JSON to **YOLO** format and train a segmentation/pose model.
- Train Mask R-CNN via Detectron2 (optional, best on GPUs / Colab).

## Practical performance notes
Performance depends strongly on video resolution, model choice, and hardware (CPU vs CUDA/MPS GPU). In general:
- Downsampling/cropping videos speeds up both interactive work and batch processing.
- Foundation-model tracking helps with occlusions and reduces the number of labeled frames needed.
