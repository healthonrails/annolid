# INSID3 Video Segmentation

INSID3 Video is a training-free DINOv3 segmentation backend for videos. It uses
one manually labeled polygon frame as an in-context reference, then segments each
target frame with DINO feature matching, positional debiasing, patch clustering,
and cluster aggregation.

## Requirements

- A video with at least one saved Annolid polygon annotation.
- Access to the configured DINOv3 checkpoint. The default uses
  `facebook/dinov3-vits16-pretrain-lvd1689m`, which may require Hugging Face
  access.
- The usual Annolid Python environment with PyTorch and OpenCV.

## GUI Workflow

1. Open the video.
2. Draw polygon masks for the objects on a clear reference frame.
3. Save the frame so Annolid creates the PNG and JSON seed pair.
4. Select **INSID3 Video (DINOv3)** in the model dropdown.
5. Run prediction for the desired frame range.

Annolid writes predicted frames to the video folder's AnnotationStore NDJSON
file, for example `mouse_annotations.ndjson`, instead of creating one JSON file
per predicted frame. Manual seed frames remain normal PNG+JSON pairs so they can
be discovered as in-context references.

## Behavior

- The backend is training-free and does not use a segmentation decoder.
- Each instance is segmented from its reference polygon prototype.
- Each video frame is inferred independently from the reference frame.
- Optional CRF-style boundary refinement can be enabled after the DINO/clustering
  mask is produced. Annolid first tries the external dense CRF package when
  available, then falls back to an OpenCV guided-filter cleanup backend when
  `auto` is used.

## Optional CRF Refinement

The external dense CRF backend is optional because it builds native PyTorch/CUDA
extensions. To experiment with it in a CUDA-capable environment:

```bash
python -m pip install "git+https://github.com/netw0rkf10w/CRF.git"
```

Then construct the processor or run scripts with `insid3_crf_refine=True`. On
machines where the dense CRF package is unavailable, `insid3_crf_backend="auto"`
uses the OpenCV refinement fallback instead of failing the prediction run.

## Tuning

The backend accepts runtime keyword arguments when constructed from scripts:

- `patch_model_name` or `dinov3_model_name`
- `insid3_short_side`
- `insid3_svd_components`
- `insid3_tau`
- `insid3_merge_threshold`
- `insid3_max_cluster_area_growth`
- `insid3_crf_refine`
- `insid3_crf_backend` (`auto`, `crf`, or `opencv`)
- `insid3_crf_band_px`
- `insid3_crf_p_core`
- `insid3_crf_iterations`

Lower `insid3_short_side` values are faster. Higher `insid3_tau` values split
clusters more aggressively. Higher `insid3_merge_threshold` values make the
final mask more conservative. Lower `insid3_max_cluster_area_growth` values
reject larger cluster expansions relative to the reference-matched candidate
patches. Smaller `insid3_crf_band_px` values restrict refinement closer to the
predicted boundary; larger values allow more boundary movement.
