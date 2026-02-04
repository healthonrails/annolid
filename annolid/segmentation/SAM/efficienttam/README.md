# EfficientTAM Integration in Annolid

This directory vendors the core components of **Efficient Track Anything (EfficientTAM)** for use inside Annolid, so users can select EfficientTAM as a video segmentation / tracking backend without installing or managing a separate repository.

The original project lives at:

- GitHub: https://github.com/yformer/EfficientTAM
- Project page: https://yformer.github.io/efficient-track-anything/

The code here is lightly adapted to:

- Live under `annolid.segmentation.SAM.efficienttam` as a standard Python package.
- Use Annolid’s device selection (`cuda` → `mps` → `cpu`).
- Load frames directly from video files (via `decord` when available, with an OpenCV fallback).
- Auto-download EfficientTAM checkpoints from Hugging Face into a user cache (no manual downloads required).

## How Annolid Uses EfficientTAM

- **Wrapper:** `EfficientTAMVideoProcessor` in `annolid/segmentation/SAM/sam_v2.py` wraps the vendored EfficientTAM predictor and converts its outputs into LabelMe JSON/NDJSON, reusing the same pipeline as SAM2/SAM3.
- **Config:** Hydra configs for the main variants (`efficienttam_s`, `efficienttam_ti`, etc.) are under `configs/efficienttam/`.
- **Checkpoints:** On first use of a given model key (e.g. `efficienttam_s`), Annolid downloads the corresponding `.pt` from `yunyangx/efficient-track-anything` into:
  - `~/.cache/annolid/efficienttam/checkpoints` by default, or
  - the directory pointed to by `ANNOLID_EFFICIENTTAM_CACHE_DIR`.
- **GUI:** The Annolid GUI exposes `EfficientTAM_s` and `EfficientTAM_ti` as selectable models in the AI model combo box. When selected, `Predict` runs EfficientTAM over the active video and writes per-frame annotations for downstream CSV/export.

## Usage (High Level)

1. Open a video in Annolid and label at least one frame (polygons or rectangles) for the objects you want to track.
2. In the model selector, choose `EfficientTAM_s` (small) or `EfficientTAM_ti` (tiny).
3. Click `Pred` to start tracking:
   - The first run downloads the appropriate EfficientTAM checkpoint.
   - Frames are decoded from the original video file, processed on GPU/MPS/CPU, and saved as LabelMe JSONs + NDJSON.
4. Annolid converts those JSONs to a tracking CSV using the standard post-processing pipeline.

## Citing EfficientTAM

If you use EfficientTAM through Annolid in your research, please cite the original EfficientTAM work in addition to Annolid:

```bibtex
@article{xiong2024efficienttam,
  title={Efficient Track Anything},
  author={Yunyang Xiong, Chong Zhou, Xiaoyu Xiang, Lemeng Wu, Chenchen Zhu, Zechun Liu, Saksham Suri, Balakrishnan Varadarajan, Ramya Akula, Forrest Iandola, Raghuraman Krishnamoorthi, Bilge Soran, Vikas Chandra},
  journal={preprint arXiv:2411.18933},
  year={2024}
}
```

For details about model design, training data, and the broader ecosystem (SAM2, EfficientSAM, etc.), refer to the upstream EfficientTAM repository and paper.
