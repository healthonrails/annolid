# Annolid on Google Colab

Google Colab is often the easiest way to run GPU-heavy training or inference (especially for Detectron2 / Mask R-CNN workflows).

## Open the official notebook
- From the Annolid GUI: *File → Open in Colab*
- Or open directly:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb)

## Typical workflow
1. On your local machine, label frames in Annolid and export a dataset (COCO or YOLO depending on the notebook/tooling you use).
2. Upload the dataset to Google Drive (or point the notebook at a GitHub URL if the data is public).
3. Run training/inference in Colab.
4. Download the trained weights and use them in Annolid (e.g., “Browse Custom YOLO…” or your Detectron2 model path).

```{note}
For many tracking tasks you don’t need Colab: you can label one frame and track with Annolid’s built-in tracking backends, then export to CSV for analysis.
```
