# Annolid on Google Colab

Google Colab lets you run GPU-heavy Annolid training and inference (YOLO, DINO KPSEG, Mask R-CNN, Keypoint R-CNN, and video batch tracking) without needing a local CUDA GPU.

## Open Google Colab Studio in Annolid

- From the Annolid GUI: **AI & Models → Google Colab…** (or click the **Google Colab** toolbar button, or click **Run on Google Colab…** from the Training / Inference Wizards).
- From the CLI:

```bash
annolid-run colab status
annolid-run colab auth login
annolid-run colab train yolo --data /path/to/data.yaml --weights yolo11s-seg.pt --gpu T4
annolid-run colab predict yolo_labelme --source /path/to/video.mp4 --weights /path/to/best.pt --gpu T4
```

## Official Notebooks

- **Unified Annolid Train & Inference (recommended):**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_Colab_Train_and_Inference.ipynb)

- **YOLO Instance Segmentation:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_Instance_Segmentation_on_YOLO_Tutorial.ipynb)

- **YOLO Pose Estimation:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_Pose_Estimation_on_YOLO_Tutorial.ipynb)

- **Legacy Detectron2 Mask R-CNN:**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/healthonrails/annolid/blob/main/docs/tutorials/Annolid_on_Detectron2_Tutorial.ipynb)

## Typical Workflow

1. Label frames locally in Annolid and export a dataset (YOLO `data.yaml`, LabelMe `spec.yaml`, or COCO folder).
2. Authenticate via **Google Colab Studio** or `annolid-run colab auth login` (supports `google-colab-cli`, Google OAuth2 for Colab API + Drive, or `gcloud` ADC).
3. Run training or inference directly on a Colab GPU (`cli` / `api` backend) or generate a Google Drive-synced `.ipynb` notebook (`notebook` backend).
4. Trained checkpoints and inference LabelMe JSONs / tracking CSVs are downloaded locally (or pulled via `annolid-run colab pull <job_id>`).

```{note}
For full authentication options, CLI flags, and Annolid Bot setup, see `docs/google_colab.md`.
```
