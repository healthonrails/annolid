---
name: model-training
description: Inspect datasets, select an Annolid model family, launch local or remote training jobs, monitor progress, and evaluate trained checkpoints safely.
metadata: '{"annolid":{"always":false,"tags":["train","training","fine-tune","yolo","dino_kpseg","maskrcnn","keypoint","behavior_classifier","checkpoint","epochs","weights","annolid_train_start","runs"]}}'
---

# Model Training & Fine-Tuning

Use this skill when the user wants to train or fine-tune a detection, segmentation, keypoint/pose, or behavior model in Annolid.

## Recommended Pipeline

### 1. Inspect & Prepare Dataset First (Never Train Blind)
- Read-only audit: `annolid_dataset_inspect(dataset_dir="...")`
- Convert/prepare if needed (mutating — set `allow_mutation=true`):
  - `annolid_dataset_prepare(mode="labelme_spec"|"yolo_from_labelme"|"coco_spec"|"coco_to_yolo"|"deeplabcut_import"|"sleap_import", source_dir="...", output_dir="...", allow_mutation=true)`
- For polygon/TCN behavior classifiers, follow the `polygon-classifier-training` skill.

### 2. Discover Supported Models & Exact Arguments
- List available local training families: `annolid_train_models()`
- Check model-specific flags before launching: `annolid_train_help(model="...")`

### 3. Launch Training (Local or Remote Colab)
- **Local training** (mutating — requires `allow_mutation=true`; prefer `background=true` for multi-epoch runs so the session stays responsive):
  ```text
  annolid_train_start(
    model="yolo",
    task="segment",
    data="/path/to/data.yaml",
    epochs=50,
    batch=16,
    imgsz=640,
    device="auto",
    run_name="mouse_seg_v1",
    background=true,
    allow_mutation=true
  )
  ```
- **Remote GPU training**: when the machine lacks GPU VRAM or the user asks for Colab, switch to the `google-colab` skill (`google_colab(action="train", ...)`).

### 4. Monitor & Evaluate
- Monitor background jobs via `exec_process(action="poll"|"log", session_id=...)` or `annolid-run runs list|show|logs`.
- After training completes, evaluate on a held-out split using `annolid_eval_start` and `annolid_eval_report` (see the `behavior-evaluation` skill).

## Guardrails

1. Always honor explicit `device="cpu"` requests.
2. Split datasets by recording/video group before training to avoid frame-level data leakage.
3. Confirm output directory (`output_dir` / `runs_root` / `run_name`) and checkpoint path before starting so existing runs are not clobbered.
4. Report exact command/config, dataset split counts, device used, checkpoint path (`best.pt` / `.pth`), and held-out validation metrics.
