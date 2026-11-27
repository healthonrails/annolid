---
name: google-colab
description: Run remote GPU/TPU training, inference, notebook generation, session management, and result syncing on Google Colab via the google_colab tool or annolid-run colab.
metadata: '{"annolid":{"always":false,"tags":["colab","google colab","gpu","t4","l4","a100","h100","tpu","remote training","remote gpu","notebook","colab-cli","pull results","cloud gpu"]}}'
---

# Google Colab Remote Execution

Use this skill when the user asks to train or run inference on Google Colab GPUs/TPUs, check Colab auth/doctor status, manage warm Colab sessions, generate Colab notebooks, or pull training artifacts back locally.

## Preferred Tool: `google_colab`

Always use the typed `google_colab` tool instead of ad hoc shell commands when available (CLI equivalent: `annolid-run colab ...`).

### Preflight & Discovery Actions (Read-Only)

1. `google_colab(action="auth_status")`: verify Colab credentials / active backend readiness.
2. `google_colab(action="doctor")`: diagnose local `colab-cli`, Google Drive auth, and backend health.
3. `google_colab(action="list_specs")`: list supported models, accelerators, and backends.
4. `google_colab(action="list_sessions")`: inspect warm reusable Colab runtimes.
5. `google_colab(action="list_jobs", limit=20)` / `google_colab(action="list_templates")`: inspect recent runs or starter notebook templates.

### Execution & Artifact Actions

- **Train remotely**:
  ```text
  google_colab(
    action="train",
    model="yolo",
    dataset_path="/path/to/dataset",
    accelerator="L4",
    backend="auto",
    epochs=50,
    batch=16,
    imgsz=640,
    keep_runtime=true
  )
  ```
- **Run remote inference**:
  ```text
  google_colab(
    action="predict",
    model="yolo",
    source_path="/path/to/video.mp4",
    weights="/path/to/best.pt",
    accelerator="T4",
    save_csv=true,
    save_video=true
  )
  ```
- **Generate a standalone notebook** (useful when `backend="notebook"` or for manual Colab browser runs):
  `google_colab(action="generate_notebook", model="...", dataset_path="...", output_dir="...")`
- **Pull results back locally**:
  `google_colab(action="pull_results", job_id="...", output_dir="...")`
- **Release compute units when finished**:
  `google_colab(action="stop_session", session="...")`

## Supported Options

- **Models**: `yolo`, `yolo_labelme`, `dino_kpseg`, `maskrcnn_detectron2`, `keypoint_rcnn`, `tcn_behavior`, `polygon_classifier`, `behavior_classifier`.
- **Accelerators**: `T4`, `L4`, `A100`, `H100`, `CPU`, `TPU_V5E1`, `TPU_V6E1`.
- **Backends**: `auto` (default), `cli` (`colab-cli`), `api`, `notebook`.

## Best Practices & Guardrails

1. Check `auth_status` or `doctor` first before launching a long remote job.
2. Inspect datasets locally with `annolid_dataset_inspect` before uploading/packaging.
3. Reuse warm sessions (`session` + `keep_runtime=true`) during iterative tuning, and explicitly call `stop_session` when done to avoid wasting compute units.
4. Never embed tokens or credentials in prompts or generated notebooks.
5. Return exact `job_id`, `session`, accelerator, and pulled artifact paths (`weights`, metrics, predictions) so downstream evaluation (`behavior-evaluation`) can consume them.
