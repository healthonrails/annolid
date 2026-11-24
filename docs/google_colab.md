# Google Colab Training and Inference

Annolid supports authenticating with **Google Colab** and **Google Drive** to run GPU/TPU model training and video/image inference remotely from the **GUI**, the **`annolid-run colab` CLI**, or **Annolid Bot**.

Supported models include:

- **Training:** `yolo`, `dino_kpseg`, `maskrcnn_detectron2`, `keypoint_rcnn`, `tcn_behavior`, `polygon_classifier`, `behavior_classifier`
- **Inference:** `yolo_labelme` (exports LabelMe JSON + tracking CSVs), `yolo`, `dino_kpseg`, `maskrcnn_detectron2`, `keypoint_rcnn`

## Choose Your Execution Backend

Annolid automatically picks the best available backend (`--backend auto`), or you can choose one explicitly:

| Backend | Best For | Platforms | Auth Mechanism |
| --- | --- | --- | --- |
| **`cli`** (`google-colab-cli`) | Headless GPU/TPU runs with a personal Google account (no GCP project required) | macOS, Linux (WSL on Windows) | Browser OAuth2 via `colab` (saved in OS keyring) or `gcloud` ADC |
| **`api`** (`colaboratory.googleapis.com/v1beta`) | Cross-platform headless execution directly from Python | macOS, Linux, Windows | Shared Annolid Google OAuth2 (`colaboratory` scope) or `gcloud` ADC |
| **`notebook`** (Google Drive + `.ipynb`) | Interactive execution in Colab browser, Free tier, or restricted networks | macOS, Linux, Windows | Optional Google Drive OAuth2 (or manual upload of generated `.ipynb` + `.zip`) |

## 1. Authenticate with Google Colab

If you run a training or inference job without active credentials (or with expired tokens), Annolid guides you through sign-in and automatically resumes your job:

- **Interactive CLI:** prompts you to sign in via `google-colab-cli`, Google OAuth2, or `gcloud` ADC, or switch to `.ipynb` Notebook mode. Pass `--no-prompt` for non-interactive scripts/CI.
- **GUI (`Colab Studio`):** opens a guided **Sign In to Google Colab** dialog (with one-click **Install Colab CLI**, automatic browser launch, and a paste-code prompt if `google-colab-cli` uses its OAuth2 copy-paste flow) and automatically resumes your job after sign-in.

Check current readiness across all authentication paths:

```bash
annolid-run colab status
annolid-run colab auth status
annolid-run colab doctor
# Optional: verify real live connectivity against Colab API, Drive API, and Colab CLI
annolid-run colab doctor --live
```

### Option A: Official Google Colab CLI (Fastest on macOS / Linux)

Works with personal Google accounts and Colab Free/Pro/Pro+ tiers without creating a Google Cloud OAuth app:

```bash
# Install automatically if missing and sign in:
annolid-run colab auth login --method colab-cli --install-cli

# Or install manually first:
uv tool install google-colab-cli
# or: pip install google-colab-cli
annolid-run colab auth login --method colab-cli
```

Note: `colab --auth=oauth2` opens a Google sign-in URL and displays an authorization code tied to that exact run (PKCE). Always approve the newest browser tab opened by the current command and paste the code promptly. If `gcloud` is installed, Option C (`--method adc`) uses an automatic `localhost` browser redirect with no code copy/paste and also powers `google-colab-cli`.

### Option B: Shared Annolid Google OAuth2 (Colab API + Google Drive Sync)

Uses Annolid's shared OAuth files (`~/.annolid/agent/google_oauth_credentials.json` and `~/.annolid/agent/google_oauth_token.json`, mode `0600`):

1. Install the Google OAuth extra if not already installed:

```bash
pip install "annolid[annolid_bot]"
```

2. Place your Google Cloud OAuth Desktop client JSON at `~/.annolid/agent/google_oauth_credentials.json`.
3. Sign in once from CLI or GUI (merges with existing Drive/Calendar scopes instead of overwriting them):

```bash
# Request both Colab API and Google Drive scopes:
annolid-run colab auth login --method oauth2

# Or Google Drive sync only (for Notebook + Drive mode):
annolid-run colab auth login --method drive
```

### Option C: Google Cloud Application Default Credentials (ADC)

Recommended when `gcloud` is installed — opens your browser with automatic `localhost` redirect (no code copy/paste) and automatically configures `google-colab-cli` (`--cli-auth auto` -> `adc`):

```bash
annolid-run colab auth login --method adc
```

On Google's consent screen, make sure **all** requested permission checkboxes (including Google Cloud and Google Colab) are selected before clicking **Continue**.

Or run `gcloud` directly:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
```

To check which personal Google account is active and view your Colab compute-unit balance, or sign out to switch accounts:

```bash
annolid-run colab auth whoami
annolid-run colab auth logout --all
```

## 2. Inspect Eligible GPUs and Manage Runtimes

```bash
# List eligible CPU/GPU/TPU runtime specs via Colab API
annolid-run colab specs

# List active sessions / runtimes
annolid-run colab sessions

# Provision a persistent GPU session manually
annolid-run colab new --session annolid-gpu --gpu T4

# Stop a session or runtime
annolid-run colab stop annolid-gpu
```

## 3. Train a Model on Google Colab

When packaging a YOLO `data.yaml` or LabelMe/COCO `spec.yaml`, Annolid automatically builds a portable `.zip` archive with normalized relative paths (`path: .`) without modifying your original dataset on disk, so macOS/Windows paths never break on Linux Colab containers.

### YOLO Instance Segmentation / Pose Training

```bash
annolid-run colab train yolo \
  --data /path/to/my_dataset/data.yaml \
  --weights yolo11s-seg.pt \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --gpu T4 \
  --output ./colab_runs/yolo_run
```

### DINO KPSEG Training

```bash
annolid-run colab train dino_kpseg \
  --data /path/to/labelme_spec.yaml \
  --epochs 150 \
  --batch 8 \
  --gpu L4 \
  --output ./colab_runs/dino_run
```

### Torchvision Mask R-CNN or Keypoint R-CNN Training

```bash
annolid-run colab train maskrcnn_detectron2 \
  --data /path/to/coco_dataset \
  --max-iterations 3000 \
  --batch 4 \
  --gpu T4
```

### Reuse a Warm GPU Session Across Training and Inference

Pass `--session <name> --keep-runtime` on training, then reuse `--session <name>` on inference to skip VM cold-start and reuse cached Annolid packages on the same GPU:

```bash
annolid-run colab train yolo \
  --data /path/to/data.yaml --weights yolo11s-seg.pt --epochs 100 \
  --gpu T4 --session annolid-gpu --keep-runtime --output ./colab_runs/train

annolid-run colab predict yolo_labelme \
  --source /path/to/video.mp4 --weights ./colab_runs/train/train/weights/best.pt \
  --gpu T4 --session annolid-gpu --output ./colab_runs/predict

annolid-run colab stop annolid-gpu
```

In **Colab Studio** (GUI), when a training run finishes and downloads `best.pt`, Annolid automatically fills the **Inference on Colab** tab's weights (and session name when **Keep runtime alive** is checked) for one-click handoff.

### Notebook + Google Drive Mode (Interactive Browser)

```bash
annolid-run colab train yolo \
  --data /path/to/data.yaml \
  --weights yolo11s-seg.pt \
  --backend notebook \
  --drive-folder Annolid/colab \
  --open-browser
```

After running the notebook in Colab, pull the trained weights (`best.pt`, metrics, plots) back locally:

```bash
annolid-run colab jobs
annolid-run colab pull <job_id> --output ./colab_runs/pulled
```

## 4. Run Inference on Google Colab

Run video or image inference remotely and automatically download LabelMe JSON predictions, `*_tracking.csv`, `*_tracked.csv`, and optional annotated preview video:

```bash
annolid-run colab predict yolo_labelme \
  --source /path/to/video.mp4 \
  --weights /path/to/best.pt \
  --conf 0.3 \
  --save-video \
  --gpu T4 \
  --output ./colab_preds/video_01
```

## 5. Use Google Colab Studio from the Annolid GUI

1. Open **AI & Models -> Google Colab…** (or click the **Google Colab** toolbar button).
2. You can also configure a job in **Training Wizard** or **Inference Wizard** and click **Run on Google Colab…** to prefill Google Colab Studio with your selected dataset, video, model, and hyperparameters.
3. In the top **Personal Account Bar** (or the **Account & Runtimes** tab), click **Sign In with Google** to connect your own personal Google Colab account (Free, Pro, or Pro+) — no Google Cloud project setup required. Your signed-in email, compute-unit balance, and active runtimes appear directly in the header, and **Switch Account** lets another lab member sign out and log into their own personal account cleanly.
4. In **Train on Colab** or **Inference on Colab**, launch headless Colab jobs (with optional warm **Session Name** reuse and **Colab Pip Spec** selection) or generate a Drive-synced `.ipynb` notebook. When training completes, `best.pt` is automatically prefilled into the Inference tab.
5. In **Live Monitor** (opened automatically when a direct run starts), watch real-time **Stage**, **GPU (`nvidia-smi`)**, **Elapsed Timer**, **Epoch/Step Progress**, **KPI Metric Cards** (`Train Loss`, `Box mAP50`, `Mask mAP50`, `GPU Mem`, `Artifacts`), **Live Loss & mAP50 Curves**, **Training Plots Preview** (`results.png`, `confusion_matrix.png`, validation overlays), **Open Runtime in Colab Browser**, and a **filterable live log console** with Copy/Save/Auto-scroll.
6. In **Jobs & Notebooks**, click **Inspect in Monitor** to replay any past run's `results.csv` curves and plots, pull outputs from Google Drive, open output folders, click **Load Predictions in Annolid** to import LabelMe predictions directly onto the canvas, or launch curated official Annolid Colab notebooks.

## 6. Annolid Bot Integration

Enable the `google_colab` tool in `~/.annolid/config.json`:

```json
{
  "tools": {
    "google_auth": {
      "credentialsFile": "~/.annolid/agent/google_oauth_credentials.json",
      "tokenFile": "~/.annolid/agent/google_oauth_token.json",
      "allowInteractiveAuth": false
    },
    "google_drive": {
      "enabled": true
    },
    "google_colab": {
      "enabled": true,
      "backend": "auto",
      "defaultAccelerator": "T4",
      "driveFolder": "Annolid/colab"
    }
  }
}
```

## Large Datasets & Long Training Runs

- **Proxy token auto-refresh (`api` backend):** Colab runtime proxy tokens expire roughly every 60 minutes. Annolid automatically inspects `connectionInfo.expireTime` and refreshes the runtime proxy token before upload, execution, and download (as well as retrying once on proxy HTTP `401`/`403`) so multi-hour training jobs do not fail when downloading checkpoints.
- **O(1)-memory streaming downloads (`api` backend):** Output archives (`output_bundle.zip`) are streamed as binary bytes via the runtime `/files/` endpoint (with automatic fallback to `/api/contents/`).
- **Multi-GB video / dataset throughput:** For very large (>100 MB) datasets or raw videos, `--backend cli` (`google-colab-cli`) or `--backend notebook` (resumable Google Drive upload) avoids base64 overhead on the Jupyter Contents API.

## Opt-In Live Integration Tests

Default `pytest` runs stay completely offline and deterministic. When you have signed in to Google Colab / Drive on your machine, you can run read-only live verification probes with:

```bash
ANNOLID_COLAB_LIVE_TESTS=1 pytest tests/test_colab_live_integration.py -v
```

## Related Docs

- [Sim-to-Real Studio (3D Mouse/Fly Simulation → Colab)](sim2real_colab_workflow.md)
- [Google Integrations](agent_workspace.md)
- [YOLO & YOLOE Fine-Tuning and Inference](yolo_workflow.md)
- [Pose Estimation Workflow](pose_estimation_workflow.md)
- [Tutorials](tutorials.md)
