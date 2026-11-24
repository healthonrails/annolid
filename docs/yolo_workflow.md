# YOLO & YOLOE Fine-Tuning and Inference Workflow

Annolid provides a unified workflow for fine-tuning and running inference with
Ultralytics **YOLO11**, **YOLO26**, and open-vocabulary **YOLOE** models across
the GUI and `annolid-run` CLI.

## Supported Tasks & Model Families

- **Families**: `YOLO11` (`yolo11n/s/m/l/x`), `YOLO26` (`yolo26n/s/m/l/x`), and
  open-vocabulary `YOLOE` (`yoloe-11*`, `yoloe-26*`, plus prompt-free `*-pf.pt`).
- **Tasks**:
  - Detection (`yolo11s.pt`, `yolo26s.pt`)
  - Instance Segmentation (`yolo11s-seg.pt`, `yolo26s-seg.pt`)
  - Pose Estimation (`yolo11s-pose.pt`, `yolo26s-pose.pt`)
  - Oriented Detection (`yolo11s-obb.pt`)
  - Classification (`yolo11s-cls.pt`)

## 1. Prepare a Dataset

Annolid accepts any of the following inputs directly for YOLO fine-tuning (both
in the GUI and CLI):

- A standard YOLO `data.yaml` file
- A dataset folder containing `data.yaml` or `YOLO_dataset/data.yaml`
- A COCO dataset folder (`annotations/*.json`, `train.json`/`val.json`, or
  split-local `train/annotations.json`) or COCO spec YAML — Annolid
  automatically converts and hardlinks COCO **detection boxes**, **instance
  segmentation masks** (polygon lists and uncompressed/compressed RLE masks),
  and **pose keypoints** into a temporary YOLO dataset without modifying your
  original files.

To export LabelMe annotations from an Annolid project to YOLO format:

- **GUI**: Use **Dataset Export Wizard** (or launch it directly from the
  **Training Wizard** dataset step).
- **CLI**:

```bash
annolid-run collect-labels --input /path/to/labeled_frames --output /path/to/dataset_index.jsonl
annolid-run index-to-yolo --index /path/to/dataset_index.jsonl --output /path/to/yolo_dataset
```

## 2. Choose a Fine-Tuning Preset

Curated presets provide sensible defaults for lab datasets while remaining fully
customizable:

| Preset key | GUI name | Best for | Key defaults |
| --- | --- | --- | --- |
| `balanced` | Balanced (Recommended) | General lab datasets (150+ labeled frames) | 100 epochs, 640px, batch 8, patience 50, cosine LR |
| `small_dataset` | Small Lab Dataset (Fine-tune) | Small datasets (~20–150 labeled frames) | 120 epochs, AdamW, `lr0=0.002`, `freeze=10`, cosine LR, RAM cache |
| `fast` | Quick Smoke Test | Verifying labels, classes, and pipeline quickly | 30 epochs, batch 16, patience 15 |
| `high_accuracy` | High Accuracy (960px) | Small animals, fine body parts, or dense scenes | 200 epochs, 960px, AdamW, `lr0=0.005`, patience 60, cosine LR |

In the **Training Wizard**, Annolid inspects your dataset automatically,
pre-selects the matching YOLO task (Detection / Segmentation / Pose), and
recommends `Small Lab Dataset (Fine-tune)` when the training split has 150 or
fewer images.

## 3. Fine-Tune in the GUI

### Training Wizard (Recommended)

1. Open **AI & Models -> Training Wizard...**
2. **Select Dataset**: Choose a YOLO `data.yaml` (or folder) on the **YOLO** tab
   or a COCO dataset folder on the **COCO** tab. Review the detected task,
   class list, and train/val image counts.
3. **Choose Backend**: Select **YOLO (Ultralytics)**.
4. **Configure**:
   - Pick **Family** (`YOLO11` or `YOLO26`), **Task**, and **Size**, or choose
     a **Recent Trained Run** / browse to a local `.pt` checkpoint to fine-tune.
   - Check **Resume interrupted run** only when continuing an interrupted job
     from `last.pt` with the exact saved epoch and optimizer state. Leave it
     unchecked to fine-tune on a new schedule or dataset.
   - Choose a **Fine-tune preset**, **Device** (`CUDA`, `Apple MPS`, or `CPU`),
     and optional advanced knobs (`lr0`, `patience`, `freeze`, `optimizer`,
     `cos_lr`, `cache`, `plots`).
5. **Review & Launch**: Confirm the summary and optionally open the **Training
   Monitor & Visualization Dashboard** (and auto-start embedded **TensorBoard**
   if desired).

### Monitoring & Visualization (`Ctrl+Shift+V`)

Open **AI & Models -> Training Monitor & Visualization…** (or click
**Visualization** / press `Ctrl+Shift+V` at any time) to inspect live and
historical runs across YOLO, DINO KPSEG, Mask R-CNN, Keypoint R-CNN, and
downloaded Colab runs:

- **Monitor & Curves**: Inspect live progress, KPI cards, native interactive
  loss/mAP/PCK/learning-rate curves (with TensorBoard-style EMA smoothing, log
  Y-scale, hover readout, and PNG export), and check multiple runs to overlay
  them side by side without launching an external server.
- **Compare & Diff**: View a multi-run summary matrix (latest & min losses,
  best validation metric and step, tags, and health status) alongside a
  highlighted hyperparameter diff table merging `args.yaml` and CLI flags from
  `command.txt`.
- **Notes, Tags & Starred Runs**: Bookmark favorite runs (`★ Star`), assign
  short aliases, add searchable tags, and record lab notes stored
  non-destructively in `annolid_run_meta.json` inside each run folder; filter
  the runs list by `Starred only`, tag, alias, or notes.
- **Automated Health Diagnostics**: Automatically surfaces actionable alerts
  for `NaN`/`Inf` metrics, sudden loss spikes, overfitting trends, metric
  plateaus, finished runs missing checkpoints, and low free disk space.
- **Live System Telemetry**: Displays non-blocking CPU, RAM, disk free space,
  and GPU/VRAM utilization (NVIDIA via `nvidia-smi` or Apple Silicon MPS hint)
  in the header and **System** sub-tab.
- **Offline Reports & Optional Tracker Export**: Click **Export / Trackers** to
  export a self-contained offline **HTML5 + inline SVG + Markdown + JSON + CSV**
  experiment report, or push runs to optional external backends (**MLflow**,
  **Weights & Biases** offline/online, and **Rerun** `.rrd` recordings) when
  installed.
- **Plots & Images**: Preview generated `results.png`, confusion matrices,
  PR/F1 curves, and validation batch overlays directly in the dashboard.
- **TensorBoard & Projector**: Launch a non-blocking embedded (or browser)
  TensorBoard server scoped to the **Selected Run**, **Compare Checked Runs**
  (`--logdir_spec`), or **All Runs (Root)**, including one-click access to the
  **Embedding Projector** when `projector_config.pbtxt` is present.

When training completes:

- The trained `best.pt` checkpoint is automatically discovered and added to the
  main window AI model dropdown.
- The completion dialog provides one-click actions to **Use Model Now**,
  **Run Inference...**, or **Open Run Folder**.
- You can also stop an active run or launch inference on any completed run from
  the **Training Monitor & Visualization Dashboard**.

### Train Models Dialog (Advanced Single-Window Form)

**AI & Models -> Train Models... -> YOLO** exposes the same presets, family/task/size
selectors, dataset folder picker with inline inspection, backbone layer freezing,
and explicit fine-tune vs. exact `resume=True` toggle.

## 4. Fine-Tune from the CLI

```bash
# Fine-tune YOLO11s-seg on a small lab dataset (accepts data.yaml, folder, or COCO folder)
annolid-run train yolo \
  --data /path/to/dataset \
  --weights yolo11s-seg.pt \
  --preset small_dataset

# Fine-tune YOLO26s-pose with explicit settings
annolid-run train yolo \
  --data /path/to/pose_dataset/data.yaml \
  --weights yolo26s-pose.pt \
  --epochs 150 --imgsz 640 --batch 16 --freeze 10 --cos-lr

# Use a version-controlled run config YAML (CLI flags override YAML values)
annolid-run train yolo \
  --run-config annolid/configs/runs/yolo_train.yaml \
  --data /path/to/data.yaml

# Resume an interrupted training run from last.pt
annolid-run train yolo \
  --data /path/to/data.yaml \
  --weights /path/to/run/weights/last.pt \
  --resume
```

Managed CLI runs default to Annolid's shared runs root
(`<runs_root>/yolo/<model>/<timestamp>/weights/best.pt`) so models trained from
the terminal appear automatically in the GUI model selector and Inference Wizard.
Pass `--project` (and optionally `--name`) to write outputs to a custom location.

## 5. Run Inference & Export Annotations

### Inference Wizard (Batch Video Processing in GUI)

1. Open **AI & Models -> Inference Wizard...** (or click **Run Inference...**
   after training completes).
2. Pick a recently trained checkpoint from **Quick pick** / **Recent Models**,
   select a pretrained YOLO/YOLOE model, or browse to a `.pt` / `.onnx` /
   `.engine` / `.mlpackage` file.
3. Select one video, multiple videos, or a folder of videos (and an optional
   frame range).
4. Configure **Confidence threshold**, **NMS IoU threshold**, **Image size**,
   **Device**, **Tracking** (`ByteTrack` / `BoT-SORT` / `OC-SORT`) or
   **Frame batch size** (when tracking is disabled for faster GPU throughput),
   and outputs (**LabelMe annotations**, **Tracking CSV**, and optional
   **Annotated video**).

### Interactive Canvas Prediction (GUI)

Select any trained YOLO run or built-in YOLO/YOLOE model in the main toolbar AI
model dropdown and click **Predict**. The loaded YOLO processor is cached across
repeated prediction runs while the model and prompt configuration remain
unchanged.

### CLI Inference

Use `yolo_labelme` to write Annolid-compatible LabelMe JSONs (plus optional
tracking CSVs and annotated preview video):

```bash
# Custom trained YOLO model with tracking + CSV export
annolid-run predict yolo_labelme \
  --weights /path/to/weights/best.pt \
  --source /path/to/video.mp4 \
  --conf 0.30 --iou 0.70 --imgsz 640 \
  --save-csv --skip-existing

# Faster batched video inference without stateful tracking
annolid-run predict yolo_labelme \
  --weights /path/to/weights/best.pt \
  --source /path/to/video.mp4 \
  --no-tracking --batch 8 --save-video /path/to/preview.mp4

# Open-vocabulary YOLOE text prompting
annolid-run predict yolo_labelme \
  --weights yoloe-26s-seg.pt \
  --source /path/to/video.mp4 \
  --classes "black mouse,white mouse"
```

Use `yolo` for lightweight streaming Ultralytics inference (constant memory on
long videos):

```bash
annolid-run predict yolo \
  --weights /path/to/weights/best.pt \
  --source /path/to/video.mp4 \
  --track --conf 0.30 --save
```

## 6. Export Weights for Realtime / Edge Deployment

```bash
python -m annolid.realtime.convert \
  --model /path/to/weights/best.pt \
  --format onnx \
  --imgsz 640
```

Supported formats include `onnx`, `openvino`, `engine` (TensorRT), and
`coreml` (Apple Silicon).

## Hardware & Troubleshooting Notes

- **Device selection & Apple Silicon MPS**: Both training and inference
  automatically prefer CUDA -> Apple Silicon MPS -> CPU unless `--device` is
  explicitly set (inference also checks for a sibling TensorRT `.engine` on
  CUDA). On macOS, Annolid sets `PYTORCH_ENABLE_MPS_FALLBACK=1` and
  `MPLBACKEND=Agg` in YOLO CLI subprocesses so unsupported MPS ops fall back to
  CPU safely instead of aborting; if a specific PyTorch/Ultralytics build still
  hits an MPS error, pass `--device cpu` (or select **CPU** in the GUI).
- **macOS training plots**: Ultralytics training plots are disabled by default
  on macOS in GUI runs to avoid matplotlib/GUI thread crashes; enable them only
  when needed or use TensorBoard in the Training Dashboard.
- **Small datasets**: Prefer `small_dataset` (`freeze=10`, `AdamW`, cosine LR)
  and start with a smaller architecture (`n` or `s`) before scaling to `m`/`l`/`x`.
