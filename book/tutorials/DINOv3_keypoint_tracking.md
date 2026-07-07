# DINOv3 Keypoint Tracking

This tutorial shows how to track labeled body-part keypoints through a video using frozen DINO-family patch descriptors, local motion-prior search, optional mask constraints, and Lucas-Kanade pixel refinement. The backend is a sparse-label tracker: it propagates manual seed points and correction frames, but it does not train a new pose-estimation model.

## Prerequisites
- Annolid installed with the GUI, model, and tracking runtime dependencies, and
  the GUI launches (`annolid`).
- One saved annotation JSON on the starting frame containing:
  - One or more point keypoints per instance (e.g., `nose`, `leftear`, `rightear`).
  - Optional polygons or masks for each tracked instance. These are recommended when nearby body parts look similar, but point-only tracking is supported.
- Internet access for the first run to download DINO-family weights from Hugging Face, or a local cache at `DINOV3_LOCATION`.
- Python packages from Annolid's `ml` and `tracking` extras available in your
  environment.

Tip: If the DINOv3 checkpoint is gated, accept the model license on Hugging Face and run `hf auth login` once in your shell, or set `HF_TOKEN`.

## 1. Prepare the initial frame
1. Open your video in Annolid and navigate to the frame you want to start from.
2. Add point keypoints for each instance (use the Point tool). Keep names consistent across frames, e.g., `nose`, `leftear`, `rightear`, `tail_base`.
3. When available, draw a polygon around each instance you want to track (Tools → Create Polygons, or use Segment Anything to speed up). Masks help constrain ambiguous keypoints.
4. Save (`Ctrl+S`). This creates `<video_name>/<video_name>_#########.json` next to your video.

Notes on keypoint naming:
- To enable symmetry constraints (prevent left/right swaps), set `symmetry_pairs` in `CutieDinoTrackerConfig`, e.g. `symmetry_pairs=(("leftear","rightear"),)`.
- Annolid stores instance/keypoint association in the shape list; adding points while your instance is selected ensures correct linkage.

## 2. Choose and cache the DINO-family feature backbone
Open **Tools → Advanced Parameters → Tracker → DINO feature model** and pick a model. Recommended starting point: **DINOv3 ViT-S/16 (gated, recommended)**. You can also paste a Hugging Face model id into this editable field.

Common choices:

| Model | Best use |
| --- | --- |
| `facebook/dinov3-vits16-pretrain-lvd1689m` | Default interactive tracking; fastest DINOv3 option. |
| `facebook/dinov3-vits16plus-pretrain-lvd1689m` | Slightly stronger descriptors with modest extra memory. |
| `facebook/dinov3-vitb16-pretrain-lvd1426` | Medium offline runs when ViT-S is not distinctive enough. |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | Higher-memory offline runs. |
| `facebook/dinov3-vith16plus-pretrain-lvd1689m` or `facebook/dinov3-vit7b16-pretrain-lvd1689m` | Specialized high-memory workflows; pre-download before long videos. |
| `facebook/dinov2-base` or `facebook/dinov2-large` | Open fallback if DINOv3 access is unavailable. |
| `nvidia/C-RADIOv4-SO400M` | Compatible feature backbone for specialized workflows. |

To list the supported catalog:

```bash
annolid-run dinov3-models --list
```

To pre-download the default model before opening the GUI:

```bash
annolid-run dinov3-models --model facebook/dinov3-vits16-pretrain-lvd1689m
```

To use a shared local Hugging Face cache:

```bash
export DINOV3_LOCATION=/path/to/hf-cache
annolid-run dinov3-models --model facebook/dinov3-vits16-pretrain-lvd1689m
```

The patch similarity and PCA map tools still have their own model setting under **Tools → Patch Similarity Settings** and **Tools → PCA Feature Map Settings**.

## 3. Start DINOv3 keypoint tracking
1. In the model dropdown (top toolbar), select “DINOv3 Keypoint Tracker”.
2. Press Pred button. Annolid will:
   - Load the selected DINO-family model and extract dense patch features for each frame.
   - Use your labeled points as anchors and follow them with feature similarity, motion priors, and stateful track memory.
   - Use dense Farneback flow plus per-point pyramidal Lucas-Kanade flow to guide local search and sub-patch refinement.
   - Optionally constrain search to the instance mask (from your polygon, or propagated by Cutie if enabled).
3. Tracking runs from the current frame to the end of the video. Progress appears in the status bar.

Outputs (written live):
- Per‑frame LabelMe JSONs with updated keypoint locations (and masks if present) in `<video_name>/`.
- Point metadata includes backend description, quality, velocity, miss count, instance label, and display label when available.

## 4. Advanced parameters (optional)
Open Tools → Advanced Parameters to fine‑tune tracking:
- Tracker preset: choose a tuned preset such as `fly_70fps_keypoints` or `rodent_30fps_occlusions`, or keep custom values.
- DINO feature model: choose the DINO-family backbone used for video keypoint tracking. Larger models can improve matching but increase download size, startup time, and memory use.
- Clamp keypoints to instance mask and Mask snap radius: keep points inside their masks; increase radius to be more permissive near edges.
- Search tighten, Velocity gain, Flow gain, Min/Max radius, Miss boost: tune the motion prior (uses optical flow and recent velocity).
- Motion prior weight/soft radius/factor/miss relief/flow relief: adjust how strongly the motion prior penalizes unlikely jumps.
- DINOv3 positional debias: suppress low-dimensional coordinate bias in DINO features before matching, useful when a keypoint repeatedly snaps to the same image location instead of the same body part.
- DINOv3 backward consistency: prefer candidates whose nearest match in the previous frame maps back to the prior keypoint patch.
- Coherent peak refinement: refine a keypoint using the connected local similarity peak around the selected seed, rather than unrelated nearby peaks.
- Keypoint refine radius/sigma/temperature: track a small patch neighborhood around each keypoint and report the Gaussian‑weighted centroid (sub‑patch smoothing).
- Pixel refinement weight/window/error/jump: blend the DINO-selected point with a valid Lucas-Kanade estimate to recover motion inside one ViT patch.
- Appearance bundle, context, support probes, and structural consistency: add local state that helps distinguish similar landmarks and recover after short misses.

Defaults work well on most videos; increase Search tighten and reduce Max radius for tighter, slower motion; do the opposite for fast motion.

## 5. Inspect and correct
- Step through frames to verify trajectories. Edit any point or polygon and press Save — the tracker will resume from that updated ground truth if you run it again.
- Treat corrections as sparse reseed frames. Correct the first frame where drift begins, then rerun from that frame instead of relabeling every frame.
- Use Tools → Patch Similarity to click a region and visualize DINO similarity heatmaps if you need to diagnose difficult frames.

## 6. How the tracker differs from pose estimation
DINOv3 Keypoint Tracker is best for video-specific propagation from sparse manual seeds. It is useful when you need reviewable trajectories quickly and can correct drift. It is not a reusable pose model trained across many videos.

Choose a supervised pose-estimation workflow when you already have enough labeled training frames, need a reusable model for unseen sessions, or need standardized pose-model benchmarks. Choose TAPNext, CoTracker, or CoWTracker when you want another point-tracking backend under the same reviewable Annolid output contract. INSID3 is adjacent: it uses DINOv3 for in-context segmentation masks, not direct point tracking.

## Tips
- Keep keypoint labels consistent across individuals and sessions.
- Add 2–4 reliable anchor keypoints per instance (e.g., nose, ears, tail_base) for robust tracking.
- Use masks or polygons when similar body parts are close together.
- Good lighting and moderate resolution (short side ≈ 768 px) help DINO descriptors remain distinctive.
- If DINO downloads are blocked, run `annolid-run dinov3-models --local-files-only --model <model-id>` to verify that the checkpoint is already cached; see the log for the exact model id in use.

## Programmatic use (optional)
You can run the tracker from Python for scripted workflows:

```python
from annolid.tracking.dino_keypoint_tracker import DinoKeypointVideoProcessor
from annolid.tracking.configuration import CutieDinoTrackerConfig

vp = DinoKeypointVideoProcessor(
    video_path="/path/to/video.mp4",
    result_folder=None,  # defaults to /path/to/video/
    model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
    short_side=768,
    device=None,  # auto-selects CUDA/MPS/CPU
    runtime_config=CutieDinoTrackerConfig.from_preset("rodent_30fps_occlusions"),
)
vp.process_video()
```

The processor reads your initial JSON from the video's output folder, tracks to the end, and writes updated JSON files.

---
If you run into issues loading DINOv3 weights, ensure `transformers` is installed and that you have access to the selected Hugging Face checkpoint.
