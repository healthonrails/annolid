# DINOv3 Keypoint Tracking

This tutorial shows how to track labeled body-part keypoints through a video using DINOv3 patch descriptors, optionally constrained by per-instance masks propagated by Cutie.

## Prerequisites
- Annolid v1.2.2+ installed and the GUI launches (`annolid`).
- One saved annotation JSON on the starting frame containing:
  - A polygon per tracked instance (e.g., each animal).
  - One or more point keypoints per instance (e.g., `nose`, `leftear`, `rightear`).
- Internet access for the first run to download DINOv3 weights from Hugging Face, or a local cache at `DINOV3_LOCATION`.
- Python package `transformers>=4.39` available in your environment (required by the DINOv3 extractor).

Tip: If the DINOv3 checkpoint is gated, run `huggingface-cli login` once in your shell, or pre‑download the model into a folder and set `DINOV3_LOCATION=/path/to/cache`.

## 1. Prepare the initial frame
1. Open your video in Annolid and navigate to the frame you want to start from.
2. Draw a polygon around each instance you want to track (Tools → Create Polygons, or use Segment Anything to speed up).
3. Add point keypoints for each instance (use the Point tool). Keep names consistent across frames, e.g., `nose`, `leftear`, `rightear`, `tail_base`.
4. Save (`Ctrl+S`). This creates `<video_name>/<video_name>_#########.json` next to your video.

Notes on keypoint naming:
- Symmetry handling assumes lowercase pairs like `leftear` and `rightear` by default; use those names to activate symmetry constraints.
- Annolid stores instance/keypoint association in the shape list; adding points while your instance is selected ensures correct linkage.

## 2. Choose the DINOv3 backbone (once)
- Open Tools → Patch Similarity Settings…
- Pick a DINO model. Recommended starting point: “DINOv3 ViT‑S/16 (gated)”.
- Adjust overlay opacity if you plan to use the heatmap tool; this setting is also used by keypoint tracking to select the backbone.

Behind the scenes, Annolid uses the same DINO model for both patch similarity overlays and keypoint tracking.

## 3. Start DINOv3 keypoint tracking
1. In the model dropdown (top toolbar), select “DINOv3 Keypoint Tracker”.
2. Press Pred button. Annolid will:
   - Load the DINO model and extract dense patch features for each frame.
   - Use your labeled points as anchors and follow them using cosine‑similarity in feature space.
   - Optionally constrain search to the instance mask (from your polygon, or propagated by Cutie if enabled).
3. Tracking runs from the current frame to the end of the video. Progress appears in the status bar.

Outputs (written live):
- Per‑frame LabelMe JSONs with updated keypoint locations (and masks if present) in `<video_name>/`.
- A CSV summary for downstream analysis in the same folder.

## 4. Advanced parameters (optional)
Open Tools → Advanced Parameters to fine‑tune tracking:
- Clamp keypoints to instance mask and Mask snap radius: keep points inside their masks; increase radius to be more permissive near edges.
- Search tighten, Velocity gain, Flow gain, Min/Max radius, Miss boost: tune the motion prior (uses optical flow and recent velocity).
- Motion prior weight/soft radius/factor/miss relief/flow relief: adjust how strongly the motion prior penalizes unlikely jumps.

Defaults work well on most videos; increase Search tighten and reduce Max radius for tighter, slower motion; do the opposite for fast motion.

## 5. Inspect and correct
- Step through frames to verify trajectories. Edit any point or polygon and press Save — the tracker will resume from that updated ground truth if you run it again.
- Use Tools → Patch Similarity to click a region and visualize DINO similarity heatmaps if you need to diagnose difficult frames.

## Tips
- Keep keypoint labels consistent across individuals and sessions.
- Add 2–4 reliable anchor keypoints per instance (e.g., nose, ears, tail_base) for robust tracking.
- Good lighting and moderate resolution (short side ≈ 768 px) help DINO descriptors remain distinctive.
- If DINO downloads are blocked, set `DINOV3_LOCATION` to a local folder containing the model checkpoint; see the log for the exact model id in use.

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
    runtime_config=CutieDinoTrackerConfig(mask_enforce_position=True),
)
vp.process_video()
```

The processor reads your initial JSON from the video’s output folder, tracks to the end, and writes updated JSON/CSV files.

---
If you run into issues loading DINOv3 weights, ensure `transformers` is installed and that you have access to the selected Hugging Face checkpoint.
