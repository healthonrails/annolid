# Video Depth Anything

Annolid bundles the [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) models to estimate per-frame depth maps from a video.

## Run in the GUI

1. Open a video (**File → Open Video**).
2. (Optional) Tune defaults in **View → Depth Settings…** (encoder, resolution, target FPS, streaming, and which outputs to save).
3. Start processing via **View → Video Depth Anything…**.

Outputs are written next to the video by default, in a folder named after the video stem (e.g. `videos/mice.mp4` → `videos/mice/`).

## Download checkpoints (optional)

Checkpoints live under `annolid/depth/checkpoints` and are auto-downloaded when you run the tool. To prefetch them:

```bash
python -m annolid.depth.download_weights --list
python -m annolid.depth.download_weights --model video_depth_anything_vits
```

Set `HF_HUB_TOKEN` if you need authentication to download from Hugging Face.

## Run from Python

```python
from annolid.depth import run_video_depth_anything

run_video_depth_anything(
    input_video="videos/mice.mp4",
    output_dir="outputs/mice_depth",
    encoder="vits",
    streaming=True,
    target_fps=15,
    save_depth_video=True,
)
```

## Outputs

- `depth.ndjson` (always): one JSON record per processed frame, with a base64-encoded `uint16` PNG depth map and a per-frame min/max `scale`.
- `<video_stem>_vis.mp4` (optional): saved when `save_depth_video=True`.
- `depth_frames/` and `point_clouds/` (optional): saved when enabled in settings.

