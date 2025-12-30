# Video Depth Anything Tutorial

Annolid bundles the [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) suite so you can estimate per-frame depth without leaving the GUI or writing a complex pipeline. The helper exposed as `annolid.depth.run_video_depth_anything` can be used in scripts, while the GUI exposes the same functionality under **View → Video Depth Anything**.

## 1. Prerequisites

The dependencies that power Video Depth Anything are already included when you install Annolid (`pip install -e .`), so you don’t need to run any additional `pip` commands. The tool prefers GPU inference (and falls back to CPU with forced FP32 if no CUDA device is available). For EXR export you also need `OpenEXR`/`Imath` packages.

## 2. Download the checkpoints

Annolid stores the Depth Anything weights under `annolid/depth/checkpoints`. You can let the model download them when you run the tool, but you can also fetch them ahead of time with the bundled downloader:

```bash
python -m annolid.depth.download_weights --all
```

If you only need a specific encoder, replace `--all` with repeated `--model` flags such as `--model vitb`. If you are behind an authenticated Hugging Face endpoint, set `HF_HUB_TOKEN` before running the script.


## 3. Run inside the GUI

1. Load a video (File → Open Video).
2. Use **View → Depth Settings…** (or the widget inside the depth settings) to tweak the defaults (encoder, resolution, grayscale overlay, point cloud export, etc.). Changes saved here immediately affect the next depth run, so you never have to open a CLI shell once you fine-tune your preferences.
3. Choose **View → Video Depth Anything…** to start a background worker. The status bar shows progress and an overlay is painted on the canvas.

Streaming mode (default) feeds one frame at a time through the model, which keeps memory usage low and lets you preview intermediate results via the overlay. The dialog keeps `streaming`, `save_depth_video`, `save_depth_frames`, `save_point_clouds`, and other toggles in sync with the call to `run_video_depth_anything`, so updating the settings in the view propagates automatically when you click the menu action.

## 4. Run depth estimation from Python

Because `run_video_depth_anything` is a normal Python function, you can drop it into scripts for batch processing:

```python
from pathlib import Path
from annolid.depth import run_video_depth_anything

input_video = Path("videos/mice.mp4")
output_dir = Path("outputs/mice_depth")

result = run_video_depth_anything(
    input_video=str(input_video),
    output_dir=str(output_dir),
    encoder="vitb",
    max_res=900,
    max_len=500,
    target_fps=15,
    save_depth_video=True,
    save_depth_frames=True,
    save_point_clouds=True,
)

print(result["depth_ndjson"])  # point to the depth maps you can re‑use later
```

If the `result` dictionary contains keys such as `npz`, `exr`, `point_cloud_csv`, etc., those paths point to additional files the tool created.


## Output layout

The tool always writes to the directory you pass as `output_dir`. Within that directory you will typically see:

- `depth.ndjson` – one JSON record per processed frame. Each record stores the depth map as a base64-encoded `uint16` PNG under `otherData.depth_map`. The per-frame `"scale"` tells you how to map the quantized values back to float depths (metric units only when `metric=True`).
- `<video_stem>_vis.mp4` (if `save_depth_video=True`) – a short MP4 where depth maps are rendered with the inferno palette (or grayscale if you asked for it).
- `depth_frames/depth_00000.png` (if `save_depth_frames=True`) – per-frame PNGs with either palette colors or grayscale values.
- `point_clouds/point_0000.csv` (if `save_point_clouds=True`) – XYZ-intensity CSVs. Columns are `x,y,z,intensity`, followed by an optional `region` column (if you enabled `include_region_labels` and annotated regions exist) and `red,green,blue` when color data is available.
- `*_depths.npz` (if `save_npz=True`) and/or `*_depths_exr/*` (if `save_exr=True`) – compressed NumPy or EXR stacks for downstream processing.

You can decode a depth map like this:

```python
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

output_dir = Path("outputs/mice_depth")
record = json.loads(output_dir.joinpath("depth.ndjson").read_text().splitlines()[0])
encoded = record["otherData"]["depth_map"]["image_data"]
scale = record["otherData"]["depth_map"]["scale"]

array = Image.open(io.BytesIO(base64.b64decode(encoded)))
depth = (np.array(array, dtype=np.float32) / 65535.0) * (scale["max"] - scale["min"]) + scale["min"]
```

## Advanced knobs

- **Encoder** – choose between `vits`, `vitb`, and `vitl`. The larger encoders give richer detail but need more GPU memory.
- **Input size** – the spatial size the network uses internally. A lower value is faster but may blur fine geometry.
- **Max resolution** – resizes each frame so the longer side does not exceed this before inference.
- **Max frames / Target FPS** – limit how many frames are processed or skip frames to reduce runtime.
- **Metric depth model** – enables the metric variant of the checkpoints (prefix `metric_`) so depth maps are already scaled in meters.
- **FP32 inference** – Forces Float32 even on CUDA since the GPU default is often FP16 and may produce flickering depth.
- **Grayscale overlay** – changes both the GUI overlay and `depth_frames` output to use a grayscale palette.
- **Save depth video / frames / NPZ / EXR / point clouds** – turn on each artifact you care about. EXR writing requires `OpenEXR`+`Imath`.
- **Include region labels** – if a folder named like your video (e.g., `videos/mice` for `videos/mice.mp4`) exists and contains Annolid tracking JSON/annotation records, the generated CSV files get a `region` column so you can link depth back to tracked animals.
- **Streaming vs batch** – streaming (default) runs one frame at a time for long videos, whereas batch mode loads a chunk of frames before inferring them together.
- **Focal lengths** – used when projecting depths to point clouds. The defaults align with the training data but you can override them to match your camera.
- **Checkpoint root** – point `checkpoint_root` at another directory to reuse downloaded weights stored elsewhere.

## Tips

- If you process many videos, download all checkpoints once with `download_weights` to avoid repeated Hugging Face fetches.
- Keep an eye on the status bar or console log; the worker reports progress percentages and logs a warning if FG/point-cloud generation fails.
- `depth.ndjson` keeps the original frame index and video name so you can re-align depth maps with tracked skeletons or behaviors later.

With this flow, you can generate per-frame depth, save visualizations, and even pull out point clouds for downstream analysis without leaving Annolid.
