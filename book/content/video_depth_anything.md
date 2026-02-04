# Video Depth Anything (Depth Estimation)

Annolid bundles the [Video-Depth-Anything](https://github.com/DepthAnything/Video-Depth-Anything) model family so you can estimate per-frame depth for a video directly inside the GUI (or from Python).

## Quick start (GUI)

1. Launch Annolid and open a video (**File → Open Video**).
2. (Optional) Configure defaults via **View → Depth Settings…**:
   - encoder (`vits`, `vitb`, `vitl`)
   - max resolution / input size
   - target FPS / max frames
   - streaming mode (recommended)
   - save depth video / frames / point cloud CSVs
3. Run **View → Video Depth Anything…**.

By default, outputs are written next to the video in a folder named after the video stem. For example, `videos/mice.mp4` writes to `videos/mice/`. You can override the output folder by setting `video_depth_anything.output_dir` in your Annolid config (see “Configuration” below).

## Model weights (checkpoints)

Annolid stores weights under `annolid/depth/checkpoints` and auto-downloads the one you selected when you run the tool (via Hugging Face).

If you want to pre-download checkpoints (useful for offline runs), use:

```bash
python -m annolid.depth.download_weights --list
python -m annolid.depth.download_weights --model video_depth_anything_vits
python -m annolid.depth.download_weights --model metric_video_depth_anything_vitb
```

If you are behind an authenticated Hugging Face endpoint, set `HF_HUB_TOKEN` before running the downloader (and before running Annolid if you want auto-download to work).

## Run from Python (batch processing)

You can call the same runner from a script:

```python
from pathlib import Path

from annolid.depth import run_video_depth_anything

input_video = Path("videos/mice.mp4")
output_dir = Path("outputs/mice_depth")

result = run_video_depth_anything(
    input_video=str(input_video),
    output_dir=str(output_dir),
    encoder="vits",
    streaming=True,
    target_fps=15,
    max_res=1280,
    save_depth_video=True,
    save_depth_frames=True,
    save_point_clouds=False,
)

print(result.get("depth_ndjson"))
```

Device selection is automatic (`cuda` → `mps` → `cpu`). If you see unstable depths on GPU, try `fp32=True` to force FP32 inference.

## Output files

Inside `output_dir`, you will typically see:

- `depth.ndjson` (always): one JSON record per processed frame, containing a base64-encoded `uint16` PNG depth map plus per-frame `scale` metadata.
- `<video_stem>_vis.mp4` (if `save_depth_video=True`): a depth-visualization video.
- `depth_frames/depth_00000.png` (if `save_depth_frames=True`): per-frame depth images (inferno palette or grayscale).
- `point_clouds/point_0000.csv` (if `save_point_clouds=True`): point cloud CSVs (XYZ + intensity, optionally region labels + RGB).
- `<video_stem>_depths.npz` (if `save_npz=True`) and/or `<video_stem>_depths_exr/` (if `save_exr=True`).

Notes:
- `metric=True` uses “Metric Video Depth Anything” checkpoints; the resulting depth values are intended to be metric, but still depend on the model and your camera setup.
- EXR export requires Python packages `OpenEXR` and `Imath`.

## Reading `depth.ndjson`

Each record stores a quantized `uint16` PNG plus per-frame min/max values that let you reconstruct the float depth map:

```python
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

record = json.loads(Path("outputs/mice_depth/depth.ndjson").read_text().splitlines()[0])
encoded = record["otherData"]["depth_map"]["image_data"]
scale = record["otherData"]["depth_map"]["scale"]  # {"min": ..., "max": ...}

image = Image.open(io.BytesIO(base64.b64decode(encoded)))
quant = np.array(image, dtype=np.float32)
depth = (quant / 65535.0) * (scale["max"] - scale["min"]) + scale["min"]
```

## Configuration

Annolid stores GUI settings under the `video_depth_anything` key in your config (typically `~/.labelmerc`). You can set defaults (or an output directory) like:

```yaml
video_depth_anything:
  encoder: vits
  streaming: true
  target_fps: 15
  max_res: 1280
  save_depth_video: false
  save_depth_frames: false
  save_point_clouds: false
  output_dir: /path/to/depth_outputs
```
