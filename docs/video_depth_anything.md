# Video Depth Anything

Annolid includes Video Depth Anything integration for estimating per-frame depth
from a loaded video. The GUI path is the normal workflow, and the same runtime is
available from Python through `annolid.depth.run_video_depth_anything`.

Use this workflow when you need a depth sidecar for review, visualization, point
cloud export, or later 2D-to-3D workflows such as FlyBody.

## Requirements

Video Depth Anything needs the GUI plus the model runtime dependencies used by
Annolid's ML workflows, including PyTorch and `huggingface-hub` for checkpoint
downloads. Use an environment created by the one-line installer, or install the
GUI and ML extras:

```bash
pip install "annolid[gui,ml]"
```

For script-only use without the desktop GUI, `pip install "annolid[ml]"` is
enough.

If you are working from a local checkout:

```bash
pip install -e ".[gui,ml]"
```

For EXR export, also install `OpenEXR` and `Imath` in the active environment.

## Checkpoints

Annolid stores Video Depth Anything checkpoints under:

```text
annolid/depth/checkpoints
```

The GUI auto-downloads the selected checkpoint on first use. If you want to
prefetch checkpoints before a GUI session, use the bundled downloader:

```bash
python -m annolid.depth.download_weights --model video_depth_anything_vitl
```

Useful downloader commands:

```bash
python -m annolid.depth.download_weights --list
python -m annolid.depth.download_weights --all
python -m annolid.depth.download_weights --model metric_video_depth_anything_vitb
```

Existing checkpoint files are not downloaded again. If you use an authenticated
Hugging Face endpoint, set `HF_HUB_TOKEN` before running the downloader.

## GUI Workflow

1. Open a video in Annolid.
2. Open **View -> Depth Settings...**.
3. Choose the encoder, resolution, frame limit, target FPS, metric-depth option,
   and output artifacts.
4. Open **View -> Video Depth Anything...**.
5. Watch progress in the status bar and review the live depth overlay on the
   canvas.
6. Inspect the generated `depth.ndjson` and any optional rendered outputs.

Streaming mode is enabled by default. It processes one frame at a time, keeps
memory use lower on long videos, and emits records incrementally.

## Python Workflow

For batch jobs, call the same runtime directly:

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

print(result["depth_ndjson"])
```

## Outputs

Every run writes outputs to the selected output directory. The primary artifact
is:

```text
depth.ndjson
```

Each line is one frame record. The depth map is stored as a base64-encoded
`uint16` PNG under `otherData.depth_map`, with scale metadata for converting
quantized values back to floating-point depth values. Metric depth is in metric
units only when the metric-depth model is enabled.

Optional outputs include:

- `<video_stem>_vis.mp4` when `save_depth_video=True`
- `depth_frames/` when `save_depth_frames=True`
- `point_clouds/*.csv` when `save_point_clouds=True`
- `<video_stem>_depths.npz` when `save_npz=True`
- `<video_stem>_depths_exr/` when `save_exr=True`

Point-cloud CSVs contain `x`, `y`, `z`, and `intensity` columns. When region
labels are enabled and matching Annolid annotation files are available, they can
also include a `region` column.

## Decode a Depth Record

Use the scale metadata from each record to convert the stored PNG back to a
floating-point depth array:

```python
import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

record = json.loads(Path("outputs/mice_depth/depth.ndjson").read_text().splitlines()[0])
payload = record["otherData"]["depth_map"]
scale = payload["scale"]

image = Image.open(io.BytesIO(base64.b64decode(payload["image_data"])))
depth = (np.asarray(image, dtype=np.float32) / 65535.0) * (
    scale["max"] - scale["min"]
) + scale["min"]
```

## Settings Reference

| Setting | Effect |
| --- | --- |
| Encoder | `vits`, `vitb`, or `vitl`; larger encoders can preserve more detail but use more memory. |
| Max resolution | Downscales each frame so the longer side does not exceed this value before inference. |
| Max frames | Limits how many frames are processed; `-1` means unlimited. |
| Target FPS | Samples frames to a target FPS; `-1` keeps the source rate. |
| Metric depth model | Uses `metric_video_depth_anything_*` checkpoints and metric-depth scaling. |
| FP32 inference | Forces float32 inference; Annolid also forces FP32 on non-CUDA devices for stable values. |
| Grayscale overlay | Uses grayscale rendering for GUI preview and saved depth frames. |
| Save depth video | Writes `<video_stem>_vis.mp4`. |
| Save depth frames | Writes rendered PNGs under `depth_frames/`. |
| Save point clouds | Writes XYZ-intensity CSV files under `point_clouds/`. |
| Save NPZ / EXR | Writes array stacks for downstream processing; EXR requires `OpenEXR` and `Imath`. |

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Import error for `huggingface_hub` | Install the ML runtime with `pip install "annolid[ml]"` or `pip install -e ".[gui,ml]"`. |
| First run appears slow | Confirm the selected checkpoint is downloading and the active environment has network access. |
| CUDA runs out of memory | Use a smaller encoder, lower max resolution, limit frames, or run on CPU. |
| EXR export fails | Install `OpenEXR` and `Imath`, or disable EXR output. |
| No `depth.ndjson` appears | Check the status message and logs for video decoding, checkpoint, or output-directory errors. |

For related downstream 3D workflows, see [Simulation and FlyBody](simulation_flybody.md).
