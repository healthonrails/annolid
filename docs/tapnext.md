# TAPNext ONNX Point Tracking

TAPNext is a point-tracking backend for videos where you want Annolid to follow
body parts, landmarks, or a sampled set of points inside a manually labeled
region. It is useful when the target is better represented by keypoints than by
a full segmentation mask.

## What TAPNext Tracks

TAPNext starts from manually labeled prompts in Annolid:

- `point` shapes become individual tracked landmarks.
- `polygon` shapes can be converted into a grid of tracked points inside the
  polygon.
- zone shapes are ignored.

Each tracked point keeps the label from the manual annotation. For example,
labels such as `head`, `tail_base`, `left_wing`, or `fly_body` are written back
to the predicted frame JSON files.

## Requirements

TAPNext uses ONNX Runtime. Standard Annolid installs include the required
runtime dependency:

```bash
pip install annolid
```

If you are working from a local checkout, install the project environment first:

```bash
pip install -e .
```

The model file is large, about 938 MB. The first run needs an internet
connection and enough disk space for the cached ONNX model.

## Model Download and Cache

When you select **TAPNext (ONNX)** and start tracking, Annolid downloads the
official TAPNext ONNX model on first use, verifies its SHA256 checksum, and
caches it at:

```bash
~/.annolid/workspace/downloads/tapnext.onnx
```

Model URL:

```bash
https://github.com/healthonrails/annolid/releases/download/v1.6.6/tapnext.onnx
```

Expected checksum:

```bash
sha256:4fca0951802f0b745de254930c880938a74bf8b54b10786fc68d0ab4ba5c5300
```

If a file already exists at the default cache path but the checksum does not
match, Annolid replaces it with the verified release asset.

To verify the cached file manually:

```bash
python - <<'PY'
from pathlib import Path
from annolid.utils.model_assets import sha256sum
print(sha256sum(Path("~/.annolid/workspace/downloads/tapnext.onnx").expanduser()))
PY
```

## GUI Workflow

1. Open a video in Annolid.
2. Go to the frame where the target points are clearly visible.
3. Add manual `point` labels for landmarks you want to track.
4. Optionally add a `polygon` around a region if you want Annolid to sample
   multiple tracking points inside it.
5. Save the annotation for that frame.
6. Select **TAPNext (ONNX)** from the model dropdown.
7. Click the prediction/tracking button to start from the current frame.
8. Review the predicted frame JSON files and correct any drift or mislabeled
   points before using the data for analysis.

For long videos, start with a short range first. Confirm that the points follow
the right structures, then run the longer segment.

## Output

TAPNext writes LabelMe-compatible JSON files into the video result folder, the
same folder Annolid uses for other tracking outputs:

```text
<video_name>/<video_name>_000000001.json
<video_name>/<video_name>_000000002.json
...
```

Each predicted shape is a `point` with:

- the original label,
- the tracked coordinate,
- a visibility flag,
- a description identifying the TAPNext ONNX backend.

These JSON files can be reviewed in the GUI and exported through the usual
Annolid tracking/analysis tools.

## Choosing Good Prompts

Use prompts that are visually stable across frames:

- Place point prompts on distinctive body parts, joints, or landmarks.
- Avoid points on uniform texture, motion blur, shadows, or reflections.
- Prefer several named points over one ambiguous point when the animal rotates.
- For polygon prompts, use compact regions around the structure you care about.
- Save a clean manual frame before starting a long run.

If the tracker drifts, stop the run, correct the frame where the drift begins,
save that annotation, and continue tracking from the corrected frame.

## Performance Notes

TAPNext runs in fixed-length ONNX clips and saves frames incrementally. Progress
starts after the model is downloaded, loaded, and the first clip finishes.

On macOS, PyTorch MPS does not accelerate ONNX Runtime inference. Annolid uses
the ONNX Runtime providers available on the machine. CUDA is used when available;
otherwise TAPNext normally runs on CPU. CoreML can be tried explicitly:

```bash
ANNOLID_TAPNEXT_USE_COREML=1 annolid
```

Use CoreML only if it is faster and stable on your machine. Some ONNX models
with dynamic video/query shapes can load or run more reliably on CPU.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| First run appears slow | Confirm the model is downloading. The file is about 938 MB. |
| Download fails | Check internet access and write permission for `~/.annolid/workspace/downloads`. |
| Checksum mismatch | Delete `~/.annolid/workspace/downloads/tapnext.onnx` and run again. Annolid will redownload it. |
| No point prompts found | Save at least one manual point or polygon annotation before tracking. |
| Tracking results are poor | Use clearer seed points, start from a sharper frame, or correct and restart from the first drift frame. |
| No MPS provider appears | This is expected for ONNX Runtime. MPS is a PyTorch backend, not an ONNX Runtime execution provider. |

## When to Use a Different Tracker

Use TAPNext when your output should be tracked points or sampled points inside a
region.

Use a segmentation tracker such as Cutie or SAM3 when you need full object masks
or polygons. Use CoTracker/CoWTracker when you want an alternative point tracker
and already have a workflow tuned for those models.
