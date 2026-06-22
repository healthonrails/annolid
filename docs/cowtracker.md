# CoWTracker Point Tracking

CoWTracker is an optional point-tracking backend for videos where you want
Annolid to propagate labeled points through a recording. It uses the same
reviewable Annolid point-tracking contract as CoTracker and TAPNext: manually
saved point prompts, optional polygon-derived grid points, and LabelMe-compatible
JSON outputs.

## Requirements

For a workstation that needs the model runtime plus the CoWTracker-specific
dependency, install Annolid with both tracking extras:

```bash
pip install "annolid[tracking,cowtracker]"
```

If you are working from a local checkout:

```bash
pip install -e ".[tracking,cowtracker]"
```

If Annolid is already installed in an environment that has the tracking runtime,
you can add only the CoWTracker dependency:

```bash
pip install "safetensors>=0.4.0"
```

The first run needs an internet connection so Annolid can fetch the CoWTracker
checkpoint from Hugging Face.

## GUI Workflow

1. Open a video in Annolid.
2. Go to a frame where the target landmarks are visible.
3. Add manual `point` labels for the landmarks you want to track.
4. Optionally add a `polygon` around a region if you want Annolid to sample
   multiple tracking points inside it.
5. Save the annotation for that frame.
6. Select **CoWTracker** from the model dropdown.
7. Start tracking from the current frame.
8. Review the predicted frame JSON files and correct drift before using the data
   for analysis.

For long videos, validate a short representative range first. CoWTracker uses a
windowed offline tracker, so first-run latency includes dependency import,
checkpoint download if needed, model initialization, and the first tracking
window.

## Output

CoWTracker writes LabelMe-compatible JSON files into the video result folder,
the same folder Annolid uses for other tracking outputs:

```text
<video_name>/<video_name>_000000001.json
<video_name>/<video_name>_000000002.json
...
```

Each predicted shape is saved as a `point` with the original label, tracked
coordinate, visibility flag, and backend description. These files remain
reviewable in the GUI and can be exported through the usual Annolid workflows.

## VGGT Runtime Notes

CoWTracker uses a minimal vendored VGGT runtime subset under:

```text
annolid/tracker/cowtracker/cowtracker/thirdparty/vggt
```

At runtime, Annolid resolves VGGT in this order:

1. use the vendored subset when it is present and complete;
2. fall back to an externally installed `vggt` package.

If neither runtime is available, CoWTracker raises a dependency error with
install hints. Maintainer-facing vendoring details, required file lists, and
packaging checks live in `annolid/tracker/cowtracker/README.md`.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| CoWTracker is missing from the model choices | Confirm the active environment includes the tracking runtime and CoWTracker dependency. |
| First run appears slow | Confirm the checkpoint is downloading from Hugging Face and that the environment has enough disk space. |
| Dependency error mentions `safetensors` | Install `pip install "annolid[tracking,cowtracker]"` or `pip install "safetensors>=0.4.0"` in the active environment. |
| Dependency error mentions VGGT | Confirm the vendored VGGT subset is present, or install an external `vggt` package compatible with the CoWTracker runtime. |
| No point prompts found | Save at least one manual point or polygon annotation before tracking. |
| Tracking results are poor | Use clearer seed points, start from a sharper frame, or correct and restart from the first drift frame. |

## When to Use a Different Tracker

Use CoWTracker when you need an alternative point tracker and want outputs as
reviewable tracked points.

Use TAPNext for ONNX-only point tracking, or use a segmentation tracker such as
Cutie or SAM3 when you need full masks or polygons.
