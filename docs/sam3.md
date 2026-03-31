# SAM3 in Annolid

This guide documents the **SAM3/SAM3.1 integration in Annolid** for video segmentation and tracking.

## What Annolid uses

Annolid uses the bundled SAM3 runtime under:

- `annolid/segmentation/SAM/sam3/sam3`

and integrates it through:

- `annolid/segmentation/SAM/sam3/adapter.py`
- `annolid/segmentation/SAM/sam3/session.py`
- `annolid/gui/widgets/sam3_manager.py`

The runtime is configured for **inference only** (no training/eval workflow required).

## Installation

Install Annolid with SAM3 extras:

```bash
pip install ".[sam3]"
```

Minimum practical runtime dependencies include:

- `torch`
- `iopath`
- `ftfy`

If a dependency is missing, Annolid raises a startup error from the SAM3 manager/session path.

## Input modes

SAM3 runs in two main modes.

1. Seeded mode (annotation-guided):
- Uses existing per-frame prompts when available.
- Supported prompt shapes: boxes/rectangles, polygons/masks, and points.
- Polygon prompts are converted to robust box/mask seeds for SAM3 propagation.

2. Text-only mode:
- Uses a text prompt (for example, `mouse`) and runs SAM3.1 windowed propagation.
- This is the fallback when no usable geometric prompts (box/point/polygon) exist.

## Windowed inference behavior

For text-only runs, Annolid uses a windowed strategy to improve long-video stability.

Key properties:

- Reads frames from the source video timeline when input is an `.mp4`.
- Streams overlapping windows sequentially from the video instead of seeking back to each window start.
- Reuses the temporary window frame directory and only trims stale tail files between windows.
- Chooses larger default windows automatically for long CPU/CUDA runs when the user did not override `window_size` or `stride`.
- Uses overlapping windows by default (`stride = window_size - 1`) for boundary robustness.
- Carries visual prompt boxes from nearest neighbor mask frames across windows.
- Reacquires missed frames with visual+text prompts after the primary pass.
- Reacquires partially lost instances as well, and merges recovered masks back into the existing frame instead of replacing already tracked instances.
- Finalizes frame coverage by ensuring expected frame JSON outputs exist and are valid.

This is implemented in `Sam3SessionManager` in `session.py`.

## Device policy

Annolid applies defensive device handling in SAM3 paths.

- Chooses runtime device from user/default config.
- Falls back from unstable MPS paths to CPU when needed.
- Aligns index tensors and data tensors to the same device before `torch.index_select`/`torch.isin` operations.
- Uses strict JSON-safe serialization for frame outputs to avoid malformed files.

## Output files

Given video `.../mouse.mp4`, outputs are written under:

- `.../mouse/`

Primary artifacts:

- Per-frame LabelMe JSON: `000000000.json`, `000000001.json`, ...
- Annotation store: `mouse_annotations.ndjson`
- CSV exports (post-processing):
  - `mouse_tracking.csv`
  - `mouse_tracked.csv`

Notes:

- Frames with no masks are still materialized as valid JSON with an empty `shapes` list.
- CSV files contain object rows. A frame with zero detections will not necessarily have a row in `*_tracked.csv`.

## Runtime knobs

SAM3-related runtime settings are read from Annolid config/GUI state (advanced parameters).

Common knobs:

- `checkpoint_path`
- `max_frame_num_to_track`
- `device`
- `score_threshold_detection`
- `new_det_thresh`
- `max_num_objects`
- `multiplex_count`
- `compile_model`
- `offload_video_to_cpu`
- `sliding_window_size`
- `sliding_window_stride`
- `use_sliding_window_for_text_prompt`

If `sliding_window_size` and `sliding_window_stride` are not set explicitly, Annolid now derives them from runtime context:

- short CPU runs: smaller windows
- long CPU runs: moderate windows to reduce session churn
- CUDA runs: larger windows with moderate overlap
- explicit user values still take priority and are normalized to keep at least 1-frame overlap

## Troubleshooting

### 1) Missing dependency error

Symptoms:

- startup failure mentioning `iopath`, `ftfy`, or SAM3 extras.

Fix:

```bash
pip install ".[sam3]"
```

### 2) Device mismatch errors (`cuda` vs `cpu`)

Symptoms:

- `RuntimeError: Expected all tensors to be on the same device ... index_select`

Status:

- Patched in SAM3 postprocess paths to align indices/tensors by device.

If still seen:

- collect traceback + device setting + model config
- verify same Annolid revision is running in target environment

### 3) Malformed per-frame JSON (`Expecting value ...`)

Symptoms:

- frame JSON fails to parse in GUI or tracking reports

Status:

- Frame writes now use strict JSON-safe serialization and atomic replace.
- Coverage finalization repairs missing/corrupt frame JSON files.

### 4) Window-boundary frame drop

Symptoms:

- sparse or unstable detections around boundaries (for example, near frames 15/16 with window size 15)

Mitigations in current integration:

- forced overlap by default (`stride < window_size`)
- nearest-neighbor carry prompts
- post-pass reacquisition on missed frames

If results are still sparse:

- reduce `sliding_window_size`
- keep overlap (do not set `stride >= window_size`)
- lower detection thresholds conservatively

## Debug checklist

When debugging a run, capture:

1. Model + runtime options (window size, stride, thresholds, device)
2. Session logs from `sam3_manager` and `session`
3. Whether per-frame JSONs are valid JSON for expected frame range
4. `max_frame` from GUI prediction summary vs video frame count
5. Boundary frame behavior (`N-1`, `N`, `N+1` around each window split)

## Related docs

- [Installation](installation.md)
- [One-Line Installer](one_line_install_choices.md)
- [Workflows](workflows.md)
- [SAM 3D](sam3d.md)
