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

## Annolid Bot workflow for long videos

If you want Annolid Bot to do the tracking for you, use the bot-facing long-video tool:

- `sam3_agent_video_track`

Use this when you need the agent to refine each window before propagation. It is the best fit for:

- long recordings that drift with a single pass
- repeated appearances after occlusion
- multiple objects that need stable identities across windows

Recommended workflow:

1. Inspect the source video first if you do not know its frame count or FPS.
   - In Annolid Bot, call `video_info(path)` first.
2. Run a dry run to confirm the output folder and settings.
   - Set `dry_run=true`.
3. Run the full SAM3 agent-seeded tracking pass.
4. Review the returned JSON summary and the saved artifact path.

Example bot call:

```text
sam3_agent_video_track(
  video_path="/path/to/video.mp4",
  agent_prompt="mouse",
  window_size=5,
  stride=5,
  propagation_direction="forward",
  dry_run=true
)
```

When the settings look right, rerun with `dry_run=false`.

### Output artifacts

The tool returns a JSON summary and writes the same summary to disk.

Default output folder:

- `<video_stem>_sam3_agent/`

Default summary filename:

- `<video_stem>_sam3_agent_tracking.json`

The summary includes:

- `frames_processed`
- `masks_written`
- `summary_path`
- the resolved runtime config

If you specify `output_dir`, Annolid writes the summary there instead.

### Choosing the right tool

Use `sam3_agent_video_track` when:

- you want agent-assisted long-video tracking
- the object may disappear and reappear
- you want overlap-aware carry-over between windows

Use `gui_segment_track_video` when:

- you are working interactively inside the GUI
- you want to open a video and run the standard GUI-assisted segment/track flow

Use `video_run_model_inference` when:

- you only want to run a normal model predict command on a video
- you do not need SAM3 agent reseeding or window overlap carry-over

## GUI slash track

If you are working in the Annolid GUI chat box, the `/track` slash command is a
shorter front-end to the interactive video workflow. Typing `/track` with no
arguments opens a guided form; filling it out inserts a structured command that
routes to either `gui_segment_track_video` or `sam3_agent_video_track` depending
on the selected model.

Examples:

```text
/track
/track video=/path/to/video.mp4 prompt="mouse" model=Cutie
/track video=/path/to/video.mp4 prompt="mouse" model=Cutie to_frame=400
/track /path/to/video.mp4 mouse
```

Field guide:

- `video=` or `path=` points to the source video
- `prompt=` or `text_prompt=` provides the text prompt
- `model=` selects from the AI Models dropdown list in the main window
- `to_frame=` jumps tracking to a target frame when you want a bounded run

The guided form also shows which bot provider/model SAM3 will reuse for the
vision seed call, so you can confirm the active LLM before inserting the
command.

This slash command is the right choice when you want a quick GUI shortcut for
the standard open-video + prompt + track flow. When you choose `SAM3` from the
AI Models dropdown, the same slash command uses the SAM3 agent text-prompt
pipeline instead of the legacy GroundingDINO+SAM workflow, and it reuses the
current Annolid bot provider/model for the agent VLM call. If that agent path
cannot complete, Annolid falls back to the standard GUI video workflow so the
video still opens and tracking can continue with the selected GUI tracking
model.

## Prompt transaction model

Annolid enforces a **single prompt type per SAM3 request** at a common boundary in `Sam3SessionManager`.

Transaction rules:

1. One request contains exactly one prompt kind:
- `text` **or** `boxes` **or** `points`
2. Mixed inputs are split into an ordered transaction:
- `text -> boxes -> points`
3. Point prompts are always tracker prompts:
- point transactions use `obj_id` (required by SAM3.1 point refinement)
- point labels are normalized to binary foreground/background (`0/1`)
4. Box labels are normalized to binary (`0/1`) for SAM3 geometric prompt compatibility.

Why this exists:

- SAM3.1 point prompting rejects mixed text/box payloads in the same request.
- Explicit transaction sequencing removes mixed-prompt edge failures and makes behavior deterministic across normal and windowed runs.

Where implemented:

- `annolid/segmentation/SAM/sam3/session.py`
  - `_build_prompt_transaction_steps(...)`
  - `_execute_prompt_transaction(...)`
  - `add_prompt(...)`

## Multi-object prompt identity

For canvas-driven prompting, Annolid maps prompts to stable per-instance object ids.

Identity priority:

1. `group_id` (if present and valid)
2. existing label-to-id mapping from loaded annotations/session
3. deterministic new id allocation

Effects:

- point/polygon/box prompts can refine/add-back the correct object instead of collapsing to a single default object id.
- object identity remains stable across repeated prompt edits in the same run.

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

## Interactive session controls (GUI)

Annolid exposes notebook-like SAM3 session controls in the GUI:

- `Reset SAM3 Session`
- `Close SAM3 Session`
- `Remove SAM3 Object…` (by object id)

These actions are available from:

- **AI & Models** menu
- canvas right-click context menu

Safety behavior:

- controls are blocked while prediction is actively running (stop prediction first)
- remove-object runs at the current frame and refreshes loaded prediction shapes
- remove-object dialog prefills object id from selected shape when possible (`group_id` first, then label mapping)

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

### 5) Annolid Bot run succeeds but the summary file is missing

Symptoms:

- the tool returns `ok=true`, but you cannot find the saved JSON summary

Check:

- whether `output_dir` was set explicitly
- whether the selected workspace is writable
- whether `dry_run=true` was used

Remember:

- `dry_run=true` does not execute tracking and does not create the output folder
- the summary path reported in the JSON payload is the canonical artifact location

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
