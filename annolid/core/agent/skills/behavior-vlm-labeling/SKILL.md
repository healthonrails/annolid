---
name: behavior-vlm-labeling
description: Label video behavior segments from frame-grid images with a defined behavior list, model-ready JSON, resumable outputs, and no_behavior handling for sparse labels.
metadata: '{"annolid":{"always":false}}'
---

# Behavior VLM Labeling

Use this skill when the user asks Annolid Bot to label behavior segments from a
video, frame grid, or fixed interval, especially with phrasing such as "label
behavior", "defined list", "every 1s", or "frames per grid".

## Workflow

1. Preserve the user-provided behavior list exactly when the message includes
   explicit labels.
2. Use project-defined behavior labels only when the user asks for the defined
   list and does not provide explicit labels.
3. Segment to the end of the video unless the user gives an explicit positive
   segment cap.
4. Build chronological frame grids with the requested number of frames per grid.
5. Resume from existing valid segment-label JSON instead of repeating processed
   intervals.
6. Keep model calls bounded, sequential, and resumable so rate limits or empty
   model responses do not corrupt existing outputs.

## Model Prompt Contract

- The allowed labels are the behavior labels for this run plus `no_behavior`.
- Return strict JSON only.
- The `label` value must be exactly one allowed behavior label or `no_behavior`.
- Use `no_behavior` when none of the listed behaviors is clearly visible.
- Do not force sparse labels onto still, walking, resting, or unclear frames.
- Ground the description in visible frame-grid evidence only.
- Do not invent species-specific anatomy or actions that are not visible.
- Keep descriptions concise and observable.

## Output Contract

- Save only real behavior predictions to behavior timestamp outputs.
- Treat `no_behavior` as a skipped segment, not as a persisted behavior interval.
- Preserve frame indices, time intervals, model metadata, grid image paths, and
  description provenance for each saved prediction.
- Never overwrite or reinterpret existing predictions unless the user requested
  overwrite.
