---
name: behavior-vlm-labeling
description: Label video behavior segments from frame-grid images with a defined behavior list, model-ready JSON, resumable outputs, and no_behavior handling for sparse labels.
metadata: '{"annolid":{"always":false,"tags":["behavior","vlm","frame grid","behavior labeling","segments","ethogram","timestamps","defined list","every 1s","frames per grid","grooming","rearing","no_behavior"]}}'
---

# Behavior VLM Labeling

Use this skill when the user asks Annolid Bot to label behavior segments from a
video, frame grid, or fixed interval, especially with phrasing such as "label
behavior", "defined list", "every 1s", or "frames per grid".

## Preferred Annolid Tools

1. **Direct behavior segment labeling**:
   - `gui_label_behavior_segments(path?, behavior_labels?, use_defined_behavior_list?, segment_mode?, segment_frames?, segment_seconds?, sample_frames_per_segment?, frames_per_grid?, max_segments?, subject?, subject_term?, overwrite_existing?, llm_profile?, video_description?, instance_count?, experiment_context?, behavior_definitions?, focus_points?)`
   - Saves timeline labels, exports `<video_stem>_timestamps.csv`, and writes resumable segment logs to `<video_stem>_behavior_segment_labels.json`.
   - Use `segment_mode="uniform"` with `segment_seconds=1.0` for fixed 1-second windows, and set `max_segments=0` (or omit) to process to the end of the video.
   - Use `subject_term` for the visible organism named in the prompt (e.g., `fly`, `mouse`); keep `subject` for the exported timeline subject column.
2. **Manage the project behavior catalog first when needed**:
   - `gui_behavior_catalog(action="list"|"add"|"update"|"remove"|"reset", ...)`
   - Set `use_defined_behavior_list=true` when the user asks to label with their project-defined behaviors without naming explicit labels.
3. **Combined track + label pipeline**:
   - `gui_process_video_behaviors(path, text_prompt?, behavior_labels?, use_defined_behavior_list?, run_tracking?, run_behavior_labeling?, ...)`
4. **Standalone frame-grid rendering for manual inspection**:
   - `video_segment_frame_grid(path, start_sec?, end_sec?, sample_count?, columns?, annotate?)`

## Workflow

1. Preserve the user-provided behavior list exactly when the message includes
   explicit labels.
2. Use project-defined behavior labels only when the user asks for the defined
   list and does not provide explicit labels.
3. Segment to the end of the video unless the user gives an explicit positive
   segment cap.
4. Build chronological frame grids with the requested number of frames per grid
   (`sample_frames_per_segment` / `frames_per_grid`).
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
  overwrite (`overwrite_existing=true`).
