---
name: shape-workflows
description: Inspect, select, relabel, or delete shapes in Annolid live canvas annotations, LabelMe JSON files, or annotation-store NDJSON files safely.
metadata: '{"annolid":{"always":false,"tags":["shapes","shape","relabel","delete shapes","select shapes","list shapes","canvas","labelme","polygon","keypoints","bbox","annotations"]}}'
---

# Shape Workflows

Use this skill when the user asks to inspect, filter, select, relabel, or delete shapes in the live Annolid canvas, LabelMe `.json` files, or `*_annotations.ndjson` stores.

## Core Rules

- Use **live-canvas tools** when the request refers to currently opened, visible, or selected canvas shapes.
- Use **file-backed tools** when the user provides a `.json` or `*_annotations.ndjson` file path, folder, or frame index.
- Never claim you cannot edit shapes due to canvas limits when file-backed tools apply.
- Keep all paths inside workspace / allowed read roots.
- For relabel or delete mutations, list first or run `dry_run=true` when scope is ambiguous; require explicit filters (or explicit `delete_all=true`) for deletion.

## Tool Map

### Live Canvas Tools
- `gui_list_shapes(label_contains?, shape_type?, selected_only?, max_results?)`
- `gui_select_shapes(label_contains?, shape_type?, max_select?, clear_existing?)`
- `gui_set_selected_shape_label(new_label)`
- `gui_delete_selected_shapes()`

### File-Backed Annotation Tools (LabelMe `.json`, NDJSON, or JSON Stub)
- `gui_list_shapes_in_annotation(path, frame?, label_contains?, exact_label?, shape_type?, max_results?, include_points?)`
- `gui_relabel_shapes_in_annotation(path, old_label, new_label, frame?, shape_type?, apply_all_frames?, dry_run?)`
- `gui_delete_shapes_in_annotation(path, frame?, label_contains?, exact_label?, shape_type?, max_delete?, apply_all_frames?, delete_all?, dry_run?)`

## Safe Usage Patterns

- List shapes in a file or NDJSON frame:
  - `gui_list_shapes_in_annotation(path="...", frame=42, label_contains="mouse", shape_type="polygon")`
- Preview a cross-frame relabel before writing:
  - `gui_relabel_shapes_in_annotation(path="...", old_label="nose", new_label="snout", apply_all_frames=true, dry_run=true)`
- Delete filtered shapes on one frame:
  - `gui_delete_shapes_in_annotation(path="...", exact_label="tail", frame=42, dry_run=false)`
- Delete all shapes on a frame (only with explicit user intent):
  - `gui_delete_shapes_in_annotation(path="...", frame=42, delete_all=true)`

## Response Style

- Report the source type used: `canvas`, `labelme_json`, `annotation_store`, or `annotation_store_stub`.
- Report concrete counts (`total_shapes`, `returned_count`, `changed_shapes`, `deleted_shapes`, `dry_run`).
- If zero shapes matched, state the exact filters used and suggest a broader `gui_list_shapes*` check.

## Common Failure Handling

- **Unsupported `shape_type`**: report the allowed shape types and retry with a valid type (`polygon`, `rectangle`, `circle`, `line`, `point`, `linestrip`).
- **Missing `frame` for NDJSON store**: ask for a frame index, use the stub's frame, or set `apply_all_frames=true` only when requested.
- **Outside allowed roots**: explain the path restriction and ask for a path inside allowed directories.
