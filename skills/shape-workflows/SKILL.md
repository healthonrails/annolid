---
name: shape-workflows
description: Use this skill when the user asks the bot to inspect, select, relabel, or delete shapes in Annolid canvas annotations, LabelMe JSON files, or annotation-store NDJSON files.
---

# Shape Workflows

## Use This Skill When
- The user asks to work with shapes (list, filter, select, relabel, delete).
- The request references LabelMe `.json` files or annotation-store `*_annotations.ndjson` files.
- The user is switching between live canvas work and file-backed shape updates.

## Core Rules
- Prefer live-canvas tools only when the request is clearly about currently opened shapes.
- Prefer file-backed tools when the user gives a file path, folder, JSON, or NDJSON context.
- Never claim you cannot work with shapes because of canvas access limits when file-backed tools are applicable.
- Keep all file operations inside workspace/allowed read roots.
- For destructive operations, require explicit filters or explicit full-delete intent.

## Tool Map
- Live canvas tools:
  - `gui_list_shapes`
  - `gui_select_shapes`
  - `gui_set_selected_shape_label`
  - `gui_delete_selected_shapes`
- File-backed tools:
  - `gui_list_shapes_in_annotation`
  - `gui_relabel_shapes_in_annotation`
  - `gui_delete_shapes_in_annotation`

## Decision Flow
1. Determine target type.
- If user references "selected", "current", or visible canvas state, use live canvas tools.
- If user provides a path or asks about stored annotations, use file-backed tools.

2. Determine file format.
- For LabelMe file payloads: use `.json` path.
- For annotation stores: use `*_annotations.ndjson` directly, or a JSON stub that contains `annotation_store` + `frame`.

3. Validate scope.
- Respect allowed directories and avoid paths outside allowed roots.
- If blocked by allowed-root constraints, explain that and ask for a path within allowed roots.

4. Perform operation.
- List first when uncertain.
- Apply relabel/delete only after clear filters (or explicit all-shapes confirmation).

## Safe Usage Patterns
- List shapes in file:
  - `gui_list_shapes_in_annotation(path="...", frame=..., label_contains="...", shape_type="point")`
- Relabel in file:
  - `gui_relabel_shapes_in_annotation(path="...", old_label="nose", new_label="snout", frame=...)`
- Delete in file (filtered):
  - `gui_delete_shapes_in_annotation(path="...", exact_label="tail", frame=...)`
- Delete all shapes in file (explicit):
  - `gui_delete_shapes_in_annotation(path="...", delete_all=true, frame=...)`

## Response Style
- Report what source was used: `canvas`, `labelme_json`, `annotation_store`, or `annotation_store_stub`.
- Return counts when possible: `total_shapes`, `returned_count`, `changed_shapes`, `deleted_shapes`.
- If no shapes matched, state filters used and suggest a quick list call with relaxed filters.

## Common Failure Handling
- Unsupported shape type:
  - Tell the user the allowed shape types and retry with a valid one.
- Missing frame for store context:
  - Ask for a frame index, or infer from stub if available.
- Outside allowed directories:
  - Explain the path policy and request a path under allowed roots.
