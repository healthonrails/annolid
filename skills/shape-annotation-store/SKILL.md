---
name: shape-annotation-store
description: Read, relabel, or delete shapes stored in Annolid annotation-store NDJSON files (*_annotations.ndjson) or LabelMe JSON stubs referencing a store.
metadata: '{"annolid":{"always":false,"tags":["annotation store","ndjson","annotations.ndjson","json stub","relabel ndjson","delete ndjson","shapes","frame records"]}}'
---

# Shape Annotation Store

Use this skill when the user works with shapes persisted in Annolid's dual annotation storage:

- `*_annotations.ndjson` (append/compacted frame records when `persist_json=False`)
- LabelMe `.json` stubs containing an `annotation_store` reference and `frame` index

## Workflow

1. **Resolve Target File**:
   - Accept either a direct `*_annotations.ndjson` path or a JSON stub pointing to the store.
2. **Choose Frame Scope**:
   - Single frame: pass explicit `frame=<int>` (or let a JSON stub supply its frame).
   - Multi-frame batch edit: pass `apply_all_frames=true` only when the user explicitly wants all frames updated.
3. **Run File-Backed Shape Tools**:
   - Read/inspect: `gui_list_shapes_in_annotation(path, frame?, label_contains?, exact_label?, shape_type?, max_results?, include_points?)`
   - Relabel: `gui_relabel_shapes_in_annotation(path, old_label, new_label, frame?, shape_type?, apply_all_frames?, dry_run?)`
   - Delete: `gui_delete_shapes_in_annotation(path, frame?, label_contains?, exact_label?, shape_type?, max_delete?, apply_all_frames?, delete_all?, dry_run?)`
4. **Respect Guardrails**:
   - Keep all paths within workspace / allowed read roots.
   - Preview broad or cross-frame mutations with `dry_run=true` before writing.
   - Require explicit filters (`exact_label`, `label_contains`, `shape_type`) unless the user explicitly requests `delete_all=true`.
   - Never delete standalone user source images when editing or clearing annotation shapes.

## Output Reporting

Always report:

- resolved store path and source type (`annotation_store` or `annotation_store_stub`),
- affected frame(s) or `apply_all_frames` scope,
- exact changed/deleted record and shape counts, and whether `dry_run` was active.
