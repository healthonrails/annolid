---
name: shape-annotation-store
description: Use this skill when the user asks to work with shapes in annotation-store NDJSON files or JSON stubs that reference annotation stores.
---

# Shape Annotation Store

## Scope
This skill is for annotation-store workflows:
- `*_annotations.ndjson`
- LabelMe JSON stubs containing `annotation_store`

## Workflow
1. Resolve target file.
- Accept direct NDJSON path.
- Accept JSON stub and resolve to store path + frame.

2. Choose frame behavior.
- If frame is provided, use it.
- If frame is not provided, use stub frame or latest available frame record.

3. Run operation.
- Read: `gui_list_shapes_in_annotation`
- Relabel: `gui_relabel_shapes_in_annotation`
- Delete: `gui_delete_shapes_in_annotation`

4. Respect guardrails.
- Keep paths under allowed roots.
- Require filters unless user explicitly requests full delete (`delete_all=true`).

## Practical Defaults
- Start with list before mutating when request is ambiguous.
- Use `dry_run=true` first for risky relabel/delete requests.
- Report changed record count and shape count after mutation.
