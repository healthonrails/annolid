---
name: provenance
description: Preserve evidence links, manifest paths, run identifiers, and artifact lineage so behavior-analysis outputs stay reviewable and replayable.
metadata: '{"annolid":{"always":false}}'
---

# Provenance

Use this skill whenever the bot reports behavior-analysis results.

## Required provenance

- run id or session context,
- manifest path when available,
- artifact paths for tracks, behaviors, metrics, or reports,
- validation warnings if present.

## Rules

1. Prefer immutable run artifacts over ad hoc summaries.
2. Report where the result came from.
3. If evidence is incomplete, say what is missing.
4. Preserve existing export and labeling contracts unless the task explicitly
   changes them.
