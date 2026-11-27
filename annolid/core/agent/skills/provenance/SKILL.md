---
name: provenance
description: Preserve evidence links, manifest paths, run identifiers, and artifact lineage so behavior-analysis outputs stay reviewable and replayable.
metadata: '{"annolid":{"always":false,"tags":["provenance","manifest","lineage","run id","reproducibility","replayable","audit trail","artifact paths","scientific report"]}}'
---

# Provenance

Use this skill whenever the bot reports behavior-analysis, training, evaluation, or simulation results.

## Required provenance

- `run_id` or session context,
- immutable manifest path when available (e.g., `<results-dir>/analysis_runs/<run-id>/manifest.json`),
- exact artifact paths for tracks (`*_annotations.ndjson`, `*_tracking.csv`), behaviors (`*_timestamps.csv`, `*_behavior_segment_labels.json`), checkpoints, metrics, and reports (`analysis_report.html`/`.md`/`.json`),
- model/policy/calibration metadata (`model_policy`, `fps`, `px_per_cm`, seed, or `allow_fallback` status),
- validation or QC warnings (`annolid_track_qc`, `annolid_behavior_qc`) if present.

## Rules

1. Prefer immutable run artifacts over ad hoc summaries.
2. Always report where every number and conclusion came from.
3. If evidence, calibration, or provenance is incomplete, state explicitly what is missing.
4. Preserve existing export, schema, and labeling contracts unless the task explicitly changes them.
