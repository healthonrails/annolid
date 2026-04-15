# Identity Governor

The Identity Governor is a post-processing pass that repairs identity swaps in frame JSON annotations using user-defined evidence rules.

It is generic by design:

- no hardcoded rover/tether labels,
- configurable metrics and thresholds,
- configurable zones and interesting instances,
- conservative repair with audit output.

## What It Does

Given a folder of LabelMe-style frame JSON files, the governor:

1. reads per-shape observations (`label`, `instance_label`, IDs, geometry),
2. computes metrics per frame (distance, zone, area),
3. evaluates your rules to produce identity evidence,
4. backtracks to the last ambiguous frame when configured,
5. forward-propagates through short uncertainty gaps,
6. writes a report (always),
7. optionally rewrites JSON atomically (apply mode).

## Current Interface

Current entry point is Python API:

- `annolid.postprocessing.run_identity_governor`
- `annolid.postprocessing.IdentityGovernor`
- `annolid.postprocessing.GovernorPolicy`

GUI entry points:

- `Video Tools -> Identity Governor...`
- `Analysis -> Identity Governor...`

Built-in GUI policy snippets:

- Generic identity template
- 2-subject arena (zone + distance)
- 3-vole social assay
- Distance-only fallback

## Quick Start

### 1. Prepare Inputs

- Annotation directory: one JSON per frame (for example `session_000000123.json`)
- Optional zone file: `*_zones.json` created from Annolid zone tools

### 2. Define a Policy

Use a plain dictionary (or JSON) with:

- `rules`: evidence rules that assign a corrected label,
- `metric_aliases`: optional readable aliases,
- `ambiguity_conditions`: conditions for backtracking window,
- `interesting_labels` / `interesting_track_ids`: optional scope filter,
- `canonical_track_ids`: optional label-to-canonical-ID mapping.

### 3. Run Dry-Run First

```python
from annolid.postprocessing import run_identity_governor

policy = {
    "metric_aliases": {
        "in_left": "zone.inside.left_zone",
        "in_right": "zone.inside.right_zone",
        "nearest": "distance.nearest",
        "area_px": "area",
    },
    "rules": [
        {
            "name": "alpha_when_right_and_large",
            "assign_label": "alpha",
            "conditions": [
                {"metric": "in_right", "op": "eq", "value": True},
                {"metric": "area_px", "op": "gte", "value": 80.0},
            ],
            "min_streak_frames": 2,
            "priority": 10,
        },
        {
            "name": "beta_when_left_and_large",
            "assign_label": "beta",
            "conditions": [
                {"metric": "in_left", "op": "eq", "value": True},
                {"metric": "area_px", "op": "gte", "value": 80.0},
            ],
            "min_streak_frames": 2,
            "priority": 10,
        },
    ],
    "ambiguity_conditions": [
        {"metric": "nearest", "op": "lte", "value": 5.0}
    ],
    "max_backtrack_frames": 500,
    "max_forward_gap_frames": 1,
    "min_correction_span_frames": 1,
    "canonical_track_ids": {
        "alpha": "1",
        "beta": "2",
    },
}

result = run_identity_governor(
    annotation_dir="/path/to/session",
    policy=policy,
    zone_file="/path/to/session_zones.json",  # optional
    apply_changes=False,  # dry-run
)

print(result.report_path)
print(len(result.proposed_corrections))
```

### 4. Inspect Report

`identity_governor_report.json` includes:

- scanned files/observations,
- proposed correction spans,
- corrected label per span,
- trigger rule and evidence span,
- dry-run/apply mode and update counts.

### 5. Apply Changes

```python
result = run_identity_governor(
    annotation_dir="/path/to/session",
    policy=policy,
    zone_file="/path/to/session_zones.json",
    apply_changes=True,
)
print(result.updated_files, result.updated_shapes)
```

## Supported Metrics

You can reference these directly in rule conditions.

### Distance Metrics

- `distance.nearest`
- `distance.to_track.<track_id>`
- `distance.to_label.<instance_label>`

Example:

```json
{"metric": "distance.to_track.2", "op": "gte", "value": 50.0}
```

### Zone Metrics

From the loaded zone file:

- `zone.inside.<zone_name>` (boolean)
- `zone.distance.<zone_name>` (float pixels)
- `zone.inside_kind.<zone_kind>` (boolean; inside any zone of that kind)
- `zone.distance_kind.<zone_kind>` (float pixels; min distance to that kind)
- `zone.inside_role.<occupant_role>` (boolean; inside any zone with that role)
- `zone.distance_role.<occupant_role>` (float pixels; min distance to that role)
- `zone.inside.stim_chamber` / `zone.distance.stim_chamber` (aggregate stim chamber metric)
- `zone.inside.neutral_transit` / `zone.distance.neutral_transit` (aggregate neutral tube/transit metric)

Example:

```json
{"metric": "zone.inside.right_zone", "op": "eq", "value": true}
```

### Geometry Metrics

- `area` (shape area in pixels)
- `x`, `y` (shape centroid)

## Condition Operators

Supported operators:

- `eq`, `==`
- `ne`, `!=`
- `gt`, `>`
- `gte`, `>=`
- `lt`, `<`
- `lte`, `<=`
- `in`
- `not_in`

## Scope and Safety Controls

- `interesting_labels`: only evaluate listed labels
- `interesting_track_ids`: only evaluate listed track IDs
- `min_streak_frames`: require evidence persistence before correction
- `ambiguity_conditions`: define what counts as uncertain overlap
- `max_backtrack_frames`: bound reverse extension
- `max_forward_gap_frames`: allow short forward missing gaps
- `min_correction_span_frames`: minimum relabeled span length

## ID Repair Behavior

When `canonical_track_ids` is set, apply mode also normalizes ID fields for corrected shapes:

- top-level: `track_id`, `tracking_id`, `instance_id`, `group_id`
- flags: same keys when present

This keeps labels and numeric IDs aligned.

## Recommended Workflow

1. Start with strict rules and dry-run.
2. Review report spans against a few key frames.
3. Relax thresholds only when needed.
4. Apply changes.
5. Re-run downstream CSV/zone summaries.

## Troubleshooting

- No corrections proposed:
  - verify IDs and labels are present in shapes,
  - verify metric names/aliases match policy,
  - inspect zone names from your saved `*_zones.json`.
- Too many corrections:
  - increase `min_streak_frames`,
  - tighten distance/area thresholds,
  - restrict `interesting_labels` or `interesting_track_ids`.
- Unexpected backtracking:
  - tighten `ambiguity_conditions`,
  - reduce `max_backtrack_frames`.
