# Identity Governor

The Identity Governor is a post-processing pass that repairs identity swaps in frame JSON annotations using user-defined evidence rules.

For common CUTIE ID switches, start with the manual seed-frame repair workflow:
correct the first switched frame, save it, and rerun CUTIE from that frame or
from a bounded segment. Saved manual frames supersede automatic prediction and
become reset points for the following tracking window. See
[Segment-Based Batch Tracking](tutorials/segment-based-batch-tracking.md#10-repair-id-switches-with-manual-seed-frames).

Use this Identity Governor page for advanced post-processing after tracking:
policy rules, report-driven temporal repair, and batch audits. These tools are
useful when manual reseeding is impractical, when you need a dry-run report, or
when assay-specific evidence can identify an animal after a period of ambiguity.

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
- `annolid.postprocessing.run_temporal_identity_repair`

GUI entry points:

- `Video Tools -> Identity Governor...`
- `Analysis -> Identity Governor...`

If these menu items are not visible, update Annolid to a build that includes the
Identity Governor dialog. Older builds may only have the Python post-processing
API, or may not include the temporal repair mode.

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

## CUTIE Home-Cage Temporal Repair

For crowded home-cage CUTIE runs, especially five or more visually similar mice,
the most practical first repair pass is often temporal continuity rather than a
zone policy. This mode treats the labels from a good reference frame as the
identity set, then walks forward frame by frame and reassigns swapped labels when
the multi-frame evidence is more consistent with another identity.

The temporal matcher uses a global assignment across all visible animals in each
frame. Its score combines centroid distance, constant-velocity prediction,
motion-compensated shape overlap, area consistency, and body-axis orientation
when polygon geometry is available. For LabelMe polygon output, "overlap" means
overlap of the tracked shape extent after motion compensation; raster-mask IoU
can be added later if mask bitmaps are stored in the JSON.

The repair does not wait for a separate CUTIE "ID switch" flag. It scans forward
from the reference frame and proposes a correction when the best temporal
assignment for a shape disagrees with the ID saved by CUTIE. The dry-run report
also flags frames that made the prediction suspicious before repair:

- duplicate IDs in one frame,
- missing expected IDs,
- unexpected IDs,
- instance-count mismatches,
- implausible same-ID jumps,
- CUTIE recovery/fallback notes saved in the shape metadata.

CUTIE's own missing-instance recovery still runs during tracking when enabled.
It can reseed from a previous complete frame, fill from recent masks, or pause
for manual correction when recovery fails. Temporal repair is the post-processing
layer after that: it audits recovered frames and corrects the label/ID if a
recovered mask came back attached to the wrong identity.

Use this when:

- CUTIE outputs one LabelMe JSON file per frame,
- each animal has a stable unique label in a good starting frame,
- labels switch after close overlap or crossing,
- you want a dry-run report before rewriting JSON files.

GUI path:

1. Open `Video Tools -> Identity Governor...` or `Analysis -> Identity Governor...`.
2. Set `Repair Mode` to `Temporal continuity`.
3. Choose the annotation folder produced by CUTIE.
4. Set `Count` to the expected number of visible mice, for example `5`.
5. Start with `Distance` around the largest plausible frame-to-frame or predicted
   movement in pixels, then run `Preview (Dry-Run)`.
6. Review `temporal_identity_repair_report.json` and key frames before
   `Apply Fixes`.

Python API:

```python
from annolid.postprocessing import run_temporal_identity_repair

result = run_temporal_identity_repair(
    annotation_dir="/path/to/cutie_results/session",
    start_frame=0,
    expected_instance_count=5,
    max_gap_frames=5,
    max_match_distance=80.0,
    apply_changes=False,  # dry-run first
)

print(result.report_path)
print(len(result.proposed_corrections))
```

After reviewing the report:

```python
run_temporal_identity_repair(
    annotation_dir="/path/to/cutie_results/session",
    expected_instance_count=5,
    max_gap_frames=5,
    max_match_distance=80.0,
    apply_changes=True,
)
```

`max_match_distance` is video-scale dependent. It gates both the previous
centroid and the constant-velocity prediction. If it is too small, fast-moving
animals will not be matched after crossings. If it is too large, nearby animals
can be over-corrected during dense contact. Dry-run on a short representative
span first.

`temporal_identity_repair_report.json` contains both `corrections` and
`quality_events`. The `corrections` list is what apply mode would rewrite. The
`quality_events` list is an audit trail for frames that likely need review even
if no safe correction was made.

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

## Reducing CUTIE ID Switches Before Tracking

For five or more mice in a 38 cm by 21.5 cm home cage, identity switches are
expected to become more frequent because animals overlap and have similar
appearance. These practices reduce the problem before post-processing:

1. Start from a frame where all animals are separated and every animal has a
   unique label.
2. Add extra manual seed frames just before and after known dense crossings,
   sleeping piles, or long occlusions.
3. Track shorter segments instead of one long uninterrupted run when crowding is
   heavy.
4. Enable `Automatic Pause on Error Detection` in Advanced Parameters so tracking
   stops when a seeded animal disappears.
5. Enable CUTIE missing-instance recovery only when short occlusions are expected
   and review the recovered frames before downstream analysis.
6. Avoid unnecessary downsampling; use enough image resolution for separate body
   masks.
7. Keep the expected animal count consistent with the visible animals in the
   segment.
8. Review the first few crowded crossings before running downstream behavior or
   zone analysis.

## Troubleshooting

- No corrections proposed:
  - verify IDs and labels are present in shapes,
  - verify metric names/aliases match policy,
  - inspect zone names from your saved `*_zones.json`.
- Temporal repair proposes no corrections:
  - verify the reference frame has one unique label per animal,
  - increase `max_match_distance` if animals move farther between frames,
  - set `expected_instance_count` to the number of visible animals.
- Too many corrections:
  - increase `min_streak_frames`,
  - tighten distance/area thresholds,
  - restrict `interesting_labels` or `interesting_track_ids`.
- Temporal repair over-corrects:
  - lower `max_match_distance`,
  - start from a more stable frame,
  - add manual seed frames and repair shorter spans.
- Unexpected backtracking:
  - tighten `ambiguity_conditions`,
  - reduce `max_backtrack_frames`.
