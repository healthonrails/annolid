---
name: behavior-segmentation
description: Segment behavior timelines from tracks, pose, contact, speed, and proximity signals into typed intervals with stable labels and rationales.
metadata: '{"annolid":{"always":false,"tags":["behavior segmentation","aggression","bouts","bout counts","sub-events","slap_in_face","run_away","fight_initiation","ethogram","timeline","intervals"]}}'
---

# Behavior Segmentation

Use this skill when the task is to convert low-level evidence into a timeline of
behavior intervals or aggression bouts.

## Goal

Produce typed behavior segments that are deterministic, reviewable, and
compatible with downstream aggregation.

## Annolid Entrypoints

- **GUI aggression/bout scoring**: `gui_score_aggression_bouts(path?, artifacts_ndjson?, assay?, model_policy?, bout_frame_gap?, ...)`
- **Delegated subagent**: `spawn_behavior_subagent(profile="behavior_segmentation", ...)`
- **Headless CLI**: `annolid-run agent-behavior ...`
- **Post-segmentation validation & summaries**: `annolid_behavior_qc` and `annolid_behavior_time_budget`

## Rules

1. Base segments on signal changes, not narrative intuition.
2. Keep labels canonical and stable across similar clips.
3. Include a concise rationale per segment.
4. Prefer additive evidence such as contact, approach speed, retreat, and zone
   entry over ambiguous visual impressions.
5. Preserve interval boundaries and frame indices when available.

## Aggression-specific guidance

For aggression scoring, focus on canonical sub-events such as:

- `slap_in_face`
- `run_away`
- `fight_initiation`

Normalize common aliases before aggregation and merge nearby sub-events using an explicit frame-gap rule (`bout_frame_gap`).
