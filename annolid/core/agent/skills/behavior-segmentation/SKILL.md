---
name: behavior-segmentation
description: Segment behavior timelines from tracks, pose, contact, speed, and proximity signals into typed intervals with stable labels and rationales.
metadata: '{"annolid":{"always":false}}'
---

# Behavior Segmentation

Use this skill when the task is to convert low-level evidence into a timeline of
behavior intervals.

## Goal

Produce typed behavior segments that are deterministic, reviewable, and
compatible with downstream aggregation.

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

Normalize common aliases before aggregation.
