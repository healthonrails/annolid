---
name: timeline-reasoning
description: Reason over ordered behavior events, bout boundaries, escalation patterns, and frame gaps without losing temporal causality.
metadata: '{"annolid":{"always":false,"tags":["timeline","bout","bouts","bout counts","sub-events","frame gap","merge","escalation","temporal","event order","onset","offset","aggression"]}}'
---

# Timeline Reasoning

Use this skill when the behavior meaning depends on event order, bout merging, or escalation sequences rather than isolated frames.

## Core tasks

- detect bout starts and ends,
- merge nearby sub-events using a stable frame-gap rule (`bout_frame_gap` / `--bout-frame-gap`),
- distinguish escalation from repeated unrelated events,
- summarize temporal rationale without losing exact ordering.

## Annolid Tools

- `gui_score_aggression_bouts(..., bout_frame_gap=...)`: merge sub-events into deterministic bout intervals.
- `annolid_behavior_qc`: verify every `START` state event has a matching `STOP` and no mutually exclusive overlaps exist.
- `annolid_behavior_time_budget`: compute deterministic bout counts, bout duration distributions, interval-union percentages, and binned timelines.

## Rules

1. Time order matters more than single-frame salience.
2. Keep bout aggregation deterministic and parameterized by the configured frame
   gap or interval rule.
3. Do not merge events across long silent gaps unless the policy explicitly
   allows it.
4. Report boundary uncertainty instead of inventing continuity.
