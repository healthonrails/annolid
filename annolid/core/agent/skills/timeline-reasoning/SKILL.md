---
name: timeline-reasoning
description: Reason over ordered behavior events, bout boundaries, escalation patterns, and frame gaps without losing temporal causality.
metadata: '{"annolid":{"always":false}}'
---

# Timeline Reasoning

Use this skill when the behavior meaning depends on event order rather than
isolated frames.

## Core tasks

- detect bout starts and ends,
- merge nearby sub-events using a stable frame-gap rule,
- distinguish escalation from repeated unrelated events,
- summarize temporal rationale without losing exact ordering.

## Rules

1. Time order matters more than single-frame salience.
2. Keep bout aggregation deterministic and parameterized by the configured frame
   gap or interval rule.
3. Do not merge events across long silent gaps unless the policy explicitly
   allows it.
4. Report boundary uncertainty instead of inventing continuity.
