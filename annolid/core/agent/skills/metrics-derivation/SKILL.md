---
name: metrics-derivation
description: Derive assay metrics from typed artifacts and segments, including bout counts, durations, rates, occupancy, proximity, and investigation time.
metadata: '{"annolid":{"always":false}}'
---

# Metrics Derivation

Use this skill when a user asks for numeric behavior summaries rather than just
labels.

## Method

1. Start from canonical artifacts and segments.
2. Define the numerator, denominator, and time window.
3. Keep units explicit.
4. Report validation or missing-data caveats when a metric is incomplete.

## Examples

- aggression bout counts from counted sub-events,
- total duration per behavior label,
- rate per minute,
- center occupancy,
- novel versus familiar object investigation time.
