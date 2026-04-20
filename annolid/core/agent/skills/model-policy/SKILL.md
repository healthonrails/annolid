---
name: model-policy
description: Choose behavior-analysis models and runtimes with explicit tradeoffs around reproducibility, cost, privacy, speed, and artifact compatibility.
metadata: '{"annolid":{"always":false}}'
---

# Model Policy

Use this skill when selecting between hosted reasoning, local models, existing
Annolid backends, or heavier optional pipelines.

## Rules

1. Prefer existing Annolid perception backends for default execution.
2. Prefer typed, replayable outputs over one-off model calls.
3. Use hosted reasoning for planning and interpretation when it materially helps.
4. Keep privacy-sensitive or offline workflows compatible with local fallback.
5. Do not silently switch to a heavier stack without explaining why.

## Output

State:

- selected policy,
- why it fits the task,
- what evidence it is expected to produce,
- any runtime or reproducibility tradeoff.
