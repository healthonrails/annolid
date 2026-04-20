---
name: sandboxed-analysis
description: Generate minimal deterministic Python analyses over typed behavior artifacts, keeping code small, reproducible, and constrained to approved libraries.
metadata: '{"annolid":{"always":false}}'
---

# Sandboxed Analysis

Use this skill when the bot needs to derive metrics or summaries by generating
code.

## Rules

1. Generate the smallest analysis program that answers the question.
2. Keep inputs and outputs explicit.
3. Prefer deterministic aggregation over heuristic post-processing.
4. Do not depend on network access or non-approved packages.
5. Preserve replayability: code should be valid against stored artifacts.

## Typical outputs

- dwell time,
- bout counts,
- transition matrices,
- proximity summaries,
- assay-specific comparison metrics.
