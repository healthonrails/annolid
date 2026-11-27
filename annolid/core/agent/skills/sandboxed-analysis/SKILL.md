---
name: sandboxed-analysis
description: Generate minimal deterministic Python analyses over typed behavior artifacts inside the AST-checked subprocess sandbox, keeping code small, reproducible, and constrained to approved libraries.
metadata: '{"annolid":{"always":false,"tags":["sandbox","python analysis","custom script","deterministic","tracking csv","dataframe","statistics","bootstrap","reproducible code"]}}'
---

# Sandboxed Analysis

Use this skill when custom metrics must be computed via deterministic Python scripts after checking whether built-in Annolid analytics tools already cover the request.

## Prefer Built-In Tools First

Before writing custom scripts, check whether one of Annolid's validated tools solves the task directly:

- `annolid_kinematics_stats` (distance, speed, acceleration, tortuosity, freezing bouts, pairwise distance)
- `annolid_zone_analysis` (polygon/rectangle/circle ROI occupancy, entries, latency, assay profiles)
- `annolid_behavior_time_budget` / `annolid_behavior_qc` / `annolid_behavior_irr` / `annolid_behavior_cohort`
- `annolid_tracking_stats` / `annolid_track_qc`
- `annolid_generate_report` (HTML5 + inline SVG + Markdown + JSON report bundle)

## Rules for Custom Sandboxed Code

1. Generate the smallest deterministic Python program using core dependencies (`numpy`, `pandas`, `shapely`, standard library `math`/`statistics`/`csv`/`json`); guard optional imports like `scipy` or `matplotlib`.
2. Keep generated code compatible with Annolid's AST-checked subprocess sandbox (`annolid/infrastructure/sandbox/execution.py`) — no shell escapes, no `eval`/`exec`, no network calls, and no writing outside the designated output directory.
3. Keep input paths, column names (`frame_number`, `instance_name`, `cx`, `cy`, `Recording time`, `Subject`, `Behavior`, `Event`), and output units explicit.
4. Never mutate raw tracking CSVs, NDJSON stores, or frame JSON annotations in place; write derived summaries to new files.
5. Do not depend on interactive GUI windows (`cv2.imshow`, `plt.show`).
6. Record explicit seed values (e.g., `rng = random.Random(42)`) for any bootstrap or permutation calculations.
