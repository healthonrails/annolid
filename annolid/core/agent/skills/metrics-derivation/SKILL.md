---
name: metrics-derivation
description: Derive quantitative behavior, kinematics, spatial zone, tracking, and ethogram metrics from typed artifacts, including bout counts, durations, rates, occupancy, proximity, and investigation time.
metadata: '{"annolid":{"always":false,"tags":["metrics","kinematics","speed","distance","freezing","zone","roi","occupancy","latency","time budget","ethogram","proximity","bout counts","duration","tracking stats"]}}'
---

# Metrics Derivation

Use this skill when a user asks for quantitative kinematics, spatial zone, tracking, or behavioral ethogram endpoints rather than raw frame labels.

## Canonical Annolid Tools & CLI Entrypoints

1. **Trajectory Kinematics & Freezing**:
   - Tool: `annolid_kinematics_stats` | CLI: `annolid-run kinematics <tracking_csv_or_dir> [--fps ...] [--px-per-cm ...] [--calibration ...]`
   - Endpoints: total distance (`px` and `cm`), mean/median/P95/max speed, acceleration, angular velocity, path straightness index, tortuosity, freezing bouts & `% freezing`, high-activity bouts, and pairwise inter-animal distance + velocity correlation.
2. **Spatial Zones & Place Preference**:
   - Tool: `annolid_zone_analysis` | CLI: `annolid-run zone-analysis <video_or_csv> --zone-file <zones.json> --profile <generic|open_field|elevated_plus_maze|three_chamber_social>`
   - Endpoints: occupancy `%`, entries, transitions, first-entry latency (`s`), mean/max visit dwell (`s`), and signed boundary distance (supporting polygon, rectangle, and 2-point circle ROIs).
3. **Behavior Time Budgets & Binned Summaries**:
   - Tool: `annolid_behavior_time_budget` | CLI: `annolid-run time-budget <events.csv> [--session-duration ...] [--bin-size ...] [--bout-profile aggression]`
   - Endpoints: occurrences, total/mean/median/min/max bout duration (`s`), interval-union `% of session` (never exceeding 100%), and per-bin trajectories (`*_binned.csv`).
4. **Tracking Summaries & Quality Audits**:
   - Tools: `annolid_tracking_stats` / `gui_analyze_tracking_stats` (tracking coverage, instance counts, trajectory overview), `annolid_track_qc` (gaps, speed spikes, ID swaps), `annolid_behavior_qc` (unpaired state events, exclusion violations).
5. **Anatomical Body-Part Ethograms**:
   - CLI: `annolid-run body-part-ethogram <annotation_dir> [--fps ...] [--events-csv ...]`

## Method

1. Always verify spatial (`px_per_cm` / `--calibration`) and temporal (`fps`, `session_duration`) calibration before comparing across recordings.
2. State exact numerators, denominators, and censoring rules (e.g., animals that never enter a zone have right-censored latency rather than `0 s`).
3. Keep physical units (`cm`, `cm/s`, `s`, `%`) vs image-plane units (`px`, `px/frame`) explicit in every table.
4. Surface Track QC (`annolid_track_qc`) or Ethogram QC (`annolid_behavior_qc`) caveats whenever missing frames or unpaired state events exist.
