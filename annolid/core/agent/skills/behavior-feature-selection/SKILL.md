---
name: behavior-feature-selection
description: Choose assay-specific features and measurable objectives for behavior analysis, including distances, zones, speed, contact, orientation, and object interaction.
metadata: '{"annolid":{"always":false,"tags":["feature selection","behavior features","assay features","distance","zones","speed","contact","orientation","measurement plan","metrics planning"]}}'
---

# Behavior Feature Selection

Use this skill to convert assay context into a concrete measurement plan before computing kinematics, zone metrics, or behavior timelines.

## Goal

Pick the smallest feature set that supports the requested assay outcome.

## Feature patterns

- `aggression` / `resident_intruder`: inter-animal distance, approach velocity, retreat, contact,
  attack initiation, escape trajectories.
- `social_interaction`: body distance, nose-to-nose distance, following, orientation,
  proximity dwell time, social preference index.
- `open_field`: centroid trajectory, speed, freezing/immobility, center/periphery occupancy (`annolid_zone_analysis(..., assay_profile="open_field")`).
- `elevated_plus_maze`: open vs closed arm occupancy, arm entries, latency (`annolid_zone_analysis(..., assay_profile="elevated_plus_maze")`).
- `novel_object_recognition`: nose/object distance, investigation dwell time,
  transitions between familiar and novel objects.
- `courtship`: pursuit, display posture, contact sequences, proximity dynamics.

## Annolid Integration

- Delegate focused planning via `spawn_behavior_subagent(profile="behavior_feature_planning", ...)` when running the specialized behavior pipeline.
- Map selected spatial and kinematic features directly to `annolid_zone_analysis`, `annolid_kinematics_stats`, and `annolid_behavior_time_budget` rather than inventing ad hoc definitions.

## Output

Return:

- target features,
- derived metrics (with units and denominators),
- why each feature matters for the assay,
- any missing evidence (such as `px_per_cm`, `fps`, or a zone file) needed to compute them reliably.

## Constraint

Do not request generic features "just in case". Select features that are tied to
the assay and the requested report.
