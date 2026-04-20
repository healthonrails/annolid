---
name: behavior-feature-selection
description: Choose assay-specific features and measurable objectives for behavior analysis, including distances, zones, speed, contact, orientation, and object interaction.
metadata: '{"annolid":{"always":false}}'
---

# Behavior Feature Selection

Use this skill to convert assay context into a concrete measurement plan.

## Goal

Pick the smallest feature set that supports the requested assay outcome.

## Feature patterns

- `aggression`: inter-animal distance, approach velocity, retreat, contact,
  attack initiation, escape trajectories.
- `social_interaction`: body distance, nose-to-nose, following, orientation,
  proximity dwell time.
- `open_field`: centroid, speed, immobility, center/periphery occupancy.
- `novel_object_recognition`: nose/object distance, investigation dwell time,
  transitions between objects.
- `courtship`: pursuit, display posture, contact sequences, proximity dynamics.

## Output

Return:

- target features,
- derived metrics,
- why each feature matters,
- any missing evidence needed to compute them reliably.

## Constraint

Do not request generic features "just in case". Select features that are tied to
the assay and the requested report.
