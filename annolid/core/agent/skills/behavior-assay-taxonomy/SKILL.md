---
name: behavior-assay-taxonomy
description: Infer the assay or paradigm from video context, task text, tracked entities, and experimental cues such as social interaction, open field, courtship, resident-intruder, and novel object recognition.
metadata: '{"annolid":{"always":false,"tags":["assay","paradigm","taxonomy","resident intruder","aggression","social interaction","open field","novel object recognition","elevated plus maze","courtship","behavior video"]}}'
---

# Behavior Assay Taxonomy

Use this skill when the user asks the bot to infer or confirm what kind of
behavior experiment a video represents.

## Goal

Map the available evidence into a coarse assay label that downstream planning
and segmentation can rely on.

## Core output

Return:

- the most likely assay type,
- confidence,
- a short rationale grounded in observable evidence,
- ambiguity notes when multiple assays remain plausible.

## Common assay families

- `aggression` or `resident_intruder`: chasing, attacks, slap-in-face, retreat,
  escalation into fighting.
- `social_interaction`: approach, sniffing, following, nose-to-nose distance,
  orientation between animals.
- `open_field`: center versus periphery occupancy, locomotion, rearing, freezing.
- `elevated_plus_maze`: open-arm vs closed-arm occupancy, head dips, entries.
- `novel_object_recognition`: object investigation, nose-to-object distance,
  dwell time near familiar versus novel objects.
- `courtship`: orientation, pursuit, mounting, song/dance/display sequences.

## Decision rules

1. Prefer explicit metadata or user instructions over guesswork.
2. If no metadata is available, infer from repeated observable motifs, not from
   speculation.
3. Distinguish social interaction from aggression by whether the interaction is
   primarily affiliative/exploratory or includes overt escalation and retreat.
4. If confidence is low, say so and keep the assay generic instead of forcing a
   wrong label.

## Integration

- Use this before feature planning (`behavior-feature-selection`).
- For delegated assay classification in multi-step runs, use
  `spawn_behavior_subagent(profile="behavior_assay_inference", ...)`.
- For full aggression/resident-intruder scoring runs, feed the confirmed assay
  into `gui_score_aggression_bouts(assay=..., ...)` or `annolid-run agent-behavior`.
- If the current run already provides a typed `TaskPlan`, do not contradict it
  unless the evidence is materially stronger.
