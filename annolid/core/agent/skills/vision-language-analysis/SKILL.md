---
name: vision-language-analysis
description: Ground behavior analysis in visible evidence from frames, tracks, masks, and captions, and separate direct observation from interpretation.
metadata: '{"annolid":{"always":false}}'
---

# Vision-Language Analysis

Use this skill when behavior reasoning depends on image or video evidence.

## Rules

1. State what is directly visible before inferring higher-level behavior.
2. Prefer tracked motion, pose, distance, overlap, and temporal change over
   vague scene description.
3. Separate observations from inferences.
4. When evidence is weak, report uncertainty explicitly.

## Good evidence types

- frame-local pose or contact,
- direction of travel,
- repeated approach or avoidance,
- object proximity,
- orientation between animals,
- escalation over time.

## Avoid

- attributing intent or emotion,
- over-reading a single frame,
- claiming contact, attack, or investigation without visible support.
