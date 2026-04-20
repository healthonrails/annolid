---
name: perception-routing
description: Route behavior-analysis perception to the lightest Annolid backend that can satisfy the task, preferring existing trackers and typed artifacts before heavier open-set grounding or propagation.
metadata: '{"annolid":{"always":false}}'
---

# Perception Routing

Use this skill when the bot must decide how to obtain the evidence required for
behavior analysis.

## Preferred order

1. Reuse existing typed artifacts and replayable sidecars if available.
2. Reuse Annolid tracking or segmentation outputs already attached to the video.
3. Use GUI/model workflows only when the required evidence is missing.
4. Escalate to heavier routing only when the task truly needs open-set grounding
   or propagation across long clips.

## Decision factors

- whether typed `TrackArtifact` rows already exist,
- number of animals or objects,
- occlusion risk,
- need for identity continuity,
- need for open-set object discovery,
- cost versus expected gain.

## Output

Return a short backend choice plus a concrete reason, not a generic model list.
