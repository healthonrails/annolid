---
name: perception-routing
description: Route behavior-analysis perception to the lightest Annolid backend that can satisfy the task, preferring existing trackers and typed artifacts before heavier open-set grounding or propagation.
metadata: '{"annolid":{"always":false,"tags":["perception","routing","backend selection","tracker selection","cutie","sam3","yolo","tracking","artifacts","ndjson"]}}'
---

# Perception Routing

Use this skill when the bot must decide how to obtain the tracking, mask, or pose evidence required for behavior analysis.

## Preferred order (Lightest Sufficient Backend First)

1. **Reuse existing typed artifacts and sidecars**: inspect existing `*_annotations.ndjson`, LabelMe `.json` directories, `*_tracking.csv`, or `TrackArtifact` outputs before running any model.
2. **Reuse interactive/single-pass Annolid tracking**: when short or clean videos need masks/tracks in the GUI, use `gui_segment_track_video` (Cutie / GroundingDINO+SAM) or an existing trained YOLO/keypoint model via `video_run_model_inference`.
3. **Escalate to windowed SAM3 Agent tracking**: use `sam3_agent_video_track` (or `gui_correct_tracking_ndjson`) when videos are long, crowded, or occlusion-prone and need overlap-aware reseeding and identity carry-over.
4. **Use frame-grid VLM labeling**: use `gui_label_behavior_segments` when the user asks directly for semantic behavior interval labels without needing per-frame pixel masks.

## Decision factors

- whether typed `TrackArtifact` / NDJSON / tracking CSV outputs already exist,
- number of animals or objects,
- occlusion risk and clip duration,
- need for identity continuity vs coarse segment labels,
- need for open-set object discovery vs trained model weights (`video_list_inference_models`),
- compute cost and runtime versus expected gain.

## Output

Return a short backend choice plus a concrete reason grounded in the video/data state, not a generic model list.
