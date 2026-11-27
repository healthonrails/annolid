---
name: sam3-agent-video
description: Use SAM3 Agent windowed tracking for long videos with occlusions, repeated instances, or identity carry-over, plus post-hoc NDJSON tracking repair.
metadata: '{"annolid":{"always":false,"tags":["sam3","tracking","video tracking","long video","occlusion","identity drift","id switch","windowed propagation","reseed","ndjson","mice","masks"]}}'
---

# SAM3 Agent Video Tracking

Use this skill when the user wants to track one or more instances across a long video and the problem is likely to need:

- windowed propagation
- overlap-aware state carry-over
- reseeding after occlusions or drift
- multiple instances with stable identities
- post-hoc repair of missing frames or ID swaps in `*_annotations.ndjson`

## Prefer this path

Call `sam3_agent_video_track` when:

- the video is long enough that one-pass tracking is likely to drift
- the objects reappear after occlusion
- the user wants SAM3 Agent to refine the first frame of each window
- you need a stable bot-facing summary artifact after the run

Suggested tool call:

```text
sam3_agent_video_track(
  video_path="...",
  agent_prompt="mouse",
  window_size=5,
  stride=5,
  propagation_direction="forward",
  use_explicit_window_reseed=true,
  reject_implausible_masks=true,
  dry_run=false
)
```

Useful optional parameters:

- `dry_run=true`: validate config and window plan without running heavy inference.
- `llm_provider`, `llm_model`, `llm_profile`: override the VLM seed model (defaults to the active Annolid Bot model in GUI).
- `max_num_objects`, `agent_det_thresh`, `score_threshold_detection`, `new_det_thresh`: bound false positives in cluttered arenas.
- `offload_video_to_cpu=true`: reduce GPU VRAM footprint on long videos.
- Slash shortcut in GUI chat: `/track <prompt> model=SAM3` (routes through `gui_segment_track_video(..., model_name="SAM3", mode="track")` -> `sam3_agent_video_track`).

## Post-hoc tracking repair

When a run already produced `*_annotations.ndjson` with empty frames or occlusion-induced ID swaps, use `gui_correct_tracking_ndjson`:

```text
gui_correct_tracking_ndjson(
  ndjson_path="...",
  video_path="...",
  agent_prompt="mouse",
  run_sam3_agent=true,
  temporal_repair=true,
  start_frame=120,
  expected_instance_count=2,
  output_ndjson_path="..."
)
```

- By default, only frames with empty `shapes` are filled (`replace_only_empty_shapes=true`).
- Pass `output_ndjson_path` to write a corrected copy instead of mutating in place.

## Prefer lighter tools instead when

- you only need video metadata: use `video_info`
- you only need frame sampling or inspection: use `video_sample_frames`
- you need generic model inference on the whole clip: use `video_run_model_inference`
- you are already inside the GUI and want interactive single-pass Cutie/GroundingDINO+SAM tracking: use `gui_segment_track_video`

## Input checklist

Before running the tool, confirm:

- the video path is local and readable
- the prompt names the object clearly
- window size and stride are appropriate for video length
- output directory is writable if the user wants artifacts saved

## Output contract

The tool returns a JSON summary containing:

- `frames_processed`
- `masks_written`
- `summary_path`
- the resolved configuration used for the run

Use the summary path as the canonical artifact for downstream review, `annolid_track_qc`, or follow-up automation.

## Prompt template

When responding to the user, keep the request concrete:

1. State the target object(s).
2. State whether the video is long, crowded, or occluded.
3. State the preferred output folder if one was requested.
4. Call `sam3_agent_video_track` with the smallest window size that still preserves identity.
