---
name: sam3-agent-video
description: Use SAM3 Agent windowed tracking for long videos with occlusions, repeated instances, or identity carry-over.
metadata: '{"annolid":{"always":false}}'
---

# SAM3 Agent Video Tracking

Use this skill when the user wants to track one or more instances across a long video and the problem is likely to need:

- windowed propagation
- overlap-aware state carry-over
- reseeding after occlusions or drift
- multiple instances with stable identities

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
  dry_run=false
)
```

## Prefer lighter tools instead when

- you only need video metadata: use `video_info`
- you only need frame sampling or inspection: use `video_sample_frames`
- you need generic model inference on the whole clip: use `video_run_model_inference`
- you are already inside the GUI and want interactive open+track behavior: use `gui_segment_track_video`

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

Use the summary path as the canonical artifact for downstream review or follow-up automation.

## Prompt template

When responding to the user, keep the request concrete:

1. State the target object(s).
2. State whether the video is long, crowded, or occluded.
3. State the preferred output folder if one was requested.
4. Call `sam3_agent_video_track` with the smallest window size that still preserves identity.
