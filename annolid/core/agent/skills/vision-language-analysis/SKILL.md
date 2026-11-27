---
name: vision-language-analysis
description: Ground behavior analysis in visible evidence from frames, frame grids, tracks, masks, and captions, and separate direct observation from interpretation.
metadata: '{"annolid":{"always":false,"tags":["vision","vlm","visual","frames","frame grid","caption","image","observation","visible evidence","multimodal"]}}'
---

# Vision-Language Analysis

Use this skill when behavior reasoning depends on image, frame-grid, or video evidence.

## Annolid Visual Evidence Tools

- `video_segment_frame_grid(path, start_sec?, end_sec?, sample_count?, columns?, annotate?)`: build chronological annotated contact sheets for interval inspection.
- `video_sample_frames(path, output_dir?, mode?, step?, target_fps?, indices?, max_frames?)`: extract individual frames for closer inspection.
- `gui_shared_image_path()`: locate the current GUI frame image shared with the bot.
- `caption(image_path, prompt?)`: generate grounded captions for a local frame or contact sheet when available.

## Rules

1. State what is directly visible before inferring higher-level behavior.
2. Prefer tracked motion, pose, distance, overlap, and temporal change over
   vague scene description.
3. Separate observations from inferences.
4. When evidence is weak, blurry, or occluded, report uncertainty explicitly.

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
