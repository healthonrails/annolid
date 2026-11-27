---
name: video-ffmpeg
description: Process videos with FFmpeg (improve quality, auto-contrast, downsample, denoise, crop) and inspect, trim, or sample frames using Annolid video tools.
metadata: '{"annolid":{"always":false,"tags":["video","ffmpeg","denoise","downsample","compress","contrast","brighter","crop","trim","clip","fps"]}}'
requires:
  any_bins:
    - ffmpeg
---

# Video FFmpeg Processing

Use `video_ffmpeg_process` for FFmpeg filtering, rescaling, denoising, contrast enhancement, and cropping. Always call `video_info` first to inspect resolution, FPS, frame count, and duration.

## Tool: `video_ffmpeg_process`

**Required:** `path` (input video file).

**Presets (`action`):**

| Action | Behavior |
|---|---|
| `improve_quality` | Denoise (`hqdn3d`) + auto-contrast (good default for noisy lab recordings) |
| `auto_contrast` | Brightness/contrast/saturation enhancement only |
| `downsample` | Halve spatial resolution (`scale_factor=0.5`) and halve FPS |
| `denoise` | Temporal + spatial denoising (`hqdn3d`) |
| `crop` | Crop to a pixel region (requires `crop`) |
| `custom` | Apply only explicit parameters |

**Fine-grained overrides** (take priority over preset defaults):

- `output_path`: target path (defaults to `<stem>_processed.mp4` beside input)
- `scale_factor`: `(0, 1]`, e.g. `0.25` for quarter resolution
- `target_fps`: output FPS (`> 0`)
- `auto_contrast`: boolean
- `contrast_strength`: `[0, 3]` (higher = stronger contrast/saturation)
- `denoise`: boolean
- `crop`: `{"x": int, "y": int, "width": int, "height": int}`
- `overwrite`: boolean (only set `true` when explicitly requested)

## Companion Non-FFmpeg Video Tools

Prefer lighter video tools when FFmpeg re-encoding is unnecessary:

- `video_info(path)`: Inspect FPS, total frames, width, height, duration.
- `video_segment(path, start_frame?, end_frame?, start_sec?, end_sec?, output_path?)`: Trim a single clip.
- `video_process_segments(path, segments, output_dir?)`: Export multiple clips.
- `video_sample_frames(path, mode?, step?, target_fps?, indices?, start_frame?, max_frames?)`: Extract sampled PNG frames.
- `video_segment_frame_grid(path, start_frame?, end_frame?, start_sec?, end_sec?, sample_count?, columns?)`: Build a chronological tiled frame grid for visual inspection or VLM analysis.

## Natural-Language Mapping

| User request | Recommended parameters |
|---|---|
| "improve video quality" | `action="improve_quality"` |
| "make video brighter" | `action="auto_contrast", contrast_strength=1.5` |
| "compress / reduce file size" | `action="downsample"` |
| "downsample to 15 fps" | `action="custom", target_fps=15` |
| "half the resolution" | `action="custom", scale_factor=0.5` |
| "denoise this recording" | `action="denoise"` |
| "crop x=100 y=50 w=640 h=480" | `action="crop", crop={"x": 100, "y": 50, "width": 640, "height": 480}` |

## Recommended Flow

1. Call `video_info(path=...)` first.
2. Call `video_ffmpeg_process` with the matching preset and overrides.
3. Confirm `success=true` and report `output_path`, applied FPS/scale/filters, or the exact `error` if processing fails.
