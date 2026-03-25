---
name: Video FFmpeg Processing
description: Process videos with FFmpeg — improve quality, auto-contrast, downsample, denoise, or crop using the video_ffmpeg_process tool.
requires:
  any_bins:
    - ffmpeg
---

# Video FFmpeg Processing

Use the `video_ffmpeg_process` tool to process video files with FFmpeg.
Use `video_info` first to inspect the source video (resolution, FPS, duration).

## Tool: `video_ffmpeg_process`

**Required:** `path` — the input video file.

**Presets** — use `action` to apply common operations:

| Action | What it does |
|---|---|
| `improve_quality` | Denoise + auto-contrast (good default for noisy lab videos) |
| `auto_contrast` | Brightness/contrast/saturation enhancement only |
| `downsample` | Halve resolution and FPS (reduce file size) |
| `denoise` | Temporal + spatial denoising (hqdn3d) |
| `crop` | Crop to a region (requires `crop` object) |
| `custom` | Only explicit parameters are applied |

**Fine-grained parameters** — override any preset default:

- `scale_factor`: spatial scale (0–1], e.g. 0.25 for quarter resolution
- `target_fps`: output frame rate
- `auto_contrast`: boolean
- `contrast_strength`: 0–3 (higher = more contrast/saturation)
- `denoise`: boolean
- `crop`: `{x, y, width, height}` in pixels

## Natural-Language Mapping

| User request | Recommended parameters |
|---|---|
| "improve the quality of this video" | `action=improve_quality` |
| "make the video brighter" | `action=auto_contrast`, `contrast_strength=1.5` |
| "enhance contrast" | `action=auto_contrast` |
| "reduce file size" / "compress this video" | `action=downsample` |
| "downsample to 15 fps" | `action=custom`, `target_fps=15` |
| "downsample spatially and temporally" | `action=downsample` |
| "half the resolution" | `action=custom`, `scale_factor=0.5` |
| "denoise this recording" | `action=denoise` |
| "remove noise from the video" | `action=denoise` |
| "crop to region x=100 y=50 w=640 h=480" | `action=crop`, `crop={x:100, y:50, width:640, height:480}` |
| "clean up and shrink" | `action=improve_quality`, `scale_factor=0.5` |

## Recommended Flow

1. Call `video_info` to inspect the source video.
2. Call `video_ffmpeg_process` with the appropriate action/params.
3. Report the `output_path` and key settings from the result to the user.
4. If processing fails, report the `error` and suggest the user check FFmpeg availability.

## Tips

- Combine presets with overrides: `action=downsample` + `scale_factor=0.25` gives quarter resolution at half FPS.
- For very noisy videos, use `action=improve_quality` with `contrast_strength=2.0`.
- Always confirm the output file was created before reporting success.
- Set `overwrite=true` only when the user explicitly says to replace the existing file.
