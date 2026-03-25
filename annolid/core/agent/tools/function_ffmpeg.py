"""FFmpeg video-processing tool for the Annolid agent.

Provides the ``video_ffmpeg_process`` function tool that wraps the
existing :func:`annolid.utils.videos.compress_and_rescale_video` utility
behind a user-friendly preset/parameter interface.
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Sequence

from annolid.utils.logger import logger

from .function_base import FunctionTool
from .function_video import (
    _is_within_root,
    _resolve_read_path,
)


# ---------------------------------------------------------------------------
# Preset defaults
# ---------------------------------------------------------------------------

_PRESET_DEFAULTS: dict[str, dict[str, Any]] = {
    "improve_quality": {
        "denoise": True,
        "auto_contrast": True,
        "contrast_strength": 1.0,
    },
    "auto_contrast": {
        "auto_contrast": True,
        "contrast_strength": 1.0,
    },
    "downsample": {
        "scale_factor": 0.5,
        # target_fps is computed at runtime as source_fps / 2
        "_halve_fps": True,
    },
    "denoise": {
        "denoise": True,
    },
    "crop": {
        # Requires explicit crop object; validated at runtime.
    },
    "custom": {},
}


def _apply_preset(action: str, params: dict[str, Any]) -> dict[str, Any]:
    """Merge preset defaults *under* explicit caller values."""
    preset = dict(_PRESET_DEFAULTS.get(action, {}))
    # Explicit params take priority over preset defaults.
    for key, value in params.items():
        if value is not None:
            preset[key] = value
    return preset


# ---------------------------------------------------------------------------
# Tool
# ---------------------------------------------------------------------------


class VideoFFmpegProcessTool(FunctionTool):
    """User-friendly FFmpeg video processing tool.

    Accepts high-level *actions* (``improve_quality``, ``auto_contrast``,
    ``downsample``, ``denoise``, ``crop``) alongside fine-grained parameters.
    Presets provide sensible defaults; explicit parameters always override.
    """

    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "video_ffmpeg_process"

    @property
    def description(self) -> str:
        return (
            "Process a video with FFmpeg: improve quality, auto-contrast, "
            "downsample (space/time), denoise, or crop.  Use the 'action' "
            "parameter for common presets, or set individual parameters for "
            "full control."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Input video file path.",
                },
                "output_path": {
                    "type": "string",
                    "description": (
                        "Output video file path.  "
                        "Defaults to <stem>_processed.mp4 next to the input."
                    ),
                },
                "action": {
                    "type": "string",
                    "enum": [
                        "improve_quality",
                        "auto_contrast",
                        "downsample",
                        "denoise",
                        "crop",
                        "custom",
                    ],
                    "description": (
                        "Preset action. 'improve_quality' enables denoise + "
                        "auto-contrast. 'downsample' halves resolution and FPS. "
                        "'custom' uses only explicit parameters."
                    ),
                },
                "scale_factor": {
                    "type": "number",
                    "description": "Spatial scale factor (0, 1]. 1.0 = original size.",
                    "exclusiveMinimum": 0,
                    "maximum": 1,
                },
                "target_fps": {
                    "type": "number",
                    "description": "Target output FPS.  Omit to keep original.",
                    "exclusiveMinimum": 0,
                },
                "auto_contrast": {
                    "type": "boolean",
                    "description": "Apply auto-contrast enhancement.",
                },
                "contrast_strength": {
                    "type": "number",
                    "description": "Contrast strength multiplier [0, 3].",
                    "minimum": 0,
                    "maximum": 3,
                },
                "denoise": {
                    "type": "boolean",
                    "description": "Apply hqdn3d temporal+spatial denoising.",
                },
                "crop": {
                    "type": "object",
                    "description": "Crop region in pixels.",
                    "properties": {
                        "x": {"type": "integer", "minimum": 0},
                        "y": {"type": "integer", "minimum": 0},
                        "width": {"type": "integer", "minimum": 1},
                        "height": {"type": "integer", "minimum": 1},
                    },
                    "required": ["x", "y", "width", "height"],
                },
                "overwrite": {
                    "type": "boolean",
                    "description": "Overwrite output if it already exists.",
                },
            },
            "required": ["path"],
        }

    async def execute(  # noqa: C901 – intentional single entry-point
        self,
        path: str,
        output_path: str | None = None,
        action: str = "custom",
        scale_factor: float | None = None,
        target_fps: float | None = None,
        auto_contrast: bool | None = None,
        contrast_strength: float | None = None,
        denoise: bool | None = None,
        crop: dict[str, int] | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            return await self._run(
                path=path,
                output_path=output_path,
                action=str(action or "custom").strip().lower(),
                scale_factor=scale_factor,
                target_fps=target_fps,
                auto_contrast=auto_contrast,
                contrast_strength=contrast_strength,
                denoise=denoise,
                crop=crop,
                overwrite=bool(overwrite),
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})
        except Exception as exc:
            logger.warning("video_ffmpeg_process failed: %s", exc, exc_info=True)
            return json.dumps({"error": str(exc), "path": path})

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _run(
        self,
        *,
        path: str,
        output_path: str | None,
        action: str,
        scale_factor: float | None,
        target_fps: float | None,
        auto_contrast: bool | None,
        contrast_strength: float | None,
        denoise: bool | None,
        crop: dict[str, int] | None,
        overwrite: bool,
    ) -> str:
        # --- resolve input ---
        video_path = _resolve_read_path(
            path,
            allowed_dir=self._allowed_dir,
            allowed_read_roots=self._allowed_read_roots,
        )
        if not video_path.exists():
            return json.dumps({"error": f"File not found: {path}", "path": path})
        if not video_path.is_file():
            return json.dumps({"error": f"Not a file: {path}", "path": path})

        # --- resolve output ---
        if output_path:
            out_path = Path(output_path).expanduser().resolve()
        else:
            stem = video_path.stem
            suffix = "_processed.mp4"
            out_path = video_path.parent / f"{stem}{suffix}"

        # Validate write permissions. We allow writing to the default workspace
        # (allowed_dir) or right next to the allowed input video.
        is_allowed = False
        allowed = self._allowed_dir
        if allowed is not None and _is_within_root(
            out_path, allowed.expanduser().resolve()
        ):
            is_allowed = True
        elif _is_within_root(out_path, video_path.parent):
            is_allowed = True

        if not is_allowed:
            return json.dumps(
                {
                    "error": f"Path {out_path} is outside allowed directory.",
                    "path": str(video_path),
                }
            )

        if out_path.exists() and not overwrite:
            return json.dumps(
                {
                    "error": "Output file exists; set overwrite=true to replace.",
                    "output_path": str(out_path),
                }
            )

        # --- apply preset + explicit overrides ---
        merged = _apply_preset(
            action,
            {
                "scale_factor": scale_factor,
                "target_fps": target_fps,
                "auto_contrast": auto_contrast,
                "contrast_strength": contrast_strength,
                "denoise": denoise,
            },
        )

        effective_scale = float(merged.get("scale_factor") or 1.0)
        effective_denoise = bool(merged.get("denoise", False))
        effective_auto_contrast = bool(merged.get("auto_contrast", False))
        effective_contrast_strength = float(merged.get("contrast_strength") or 1.0)

        # Compute effective FPS.
        from annolid.core.media.video import get_video_fps as _get_fps

        source_fps = _get_fps(str(video_path))
        if source_fps is None:
            return json.dumps(
                {
                    "error": "Failed to detect source FPS.",
                    "path": str(video_path),
                }
            )

        if merged.get("target_fps") is not None:
            effective_fps: float | int = float(merged["target_fps"])
        elif merged.get("_halve_fps"):
            effective_fps = max(1.0, float(source_fps) / 2.0)
        else:
            effective_fps = float(source_fps)

        # Crop params.
        crop_x = crop_y = crop_w = crop_h = None
        if crop and isinstance(crop, dict):
            crop_x = int(crop.get("x", 0))
            crop_y = int(crop.get("y", 0))
            crop_w = int(crop.get("width", 0))
            crop_h = int(crop.get("height", 0))
            if crop_w <= 0 or crop_h <= 0:
                return json.dumps(
                    {
                        "error": "crop width and height must be positive.",
                        "crop": crop,
                    }
                )
        elif action == "crop":
            return json.dumps(
                {
                    "error": "action='crop' requires a crop object with x, y, width, height.",
                }
            )

        # --- delegate to compress_and_rescale_video (single-file mode) ---
        # The utility processes all videos in a folder.  We create a
        # temporary directory containing a symlink to the input file so
        # that we can process exactly one file.
        try:
            with tempfile.TemporaryDirectory(prefix="annolid_ffmpeg_") as tmpdir:
                tmp_input = Path(tmpdir) / "input"
                tmp_output = Path(tmpdir) / "output"
                tmp_input.mkdir()
                tmp_output.mkdir()

                # Symlink the source video into the temp input dir.
                link_name = tmp_input / video_path.name
                try:
                    os.symlink(str(video_path), str(link_name))
                except OSError:
                    # Fallback: copy if symlinks are unsupported.
                    shutil.copy2(str(video_path), str(link_name))

                from annolid.utils.videos import compress_and_rescale_video

                command_log = compress_and_rescale_video(
                    input_folder=str(tmp_input),
                    output_folder=str(tmp_output),
                    scale_factor=effective_scale,
                    fps=effective_fps,
                    apply_denoise=effective_denoise,
                    auto_contrast=effective_auto_contrast,
                    auto_contrast_strength=effective_contrast_strength,
                    crop_x=crop_x,
                    crop_y=crop_y,
                    crop_width=crop_w,
                    crop_height=crop_h,
                )

                if not command_log:
                    return json.dumps(
                        {
                            "error": "FFmpeg processing failed for all encoder profiles.",
                            "path": str(video_path),
                            "action": action,
                        }
                    )

                # Find the produced output and move it to the final location.
                produced_files = list(tmp_output.iterdir())
                if not produced_files:
                    return json.dumps(
                        {
                            "error": "No output file produced.",
                            "path": str(video_path),
                            "action": action,
                        }
                    )

                produced = produced_files[0]
                out_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(produced), str(out_path))

                # Extract the ffmpeg command that was used.
                ffmpeg_cmd = next(iter(command_log.values()), "")

        except Exception as exc:
            return json.dumps(
                {
                    "error": f"Processing failed: {exc}",
                    "path": str(video_path),
                    "action": action,
                }
            )

        return json.dumps(
            {
                "success": True,
                "path": str(video_path),
                "output_path": str(out_path),
                "action": action,
                "scale_factor": effective_scale,
                "fps": effective_fps,
                "denoise": effective_denoise,
                "auto_contrast": effective_auto_contrast,
                "contrast_strength": effective_contrast_strength,
                "crop": crop,
                "ffmpeg_command": ffmpeg_cmd,
            }
        )
