from __future__ import annotations

import argparse
import asyncio
import json
import contextlib
import shlex
import sys
from pathlib import Path
from typing import Any, List, Sequence

import cv2

from annolid.core.media.video import CV2Video
from annolid.core.agent.tools.sampling import FPSampler, UniformSampler

from .function_base import FunctionTool


def _normalize_allowed_read_roots(
    allowed_dir: Path | None, allowed_read_roots: Sequence[str | Path] | None
) -> tuple[Path, ...]:
    roots: list[Path] = []
    if allowed_dir is not None:
        roots.append(Path(allowed_dir).expanduser().resolve())
    if allowed_read_roots:
        for raw in allowed_read_roots:
            text = str(raw).strip()
            if not text:
                continue
            with contextlib.suppress(Exception):
                candidate = Path(text).expanduser().resolve()
                if candidate not in roots:
                    roots.append(candidate)
    return tuple(roots)


def _is_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _resolve_read_path(
    path: str,
    *,
    allowed_dir: Path | None = None,
    allowed_read_roots: Sequence[str | Path] | None = None,
) -> Path:
    resolved = Path(path).expanduser().resolve()
    roots = _normalize_allowed_read_roots(allowed_dir, allowed_read_roots)
    if roots and not any(_is_within_root(resolved, root) for root in roots):
        allowed = ", ".join(str(root) for root in roots)
        raise PermissionError(f"Path {path} is outside allowed read roots: [{allowed}]")
    return resolved


def _resolve_write_path(path: str, *, allowed_dir: Path | None = None) -> Path:
    resolved = Path(path).expanduser().resolve()
    if allowed_dir is not None and not _is_within_root(
        resolved, Path(allowed_dir).expanduser().resolve()
    ):
        raise PermissionError(f"Path {path} is outside allowed directory {allowed_dir}")
    return resolved


def _clamp_frame_index(index: int, total_frames: int) -> int:
    if total_frames <= 0:
        return 0
    return max(0, min(int(index), total_frames - 1))


def _resolve_segment_bounds(
    *,
    total_frames: int,
    fps: float,
    start_frame: int | None,
    end_frame: int | None,
    start_sec: float | None,
    end_sec: float | None,
) -> tuple[int, int]:
    max_index = max(0, total_frames - 1)
    if start_frame is not None or end_frame is not None:
        start_idx = int(start_frame or 0)
        end_idx = int(end_frame if end_frame is not None else max_index)
    elif start_sec is not None or end_sec is not None:
        start_idx = int(round(float(start_sec or 0.0) * float(fps)))
        end_idx = int(
            round(float(end_sec if end_sec is not None else (max_index / fps)) * fps)
        )
    else:
        start_idx, end_idx = 0, max_index

    start_idx = _clamp_frame_index(start_idx, max(total_frames, 1))
    end_idx = _clamp_frame_index(end_idx, max(total_frames, 1))
    return start_idx, end_idx


def _extract_segment_to_file(
    *,
    video_path: Path,
    out_path: Path,
    start_frame: int | None,
    end_frame: int | None,
    start_sec: float | None,
    end_sec: float | None,
    overwrite: bool,
) -> dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return {
            "error": "Output file exists; set overwrite=true to replace.",
            "output_path": str(out_path),
        }

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"error": f"Unable to open video: {video_path}"}
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if width <= 0 or height <= 0:
            return {
                "error": "Video has invalid frame dimensions.",
                "path": str(video_path),
            }

        start_idx, end_idx = _resolve_segment_bounds(
            total_frames=total_frames,
            fps=fps,
            start_frame=start_frame,
            end_frame=end_frame,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        if end_idx < start_idx:
            return {
                "error": "Invalid segment range: end before start.",
                "start_frame": start_idx,
                "end_frame": end_idx,
            }

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            return {"error": f"Unable to create output video: {out_path}"}
        written = 0
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_idx))
            for _ in range(start_idx, end_idx + 1):
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                writer.write(frame)
                written += 1
        finally:
            writer.release()
    finally:
        cap.release()

    return {
        "path": str(video_path),
        "output_path": str(out_path),
        "start_frame": int(start_idx),
        "end_frame": int(end_idx),
        "frames_written": int(written),
    }


class VideoInfoTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "video_info"

    @property
    def description(self) -> str:
        return "Read whole-video metadata (fps, frame count, resolution, duration)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        }

    async def execute(self, path: str, **kwargs: Any) -> str:
        del kwargs
        try:
            video_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            if not video_path.exists():
                return json.dumps({"error": f"File not found: {path}", "path": path})
            if not video_path.is_file():
                return json.dumps({"error": f"Not a file: {path}", "path": path})

            video = CV2Video(video_path)
            try:
                total_frames = int(video.total_frames())
                fps = float(video.get_fps() or 0.0)
                first = video.get_first_frame()
                height = int(first.shape[0])
                width = int(first.shape[1])
                duration_sec = (
                    float(total_frames) / float(fps)
                    if total_frames > 0 and fps > 0
                    else None
                )
            finally:
                video.release()

            return json.dumps(
                {
                    "path": str(video_path),
                    "total_frames": total_frames,
                    "fps": fps,
                    "width": width,
                    "height": height,
                    "duration_sec": duration_sec,
                }
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})
        except Exception as exc:
            return json.dumps({"error": str(exc), "path": path})


class VideoSampleFramesTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "video_sample_frames"

    @property
    def description(self) -> str:
        return (
            "Sample frames from a video for streaming analysis or sparse review "
            "and save sampled frames to image files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "output_dir": {"type": "string"},
                "mode": {
                    "type": "string",
                    "enum": ["stream", "uniform", "fps", "indices"],
                },
                "step": {"type": "integer", "minimum": 1},
                "target_fps": {"type": "number", "minimum": 0.1},
                "indices": {"type": "array", "items": {"type": "integer"}},
                "start_frame": {"type": "integer", "minimum": 0},
                "max_frames": {"type": "integer", "minimum": 1},
                "overwrite": {"type": "boolean"},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        output_dir: str | None = None,
        mode: str = "stream",
        step: int = 1,
        target_fps: float | None = None,
        indices: list[int] | None = None,
        start_frame: int = 0,
        max_frames: int = 32,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            video_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            if not video_path.exists():
                return json.dumps({"error": f"File not found: {path}", "path": path})
            if not video_path.is_file():
                return json.dumps({"error": f"Not a file: {path}", "path": path})

            if output_dir:
                out_dir = _resolve_write_path(output_dir, allowed_dir=self._allowed_dir)
            else:
                if self._allowed_dir is not None:
                    out_dir = _resolve_write_path(
                        str(Path(self._allowed_dir) / f"{video_path.stem}_frames"),
                        allowed_dir=self._allowed_dir,
                    )
                else:
                    out_dir = video_path.parent / f"{video_path.stem}_frames"
            out_dir.mkdir(parents=True, exist_ok=True)

            video = CV2Video(video_path)
            try:
                total_frames = int(video.total_frames())
                fps = float(video.get_fps() or 0.0)
                frame_indices = self._select_indices(
                    total_frames=total_frames,
                    mode=str(mode or "stream"),
                    step=int(step),
                    target_fps=target_fps,
                    indices=indices or [],
                    start_frame=int(start_frame),
                    max_frames=int(max_frames),
                    source_fps=fps,
                )
                outputs: List[dict[str, Any]] = []
                for frame_index in frame_indices:
                    image = video.load_frame(int(frame_index))
                    timestamp_sec = video.last_timestamp_sec()
                    out_path = out_dir / f"{video_path.stem}_{int(frame_index):09d}.png"
                    if out_path.exists() and not overwrite:
                        pass
                    else:
                        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        if not cv2.imwrite(str(out_path), image_bgr):
                            raise RuntimeError(
                                f"Failed to save frame image: {out_path}"
                            )
                    outputs.append(
                        {
                            "frame_index": int(frame_index),
                            "timestamp_sec": timestamp_sec,
                            "image_path": str(out_path),
                        }
                    )
            finally:
                video.release()

            return json.dumps(
                {
                    "path": str(video_path),
                    "mode": mode,
                    "output_dir": str(out_dir),
                    "frames": outputs,
                    "count": len(outputs),
                }
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})
        except Exception as exc:
            return json.dumps({"error": str(exc), "path": path})

    def _select_indices(
        self,
        *,
        total_frames: int,
        mode: str,
        step: int,
        target_fps: float | None,
        indices: list[int],
        start_frame: int,
        max_frames: int,
        source_fps: float,
    ) -> list[int]:
        if total_frames <= 0:
            return []

        start = _clamp_frame_index(start_frame, total_frames)
        n = max(1, int(max_frames))
        uniform_step = max(1, int(step))
        normalized_mode = str(mode or "stream").strip().lower()

        if normalized_mode == "indices":
            seen = set()
            unique = []
            for raw in indices:
                value = _clamp_frame_index(int(raw), total_frames)
                if value in seen:
                    continue
                seen.add(value)
                unique.append(value)
            if not unique:
                return [start]
            return sorted(unique)[:n]

        if normalized_mode == "fps":
            if target_fps is None or float(target_fps) <= 0 or source_fps <= 0:
                picks = UniformSampler(step=1).sample_indices(total_frames)
            else:
                picks = FPSampler(target_fps=float(target_fps)).sample_indices(
                    total_frames, fps=float(source_fps)
                )
            return [idx for idx in picks if idx >= start][:n]

        if normalized_mode in {"stream", "uniform"}:
            picks = UniformSampler(step=uniform_step).sample_indices(total_frames)
            return [idx for idx in picks if idx >= start][:n]

        return list(range(start, total_frames, 1))[:n]


class VideoSegmentTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "video_segment"

    @property
    def description(self) -> str:
        return "Export a video segment by frame or time range into a new video file."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "output_path": {"type": "string"},
                "start_frame": {"type": "integer", "minimum": 0},
                "end_frame": {"type": "integer", "minimum": 0},
                "start_sec": {"type": "number", "minimum": 0},
                "end_sec": {"type": "number", "minimum": 0},
                "overwrite": {"type": "boolean"},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        output_path: str | None = None,
        start_frame: int | None = None,
        end_frame: int | None = None,
        start_sec: float | None = None,
        end_sec: float | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            video_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            if not video_path.exists():
                return json.dumps({"error": f"File not found: {path}", "path": path})
            if not video_path.is_file():
                return json.dumps({"error": f"Not a file: {path}", "path": path})

            if output_path:
                out_path = _resolve_write_path(
                    output_path, allowed_dir=self._allowed_dir
                )
            else:
                if self._allowed_dir is not None:
                    out_path = _resolve_write_path(
                        str(Path(self._allowed_dir) / f"{video_path.stem}_segment.avi"),
                        allowed_dir=self._allowed_dir,
                    )
                else:
                    out_path = video_path.parent / f"{video_path.stem}_segment.avi"
            result = _extract_segment_to_file(
                video_path=video_path,
                out_path=out_path,
                start_frame=start_frame,
                end_frame=end_frame,
                start_sec=start_sec,
                end_sec=end_sec,
                overwrite=bool(overwrite),
            )
            return json.dumps(result)
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})
        except Exception as exc:
            return json.dumps({"error": str(exc), "path": path})


class VideoProcessSegmentsTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "video_process_segments"

    @property
    def description(self) -> str:
        return "Process multiple video segments and export each as an output clip."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "segments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start_frame": {"type": "integer", "minimum": 0},
                            "end_frame": {"type": "integer", "minimum": 0},
                            "start_sec": {"type": "number", "minimum": 0},
                            "end_sec": {"type": "number", "minimum": 0},
                            "output_path": {"type": "string"},
                        },
                    },
                    "minItems": 1,
                },
                "output_dir": {"type": "string"},
                "overwrite": {"type": "boolean"},
            },
            "required": ["path", "segments"],
        }

    async def execute(
        self,
        path: str,
        segments: list[dict[str, Any]],
        output_dir: str | None = None,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            video_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            if not video_path.exists():
                return json.dumps({"error": f"File not found: {path}", "path": path})
            if not video_path.is_file():
                return json.dumps({"error": f"Not a file: {path}", "path": path})
            if not isinstance(segments, list) or not segments:
                return json.dumps({"error": "segments must be a non-empty list."})

            if output_dir:
                out_dir = _resolve_write_path(output_dir, allowed_dir=self._allowed_dir)
            else:
                if self._allowed_dir is not None:
                    out_dir = _resolve_write_path(
                        str(Path(self._allowed_dir) / f"{video_path.stem}_segments"),
                        allowed_dir=self._allowed_dir,
                    )
                else:
                    out_dir = video_path.parent / f"{video_path.stem}_segments"
            out_dir.mkdir(parents=True, exist_ok=True)

            results: list[dict[str, Any]] = []
            for idx, item in enumerate(segments):
                if not isinstance(item, dict):
                    results.append(
                        {"index": idx, "error": "segment item must be object"}
                    )
                    continue
                custom_output = item.get("output_path")
                if custom_output:
                    out_path = _resolve_write_path(
                        str(custom_output), allowed_dir=self._allowed_dir
                    )
                else:
                    out_path = out_dir / f"{video_path.stem}_segment_{idx:03d}.avi"
                    if self._allowed_dir is not None:
                        out_path = _resolve_write_path(
                            str(out_path), allowed_dir=self._allowed_dir
                        )
                result = _extract_segment_to_file(
                    video_path=video_path,
                    out_path=out_path,
                    start_frame=(
                        int(item.get("start_frame"))
                        if item.get("start_frame") is not None
                        else None
                    ),
                    end_frame=(
                        int(item.get("end_frame"))
                        if item.get("end_frame") is not None
                        else None
                    ),
                    start_sec=(
                        float(item.get("start_sec"))
                        if item.get("start_sec") is not None
                        else None
                    ),
                    end_sec=(
                        float(item.get("end_sec"))
                        if item.get("end_sec") is not None
                        else None
                    ),
                    overwrite=bool(overwrite),
                )
                result["index"] = idx
                results.append(result)

            return json.dumps(
                {
                    "path": str(video_path),
                    "output_dir": str(out_dir),
                    "segments_processed": len(results),
                    "results": results,
                }
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "path": path})
        except Exception as exc:
            return json.dumps({"error": str(exc), "path": path})


def _find_optional_flag(parser: argparse.ArgumentParser, dest: str) -> str | None:
    for action in getattr(parser, "_actions", []):
        if str(getattr(action, "dest", "")).strip() != str(dest):
            continue
        option_strings = list(getattr(action, "option_strings", []) or [])
        for opt in option_strings:
            if isinstance(opt, str) and opt.startswith("--"):
                return opt
    return None


def _has_flag(args: Sequence[str], flag: str) -> bool:
    target = str(flag or "").strip()
    if not target:
        return False
    prefix = f"{target}="
    for token in args:
        raw = str(token)
        if raw == target or raw.startswith(prefix):
            return True
    return False


def _infer_predict_flags(
    model: str,
) -> tuple[list[str], list[str], bool, str | None]:
    from annolid.engine.registry import get_model

    plugin = get_model(model)
    if not plugin.__class__.supports_predict():
        return [], [], False, f"Model {model!r} does not support inference."

    parser = argparse.ArgumentParser(
        prog=f"annolid-run predict {model}", add_help=False
    )
    plugin.add_predict_args(parser)

    input_dest_candidates = (
        "source",
        "video",
        "video_path",
        "input",
        "input_video",
        "path",
        "video_folder",
    )
    output_dest_candidates = (
        "output_dir",
        "project",
        "output",
        "out",
        "save_dir",
    )
    input_flags = [
        flag
        for flag in (
            _find_optional_flag(parser, dest) for dest in input_dest_candidates
        )
        if flag
    ]
    output_flags = [
        flag
        for flag in (
            _find_optional_flag(parser, dest) for dest in output_dest_candidates
        )
        if flag
    ]
    has_positional_path = any(
        (not getattr(action, "option_strings", []))
        and str(getattr(action, "dest", "")).strip() in input_dest_candidates
        for action in getattr(parser, "_actions", [])
    )
    help_text = parser.format_help()
    return input_flags, output_flags, has_positional_path, help_text


class VideoListInferenceModelsTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        del allowed_dir, allowed_read_roots

    @property
    def name(self) -> str:
        return "video_list_inference_models"

    @property
    def description(self) -> str:
        return (
            "List Annolid predict models and indicate whether each can be invoked "
            "for video inference with standard input/output flags."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "video_only": {"type": "boolean"},
            },
        }

    async def execute(self, video_only: bool = True, **kwargs: Any) -> str:
        del kwargs
        from annolid.engine.registry import list_models

        models = list_models(load_builtins=True)
        items: list[dict[str, Any]] = []
        for info in models:
            if not bool(getattr(info, "supports_predict", False)):
                continue
            input_flags, output_flags, has_positional_path, _ = _infer_predict_flags(
                str(info.name)
            )
            video_compatible = bool(input_flags or has_positional_path)
            if video_only and not video_compatible:
                continue
            items.append(
                {
                    "name": str(info.name),
                    "description": str(getattr(info, "description", "") or ""),
                    "video_compatible": video_compatible,
                    "input_flags": input_flags,
                    "output_flags": output_flags,
                }
            )
        return json.dumps(
            {
                "count": len(items),
                "video_only": bool(video_only),
                "models": items,
            }
        )


class VideoRunModelInferenceTool(FunctionTool):
    def __init__(
        self,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ):
        self._allowed_dir = allowed_dir
        self._allowed_read_roots = tuple(allowed_read_roots or ())

    @property
    def name(self) -> str:
        return "video_run_model_inference"

    @property
    def description(self) -> str:
        return (
            "Run annolid-run predict for a model on a video path. "
            "Supports automatic input/output flag inference plus optional extra args."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "video_path": {"type": "string"},
                "output_dir": {"type": "string"},
                "extra_args": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "input_flag": {"type": "string"},
                "output_flag": {"type": "string"},
                "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                "dry_run": {"type": "boolean"},
            },
            "required": ["model", "video_path"],
        }

    async def execute(
        self,
        model: str,
        video_path: str,
        output_dir: str | None = None,
        extra_args: list[str] | None = None,
        input_flag: str | None = None,
        output_flag: str | None = None,
        timeout_seconds: int = 3600,
        dry_run: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            resolved_video = _resolve_read_path(
                video_path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            if not resolved_video.exists():
                return json.dumps(
                    {"error": f"File not found: {video_path}", "video_path": video_path}
                )
            if not resolved_video.is_file():
                return json.dumps(
                    {"error": f"Not a file: {video_path}", "video_path": video_path}
                )

            resolved_output_dir: Path | None = None
            if output_dir:
                resolved_output_dir = _resolve_write_path(
                    output_dir, allowed_dir=self._allowed_dir
                )
                resolved_output_dir.mkdir(parents=True, exist_ok=True)

            (
                inferred_input_flags,
                inferred_output_flags,
                has_positional_path,
                help_text,
            ) = _infer_predict_flags(str(model))
            if help_text is None:
                help_text = ""
            if not inferred_input_flags and not has_positional_path and not input_flag:
                return json.dumps(
                    {
                        "error": (
                            f"Cannot infer a video input argument for model {model!r}. "
                            "Pass input_flag explicitly and/or include proper args in extra_args."
                        ),
                        "model": str(model),
                        "help": help_text,
                    }
                )

            rendered_args = [str(arg) for arg in list(extra_args or [])]
            rendered_args = [
                arg.replace("{video_path}", str(resolved_video)).replace(
                    "{output_dir}",
                    str(resolved_output_dir) if resolved_output_dir else "",
                )
                for arg in rendered_args
            ]

            chosen_input_flag = str(input_flag).strip() if input_flag else None
            if chosen_input_flag and not chosen_input_flag.startswith("-"):
                chosen_input_flag = f"--{chosen_input_flag.lstrip('-')}"
            if not chosen_input_flag and inferred_input_flags:
                chosen_input_flag = inferred_input_flags[0]

            if chosen_input_flag and not _has_flag(rendered_args, chosen_input_flag):
                rendered_args.extend([chosen_input_flag, str(resolved_video)])
            elif not chosen_input_flag and has_positional_path:
                rendered_args.append(str(resolved_video))

            chosen_output_flag = str(output_flag).strip() if output_flag else None
            if chosen_output_flag and not chosen_output_flag.startswith("-"):
                chosen_output_flag = f"--{chosen_output_flag.lstrip('-')}"
            if not chosen_output_flag and inferred_output_flags:
                chosen_output_flag = inferred_output_flags[0]

            if (
                resolved_output_dir is not None
                and chosen_output_flag is not None
                and not _has_flag(rendered_args, chosen_output_flag)
            ):
                rendered_args.extend([chosen_output_flag, str(resolved_output_dir)])

            cmd = [
                str(sys.executable),
                "-m",
                "annolid.engine.cli",
                "predict",
                str(model),
                *rendered_args,
            ]

            if dry_run:
                return json.dumps(
                    {
                        "ok": True,
                        "dry_run": True,
                        "model": str(model),
                        "video_path": str(resolved_video),
                        "output_dir": str(resolved_output_dir)
                        if resolved_output_dir is not None
                        else None,
                        "input_flag": chosen_input_flag,
                        "output_flag": chosen_output_flag,
                        "command": cmd,
                        "shell_command": " ".join(shlex.quote(part) for part in cmd),
                    }
                )

            cwd = (
                str(Path(self._allowed_dir).expanduser().resolve())
                if self._allowed_dir is not None
                else None
            )
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    proc.communicate(), timeout=float(timeout_seconds)
                )
            except asyncio.TimeoutError:
                with contextlib.suppress(Exception):
                    proc.kill()
                return json.dumps(
                    {
                        "ok": False,
                        "error": f"Inference timed out after {int(timeout_seconds)} seconds.",
                        "model": str(model),
                        "video_path": str(resolved_video),
                        "command": cmd,
                    }
                )

            stdout_text = (stdout_bytes or b"").decode("utf-8", errors="replace")
            stderr_text = (stderr_bytes or b"").decode("utf-8", errors="replace")
            return json.dumps(
                {
                    "ok": int(proc.returncode or 0) == 0,
                    "exit_code": int(proc.returncode or 0),
                    "model": str(model),
                    "video_path": str(resolved_video),
                    "output_dir": str(resolved_output_dir)
                    if resolved_output_dir is not None
                    else None,
                    "input_flag": chosen_input_flag,
                    "output_flag": chosen_output_flag,
                    "command": cmd,
                    "stdout": stdout_text,
                    "stderr": stderr_text,
                }
            )
        except PermissionError as exc:
            return json.dumps({"error": str(exc), "video_path": video_path})
        except Exception as exc:
            return json.dumps(
                {"error": str(exc), "model": str(model), "video_path": video_path}
            )
