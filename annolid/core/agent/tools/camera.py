from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

from annolid.gui.realtime_launch import parse_camera_source

from .function_base import FunctionTool


def _unwrap_safelinks_url(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    with contextlib.suppress(Exception):
        parts = urlsplit(text)
        host = str(parts.netloc or "").lower()
        if host.endswith("safelinks.protection.outlook.com"):
            query = parse_qs(parts.query or "", keep_blank_values=True)
            wrapped = str((query.get("url") or [""])[0] or "").strip()
            if wrapped:
                return unquote(wrapped)
    return text


def _normalize_stream_source(value: str) -> object:
    text = str(value or "").strip().strip("<>").rstrip(").,;!?`")
    if not text:
        return 0
    text = _unwrap_safelinks_url(text).strip().strip("<>").rstrip(").,;!?`")
    if not text:
        return 0
    return parse_camera_source(text)


def _validate_stream_source(value: object) -> tuple[bool, str]:
    if isinstance(value, int):
        if value < 0 or value > 9999:
            return False, "camera_source index must be between 0 and 9999."
        return True, ""
    text = str(value or "").strip()
    if not text:
        return True, ""
    if re.fullmatch(r"\d{1,4}", text):
        return True, ""
    has_scheme = bool(re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", text))
    if not has_scheme:
        return (
            False,
            "camera_source must be a camera index or stream URL "
            "(rtsp/rtsps/rtp/http/https/srt/tcp/udp).",
        )
    scheme = text.split("://", 1)[0].strip().lower()
    if scheme not in {"rtsp", "rtsps", "rtp", "http", "https", "srt", "tcp", "udp"}:
        return False, f"Unsupported stream URL scheme: {scheme}"
    return True, ""


def _safe_snapshot_filename(filename: str) -> str:
    text = Path(str(filename or "").strip()).name
    if not text:
        return ""
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._")
    if not sanitized:
        return ""
    lower = sanitized.lower()
    if not lower.endswith((".jpg", ".jpeg", ".png")):
        sanitized = f"{sanitized}.jpg"
    return sanitized


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except Exception:
        return False


def _resolve_snapshot_output_path(
    *,
    workspace_root: Path,
    filename: str,
) -> tuple[Path | None, str]:
    root = Path(workspace_root).expanduser().resolve()
    snapshots_dir = root / "camera_snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    snapshots_real = snapshots_dir.resolve()
    if not _is_relative_to(snapshots_real, root):
        return None, "camera_snapshots directory resolves outside workspace root."

    safe_filename = _safe_snapshot_filename(filename)
    if not safe_filename:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_filename = f"camera_probe_{stamp}.jpg"
    out_path = snapshots_real / safe_filename

    parent_real = out_path.parent.resolve()
    if not _is_relative_to(parent_real, snapshots_real):
        return None, "Snapshot output path resolves outside camera_snapshots directory."
    if out_path.exists() and out_path.is_symlink():
        return None, "Snapshot output path cannot be a symlink."
    return out_path, ""


class CameraSnapshotTool(FunctionTool):
    def __init__(self, allowed_dir: Path | None = None):
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve() if allowed_dir else None
        )

    @property
    def name(self) -> str:
        return "camera_snapshot"

    @property
    def description(self) -> str:
        return (
            "Probe a local or network camera stream and save a snapshot into the "
            "workspace camera_snapshots directory."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "camera_source": {"type": "string"},
                "timeout_sec": {"type": "number", "minimum": 0.5, "maximum": 30.0},
                "probe_frames": {"type": "integer", "minimum": 1, "maximum": 60},
                "filename": {"type": "string"},
            },
            "required": [],
        }

    async def execute(
        self,
        camera_source: str = "0",
        timeout_sec: float = 3.0,
        probe_frames: int = 3,
        filename: str = "",
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            return await asyncio.to_thread(
                self._capture_snapshot_sync,
                camera_source=str(camera_source or ""),
                timeout_sec=float(timeout_sec),
                probe_frames=int(probe_frames),
                filename=str(filename or ""),
            )
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"camera snapshot failed: {exc}"})

    def _capture_snapshot_sync(
        self,
        *,
        camera_source: str,
        timeout_sec: float,
        probe_frames: int,
        filename: str,
    ) -> str:
        try:
            import cv2
        except Exception as exc:
            return json.dumps({"ok": False, "error": f"OpenCV is unavailable: {exc}"})

        timeout_value = max(0.5, min(30.0, float(timeout_sec)))
        probe_value = max(1, min(60, int(probe_frames)))

        source_value = _normalize_stream_source(camera_source)
        source_display = str(source_value if source_value != "" else 0)
        is_valid, error = _validate_stream_source(source_value)
        if not is_valid:
            return json.dumps(
                {"ok": False, "error": error, "camera_source": source_display}
            )

        cap = None
        try:
            cap = cv2.VideoCapture()
            with contextlib.suppress(Exception):
                open_timeout_key = getattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC", None)
                if open_timeout_key is not None:
                    cap.set(open_timeout_key, timeout_value * 1000.0)
            with contextlib.suppress(Exception):
                read_timeout_key = getattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC", None)
                if read_timeout_key is not None:
                    cap.set(read_timeout_key, timeout_value * 1000.0)

            opened = bool(cap.open(source_value))
            if not opened:
                return json.dumps(
                    {
                        "ok": False,
                        "camera_source": source_display,
                        "error": f"Failed to open stream source: {source_display}",
                    }
                )

            got_frame = False
            frame = None
            frame_shape = None
            attempts = probe_value
            t0 = time.perf_counter()
            while attempts > 0 and (time.perf_counter() - t0) <= timeout_value:
                ret, candidate = cap.read()
                if ret and candidate is not None:
                    frame = candidate
                    got_frame = True
                    with contextlib.suppress(Exception):
                        frame_shape = tuple(int(v) for v in candidate.shape[:2])
                    break
                attempts -= 1
                time.sleep(0.05)

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            if not got_frame or frame is None:
                return json.dumps(
                    {
                        "ok": False,
                        "camera_source": source_display,
                        "opened": True,
                        "got_frame": False,
                        "frame_width": width,
                        "frame_height": height,
                        "fps": fps,
                        "error": "Stream opened but no frame received within timeout.",
                    }
                )

            workspace = self._allowed_dir or Path.cwd()
            snapshot_path, path_error = _resolve_snapshot_output_path(
                workspace_root=workspace,
                filename=filename,
            )
            if snapshot_path is None:
                return json.dumps(
                    {
                        "ok": False,
                        "camera_source": source_display,
                        "opened": True,
                        "got_frame": True,
                        "error": path_error,
                    }
                )
            saved = bool(cv2.imwrite(str(snapshot_path), frame))
            if not saved:
                return json.dumps(
                    {
                        "ok": False,
                        "camera_source": source_display,
                        "opened": True,
                        "got_frame": True,
                        "error": f"Failed to save snapshot: {snapshot_path}",
                    }
                )

            return json.dumps(
                {
                    "ok": True,
                    "camera_source": source_display,
                    "opened": True,
                    "got_frame": True,
                    "frame_height": int(frame_shape[0]) if frame_shape else height,
                    "frame_width": int(frame_shape[1]) if frame_shape else width,
                    "fps": fps,
                    "timeout_sec": timeout_value,
                    "probe_frames": probe_value,
                    "snapshot_path": str(snapshot_path),
                }
            )
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "camera_source": source_display,
                    "error": f"Stream probe failed: {exc}",
                }
            )
        finally:
            if cap is not None:
                with contextlib.suppress(Exception):
                    cap.release()


__all__ = [
    "CameraSnapshotTool",
    "_unwrap_safelinks_url",
    "_normalize_stream_source",
    "_validate_stream_source",
    "_resolve_snapshot_output_path",
]
