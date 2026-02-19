from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


def open_video_tool(
    path: str,
    *,
    resolve_video_path: Callable[[str], Optional[Path]],
    invoke_open_video: Callable[[Path], bool],
) -> Dict[str, object]:
    video_path = resolve_video_path(path)
    if video_path is None:
        raw_text = str(path or "").strip()
        basename = Path(raw_text).name if raw_text else ""
        return {
            "ok": False,
            "error": "Video not found from provided path/text.",
            "input": raw_text,
            "basename": basename,
            "hint": (
                "Provide an absolute path, or a filename located in workspace/read-roots."
            ),
        }
    if not invoke_open_video(video_path):
        return {"ok": False, "error": "Failed to queue GUI video open action"}
    return {"ok": True, "queued": True, "path": str(video_path)}


def resolve_video_path_for_gui_tool(
    raw_path: str,
    *,
    widget: Any,
    load_config_fn: Callable[[], Any],
    get_workspace_path_fn: Callable[[], Path],
    build_workspace_roots_fn: Callable[[Path, List[str]], List[Path]],
    resolve_video_path_for_roots_fn: Callable[..., Optional[Path]],
) -> Optional[Path]:
    try:
        cfg = load_config_fn()
        read_roots_cfg = list(getattr(cfg.tools, "allowed_read_roots", []) or [])
    except Exception:
        read_roots_cfg = []
    roots = build_workspace_roots_fn(get_workspace_path_fn(), read_roots_cfg)

    active_video_raw: str = ""
    try:
        host = getattr(widget, "host_window_widget", None) if widget else None
        if host is None and widget is not None:
            host_getter = getattr(widget, "window", None)
            if callable(host_getter):
                host = host_getter()
        active_video_raw = str(getattr(host, "video_file", "") or "").strip()
        if active_video_raw:
            candidate_video = Path(active_video_raw).expanduser()
            if candidate_video.exists():
                roots.append(candidate_video.parent)
    except Exception:
        active_video_raw = ""

    return resolve_video_path_for_roots_fn(
        raw_path,
        roots,
        active_video_raw=active_video_raw,
        max_scan=30000,
    )
