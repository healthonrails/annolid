"""Service helpers for GUI chat runtime bootstrap and path resolution."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from annolid.core.agent.config import load_config
from annolid.core.agent.utils import get_agent_workspace_path
from annolid.core.agent.gui_backend.paths import (
    build_pdf_search_roots,
    build_workspace_roots,
    resolve_pdf_path_for_roots,
    resolve_video_path_for_roots,
)


def get_chat_workspace() -> Path:
    return get_agent_workspace_path()


def get_chat_allowed_read_roots() -> list[str]:
    try:
        cfg = load_config()
        return list(getattr(cfg.tools, "allowed_read_roots", []) or [])
    except Exception:
        return []


def build_chat_workspace_roots() -> list[Path]:
    return build_workspace_roots(get_chat_workspace(), get_chat_allowed_read_roots())


def build_chat_pdf_search_roots() -> list[Path]:
    return build_pdf_search_roots(get_chat_workspace(), get_chat_allowed_read_roots())


def build_chat_vcs_read_roots() -> list[str]:
    return [str(path) for path in build_chat_workspace_roots()]


def resolve_chat_pdf_path(raw_path: str) -> Optional[Path]:
    return resolve_pdf_path_for_roots(raw_path, build_chat_workspace_roots())


def resolve_chat_video_path(raw_path: str) -> Optional[Path]:
    return resolve_video_path_for_roots(raw_path, build_chat_workspace_roots())


def get_chat_realtime_defaults() -> dict[str, Any]:
    try:
        cfg = load_config()
        realtime_cfg = getattr(getattr(cfg, "tools", None), "realtime", None)
        return dict(realtime_cfg) if isinstance(realtime_cfg, dict) else {}
    except Exception:
        return {}


def get_chat_email_defaults() -> dict[str, Any]:
    try:
        cfg = load_config()
        email_cfg = getattr(getattr(cfg, "tools", None), "email", None)
        if email_cfg is None:
            return {}
        return {
            "enabled": bool(getattr(email_cfg, "enabled", False)),
            "default_to": str(getattr(email_cfg, "default_to", "") or ""),
            "smtp_host": str(getattr(email_cfg, "smtp_host", "") or ""),
            "smtp_port": int(getattr(email_cfg, "smtp_port", 587) or 587),
            "imap_host": str(getattr(email_cfg, "imap_host", "") or ""),
            "imap_port": int(getattr(email_cfg, "imap_port", 993) or 993),
            "user": str(getattr(email_cfg, "user", "") or ""),
            "password": str(getattr(email_cfg, "password", "") or ""),
        }
    except Exception:
        return {}


def get_chat_attachment_roots() -> list[str | Path]:
    return [get_chat_workspace(), *get_chat_allowed_read_roots()]


def read_chat_memory_text(filename: str = "MEMORY.md") -> str:
    memory_file = get_chat_workspace() / "memory" / str(filename)
    try:
        if memory_file.exists():
            return memory_file.read_text(encoding="utf-8")
    except Exception:
        return ""
    return ""


def get_chat_tutorials_dir() -> Path:
    path = get_chat_workspace() / "tutorials"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_chat_camera_snapshots_dir() -> Path:
    path = get_chat_workspace() / "camera_snapshots"
    path.mkdir(parents=True, exist_ok=True)
    return path


__all__ = [
    "build_chat_pdf_search_roots",
    "build_chat_vcs_read_roots",
    "build_chat_workspace_roots",
    "get_chat_attachment_roots",
    "get_chat_allowed_read_roots",
    "get_chat_camera_snapshots_dir",
    "get_chat_email_defaults",
    "get_chat_realtime_defaults",
    "get_chat_tutorials_dir",
    "get_chat_workspace",
    "read_chat_memory_text",
    "resolve_chat_pdf_path",
    "resolve_chat_video_path",
]
