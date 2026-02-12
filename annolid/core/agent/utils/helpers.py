from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple


def ensure_dir(path: Path) -> Path:
    """Ensure directory exists and return the normalized path."""
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_agent_data_path() -> Path:
    """Get Annolid agent data directory (~/.annolid)."""
    return ensure_dir(Path.home() / ".annolid")


def get_agent_workspace_path(workspace: Optional[str] = None) -> Path:
    """
    Resolve and ensure workspace path.

    Defaults to ~/.annolid/workspace for agent-local assets.
    """
    if workspace:
        return ensure_dir(Path(workspace).expanduser())
    return ensure_dir(get_agent_data_path() / "workspace")


def get_sessions_path() -> Path:
    return ensure_dir(get_agent_data_path() / "sessions")


def get_memory_path(workspace: Optional[Path] = None) -> Path:
    ws = (
        Path(workspace).expanduser()
        if workspace is not None
        else get_agent_workspace_path()
    )
    return ensure_dir(ws / "memory")


def get_skills_path(workspace: Optional[Path] = None) -> Path:
    ws = (
        Path(workspace).expanduser()
        if workspace is not None
        else get_agent_workspace_path()
    )
    return ensure_dir(ws / "skills")


def today_date() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def timestamp() -> str:
    return datetime.now().isoformat()


def truncate_string(text: str, max_len: int = 100, suffix: str = "...") -> str:
    value = str(text or "")
    max_len = max(1, int(max_len))
    if len(value) <= max_len:
        return value
    suffix_text = str(suffix or "")
    if len(suffix_text) >= max_len:
        return suffix_text[:max_len]
    return value[: max_len - len(suffix_text)] + suffix_text


def safe_filename(name: str) -> str:
    value = str(name or "")
    unsafe = '<>:"/\\|?*'
    for char in unsafe:
        value = value.replace(char, "_")
    value = value.strip()
    return value or "unnamed"


def parse_session_key(key: str) -> Tuple[str, str]:
    text = str(key or "")
    parts = text.split(":", 1)
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(f"Invalid session key: {key}")
    return parts[0], parts[1]
