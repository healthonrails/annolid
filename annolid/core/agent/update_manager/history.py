from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from annolid.utils.atomic_io import atomic_write_json

_MAX_HISTORY_ENTRIES = 25


def _default_updates_dir() -> Path:
    return Path.home() / ".annolid" / "updates"


def get_update_history_path(base_dir: Optional[Path] = None) -> Path:
    root = Path(base_dir).expanduser().resolve() if base_dir else _default_updates_dir()
    return root / "history.json"


def get_auto_update_state_path(base_dir: Optional[Path] = None) -> Path:
    root = Path(base_dir).expanduser().resolve() if base_dir else _default_updates_dir()
    return root / "auto_update_state.json"


def load_update_history(
    *, base_dir: Optional[Path] = None, limit: int = 10
) -> List[Dict[str, Any]]:
    path = get_update_history_path(base_dir)
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    entries = raw.get("entries") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return []
    rows = [dict(item) for item in entries if isinstance(item, dict)]
    if limit > 0:
        return rows[-limit:]
    return rows


def record_update_history(
    *,
    project: str,
    channel: str,
    install_mode: str,
    previous_version: str,
    target_version: str,
    status: str,
    previous_commit: str = "",
    source_root: str = "",
    base_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {
        "timestamp_epoch_s": round(time.time(), 3),
        "project": str(project or "annolid"),
        "channel": str(channel or "stable"),
        "install_mode": str(install_mode or "package"),
        "previous_version": str(previous_version or ""),
        "target_version": str(target_version or ""),
        "status": str(status or ""),
    }
    if previous_commit:
        entry["previous_commit"] = str(previous_commit)
    if source_root:
        entry["source_root"] = str(source_root)
    try:
        path = get_update_history_path(base_dir)
        existing = load_update_history(base_dir=base_dir, limit=0)
        existing.append(entry)
        existing = existing[-_MAX_HISTORY_ENTRIES:]
        atomic_write_json(path, {"schema_version": 1, "entries": existing})
    except Exception:
        pass
    return entry


def last_known_good_version(
    *, project: str = "annolid", base_dir: Optional[Path] = None
) -> str:
    entries = load_update_history(base_dir=base_dir, limit=0)
    proj = str(project or "annolid").strip().lower()
    for item in reversed(entries):
        if str(item.get("project") or "annolid").strip().lower() != proj:
            continue
        if str(item.get("status") or "").strip().lower() != "updated":
            continue
        prev = str(item.get("previous_version") or "").strip()
        if prev:
            return prev
    return ""


def load_auto_update_state(*, base_dir: Optional[Path] = None) -> Dict[str, Any]:
    path = get_auto_update_state_path(base_dir)
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return dict(raw) if isinstance(raw, dict) else {}
    except Exception:
        return {}


def save_auto_update_state(
    state: Dict[str, Any], *, base_dir: Optional[Path] = None
) -> None:
    try:
        path = get_auto_update_state_path(base_dir)
        atomic_write_json(path, dict(state))
    except Exception:
        pass
