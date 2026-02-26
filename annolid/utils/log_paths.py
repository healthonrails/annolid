from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


ANNOLID_LOGS_DIRNAME = "annolid_logs"
ANNOLID_REALTIME_LOGS_DIRNAME = "annolid_realtime_logs"
APP_LOGS_DIRNAME = "logs"
APP_REALTIME_LOGS_SUBDIR = "realtime"
APP_RUNS_SUBDIR = "runs"
APP_LABEL_INDEX_SUBDIR = "label_index"


def _resolve_agent_workspace_root() -> Path:
    try:
        from annolid.core.agent.utils import get_agent_workspace_path

        return Path(get_agent_workspace_path()).expanduser().resolve()
    except Exception:
        return (Path.home() / ".annolid" / "workspace").expanduser().resolve()


def resolve_annolid_logs_root(dataset_root: Optional[Path] = None) -> Path:
    if dataset_root is not None:
        return Path(dataset_root).expanduser().resolve() / APP_LOGS_DIRNAME
    return _resolve_agent_workspace_root().parent / APP_LOGS_DIRNAME


def resolve_annolid_realtime_logs_root() -> Path:
    env_dir = os.environ.get("ANNOLID_REALTIME_LOG_DIR", "").strip()
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    legacy = (Path.home() / ANNOLID_REALTIME_LOGS_DIRNAME).expanduser().resolve()
    if legacy.exists():
        return legacy
    return resolve_annolid_logs_root() / APP_REALTIME_LOGS_SUBDIR
