"""Infrastructure wrappers for agent config persistence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from annolid.core.agent.config import (
    get_config_path as _get_config_path,
    load_config as _load_config,
    save_config as _save_config,
)


def get_agent_config_path() -> Path:
    return _get_config_path()


def load_agent_config() -> Any:
    return _load_config()


def save_agent_config(config: Any) -> None:
    _save_config(config)


__all__ = [
    "get_agent_config_path",
    "load_agent_config",
    "save_agent_config",
]
