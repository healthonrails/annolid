"""Infrastructure wrappers for agent workspace paths."""

from __future__ import annotations

from pathlib import Path

from annolid.core.agent.utils import get_agent_workspace_path as _get_workspace_path


def get_agent_workspace_path() -> Path:
    return _get_workspace_path()


__all__ = ["get_agent_workspace_path"]
