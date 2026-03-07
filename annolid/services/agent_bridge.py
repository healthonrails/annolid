"""Service-layer adapter for ACP bridge execution."""

from __future__ import annotations


def run_agent_acp_bridge(*, workspace: str | None = None) -> int:
    from annolid.core.agent.acp_stdio_bridge import run_stdio_acp_bridge

    return int(run_stdio_acp_bridge(workspace=workspace))


__all__ = ["run_agent_acp_bridge"]
