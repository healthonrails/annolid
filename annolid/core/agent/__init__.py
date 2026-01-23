"""Core (GUI-free) agent orchestration primitives.

Note: this module uses lazy imports so that lightweight subpackages like
`annolid.core.agent.tools` can be imported without pulling in video/ML
dependencies during module import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .runner import AgentRunConfig, AgentRunner
    from .service import AgentServiceResult, run_agent_to_results

__all__ = [
    "AgentRunConfig",
    "AgentRunner",
    "AgentServiceResult",
    "run_agent_to_results",
]


def __getattr__(name: str):  # noqa: ANN001
    if name in {"AgentRunConfig", "AgentRunner"}:
        from .runner import AgentRunConfig, AgentRunner

        return {"AgentRunConfig": AgentRunConfig, "AgentRunner": AgentRunner}[name]

    if name in {"AgentServiceResult", "run_agent_to_results"}:
        from .service import AgentServiceResult, run_agent_to_results

        return {
            "AgentServiceResult": AgentServiceResult,
            "run_agent_to_results": run_agent_to_results,
        }[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(list(globals().keys()) + __all__)
