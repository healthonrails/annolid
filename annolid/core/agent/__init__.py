"""Core (GUI-free) agent orchestration primitives.

Note: this module uses lazy imports so that lightweight subpackages like
`annolid.core.agent.tools` can be imported without pulling in video/ML
dependencies during module import time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from .behavior_engine import (
        BehaviorEngine,
        BehaviorEngineConfig,
        BehaviorEvent,
        BehaviorUpdate,
    )
    from .frame_source import FrameSource
    from .orchestrator import AnnolidAgent
    from .pipeline import AgentPipelineConfig
    from .runner import AgentRunConfig, AgentRunner
    from .service import AgentServiceResult, run_agent_to_results
    from .track_store import TrackStore

__all__ = [
    "BehaviorEngine",
    "BehaviorEngineConfig",
    "BehaviorEvent",
    "BehaviorUpdate",
    "FrameSource",
    "AnnolidAgent",
    "AgentPipelineConfig",
    "AgentRunConfig",
    "AgentRunner",
    "AgentServiceResult",
    "run_agent_to_results",
    "TrackStore",
]


def __getattr__(name: str):  # noqa: ANN001
    if name in {"AgentPipelineConfig"}:
        from .pipeline import AgentPipelineConfig

        return {"AgentPipelineConfig": AgentPipelineConfig}[name]

    if name in {"AnnolidAgent"}:
        from .orchestrator import AnnolidAgent

        return {"AnnolidAgent": AnnolidAgent}[name]

    if name in {"FrameSource"}:
        from .frame_source import FrameSource

        return {"FrameSource": FrameSource}[name]

    if name in {"TrackStore"}:
        from .track_store import TrackStore

        return {"TrackStore": TrackStore}[name]

    if name in {
        "BehaviorEngine",
        "BehaviorEngineConfig",
        "BehaviorEvent",
        "BehaviorUpdate",
    }:
        from .behavior_engine import (
            BehaviorEngine,
            BehaviorEngineConfig,
            BehaviorEvent,
            BehaviorUpdate,
        )

        return {
            "BehaviorEngine": BehaviorEngine,
            "BehaviorEngineConfig": BehaviorEngineConfig,
            "BehaviorEvent": BehaviorEvent,
            "BehaviorUpdate": BehaviorUpdate,
        }[name]

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
