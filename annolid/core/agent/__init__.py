"""Core (GUI-free) agent orchestration primitives."""

from .runner import AgentRunConfig, AgentRunner
from .service import AgentServiceResult, run_agent_to_results

__all__ = [
    "AgentRunConfig",
    "AgentRunner",
    "AgentServiceResult",
    "run_agent_to_results",
]
