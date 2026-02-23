from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Callable

from .function_base import FunctionTool


class SwarmTool(FunctionTool):
    def __init__(
        self,
        run_swarm_callback: Callable[[str, int], Awaitable[str] | str] | None = None,
    ):
        self._run_swarm_callback = run_swarm_callback

    def set_swarm_callback(
        self, callback: Callable[[str, int], Awaitable[str] | str] | None
    ) -> None:
        self._run_swarm_callback = callback

    @property
    def name(self) -> str:
        return "run_swarm"

    @property
    def description(self) -> str:
        return (
            "Launch a multi-agent swarm to collaborate on a complex task. "
            "Use this when a task is too complex for a single agent or requires multiple perspectives "
            "(e.g., Planning, Researching, Coding, Reviewing, Architecture, Performance, Security). "
            "You MUST use this tool to spawn subagents for 'brainstorming', 'roundtable' discussions, debates, or collaborative reviews. "
            "Do NOT simulate, roleplay, or transcribe the discussion yourself in text; always use this tool so the system can track subagent progress and display them in the COMMAND CENTER. "
            "The swarm will work autonomously and return a final consolidated answer with consensus on roadmap and priorities."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "A clear, detailed description of the objective the swarm needs to accomplish.",
                },
                "max_turns": {
                    "type": "integer",
                    "description": "Maximum number of turn cycles the swarm is allowed to take (default: 5).",
                    "default": 5,
                },
                "agents": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of agent roles/names to participate (e.g. ['Architecture', 'Security']). If omitted, uses default roles.",
                },
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        max_turns: int = 5,
        agents: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if self._run_swarm_callback is None:
            return "Error: swarm callback not configured in AgentLoop"

        try:
            ret = self._run_swarm_callback(task, max_turns, agents)
        except Exception as exc:
            return f"Error triggering swarm: {exc}"

        if asyncio.iscoroutine(ret):
            return str(await ret)
        return str(ret)
