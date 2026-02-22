from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Protocol

from annolid.core.agent.loop import AgentLoop


class _SupportsRun(Protocol):
    async def run(self, *args: Any, **kwargs: Any) -> Any: ...


@dataclass
class SwarmAgent:
    """Configuration for a single swarm participant."""

    name: str
    role: str
    system_prompt: str
    loop_factory: Callable[[], _SupportsRun | Awaitable[_SupportsRun]]

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        loop_factory: Callable[
            [],
            AgentLoop
            | _SupportsRun
            | asyncio.Future[AgentLoop]
            | asyncio.Task[AgentLoop],
        ],
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.loop_factory = loop_factory
        self.session_id = f"swarm:{name}"


class SwarmManager:
    """Orchestrates multiple agents to collaborate on complex tasks."""

    def __init__(self):
        self.agents: Dict[str, SwarmAgent] = {}
        self.history: List[Dict[str, str]] = []
        self.max_context_chars = 24000

    def register_agent(self, agent: SwarmAgent) -> None:
        self.agents[agent.name] = agent

    async def _resolve_loop(self, agent: SwarmAgent) -> _SupportsRun:
        candidate = agent.loop_factory()
        if inspect.isawaitable(candidate):
            candidate = await candidate
        if not hasattr(candidate, "run"):
            raise TypeError(
                f"loop_factory for agent '{agent.name}' did not return a loop-like object with .run()."
            )
        return candidate  # type: ignore[return-value]

    def _append_context(self, context: str, turn_output: str) -> str:
        updated = f"{context}\n{turn_output}".strip()
        if len(updated) <= self.max_context_chars:
            return updated
        return (
            "[CONTEXT TRUNCATED]\n"
            + updated[-(self.max_context_chars - len("[CONTEXT TRUNCATED]\n")) :]
        )

    async def run_swarm(self, task: str, max_turns: int = 5) -> str:
        """
        Runs a collaborative swarm loop where agents take turns processing the task context.
        Agents can observe each other's outputs.
        """
        if not self.agents:
            return "No agents registered in swarm."
        if max_turns <= 0:
            return "Swarm max turns must be greater than 0."

        current_context = task
        output = f"Starting swarm with task: {task}\n"

        for turn in range(max_turns):
            for name, agent in self.agents.items():
                try:
                    loop = await self._resolve_loop(agent)
                except Exception as exc:
                    error_block = (
                        f"\n--- {agent.name} ({agent.role}) ---\n[ERROR] {exc}\n"
                    )
                    output += error_block
                    current_context = self._append_context(current_context, error_block)
                    continue

                swarm_prompt = (
                    f"You are part of an Agent Swarm.\n"
                    f"Your role is: {agent.role}\n\n"
                    f"Current Swarm Context:\n{current_context}\n\n"
                    f"Please contribute to the task based on your role. "
                    f"If the task is fully resolved, include 'TASK COMPLETE' in your output."
                )

                try:
                    result = await loop.run(
                        swarm_prompt,
                        session_id=agent.session_id,
                        system_prompt=agent.system_prompt,
                        use_memory=True,
                    )
                except Exception as exc:
                    error_block = (
                        f"\n--- {agent.name} ({agent.role}) ---\n[ERROR] {exc}\n"
                    )
                    output += error_block
                    current_context = self._append_context(current_context, error_block)
                    continue

                content = str(getattr(result, "content", "") or "").strip()
                if len(content) > 6000:
                    content = (
                        content[:3000] + "\n... [TRUNCATED] ...\n" + content[-3000:]
                    )

                turn_output = f"\n--- {agent.name} ({agent.role}) ---\n{content}\n"
                output += turn_output
                current_context = self._append_context(current_context, turn_output)

                if "TASK COMPLETE" in content:
                    output += "\nSwarm reached consensus: TASK COMPLETE."
                    return output

        output += "\nSwarm max turns reached."
        return output
