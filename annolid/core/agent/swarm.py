from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import dataclass
from contextlib import suppress
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

    async def _close_loop(self, loop: Any) -> None:
        close = getattr(loop, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                with suppress(Exception):
                    await result
            return
        aclose = getattr(loop, "aclose", None)
        if callable(aclose):
            result = aclose()
            if inspect.isawaitable(result):
                with suppress(Exception):
                    await result

    @staticmethod
    def _shorten_text(text: str, limit: int = 700) -> str:
        value = " ".join(str(text or "").split()).strip()
        if not value:
            return ""
        if len(value) <= limit:
            return value
        head = max(120, int(limit * 0.6))
        tail = max(80, min(int(limit * 0.25), limit - head - 6))
        return f"{value[:head].rstrip()} ... {value[-tail:].lstrip()}"

    @staticmethod
    def _summarize_context_entries(entries: List[Dict[str, str]]) -> str:
        if not entries:
            return ""
        counts: Dict[str, int] = {}
        latest_by_agent: Dict[str, str] = {}
        latest_error_by_agent: Dict[str, str] = {}
        recent_lines: List[str] = []
        for entry in entries:
            agent = str(entry.get("agent") or "").strip()
            role = str(entry.get("role") or "").strip()
            kind = str(entry.get("kind") or "turn").strip().lower()
            content = str(entry.get("content") or "").strip()
            if not agent:
                continue
            counts[agent] = counts.get(agent, 0) + 1
            snippet = SwarmManager._shorten_text(content, 520)
            if kind == "error":
                latest_error_by_agent[agent] = snippet
            else:
                latest_by_agent[agent] = snippet
            if len(recent_lines) < 6:
                prefix = f"{agent} ({role})" if role else agent
                recent_lines.append(f"- {prefix}: {snippet}")

        condensed: List[str] = []
        for agent in sorted(counts):
            role = ""
            for entry in reversed(entries):
                if str(entry.get("agent") or "").strip() == agent:
                    role = str(entry.get("role") or "").strip()
                    break
            latest = latest_by_agent.get(agent, "")
            latest_error = latest_error_by_agent.get(agent, "")
            parts = [f"{agent}"]
            if role:
                parts.append(f"role={role}")
            parts.append(f"turns={counts[agent]}")
            if latest_error:
                parts.append(f"latest_error={latest_error}")
            elif latest:
                parts.append(f"latest={latest}")
            condensed.append("; ".join(parts))

        sections: List[str] = []
        if recent_lines:
            sections.append("# Recent Swarm Turns\n" + "\n".join(recent_lines))
        if condensed:
            sections.append(
                "# Condensed Agent State\n"
                + "\n".join(f"- {line}" for line in condensed)
            )
        return "\n\n".join(sections).strip()

    def _build_swarm_context(self, task: str) -> str:
        base = f"# Swarm Task\n{task.strip()}\n"
        summary = self._summarize_context_entries(self.history)
        if summary:
            base = f"{base}\n{summary}"
        compact = base.strip()
        if len(compact) <= self.max_context_chars:
            return compact
        return self._shorten_text(compact, self.max_context_chars)

    def _record_context_entry(
        self,
        *,
        agent: str,
        role: str,
        kind: str,
        content: str,
    ) -> None:
        self.history.append(
            {
                "agent": agent,
                "role": role,
                "kind": kind,
                "content": content,
            }
        )
        if len(self.history) > 128:
            self.history = self.history[-128:]

    async def run_swarm(self, task: str, max_turns: int = 8) -> str:
        """
        Runs a collaborative swarm loop where agents take turns processing the task context.
        Agents can observe each other's outputs.
        """
        try:
            from annolid.gui.widgets.threejs_viewer_server import (
                clear_swarm_state,
                update_swarm_node,
            )
        except ImportError:

            def clear_swarm_state() -> None:
                pass

            def update_swarm_node(
                node_id: str,
                status: str,
                current_task: str,
                **_kwargs: Any,
            ) -> None:
                pass

        if not self.agents:
            return "No agents registered in swarm."
        if max_turns <= 0:
            return "Swarm max turns must be greater than 0."

        self.history = []
        current_context = self._build_swarm_context(task)
        output = f"Starting swarm with task: {task}\n"
        previous_agent_name = ""
        resolved_loops: Dict[str, _SupportsRun] = {}
        turns_used = 0
        completed = False
        timed_out = False
        final_output = output

        # Clear old state and register all new agents in the visualizer immediately
        try:
            clear_swarm_state()
        except Exception:
            pass

        for name, agent in self.agents.items():
            try:
                update_swarm_node(agent.name.lower(), "idle", "Awaiting Tasks")
            except Exception:
                pass
        try:
            for agent in self.agents.values():
                try:
                    resolved_loops[agent.name] = await self._resolve_loop(agent)
                except Exception as exc:
                    error_block = (
                        f"\n--- {agent.name} ({agent.role}) ---\n[ERROR] {exc}\n"
                    )
                    output += error_block
                    self._record_context_entry(
                        agent=agent.name,
                        role=agent.role,
                        kind="error",
                        content=str(exc),
                    )
                    current_context = self._build_swarm_context(task)

            for turn in range(max_turns):
                turns_used = turn + 1
                for agent in self.agents.values():
                    loop = resolved_loops.get(agent.name)
                    if loop is None:
                        continue

                    swarm_prompt = (
                        f"You are part of an Agent Swarm.\n"
                        f"Your role is: {agent.role}\n\n"
                        f"Current Swarm Context:\n{current_context}\n\n"
                        f"Please contribute to the task based on your role. "
                        f"If the task is fully resolved, include 'TASK COMPLETE' in your output."
                    )

                    def _progress_cb(text: str, p_agent=previous_agent_name) -> None:
                        try:
                            thinking_match = ""
                            if text.strip().startswith("<think>"):
                                thinking_match = (
                                    text.replace("<think>", "")
                                    .replace("</think>", "")
                                    .strip()
                                )

                            short_thought = (
                                (thinking_match[:100] + "...")
                                if len(thinking_match) > 100
                                else thinking_match
                            )
                            short_task = (text[:40] + "...") if len(text) > 40 else text

                            update_swarm_node(
                                agent.name.lower(),
                                "active",
                                f"Thinking: {short_task}",
                                thinking=short_thought,
                                parent=p_agent.lower() if p_agent else "",
                            )
                        except Exception:
                            pass

                    try:
                        turn_started = time.perf_counter()
                        update_swarm_node(
                            agent.name.lower(),
                            "active",
                            "Thinking...",
                            parent=previous_agent_name.lower()
                            if previous_agent_name
                            else "",
                            turn_latency_ms=None,
                        )
                        result = await loop.run(
                            swarm_prompt,
                            session_id=agent.session_id,
                            system_prompt=agent.system_prompt,
                            use_memory=True,
                            on_progress=_progress_cb,
                            inbound_metadata={"parent_id": previous_agent_name},
                        )
                    except Exception as exc:
                        turn_latency_ms = (time.perf_counter() - turn_started) * 1000.0
                        error_block = (
                            f"\n--- {agent.name} ({agent.role}) ---\n[ERROR] {exc}\n"
                        )
                        output += error_block
                        self._record_context_entry(
                            agent=agent.name,
                            role=agent.role,
                            kind="error",
                            content=str(exc),
                        )
                        current_context = self._build_swarm_context(task)
                        try:
                            update_swarm_node(
                                agent.name.lower(),
                                "idle",
                                "Awaiting Tasks",
                                output=f"[ERROR] {exc}",
                                turn_latency_ms=turn_latency_ms,
                            )
                        except Exception:
                            pass
                        continue

                    turn_latency_ms = (time.perf_counter() - turn_started) * 1000.0

                    content = str(getattr(result, "content", "") or "").strip()

                    clean_output = content
                    thinking_block = ""
                    if "<think>" in content and "</think>" in content:
                        parts = content.split("</think>", 1)
                        thinking_block = parts[0].replace("<think>", "").strip()
                        clean_output = parts[1].strip()

                    try:
                        update_swarm_node(
                            agent.name.lower(),
                            "idle",
                            "Awaiting Tasks",
                            thinking=thinking_block[:100] + "..."
                            if len(thinking_block) > 100
                            else thinking_block,
                            output=clean_output[:150] + "..."
                            if len(clean_output) > 150
                            else clean_output,
                            turn_latency_ms=turn_latency_ms,
                        )
                    except Exception:
                        pass

                    if len(content) > 6000:
                        content = (
                            content[:3000] + "\n... [TRUNCATED] ...\n" + content[-3000:]
                        )

                    turn_output = f"\n--- {agent.name} ({agent.role}) ---\n{content}\n"
                    output += turn_output
                    self._record_context_entry(
                        agent=agent.name,
                        role=agent.role,
                        kind="turn",
                        content=content,
                    )
                    current_context = self._build_swarm_context(task)
                    previous_agent_name = agent.name

                    if "TASK COMPLETE" in content:
                        final_output = (
                            output + "\nSwarm reached consensus: TASK COMPLETE."
                        )
                        completed = True
                        break
                if completed:
                    break

            if not completed:
                final_output = output + "\nSwarm max turns reached."
                timed_out = True
        finally:
            if turns_used > 0 or completed or timed_out:
                try:
                    from annolid.core.agent.swarm_budget import (
                        record_swarm_budget_observation,
                    )

                    record_swarm_budget_observation(
                        task,
                        requested_turns=max_turns,
                        used_turns=max(1, turns_used),
                        completed=completed,
                        timed_out=timed_out,
                        agent_count=len(self.agents),
                    )
                except Exception:
                    pass
            for loop in resolved_loops.values():
                with suppress(Exception):
                    await self._close_loop(loop)
        return final_output
