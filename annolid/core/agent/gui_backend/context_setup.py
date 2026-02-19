from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List

from annolid.core.agent.config import load_config
from annolid.core.agent.tools import FunctionToolRegistry, register_nanobot_style_tools
from annolid.core.agent.tools.policy import resolve_allowed_tools
from annolid.core.agent.utils import get_agent_workspace_path


@dataclass(frozen=True)
class ExecutionPrerequisites:
    workspace: Path
    agent_cfg: Any
    allowed_read_roots: List[str]
    t_after_workspace: float
    t_after_config: float


@dataclass(frozen=True)
class ContextToolPreparation:
    tools: FunctionToolRegistry
    policy_profile: str
    policy_source: str
    t_before_register: float
    t_after_register: float
    t_before_policy: float
    t_after_policy: float


def load_execution_prerequisites() -> ExecutionPrerequisites:
    workspace = get_agent_workspace_path()
    t_after_workspace = time.perf_counter()
    agent_cfg = load_config()
    t_after_config = time.perf_counter()
    allowed_read_roots = list(getattr(agent_cfg.tools, "allowed_read_roots", []) or [])
    return ExecutionPrerequisites(
        workspace=workspace,
        agent_cfg=agent_cfg,
        allowed_read_roots=allowed_read_roots,
        t_after_workspace=t_after_workspace,
        t_after_config=t_after_config,
    )


async def prepare_context_tools(
    *,
    include_tools: bool,
    workspace: Path,
    allowed_read_roots: List[str],
    agent_cfg: Any,
    register_gui_tools: Callable[[FunctionToolRegistry], None],
    provider: str,
    model: str,
    enable_web_tools: bool,
    always_disabled_tools: Iterable[str],
    web_tools: Iterable[str],
    resolve_policy: Callable[..., Any] = resolve_allowed_tools,
) -> ContextToolPreparation:
    tools = FunctionToolRegistry()
    t_before_register = time.perf_counter()
    if include_tools:
        calendar_cfg = getattr(agent_cfg.tools, "calendar", None)
        await register_nanobot_style_tools(
            tools,
            allowed_dir=workspace,
            allowed_read_roots=allowed_read_roots,
            email_cfg=agent_cfg.tools.email,
            calendar_cfg=calendar_cfg,
        )
        register_gui_tools(tools)
    t_after_register = time.perf_counter()

    t_before_policy = time.perf_counter()
    policy_profile, policy_source = apply_tool_policy(
        tools=tools,
        include_tools=include_tools,
        tools_cfg=agent_cfg.tools,
        provider=provider,
        model=model,
        enable_web_tools=enable_web_tools,
        always_disabled_tools=always_disabled_tools,
        web_tools=web_tools,
        resolve_policy=resolve_policy,
    )
    t_after_policy = time.perf_counter()

    return ContextToolPreparation(
        tools=tools,
        policy_profile=policy_profile,
        policy_source=policy_source,
        t_before_register=t_before_register,
        t_after_register=t_after_register,
        t_before_policy=t_before_policy,
        t_after_policy=t_after_policy,
    )


def apply_tool_policy(
    *,
    tools: FunctionToolRegistry,
    include_tools: bool,
    tools_cfg: Any,
    provider: str,
    model: str,
    enable_web_tools: bool,
    always_disabled_tools: Iterable[str],
    web_tools: Iterable[str],
    resolve_policy: Callable[..., Any] = resolve_allowed_tools,
) -> tuple[str, str]:
    if not include_tools:
        return "none", "tool_intent_skip"

    disabled_tools = set(always_disabled_tools)
    if not enable_web_tools:
        disabled_tools.update(web_tools)
    for tool_name in disabled_tools:
        tools.unregister(tool_name)

    resolved_policy = resolve_policy(
        all_tool_names=tools.tool_names,
        tools_cfg=tools_cfg,
        provider=provider,
        model=model,
    )
    for tool_name in list(tools.tool_names):
        if tool_name not in resolved_policy.allowed_tools:
            tools.unregister(tool_name)
    return resolved_policy.profile, resolved_policy.source
