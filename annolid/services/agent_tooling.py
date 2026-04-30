"""Service-layer validation helpers for agent tooling surfaces."""

from __future__ import annotations

import asyncio
import re
import tempfile
from pathlib import Path
from typing import Any

_LEGACY_GWS_NAME_RE = re.compile(
    r"^(?:gws(?:$|[-_].*)|google_workspace(?:$|[-_].*))$",
    flags=re.IGNORECASE,
)


def _is_legacy_gws_name(name: str) -> bool:
    return bool(_LEGACY_GWS_NAME_RE.match(str(name or "").strip()))


def _sanitize_capability_payload(
    tool_pool: dict[str, Any],
    skill_pool: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    sanitized_tool_pool = dict(tool_pool or {})
    allowed_tools = [
        str(name)
        for name in list(sanitized_tool_pool.get("allowed_tools") or [])
        if str(name).strip() and not _is_legacy_gws_name(str(name))
    ]
    denied_tools = [
        str(name)
        for name in list(sanitized_tool_pool.get("denied_tools") or [])
        if str(name).strip() and not _is_legacy_gws_name(str(name))
    ]
    counts = dict(sanitized_tool_pool.get("counts") or {})
    sanitized_tool_pool["allowed_tools"] = allowed_tools
    sanitized_tool_pool["denied_tools"] = denied_tools
    counts["allowed"] = len(allowed_tools)
    counts["denied"] = len(denied_tools)
    counts["registered"] = counts.get(
        "registered", len(allowed_tools) + len(denied_tools)
    )
    sanitized_tool_pool["counts"] = counts

    sanitized_skill_pool = dict(skill_pool or {})
    nested_pool = dict(sanitized_skill_pool.get("skill_pool") or {})
    original_preview_rows = list(nested_pool.get("preview") or [])
    original_unavailable_rows = list(nested_pool.get("unavailable_skills") or [])
    preview_rows = [
        row
        for row in original_preview_rows
        if not _is_legacy_gws_name(str((row or {}).get("name") or ""))
    ]
    unavailable_rows = [
        row
        for row in original_unavailable_rows
        if not _is_legacy_gws_name(str((row or {}).get("name") or ""))
    ]
    always_skills = [
        str(name)
        for name in list(nested_pool.get("always_skills") or [])
        if str(name).strip() and not _is_legacy_gws_name(str(name))
    ]
    suggested_skills = [
        row
        for row in list(sanitized_skill_pool.get("suggested_skills") or [])
        if not _is_legacy_gws_name(str((row or {}).get("name") or ""))
    ]
    nested_counts = dict(nested_pool.get("counts") or {})
    if nested_counts:
        nested_counts["always"] = len(always_skills)
        removed_available = max(0, len(original_preview_rows) - len(preview_rows))
        removed_unavailable = max(
            0, len(original_unavailable_rows) - len(unavailable_rows)
        )
        nested_counts["available"] = max(
            0, int(nested_counts.get("available") or 0) - removed_available
        )
        nested_counts["unavailable"] = max(
            0, int(nested_counts.get("unavailable") or 0) - removed_unavailable
        )
        nested_counts["total"] = (
            nested_counts["available"] + nested_counts["unavailable"]
        )
    nested_pool["preview"] = preview_rows
    nested_pool["unavailable_skills"] = unavailable_rows
    nested_pool["always_skills"] = always_skills
    nested_pool["counts"] = nested_counts
    sanitized_skill_pool["skill_pool"] = nested_pool
    sanitized_skill_pool["suggested_skills"] = suggested_skills
    return sanitized_tool_pool, sanitized_skill_pool


def validate_agent_tools() -> tuple[dict[str, object], int]:
    summary: dict[str, object] = {"status": "ok", "checks": []}

    def _record(name: str, *, ok: bool, detail: str) -> None:
        summary["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            summary["status"] = "error"

    try:
        from annolid.core.agent.tools.artifacts import FileArtifactStore, content_hash

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(base_dir=Path(tmpdir), run_id="validate")
            meta_path = store.resolve("agent_cache.json", kind="cache")
            payload = {"hello": "world"}
            store.write_meta(meta_path, {"content_hash": content_hash(payload)})
            ok = store.should_reuse_cache(meta_path, content_hash(payload))
            _record("artifacts", ok=bool(ok), detail="cache metadata round-trip")
    except Exception as exc:
        _record("artifacts", ok=False, detail=str(exc))

    try:
        from annolid.core.agent.tools.sampling import (
            FPSampler,
            RandomSampler,
            UniformSampler,
        )

        uniform = UniformSampler(step=2).sample_indices(10)
        fps = FPSampler(target_fps=5).sample_indices(30, fps=30)
        random = RandomSampler(count=2, seed=1).sample_indices(5)
        ok = bool(uniform) and bool(fps) and bool(random)
        _record("sampling", ok=ok, detail="uniform/fps/random sampling")
    except Exception as exc:
        _record("sampling", ok=False, detail=str(exc))

    try:
        from annolid.core.agent.tools.registry import ToolRegistry
        from annolid.core.agent.tools.base import Tool, ToolContext
        from annolid.core.agent.tools.function_registry import FunctionToolRegistry
        from annolid.core.agent.tools.function_builtin import (
            register_nanobot_style_tools,
        )
        from annolid.core.agent.tools.utility import register_builtin_utility_tools

        class _DummyTool(Tool[int, int]):
            name = "dummy"

            def run(self, ctx: ToolContext, payload: int) -> int:
                _ = ctx
                return payload + 1

        registry = ToolRegistry()
        registry.register("dummy", _DummyTool)
        register_builtin_utility_tools(registry)
        fn_registry = FunctionToolRegistry()
        asyncio.run(register_nanobot_style_tools(fn_registry))
        instance = registry.create("dummy")
        ok = (
            isinstance(instance, _DummyTool)
            and registry.has("calculator")
            and fn_registry.has("read_file")
            and fn_registry.has("exec")
        )
        _record(
            "registry",
            ok=ok,
            detail="register/create tool + utility + nanobot-style function tools",
        )
    except Exception as exc:
        _record("registry", ok=False, detail=str(exc))

    try:
        from annolid.core.agent.tools.vector_index import NumpyEmbeddingIndex
        from annolid.domain import FrameRef

        index = NumpyEmbeddingIndex(
            embeddings=[[0.1, 0.0], [0.0, 1.0]],
            frames=[FrameRef(frame_index=0), FrameRef(frame_index=1)],
        )
        results = index.search([0.2, 0.1], top_k=1)
        ok = bool(results)
        _record("vector_index", ok=ok, detail="numpy cosine search")
    except ImportError as exc:
        _record("vector_index", ok=True, detail=f"skipped: {exc}")
    except Exception as exc:
        _record("vector_index", ok=False, detail=str(exc))

    return summary, (0 if summary.get("status") == "ok" else 1)


def describe_agent_tool_pool(
    *,
    workspace: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    config_path: str | None = None,
) -> dict[str, Any]:
    from annolid.core.agent.config import load_config
    from annolid.core.agent.tools.function_registry import FunctionToolRegistry
    from annolid.core.agent.tools.nanobot import register_nanobot_style_tools
    from annolid.core.agent.tools.policy import (
        build_tool_permission_context,
        resolve_allowed_tools,
    )
    from annolid.core.agent.utils import get_agent_workspace_path

    cfg = load_config(Path(config_path).expanduser() if config_path else None)
    resolved_workspace = get_agent_workspace_path(workspace)
    resolved_model = str(model or cfg.agents.defaults.model or "").strip()
    resolved_provider = str(
        provider or cfg.get_provider_name(model=resolved_model) or ""
    )
    if not resolved_provider:
        resolved_provider = "unknown"

    registry = FunctionToolRegistry()
    asyncio.run(
        register_nanobot_style_tools(
            registry,
            allowed_dir=resolved_workspace,
            allowed_read_roots=cfg.tools.allowed_read_roots,
            email_cfg=cfg.tools.email,
            calendar_cfg=cfg.tools.calendar,
            google_auth_cfg=cfg.tools.google_auth,
            google_drive_enabled=bool(cfg.tools.google_drive_enabled),
            box_cfg=cfg.tools.box,
            mcp_servers=cfg.tools.mcp_servers,
            stack=None,
        )
    )
    all_names = sorted(set(registry.tool_names))
    resolved_policy = resolve_allowed_tools(
        all_tool_names=all_names,
        tools_cfg=cfg.tools,
        provider=resolved_provider,
        model=resolved_model,
    )
    permission_ctx = build_tool_permission_context(
        all_tool_names=all_names,
        resolved_policy=resolved_policy,
    )
    denied = sorted(
        tool_name
        for tool_name in all_names
        if tool_name not in resolved_policy.allowed_tools
    )

    return {
        "workspace": str(resolved_workspace),
        "provider": resolved_provider,
        "model": resolved_model,
        "policy_profile": resolved_policy.profile,
        "policy_source": resolved_policy.source,
        "allow_patterns": list(resolved_policy.allow_patterns),
        "deny_patterns": list(resolved_policy.deny_patterns),
        "counts": {
            "registered": len(all_names),
            "allowed": len(resolved_policy.allowed_tools),
            "denied": len(denied),
        },
        "allowed_tools": sorted(resolved_policy.allowed_tools),
        "denied_tools": denied,
        "permission_context": permission_ctx.to_dict(),
    }


def describe_agent_skill_pool(
    *,
    workspace: str | None = None,
    task_hint: str | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    from annolid.core.agent.skills import AgentSkillsLoader
    from annolid.core.agent.utils import get_agent_workspace_path

    resolved_workspace = get_agent_workspace_path(workspace)
    loader = AgentSkillsLoader(resolved_workspace)
    pool = loader.describe_skill_pool()
    hint = str(task_hint or "").strip()
    suggested = (
        loader.suggest_skills_for_task_scored(hint, top_k=max(1, int(top_k)))
        if hint
        else []
    )
    return {
        "workspace": str(resolved_workspace),
        "task_hint": hint,
        "skill_pool": pool,
        "suggested_skills": suggested,
    }


def describe_agent_capabilities(
    *,
    workspace: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    task_hint: str | None = None,
    top_k: int = 5,
    config_path: str | None = None,
) -> dict[str, Any]:
    tool_pool = describe_agent_tool_pool(
        workspace=workspace,
        provider=provider,
        model=model,
        config_path=config_path,
    )
    skill_pool = describe_agent_skill_pool(
        workspace=workspace,
        task_hint=task_hint,
        top_k=top_k,
    )
    tool_pool, skill_pool = _sanitize_capability_payload(tool_pool, skill_pool)
    return {
        "workspace": tool_pool["workspace"],
        "provider": tool_pool["provider"],
        "model": tool_pool["model"],
        "tool_pool": tool_pool,
        "skill_pool": skill_pool,
        "summary": {
            "registered_tools": int(tool_pool["counts"]["registered"]),
            "available_tools": int(tool_pool["counts"]["allowed"]),
            "available_skills": int(skill_pool["skill_pool"]["counts"]["available"]),
            "suggested_skills": len(skill_pool["suggested_skills"]),
        },
    }


__all__ = [
    "describe_agent_capabilities",
    "describe_agent_skill_pool",
    "describe_agent_tool_pool",
    "validate_agent_tools",
]
