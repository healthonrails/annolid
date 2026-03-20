from __future__ import annotations

import asyncio
import hashlib
import hmac
import platform
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence

import pytest

from annolid.core.agent.context import AgentContextBuilder
from annolid.core.agent.coding_harness import CodingHarnessManager
from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.session_manager import AgentSessionManager
from annolid.core.agent import skills as skills_module
from annolid.core.agent.skills import AgentSkillsLoader
from annolid.core.agent.subagent import (
    RuntimeSessionRouter,
    SubagentManager,
    build_subagent_tools_registry,
)
from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_registry import FunctionToolRegistry


def test_skills_loader_lists_workspace_skill(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "description: demo skill\n"
        'metadata: \'{"annolid": {"always": true}}\'\n'
        "---\n"
        "Use this for demo tasks.\n",
        encoding="utf-8",
    )
    loader = AgentSkillsLoader(tmp_path)
    skills = loader.list_skills(filter_unavailable=False)
    assert any(s["name"] == "demo" for s in skills)
    assert "demo skill" in loader.build_skills_summary()


def test_skills_loader_lists_builtin_skills() -> None:
    loader = AgentSkillsLoader(Path.cwd())
    skills = loader.list_skills(filter_unavailable=False)
    names = {s["name"] for s in skills}
    assert "github" in names
    assert "weather" in names
    assert "web-scraping" in names
    summary = loader.build_skills_summary()
    assert "<skills>" in summary
    assert "<location>" in summary


def test_skills_loader_precedence_workspace_over_managed_over_builtin(
    tmp_path: Path,
) -> None:
    builtin = tmp_path / "builtin"
    managed = tmp_path / "managed"
    workspace = tmp_path / "workspace"
    for root in (builtin, managed, workspace):
        (root / "skills" / "same").mkdir(parents=True, exist_ok=True)
    (builtin / "skills" / "same" / "SKILL.md").write_text(
        "---\ndescription: builtin\n---\nBuiltin skill\n",
        encoding="utf-8",
    )
    (managed / "skills" / "same" / "SKILL.md").write_text(
        "---\ndescription: managed\n---\nManaged skill\n",
        encoding="utf-8",
    )
    (workspace / "skills" / "same" / "SKILL.md").write_text(
        "---\ndescription: workspace\n---\nWorkspace skill\n",
        encoding="utf-8",
    )

    loader = AgentSkillsLoader(
        workspace=workspace,
        builtin_skills_dir=builtin / "skills",
        managed_skills_dir=managed / "skills",
    )
    skills = loader.list_skills(filter_unavailable=False)
    selected = next(s for s in skills if s["name"] == "same")
    assert selected["source"] == "workspace"
    assert str(workspace / "skills" / "same" / "SKILL.md") == selected["path"]


def test_skills_loader_honors_disable_model_invocation(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "ops"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "description: ops\n"
        "always: true\n"
        "disable-model-invocation: true\n"
        "---\n"
        "Do not inject into model context.\n",
        encoding="utf-8",
    )
    loader = AgentSkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
    assert "ops" not in loader.get_always_skills()
    injected = loader.load_skills_for_context(["ops"])
    assert injected == ""
    assert "ops" not in loader.build_skills_summary()


def test_skills_loader_honors_os_requirement(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "linux_only"
    skill_dir.mkdir(parents=True, exist_ok=True)
    disallowed_os = "darwin" if platform.system().lower() != "darwin" else "linux"
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "description: linux only\n"
        f'metadata: \'{{"openclaw": {{"os": ["{disallowed_os}"]}}}}\'\n'
        "---\n"
        "OS-gated skill.\n",
        encoding="utf-8",
    )
    loader = AgentSkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
    skills = loader.list_skills(filter_unavailable=True)
    assert not any(s["name"] == "linux_only" for s in skills)


def test_skills_loader_watch_reload_detects_skill_updates(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True, exist_ok=True)
    skill_file = skill_dir / "SKILL.md"
    skill_file.write_text(
        "---\ndescription: first\n---\nv1\n",
        encoding="utf-8",
    )
    loader = AgentSkillsLoader(
        tmp_path,
        builtin_skills_dir=tmp_path / "builtin",
        managed_skills_dir=tmp_path / "managed",
        watch=True,
        watch_poll_seconds=0.0,
    )
    initial = next(
        s for s in loader.list_skills(filter_unavailable=False) if s["name"] == "demo"
    )
    assert initial["description"] == "first"

    skill_file.write_text(
        "---\ndescription: second\n---\nv2\n",
        encoding="utf-8",
    )
    updated = next(
        s for s in loader.list_skills(filter_unavailable=False) if s["name"] == "demo"
    )
    assert updated["description"] == "second"


def test_skills_loader_watch_defaults_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = tmp_path / "config.json"
    cfg.write_text(
        '{"skills": {"load": {"watch": true, "pollSeconds": 0.0}}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(skills_module, "get_config_path", lambda: cfg)
    loader = AgentSkillsLoader(
        tmp_path,
        builtin_skills_dir=tmp_path / "builtin",
        managed_skills_dir=tmp_path / "managed",
    )
    assert loader.registry.watch_enabled is True
    assert loader.registry.watch_poll_seconds == 0.0


def test_skills_loader_validates_manifest_at_load_time(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "invalid_manifest"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        '---\ndescription: invalid manifest\nalways: "sometimes"\n---\nbad\n',
        encoding="utf-8",
    )
    loader = AgentSkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")

    all_skills = loader.list_skills(filter_unavailable=False)
    row = next(s for s in all_skills if s["name"] == "invalid_manifest")
    assert row["manifest_valid"] is False
    assert any("always must be a boolean" in e for e in row["manifest_errors"])

    available = loader.list_skills(filter_unavailable=True)
    assert not any(s["name"] == "invalid_manifest" for s in available)


def test_skills_loader_requires_signature_in_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    skill_dir = tmp_path / "skills" / "unsigned"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\ndescription: unsigned\n---\nUnsigned body\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ANNOLID_PRODUCTION_MODE", "1")
    loader = AgentSkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
    row = next(
        s
        for s in loader.list_skills(filter_unavailable=False)
        if s["name"] == "unsigned"
    )
    assert row["manifest_valid"] is False
    assert any("signature is required" in e for e in row["manifest_errors"])


def test_skills_loader_accepts_valid_signature_in_production(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret = "skill-secret"
    body = "Signed body\n"
    digest = hmac.new(
        secret.encode("utf-8"), body.encode("utf-8"), hashlib.sha256
    ).hexdigest()
    skill_dir = tmp_path / "skills" / "signed"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            "description: signed\n"
            f"signature: {digest}\n"
            "signature_alg: hmac-sha256\n"
            "---\n"
            f"{body}"
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("ANNOLID_PRODUCTION_MODE", "1")
    monkeypatch.setenv("ANNOLID_SKILL_SIGNING_KEY", secret)
    loader = AgentSkillsLoader(tmp_path, builtin_skills_dir=tmp_path / "builtin")
    row = next(
        s for s in loader.list_skills(filter_unavailable=False) if s["name"] == "signed"
    )
    assert row["manifest_valid"] is True


def test_context_builder_builds_user_media_payload(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("workspace instructions", encoding="utf-8")
    image_path = tmp_path / "image.png"
    image_path.write_bytes(b"\x89PNG\r\n\x1a\nfake")
    ctx = AgentContextBuilder(tmp_path)
    messages = ctx.build_messages(
        history=[],
        current_message="describe",
        media=[str(image_path)],
    )
    assert messages[0]["role"] == "system"
    assert "workspace instructions" in str(messages[0]["content"])
    assert "## Current Time" in str(messages[0]["content"])
    assert "(" in str(messages[0]["content"])
    assert ")" in str(messages[0]["content"])
    assert isinstance(messages[-1]["content"], list)
    assert messages[-1]["content"][-1]["text"] == "describe"


def test_context_builder_redacts_session_identifiers(tmp_path: Path) -> None:
    ctx = AgentContextBuilder(tmp_path)
    messages = ctx.build_messages(
        history=[],
        current_message="hello",
        channel="email",
        chat_id="user@example.com",
    )
    system_content = str(messages[0]["content"])
    assert "Channel: ***" in system_content
    assert "user@example.com" not in system_content
    assert "us***r@example.com" in system_content


def test_context_builder_auto_selects_skills_from_task_hint(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "weather-helper"
    skill_dir.mkdir(parents=True, exist_ok=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        "description: Use for weather and forecast questions\n"
        "---\n"
        "Use weather tools and summarize forecast.\n",
        encoding="utf-8",
    )
    ctx = AgentContextBuilder(tmp_path)
    messages = ctx.build_messages(
        history=[],
        current_message="Can you check today's weather forecast in Boston?",
    )
    system_content = str(messages[0]["content"])
    assert "# Auto-selected Skills" in system_content
    assert "### Skill: weather-helper" in system_content


def test_context_builder_bounds_system_prompt_size(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANNOLID_AGENT_SYSTEM_PROMPT_MAX_CHARS", "1800")
    (tmp_path / "AGENTS.md").write_text("A" * 6000, encoding="utf-8")
    (tmp_path / "SOUL.md").write_text("B" * 6000, encoding="utf-8")
    ctx = AgentContextBuilder(tmp_path)
    prompt = ctx.build_system_prompt()
    assert len(prompt) <= 1800
    assert "truncated to fit system prompt budget" in prompt


def test_subagent_manager_runs_background_task(tmp_path: Path) -> None:
    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": "subagent-result"}

    def loop_factory() -> AgentLoop:
        return AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=fake_llm,
            model="fake",
            workspace=str(tmp_path),
        )

    manager = SubagentManager(loop_factory=loop_factory, workspace=tmp_path)

    async def _run() -> None:
        msg = await manager.spawn(
            task="do work",
            label="worker",
            origin_channel="local",
            origin_chat_id="u1",
        )
        assert "started" in msg
        task_id = msg.split("id: ")[-1].split(")")[0]
        ok = await manager.wait(task_id, timeout=2.0)
        assert ok is True
        task = manager.get_task(task_id)
        assert task is not None
        assert task.status == "ok"
        assert task.result == "subagent-result"

    asyncio.run(_run())


def test_coding_harness_manager_processes_long_lived_session_messages(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str]] = []

    def _fake_invoke_turn(
        *,
        prompt: str,
        image_path: str,
        model: str,
        provider_name: str,
        settings: dict[str, Any],
        load_history_messages: Callable[[], list[dict[str, Any]]],
        session_id: str = "",
        runtime: str = "",
        timeout_s: float | None = None,
        max_tokens: int = 4096,
    ) -> tuple[str, str]:
        del image_path, provider_name, settings, timeout_s, max_tokens
        history = load_history_messages()
        calls.append((session_id, prompt))
        assert runtime == "acp"
        return prompt, f"reply-{len(history)}-{model}"

    manager = CodingHarnessManager(
        session_manager=AgentSessionManager(sessions_dir=tmp_path / "sessions"),
        invoke_turn=_fake_invoke_turn,
    )

    async def _run() -> None:
        started = await manager.start(
            task="inspect repo",
            label="coder",
            workspace=str(tmp_path),
            origin_channel="local",
            origin_chat_id="u1",
        )
        session_id = started.split("id: ")[-1].split(",")[0]
        await asyncio.wait_for(
            _wait_for_status(manager, session_id, "idle"), timeout=2.0
        )
        ok = await manager.send_message(session_id, "apply fix")
        assert ok is True
        await asyncio.wait_for(
            _wait_for_status(manager, session_id, "idle", turns=2), timeout=2.0
        )
        payload = await manager.poll(session_id, tail_messages=4)
        assert payload["ok"] is True
        assert payload["turn_count"] == 2
        assert payload["last_response"] == "reply-3-codex-cli/gpt-5.1-codex"
        tail = payload["tail_messages"]
        assert tail[-1]["role"] == "assistant"
        assert calls[0][0].startswith("acp:")
        closed = await manager.close(session_id)
        assert closed is True
        await asyncio.wait_for(
            _wait_for_status(manager, session_id, "closed"), timeout=2.0
        )

    asyncio.run(_run())


def test_coding_harness_manager_persists_error_and_stays_open(tmp_path: Path) -> None:
    seen = {"count": 0}

    def _fake_invoke_turn(**kwargs: Any) -> tuple[str, str]:
        seen["count"] += 1
        if seen["count"] == 1:
            raise RuntimeError("boom")
        return kwargs["prompt"], "ok-after-error"

    manager = CodingHarnessManager(
        session_manager=AgentSessionManager(sessions_dir=tmp_path / "sessions"),
        invoke_turn=_fake_invoke_turn,
    )

    async def _run() -> None:
        started = await manager.start(task="first task", workspace=str(tmp_path))
        session_id = started.split("id: ")[-1].split(",")[0]
        await asyncio.wait_for(
            _wait_for_status(manager, session_id, "error"), timeout=2.0
        )
        payload = await manager.poll(session_id)
        assert payload["last_error"] == "boom"
        queued = await manager.send_message(session_id, "retry")
        assert queued is True
        await asyncio.wait_for(
            _wait_for_status(manager, session_id, "idle", turns=1), timeout=2.0
        )
        payload = await manager.poll(session_id)
        assert payload["last_response"] == "ok-after-error"

    asyncio.run(_run())


async def _wait_for_status(
    manager: CodingHarnessManager,
    session_id: str,
    expected: str,
    *,
    turns: int | None = None,
) -> None:
    while True:
        payload = await manager.poll(session_id)
        if payload.get("ok") and payload.get("status") == expected:
            if turns is None or int(payload.get("turn_count") or 0) >= turns:
                return
        await asyncio.sleep(0.01)


def test_runtime_session_router_dispatches_subagent_and_acp(tmp_path: Path) -> None:
    subagent_calls: list[str] = []

    class _FakeSubagent:
        async def spawn(
            self,
            task: str,
            label: str | None = None,
            origin_channel: str = "cli",
            origin_chat_id: str = "direct",
        ) -> str:
            del label, origin_channel, origin_chat_id
            subagent_calls.append(task)
            return "subagent-ok"

        def list_tasks(self) -> dict[str, Any]:
            return {"sub_1": SimpleNamespace(status="ok", label="sub", result="ok")}

        def cancel(self, task_id: str) -> bool:
            return task_id == "sub_1"

    class _FakeACP:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def start(self, **kwargs: Any) -> str:
            self.calls.append(dict(kwargs))
            return "acp-ok"

        def list_sessions(self) -> dict[str, Any]:
            return {"acp_1": SimpleNamespace(status="idle", label="acp")}

        async def close(self, task_id: str) -> bool:
            return task_id == "acp_1"

    router = RuntimeSessionRouter(
        subagent_manager=_FakeSubagent(),  # type: ignore[arg-type]
        acp_manager=_FakeACP(),  # type: ignore[arg-type]
        workspace=tmp_path,
    )

    async def _run() -> None:
        subagent_reply = await router.spawn(task="inspect", runtime="subagent")
        acp_reply = await router.spawn(
            task="code",
            runtime="acp",
            provider="codex_cli",
            model="codex-cli/gpt-5.1-codex",
        )
        assert subagent_reply == "subagent-ok"
        assert acp_reply == "acp-ok"
        assert subagent_calls == ["inspect"]
        rows = router.list_tasks()
        assert "sub_1" in rows
        assert "acp_1" in rows
        assert await router.cancel("sub_1") is True
        assert await router.cancel("acp_1") is True

    asyncio.run(_run())


def test_coding_harness_manager_abort_cancels_active_turn_without_closing(
    tmp_path: Path,
) -> None:
    from threading import Event as ThreadEvent
    import time

    sessions_dir = tmp_path / "sessions"
    session_manager = AgentSessionManager(sessions_dir=sessions_dir)
    started = ThreadEvent()

    def _invoke_turn(**kwargs):
        prompt = str(kwargs.get("prompt") or "")
        cancel_event = kwargs.get("cancel_event")
        if prompt != "run forever":
            return prompt, "follow-up-ok"
        assert cancel_event is not None
        started.set()
        while not cancel_event.is_set():
            time.sleep(0.01)
        raise RuntimeError("Codex CLI request cancelled.")

    manager = CodingHarnessManager(
        session_manager=session_manager,
        invoke_turn=_invoke_turn,
    )

    async def _run() -> None:
        meta = await manager.start_session(
            task="run forever",
            workspace=str(tmp_path),
        )
        for _ in range(20):
            if started.is_set():
                break
            await asyncio.sleep(0.05)
        assert started.is_set()
        assert meta.status == "running"
        assert await manager.abort(meta.session_id) is True
        for _ in range(20):
            if meta.status == "idle" and meta.active_cancel_event is None:
                break
            await asyncio.sleep(0.05)
        assert meta.status == "idle"
        assert meta.close_requested is False
        assert await manager.send_message(meta.session_id, "follow-up") is True
        assert await manager.close(meta.session_id) is True
        assert meta.worker_task is not None
        await asyncio.wait_for(meta.worker_task, timeout=1.0)

    asyncio.run(_run())


def test_build_subagent_tools_registry_excludes_recursive_tools(
    tmp_path: Path,
) -> None:
    registry = asyncio.run(build_subagent_tools_registry(tmp_path))
    assert registry.has("read_file")
    assert registry.has("write_file")
    assert registry.has("edit_file")
    assert registry.has("list_dir")
    assert registry.has("exec")
    assert not registry.has("spawn")
    assert not registry.has("message")
    assert not registry.has("cron")
    assert not registry.has("coding_session_start")


def test_subagent_prompt_includes_time_and_skills_path(tmp_path: Path) -> None:
    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": "ok"}

    manager = SubagentManager(
        loop_factory=lambda: AgentLoop(
            tools=FunctionToolRegistry(),
            llm_callable=fake_llm,
            model="fake",
            workspace=str(tmp_path),
        ),
        workspace=tmp_path,
    )
    prompt = manager._build_subagent_prompt("review files")
    assert "## Current Time" in prompt
    assert "UTC" in prompt
    assert "## Skills" in prompt
    assert str(tmp_path / "skills") in prompt


def test_agent_loop_connects_mcp_without_overwriting_existing_tools(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ExistingTool(FunctionTool):
        @property
        def name(self) -> str:
            return "existing_tool"

        @property
        def description(self) -> str:
            return "Existing"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs: Any) -> str:
            del kwargs
            return "existing-ok"

    class _MCPTool(FunctionTool):
        @property
        def name(self) -> str:
            return "mcp_demo_ping"

        @property
        def description(self) -> str:
            return "Ping"

        @property
        def parameters(self) -> dict[str, Any]:
            return {"type": "object", "properties": {}}

        async def execute(self, **kwargs: Any) -> str:
            del kwargs
            return "pong"

    calls: list[dict[str, Any]] = []

    async def _fake_connect(mcp_servers, registry, stack) -> None:  # type: ignore[no-untyped-def]
        del stack
        calls.append(dict(mcp_servers))
        registry.register(_MCPTool())

    import annolid.core.agent.tools.mcp as mcp_tools

    monkeypatch.setattr(mcp_tools, "connect_mcp_servers", _fake_connect)

    async def _fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> Mapping[str, Any]:
        del messages, tools, model, on_token
        return {"content": "done"}

    registry = FunctionToolRegistry()
    existing = _ExistingTool()
    registry.register(existing)

    loop = AgentLoop(
        tools=registry,
        llm_callable=_fake_llm,
        model="fake",
        workspace=str(tmp_path),
        mcp_servers={"demo": {"command": "echo", "args": ["ok"]}},
    )
    result = asyncio.run(loop.run("hello", use_memory=False))

    assert result.content == "done"
    assert len(calls) == 1
    assert registry.get("existing_tool") is existing
    assert registry.has("mcp_demo_ping")
    assert loop._mcp_connected is False  # cleanup happens at end of run
    assert loop._mcp_stack is None
