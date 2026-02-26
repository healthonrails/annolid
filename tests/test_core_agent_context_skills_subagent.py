from __future__ import annotations

import asyncio
import hashlib
import hmac
import platform
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import pytest

from annolid.core.agent.context import AgentContextBuilder
from annolid.core.agent.loop import AgentLoop
from annolid.core.agent import skills as skills_module
from annolid.core.agent.skills import AgentSkillsLoader
from annolid.core.agent.subagent import SubagentManager, build_subagent_tools_registry
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
