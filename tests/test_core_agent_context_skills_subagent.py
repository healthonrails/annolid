from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Mapping, Sequence

from annolid.core.agent.context import AgentContextBuilder
from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.skills import AgentSkillsLoader
from annolid.core.agent.subagent import SubagentManager, build_subagent_tools_registry
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


def test_subagent_manager_runs_background_task(tmp_path: Path) -> None:
    async def fake_llm(
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
        model: str,
    ) -> Mapping[str, Any]:
        del messages, tools, model
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
    registry = build_subagent_tools_registry(tmp_path)
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
    ) -> Mapping[str, Any]:
        del messages, tools, model
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
