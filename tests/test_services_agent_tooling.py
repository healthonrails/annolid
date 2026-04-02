from __future__ import annotations

from pathlib import Path


def test_describe_agent_tool_pool_resolves_allowed_and_denied(monkeypatch) -> None:
    from annolid.core.agent.config import AgentConfig
    import annolid.services.agent_tooling as tooling_mod

    cfg = AgentConfig()
    cfg.tools.profile = "minimal"
    cfg.tools.allow = ["read_file"]
    cfg.tools.deny = ["write_*"]

    class _StubTool:
        def __init__(self, name: str):
            self.name = name

        def to_schema(self):
            return {"type": "function", "function": {"name": self.name}}

    async def _register_stub_tools(registry, **kwargs):
        del kwargs
        registry.register(_StubTool("read_file"))
        registry.register(_StubTool("write_file"))
        registry.register(_StubTool("exec"))

    monkeypatch.setattr(
        "annolid.core.agent.config.load_config",
        lambda config_path=None: cfg,
    )
    monkeypatch.setattr(
        "annolid.core.agent.tools.nanobot.register_nanobot_style_tools",
        _register_stub_tools,
    )
    monkeypatch.setattr(
        "annolid.core.agent.utils.get_agent_workspace_path",
        lambda workspace=None: Path(workspace) if workspace else Path("/tmp/ws"),
    )

    payload = tooling_mod.describe_agent_tool_pool(
        workspace="/tmp/ws",
        provider="ollama",
        model="qwen3",
    )
    assert payload["workspace"] == "/tmp/ws"
    assert payload["counts"]["registered"] == 3
    assert "read_file" in payload["allowed_tools"]
    assert "write_file" in payload["denied_tools"]
    assert payload["permission_context"]["deny_names"]


def test_describe_agent_skill_pool_returns_summary_and_suggestions(
    monkeypatch,
) -> None:
    import annolid.services.agent_tooling as tooling_mod

    class _StubLoader:
        def __init__(self, workspace):
            self.workspace = workspace

        def describe_skill_pool(self):
            return {
                "retrieval_mode": "lexical",
                "counts": {
                    "total": 2,
                    "available": 2,
                    "unavailable": 0,
                    "always": 1,
                },
            }

        def suggest_skills_for_task_scored(self, task_hint, top_k=5):
            del top_k
            return [
                {
                    "name": "weather",
                    "score": 1.0,
                    "strategy": "lexical",
                    "source": "workspace",
                    "description": str(task_hint),
                }
            ]

    monkeypatch.setattr(
        "annolid.core.agent.skills.AgentSkillsLoader",
        _StubLoader,
    )
    monkeypatch.setattr(
        "annolid.core.agent.utils.get_agent_workspace_path",
        lambda workspace=None: Path(workspace) if workspace else Path("/tmp/ws"),
    )

    payload = tooling_mod.describe_agent_skill_pool(
        workspace="/tmp/ws",
        task_hint="weather forecast",
        top_k=3,
    )
    assert payload["workspace"] == "/tmp/ws"
    assert payload["task_hint"] == "weather forecast"
    assert payload["skill_pool"]["counts"]["total"] == 2
    assert payload["suggested_skills"][0]["name"] == "weather"


def test_describe_agent_capabilities_combines_tools_and_skills(monkeypatch) -> None:
    import annolid.services.agent_tooling as tooling_mod

    monkeypatch.setattr(
        tooling_mod,
        "describe_agent_tool_pool",
        lambda **kwargs: {
            "workspace": str(kwargs.get("workspace") or "/tmp/ws"),
            "provider": str(kwargs.get("provider") or "ollama"),
            "model": str(kwargs.get("model") or "qwen3"),
            "counts": {"registered": 4, "allowed": 3, "denied": 1},
        },
    )
    monkeypatch.setattr(
        tooling_mod,
        "describe_agent_skill_pool",
        lambda **kwargs: {
            "workspace": str(kwargs.get("workspace") or "/tmp/ws"),
            "task_hint": str(kwargs.get("task_hint") or ""),
            "skill_pool": {
                "counts": {"total": 2, "available": 2, "unavailable": 0, "always": 1}
            },
            "suggested_skills": [{"name": "weather"}],
        },
    )

    payload = tooling_mod.describe_agent_capabilities(
        workspace="/tmp/ws",
        provider="ollama",
        model="qwen3",
        task_hint="weather forecast",
        top_k=3,
    )
    assert payload["workspace"] == "/tmp/ws"
    assert payload["tool_pool"]["counts"]["registered"] == 4
    assert payload["skill_pool"]["skill_pool"]["counts"]["available"] == 2
    assert payload["summary"]["suggested_skills"] == 1
