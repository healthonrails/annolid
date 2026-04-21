from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import annolid.agents.coding_agent.agent as coding_agent_module
from annolid.domain.behavior_agent import TaskPlan


def test_generate_code_uses_annolid_bot_acp_when_codex_cli_is_configured(
    monkeypatch,
) -> None:
    resolve_calls: list[dict[str, Any]] = []
    invoke_calls: list[dict[str, Any]] = []

    def _fake_resolve_llm_config(**kwargs):
        resolve_calls.append(dict(kwargs))
        return SimpleNamespace(
            provider="codex_cli",
            model="codex-cli/gpt-5.1-codex",
            profile=None,
            params={},
        )

    def _fake_invoke_turn(**kwargs):
        invoke_calls.append(dict(kwargs))
        history = kwargs["load_history_messages"]()
        assert history[0]["role"] == "system"
        assert "Annolid Bot" in history[0]["content"]
        return (
            kwargs["prompt"],
            "```python\n"
            "def run(inputs):\n"
            "    return {'status': 'ok', 'assay_type': str(inputs.get('assay_type', 'unknown')), 'artifact_count': int(inputs.get('artifact_count', 0)), 'segment_count': int(inputs.get('segment_count', 0))}\n"
            "```",
        )

    monkeypatch.setattr(
        coding_agent_module,
        "resolve_llm_config",
        _fake_resolve_llm_config,
    )
    monkeypatch.setattr(coding_agent_module, "ensure_provider_env", lambda _cfg: None)
    monkeypatch.setattr(
        coding_agent_module,
        "provider_kind",
        lambda settings, provider: "codex_cli",
    )
    monkeypatch.setattr(
        coding_agent_module,
        "dependency_error_for_kind",
        lambda kind: None,
    )

    agent = coding_agent_module.AnalysisCodingAgent(
        settings={"provider_definitions": {}, "codex_cli": {}},
        invoke_turn=_fake_invoke_turn,
    )

    code = agent.generate_code(TaskPlan(assay_type="aggression"))

    assert resolve_calls == [
        {"profile": None, "provider": None, "model": None, "persist": False}
    ]
    assert code.startswith("def run(inputs):")
    assert "```" not in code
    assert len(invoke_calls) == 1
    assert invoke_calls[0]["runtime"] == "acp"
    assert invoke_calls[0]["provider_name"] == "codex_cli"
    assert invoke_calls[0]["session_id"].startswith("acp:analysis_coding:")


def test_generate_code_uses_active_openai_compat_provider(monkeypatch) -> None:
    def _fake_resolve_llm_config(**kwargs):
        del kwargs
        return SimpleNamespace(
            provider="openai",
            model="gpt-4o-mini",
            profile=None,
            params={},
        )

    calls: list[dict[str, Any]] = []

    def _fake_openai_compat(**kwargs):
        calls.append(dict(kwargs))
        return kwargs["prompt"], "def run(inputs):\n    return {'status': 'ok'}\n"

    monkeypatch.setattr(
        coding_agent_module,
        "resolve_llm_config",
        _fake_resolve_llm_config,
    )
    monkeypatch.setattr(coding_agent_module, "ensure_provider_env", lambda _cfg: None)
    monkeypatch.setattr(
        coding_agent_module,
        "provider_kind",
        lambda settings, provider: "openai_compat",
    )
    monkeypatch.setattr(
        coding_agent_module,
        "dependency_error_for_kind",
        lambda kind: None,
    )

    agent = coding_agent_module.AnalysisCodingAgent(
        invoke_openai_compat=_fake_openai_compat
    )
    code = agent.generate_code(TaskPlan(assay_type="open_field"))

    assert code.startswith("def run(inputs):")
    assert len(calls) == 1
    assert calls[0]["provider_name"] == "openai"
    assert calls[0]["model"] == "gpt-4o-mini"


def test_run_falls_back_when_provider_kind_is_unsupported(monkeypatch) -> None:
    def _fake_resolve_llm_config(**kwargs):
        del kwargs
        return SimpleNamespace(
            provider="custom",
            model="custom-model",
            profile=None,
            params={},
        )

    monkeypatch.setattr(
        coding_agent_module,
        "resolve_llm_config",
        _fake_resolve_llm_config,
    )
    monkeypatch.setattr(
        coding_agent_module,
        "provider_kind",
        lambda settings, provider: "custom_kind",
    )

    agent = coding_agent_module.AnalysisCodingAgent(
        invoke_turn=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("provider invocation should not happen for unsupported kind")
        )
    )

    result = agent.run(
        plan=TaskPlan(assay_type="open_field"),
        artifacts=[],
        segments=[],
    )

    assert result.code.startswith("def run(inputs):")
    assert result.execution_output["status"] == "ok"
    assert result.evidence[0]["generation_mode"] == "deterministic_fallback"
    assert (
        result.evidence[0]["fallback_reason"]
        == "provider_kind_not_supported:custom_kind"
    )


def test_run_falls_back_when_generated_code_is_not_sandbox_compatible(
    monkeypatch,
) -> None:
    def _fake_resolve_llm_config(**kwargs):
        del kwargs
        return SimpleNamespace(
            provider="openai",
            model="gpt-4o-mini",
            profile=None,
            params={},
        )

    def _fake_openai_compat(**kwargs):
        del kwargs
        return "ignored", (
            "def run(inputs):\n"
            "    return {'status': 'ok' if isinstance(inputs, dict) else 'bad'}\n"
        )

    monkeypatch.setattr(
        coding_agent_module,
        "resolve_llm_config",
        _fake_resolve_llm_config,
    )
    monkeypatch.setattr(coding_agent_module, "ensure_provider_env", lambda _cfg: None)
    monkeypatch.setattr(
        coding_agent_module,
        "provider_kind",
        lambda settings, provider: "openai_compat",
    )
    monkeypatch.setattr(
        coding_agent_module,
        "dependency_error_for_kind",
        lambda kind: None,
    )

    agent = coding_agent_module.AnalysisCodingAgent(
        invoke_openai_compat=_fake_openai_compat
    )
    result = agent.run(
        plan=TaskPlan(assay_type="open_field"), artifacts=[], segments=[]
    )

    assert result.execution_output["status"] == "ok"
    assert result.evidence[0]["generation_mode"] == "deterministic_fallback"
    assert "isinstance" in str(result.evidence[0]["fallback_reason"])


def test_run_falls_back_when_generated_code_breaks_output_contract(
    monkeypatch,
) -> None:
    def _fake_resolve_llm_config(**kwargs):
        del kwargs
        return SimpleNamespace(
            provider="openai",
            model="gpt-4o-mini",
            profile=None,
            params={},
        )

    def _fake_openai_compat(**kwargs):
        del kwargs
        return "ignored", (
            "def run(inputs):\n"
            "    return {'status': 'success', 'assay_type': 'x', 'artifact_count': 0, 'segment_count': 0}\n"
        )

    monkeypatch.setattr(
        coding_agent_module,
        "resolve_llm_config",
        _fake_resolve_llm_config,
    )
    monkeypatch.setattr(coding_agent_module, "ensure_provider_env", lambda _cfg: None)
    monkeypatch.setattr(
        coding_agent_module,
        "provider_kind",
        lambda settings, provider: "openai_compat",
    )
    monkeypatch.setattr(
        coding_agent_module,
        "dependency_error_for_kind",
        lambda kind: None,
    )

    agent = coding_agent_module.AnalysisCodingAgent(
        invoke_openai_compat=_fake_openai_compat
    )
    result = agent.run(
        plan=TaskPlan(assay_type="open_field"), artifacts=[], segments=[]
    )

    assert result.execution_output["status"] == "ok"
    assert result.evidence[0]["generation_mode"] == "deterministic_fallback"
    assert "status='ok'" in str(result.evidence[0]["fallback_reason"])
