from __future__ import annotations

from types import SimpleNamespace

from annolid.agents import behavior_agent


def test_initialize_agent_uses_annolid_bot_defaults_by_default(monkeypatch) -> None:
    resolve_calls: list[dict[str, object]] = []
    ensure_calls: list[object] = []

    def _fake_resolve_llm_config(**kwargs):
        resolve_calls.append(dict(kwargs))
        return SimpleNamespace(
            provider="openai",
            model="gpt-4o-mini",
            profile=None,
            api_key=None,
        )

    monkeypatch.setattr(behavior_agent, "resolve_llm_config", _fake_resolve_llm_config)
    monkeypatch.setattr(behavior_agent, "ensure_provider_env", ensure_calls.append)

    agent = behavior_agent.initialize_agent()

    assert resolve_calls == [{"profile": None, "provider": None, "model": None}]
    assert len(ensure_calls) == 1
    assert agent._provider == "openai"
    assert agent._model_name == "gpt-4o-mini"
    assert agent._llm_profile is None


def test_initialize_agent_can_opt_into_legacy_behavior_profile(
    monkeypatch,
) -> None:
    resolve_calls: list[dict[str, object]] = []

    def _fake_resolve_llm_config(**kwargs):
        resolve_calls.append(dict(kwargs))
        return SimpleNamespace(
            provider="openai",
            model="gpt-4o-mini",
            profile=behavior_agent.LEGACY_PROFILE_NAME,
            api_key=None,
        )

    monkeypatch.setattr(behavior_agent, "resolve_llm_config", _fake_resolve_llm_config)
    monkeypatch.setattr(behavior_agent, "ensure_provider_env", lambda _cfg: None)

    agent = behavior_agent.initialize_agent(use_annolid_bot=False)

    assert resolve_calls == [
        {
            "profile": behavior_agent.LEGACY_PROFILE_NAME,
            "provider": None,
            "model": None,
        }
    ]
    assert agent._llm_profile == behavior_agent.LEGACY_PROFILE_NAME
