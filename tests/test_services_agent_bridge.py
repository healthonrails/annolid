from __future__ import annotations

from annolid.services.agent_bridge import run_agent_acp_bridge


def test_run_agent_acp_bridge_delegates(monkeypatch) -> None:
    import annolid.core.agent.acp_stdio_bridge as bridge_mod

    captured = {}

    def _run_stdio_acp_bridge(*, workspace=None):
        captured["workspace"] = workspace
        return 7

    monkeypatch.setattr(bridge_mod, "run_stdio_acp_bridge", _run_stdio_acp_bridge)

    result = run_agent_acp_bridge(workspace="demo-workspace")

    assert result == 7
    assert captured["workspace"] == "demo-workspace"
