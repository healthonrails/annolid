from __future__ import annotations

from annolid.core.agent.tools.base import Tool, ToolContext
from annolid.core.agent.tools.registry import ToolRegistry, build_pipeline


class _DummyTool(Tool[str, str]):
    name = "dummy"

    def run(self, ctx: ToolContext, payload: str) -> str:  # noqa: ARG002
        return payload.upper()


def test_registry_create_and_pipeline() -> None:
    registry = ToolRegistry()
    registry.register("dummy", lambda cfg: _DummyTool(config=cfg))
    tool = registry.create("dummy", {"x": 1})
    assert isinstance(tool, _DummyTool)
    assert tool.config["x"] == 1

    pipeline = build_pipeline(
        registry,
        [{"tool": "dummy", "config": {"y": 2}}],
    )
    assert len(pipeline) == 1
    assert isinstance(pipeline[0], _DummyTool)
