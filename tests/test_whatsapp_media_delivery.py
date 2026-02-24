from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from annolid.core.agent.loop import AgentLoop
from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_registry import FunctionToolRegistry


class MockCameraSnapshotTool(FunctionTool):
    @property
    def name(self) -> str:
        return "camera_snapshot"

    @property
    def description(self) -> str:
        return "Mock camera snapshot tool."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        # Return a top-level path (Like real CameraSnapshotTool)
        payload = {"ok": True, "snapshot_path": "camera_snapshots/mock_snapshot.jpg"}
        return json.dumps(payload)


class MockGuiPayloadTool(FunctionTool):
    @property
    def name(self) -> str:
        return "gui_check_stream_source"

    @property
    def description(self) -> str:
        return "Mock GUI tool returning payload."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        # Structure often used by tool wrappers return payload
        payload = {
            "ok": True,
            "payload": {
                "ok": True,
                "snapshot_path": "camera_snapshots/mock_payload_snapshot.jpg",
            },
        }
        return json.dumps(payload)


class MockNestedTool(FunctionTool):
    @property
    def name(self) -> str:
        return "check_stream_source"

    @property
    def description(self) -> str:
        return "Mock tool with deep nesting."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        # Deep nesting (steps/capture)
        payload = {
            "ok": True,
            "steps": {
                "capture": {
                    "ok": True,
                    "snapshot_path": "camera_snapshots/mock_nested_snapshot.jpg",
                }
            },
        }
        return json.dumps(payload)


def test_agent_loop_intercepts_top_level(tmp_path: Path):
    registry = FunctionToolRegistry()
    registry.register(MockCameraSnapshotTool())

    async def fake_llm(*args, **kwargs):
        if not hasattr(fake_llm, "n"):
            fake_llm.n = 0
        fake_llm.n += 1
        if fake_llm.n == 1:
            return {
                "content": "Taking snapshot",
                "tool_calls": [
                    {"id": "c1", "name": "camera_snapshot", "arguments": {}}
                ],
            }
        return {"content": "Done", "tool_calls": []}

    loop = AgentLoop(
        tools=registry, llm_callable=fake_llm, model="fake", workspace=str(tmp_path)
    )
    result = asyncio.run(loop.run("test"))

    assert len(result.media) == 1
    assert result.media[0].endswith("camera_snapshots/mock_snapshot.jpg")


def test_agent_loop_intercepts_payload_nested(tmp_path: Path):
    registry = FunctionToolRegistry()
    registry.register(MockGuiPayloadTool())

    async def fake_llm(*args, **kwargs):
        if not hasattr(fake_llm, "n"):
            fake_llm.n = 0
        fake_llm.n += 1
        if fake_llm.n == 1:
            return {
                "content": "GUI tool",
                "tool_calls": [
                    {"id": "c2", "name": "gui_check_stream_source", "arguments": {}}
                ],
            }
        return {"content": "Done", "tool_calls": []}

    loop = AgentLoop(
        tools=registry, llm_callable=fake_llm, model="fake", workspace=str(tmp_path)
    )
    result = asyncio.run(loop.run("test"))

    assert len(result.media) == 1
    assert result.media[0].endswith("camera_snapshots/mock_payload_snapshot.jpg")


def test_agent_loop_intercepts_steps_capture_nested(tmp_path: Path):
    registry = FunctionToolRegistry()
    registry.register(MockNestedTool())

    async def fake_llm(*args, **kwargs):
        if not hasattr(fake_llm, "n"):
            fake_llm.n = 0
        fake_llm.n += 1
        if fake_llm.n == 1:
            return {
                "content": "Nested tool",
                "tool_calls": [
                    {"id": "c3", "name": "check_stream_source", "arguments": {}}
                ],
            }
        return {"content": "Done", "tool_calls": []}

    loop = AgentLoop(
        tools=registry, llm_callable=fake_llm, model="fake", workspace=str(tmp_path)
    )
    result = asyncio.run(loop.run("test"))

    assert len(result.media) == 1
    assert result.media[0].endswith("camera_snapshots/mock_nested_snapshot.jpg")
