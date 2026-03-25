import asyncio
import pytest
from unittest.mock import MagicMock
from annolid.core.agent.loop import AgentLoop


@pytest.mark.anyio
async def test_agent_loop_streaming_visualizer_updates(monkeypatch):
    """Test that AgentLoop correctly calls update_swarm_node during streaming."""

    # Mock update_swarm_node
    # We need to mock it where it's imported in loop.py
    # Since it's imported inside the function, we might need a different strategy
    # but for now let's mock it in the module scope if possible or just check the call chain.

    tools = MagicMock()
    tools.get_all_definitions.return_value = []

    # Mock LLM provider that streams tokens
    async def mock_llm_callable(messages, tools, model, on_token=None):
        if on_token:
            on_token("<think>Thinking carefully...</think>")
            await asyncio.sleep(0.6)
            on_token("Hello ")
            on_token("world!")
        return {
            "content": "Hello world!",
            "reasoning_content": "Thinking carefully...",
            "finish_reason": "stop",
            "usage": {"total_tokens": 10},
        }

    loop = AgentLoop(tools=tools, llm_callable=mock_llm_callable, model="test-model")

    # We want to verify that update_swarm_node is called.
    # Since update_swarm_node is imported locally in loop.py:
    # from annolid.gui.widgets.threejs_viewer_server import update_swarm_node

    mock_update = MagicMock()
    monkeypatch.setattr(
        "annolid.gui.widgets.threejs_viewer_server.update_swarm_node",
        mock_update,
    )

    await loop.run("test", session_id="swarm:agent1")

    # Check if update_swarm_node was called during streaming
    # Note: the mock_update might be called multiple times due to throttling
    assert mock_update.called
    # Check one of the calls
    found_active = False
    for call in mock_update.call_args_list:
        if call.args[0] == "agent1" and call.args[1] == "active":
            found_active = True
            break
    assert found_active


@pytest.mark.anyio
async def test_unified_provider_streaming_mock():
    """Verify UnifiedLLMProvider correctly handles on_token callback."""
    from annolid.core.agent.providers.unified_provider import UnifiedLLMProvider

    provider = UnifiedLLMProvider(default_model="test")

    class _FakeCompletions:
        async def create(self, **kwargs):  # noqa: ANN003
            assert kwargs["stream"] is True

            async def _stream():
                class _Chunk:
                    def __init__(self, content, reasoning=None, finish_reason=None):
                        self.choices = [MagicMock()]
                        self.choices[0].delta = MagicMock()
                        self.choices[0].delta.content = content
                        self.choices[0].delta.reasoning_content = reasoning
                        self.choices[0].finish_reason = finish_reason

                yield _Chunk(None, "Thinking")
                yield _Chunk("Hello")
                yield _Chunk(" World")
                yield _Chunk(None, finish_reason="stop")

            return _stream()

    class _FakeClient:
        def __init__(self):
            self.chat = MagicMock(completions=_FakeCompletions())

    provider._ensure_openai_client = lambda: _FakeClient()  # type: ignore[method-assign]

    tokens = []

    def on_token(t):
        tokens.append(t)

    await provider.chat(messages=[], on_token=on_token)

    assert "<think>Thinking</think>" in tokens
    assert "Hello" in tokens
    assert " World" in tokens
