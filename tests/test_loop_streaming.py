import asyncio
import pytest
from unittest.mock import MagicMock, AsyncMock
from annolid.core.agent.loop import AgentLoop


@pytest.mark.anyio
async def test_agent_loop_streaming_visualizer_updates():
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

    with MagicMock() as mock_update:
        import annolid.gui.widgets.threejs_viewer_server

        annolid.gui.widgets.threejs_viewer_server.update_swarm_node = mock_update

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
async def test_litellm_provider_streaming_mock():
    """Verify LiteLLMProvider correctly handles on_token callback."""
    from annolid.core.agent.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(default_model="test")

    # Mock litellm.acompletion
    with MagicMock() as mock_litellm:
        import sys

        sys.modules["litellm"] = mock_litellm

        async def mock_stream(*args, **kwargs):
            class Chunk:
                def __init__(self, content, reasoning=None):
                    self.choices = [MagicMock()]
                    self.choices[0].delta = MagicMock()
                    self.choices[0].delta.content = content
                    self.choices[0].delta.reasoning_content = reasoning
                    self.choices[0].finish_reason = None

            yield Chunk(None, "Thinking")
            yield Chunk("Hello")
            yield Chunk(" World")

            final_chunk = Chunk(None)
            final_chunk.choices[0].finish_reason = "stop"
            yield final_chunk

        mock_litellm.acompletion = AsyncMock(return_value=mock_stream())

        tokens = []

        def on_token(t):
            tokens.append(t)

        await provider.chat(messages=[], on_token=on_token)

        assert "<think>Thinking</think>" in tokens
        assert "Hello" in tokens
        assert " World" in tokens
