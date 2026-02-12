from __future__ import annotations

from annolid.gui.widgets.ai_chat_backend import StreamingChatTask


def test_normalize_messages_for_ollama_handles_openai_multimodal_content() -> None:
    raw = [
        {"role": "system", "content": "rules"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe image"},
                {
                    "type": "image_url",
                    "image_url": {"url": "data:image/png;base64,aGVsbG8="},
                },
            ],
        },
    ]

    out = StreamingChatTask._normalize_messages_for_ollama(raw)
    assert out[0]["role"] == "system"
    assert out[0]["content"] == "rules"
    assert out[1]["role"] == "user"
    assert out[1]["content"] == "describe image"
    assert isinstance(out[1]["images"], list)
    assert out[1]["images"][0] == b"hello"


def test_extract_ollama_text_prefers_content_then_thinking_then_response() -> None:
    r1 = {"message": {"content": "hello"}}
    assert StreamingChatTask._extract_ollama_text(r1) == "hello"

    r2 = {"message": {"content": [{"type": "text", "text": "hi"}]}}
    assert StreamingChatTask._extract_ollama_text(r2) == "hi"

    r3 = {"message": {"content": "", "thinking": "draft answer"}}
    assert StreamingChatTask._extract_ollama_text(r3) == "draft answer"

    r4 = {"message": {"content": ""}, "response": "fallback text"}
    assert StreamingChatTask._extract_ollama_text(r4) == "fallback text"
