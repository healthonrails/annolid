from __future__ import annotations

from annolid.core.agent.providers.transcription import OpenAICompatTranscriptionProvider


def test_transcription_provider_returns_empty_without_api_key(tmp_path) -> None:
    provider = OpenAICompatTranscriptionProvider(api_key="")
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF")
    text = __import__("asyncio").run(provider.transcribe(path))
    assert text == ""
