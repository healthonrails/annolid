from __future__ import annotations

import asyncio
import importlib
import importlib.abc
import json
import sys
import types
import urllib.request

from annolid.core.agent.providers.transcription import OpenAICompatTranscriptionProvider


class _BlockHttpxFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None):
        del path, target
        if fullname == "httpx" or fullname.startswith("httpx."):
            raise ModuleNotFoundError("No module named 'httpx'", name="httpx")
        return None


def test_transcription_provider_returns_empty_without_api_key(tmp_path) -> None:
    provider = OpenAICompatTranscriptionProvider(api_key="")
    path = tmp_path / "audio.wav"
    path.write_bytes(b"RIFF")
    text = asyncio.run(provider.transcribe(path))
    assert text == ""


def test_providers_and_gui_backend_import_cleanly_without_httpx(monkeypatch) -> None:
    finder = _BlockHttpxFinder()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    for mod_name in [
        name
        for name in list(sys.modules)
        if name == "httpx" or name.startswith("httpx.")
    ]:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    transcription_mod = importlib.reload(
        importlib.import_module("annolid.core.agent.providers.transcription")
    )
    providers_mod = importlib.reload(
        importlib.import_module("annolid.core.agent.providers")
    )
    gui_backend_mod = importlib.reload(
        importlib.import_module("annolid.core.agent.gui_backend")
    )

    assert hasattr(transcription_mod, "OpenAICompatTranscriptionProvider")
    assert hasattr(providers_mod, "OpenAICompatTranscriptionProvider")
    assert hasattr(gui_backend_mod, "parse_direct_gui_command")


def test_transcription_provider_uses_urllib_fallback_without_httpx(
    tmp_path, monkeypatch
) -> None:
    finder = _BlockHttpxFinder()
    monkeypatch.setattr(sys, "meta_path", [finder, *sys.meta_path])
    for mod_name in [
        name
        for name in list(sys.modules)
        if name == "httpx" or name.startswith("httpx.")
    ]:
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    captured: dict[str, object] = {}

    class _FakeHttpResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def read(self) -> bytes:
            return json.dumps({"text": "mouse grooming bout"}).encode("utf-8")

    def _fake_urlopen(request: urllib.request.Request, timeout: float = 0.0):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["headers"] = dict(request.header_items())
        captured["body"] = bytes(request.data or b"")
        captured["timeout"] = timeout
        return _FakeHttpResponse()

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)

    audio_path = tmp_path / "clip.wav"
    audio_path.write_bytes(b"RIFF\x00\x00\x00\x00WAVE")
    provider = OpenAICompatTranscriptionProvider(
        api_key="sk-test",
        api_base="https://api.example.org/v1/",
        model="whisper-large-v3",
    )
    result = asyncio.run(provider.transcribe(audio_path))

    assert result == "mouse grooming bout"
    assert captured["url"] == "https://api.example.org/v1/audio/transcriptions"
    assert captured["method"] == "POST"
    headers = {k.lower(): v for k, v in dict(captured["headers"]).items()}
    assert headers["authorization"] == "Bearer sk-test"
    assert headers["content-type"].startswith("multipart/form-data; boundary=")
    body = bytes(captured["body"])
    assert b'name="model"' in body
    assert b"whisper-large-v3" in body
    assert b'filename="clip.wav"' in body
    assert b"RIFF\x00\x00\x00\x00WAVE" in body


def test_transcription_provider_uses_httpx_when_available(
    tmp_path, monkeypatch
) -> None:
    captured: dict[str, object] = {}

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, str]:
            return {"text": "rearing detected"}

    class _FakeAsyncClient:
        def __init__(self, *, timeout: float = 0.0) -> None:
            captured["timeout"] = timeout

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def post(self, url: str, *, headers=None, files=None):
            captured["url"] = url
            captured["headers"] = dict(headers or {})
            captured["model"] = files["model"]
            captured["filename"] = files["file"][0]
            captured["bytes"] = files["file"][1].read()
            return _FakeResponse()

    monkeypatch.setitem(
        sys.modules, "httpx", types.SimpleNamespace(AsyncClient=_FakeAsyncClient)
    )

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"RIFFDATA")
    provider = OpenAICompatTranscriptionProvider(
        api_key="sk-httpx",
        api_base="https://api.example.org/v1",
        model="whisper-1",
    )
    result = asyncio.run(provider.transcribe(audio_path))

    assert result == "rearing detected"
    assert captured["url"] == "https://api.example.org/v1/audio/transcriptions"
    assert captured["headers"] == {"Authorization": "Bearer sk-httpx"}
    assert captured["model"] == (None, "whisper-1")
    assert captured["filename"] == "sample.wav"
    assert captured["bytes"] == b"RIFFDATA"
