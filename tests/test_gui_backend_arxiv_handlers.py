from __future__ import annotations

import asyncio
import json
import urllib.error

import pytest

import annolid.core.agent.gui_backend.tool_handlers_arxiv as arxiv_handlers


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        return None

    def read(self) -> bytes:
        return self._payload


def test_fetch_arxiv_feed_bytes_retries_on_429(monkeypatch) -> None:
    arxiv_handlers._reset_arxiv_circuit_breaker()
    calls = {"n": 0}
    slept: list[float] = []

    def _fake_urlopen(req, timeout=0):  # type: ignore[no-untyped-def]
        del timeout
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                getattr(req, "full_url", str(req)),
                429,
                "rate limit",
                {"Retry-After": "2"},
                None,
            )
        return _FakeResponse(b"<feed />")

    monkeypatch.setattr(arxiv_handlers.urllib.request, "urlopen", _fake_urlopen)
    monkeypatch.setattr(
        arxiv_handlers.time, "sleep", lambda value: slept.append(float(value))
    )
    progress: list[str] = []

    payload = arxiv_handlers._fetch_arxiv_feed_bytes(
        url="https://export.arxiv.org/api/query?search_query=all:test&start=0&max_results=1",
        emit_progress=progress.append,
    )

    assert payload == b"<feed />"
    assert calls["n"] == 2
    assert slept and slept[0] >= 2.0
    assert any("429" in msg for msg in progress)


def test_fetch_arxiv_feed_bytes_circuit_breaker_blocks_after_failures(
    monkeypatch,
) -> None:
    arxiv_handlers._reset_arxiv_circuit_breaker()
    calls = {"n": 0}

    def _always_429(req, timeout=0):  # type: ignore[no-untyped-def]
        del timeout
        calls["n"] += 1
        raise urllib.error.HTTPError(
            getattr(req, "full_url", str(req)),
            429,
            "rate limit",
            {"Retry-After": "0"},
            None,
        )

    monkeypatch.setattr(arxiv_handlers.urllib.request, "urlopen", _always_429)
    monkeypatch.setattr(arxiv_handlers.time, "sleep", lambda _: None)

    with pytest.raises(RuntimeError):
        arxiv_handlers._fetch_arxiv_feed_bytes(
            url="https://export.arxiv.org/api/query?search_query=all:test&start=0&max_results=1",
            max_attempts=3,
        )
    assert calls["n"] == 3

    with pytest.raises(RuntimeError, match="temporarily unavailable"):
        arxiv_handlers._fetch_arxiv_feed_bytes(
            url="https://export.arxiv.org/api/query?search_query=all:test&start=0&max_results=1",
            max_attempts=1,
        )
    assert calls["n"] == 3


def test_arxiv_search_tool_uses_https_pdf_fallback(monkeypatch, tmp_path) -> None:
    arxiv_handlers._reset_arxiv_circuit_breaker()
    captured: dict[str, str] = {}

    monkeypatch.setattr(
        arxiv_handlers,
        "search_literature",
        lambda *_, **__: {
            "results": [
                {
                    "source": "arxiv",
                    "title": "Sample Paper",
                    "arxiv_id": "2602.17594v1",
                    "pdf_url": "",
                }
            ],
            "source_counts": {"arxiv": 1},
            "degraded_sources": [],
            "cache_hit": False,
        },
    )

    class _FakeDownloader:
        def __init__(self, allowed_dir):  # type: ignore[no-untyped-def]
            del allowed_dir

        async def execute(self, *, url, output_path, overwrite):  # type: ignore[no-untyped-def]
            del overwrite
            captured["download_url"] = str(url)
            return json.dumps({"output_path": str(output_path)})

    monkeypatch.setattr(arxiv_handlers, "DownloadPdfTool", _FakeDownloader)

    async def _open_pdf(path: str) -> dict[str, object]:
        captured["open_path"] = path
        return {"ok": True}

    progress: list[str] = []
    result = asyncio.run(
        arxiv_handlers.arxiv_search_tool(
            query="id:2602.17594",
            max_results=1,
            workspace=tmp_path,
            emit_progress=progress.append,
            open_pdf=_open_pdf,
        )
    )

    assert result["ok"] is True
    assert captured["download_url"] == "https://arxiv.org/pdf/2602.17594v1.pdf"
    assert captured["open_path"].endswith(".pdf")
    assert result["search"]["source_counts"] == {"arxiv": 1}
