from __future__ import annotations

import sys
from pathlib import Path

from annolid.core.agent.gui_backend.heuristics import looks_like_pdf_summary_request
from annolid.core.agent.gui_backend.pdf_summary import summarize_active_pdf_with_cache


def test_looks_like_pdf_summary_request_accepts_summary_typos() -> None:
    assert looks_like_pdf_summary_request("summarzie this paper")
    assert looks_like_pdf_summary_request("please summarise this pdf")


def test_summarize_active_pdf_with_cache_emits_telemetry_and_reuses_cache(
    monkeypatch, tmp_path: Path
) -> None:
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 test")
    telemetry_events: list[dict[str, object]] = []
    fitz_open_calls = {"count": 0}

    class _Page:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, _mode: str) -> str:
            return self._text

    class _Doc:
        page_count = 2

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            del exc_type, exc, tb

        def load_page(self, index: int) -> _Page:
            return _Page(
                f"Abstract. Page {index + 1} introduces finetuning-free behavioral understanding."
            )

    class _FitzModule:
        @staticmethod
        def open(path: str) -> _Doc:
            assert path == str(pdf_path)
            fitz_open_calls["count"] += 1
            return _Doc()

    monkeypatch.setitem(sys.modules, "fitz", _FitzModule)

    def _state() -> dict[str, object]:
        return {
            "ok": True,
            "has_pdf": True,
            "path": str(pdf_path),
            "title": "paper.pdf",
        }

    summary_1 = summarize_active_pdf_with_cache(
        workspace=tmp_path,
        get_pdf_state=_state,
        build_summary=lambda text, **_: f"summary({len(text)})",
        on_telemetry=lambda payload: telemetry_events.append(dict(payload)),
    )
    assert "Summary of the open PDF (paper.pdf):" in summary_1
    assert "Cached extraction:" in summary_1
    assert fitz_open_calls["count"] == 1
    assert telemetry_events[-1]["ok"] is True
    assert telemetry_events[-1]["cache_hit"] is False

    summary_2 = summarize_active_pdf_with_cache(
        workspace=tmp_path,
        get_pdf_state=_state,
        build_summary=lambda text, **_: f"summary({len(text)})",
        on_telemetry=lambda payload: telemetry_events.append(dict(payload)),
    )
    assert "Summary of the open PDF (paper.pdf):" in summary_2
    assert fitz_open_calls["count"] == 1
    assert telemetry_events[-1]["ok"] is True
    assert telemetry_events[-1]["cache_hit"] is True
