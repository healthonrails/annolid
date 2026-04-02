from __future__ import annotations

from annolid.core.agent.gui_backend.heuristics import (
    looks_like_pdf_read_promise,
    looks_like_pdf_summary_request,
    looks_like_web_lookup_promise,
    prompt_may_need_mcp,
)


def test_web_lookup_promise_is_bounded_and_keyword_gated() -> None:
    assert looks_like_web_lookup_promise("I'll check the weather skill first.")
    assert not looks_like_web_lookup_promise("I'll check the weather skill first." * 20)
    assert not looks_like_web_lookup_promise("I will review the situation.")
    assert not looks_like_web_lookup_promise("The skillful analysis is complete.")


def test_pdf_heuristics_use_shared_phrase_matching() -> None:
    assert looks_like_pdf_read_promise("Let me read the PDF page.")
    assert looks_like_pdf_summary_request("Please summarize this pdf document.")
    assert not looks_like_pdf_read_promise("Let me read this." * 50)
    assert not looks_like_pdf_summary_request("Please summarize this note.")


def test_prompt_may_need_mcp_covers_web_and_browser_intents() -> None:
    assert prompt_may_need_mcp("open website in browser")
    assert prompt_may_need_mcp("what is the weather today?")
    assert not prompt_may_need_mcp("write a short note")
    assert not prompt_may_need_mcp("search")
