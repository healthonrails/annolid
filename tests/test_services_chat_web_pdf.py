from __future__ import annotations

import asyncio

from annolid.services.chat_web_pdf import (
    capture_chat_web_screenshot,
    click_chat_web,
    describe_chat_web_view,
    extract_chat_first_web_url,
    extract_chat_web_structured,
    find_chat_pdf_sections,
    find_chat_web_forms,
    get_chat_pdf_state,
    get_chat_pdf_text,
    get_chat_web_dom_text,
    get_chat_web_state,
    open_chat_in_browser_tool,
    open_chat_pdf_tool,
    open_chat_url_tool,
    run_chat_web_steps,
    scroll_chat_web,
    type_chat_web,
)


def test_chat_web_pdf_wrappers(monkeypatch) -> None:
    import annolid.services.chat_web_pdf as web_pdf_mod

    monkeypatch.setattr(
        web_pdf_mod,
        "gui_extract_first_web_url",
        lambda text, extract_web_urls: f"url:{text}",
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_open_in_browser_tool",
        lambda url, **kwargs: {"ok": True, "url": url},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_web_get_dom_text",
        lambda **kwargs: {"ok": True, "kind": "dom", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_web_get_state",
        lambda **kwargs: {"ok": True, "kind": "state", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_web_capture_screenshot",
        lambda **kwargs: {"ok": True, "kind": "shot", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_web_describe_view",
        lambda **kwargs: {"ok": True, "kind": "describe", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_web_extract_structured",
        lambda **kwargs: {"ok": True, "kind": "structured", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_web_click",
        lambda **kwargs: {"ok": True, "kind": "click", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_web_type",
        lambda **kwargs: {"ok": True, "kind": "type", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_web_scroll",
        lambda **kwargs: {"ok": True, "kind": "scroll", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_web_find_forms",
        lambda **kwargs: {"ok": True, "kind": "forms", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_pdf_get_state",
        lambda **kwargs: {"ok": True, "kind": "pdf_state", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_pdf_get_text",
        lambda **kwargs: {"ok": True, "kind": "pdf_text", **kwargs},
    )
    monkeypatch.setattr(
        web_pdf_mod,
        "gui_pdf_find_sections",
        lambda **kwargs: {"ok": True, "kind": "pdf_sections", **kwargs},
    )

    async def _open_url(url, **kwargs):
        return {"ok": True, "kind": "open_url", "url": url, **kwargs}

    async def _open_pdf(path="", **kwargs):
        return {"ok": True, "kind": "open_pdf", "path": path, **kwargs}

    async def _run_steps(**kwargs):
        return {"ok": True, "kind": "run_steps", **kwargs}

    monkeypatch.setattr(web_pdf_mod, "gui_open_url_tool", _open_url)
    monkeypatch.setattr(web_pdf_mod, "gui_open_pdf_tool", _open_pdf)
    monkeypatch.setattr(web_pdf_mod, "gui_web_run_steps", _run_steps)

    assert (
        extract_chat_first_web_url("example.com", extract_web_urls=lambda _t: [])
        == "url:example.com"
    )
    assert (
        open_chat_in_browser_tool("https://example.com")["url"] == "https://example.com"
    )
    assert get_chat_web_dom_text(invoke_widget_json_slot="slot")["kind"] == "dom"
    assert get_chat_web_state(invoke_widget_json_slot="slot")["kind"] == "state"
    assert capture_chat_web_screenshot(invoke_widget_json_slot="slot")["kind"] == "shot"
    assert describe_chat_web_view(invoke_widget_json_slot="slot")["kind"] == "describe"
    assert (
        extract_chat_web_structured(get_state=lambda: {}, get_dom_text=lambda _n: {})[
            "kind"
        ]
        == "structured"
    )
    assert (
        click_chat_web(invoke_widget_json_slot="slot", selector="#a")["kind"] == "click"
    )
    assert (
        type_chat_web(invoke_widget_json_slot="slot", selector="#a", text="hi")["kind"]
        == "type"
    )
    assert (
        scroll_chat_web(invoke_widget_json_slot="slot", delta_y=10)["kind"] == "scroll"
    )
    assert find_chat_web_forms(invoke_widget_json_slot="slot")["kind"] == "forms"
    assert get_chat_pdf_state(invoke_widget_json_slot="slot")["kind"] == "pdf_state"
    assert get_chat_pdf_text(invoke_widget_json_slot="slot")["kind"] == "pdf_text"
    assert (
        find_chat_pdf_sections(invoke_widget_json_slot="slot")["kind"] == "pdf_sections"
    )
    assert asyncio.run(open_chat_url_tool("https://example.com"))["kind"] == "open_url"
    assert asyncio.run(open_chat_pdf_tool("/tmp/a.pdf"))["kind"] == "open_pdf"
    assert asyncio.run(run_chat_web_steps(steps=[]))["kind"] == "run_steps"
