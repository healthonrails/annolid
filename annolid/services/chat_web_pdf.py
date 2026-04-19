"""Service helpers for GUI chat web and PDF actions."""

from __future__ import annotations

from typing import Any

from annolid.core.agent.gui_backend.tool_handlers_openers import (
    extract_first_web_url as gui_extract_first_web_url,
    open_in_browser_tool as gui_open_in_browser_tool,
    open_pdf_tool as gui_open_pdf_tool,
    open_url_tool as gui_open_url_tool,
)
from annolid.core.agent.gui_backend.tool_handlers_web_pdf import (
    pdf_find_sections as gui_pdf_find_sections,
    pdf_get_state as gui_pdf_get_state,
    pdf_get_text as gui_pdf_get_text,
    web_capture_screenshot as gui_web_capture_screenshot,
    web_click as gui_web_click,
    web_describe_view as gui_web_describe_view,
    web_extract_structured as gui_web_extract_structured,
    web_find_forms as gui_web_find_forms,
    web_get_dom_text as gui_web_get_dom_text,
    web_save_current as gui_web_save_current,
    web_get_state as gui_web_get_state,
    web_run_steps as gui_web_run_steps,
    web_scroll as gui_web_scroll,
    web_type as gui_web_type,
)


def extract_chat_first_web_url(text: str, *, extract_web_urls: Any) -> str:
    return gui_extract_first_web_url(text, extract_web_urls=extract_web_urls)


async def open_chat_url_tool(url: str, **kwargs: Any) -> dict[str, object]:
    return await gui_open_url_tool(url, **kwargs)


def open_chat_in_browser_tool(url: str, **kwargs: Any) -> dict[str, object]:
    return gui_open_in_browser_tool(url, **kwargs)


async def open_chat_pdf_tool(path: str = "", **kwargs: Any) -> dict[str, object]:
    return await gui_open_pdf_tool(path, **kwargs)


def get_chat_web_dom_text(**kwargs: Any) -> dict[str, Any]:
    return gui_web_get_dom_text(**kwargs)


def get_chat_web_state(**kwargs: Any) -> dict[str, Any]:
    return gui_web_get_state(**kwargs)


def capture_chat_web_screenshot(**kwargs: Any) -> dict[str, Any]:
    return gui_web_capture_screenshot(**kwargs)


def describe_chat_web_view(**kwargs: Any) -> dict[str, Any]:
    return gui_web_describe_view(**kwargs)


def extract_chat_web_structured(**kwargs: Any) -> dict[str, Any]:
    return gui_web_extract_structured(**kwargs)


def click_chat_web(**kwargs: Any) -> dict[str, Any]:
    return gui_web_click(**kwargs)


def type_chat_web(**kwargs: Any) -> dict[str, Any]:
    return gui_web_type(**kwargs)


def scroll_chat_web(**kwargs: Any) -> dict[str, Any]:
    return gui_web_scroll(**kwargs)


def find_chat_web_forms(**kwargs: Any) -> dict[str, Any]:
    return gui_web_find_forms(**kwargs)


def save_chat_web_current(**kwargs: Any) -> dict[str, Any]:
    return gui_web_save_current(**kwargs)


async def run_chat_web_steps(**kwargs: Any) -> dict[str, Any]:
    return await gui_web_run_steps(**kwargs)


def get_chat_pdf_state(**kwargs: Any) -> dict[str, Any]:
    return gui_pdf_get_state(**kwargs)


def get_chat_pdf_text(**kwargs: Any) -> dict[str, Any]:
    return gui_pdf_get_text(**kwargs)


def find_chat_pdf_sections(**kwargs: Any) -> dict[str, Any]:
    return gui_pdf_find_sections(**kwargs)


__all__ = [
    "capture_chat_web_screenshot",
    "click_chat_web",
    "describe_chat_web_view",
    "extract_chat_first_web_url",
    "extract_chat_web_structured",
    "find_chat_pdf_sections",
    "find_chat_web_forms",
    "get_chat_pdf_state",
    "get_chat_pdf_text",
    "get_chat_web_dom_text",
    "get_chat_web_state",
    "open_chat_in_browser_tool",
    "open_chat_pdf_tool",
    "open_chat_url_tool",
    "run_chat_web_steps",
    "save_chat_web_current",
    "scroll_chat_web",
    "type_chat_web",
]
