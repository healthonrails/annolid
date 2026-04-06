from __future__ import annotations

from typing import Any, Optional

from .function_gui_base import ActionCallback, _run_callback
from .function_base import FunctionTool


class GuiOpenUrlTool(FunctionTool):
    def __init__(self, open_url_callback: Optional[ActionCallback] = None):
        self._open_url_callback = open_url_callback

    @property
    def name(self) -> str:
        return "gui_open_url"

    @property
    def description(self) -> str:
        return (
            "Open a web URL or local file path in Annolid's embedded canvas web "
            "viewer. Accepts full URLs, domains like google.com, and local "
            "markdown/html files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "minLength": 1}},
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_url_callback, **kwargs)


class GuiOpenInBrowserTool(FunctionTool):
    def __init__(self, open_in_browser_callback: Optional[ActionCallback] = None):
        self._open_in_browser_callback = open_in_browser_callback

    @property
    def name(self) -> str:
        return "gui_open_in_browser"

    @property
    def description(self) -> str:
        return (
            "Open a web URL in the system browser instead of the embedded Annolid "
            "web viewer."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"url": {"type": "string", "minLength": 1}},
            "required": ["url"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_in_browser_callback, **kwargs)


class GuiWebGetDomTextTool(FunctionTool):
    def __init__(self, web_get_dom_text_callback: Optional[ActionCallback] = None):
        self._web_get_dom_text_callback = web_get_dom_text_callback

    @property
    def name(self) -> str:
        return "gui_web_get_dom_text"

    @property
    def description(self) -> str:
        return "Read visible text content from the active embedded web page."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"max_chars": {"type": "integer", "minimum": 200}},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_get_dom_text_callback, **kwargs)


class GuiWebCaptureScreenshotTool(FunctionTool):
    def __init__(
        self, web_capture_screenshot_callback: Optional[ActionCallback] = None
    ):
        self._web_capture_screenshot_callback = web_capture_screenshot_callback

    @property
    def name(self) -> str:
        return "gui_web_capture_screenshot"

    @property
    def description(self) -> str:
        return (
            "Capture a screenshot of the active embedded web page and save it to "
            "the Annolid workspace."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_width": {"type": "integer", "minimum": 320, "maximum": 4096}
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_capture_screenshot_callback, **kwargs)


class GuiWebDescribeViewTool(FunctionTool):
    def __init__(self, web_describe_view_callback: Optional[ActionCallback] = None):
        self._web_describe_view_callback = web_describe_view_callback

    @property
    def name(self) -> str:
        return "gui_web_describe_view"

    @property
    def description(self) -> str:
        return (
            "Capture the active embedded web page as an image and ask Annolid Bot "
            "vision to describe it."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_width": {"type": "integer", "minimum": 320, "maximum": 4096}
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_describe_view_callback, **kwargs)


class GuiWebClickTool(FunctionTool):
    def __init__(self, web_click_callback: Optional[ActionCallback] = None):
        self._web_click_callback = web_click_callback

    @property
    def name(self) -> str:
        return "gui_web_click"

    @property
    def description(self) -> str:
        return "Click an element in the embedded web page by CSS selector or index tag."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"selector": {"type": "string", "minLength": 1}},
            "required": ["selector"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_click_callback, **kwargs)


class GuiWebTypeTool(FunctionTool):
    def __init__(self, web_type_callback: Optional[ActionCallback] = None):
        self._web_type_callback = web_type_callback

    @property
    def name(self) -> str:
        return "gui_web_type"

    @property
    def description(self) -> str:
        return (
            "Type text into an input-like element in embedded web page by CSS selector "
            "or index tag."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "selector": {"type": "string", "minLength": 1},
                "text": {"type": "string"},
                "submit": {"type": "boolean"},
            },
            "required": ["selector", "text"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_type_callback, **kwargs)


class GuiWebScrollTool(FunctionTool):
    def __init__(self, web_scroll_callback: Optional[ActionCallback] = None):
        self._web_scroll_callback = web_scroll_callback

    @property
    def name(self) -> str:
        return "gui_web_scroll"

    @property
    def description(self) -> str:
        return "Scroll the active embedded web page by delta pixels."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"delta_y": {"type": "integer"}},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_scroll_callback, **kwargs)


class GuiWebFindFormsTool(FunctionTool):
    def __init__(self, web_find_forms_callback: Optional[ActionCallback] = None):
        self._web_find_forms_callback = web_find_forms_callback

    @property
    def name(self) -> str:
        return "gui_web_find_forms"

    @property
    def description(self) -> str:
        return (
            "List forms and input fields on the active embedded web page for "
            "automation planning."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._web_find_forms_callback)


class GuiWebRunStepsTool(FunctionTool):
    def __init__(self, web_run_steps_callback: Optional[ActionCallback] = None):
        self._web_run_steps_callback = web_run_steps_callback

    @property
    def name(self) -> str:
        return "gui_web_run_steps"

    @property
    def description(self) -> str:
        return (
            "Run a sequence of embedded web actions for browser automation. "
            "Supported actions: open_url, open_in_browser, get_text, click, type, "
            "scroll, find_forms, capture_screenshot, describe_view, wait."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "action": {"type": "string", "minLength": 1},
                            "url": {"type": "string"},
                            "selector": {"type": "string"},
                            "text": {"type": "string"},
                            "submit": {"type": "boolean"},
                            "delta_y": {"type": "integer"},
                            "max_chars": {"type": "integer", "minimum": 200},
                            "max_width": {
                                "type": "integer",
                                "minimum": 320,
                                "maximum": 4096,
                            },
                            "wait_ms": {"type": "integer", "minimum": 0},
                        },
                        "required": ["action"],
                    },
                    "minItems": 1,
                },
                "stop_on_error": {"type": "boolean"},
                "max_steps": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["steps"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_run_steps_callback, **kwargs)


class GuiWebExtractStructuredTool(FunctionTool):
    def __init__(
        self, web_extract_structured_callback: Optional[ActionCallback] = None
    ):
        self._web_extract_structured_callback = web_extract_structured_callback

    @property
    def name(self) -> str:
        return "gui_web_extract_structured"

    @property
    def description(self) -> str:
        return (
            "Extract structured fields from the active embedded web page text. "
            "Useful for titles, summaries, weather facts, prices, and source URLs."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                },
                "regex_overrides": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "selector_hints": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                },
                "extraction_mode": {
                    "type": "string",
                    "enum": ["auto", "regex", "hint"],
                },
                "max_chars": {"type": "integer", "minimum": 200, "maximum": 50000},
                "include_excerpt": {"type": "boolean"},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._web_extract_structured_callback, **kwargs)
