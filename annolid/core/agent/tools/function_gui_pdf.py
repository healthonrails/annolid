from __future__ import annotations

from typing import Any, Optional

from .function_gui_base import ActionCallback, _run_callback
from .function_base import FunctionTool


class GuiOpenPdfTool(FunctionTool):
    def __init__(self, open_pdf_callback: Optional[ActionCallback] = None):
        self._open_pdf_callback = open_pdf_callback

    @property
    def name(self) -> str:
        return "gui_open_pdf"

    @property
    def description(self) -> str:
        return (
            "Open a PDF in Annolid using the same workflow as File > Open PDF... "
            "If path is provided, open that file directly without prompting."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._open_pdf_callback, **kwargs)


class GuiPdfGetStateTool(FunctionTool):
    def __init__(self, pdf_get_state_callback: Optional[ActionCallback] = None):
        self._pdf_get_state_callback = pdf_get_state_callback

    @property
    def name(self) -> str:
        return "gui_pdf_get_state"

    @property
    def description(self) -> str:
        return "Get state of the currently opened PDF in Annolid."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}, "required": []}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        return await _run_callback(self._pdf_get_state_callback)


class GuiPdfGetTextTool(FunctionTool):
    def __init__(self, pdf_get_text_callback: Optional[ActionCallback] = None):
        self._pdf_get_text_callback = pdf_get_text_callback

    @property
    def name(self) -> str:
        return "gui_pdf_get_text"

    @property
    def description(self) -> str:
        return (
            "Read text from the currently opened PDF in Annolid, starting from the "
            "current page."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_chars": {"type": "integer", "minimum": 200},
                "pages": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._pdf_get_text_callback, **kwargs)


class GuiPdfFindSectionsTool(FunctionTool):
    def __init__(self, pdf_find_sections_callback: Optional[ActionCallback] = None):
        self._pdf_find_sections_callback = pdf_find_sections_callback

    @property
    def name(self) -> str:
        return "gui_pdf_find_sections"

    @property
    def description(self) -> str:
        return (
            "Detect likely section headings in the currently opened PDF and return "
            "their page numbers."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "max_sections": {"type": "integer", "minimum": 1, "maximum": 200},
                "max_pages": {"type": "integer", "minimum": 1, "maximum": 100},
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._pdf_find_sections_callback, **kwargs)


class GuiArxivSearchTool(FunctionTool):
    def __init__(self, arxiv_search_callback: Optional[ActionCallback] = None):
        self._arxiv_search_callback = arxiv_search_callback

    @property
    def name(self) -> str:
        return "gui_arxiv_search"

    @property
    def description(self) -> str:
        return "Search arXiv for papers, download the best match, and open it in Annolid's PDF viewer."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 5,
                    "default": 1,
                },
            },
            "required": ["query"],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._arxiv_search_callback, **kwargs)


class GuiListPdfsTool(FunctionTool):
    def __init__(self, list_pdfs_callback: Optional[ActionCallback] = None):
        self._list_pdfs_callback = list_pdfs_callback

    @property
    def name(self) -> str:
        return "gui_list_pdfs"

    @property
    def description(self) -> str:
        return (
            "List all local PDF files available in the workspace downloads or "
            "other accessible directories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Optional search query to filter PDF files by name.",
                }
            },
            "required": [],
        }

    async def execute(self, **kwargs: Any) -> str:
        return await _run_callback(self._list_pdfs_callback, **kwargs)
