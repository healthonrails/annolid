"""Service helpers for GUI chat arXiv and local PDF actions."""

from __future__ import annotations

from annolid.core.agent.gui_backend.tool_handlers_arxiv import (
    arxiv_search_tool,
    list_local_pdfs,
    safe_run_arxiv_search,
)

__all__ = [
    "arxiv_search_tool",
    "list_local_pdfs",
    "safe_run_arxiv_search",
]
