from __future__ import annotations

import asyncio
from typing import Any

from .function_base import FunctionTool
from annolid.services.literature_search import search_literature

try:
    from annolid.services.paper_writer import run_paper_drafting_swarm
except ImportError:
    run_paper_drafting_swarm = None


class LiteratureSearchTool(FunctionTool):
    @property
    def name(self) -> str:
        return "search_literature"

    @property
    def description(self) -> str:
        return "Search academic literature across OpenAlex, Crossref, and arXiv for papers on a given topic."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The research topic or keywords to search for.",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of papers to return (default 8).",
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        query: str,
        max_results: int = 8,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            # Run the synchronous search_literature in a thread pool
            loop = asyncio.get_running_loop()
            payload = await loop.run_in_executor(
                None, lambda: search_literature(query, max_results=max_results)
            )
            raw = payload.get("results", [])
            lines = []
            for i, row in enumerate(raw, 1):
                if not isinstance(row, dict):
                    continue
                title = str(row.get("title") or "")
                year = row.get("year") or ""
                authors = ", ".join(
                    str(a) for a in (row.get("authors") or []) if str(a).strip()
                )
                summary = str(row.get("summary") or "").replace("\n", " ")
                url = (
                    row.get("pdf_url") or row.get("abs_url") or row.get("id_url") or ""
                )
                lines.append(
                    f"{i}. Title: {title} ({year})\n"
                    f"   Authors: {authors}\n"
                    f"   URL: {url}\n"
                    f"   Abstract: {summary}\n"
                )
            if not lines:
                return "No literature found for that query."
            return "\n".join(lines)
        except Exception as exc:
            return f"Error executing literature search: {exc}"


class DraftPaperSwarmTool(FunctionTool):
    def __init__(self) -> None:
        self._loop_factory = None

    def set_loop_factory(self, factory: Any) -> None:
        self._loop_factory = factory

    @property
    def name(self) -> str:
        return "draft_paper_swarm"

    @property
    def description(self) -> str:
        return "Draft a full academic paper by launching a collaborative swarm of specialized agents (Outliner, Writer, Reviewer)."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "The target research topic.",
                },
            },
            "required": ["topic"],
        }

    async def execute(
        self,
        topic: str,
        **kwargs: Any,
    ) -> str:
        del kwargs
        if self._loop_factory is None:
            return "Error: DraftPaperSwarmTool requires a loop_factory to be provided by the environment."
        if run_paper_drafting_swarm is None:
            return "Error: paper_writer module is not correctly installed or imported."

        try:
            # We don't have paper LiteratureResult objects readily here without doing a search,
            # but the swarm will operate with empty literature sequence and can rely on the topic
            # and its intrinsic knowledge for now, or the user can do a literature search first.
            result = await run_paper_drafting_swarm(
                topic=topic, papers=[], loop_factory=self._loop_factory, max_turns=6
            )
            return f"Paper drafting swarm completed successfully:\n\n{result}"
        except Exception as exc:
            return f"Failed to draft paper via swarm: {exc}"


__all__ = ["LiteratureSearchTool", "DraftPaperSwarmTool"]
