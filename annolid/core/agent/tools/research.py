from __future__ import annotations

import asyncio
import json
from typing import Any, Mapping, Sequence

from annolid.core.agent.swarm_budget import resolve_swarm_turn_budget
from .function_base import FunctionTool
from annolid.services.literature_search import LiteratureResult, search_literature

assess_paper_draft_quality = None
build_research_preflight_packet = None
run_paper_drafting_swarm = None


def _ensure_paper_writer_services() -> str:
    global assess_paper_draft_quality
    global build_research_preflight_packet
    global run_paper_drafting_swarm

    try:
        from annolid.services import paper_writer

        if assess_paper_draft_quality is None:
            assess_paper_draft_quality = paper_writer.assess_paper_draft_quality
        if build_research_preflight_packet is None:
            build_research_preflight_packet = (
                paper_writer.build_research_preflight_packet
            )
        if run_paper_drafting_swarm is None:
            run_paper_drafting_swarm = paper_writer.run_paper_drafting_swarm
    except Exception as exc:  # noqa: BLE001
        return str(exc)
    return ""


def _coerce_literature_rows(rows: Sequence[object]) -> list[LiteratureResult]:
    papers: list[LiteratureResult] = []
    for row in rows:
        if isinstance(row, LiteratureResult):
            papers.append(row)
        elif isinstance(row, Mapping):
            paper = LiteratureResult.from_dict(row)
            if paper.title:
                papers.append(paper)
        else:
            title = str(row or "").strip()
            if title:
                papers.append(LiteratureResult(source="user", title=title))
    return papers


async def _search_literature_results(
    query: str,
    *,
    max_results: int = 8,
) -> list[LiteratureResult]:
    loop = asyncio.get_running_loop()
    payload = await loop.run_in_executor(
        None,
        lambda: search_literature(query, max_results=max_results),
    )
    raw = payload.get("results", [])
    rows = (
        raw if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else []
    )
    return _coerce_literature_rows(list(rows))


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


class ResearchPacketTool(FunctionTool):
    @property
    def name(self) -> str:
        return "build_research_packet"

    @property
    def description(self) -> str:
        return (
            "Build a grounded research preflight packet for paper writing, "
            "including literature, novelty risk, rubric, and drafting guardrails."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "minLength": 1,
                    "description": "The research topic or manuscript idea.",
                },
                "idea_summary": {
                    "type": "string",
                    "description": "Optional fuller hypothesis or contribution summary.",
                },
                "related_work": {
                    "type": "array",
                    "items": {},
                    "description": (
                        "Optional already-collected papers as strings or result objects. "
                        "If omitted, the tool searches literature."
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 30,
                    "description": "Maximum literature results to search when related_work is omitted.",
                },
            },
            "required": ["topic"],
        }

    async def execute(
        self,
        topic: str,
        idea_summary: str = "",
        related_work: Sequence[object] | None = None,
        max_results: int = 8,
        **kwargs: Any,
    ) -> str:
        del kwargs
        service_error = _ensure_paper_writer_services()
        if build_research_preflight_packet is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": "paper_writer service is unavailable.",
                    "detail": service_error,
                },
                ensure_ascii=False,
            )
        try:
            papers = _coerce_literature_rows(list(related_work or []))
            search_error = ""
            if not papers:
                try:
                    papers = await _search_literature_results(
                        topic,
                        max_results=max(1, int(max_results or 8)),
                    )
                except Exception as exc:  # noqa: BLE001
                    search_error = str(exc)
            packet = build_research_preflight_packet(
                topic,
                papers,
                idea_summary=idea_summary,
                max_literature_items=max(1, int(max_results or 8)),
            )
            packet["ok"] = True
            if search_error:
                packet["literature_search_error"] = search_error
            return json.dumps(packet, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
            )


class PaperDraftQualityTool(FunctionTool):
    @property
    def name(self) -> str:
        return "assess_paper_draft"

    @property
    def description(self) -> str:
        return (
            "Assess a paper draft for placeholder text, missing manuscript "
            "sections, citation coverage, numeric-claim risk, and review rubric."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paper_text": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Markdown or plain-text paper draft to assess.",
                },
                "references": {
                    "type": "array",
                    "items": {},
                    "description": "Optional references used by the draft.",
                },
            },
            "required": ["paper_text"],
        }

    async def execute(
        self,
        paper_text: str,
        references: Sequence[object] | None = None,
        **kwargs: Any,
    ) -> str:
        del kwargs
        service_error = _ensure_paper_writer_services()
        if assess_paper_draft_quality is None:
            return json.dumps(
                {
                    "ok": False,
                    "error": "paper_writer service is unavailable.",
                    "detail": service_error,
                },
                ensure_ascii=False,
            )
        try:
            result = assess_paper_draft_quality(
                paper_text,
                references=list(references or []),
            )
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
            )


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
                "max_literature_results": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 30,
                    "description": (
                        "Maximum papers to search before drafting. Use 0 to skip search."
                    ),
                },
            },
            "required": ["topic"],
        }

    async def execute(
        self,
        topic: str,
        max_literature_results: int = 8,
        **kwargs: Any,
    ) -> str:
        del kwargs
        service_error = _ensure_paper_writer_services()
        if self._loop_factory is None:
            return "Error: DraftPaperSwarmTool requires a loop_factory to be provided by the environment."
        if run_paper_drafting_swarm is None:
            detail = f" Detail: {service_error}" if service_error else ""
            return f"Error: paper_writer module is not correctly installed or imported.{detail}"

        try:
            papers: list[LiteratureResult] = []
            preflight_context = ""
            preflight_note = ""
            if max_literature_results > 0:
                try:
                    papers = await _search_literature_results(
                        topic,
                        max_results=max_literature_results,
                    )
                except Exception as exc:  # noqa: BLE001
                    preflight_note = f"\n\nLiterature search degraded: {exc}"

            if build_research_preflight_packet is not None:
                try:
                    packet = build_research_preflight_packet(
                        topic,
                        papers,
                        idea_summary=topic,
                        max_literature_items=max(1, max_literature_results or 1),
                    )
                    preflight_context = str(packet.get("prompt_context") or "")
                    novelty = packet.get("novelty")
                    if isinstance(novelty, Mapping):
                        preflight_note += (
                            "\n\nResearch preflight: "
                            f"recommendation={novelty.get('recommendation', '')}; "
                            f"coverage={novelty.get('coverage_quality', '')}."
                        )
                except Exception as exc:  # noqa: BLE001
                    preflight_note += f"\n\nResearch preflight degraded: {exc}"

            result = await run_paper_drafting_swarm(
                topic=topic,
                papers=papers,
                loop_factory=self._loop_factory,
                max_turns=resolve_swarm_turn_budget(
                    topic,
                    8,
                    paper_context=True,
                    agent_count=3,
                ),
                research_context=preflight_context,
            )
            literature_note = (
                f"Grounded with {len(papers)} literature reference(s)."
                if papers
                else "No literature references were available before drafting."
            )
            return (
                "Paper drafting swarm completed successfully.\n"
                f"{literature_note}{preflight_note}\n\n{result}"
            )
        except Exception as exc:
            return f"Failed to draft paper via swarm: {exc}"


__all__ = [
    "DraftPaperSwarmTool",
    "LiteratureSearchTool",
    "PaperDraftQualityTool",
    "ResearchPacketTool",
]
