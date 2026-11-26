from __future__ import annotations

import asyncio
import inspect
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from annolid.core.agent.swarm_budget import resolve_swarm_turn_budget
from annolid.services.literature_search import LiteratureResult, search_literature
from annolid.utils.atomic_io import atomic_write_text

from .common import (
    _normalize_allowed_read_roots,
    _resolve_read_path,
    _resolve_write_path,
)
from .function_base import FunctionTool

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


def _sanitize_export_stem(base_name: str, *, default: str = "paper_draft") -> str:
    raw = str(base_name or "").strip()
    if not raw:
        return default
    if "/" in raw or "\\" in raw or ".." in raw:
        raise ValueError(
            "report_basename must be a plain filename stem without path separators or '..'."
        )
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._")
    if not cleaned:
        raise ValueError("report_basename must contain alphanumeric characters.")
    return cleaned


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
                venue = str(row.get("venue") or "").strip()
                doi = str(row.get("doi") or "").strip()
                arxiv_id = str(row.get("arxiv_id") or "").strip()
                ids = ", ".join(
                    part
                    for part in [
                        f"doi:{doi}" if doi else "",
                        f"arXiv:{arxiv_id}" if arxiv_id else "",
                    ]
                    if part
                )
                summary = str(row.get("summary") or "").replace("\n", " ").strip()
                url = (
                    row.get("pdf_url") or row.get("abs_url") or row.get("id_url") or ""
                )
                entry_lines = [f"{i}. Title: {title} ({year})"]
                entry_lines.append(f"   Authors: {authors}")
                if venue:
                    entry_lines.append(f"   Venue: {venue}")
                if ids:
                    entry_lines.append(f"   IDs: {ids}")
                entry_lines.append(f"   URL: {url}")
                entry_lines.append(f"   Abstract: {summary}")
                lines.append("\n".join(entry_lines) + "\n")
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
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir,
            allowed_read_roots,
        )

    @property
    def name(self) -> str:
        return "assess_paper_draft"

    @property
    def description(self) -> str:
        return (
            "Assess a paper draft (inline Markdown/LaTeX or file path) for placeholder "
            "text, missing manuscript sections, citation coverage, unresolved keys, "
            "quantitative-claim risk, and review rubric."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "paper_text": {
                    "type": "string",
                    "description": "Markdown or LaTeX paper draft text to assess.",
                },
                "paper_path": {
                    "type": "string",
                    "description": "Optional path to a .md or .tex draft file to read and assess.",
                },
                "references": {
                    "type": "array",
                    "items": {},
                    "description": "Optional references used by the draft.",
                },
                "bib_path": {
                    "type": "string",
                    "description": "Optional path to a BibTeX (.bib) file to cross-check inline citation keys.",
                },
            },
        }

    async def execute(
        self,
        paper_text: str = "",
        references: Sequence[object] | None = None,
        paper_path: str = "",
        bib_path: str = "",
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
            text_to_assess = str(paper_text or "")
            if not text_to_assess.strip() and str(paper_path or "").strip():
                resolved_paper = _resolve_read_path(
                    paper_path,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                )
                text_to_assess = resolved_paper.read_text(encoding="utf-8")
            if not text_to_assess.strip():
                return json.dumps(
                    {
                        "ok": False,
                        "error": "Provide paper_text or paper_path to assess.",
                    },
                    ensure_ascii=False,
                )

            ref_items: list[object] = list(references or [])
            if str(bib_path or "").strip():
                from annolid.utils.citations import load_bibtex

                resolved_bib = _resolve_read_path(
                    bib_path,
                    allowed_dir=self._allowed_dir,
                    allowed_read_roots=self._allowed_read_roots,
                )
                for entry in load_bibtex(resolved_bib):
                    ref_items.append(
                        {
                            "key": entry.key,
                            "title": entry.fields.get("title") or entry.key,
                            "author": entry.fields.get("author") or "",
                            "year": entry.fields.get("year") or "",
                            "doi": entry.fields.get("doi") or "",
                            "venue": entry.fields.get("journal")
                            or entry.fields.get("booktitle")
                            or "",
                            "source": "bibtex",
                        }
                    )

            result = assess_paper_draft_quality(
                text_to_assess,
                references=ref_items,
            )
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as exc:
            return json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
            )


class DraftPaperSwarmTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._loop_factory = None
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir,
            allowed_read_roots,
        )

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
                "idea_summary": {
                    "type": "string",
                    "description": "Optional hypothesis or contribution summary for novelty preflight.",
                },
                "related_work": {
                    "type": "array",
                    "items": {},
                    "description": "Optional pre-collected references to ground the draft.",
                },
                "max_literature_results": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 30,
                    "description": (
                        "Maximum papers to search before drafting. Use 0 to skip search."
                    ),
                },
                "output_dir": {
                    "type": "string",
                    "description": (
                        "Optional directory to export paper_draft.md, .bib, and quality JSON "
                        "(requires allow_mutation=true)."
                    ),
                },
                "report_basename": {
                    "type": "string",
                    "description": "Base filename stem for exported draft artifacts (default paper_draft).",
                },
                "allow_mutation": {
                    "type": "boolean",
                    "description": "Must be true to write exported files to output_dir.",
                },
            },
            "required": ["topic"],
        }

    async def execute(
        self,
        topic: str,
        max_literature_results: int = 8,
        idea_summary: str = "",
        related_work: Sequence[object] | None = None,
        output_dir: str = "",
        report_basename: str = "paper_draft",
        allow_mutation: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        service_error = _ensure_paper_writer_services()
        if self._loop_factory is None:
            return "Error: DraftPaperSwarmTool requires a loop_factory to be provided by the environment."
        if run_paper_drafting_swarm is None:
            detail = f" Detail: {service_error}" if service_error else ""
            return f"Error: paper_writer module is not correctly installed or imported.{detail}"

        if str(output_dir or "").strip() and not bool(allow_mutation):
            return (
                "Error: Writing exported draft files modifies disk state. "
                "Retry with allow_mutation=true only when file export is intended."
            )

        try:
            papers: list[LiteratureResult] = _coerce_literature_rows(
                list(related_work or [])
            )
            preflight_context = ""
            preflight_note = ""
            if not papers and max_literature_results > 0:
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
                        idea_summary=str(idea_summary or "").strip() or topic,
                        max_literature_items=max(
                            1, len(papers) or max_literature_results or 1
                        ),
                    )
                    preflight_context = str(packet.get("prompt_context") or "")
                    novelty = packet.get("novelty")
                    if isinstance(novelty, Mapping):
                        rec = str(novelty.get("recommendation") or "").strip().lower()
                        preflight_note += (
                            "\n\nResearch preflight: "
                            f"recommendation={rec}; "
                            f"coverage={novelty.get('coverage_quality', '')}."
                        )
                        if rec == "abort":
                            preflight_note += (
                                " WARNING: High overlap with existing literature detected; "
                                "differentiate claims carefully before submission."
                            )
                except Exception as exc:  # noqa: BLE001
                    preflight_note += f"\n\nResearch preflight degraded: {exc}"

            resolved_turns = resolve_swarm_turn_budget(
                topic,
                8,
                paper_context=True,
                agent_count=3,
            )
            run_kwargs: dict[str, Any] = {
                "topic": topic,
                "papers": papers,
                "loop_factory": self._loop_factory,
                "max_turns": resolved_turns,
                "research_context": preflight_context,
            }
            try:
                sig = inspect.signature(run_paper_drafting_swarm)
                if "resolve_budget" in sig.parameters or any(
                    p.kind == inspect.Parameter.VAR_KEYWORD
                    for p in sig.parameters.values()
                ):
                    run_kwargs["resolve_budget"] = False
            except Exception:
                pass

            result = await run_paper_drafting_swarm(**run_kwargs)

            from annolid.services.paper_writer import (
                build_bibtex,
                extract_final_draft_from_swarm_output,
            )

            extracted_draft = extract_final_draft_from_swarm_output(result)
            quality_note = ""
            quality_report: dict[str, object] | None = None
            if assess_paper_draft_quality is not None and extracted_draft:
                try:
                    quality_report = assess_paper_draft_quality(
                        extracted_draft,
                        references=papers,
                    )
                    metrics = quality_report.get("metrics")
                    m_map = metrics if isinstance(metrics, Mapping) else {}
                    quality_note = (
                        "\n\nDraft quality check: "
                        f"status={quality_report.get('quality_status', 'warn')}; "
                        f"words={m_map.get('word_count', 0)}; "
                        f"citations={m_map.get('citation_count', 0)}; "
                        f"placeholders={m_map.get('placeholder_count', 0)}."
                    )
                except Exception:
                    quality_report = None

            export_note = ""
            if str(output_dir or "").strip():
                resolved_out = _resolve_write_path(
                    output_dir, allowed_dir=self._allowed_dir
                )
                resolved_out.mkdir(parents=True, exist_ok=True)
                stem = _sanitize_export_stem(report_basename, default="paper_draft")
                md_path = resolved_out / f"{stem}.md"
                atomic_write_text(md_path, extracted_draft or result)
                written_paths = [str(md_path)]
                bib_text = build_bibtex(papers)
                if bib_text:
                    bib_path = resolved_out / f"{stem}.bib"
                    atomic_write_text(bib_path, bib_text)
                    written_paths.append(str(bib_path))
                if quality_report is not None:
                    q_path = resolved_out / f"{stem}_quality.json"
                    atomic_write_text(
                        q_path,
                        json.dumps(quality_report, indent=2, ensure_ascii=False),
                    )
                    written_paths.append(str(q_path))
                export_note = "\n\nExported draft artifacts: " + ", ".join(
                    written_paths
                )

            literature_note = (
                f"Grounded with {len(papers)} literature reference(s)."
                if papers
                else "No literature references were available before drafting."
            )
            return (
                "Paper drafting swarm completed successfully.\n"
                f"{literature_note}{preflight_note}{quality_note}{export_note}\n\n{result}"
            )
        except Exception as exc:
            return f"Failed to draft paper via swarm: {exc}"


__all__ = [
    "DraftPaperSwarmTool",
    "LiteratureSearchTool",
    "PaperDraftQualityTool",
    "ResearchPacketTool",
]
