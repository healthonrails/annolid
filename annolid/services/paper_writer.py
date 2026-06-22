"""Research paper drafting service for Annolid.

Provides lightweight, LLM-provider-agnostic helpers for generating structured
academic paper drafts grounded in literature search results.

The service intentionally has no Qt or network dependencies — the caller
supplies an ``llm_call`` callable so it can be used from GUI or CLI contexts.

Public API
----------
- ``PaperSection`` — dataclass for one paper section.
- ``PaperDraft``   — dataclass for the complete draft.
- ``generate_outline``         — ask the LLM for a section outline JSON.
- ``draft_section``            — ask the LLM to write one section.
- ``build_bibtex``             — emit BibTeX from ``LiteratureResult`` objects.
- ``format_draft_markdown``    — stitch sections + references into markdown.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from annolid.core.agent.swarm_budget import resolve_swarm_turn_budget
from annolid.services.literature_search import LiteratureResult
from annolid.services.novelty import novelty_preflight_check
from annolid.core.agent.swarm import SwarmAgent, SwarmManager

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class PaperSection:
    """One section of a research paper draft."""

    title: str
    content: str = ""
    guidance: str = ""  # LLM hint used during drafting


@dataclass
class PaperDraft:
    """A fully drafted research paper."""

    topic: str
    sections: list[PaperSection] = field(default_factory=list)
    references: list[LiteratureResult] = field(default_factory=list)

    # Running status used by the widget progress label.
    status: str = "idle"  # idle | drafting | done | error


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

_OUTLINE_SYSTEM = """\
You are an expert academic writer. Given a research topic and a list of
relevant papers, produce a JSON outline for a complete research paper.

Return ONLY a valid JSON array — no prose, no markdown fences.
Each element must be an object with two string fields:
  "title"    — the section heading (e.g. "Introduction", "Related Work", ...)
  "guidance" — one sentence describing what this section should cover.

Target 5–7 sections appropriate for a venue like NeurIPS or CVPR.
"""

_OUTLINE_USER_TPL = """\
Research topic: {topic}

Relevant papers (use these to shape the outline):
{paper_list}

Return the JSON array now.
"""

_SECTION_SYSTEM = """\
You are an expert academic author. Write a single section of a research paper
in flowing academic prose. Target 350–600 words.

Rules:
- Do NOT add section headings yourself; the caller will insert them.
- Cite papers using their title in parentheses when appropriate — do NOT
  invent DOIs or arXiv IDs.
- Do NOT include a references list.
- Write in plain markdown (bold/italic/lists are fine; no LaTeX math).
"""

_SECTION_USER_TPL = """\
Paper topic: {topic}
Section to write: {section_title}
Guidance: {guidance}

Relevant papers to draw from:
{paper_list}

Write the section now.
"""

_PAPER_RUBRIC: tuple[dict[str, object], ...] = (
    {
        "id": "novelty",
        "name": "Novelty",
        "weight": 1.0,
        "criteria": (
            "The manuscript states a concrete gap and differentiates the "
            "contribution from adjacent prior work."
        ),
        "scale": "1=rehash, 3=incremental, 5=clear contribution, 7=strong, 10=field-shifting",
    },
    {
        "id": "rigor",
        "name": "Rigor",
        "weight": 1.0,
        "criteria": (
            "Methods, datasets, controls, metrics, and limitations are "
            "specific enough to audit or reproduce."
        ),
        "scale": "1=unsupported, 3=thin, 5=adequate, 7=thorough, 10=exemplary",
    },
    {
        "id": "evidence",
        "name": "Evidence",
        "weight": 1.0,
        "criteria": (
            "Every quantitative claim is tied to measured data, a cited "
            "source, or is explicitly marked as a hypothesis."
        ),
        "scale": "1=fabricated-looking, 3=weakly grounded, 5=mixed, 7=well grounded, 10=fully traceable",
    },
    {
        "id": "clarity",
        "name": "Clarity",
        "weight": 1.0,
        "criteria": "The paper has a coherent structure, clear paragraphs, and precise terminology.",
        "scale": "1=incoherent, 3=hard to follow, 5=adequate, 7=clear, 10=excellent",
    },
    {
        "id": "impact",
        "name": "Impact",
        "weight": 1.0,
        "criteria": (
            "The manuscript explains who can use the work, why it matters, "
            "and what workflow or scientific burden it reduces."
        ),
        "scale": "1=unclear, 3=limited, 5=moderate, 7=important, 10=transformative",
    },
)

_DRAFTING_GUARDRAILS: tuple[str, ...] = (
    "Use only the listed references unless additional literature is searched.",
    "Do not invent DOI, arXiv, PMID, journal, author, or year metadata.",
    "Mark unverified numerical results as TBD instead of fabricating values.",
    "Separate measured results, cited prior work, and hypotheses in the prose.",
    "Tie Annolid claims to inspectable artifacts such as annotations, masks, tracks, metrics, or saved reports.",
    "When novelty preflight recommends differentiation, state the specific boundary against related work before drafting claims.",
)

_WRITING_CHECKLIST: tuple[str, ...] = (
    "Abstract states objective, approach, key result, and implication.",
    "Introduction moves from field need to gap to concrete contribution.",
    "Methods include enough implementation and data detail for replication.",
    "Results cite figures or tables before interpretation.",
    "Discussion compares against literature, names limitations, and avoids overclaiming.",
    "References match the cited works and citation keys exactly.",
)

_PLACEHOLDER_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\[INSERT\s+.*?\]", re.IGNORECASE), "insert placeholder"),
    (re.compile(r"\[TODO\s*:?\s*.*?\]", re.IGNORECASE), "TODO placeholder"),
    (
        re.compile(r"\[PLACEHOLDER\s*:?\s*.*?\]", re.IGNORECASE),
        "explicit placeholder",
    ),
    (re.compile(r"lorem\s+ipsum", re.IGNORECASE), "lorem ipsum filler"),
    (
        re.compile(
            r"this\s+section\s+will\s+(describe|discuss|present|outline|explain)",
            re.IGNORECASE,
        ),
        "future-tense placeholder",
    ),
    (
        re.compile(
            r"add\s+(your|the)\s+(content|text|description)\s+here", re.IGNORECASE
        ),
        "add-content placeholder",
    ),
    (
        re.compile(r"replace\s+this\s+(text|content|section)", re.IGNORECASE),
        "replace-content placeholder",
    ),
)

_SECTION_ALIASES: Mapping[str, tuple[str, ...]] = {
    "abstract": ("abstract", "summary"),
    "introduction": ("introduction", "background"),
    "methods": ("method", "methods", "materials and methods", "approach"),
    "results": ("result", "results", "evaluation", "experiments"),
    "discussion": ("discussion", "limitations", "conclusion", "conclusions"),
    "references": ("references", "bibliography"),
}


def _format_paper_list(
    papers: Sequence[LiteratureResult], *, max_items: int = 12
) -> str:
    """Format a short numbered list of papers for use in prompts."""
    lines: list[str] = []
    for i, p in enumerate(papers[:max_items], 1):
        year = f" ({p.year})" if p.year else ""
        doi_or_arxiv = ""
        if p.doi:
            doi_or_arxiv = f" doi:{p.doi}"
        elif p.arxiv_id:
            doi_or_arxiv = f" arXiv:{p.arxiv_id}"
        summary_snip = (p.summary[:120].rstrip() + "…") if p.summary else ""
        lines.append(
            f"{i}. {p.title}{year}{doi_or_arxiv}"
            + (f"\n   {summary_snip}" if summary_snip else "")
        )
    return "\n".join(lines) if lines else "(no papers provided)"


def _paper_url(paper: LiteratureResult) -> str:
    return paper.pdf_url or paper.abs_url or paper.id_url or ""


def _coerce_literature_result(item: object) -> LiteratureResult | None:
    if isinstance(item, LiteratureResult):
        return item
    if isinstance(item, Mapping):
        result = LiteratureResult.from_dict(item)
        return result if result.title else None
    text = str(item or "").strip()
    if not text:
        return None
    return LiteratureResult(source="user", title=text)


def _compact_reference(paper: LiteratureResult) -> dict[str, object]:
    return {
        "title": paper.title,
        "year": paper.year,
        "source": paper.source,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "url": _paper_url(paper),
        "summary": paper.summary[:500].strip(),
    }


def _research_packet_next_steps(novelty: Mapping[str, object]) -> list[str]:
    recommendation = str(novelty.get("recommendation") or "").strip().lower()
    if recommendation == "abort":
        return [
            "Do not draft contribution claims yet; revise the idea or narrow the target gap.",
            "Review the highest-overlap papers and write an explicit differentiation paragraph.",
            "Search additional related work before committing to a manuscript outline.",
        ]
    if recommendation == "differentiate":
        return [
            "Draft a related-work paragraph that names the closest overlap and the boundary of the Annolid contribution.",
            "Avoid strong novelty language until the differentiation is stated.",
            "Add more literature if coverage quality is low.",
        ]
    return [
        "Proceed with an outline grounded in the collected literature.",
        "Keep numerical claims tied to verified Annolid artifacts or cited references.",
    ]


def build_research_preflight_packet(
    topic: str,
    papers: Sequence[object] = (),
    *,
    idea_summary: str = "",
    top_k: int = 5,
    max_literature_items: int = 12,
) -> dict[str, object]:
    """Build an offline-testable research packet before drafting a paper.

    The packet ports the useful AutoResearchClaw discipline into Annolid's
    smaller bot surface: novelty preflight, quality rubric, writing checklist,
    and explicit guardrails against fabricated citations or numbers.
    """
    topic_text = str(topic or "").strip()
    if not topic_text:
        raise ValueError("topic is required.")

    references = [
        result
        for result in (_coerce_literature_result(item) for item in papers)
        if result is not None
    ]
    related_work = [paper.to_dict() for paper in references]
    summary_text = str(idea_summary or "").strip() or topic_text
    novelty = novelty_preflight_check(
        idea_title=topic_text,
        idea_summary=summary_text,
        related_work=related_work,
        top_k=max(1, int(top_k or 5)),
    )
    packet: dict[str, object] = {
        "schema_version": "1.0",
        "topic": topic_text,
        "idea_summary": summary_text,
        "literature": [
            _compact_reference(paper)
            for paper in references[: max(1, int(max_literature_items or 12))]
        ],
        "novelty": novelty,
        "paper_rubric": [dict(item) for item in _PAPER_RUBRIC],
        "drafting_guardrails": list(_DRAFTING_GUARDRAILS),
        "writing_checklist": list(_WRITING_CHECKLIST),
        "recommended_next_steps": _research_packet_next_steps(novelty),
    }
    packet["prompt_context"] = format_research_packet_for_prompt(packet)
    return packet


def format_research_packet_for_prompt(packet: Mapping[str, object]) -> str:
    """Render a research packet as compact prompt context for a paper agent."""
    lines = [
        "RESEARCH PREFLIGHT PACKET",
        f"Topic: {packet.get('topic', '')}",
    ]
    novelty = packet.get("novelty")
    if isinstance(novelty, Mapping):
        lines.append(
            "Novelty: "
            f"recommendation={novelty.get('recommendation', '')}; "
            f"coverage={novelty.get('coverage_quality', '')}; "
            f"reason={novelty.get('reason', '')}"
        )

    literature = packet.get("literature")
    if isinstance(literature, Sequence) and not isinstance(literature, (str, bytes)):
        lines.append("References available for citation:")
        for idx, item in enumerate(literature[:8], 1):
            if not isinstance(item, Mapping):
                continue
            year = f" ({item.get('year')})" if item.get("year") else ""
            identifiers = ", ".join(
                str(value)
                for value in [
                    f"doi:{item.get('doi')}" if item.get("doi") else "",
                    f"arXiv:{item.get('arxiv_id')}" if item.get("arxiv_id") else "",
                    str(item.get("url") or ""),
                ]
                if value
            )
            suffix = f" [{identifiers}]" if identifiers else ""
            lines.append(f"{idx}. {item.get('title', '')}{year}{suffix}")
    else:
        lines.append("References available for citation: none")

    lines.append("Drafting guardrails:")
    for guardrail in packet.get("drafting_guardrails", []):
        lines.append(f"- {guardrail}")

    lines.append("Review rubric:")
    for item in packet.get("paper_rubric", []):
        if isinstance(item, Mapping):
            lines.append(f"- {item.get('name')}: {item.get('criteria')}")
    return "\n".join(lines)


def _extract_markdown_headings(text: str) -> set[str]:
    headings: set[str] = set()
    for match in re.finditer(r"^\s{0,3}#{1,6}\s+(.+?)\s*$", text, re.MULTILINE):
        heading = re.sub(r"[*_`]+", "", match.group(1)).strip().lower()
        if heading:
            headings.add(heading)
    return headings


def _find_placeholder_matches(text: str) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    lines = text.splitlines()
    for line_number, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped:
            continue
        for pattern, description in _PLACEHOLDER_PATTERNS:
            if pattern.search(stripped):
                matches.append(
                    {
                        "line": line_number,
                        "pattern": description,
                        "excerpt": stripped[:120],
                    }
                )
                break
    return matches


def assess_paper_draft_quality(
    paper_text: str,
    *,
    references: Sequence[object] = (),
    required_sections: Sequence[str] = (
        "abstract",
        "introduction",
        "methods",
        "results",
        "discussion",
        "references",
    ),
) -> dict[str, object]:
    """Run deterministic manuscript quality checks for Annolid bot drafts."""
    text = str(paper_text or "")
    words = re.findall(r"\b[\w'-]+\b", text)
    headings = _extract_markdown_headings(text)
    missing_sections: list[str] = []
    for section in required_sections:
        key = str(section or "").strip().lower()
        aliases = _SECTION_ALIASES.get(key, (key,))
        if not any(
            any(alias in heading for alias in aliases)
            for heading in headings
            if heading
        ):
            missing_sections.append(key)

    citation_count = len(
        re.findall(
            r"(?:\\cite\{[^}]+\}|\[[0-9,\s-]+\]|\([A-Z][^)]*?\b\d{4}[a-z]?\))",
            text,
        )
    )
    numeric_claim_count = len(
        re.findall(
            r"\b\d+(?:\.\d+)?\s*(?:%|percent|fps|frames?|mAP|IoU|F1|AUC|ms|s)?\b",
            text,
            re.IGNORECASE,
        )
    )
    placeholder_matches = _find_placeholder_matches(text)
    reference_count = len(
        [
            item
            for item in (_coerce_literature_result(ref) for ref in references)
            if item is not None
        ]
    )

    warnings: list[str] = []
    status = "pass"
    if placeholder_matches:
        status = "fail"
        warnings.append("Template or placeholder content remains in the draft.")
    if missing_sections:
        status = "warn" if status == "pass" else status
        warnings.append("Missing expected sections: " + ", ".join(missing_sections))
    if reference_count == 0:
        status = "warn" if status == "pass" else status
        warnings.append("No reference list was supplied for citation checks.")
    if citation_count == 0 and reference_count > 0:
        status = "warn" if status == "pass" else status
        warnings.append(
            "References were supplied but no in-text citations were detected."
        )
    if numeric_claim_count > 0 and "tbd" not in text.lower() and citation_count == 0:
        status = "warn" if status == "pass" else status
        warnings.append(
            "Numeric claims are present without detected citations or TBD markers."
        )

    return {
        "ok": True,
        "quality_status": status,
        "metrics": {
            "word_count": len(words),
            "heading_count": len(headings),
            "citation_count": citation_count,
            "reference_count": reference_count,
            "numeric_claim_count": numeric_claim_count,
            "placeholder_count": len(placeholder_matches),
        },
        "missing_sections": missing_sections,
        "placeholder_matches": placeholder_matches,
        "rubric": [dict(item) for item in _PAPER_RUBRIC],
        "warnings": warnings,
    }


def build_paper_swarm_prompt(
    topic: str,
    *,
    pdf_state: Mapping[str, Any] | None = None,
) -> str:
    topic_text = str(topic or "").strip()
    if not topic_text:
        topic_text = "the current research topic"

    pdf_title = ""
    pdf_path = ""
    if isinstance(pdf_state, Mapping):
        pdf_title = str(pdf_state.get("title") or "").strip()
        pdf_path = str(pdf_state.get("path") or "").strip()

    lines = [
        "Use the `draft_paper_swarm` tool to draft a complete research paper.",
        f"Topic: {topic_text}",
    ]
    if pdf_title or pdf_path:
        lines.append(
            "Active PDF context: "
            + ", ".join(
                value
                for value in [
                    f"title={pdf_title}" if pdf_title else "",
                    f"path={pdf_path}" if pdf_path else "",
                ]
                if value
            )
        )
    lines.extend(
        [
            "Start by grounding the draft in literature search results and any open PDF context.",
            "Produce a structured paper with outline, sections, citations, and a concise completion note.",
            "If the current context is insufficient, search literature first rather than drafting from guesswork.",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Swarm Generation
# ---------------------------------------------------------------------------


async def run_paper_drafting_swarm(
    topic: str,
    papers: Sequence[LiteratureResult],
    loop_factory: Callable[[], Any],
    max_turns: int = 8,
    research_context: str = "",
) -> str:
    """Draft a complete research paper by orchestrating subagents.

    Parameters
    ----------
    topic:
        The research topic / hypothesis.
    papers:
        Literature results to ground the paper.
    loop_factory:
        A callable that returns a fresh ``AgentLoop`` or ``_SupportsRun`` instance
        for the swarm agents to use.
    max_turns:
        Maximum number of iterative agent turns.

    Returns
    -------
    str
        The final paper draft or conversation transcript from the swarm.
    """
    manager = SwarmManager()

    outliner = SwarmAgent(
        name="Outliner",
        role="Literature Review and Planner",
        system_prompt=(
            "You are the Outliner Agent. Your job is to read the provided topic and literature, "
            "then propose a strong 5-7 section structure for the paper (e.g., Introduction, "
            "Related Work, Method, Results). Only propose the structure and provide short guidance "
            "for each section. Do not write the full paper."
        ),
        loop_factory=loop_factory,
    )

    writer = SwarmAgent(
        name="Writer",
        role="Academic Author",
        system_prompt=(
            "You are the Paper-Writing Agent. Your job is to write the paper sections according "
            "to the Outliner's proposed structure and the literature context. "
            "Write in flowing, formal academic prose. Cite papers using their titles."
        ),
        loop_factory=loop_factory,
    )

    reviewer = SwarmAgent(
        name="Reviewer",
        role="Peer Reviewer",
        system_prompt=(
            "You are the Reviewer Agent. Your job is to critique the written sections, "
            "point out weak arguments, missing citations, or poor transitions, and suggest improvements. "
            "If the drafted paper meets high academic standards, say 'TASK COMPLETE' exactly to finish the swarm."
        ),
        loop_factory=loop_factory,
    )

    manager.register_agent(outliner)
    manager.register_agent(writer)
    manager.register_agent(reviewer)

    paper_list = _format_paper_list(papers)
    initial_task = (
        f"Draft a comprehensive research paper on: {topic}\n\n"
        f"Relevant Literature to use:\n{paper_list}\n\n"
        "Outliner, please start by proposing the sections."
    )
    context_text = str(research_context or "").strip()
    if context_text:
        initial_task += (
            "\n\nResearch preflight context and guardrails:\n"
            f"{context_text}\n\n"
            "Follow these guardrails when planning, writing, and reviewing."
        )

    resolved_turns = resolve_swarm_turn_budget(
        initial_task,
        max_turns,
        agent_count=3,
        paper_context=True,
    )
    return await manager.run_swarm(initial_task, max_turns=resolved_turns)


# ---------------------------------------------------------------------------
# BibTeX helpers
# ---------------------------------------------------------------------------


def _slugify(text: str, max_len: int = 20) -> str:
    """Make a safe BibTeX key fragment from arbitrary text."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "")).strip("_")
    return slug[:max_len] if slug else "ref"


def _paper_bibtex_key(paper: LiteratureResult, index: int) -> str:
    """Derive a unique-ish BibTeX key for a paper."""
    if paper.doi:
        suffix = _slugify(paper.doi.split("/")[-1], 16)
        return f"doi_{suffix}"
    if paper.arxiv_id:
        return f"arxiv_{_slugify(paper.arxiv_id, 16)}"
    title_slug = _slugify(paper.title, 20)
    return f"ref{index}_{title_slug}" if title_slug else f"ref{index}"


def build_bibtex(papers: Sequence[LiteratureResult]) -> str:
    """Generate a BibTeX file string from a list of ``LiteratureResult`` objects.

    Produces ``@misc`` entries (conservative type that covers all sources).
    Empty sequence returns empty string.
    """
    if not papers:
        return ""
    entries: list[str] = []
    seen_keys: set[str] = set()
    for i, paper in enumerate(papers):
        key = _paper_bibtex_key(paper, i)
        # Ensure uniqueness by appending a counter suffix
        base_key = key
        counter = 1
        while key in seen_keys:
            key = f"{base_key}_{counter}"
            counter += 1
        seen_keys.add(key)

        fields: list[str] = []
        if paper.title:
            escaped_title = paper.title.replace("{", "\\{").replace("}", "\\}")
            fields.append(f"  title     = {{{escaped_title}}}")
        if paper.year:
            fields.append(f"  year      = {{{paper.year}}}")
        if paper.doi:
            fields.append(f"  doi       = {{{paper.doi}}}")
        if paper.arxiv_id:
            fields.append(f"  eprint    = {{{paper.arxiv_id}}}")
            fields.append("  archivePrefix = {arXiv}")
        url = paper.pdf_url or paper.abs_url or paper.id_url
        if url:
            fields.append(f"  url       = {{{url}}}")
        if paper.source:
            fields.append(f"  note      = {{Source: {paper.source}}}")

        body = ",\n".join(fields)
        entries.append(f"@misc{{{key},\n{body}\n}}")

    return "\n\n".join(entries) + "\n"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def format_draft_markdown(draft: PaperDraft) -> str:
    """Render a ``PaperDraft`` as a complete Markdown document.

    Structure:
      # <topic>  (H1 page title)
      ## <section title>  (H2 per section)
      <section content>
      ## References
      - numbered list of papers
    """
    lines: list[str] = [f"# {draft.topic}", ""]
    for section in draft.sections:
        lines.append(f"## {section.title}")
        lines.append("")
        if section.content:
            lines.append(section.content)
        else:
            lines.append("*(section not yet drafted)*")
        lines.append("")

    if draft.references:
        lines.append("## References")
        lines.append("")
        for i, ref in enumerate(draft.references, 1):
            year = f" ({ref.year})" if ref.year else ""
            url = ref.pdf_url or ref.abs_url or ref.id_url or ""
            link = f" — [{url}]({url})" if url else ""
            lines.append(f"{i}. {ref.title}{year}{link}")
        lines.append("")

    return "\n".join(lines)


__all__ = [
    "PaperDraft",
    "PaperSection",
    "assess_paper_draft_quality",
    "build_bibtex",
    "build_paper_swarm_prompt",
    "build_research_preflight_packet",
    "format_research_packet_for_prompt",
    "run_paper_drafting_swarm",
    "format_draft_markdown",
]
