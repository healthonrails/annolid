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

from dataclasses import dataclass, field
import re
from typing import Callable, Sequence, Any

from annolid.services.literature_search import LiteratureResult
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


# ---------------------------------------------------------------------------
# Core Swarm Generation
# ---------------------------------------------------------------------------


async def run_paper_drafting_swarm(
    topic: str,
    papers: Sequence[LiteratureResult],
    loop_factory: Callable[[], Any],
    max_turns: int = 6,
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

    return await manager.run_swarm(initial_task, max_turns=max_turns)


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
    "build_bibtex",
    "run_paper_drafting_swarm",
    "format_draft_markdown",
]
