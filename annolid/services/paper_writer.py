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

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Callable, Sequence

from annolid.services.literature_search import LiteratureResult

_LOGGER = logging.getLogger(__name__)

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
# Core generation functions
# ---------------------------------------------------------------------------

LLMCall = Callable[[str, str], str]  # (system_prompt, user_prompt) -> text


def generate_outline(
    topic: str,
    papers: Sequence[LiteratureResult],
    llm_call: LLMCall,
) -> list[dict[str, str]]:
    """Ask the LLM for a paper section outline.

    Parameters
    ----------
    topic:
        The research topic / hypothesis.
    papers:
        Literature results to ground the outline.
    llm_call:
        ``callable(system_prompt, user_prompt) -> str``  — the LLM gateway.

    Returns
    -------
    list[dict[str, str]]
        Each dict has ``"title"`` and ``"guidance"`` keys.
        Falls back to a hard-coded five-section outline on parse failure.
    """
    paper_list = _format_paper_list(papers)
    user_msg = _OUTLINE_USER_TPL.format(topic=topic, paper_list=paper_list)
    try:
        raw = llm_call(_OUTLINE_SYSTEM, user_msg)
    except Exception as exc:
        _LOGGER.warning("generate_outline: LLM call failed: %s", exc)
        return _default_outline(topic)

    # Strip markdown fences if present
    cleaned = re.sub(r"```[a-z]*\n?", "", str(raw or "")).strip()
    # Find the JSON array
    match = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if match:
        cleaned = match.group(0)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            result: list[dict[str, str]] = []
            for item in parsed:
                if isinstance(item, dict):
                    title = str(item.get("title") or "").strip()
                    guidance = str(item.get("guidance") or "").strip()
                    if title:
                        result.append({"title": title, "guidance": guidance})
            if result:
                return result
    except (json.JSONDecodeError, ValueError) as exc:
        _LOGGER.warning(
            "generate_outline: JSON parse failed (%s); using default outline.", exc
        )
    return _default_outline(topic)


def _default_outline(topic: str) -> list[dict[str, str]]:
    """Fallback five-section outline."""
    _ = topic  # reserved for future personalisation
    return [
        {
            "title": "Introduction",
            "guidance": "Motivate the problem and state contributions.",
        },
        {
            "title": "Related Work",
            "guidance": "Survey relevant prior work and highlight gaps.",
        },
        {"title": "Method", "guidance": "Describe the proposed approach in detail."},
        {
            "title": "Experiments",
            "guidance": "Present experimental setup and quantitative results.",
        },
        {
            "title": "Conclusion",
            "guidance": "Summarize findings and discuss future directions.",
        },
    ]


def draft_section(
    section_title: str,
    guidance: str,
    topic: str,
    papers: Sequence[LiteratureResult],
    llm_call: LLMCall,
) -> str:
    """Ask the LLM to write one paper section.

    Returns the written section text (plain markdown, no heading).
    Returns an empty string on failure (caller should handle gracefully).
    """
    paper_list = _format_paper_list(papers)
    user_msg = _SECTION_USER_TPL.format(
        topic=topic,
        section_title=section_title,
        guidance=guidance,
        paper_list=paper_list,
    )
    try:
        text = llm_call(_SECTION_SYSTEM, user_msg)
        return str(text or "").strip()
    except Exception as exc:
        _LOGGER.warning("draft_section(%r): LLM call failed: %s", section_title, exc)
        return ""


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
    "LLMCall",
    "PaperDraft",
    "PaperSection",
    "build_bibtex",
    "draft_section",
    "format_draft_markdown",
    "generate_outline",
]
