"""Research paper drafting service for Annolid.

Provides lightweight, LLM-provider-agnostic helpers for generating, evaluating,
and exporting structured academic paper drafts grounded in literature search
results.

The service intentionally has no Qt, network, or eager agent-runtime
dependencies — the caller supplies an ``llm_call`` or ``loop_factory`` callable
so it can be used cleanly from GUI, CLI, or agent contexts.

Public API
----------
- ``PaperSection`` — dataclass for one paper section.
- ``PaperDraft``   — dataclass for the complete draft.
- ``assign_citation_keys`` — assign deterministic BibTeX/inline citation keys.
- ``generate_outline`` — ask an LLM for a structured section outline.
- ``draft_section`` — ask an LLM to write one grounded section.
- ``draft_paper`` — run the full outline + section-by-section drafting pipeline.
- ``build_research_preflight_packet`` — build novelty + rubric + guardrails packet.
- ``assess_paper_draft_quality`` — deterministic manuscript quality & citation audit.
- ``run_paper_drafting_swarm`` — collaborative Outliner/Writer/Reviewer swarm.
- ``extract_final_draft_from_swarm_output`` — extract manuscript from swarm log.
- ``build_bibtex`` — emit BibTeX from ``LiteratureResult`` objects.
- ``format_draft_markdown`` — stitch sections + references into Markdown.
- ``format_draft_latex`` — render a ``PaperDraft`` to LaTeX with ``\\cite{}`` keys.
- ``write_draft_bundle`` — atomically export ``.md``, ``.bib``, ``.tex``, and quality JSON.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence

from annolid.services.literature_search import LiteratureResult
from annolid.services.novelty import novelty_preflight_check
from annolid.utils.atomic_io import atomic_write_text

# Lazily resolved only when run_paper_drafting_swarm() is invoked, keeping
# annolid.services.paper_writer decoupled from annolid.core.agent at import
# time while preserving monkeypatch compatibility on SwarmManager/SwarmAgent.
SwarmManager: Any = None
SwarmAgent: Any = None


def _resolve_swarm_runtime() -> tuple[Any, Any, Callable[..., int]]:
    from annolid.core.agent.swarm import (
        SwarmAgent as _DefaultSwarmAgent,
        SwarmManager as _DefaultSwarmManager,
    )
    from annolid.core.agent.swarm_budget import resolve_swarm_turn_budget

    manager_cls = SwarmManager if SwarmManager is not None else _DefaultSwarmManager
    agent_cls = SwarmAgent if SwarmAgent is not None else _DefaultSwarmAgent
    return manager_cls, agent_cls, resolve_swarm_turn_budget


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
# Prompt builders & guardrails
# ---------------------------------------------------------------------------

_DEFAULT_OUTLINE: tuple[tuple[str, str], ...] = (
    (
        "Abstract",
        "Summarize the problem, proposed approach, key findings (or TBD markers), and practical significance.",
    ),
    (
        "Introduction",
        "Motivate the scientific need, identify the specific gap in prior work, and list concrete contributions.",
    ),
    (
        "Related Work",
        "Compare the closest prior methods using the supplied references and clarify where this work differs.",
    ),
    (
        "Methods",
        "Specify datasets, annotations, model/algorithm design, experimental controls, and evaluation protocol.",
    ),
    (
        "Results",
        "Report artifact-backed metrics and comparisons, marking unmeasured values explicitly as TBD.",
    ),
    (
        "Discussion and Limitations",
        "Interpret findings against related literature, state key limitations, and note future work.",
    ),
)

_OUTLINE_SYSTEM = """\
You are an expert academic writer. Given a research topic and a list of
relevant papers, produce a JSON outline for a complete research paper.

Return ONLY a valid JSON array — no prose, no markdown fences.
Each element must be an object with two string fields:
  "title"    — the section heading (e.g. "Abstract", "Introduction", "Related Work", "Methods", "Results", "Discussion")
  "guidance" — one sentence describing what this section should cover.

Target 5–7 sections appropriate for the target venue.
"""

_OUTLINE_USER_TPL = """\
Research topic: {topic}
Target venue: {venue}
{idea_block}{context_block}
Relevant papers (use these to shape the outline):
{paper_list}

Return the JSON array now.
"""

_SECTION_SYSTEM = """\
You are an expert academic author. Write a single section of a research paper
in clear, flowing academic prose. Target 300–600 words unless writing an Abstract
(150–250 words).

Rules:
- Do NOT add the top-level section heading yourself; the caller inserts it.
- Cite ONLY the supplied references using their exact bracketed citation keys
  (e.g., [@key]) — NEVER invent DOIs, arXiv IDs, authors, or years.
- Mark unmeasured numerical values explicitly as TBD instead of fabricating numbers.
- Do NOT include a standalone References section.
- Write clean Markdown prose (bold/italic/bullet lists/tables are fine).
"""

_SECTION_USER_TPL = """\
Paper topic: {topic}
Section to write: {section_title}
Guidance: {guidance}
{prior_block}{context_block}
Available references (cite with exact [@key]):
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
    (re.compile(r"\[FIXME\s*:?\s*.*?\]", re.IGNORECASE), "FIXME placeholder"),
    (
        re.compile(r"\[PLACEHOLDER\s*:?\s*.*?\]", re.IGNORECASE),
        "explicit placeholder",
    ),
    (
        re.compile(r"\*\(section not yet drafted\)\*", re.IGNORECASE),
        "undrafted section placeholder",
    ),
    (re.compile(r"\.\.\.\s*\[TRUNCATED\]\s*\.\.\.", re.IGNORECASE), "truncated output"),
    (re.compile(r"\bXXX\b"), "XXX placeholder"),
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
    "abstract": ("abstract", "executive summary"),
    "introduction": ("introduction", "background"),
    "related_work": ("related work", "literature review", "prior work"),
    "methods": (
        "method",
        "methods",
        "methodology",
        "materials and methods",
        "approach",
        "experimental setup",
    ),
    "results": ("result", "results", "evaluation", "experiments", "findings"),
    "discussion": ("discussion", "limitations", "conclusion", "conclusions"),
    "references": ("references", "bibliography", "works cited"),
}

_MD_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")
_LATEX_HEADING_RE = re.compile(
    r"^\s*\\(?:sub)*section\*?\{([^}]+)\}\s*$", re.IGNORECASE
)
_PANDOC_CITE_RE = re.compile(
    r"\[((?:@[A-Za-z0-9_.:-]+)(?:\s*[;,]\s*@[A-Za-z0-9_.:-]+)*)\]"
)
_LATEX_CITE_RE = re.compile(
    r"\\(?:cite|citep|citet|citealp|citealt|parencite|textcite|autocite)\*?\{([^}]+)\}"
)
_NUMERIC_CITE_RE = re.compile(r"\[(\d+(?:\s*[-,]\s*\d+)*)\]")
_AUTHOR_YEAR_CITE_RE = re.compile(r"\(([A-Z][^()]{1,90}?\b(?:19|20)\d{2}[a-z]?)\)")
_QUANT_CLAIM_RE = re.compile(
    r"(?:"
    r"\b\d+(?:\.\d+)?\s*(?:%|percent|fps|frames?|mAP|AP50|AP75|IoU|mIoU|PCK|F1|AUC|ROC|RMSE|MAE|MSE|ms|sec|seconds?|min|minutes?|hours?|Hz|kHz|px|pixels?|mm|cm|m|kg|g|mg|dB|GB|MB)\b"
    r"|\b(?:mAP|IoU|F1|AUC|PCK|precision|recall|accuracy|kappa|alpha|p|q|r|R2|n|N)\s*(?:=|==|<|>|<=|>=|≈|~|of|was|reached|achieved)\s*\d+(?:\.\d+)?"
    r"|\b\d+(?:\.\d+)?\s*(?:±|\+/-)\s*\d+(?:\.\d+)?"
    r")",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Reference formatting & BibTeX key helpers
# ---------------------------------------------------------------------------


def _slugify(text: str, max_len: int = 20) -> str:
    """Make a safe BibTeX key fragment from arbitrary text."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(text or "")).strip("_")
    return slug[:max_len] if slug else "ref"


def _paper_bibtex_key(paper: LiteratureResult, index: int) -> str:
    """Derive a deterministic citation key for a paper."""
    if paper.doi:
        suffix = _slugify(paper.doi.split("/")[-1], 16)
        return f"doi_{suffix}"
    if paper.arxiv_id:
        return f"arxiv_{_slugify(paper.arxiv_id, 16)}"
    if paper.authors:
        first_author = str(paper.authors[0] or "").strip()
        surname = (
            first_author.split(",")[0].strip()
            if "," in first_author
            else (first_author.split()[-1] if first_author.split() else "")
        )
        surname_slug = _slugify(surname, 12).lower()
        year_part = str(paper.year) if paper.year else ""
        title_word = _slugify(paper.title, 12).lower()
        if surname_slug and surname_slug != "ref":
            return f"{surname_slug}{year_part}_{title_word}".strip("_")
    title_slug = _slugify(paper.title, 20)
    return f"ref{index}_{title_slug}" if title_slug else f"ref{index}"


def assign_citation_keys(
    papers: Sequence[LiteratureResult],
) -> list[tuple[str, LiteratureResult]]:
    """Assign unique, deterministic BibTeX/inline citation keys to ``papers``."""
    assigned: list[tuple[str, LiteratureResult]] = []
    seen_keys: set[str] = set()
    for idx, paper in enumerate(papers):
        key = _paper_bibtex_key(paper, idx)
        base_key = key
        counter = 1
        while key.lower() in seen_keys:
            key = f"{base_key}_{counter}"
            counter += 1
        seen_keys.add(key.lower())
        assigned.append((key, paper))
    return assigned


def _format_authors_short(authors: Sequence[str], *, max_names: int = 3) -> str:
    clean = [str(a).strip() for a in authors if str(a).strip()]
    if not clean:
        return ""
    if len(clean) <= max_names:
        return ", ".join(clean)
    return f"{clean[0]} et al."


def _format_paper_list(
    papers: Sequence[LiteratureResult],
    *,
    max_items: int = 12,
    include_keys: bool = True,
) -> str:
    """Format a numbered reference list (with citation keys & authors) for prompts."""
    selected = list(papers[: max(1, int(max_items))])
    if not selected:
        return "(no papers provided)"
    keyed = assign_citation_keys(selected)
    lines: list[str] = []
    for i, (key, p) in enumerate(keyed, 1):
        key_tag = f"[@{key}] " if include_keys else ""
        authors_txt = _format_authors_short(p.authors)
        author_prefix = f"{authors_txt}. " if authors_txt else ""
        year = f" ({p.year})" if p.year else ""
        venue_part = f" — {p.venue}" if p.venue else ""
        doi_or_arxiv = ""
        if p.doi:
            doi_or_arxiv = f" doi:{p.doi}"
        elif p.arxiv_id:
            doi_or_arxiv = f" arXiv:{p.arxiv_id}"
        summary_snip = (p.summary[:180].rstrip() + "…") if p.summary else ""
        lines.append(
            f"{i}. {key_tag}{author_prefix}{p.title}{year}{venue_part}{doi_or_arxiv}"
            + (f"\n   {summary_snip}" if summary_snip else "")
        )
    return "\n".join(lines)


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


def _compact_reference(
    paper: LiteratureResult, *, citation_key: str = ""
) -> dict[str, object]:
    payload: dict[str, object] = {
        "title": paper.title,
        "authors": list(paper.authors),
        "venue": paper.venue,
        "year": paper.year,
        "source": paper.source,
        "doi": paper.doi,
        "arxiv_id": paper.arxiv_id,
        "url": _paper_url(paper),
        "summary": paper.summary[:500].strip(),
    }
    if citation_key:
        payload["citation_key"] = citation_key
    return payload


# ---------------------------------------------------------------------------
# Provider-agnostic outline & section drafting
# ---------------------------------------------------------------------------


def _parse_outline_json(raw_response: str) -> list[PaperSection]:
    """Parse an LLM JSON outline response, tolerating markdown fences and extra text."""
    text = str(raw_response or "").strip()
    if not text:
        return []
    candidates = [text]
    fenced = re.findall(r"```(?:json)?\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    candidates.extend(f.strip() for f in fenced if f.strip())
    arr_start = text.find("[")
    arr_end = text.rfind("]")
    if arr_start >= 0 and arr_end > arr_start:
        candidates.append(text[arr_start : arr_end + 1].strip())

    parsed: object = None
    for cand in candidates:
        try:
            obj = json.loads(cand)
        except Exception:
            continue
        if isinstance(obj, list):
            parsed = obj
            break
        if isinstance(obj, dict) and isinstance(obj.get("sections"), list):
            parsed = obj["sections"]
            break

    if not isinstance(parsed, list):
        return []

    sections: list[PaperSection] = []
    seen: set[str] = set()
    for item in parsed:
        if isinstance(item, Mapping):
            title = " ".join(str(item.get("title") or "").split()).strip()
            guidance = " ".join(str(item.get("guidance") or "").split()).strip()
        elif isinstance(item, str):
            title = " ".join(item.split()).strip()
            guidance = ""
        else:
            continue
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        sections.append(PaperSection(title=title, guidance=guidance))
    return sections


def _strip_duplicate_leading_heading(content: str, section_title: str) -> str:
    lines = str(content or "").strip().splitlines()
    if not lines:
        return ""
    first = lines[0].strip()
    match = _MD_HEADING_RE.match(first)
    if match:
        heading_text = re.sub(r"[*_`]+", "", match.group(2)).strip().lower()
        target = re.sub(r"[*_`]+", "", section_title).strip().lower()
        if heading_text == target or target in heading_text:
            lines = lines[1:]
    return "\n".join(lines).strip()


def generate_outline(
    topic: str,
    papers: Sequence[object] = (),
    *,
    llm_call: Callable[[str, str], str],
    idea_summary: str = "",
    venue: str = "NeurIPS / CVPR",
    research_context: str = "",
    fallback_on_error: bool = True,
) -> list[PaperSection]:
    """Ask ``llm_call(system_prompt, user_prompt)`` for a structured section outline."""
    topic_text = str(topic or "").strip()
    if not topic_text:
        raise ValueError("topic is required.")
    references = [
        r for r in (_coerce_literature_result(p) for p in papers) if r is not None
    ]
    idea_block = (
        f"Idea summary: {str(idea_summary).strip()}\n"
        if str(idea_summary or "").strip()
        else ""
    )
    ctx_clean = str(research_context or "").strip()
    context_block = f"\nResearch preflight context:\n{ctx_clean}\n" if ctx_clean else ""
    user_prompt = _OUTLINE_USER_TPL.format(
        topic=topic_text,
        venue=str(venue or "NeurIPS / CVPR").strip(),
        idea_block=idea_block,
        context_block=context_block,
        paper_list=_format_paper_list(references, include_keys=True),
    )
    try:
        raw = llm_call(_OUTLINE_SYSTEM, user_prompt)
    except Exception:
        if not fallback_on_error:
            raise
        raw = ""
    sections = _parse_outline_json(raw)
    if sections:
        return sections
    if not fallback_on_error:
        raise ValueError("LLM did not return a valid JSON outline.")
    return [PaperSection(title=t, guidance=g) for t, g in _DEFAULT_OUTLINE]


def draft_section(
    topic: str,
    section: PaperSection | str,
    papers: Sequence[object] = (),
    *,
    llm_call: Callable[[str, str], str],
    prior_sections: Sequence[PaperSection] = (),
    research_context: str = "",
) -> PaperSection:
    """Draft a single grounded ``PaperSection`` using ``llm_call(system, user)``."""
    topic_text = str(topic or "").strip()
    if not topic_text:
        raise ValueError("topic is required.")
    sec = (
        section
        if isinstance(section, PaperSection)
        else PaperSection(title=str(section or "").strip())
    )
    if not sec.title:
        raise ValueError("section.title is required.")
    references = [
        r for r in (_coerce_literature_result(p) for p in papers) if r is not None
    ]

    prior_lines: list[str] = []
    for prev in prior_sections:
        if prev.content.strip():
            snippet = " ".join(prev.content.split())[:220].rstrip()
            prior_lines.append(f"- {prev.title}: {snippet}")
    prior_block = (
        "\nPreviously drafted sections (maintain continuity, avoid repetition):\n"
        + "\n".join(prior_lines)
        + "\n"
        if prior_lines
        else ""
    )
    ctx_clean = str(research_context or "").strip()
    context_block = (
        f"\nResearch preflight guardrails:\n{ctx_clean}\n" if ctx_clean else ""
    )

    user_prompt = _SECTION_USER_TPL.format(
        topic=topic_text,
        section_title=sec.title,
        guidance=sec.guidance or f"Write a thorough, grounded {sec.title} section.",
        prior_block=prior_block,
        context_block=context_block,
        paper_list=_format_paper_list(references, include_keys=True),
    )
    raw = llm_call(_SECTION_SYSTEM, user_prompt)
    cleaned = _strip_duplicate_leading_heading(raw, sec.title)
    return PaperSection(title=sec.title, content=cleaned, guidance=sec.guidance)


def draft_paper(
    topic: str,
    papers: Sequence[object] = (),
    *,
    llm_call: Callable[[str, str], str],
    idea_summary: str = "",
    venue: str = "NeurIPS / CVPR",
    research_context: str = "",
    outline: Sequence[PaperSection] | None = None,
) -> PaperDraft:
    """Run the full provider-agnostic outline + section-by-section drafting flow."""
    topic_text = str(topic or "").strip()
    if not topic_text:
        raise ValueError("topic is required.")
    references = [
        r for r in (_coerce_literature_result(p) for p in papers) if r is not None
    ]
    ctx = str(research_context or "").strip()
    if not ctx:
        packet = build_research_preflight_packet(
            topic_text, references, idea_summary=idea_summary
        )
        ctx = str(packet.get("prompt_context") or "")

    draft = PaperDraft(topic=topic_text, references=references, status="drafting")
    try:
        planned = (
            list(outline)
            if outline
            else generate_outline(
                topic_text,
                references,
                llm_call=llm_call,
                idea_summary=idea_summary,
                venue=venue,
                research_context=ctx,
            )
        )
        completed: list[PaperSection] = []
        for sec in planned:
            written = draft_section(
                topic_text,
                sec,
                references,
                llm_call=llm_call,
                prior_sections=completed,
                research_context=ctx,
            )
            completed.append(written)
        draft.sections = completed
        draft.status = "done"
        return draft
    except Exception:
        draft.status = "error"
        raise


# ---------------------------------------------------------------------------
# Preflight packet & deterministic quality assessment
# ---------------------------------------------------------------------------


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

    Combines literature with assigned citation keys, novelty preflight,
    manuscript quality rubric, writing checklist, and anti-fabrication
    guardrails.
    """
    topic_text = str(topic or "").strip()
    if not topic_text:
        raise ValueError("topic is required.")

    references = [
        result
        for result in (_coerce_literature_result(item) for item in papers)
        if result is not None
    ]
    keyed_refs = assign_citation_keys(
        references[: max(1, int(max_literature_items or 12))]
    )
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
            _compact_reference(paper, citation_key=key) for key, paper in keyed_refs
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
            cite_key = str(item.get("citation_key") or "").strip()
            key_prefix = f"[@{cite_key}] " if cite_key else ""
            raw_authors = item.get("authors")
            author_str = (
                _format_authors_short(raw_authors)
                if isinstance(raw_authors, Sequence)
                and not isinstance(raw_authors, (str, bytes))
                else ""
            )
            author_prefix = f"{author_str}. " if author_str else ""
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
            lines.append(
                f"{idx}. {key_prefix}{author_prefix}{item.get('title', '')}{year}{suffix}"
            )
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


def _extract_sections_from_text(text: str) -> list[tuple[int, str, str]]:
    """Extract ordered ``(heading_level, normalized_heading, body_text)`` triples."""
    sections: list[tuple[int, str, list[str]]] = []
    current_level: int = 0
    current_heading: str | None = None
    current_lines: list[str] = []

    for line in str(text or "").splitlines():
        md_m = _MD_HEADING_RE.match(line)
        tex_m = _LATEX_HEADING_RE.match(line) if md_m is None else None
        if md_m or tex_m:
            if current_heading is not None:
                sections.append((current_level, current_heading, current_lines))
            if md_m:
                current_level = len(md_m.group(1))
                raw_heading = md_m.group(2)
            else:
                low_line = line.strip().lower()
                current_level = 3 if low_line.startswith("\\sub") else 2
                raw_heading = tex_m.group(1)  # type: ignore[union-attr]
            cleaned = re.sub(r"[*_`]+", "", raw_heading).strip()
            # Strip leading numbering like "1. " or "2.1 "
            cleaned = re.sub(
                r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+", "", cleaned
            ).strip()
            current_heading = cleaned.lower()
            current_lines = []
        elif current_heading is not None:
            current_lines.append(line)

    if current_heading is not None:
        sections.append((current_level, current_heading, current_lines))
    return [(lvl, h, "\n".join(body).strip()) for lvl, h, body in sections if h]


def _extract_manuscript_body_without_references(text: str) -> str:
    """Return manuscript text excluding the trailing References/Bibliography section."""
    body_lines: list[str] = []
    in_references = False
    ref_aliases = _SECTION_ALIASES.get("references", ("references", "bibliography"))
    for line in str(text or "").splitlines():
        md_m = _MD_HEADING_RE.match(line)
        tex_m = _LATEX_HEADING_RE.match(line) if md_m is None else None
        if md_m or tex_m:
            raw_heading = md_m.group(2) if md_m else tex_m.group(1)  # type: ignore[union-attr]
            cleaned = re.sub(r"[*_`]+", "", raw_heading).strip()
            cleaned = re.sub(
                r"^(?:\d+(?:\.\d+)*|[IVXLCDM]+)\.?\s+", "", cleaned
            ).strip()
            in_references = any(
                _heading_matches_alias(cleaned.lower(), alias) for alias in ref_aliases
            )
            if not in_references:
                body_lines.append(line)
            continue
        if not in_references:
            body_lines.append(line)
    return "\n".join(body_lines)


def _heading_matches_alias(heading: str, alias: str) -> bool:
    if not heading or not alias:
        return False
    return bool(re.search(rf"\b{re.escape(alias)}\b", heading, flags=re.IGNORECASE))


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


def _extract_reference_keys_and_aliases(
    references: Sequence[object],
) -> tuple[list[str], set[str], list[tuple[str, LiteratureResult]]]:
    """Collect known BibTeX keys and author/title tokens for citation cross-checking."""
    explicit_keys: list[str] = []
    coerced: list[LiteratureResult] = []
    for ref in references:
        if isinstance(ref, Mapping):
            raw_key = str(ref.get("key") or ref.get("citation_key") or "").strip()
            if raw_key:
                explicit_keys.append(raw_key)
        lit = _coerce_literature_result(ref)
        if lit is not None:
            coerced.append(lit)

    keyed = assign_citation_keys(coerced)
    known_keys: set[str] = {k.lower() for k in explicit_keys}
    for key, _ in keyed:
        known_keys.add(key.lower())
    return explicit_keys, known_keys, keyed


def _extract_inline_citations(text: str) -> tuple[int, list[str], list[str]]:
    """Return ``(total_citation_mentions, explicit_keys, author_year_or_numeric)``."""
    explicit_keys: list[str] = []
    for m in _PANDOC_CITE_RE.finditer(text):
        for token in re.findall(r"@([A-Za-z0-9_.:-]+)", m.group(1)):
            explicit_keys.append(token)
    for m in _LATEX_CITE_RE.finditer(text):
        for token in m.group(1).split(","):
            clean = token.strip()
            if clean:
                explicit_keys.append(clean)

    numeric_mentions = _NUMERIC_CITE_RE.findall(text)
    author_year_mentions = _AUTHOR_YEAR_CITE_RE.findall(text)

    total_mentions = (
        len(list(_PANDOC_CITE_RE.finditer(text)))
        + len(list(_LATEX_CITE_RE.finditer(text)))
        + len(numeric_mentions)
        + len(author_year_mentions)
    )
    other_mentions = [*numeric_mentions, *author_year_mentions]
    return total_mentions, explicit_keys, other_mentions


def _count_quantitative_claims(text: str) -> int:
    """Count real quantitative claims while ignoring headings, years, and citations."""
    # Strip markdown heading lines and bracketed numeric/key citations before scanning.
    body_lines = [
        line
        for line in str(text or "").splitlines()
        if not _MD_HEADING_RE.match(line) and not _LATEX_HEADING_RE.match(line)
    ]
    scrubbed = "\n".join(body_lines)
    scrubbed = _PANDOC_CITE_RE.sub("", scrubbed)
    scrubbed = _LATEX_CITE_RE.sub("", scrubbed)
    scrubbed = _NUMERIC_CITE_RE.sub("", scrubbed)
    return len(_QUANT_CLAIM_RE.findall(scrubbed))


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
    """Run deterministic manuscript quality and citation-grounding checks."""
    text = str(paper_text or "")
    words = re.findall(r"\b[\w'-]+\b", text)
    extracted_sections = _extract_sections_from_text(text)
    headings = {h for _, h, _ in extracted_sections}
    section_word_counts = {
        h: len(re.findall(r"\b[\w'-]+\b", body)) for _, h, body in extracted_sections
    }
    # Only an H1 document title (level 1) is allowed to have empty prose before its first H2 section.
    empty_sections = [
        h for lvl, h, body in extracted_sections if lvl != 1 and not body.strip()
    ]

    missing_sections: list[str] = []
    for section in required_sections:
        key = str(section or "").strip().lower()
        aliases = _SECTION_ALIASES.get(key, (key,))
        if not any(
            any(_heading_matches_alias(heading, alias) for alias in aliases)
            for heading in headings
        ):
            missing_sections.append(key)

    # Scan only the manuscript body (excluding the References section) so entries listed
    # solely in "## References" do not masquerade as in-text citations or quantitative claims.
    body_text = _extract_manuscript_body_without_references(text)
    citation_count, cited_keys, other_citations = _extract_inline_citations(body_text)
    numeric_claim_count = _count_quantitative_claims(body_text)
    tbd_count = len(re.findall(r"\bTBD\b", text, flags=re.IGNORECASE))
    placeholder_matches = _find_placeholder_matches(text)

    explicit_ref_keys, known_keys, keyed_refs = _extract_reference_keys_and_aliases(
        references
    )
    reference_count = max(len(keyed_refs), len(explicit_ref_keys))

    # Cross-check explicit citation keys ([@key] / \cite{key}) against supplied references.
    unresolved_citations: list[str] = []
    if reference_count > 0 and cited_keys:
        seen_unres: set[str] = set()
        for ck in cited_keys:
            if ck.lower() not in known_keys and ck.lower() not in seen_unres:
                seen_unres.add(ck.lower())
                unresolved_citations.append(ck)

    # Identify supplied references that are never cited in the manuscript body.
    uncited_references: list[str] = []
    if reference_count > 0:
        cited_lower = {ck.lower() for ck in cited_keys}
        body_lower = body_text.lower()
        for key, lit in keyed_refs:
            if key.lower() in cited_lower:
                continue
            title_norm = " ".join(lit.title.lower().split())
            if title_norm and title_norm[:40] in body_lower:
                continue
            if lit.doi and lit.doi.lower() in body_lower:
                continue
            if lit.arxiv_id and lit.arxiv_id.lower() in body_lower:
                continue
            if other_citations and lit.authors and lit.year:
                first_author = lit.authors[0]
                surname = (
                    first_author.split(",")[0].strip().lower()
                    if "," in first_author
                    else first_author.split()[-1].strip().lower()
                )
                if surname and any(
                    surname in c.lower() and str(lit.year) in c for c in other_citations
                ):
                    continue
            uncited_references.append(f"@{key}: {lit.title}")

    warnings: list[str] = []
    status = "pass"
    if placeholder_matches:
        status = "fail"
        warnings.append("Template or placeholder content remains in the draft.")
    if empty_sections:
        status = "warn" if status == "pass" else status
        warnings.append("Empty sections detected: " + ", ".join(empty_sections))
    if missing_sections:
        status = "warn" if status == "pass" else status
        warnings.append("Missing expected sections: " + ", ".join(missing_sections))
    if unresolved_citations:
        status = "warn" if status == "pass" else status
        warnings.append(
            "Cited keys not found in supplied references: "
            + ", ".join(unresolved_citations)
        )
    if reference_count == 0:
        status = "warn" if status == "pass" else status
        warnings.append("No reference list was supplied for citation checks.")
    if citation_count == 0 and reference_count > 0:
        status = "warn" if status == "pass" else status
        warnings.append(
            "References were supplied but no in-text citations were detected."
        )
    if numeric_claim_count > 0 and tbd_count == 0 and citation_count == 0:
        status = "warn" if status == "pass" else status
        warnings.append(
            "Quantitative claims are present without detected citations or TBD markers."
        )

    return {
        "ok": True,
        "quality_status": status,
        "metrics": {
            "word_count": len(words),
            "heading_count": len(headings),
            "citation_count": citation_count,
            "unique_cited_keys": len({k.lower() for k in cited_keys}),
            "reference_count": reference_count,
            "numeric_claim_count": numeric_claim_count,
            "tbd_count": tbd_count,
            "placeholder_count": len(placeholder_matches),
            "empty_section_count": len(empty_sections),
            "unresolved_citation_count": len(unresolved_citations),
        },
        "section_word_counts": section_word_counts,
        "missing_sections": missing_sections,
        "empty_sections": empty_sections,
        "unresolved_citations": unresolved_citations,
        "uncited_references": uncited_references,
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
            "Cite supplied references with exact [@key] markers and mark unmeasured metrics as TBD.",
            "Produce a structured paper with outline, sections, citations, and a concise completion note.",
            "Run `assess_paper_draft` before finalizing so placeholders or missing sections are caught.",
            "If the current context is insufficient, search literature first rather than drafting from guesswork.",
        ]
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core Swarm Generation & Transcript Extraction
# ---------------------------------------------------------------------------


def extract_final_draft_from_swarm_output(transcript: str) -> str:
    """Extract the latest Writer manuscript from a multi-turn swarm transcript."""
    raw = str(transcript or "").strip()
    if not raw:
        return ""
    blocks = list(
        re.finditer(
            r"^---\s+(?P<agent>[^\n(]+?)\s*\((?P<role>[^)]+)\)\s+---\s*$",
            raw,
            flags=re.MULTILINE,
        )
    )
    if not blocks:
        return raw

    writer_turns: list[str] = []
    for idx, match in enumerate(blocks):
        agent = match.group("agent").strip().lower()
        start = match.end()
        end = blocks[idx + 1].start() if idx + 1 < len(blocks) else len(raw)
        body = raw[start:end].strip()
        # Strip trailing swarm footer lines if present on the final block.
        body = re.sub(
            r"\n(?:Swarm reached consensus: TASK COMPLETE\.|Swarm max turns reached\.)\s*$",
            "",
            body,
        ).strip()
        if agent == "writer" and body and not body.startswith("[ERROR]"):
            writer_turns.append(body)

    if writer_turns:
        return writer_turns[-1]
    return raw


async def run_paper_drafting_swarm(
    topic: str,
    papers: Sequence[LiteratureResult],
    loop_factory: Callable[[], Any],
    max_turns: int = 8,
    research_context: str = "",
    *,
    resolve_budget: bool = True,
) -> str:
    """Draft a complete research paper by orchestrating Outliner/Writer/Reviewer.

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
    research_context:
        Optional preflight packet / guardrail context.
    resolve_budget:
        When True (default), adaptively scales ``max_turns`` via
        ``resolve_swarm_turn_budget``. Callers that already resolved the budget
        can pass ``resolve_budget=False`` to avoid double-counting.
    """
    manager_cls, agent_cls, resolve_budget_fn = _resolve_swarm_runtime()
    try:
        manager = manager_cls(
            max_context_chars=32000,
            max_turn_output_chars=32000,
            completion_agents=("Reviewer",),
        )
    except TypeError:
        manager = manager_cls()

    outliner = agent_cls(
        name="Outliner",
        role="Literature Review and Planner",
        system_prompt=(
            "You are the Outliner Agent. Read the provided topic, preflight guardrails, "
            "and numbered literature list (with [@key] citation keys). Propose a tight 5-7 "
            "section Markdown outline (Abstract, Introduction, Related Work, Methods, "
            "Results, Discussion and Limitations, References) mapping specific [@key] "
            "citations to each section. Only propose the outline and section guidance; "
            "do not write the full paper."
        ),
        loop_factory=loop_factory,
    )

    writer = agent_cls(
        name="Writer",
        role="Academic Author",
        system_prompt=(
            "You are the Paper-Writing Agent. Write or revise the COMPLETE research paper "
            "in clean Markdown following the Outliner's structure and Reviewer's feedback. "
            "Cite ONLY the supplied references using their exact [@key] keys. Never invent "
            "DOIs, arXiv IDs, authors, years, or experimental metrics — mark unmeasured "
            "numbers explicitly as TBD. Do not leave [TODO] or placeholder stubs."
        ),
        loop_factory=loop_factory,
    )

    reviewer = agent_cls(
        name="Reviewer",
        role="Peer Reviewer",
        system_prompt=(
            "You are the Peer Reviewer Agent. Audit the Writer's latest manuscript against "
            "the 5-axis rubric (Novelty, Rigor, Evidence, Clarity, Impact). Check for "
            "missing sections, placeholder text, ungrounded citations outside the supplied "
            "[@key] list, and unverified numbers without TBD markers. Give concrete revision "
            "instructions if issues remain. Only when all critical issues are resolved, "
            "output 'TASK COMPLETE' on its own line."
        ),
        loop_factory=loop_factory,
    )

    manager.register_agent(outliner)
    manager.register_agent(writer)
    manager.register_agent(reviewer)

    paper_list = _format_paper_list(papers, include_keys=True)
    initial_task = (
        f"Draft a comprehensive research paper on: {topic}\n\n"
        f"Relevant Literature to use (cite with exact [@key]):\n{paper_list}\n\n"
        "Outliner, please start by proposing the sections and citation map."
    )
    context_text = str(research_context or "").strip()
    if context_text:
        initial_task += (
            "\n\nResearch preflight context and guardrails:\n"
            f"{context_text}\n\n"
            "Follow these guardrails when planning, writing, and reviewing."
        )

    resolved_turns = (
        resolve_budget_fn(
            initial_task,
            max_turns,
            agent_count=3,
            paper_context=True,
        )
        if resolve_budget
        else max(1, int(max_turns))
    )
    return await manager.run_swarm(initial_task, max_turns=resolved_turns)


# ---------------------------------------------------------------------------
# BibTeX helpers
# ---------------------------------------------------------------------------


def _escape_bibtex_value(value: str) -> str:
    """Escape braces and LaTeX comment/special tokens safely for BibTeX values."""
    text = " ".join(str(value or "").split()).strip()
    text = re.sub(r"(?<!\\)\{", r"\\{", text)
    text = re.sub(r"(?<!\\)\}", r"\\}", text)
    text = re.sub(r"(?<!\\)%", r"\\%", text)
    text = re.sub(r"(?<!\\)&", r"\\&", text)
    return text


def build_bibtex(papers: Sequence[LiteratureResult]) -> str:
    """Generate a BibTeX file string from a list of ``LiteratureResult`` objects.

    Uses ``@article`` when a non-arXiv venue/journal is provided and ``@misc``
    otherwise, with deterministic unique keys matching ``assign_citation_keys``.
    Empty sequence returns empty string.
    """
    if not papers:
        return ""
    entries: list[str] = []
    for key, paper in assign_citation_keys(papers):
        has_journal = bool(paper.venue and paper.venue.strip().lower() != "arxiv")
        entry_type = "article" if has_journal else "misc"

        fields: list[str] = []
        if paper.title:
            fields.append(f"  title     = {{{_escape_bibtex_value(paper.title)}}}")
        if paper.authors:
            author_val = " and ".join(
                _escape_bibtex_value(name) for name in paper.authors if name.strip()
            )
            if author_val:
                fields.append(f"  author    = {{{author_val}}}")
        if has_journal:
            fields.append(f"  journal   = {{{_escape_bibtex_value(paper.venue)}}}")
        if paper.year:
            fields.append(f"  year      = {{{paper.year}}}")
        if paper.doi:
            fields.append(f"  doi       = {{{paper.doi.strip()}}}")
        if paper.arxiv_id:
            fields.append(f"  eprint    = {{{paper.arxiv_id.strip()}}}")
            fields.append("  archivePrefix = {arXiv}")
        url = _paper_url(paper)
        if url:
            fields.append(f"  url       = {{{url}}}")
        if paper.source:
            fields.append(
                f"  note      = {{Source: {_escape_bibtex_value(paper.source)}}}"
            )

        body = ",\n".join(fields)
        entries.append(f"@{entry_type}{{{key},\n{body}\n}}")

    return "\n\n".join(entries) + "\n"


# ---------------------------------------------------------------------------
# Markdown, LaTeX, and atomic bundle export
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
        for i, (key, ref) in enumerate(assign_citation_keys(draft.references), 1):
            authors = _format_authors_short(ref.authors)
            author_part = f"{authors}. " if authors else ""
            year = f" ({ref.year})" if ref.year else ""
            venue = f" *{ref.venue}*." if ref.venue else ""
            url = _paper_url(ref)
            link = f" — [{url}]({url})" if url else ""
            lines.append(f"{i}. `[@{key}]` {author_part}{ref.title}{year}{venue}{link}")
        lines.append("")

    return "\n".join(lines)


def _escape_latex_text(text: str) -> str:
    replacements = (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    )
    out = str(text or "")
    for old, new in replacements:
        out = out.replace(old, new)
    return out


_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
_ORDERED_LIST_RE = re.compile(r"^\d+[.)]\s+(.+)$")


def _markdown_inline_to_latex(line: str) -> str:
    """Convert inline Markdown citations, links, and emphasis into valid LaTeX."""
    # Protect [@key] citations and [label](url) links before escaping LaTeX metacharacters.
    placeholders: dict[str, str] = {}

    def _stash_cite(match: re.Match[str]) -> str:
        keys = re.findall(r"@([A-Za-z0-9_.:-]+)", match.group(1))
        token = f"@@CITE{len(placeholders)}@@"
        placeholders[token] = f"\\cite{{{', '.join(keys)}}}"
        return token

    def _stash_link(match: re.Match[str]) -> str:
        label = _escape_latex_text(match.group(1).strip())
        url = match.group(2).strip().replace("%", r"\%").replace("#", r"\#")
        token = f"@@LINK{len(placeholders)}@@"
        placeholders[token] = f"\\href{{{url}}}{{{label}}}"
        return token

    protected = _PANDOC_CITE_RE.sub(_stash_cite, str(line or ""))
    protected = _MD_LINK_RE.sub(_stash_link, protected)
    escaped = _escape_latex_text(protected)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"\\textbf{\1}", escaped)
    escaped = re.sub(r"\*([^*]+)\*", r"\\emph{\1}", escaped)
    escaped = re.sub(r"`([^`]+)`", r"\\texttt{\1}", escaped)
    for token, latex_cmd in placeholders.items():
        escaped = escaped.replace(token, latex_cmd)
    return escaped


def format_draft_latex(
    draft: PaperDraft,
    *,
    bib_stem: str = "references",
    document_class: str = "article",
) -> str:
    """Render a ``PaperDraft`` into a compilable LaTeX manuscript."""
    safe_class = (
        re.sub(r"[^a-zA-Z0-9_-]+", "", str(document_class or "article")) or "article"
    )
    safe_bib = (
        re.sub(r"[^a-zA-Z0-9._-]+", "_", str(bib_stem or "references")).strip("._")
        or "references"
    )
    lines: list[str] = [
        f"\\documentclass{{{safe_class}}}",
        "\\usepackage[utf8]{inputenc}",
        "\\usepackage{hyperref}",
        "\\usepackage{booktabs}",
        "",
        f"\\title{{{_escape_latex_text(draft.topic)}}}",
        "\\date{}",
        "",
        "\\begin{document}",
        "\\maketitle",
        "",
    ]

    for section in draft.sections:
        title_lower = section.title.strip().lower()
        body = section.content.strip() if section.content else "TBD."
        converted_paras: list[str] = []
        list_mode: str | None = None  # "itemize" | "enumerate" | None
        in_verbatim = False

        def _close_list() -> None:
            nonlocal list_mode
            if list_mode is not None:
                converted_paras.append(f"\\end{{{list_mode}}}")
                list_mode = None

        for raw_line in body.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("```"):
                _close_list()
                if not in_verbatim:
                    converted_paras.append("\\begin{verbatim}")
                    in_verbatim = True
                else:
                    converted_paras.append("\\end{verbatim}")
                    in_verbatim = False
                continue
            if in_verbatim:
                converted_paras.append(raw_line)
                continue

            ordered_m = _ORDERED_LIST_RE.match(stripped)
            if stripped.startswith(("- ", "* ")):
                if list_mode != "itemize":
                    _close_list()
                    converted_paras.append("\\begin{itemize}")
                    list_mode = "itemize"
                converted_paras.append(
                    f"  \\item {_markdown_inline_to_latex(stripped[2:].strip())}"
                )
            elif ordered_m:
                if list_mode != "enumerate":
                    _close_list()
                    converted_paras.append("\\begin{enumerate}")
                    list_mode = "enumerate"
                converted_paras.append(
                    f"  \\item {_markdown_inline_to_latex(ordered_m.group(1).strip())}"
                )
            else:
                _close_list()
                if stripped.startswith("#### "):
                    converted_paras.append(
                        f"\\subsubsection{{{_escape_latex_text(stripped[5:].strip())}}}"
                    )
                elif stripped.startswith("### "):
                    converted_paras.append(
                        f"\\subsection{{{_escape_latex_text(stripped[4:].strip())}}}"
                    )
                elif stripped:
                    converted_paras.append(_markdown_inline_to_latex(stripped))
                else:
                    converted_paras.append("")
        _close_list()
        if in_verbatim:
            converted_paras.append("\\end{verbatim}")

        rendered_body = "\n".join(converted_paras).strip()
        if title_lower == "abstract":
            lines.extend(["\\begin{abstract}", rendered_body, "\\end{abstract}", ""])
        elif title_lower not in {"references", "bibliography"}:
            lines.extend(
                [
                    f"\\section{{{_escape_latex_text(section.title)}}}",
                    rendered_body,
                    "",
                ]
            )

    if draft.references:
        lines.extend(
            [
                "\\bibliographystyle{plain}",
                f"\\bibliography{{{safe_bib}}}",
                "",
            ]
        )

    lines.append("\\end{document}")
    return "\n".join(lines) + "\n"


def _sanitize_file_stem(base_name: str, *, default: str = "paper_draft") -> str:
    raw = Path(str(base_name or "").strip()).name
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw).strip("._")
    if not cleaned or ".." in cleaned:
        return default
    return cleaned


def write_draft_bundle(
    draft: PaperDraft,
    output_dir: Path | str,
    *,
    base_name: str = "paper_draft",
    include_latex: bool = True,
    quality_report: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Atomically write ``.md``, ``.bib``, optional ``.tex``, and quality JSON."""
    out_dir = Path(output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = _sanitize_file_stem(base_name, default="paper_draft")
    written: dict[str, str] = {}

    md_text = format_draft_markdown(draft)
    md_path = out_dir / f"{stem}.md"
    atomic_write_text(md_path, md_text)
    written["markdown"] = str(md_path)

    bib_text = build_bibtex(draft.references)
    if bib_text:
        bib_path = out_dir / f"{stem}.bib"
        atomic_write_text(bib_path, bib_text)
        written["bibtex"] = str(bib_path)

    if include_latex:
        tex_text = format_draft_latex(draft, bib_stem=stem)
        tex_path = out_dir / f"{stem}.tex"
        atomic_write_text(tex_path, tex_text)
        written["latex"] = str(tex_path)

    report_payload = (
        dict(quality_report)
        if quality_report is not None
        else assess_paper_draft_quality(md_text, references=draft.references)
    )
    quality_path = out_dir / f"{stem}_quality.json"
    atomic_write_text(
        quality_path, json.dumps(report_payload, indent=2, ensure_ascii=False)
    )
    written["quality_json"] = str(quality_path)
    return written


__all__ = [
    "PaperDraft",
    "PaperSection",
    "SwarmAgent",
    "SwarmManager",
    "assess_paper_draft_quality",
    "assign_citation_keys",
    "build_bibtex",
    "build_paper_swarm_prompt",
    "build_research_preflight_packet",
    "draft_paper",
    "draft_section",
    "extract_final_draft_from_swarm_output",
    "format_draft_latex",
    "format_draft_markdown",
    "format_research_packet_for_prompt",
    "generate_outline",
    "run_paper_drafting_swarm",
    "write_draft_bundle",
]
