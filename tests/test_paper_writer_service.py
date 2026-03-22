"""Unit tests for annolid.services.paper_writer service.

All tests are pure-Python: no Qt, no network.
"""

from __future__ import annotations

from annolid.services.literature_search import LiteratureResult
from annolid.services.paper_writer import (
    PaperDraft,
    PaperSection,
    build_bibtex,
    draft_section,
    format_draft_markdown,
    generate_outline,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paper(
    *,
    title: str = "A Test Paper",
    doi: str = "",
    arxiv_id: str = "",
    year: int | None = 2024,
    source: str = "arxiv",
    summary: str = "",
) -> LiteratureResult:
    return LiteratureResult(
        source=source,
        title=title,
        doi=doi,
        arxiv_id=arxiv_id,
        year=year,
        summary=summary,
    )


def _noop_llm(_system: str, _user: str) -> str:
    """LLM stub returning a canned response."""
    return '[{"title": "Introduction", "guidance": "Motivate the problem."}, {"title": "Conclusion", "guidance": "Summarize."}]'


def _section_llm(_system: str, user: str) -> str:
    return f"This section covers the topic as indicated. Topic extracted from prompt: {user[:20]}..."


# ---------------------------------------------------------------------------
# build_bibtex tests
# ---------------------------------------------------------------------------


def test_build_bibtex_empty() -> None:
    result = build_bibtex([])
    assert result == ""


def test_build_bibtex_basic_doi() -> None:
    paper = _make_paper(title="Neural Tracking", doi="10.1234/nt.2024")
    bib = build_bibtex([paper])
    assert "@misc{" in bib
    assert "Neural Tracking" in bib
    assert "10.1234/nt.2024" in bib


def test_build_bibtex_arxiv() -> None:
    paper = _make_paper(title="Pose Estimation Survey", arxiv_id="2401.00001")
    bib = build_bibtex([paper])
    assert "arXiv" in bib
    assert "2401.00001" in bib


def test_build_bibtex_multiple_unique_keys() -> None:
    p1 = _make_paper(title="Paper A", arxiv_id="2401.00001")
    p2 = _make_paper(title="Paper B", arxiv_id="2401.00001")  # same id
    bib = build_bibtex([p1, p2])
    # Both should be present; keys must be unique
    entries = [line for line in bib.splitlines() if line.startswith("@misc{")]
    assert len(entries) == 2
    keys = [e.split("{", 1)[1].rstrip(",") for e in entries]
    assert len(set(keys)) == 2


def test_build_bibtex_year_included() -> None:
    paper = _make_paper(title="Year Test", year=2023)
    bib = build_bibtex([paper])
    assert "2023" in bib


def test_build_bibtex_no_year() -> None:
    paper = _make_paper(title="No Year", year=None)
    bib = build_bibtex([paper])
    assert "@misc{" in bib


# ---------------------------------------------------------------------------
# format_draft_markdown tests
# ---------------------------------------------------------------------------


def test_format_draft_markdown_title() -> None:
    draft = PaperDraft(topic="Animal Tracking")
    md = format_draft_markdown(draft)
    assert md.startswith("# Animal Tracking")


def test_format_draft_markdown_sections() -> None:
    draft = PaperDraft(
        topic="Pose Estimation",
        sections=[
            PaperSection(title="Introduction", content="We study poses."),
            PaperSection(title="Conclusion", content="Poses matter."),
        ],
    )
    md = format_draft_markdown(draft)
    assert "## Introduction" in md
    assert "We study poses." in md
    assert "## Conclusion" in md


def test_format_draft_markdown_empty_section_placeholder() -> None:
    draft = PaperDraft(
        topic="Test",
        sections=[PaperSection(title="Related Work")],
    )
    md = format_draft_markdown(draft)
    assert "*(section not yet drafted)*" in md


def test_format_draft_markdown_references() -> None:
    draft = PaperDraft(
        topic="Test",
        references=[_make_paper(title="Cited Paper", year=2022)],
    )
    md = format_draft_markdown(draft)
    assert "## References" in md
    assert "Cited Paper" in md
    assert "(2022)" in md


# ---------------------------------------------------------------------------
# generate_outline tests
# ---------------------------------------------------------------------------


def test_generate_outline_parses_llm_json() -> None:
    papers = [_make_paper()]
    outline = generate_outline("pose estimation", papers, _noop_llm)
    assert isinstance(outline, list)
    assert len(outline) == 2
    assert outline[0]["title"] == "Introduction"


def test_generate_outline_fallback_on_bad_json() -> None:
    def _bad_llm(_sys: str, _usr: str) -> str:
        return "this is not JSON"

    papers = [_make_paper()]
    outline = generate_outline("test topic", papers, _bad_llm)
    # Should fall back to default 5-section outline
    assert len(outline) >= 2
    assert all("title" in item for item in outline)


def test_generate_outline_empty_papers() -> None:
    outline = generate_outline("robot navigation", [], _noop_llm)
    assert isinstance(outline, list)
    assert len(outline) >= 2


# ---------------------------------------------------------------------------
# draft_section tests
# ---------------------------------------------------------------------------


def test_draft_section_calls_llm() -> None:
    called: list[tuple[str, str]] = []

    def _tracking_llm(sys_prompt: str, user_prompt: str) -> str:
        called.append((sys_prompt, user_prompt))
        return "This is the section content."

    papers = [_make_paper()]
    result = draft_section(
        "Introduction",
        "Motivate the problem.",
        "pose estimation",
        papers,
        _tracking_llm,
    )
    assert len(called) == 1
    assert "Introduction" in called[0][1]
    assert "pose estimation" in called[0][1]
    assert result == "This is the section content."


def test_draft_section_handles_llm_error() -> None:
    def _failing_llm(_s: str, _u: str) -> str:
        raise RuntimeError("LLM unavailable")

    result = draft_section("Conclusion", "", "test topic", [], _failing_llm)
    assert result == ""
