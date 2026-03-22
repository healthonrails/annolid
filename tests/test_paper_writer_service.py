"""Unit tests for annolid.services.paper_writer service.

All tests are pure-Python: no Qt, no network.
"""

from __future__ import annotations

from annolid.services.literature_search import LiteratureResult
from annolid.services.paper_writer import (
    PaperDraft,
    PaperSection,
    build_bibtex,
    build_paper_swarm_prompt,
    run_paper_drafting_swarm,
    format_draft_markdown,
)
from annolid.core.agent.tools.research import DraftPaperSwarmTool

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
# run_paper_drafting_swarm tests
# ---------------------------------------------------------------------------


def test_run_paper_drafting_swarm(monkeypatch) -> None:
    # A dummy SwarmManager that just returns a fixed string
    class DummySwarmManager:
        def register_agent(self, agent):
            pass

        async def run_swarm(self, task: str, max_turns: int = 5) -> str:
            return "## Introduction\nDraft content."

    import annolid.services.paper_writer

    monkeypatch.setattr(
        annolid.services.paper_writer, "SwarmManager", DummySwarmManager
    )

    def dummy_loop_factory():
        return None

    papers = [_make_paper(title="Test", summary="Context")]

    import asyncio

    result = asyncio.run(
        run_paper_drafting_swarm(
            "Neuroscience", papers, dummy_loop_factory, max_turns=2
        )
    )
    assert "## Introduction" in result
    assert "Draft content." in result


def test_run_paper_drafting_swarm_defaults_to_longer_turn_budget(monkeypatch) -> None:
    captured: dict[str, int] = {}

    class DummySwarmManager:
        def register_agent(self, agent):
            pass

        async def run_swarm(self, task: str, max_turns: int = 8) -> str:
            captured["max_turns"] = max_turns
            return "## Introduction\nDraft content."

    import annolid.services.paper_writer

    monkeypatch.setattr(
        annolid.services.paper_writer, "SwarmManager", DummySwarmManager
    )

    def dummy_loop_factory():
        return None

    papers = [_make_paper(title="Test", summary="Context")]

    import asyncio

    result = asyncio.run(
        run_paper_drafting_swarm("Neuroscience", papers, dummy_loop_factory)
    )
    assert "## Introduction" in result
    assert captured["max_turns"] > 8


def test_draft_paper_swarm_tool_uses_new_default_turn_budget(monkeypatch) -> None:
    captured: dict[str, int] = {}

    async def _fake_run(topic: str, papers, loop_factory, max_turns: int = 8):
        captured["max_turns"] = max_turns
        return f"topic={topic}"

    import annolid.core.agent.tools.research as research_mod

    monkeypatch.setattr(research_mod, "run_paper_drafting_swarm", _fake_run)

    tool = DraftPaperSwarmTool()
    tool.set_loop_factory(lambda: object())

    import asyncio

    result = asyncio.run(tool.execute("Behavioral Understanding"))
    assert "topic=Behavioral Understanding" in result
    assert captured["max_turns"] > 8


def test_build_paper_swarm_prompt_includes_pdf_context() -> None:
    prompt = build_paper_swarm_prompt(
        "Behavioral understanding",
        pdf_state={
            "title": "BehaviorVLM.pdf",
            "path": "/tmp/BehaviorVLM.pdf",
        },
    )
    assert "draft_paper_swarm" in prompt
    assert "Behavioral understanding" in prompt
    assert "title=BehaviorVLM.pdf" in prompt
    assert "path=/tmp/BehaviorVLM.pdf" in prompt
