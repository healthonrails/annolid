"""Unit tests for annolid.services.paper_writer service.

All tests are pure-Python: no Qt, no network.
"""

from __future__ import annotations

from pathlib import Path
import asyncio
import json

from annolid.services.literature_search import LiteratureResult
from annolid.services.paper_writer import (
    PaperDraft,
    PaperSection,
    assess_paper_draft_quality,
    assign_citation_keys,
    build_bibtex,
    build_paper_swarm_prompt,
    build_research_preflight_packet,
    draft_paper,
    draft_section,
    extract_final_draft_from_swarm_output,
    format_draft_latex,
    format_draft_markdown,
    generate_outline,
    run_paper_drafting_swarm,
    write_draft_bundle,
)
from annolid.core.agent.tools.paper_reporting import AnnolidPaperRunReportTool
from annolid.core.agent.tools.research import (
    DraftPaperSwarmTool,
    LiteratureSearchTool,
    PaperDraftQualityTool,
    ResearchPacketTool,
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


def test_run_paper_drafting_swarm_includes_research_context(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class DummySwarmManager:
        def register_agent(self, agent):
            pass

        async def run_swarm(self, task: str, max_turns: int = 8) -> str:
            captured["task"] = task
            return "done"

    import annolid.services.paper_writer

    monkeypatch.setattr(
        annolid.services.paper_writer, "SwarmManager", DummySwarmManager
    )

    def dummy_loop_factory():
        return None

    import asyncio

    result = asyncio.run(
        run_paper_drafting_swarm(
            "Neuroscience",
            [_make_paper(title="Grounded Paper")],
            dummy_loop_factory,
            research_context="Do not fabricate numbers.",
        )
    )
    assert result == "done"
    assert "Grounded Paper" in captured["task"]
    assert "Do not fabricate numbers." in captured["task"]


def test_build_research_preflight_packet_adds_guardrails_and_novelty() -> None:
    packet = build_research_preflight_packet(
        "Sparse seed animal tracking",
        [
            _make_paper(
                title="Sparse Seed Animal Tracking",
                summary="Tracking animals from sparse seed annotations.",
            )
        ],
    )
    assert packet["schema_version"] == "1.0"
    assert packet["topic"] == "Sparse seed animal tracking"
    assert packet["literature"]
    assert packet["paper_rubric"]
    assert packet["drafting_guardrails"]
    assert "prompt_context" in packet
    novelty = packet["novelty"]
    assert isinstance(novelty, dict)
    assert novelty["ok"] is True


def test_assess_paper_draft_quality_flags_placeholders() -> None:
    result = assess_paper_draft_quality(
        "# Test Paper\n\n## Abstract\n[TODO: add content here]\n",
        references=[_make_paper(title="Reference")],
    )
    assert result["quality_status"] == "fail"
    assert result["metrics"]["placeholder_count"] == 1
    assert result["placeholder_matches"]


def test_assess_paper_draft_quality_passes_complete_cited_draft() -> None:
    draft = """# Test Paper

## Abstract
We summarize a grounded result (Reference Paper, 2024).

## Introduction
The problem is important (Reference Paper, 2024).

## Methods
We used auditable Annolid annotations.

## Results
The measured F1 was TBD until validated.

## Discussion
The workflow has limitations.

## References
Reference Paper.
"""
    result = assess_paper_draft_quality(
        draft,
        references=[_make_paper(title="Reference Paper")],
    )
    assert result["quality_status"] == "pass"
    assert result["missing_sections"] == []


def test_draft_paper_swarm_tool_uses_new_default_turn_budget(monkeypatch) -> None:
    captured: dict[str, int] = {}

    async def _fake_run(
        topic: str,
        papers,
        loop_factory,
        max_turns: int = 8,
        research_context: str = "",
    ):
        captured["max_turns"] = max_turns
        captured["paper_count"] = len(papers)
        captured["has_context"] = bool(research_context)
        return f"topic={topic}"

    import annolid.core.agent.tools.research as research_mod

    monkeypatch.setattr(research_mod, "run_paper_drafting_swarm", _fake_run)

    tool = DraftPaperSwarmTool()
    tool.set_loop_factory(lambda: object())

    import asyncio

    result = asyncio.run(
        tool.execute("Behavioral Understanding", max_literature_results=0)
    )
    assert "topic=Behavioral Understanding" in result
    assert captured["max_turns"] > 8
    assert captured["paper_count"] == 0


def test_draft_paper_swarm_tool_searches_literature_before_drafting(
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def _fake_run(
        topic: str,
        papers,
        loop_factory,
        max_turns: int = 8,
        research_context: str = "",
    ):
        captured["paper_count"] = len(papers)
        captured["research_context"] = research_context
        return f"topic={topic}"

    import annolid.core.agent.tools.research as research_mod

    monkeypatch.setattr(research_mod, "run_paper_drafting_swarm", _fake_run)
    monkeypatch.setattr(
        research_mod,
        "search_literature",
        lambda query, max_results=8: {
            "results": [
                _make_paper(
                    title="Behavior Tracking Reference",
                    summary="Reference summary.",
                ).to_dict()
            ]
        },
    )

    tool = DraftPaperSwarmTool()
    tool.set_loop_factory(lambda: object())

    import asyncio

    result = asyncio.run(tool.execute("Behavioral Understanding"))
    assert "topic=Behavioral Understanding" in result
    assert captured["paper_count"] == 1
    assert "RESEARCH PREFLIGHT PACKET" in str(captured["research_context"])


def test_research_packet_tool_uses_supplied_related_work_without_search() -> None:
    tool = ResearchPacketTool()

    import asyncio
    import json

    result = asyncio.run(
        tool.execute(
            "Animal tracking",
            related_work=[
                {
                    "title": "Animal Tracking Prior Work",
                    "summary": "Prior work about animal tracking.",
                    "year": 2024,
                }
            ],
        )
    )
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["literature"][0]["title"] == "Animal Tracking Prior Work"
    assert payload["drafting_guardrails"]


def test_paper_draft_quality_tool_returns_json() -> None:
    tool = PaperDraftQualityTool()

    import asyncio
    import json

    result = asyncio.run(tool.execute("# Paper\n\n## Abstract\nText."))
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["quality_status"] in {"pass", "warn", "fail"}


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


# ---------------------------------------------------------------------------
# New provider-agnostic drafting, BibTeX/LaTeX export, quality audit & security tests
# ---------------------------------------------------------------------------


def test_generate_outline_draft_section_draft_paper_and_latex_bundle(
    tmp_path: Path,
) -> None:
    paper = LiteratureResult(
        source="openalex",
        title="SAM2 & Behavior: 100% Modular Tracking",
        authors=("Alice Smith", "Bob Chen"),
        venue="Nature Methods",
        year=2024,
        doi="10.1000/sam2",
        summary="Video segmentation for behavioral ethology.",
    )
    key = assign_citation_keys([paper])[0][0]

    calls: list[tuple[str, str]] = []

    def _fake_llm(system_prompt: str, user_prompt: str) -> str:
        calls.append((system_prompt, user_prompt))
        if len(calls) == 1:
            return (
                "```json\n"
                '[{"title": "Abstract", "guidance": "Summarize key ideas."}, '
                '{"title": "Introduction", "guidance": "State the gap."}]\n'
                "```"
            )
        if "Section to write: Abstract" in user_prompt:
            return (
                f"## Abstract\nWe study **rodent** tracking [@{key}] with TBD accuracy."
            )
        return f"Prior work [@{key}] motivates our *modular* pipeline:\n- Fast tracking\n- Open metrics"

    outline = generate_outline("Modular Tracking", [paper], llm_call=_fake_llm)
    assert [s.title for s in outline] == ["Abstract", "Introduction"]

    sec = draft_section("Modular Tracking", outline[0], [paper], llm_call=_fake_llm)
    assert not sec.content.startswith("## Abstract")
    assert f"[@{key}]" in sec.content

    calls.clear()
    draft = draft_paper("Modular Tracking", [paper], llm_call=_fake_llm)
    assert draft.status == "done"
    assert len(draft.sections) == 2

    bib = build_bibtex([paper])
    assert f"@article{{{key}," in bib
    assert "Alice Smith and Bob Chen" in bib
    assert "Nature Methods" in bib
    assert "100\\%" in bib
    assert "\\&" in bib

    from annolid.utils.citations import parse_bibtex

    parsed_entries = parse_bibtex(bib)
    assert len(parsed_entries) == 1
    assert parsed_entries[0].key == key

    latex = format_draft_latex(draft, bib_stem="my_refs")
    assert "\\begin{abstract}" in latex
    assert f"\\cite{{{key}}}" in latex
    assert "\\textbf{rodent}" in latex
    assert "\\emph{modular}" in latex
    assert "\\begin{itemize}" in latex
    assert "\\bibliography{my_refs}" in latex

    written = write_draft_bundle(
        draft,
        tmp_path / "bundle",
        base_name="../ignored_dir/my_paper",
        include_latex=True,
    )
    assert Path(written["markdown"]).parent == (tmp_path / "bundle").resolve()
    assert Path(written["markdown"]).name == "my_paper.md"
    assert Path(written["bibtex"]).exists()
    assert Path(written["latex"]).exists()
    assert Path(written["quality_json"]).exists()
    assert list((tmp_path / "bundle").glob("*.tmp")) == []


def test_generate_outline_falls_back_on_malformed_llm_output() -> None:
    sections = generate_outline(
        "Rodent Tracking",
        [],
        llm_call=lambda _s, _u: "not valid json at all",
    )
    titles = [s.title for s in sections]
    assert "Abstract" in titles
    assert "Methods" in titles


def test_assess_paper_draft_quality_cross_checks_keys_latex_and_real_metrics() -> None:
    ref = LiteratureResult(
        source="openalex",
        title="Deep Tracking",
        authors=("Alice Smith",),
        year=2024,
        doi="10.1000/dt",
    )
    ref_key = assign_citation_keys([ref])[0][0]

    # 1) Markdown with [@key], unresolved [@ghost2029], and heading "Summary of Future Work"
    bad_draft = f"""# Paper Title

## 1. Introduction
Prior work [@{ref_key}] and fabricated [@ghost2029] appeared in 2024.

## 2. Methods
We evaluated 12 videos at 30 fps with mAP = 0.88.

## 3. Results

## 4. Summary of Future Work
More work is needed.
"""
    audit = assess_paper_draft_quality(bad_draft, references=[ref])
    assert audit["quality_status"] == "warn"
    assert "abstract" in audit["missing_sections"]
    assert "results" in audit["empty_sections"]
    assert audit["unresolved_citations"] == ["ghost2029"]
    assert audit["metrics"]["citation_count"] == 2
    assert audit["metrics"]["numeric_claim_count"] >= 2

    # 2) Undrafted placeholder from format_draft_markdown must FAIL quality check
    undrafted = format_draft_markdown(
        PaperDraft(topic="Empty", sections=[PaperSection(title="Abstract")])
    )
    assert assess_paper_draft_quality(undrafted)["quality_status"] == "fail"

    # 3) LaTeX headings & \cite{} pass cleanly without false numeric alarms on years
    latex_draft = f"""\\section{{Abstract}}
Grounded overview in 2024 \\cite{{{ref_key}}}.
\\section{{Introduction}}
Motivation \\cite{{{ref_key}}}.
\\section{{Methods}}
Auditable protocol.
\\section{{Results}}
Accuracy was TBD.
\\section{{Discussion}}
Clear limitations.
\\section{{References}}
{ref.title}
"""
    latex_audit = assess_paper_draft_quality(latex_draft, references=[ref])
    assert latex_audit["quality_status"] == "pass"
    assert latex_audit["missing_sections"] == []
    assert latex_audit["metrics"]["numeric_claim_count"] == 0
    assert latex_audit["metrics"]["tbd_count"] == 1


def test_swarm_manager_respects_completion_agents_negation_and_long_drafts() -> None:
    from annolid.core.agent.swarm import SwarmAgent, SwarmManager

    class _TurnResult:
        def __init__(self, content: str) -> None:
            self.content = content

    long_draft = "# Full Paper\n\n## Abstract\n" + ("word " * 1800)
    assert len(long_draft) > 8000

    class _FakeLoop:
        def __init__(self, replies: list[str]) -> None:
            self._replies = list(replies)
            self.prompts: list[str] = []

        async def run(self, prompt: str, **_kwargs):
            self.prompts.append(prompt)
            return _TurnResult(self._replies.pop(0))

    outliner_loop = _FakeLoop(["Outline complete. TASK COMPLETE"])
    writer_loop = _FakeLoop([long_draft, long_draft + "\n## Conclusion\nRevised."])
    reviewer_loop = _FakeLoop(
        [
            "Do NOT say TASK COMPLETE yet; add a Conclusion.",
            "Looks great.\nTASK COMPLETE",
        ]
    )

    mgr = SwarmManager(
        max_turn_output_chars=32000,
        completion_agents=("Reviewer",),
    )
    mgr.register_agent(SwarmAgent("Outliner", "Planner", "sys", lambda: outliner_loop))
    mgr.register_agent(SwarmAgent("Writer", "Author", "sys", lambda: writer_loop))
    mgr.register_agent(SwarmAgent("Reviewer", "Reviewer", "sys", lambda: reviewer_loop))

    transcript = asyncio.run(mgr.run_swarm("Draft a paper", max_turns=3))
    # Outliner's TASK COMPLETE and Reviewer's negated TASK COMPLETE must NOT abort turn 1.
    assert len(reviewer_loop.prompts) == 2
    assert "... [TRUNCATED] ..." not in transcript
    extracted = extract_final_draft_from_swarm_output(transcript)
    assert extracted.startswith("# Full Paper")
    assert "## Conclusion\nRevised." in extracted


def test_research_and_paper_reporting_tools_file_audit_export_and_security(
    tmp_path: Path, monkeypatch
) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("# Secret\n", encoding="utf-8")

    # 1) PaperDraftQualityTool reads paper_path + bib_path inside allowed_dir and blocks outside
    bib_file = workspace / "refs.bib"
    bib_file.write_text(
        "@article{smith2024,\n  title={Tracking},\n  author={Smith, Alice},\n  year={2024},\n}\n",
        encoding="utf-8",
    )
    draft_file = workspace / "draft.md"
    draft_file.write_text(
        "# Study\n\n## Abstract\nGrounded [@smith2024].\n"
        "## Introduction\nGap [@smith2024].\n"
        "## Methods\nDetails.\n## Results\nTBD.\n## Discussion\nLimits.\n## References\nSmith 2024.\n",
        encoding="utf-8",
    )
    q_tool = PaperDraftQualityTool(allowed_dir=workspace)
    res_ok = json.loads(
        asyncio.run(q_tool.execute(paper_path=str(draft_file), bib_path=str(bib_file)))
    )
    assert res_ok["ok"] is True
    assert res_ok["quality_status"] == "pass"
    assert res_ok["unresolved_citations"] == []

    res_blocked = json.loads(asyncio.run(q_tool.execute(paper_path=str(outside))))
    assert res_blocked["ok"] is False

    # 2) LiteratureSearchTool displays authors, venue, and IDs
    monkeypatch.setattr(
        "annolid.core.agent.tools.research.search_literature",
        lambda query, max_results=8: {
            "results": [
                {
                    "title": "Behavioral Ethology",
                    "year": 2024,
                    "authors": ["Alice Smith", "Bob Chen"],
                    "venue": "Nature Methods",
                    "doi": "10.1000/be",
                    "summary": "Clear summary.",
                    "pdf_url": "https://example.org/be.pdf",
                }
            ]
        },
    )
    lit_out = asyncio.run(LiteratureSearchTool().execute("ethology"))
    assert "Authors: Alice Smith, Bob Chen" in lit_out
    assert "Venue: Nature Methods" in lit_out
    assert "doi:10.1000/be" in lit_out

    # 3) DraftPaperSwarmTool enforces allow_mutation and exports .md/.bib/_quality.json atomically
    import annolid.core.agent.tools.research as research_mod

    async def _fake_swarm(**kwargs):
        assert kwargs.get("resolve_budget") is False
        return "--- Writer (Academic Author) ---\n# Study\n\n## Abstract\nText [@doi_be].\n"

    monkeypatch.setattr(research_mod, "run_paper_drafting_swarm", _fake_swarm)
    swarm_tool = DraftPaperSwarmTool(allowed_dir=workspace)
    swarm_tool.set_loop_factory(lambda: object())

    blocked_export = asyncio.run(
        swarm_tool.execute(
            "Ethology",
            related_work=[{"title": "Behavioral Ethology", "doi": "10.1000/be"}],
            output_dir=str(workspace / "out"),
            allow_mutation=False,
        )
    )
    assert "allow_mutation=true" in blocked_export

    exported = asyncio.run(
        swarm_tool.execute(
            "Ethology",
            related_work=[{"title": "Behavioral Ethology", "doi": "10.1000/be"}],
            output_dir=str(workspace / "out"),
            report_basename="final_draft",
            allow_mutation=True,
        )
    )
    assert "Draft quality check:" in exported
    assert (workspace / "out" / "final_draft.md").exists()
    assert (workspace / "out" / "final_draft.bib").exists()
    assert (workspace / "out" / "final_draft_quality.json").exists()

    # 4) AnnolidPaperRunReportTool rejects path traversal in report_basename, writes _table.tex, and audits paper_draft_path
    report_tool = AnnolidPaperRunReportTool(allowed_dir=workspace)
    eval_payload = {
        "metadata": {
            "model": "m",
            "dataset": "d",
            "split": "test",
            "source_path": str(draft_file),
        },
        "quality_status": "pass",
        "paper_table": {
            "markdown": "| Metric | Value |\n|---|---|\n| F1 | 0.91 |",
            "latex": "\\begin{tabular}{lr}F1 & 0.91\\\\\\end{tabular}",
        },
        "quality_checks": [],
    }
    trav_res = json.loads(
        asyncio.run(
            report_tool.execute(
                eval_report=eval_payload,
                report_dir=str(workspace / "reports"),
                report_basename="../../escape",
                allow_mutation=True,
            )
        )
    )
    assert trav_res["ok"] is False
    assert "report_basename" in trav_res["error"]

    good_res = json.loads(
        asyncio.run(
            report_tool.execute(
                eval_report=eval_payload,
                paper_draft_path=str(draft_file),
                report_dir=str(workspace / "reports"),
                report_basename="unified_run",
                allow_mutation=True,
            )
        )
    )
    assert good_res["ok"] is True
    written = good_res["report"]["written_files"]
    assert Path(written["latex_table"]).exists()
    assert good_res["report"]["draft_quality"]["available"] is True
    assert good_res["report"]["draft_quality"]["quality_status"] == "warn"
    assert "## Manuscript Draft Quality" in good_res["report"]["report_markdown"]
    assert list((workspace / "reports").glob("*.tmp")) == []

    # 5) Passing bib_path alongside paper_draft_path verifies citations cleanly to PASS
    with_bib = json.loads(
        asyncio.run(
            report_tool.execute(
                eval_report=eval_payload,
                paper_draft_path=str(draft_file),
                bib_path=str(bib_file),
            )
        )
    )
    assert with_bib["ok"] is True
    assert with_bib["report"]["draft_quality"]["quality_status"] == "pass"
    assert with_bib["report"]["draft_quality"]["unresolved_citation_count"] == 0

    # 6) DraftPaperSwarmTool rejects traversal in report_basename and surfaces novelty ABORT warning
    trav_swarm = asyncio.run(
        swarm_tool.execute(
            "Ethology",
            related_work=[{"title": "Ethology", "doi": "10.1000/be"}],
            output_dir=str(workspace / "out"),
            report_basename="../../escape",
            allow_mutation=True,
        )
    )
    assert "report_basename" in trav_swarm


def test_assess_paper_draft_excludes_references_section_and_flags_leading_empty_h2() -> (
    None
):
    ref = LiteratureResult(
        source="openalex",
        title="Tracking Rodents",
        authors=("Alice Smith",),
        year=2024,
        doi="10.1000/tr",
    )
    key = assign_citation_keys([ref])[0][0]

    # Leading H2 Abstract is empty (no H1 title) and ONLY ## References mentions [@key]
    draft_only_ref_section = f"""## Abstract

## Introduction
We analyze tracking without citing anything in the body.

## Methods
Standard protocol.

## Results
Results were TBD.

## Discussion
Limitations exist.

## References
1. `[@{key}]` Alice Smith. Tracking Rodents (2024), 95% accuracy.
"""
    audit = assess_paper_draft_quality(draft_only_ref_section, references=[ref])
    assert "abstract" in audit["empty_sections"]
    assert audit["metrics"]["citation_count"] == 0
    assert audit["metrics"]["numeric_claim_count"] == 0
    assert audit["uncited_references"]
    assert any("no in-text citations" in w.lower() for w in audit["warnings"])


def test_latex_export_handles_links_ordered_lists_verbatim_and_bibtex_idempotent_escape() -> (
    None
):
    paper = LiteratureResult(
        source="openalex",
        title=r"Already \{Escaped\} & 100% Clean",
        authors=("Alice Smith",),
        year=2024,
        venue="NeurIPS",
    )
    bib = build_bibtex([paper])
    assert r"\\{" not in bib
    assert r"\{Escaped\}" in bib
    assert r"100\%" in bib

    draft = PaperDraft(
        topic="Links & Lists",
        sections=[
            PaperSection(
                title="Methods",
                content=(
                    "See [Annolid](https://annolid.com/docs) for details.\n\n"
                    "1. First step\n"
                    "2. Second step\n\n"
                    "#### Implementation Detail\n"
                    "```python\nx = 1\n```"
                ),
            )
        ],
        references=[paper],
    )
    tex = format_draft_latex(draft)
    assert r"\href{https://annolid.com/docs}{Annolid}" in tex
    assert r"\begin{enumerate}" in tex
    assert r"\subsubsection{Implementation Detail}" in tex
    assert r"\begin{verbatim}" in tex


def test_media_workflow_mixin_open_research_paper_swarm_uses_extracted_pdf_title(
    tmp_path: Path, monkeypatch
) -> None:
    from qtpy import QtWidgets
    from annolid.gui.mixins.media_workflow_mixin import MediaWorkflowMixin

    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    pdf_path = tmp_path / "2401.12345v2.pdf"
    pdf_path.write_bytes(b"%PDF-1.7\n%%EOF\n")

    class _FakePdfManager:
        def get_pdf_state(self) -> dict[str, object]:
            return {"has_pdf": True, "title": pdf_path.name, "path": str(pdf_path)}

    class _FakePromptEdit:
        def __init__(self) -> None:
            self.placeholder = ""

        def setPlaceholderText(self, text: str) -> None:
            self.placeholder = text

    class _FakeChatWidget:
        def __init__(self) -> None:
            self.prompt = ""
            self.sent = False
            self.prompt_text_edit = _FakePromptEdit()

        def bot_set_chat_prompt(self, text: str) -> None:
            self.prompt = text

        def bot_send_chat_prompt(self) -> None:
            self.sent = True

    class _FakeChatManager:
        def __init__(self) -> None:
            self.ai_chat_widget = _FakeChatWidget()

        def show_chat_dock(self) -> None:
            return

    class _Host(MediaWorkflowMixin, QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.pdf_manager = _FakePdfManager()
            self.ai_chat_manager = _FakeChatManager()

    host = _Host()
    seen_default: list[str] = []

    monkeypatch.setattr(
        "annolid.services.pdf_parsing.extract_pdf_title",
        lambda _p: "Deep Multi-Animal Ethology",
    )

    def _fake_get_text(_parent, _title, _label, text=""):
        seen_default.append(text)
        return text, True

    monkeypatch.setattr(QtWidgets.QInputDialog, "getText", _fake_get_text)

    host.open_research_paper_swarm()
    assert seen_default == ["Deep Multi-Animal Ethology"]
    assert host.ai_chat_manager.ai_chat_widget.sent is True
    assert "Deep Multi-Animal Ethology" in host.ai_chat_manager.ai_chat_widget.prompt
    assert str(pdf_path) in host.ai_chat_manager.ai_chat_widget.prompt
    host.close()
