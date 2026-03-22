"""Unit tests for annolid.services.chat_research service.

All tests are pure-Python: no Qt, no network.
"""

from __future__ import annotations

from annolid.services.chat_research import (
    ResearchIntent,
    detect_research_intent,
    format_literature_results_for_chat,
)
from annolid.services.literature_search import LiteratureResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_paper(
    title: str, source: str = "arxiv", year: int = 2024
) -> LiteratureResult:
    return LiteratureResult(
        source=source,
        title=title,
        year=year,
        abs_url=f"https://example.com/{title.replace(' ', '_')}",
        summary=f"Summary of {title}.",
    )


# ---------------------------------------------------------------------------
# detect_research_intent tests
# ---------------------------------------------------------------------------


def test_detect_research_intent_none_for_greeting() -> None:
    result = detect_research_intent("Hello, how are you?")
    assert result is None


def test_detect_research_intent_none_for_empty() -> None:
    result = detect_research_intent("")
    assert result is None


def test_detect_research_intent_none_for_annotation_question() -> None:
    result = detect_research_intent("What is the label for this shape?")
    assert result is None


def test_detect_research_intent_search_basic() -> None:
    result = detect_research_intent("find papers on mouse tracking")
    assert isinstance(result, ResearchIntent)
    assert result.kind == "search"
    assert "mouse tracking" in result.topic.lower()


def test_detect_research_intent_search_research_command() -> None:
    result = detect_research_intent("research animal behavior analysis")
    assert isinstance(result, ResearchIntent)
    assert result.kind == "search"
    assert "animal behavior analysis" in result.topic.lower()


def test_detect_research_intent_search_papers_about() -> None:
    result = detect_research_intent("papers about pose estimation in videos")
    assert isinstance(result, ResearchIntent)
    assert result.kind == "search"


def test_detect_research_intent_search_what_papers_exist() -> None:
    result = detect_research_intent("what papers exist on neuroscience and behavior?")
    assert isinstance(result, ResearchIntent)
    assert result.kind == "search"
    assert "neuroscience" in result.topic.lower()


def test_detect_research_intent_draft_write_paper() -> None:
    result = detect_research_intent("write a paper about deep learning for tracking")
    assert isinstance(result, ResearchIntent)
    assert result.kind == "draft"
    assert "deep learning" in result.topic.lower()


def test_detect_research_intent_draft_generate_paper() -> None:
    result = detect_research_intent("generate a research paper on zebrafish behavior")
    assert isinstance(result, ResearchIntent)
    assert result.kind == "draft"
    assert "zebrafish" in result.topic.lower()


def test_detect_research_intent_draft_compose() -> None:
    result = detect_research_intent("compose a paper about pose estimation")
    assert isinstance(result, ResearchIntent)
    assert result.kind == "draft"


def test_detect_research_intent_topic_strip_trailing_punctuation() -> None:
    result = detect_research_intent("research neuroscience.")
    assert isinstance(result, ResearchIntent)
    assert not result.topic.endswith(".")


def test_detect_research_intent_topic_strip_please() -> None:
    result = detect_research_intent("find papers on tracking please")
    assert isinstance(result, ResearchIntent)
    assert "please" not in result.topic.lower()


# ---------------------------------------------------------------------------
# format_literature_results_for_chat tests
# ---------------------------------------------------------------------------


def test_format_literature_results_empty() -> None:
    msg = format_literature_results_for_chat([])
    assert (
        "no papers found" in msg.lower()
        or "no results" in msg.lower()
        or "no" in msg.lower()
    )


def test_format_literature_results_empty_with_topic() -> None:
    msg = format_literature_results_for_chat([], topic="animal tracking")
    assert "animal tracking" in msg.lower()


def test_format_literature_results_basic_two_papers() -> None:
    papers = [
        _make_paper("Tracking Mice with Deep Learning"),
        _make_paper("Pose Estimation Survey"),
    ]
    msg = format_literature_results_for_chat(papers)
    assert "1." in msg
    assert "2." in msg
    assert "Tracking Mice" in msg
    assert "Pose Estimation" in msg


def test_format_literature_results_source_badge() -> None:
    papers = [_make_paper("Paper One", source="openalex")]
    msg = format_literature_results_for_chat(papers)
    assert "openalex" in msg


def test_format_literature_results_topic_in_header() -> None:
    papers = [_make_paper("Some Paper")]
    msg = format_literature_results_for_chat(papers, topic="neuroscience")
    assert "neuroscience" in msg


def test_format_literature_results_max_items_truncation() -> None:
    papers = [_make_paper(f"Paper {i}") for i in range(15)]
    msg = format_literature_results_for_chat(papers, max_items=5)
    # Only 5 items numbered
    assert "5." in msg
    assert "6." not in msg
    # Should note there are more
    assert "more" in msg.lower()
