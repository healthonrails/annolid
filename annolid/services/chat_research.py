"""Research intent detection and formatting helpers for the Annolid chat system.

This module is deliberately stateless and free of Qt/network dependencies so
it can be unit-tested in isolation.

Public API
----------
- ``ResearchIntent``              — dataclass describing a detected intent.
- ``detect_research_intent``      — extract a research intent from a raw message.
- ``format_literature_results_for_chat``  — render search results as markdown.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from annolid.services.literature_search import LiteratureResult

# ---------------------------------------------------------------------------
# Intent dataclass
# ---------------------------------------------------------------------------


@dataclass
class ResearchIntent:
    """A parsed research intent extracted from a chat message.

    Attributes
    ----------
    kind:
        ``"search"``  — find papers on a topic.
        ``"draft"``   — find papers AND draft a paper.
    topic:
        The research topic string (stripped from the message).
    max_results:
        Suggested number of literature results to retrieve (default: 8).
    raw_message:
        The original unmodified message.
    """

    kind: str  # "search" | "draft"
    topic: str
    max_results: int = 8
    raw_message: str = field(default="", repr=False)


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

# Ordered from most specific to least specific.
# Each tuple is (compiled_pattern, intent_kind, topic_group_name).
_DRAFT_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:write|draft|generate|produce|create|compose)\s+a?\s*(?:research\s+)?paper\s+"
        r"(?:about|on|regarding|covering|for|titled?)\s+(?P<topic>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:write|draft|generate|produce|create|compose)\s+(?:a\s+)?paper\s+(?:about|on|regarding)\s+(?P<topic>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:help\s+me\s+)?(?:write|draft)\s+(?:a\s+)?(?:paper|article|manuscript)\s+(?:about|on)\s+(?P<topic>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"paper\s+(?:on|about|regarding)\s+(?P<topic>.+)",
        re.IGNORECASE,
    ),
]

_SEARCH_PATTERNS: list[re.Pattern[str]] = [
    re.compile(
        r"(?:search|find|look\s+up|look\s+for|get|fetch)\s+(?:papers?|articles?|literature|references?|citations?)\s+"
        r"(?:on|about|for|regarding|related\s+to|covering)\s+(?P<topic>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:search|find|look\s+for)\s+(?:papers?|literature)\s+(?P<topic>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"research\s+(?P<topic>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:papers?|literature|articles?)\s+(?:on|about|regarding)\s+(?P<topic>.+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:what\s+(?:papers?|research|literature)\s+(?:exist|is\s+there|is\s+available)\s+(?:on|about|for))\s+(?P<topic>.+)",
        re.IGNORECASE,
    ),
]

# Clutter that should be stripped from the end of a detected topic
_TRAILING_NOISE = re.compile(
    r"[\s,.?!;:]+$|(?:\s+please\.?)$",
    re.IGNORECASE,
)

_MAX_TOPIC_LEN = 200


def _clean_topic(raw: str) -> str:
    text = str(raw or "").strip()
    text = _TRAILING_NOISE.sub("", text)
    return text[:_MAX_TOPIC_LEN].strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def detect_research_intent(message: str) -> ResearchIntent | None:
    """Try to extract a research intent from a raw chat message.

    Returns ``None`` if the message does not match any research pattern.

    Detection order: draft patterns are tried first (they are a superset
    of search — if the user wants a draft they also want a search), then
    search-only patterns.
    """
    text = str(message or "").strip()
    if not text:
        return None

    # Draft patterns
    for pattern in _DRAFT_PATTERNS:
        match = pattern.search(text)
        if match:
            topic = _clean_topic(match.group("topic"))
            if topic:
                return ResearchIntent(
                    kind="draft",
                    topic=topic,
                    raw_message=text,
                )

    # Search-only patterns
    for pattern in _SEARCH_PATTERNS:
        match = pattern.search(text)
        if match:
            topic = _clean_topic(match.group("topic"))
            if topic:
                return ResearchIntent(
                    kind="search",
                    topic=topic,
                    raw_message=text,
                )

    return None


def format_literature_results_for_chat(
    results: Sequence[LiteratureResult],
    *,
    topic: str = "",
    max_items: int = 10,
) -> str:
    """Render a list of ``LiteratureResult`` objects as a markdown message.

    Designed to be posted directly into the chat pane.
    Returns a "no results" message for empty input.
    """
    if not results:
        topic_suffix = f" for **{topic}**" if topic else ""
        return (
            f"🔍 No papers found{topic_suffix}.\n\n"
            "Try a different query, or check your network connection."
        )

    lines: list[str] = []
    header = f"🔬 Found **{len(results)}** paper(s)"
    if topic:
        header += f" on **{topic}**"
    lines.append(header + ":\n")

    for i, paper in enumerate(results[:max_items], 1):
        year = f" ({paper.year})" if paper.year else ""
        source_badge = f"`{paper.source}`" if paper.source else ""
        url = paper.pdf_url or paper.abs_url or paper.id_url or ""
        if url:
            title_md = f"[{paper.title}]({url})"
        else:
            title_md = paper.title
        summary_snip = ""
        if paper.summary:
            snip = paper.summary[:120].rstrip()
            summary_snip = f"\n   _{snip}…_"
        lines.append(f"{i}. {title_md}{year} {source_badge}{summary_snip}")

    if len(results) > max_items:
        lines.append(
            f"\n…and {len(results) - max_items} more. "
            "Click **Generate Draft** to incorporate all results."
        )

    return "\n".join(lines)


__all__ = [
    "ResearchIntent",
    "detect_research_intent",
    "format_literature_results_for_chat",
]
