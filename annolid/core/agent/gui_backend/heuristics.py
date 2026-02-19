from __future__ import annotations

import re
from typing import List, Tuple


WEB_ACCESS_REFUSAL_HINTS: Tuple[str, ...] = (
    "don't have web browsing capabilities",
    "do not have web browsing capabilities",
    "don't have web search",
    "do not have web search",
    "don't have web search or web browsing capabilities",
    "do not have web search or web browsing capabilities",
    "cannot directly fetch urls",
    "can't directly fetch urls",
    "i cannot directly fetch urls",
    "i can't directly fetch urls",
    "cannot browse the web",
    "can't browse the web",
    "cannot access external websites",
    "can't access external websites",
    "cannot access the internet",
    "can't access the internet",
    "no browsing capability",
    "require an already-open web page",
    "requires an already-open web page",
    "no page is currently open",
)

KNOWLEDGE_GAP_HINTS: Tuple[str, ...] = (
    "i don't have access to",
    "i do not have access to",
    "i don't know",
    "i do not know",
    "i'm not sure",
    "i am not sure",
    "i cannot determine",
    "i can't determine",
    "you can check by",
    "check a website",
    "check an app",
    "can't check",
    "cannot check",
    "isn't configured",
    "is not configured",
    "not configured",
    "web search api",
    "api key",
    "in your browser",
)

OPEN_URL_SUGGESTION_HINTS: Tuple[str, ...] = (
    "open your browser",
    "in your browser",
    "search for",
    "go to ",
    "visit ",
    "check weather.gov",
    "check accuweather",
)

OPEN_PDF_SUGGESTION_HINTS: Tuple[str, ...] = (
    "open pdf",
    "open the pdf",
    "upload pdf",
    "share the pdf",
    "provide the pdf",
    "cannot access your local file",
    "can't access your local file",
    "cannot access local file",
    "can't access local file",
)

WEB_CONTEXT_HINTS: Tuple[str, ...] = (
    "this page",
    "current page",
    "open page",
    "web page",
    "website",
    "site",
    "browser",
    "tab",
)

PDF_CONTEXT_HINTS: Tuple[str, ...] = (
    "pdf",
    "document",
    "paper",
    "article",
    "manuscript",
)

EMBEDDED_SEARCH_URL_TEMPLATE = "https://html.duckduckgo.com/html/?q={query}"
EMBEDDED_SEARCH_SOURCE = "DuckDuckGo search results page (embedded web viewer)."


def contains_hint(text: str, hints: Tuple[str, ...]) -> bool:
    lowered = str(text or "").strip().lower()
    if not lowered:
        return False
    return any(h in lowered for h in hints)


def looks_like_url_request(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    if "http://" in lowered or "https://" in lowered or "www." in lowered:
        return True
    return bool(
        re.search(
            r"\b[a-z0-9][a-z0-9\-]{0,62}(?:\.[a-z0-9][a-z0-9\-]{0,62})+\b",
            lowered,
        )
    )


def should_attach_live_web_context(prompt: str) -> bool:
    return contains_hint(prompt, WEB_CONTEXT_HINTS) or looks_like_url_request(prompt)


def should_attach_live_pdf_context(prompt: str) -> bool:
    return contains_hint(prompt, PDF_CONTEXT_HINTS)


def looks_like_web_access_refusal(text: str) -> bool:
    return contains_hint(text, WEB_ACCESS_REFUSAL_HINTS)


def looks_like_knowledge_gap_response(text: str) -> bool:
    return contains_hint(text, KNOWLEDGE_GAP_HINTS)


def looks_like_open_url_suggestion(text: str) -> bool:
    return contains_hint(text, OPEN_URL_SUGGESTION_HINTS)


def looks_like_open_pdf_suggestion(text: str) -> bool:
    return contains_hint(text, OPEN_PDF_SUGGESTION_HINTS)


def extract_web_urls(text: str) -> List[str]:
    raw = str(text or "")
    if not raw:
        return []
    candidates = re.findall(r"https?://[^\s<>\"]+", raw, flags=re.IGNORECASE)
    urls: List[str] = []
    for item in candidates:
        cleaned = str(item or "").strip().rstrip(").,;!?")
        if cleaned and cleaned not in urls:
            urls.append(cleaned)
    return urls


def build_extractive_summary(
    text: str,
    *,
    max_sentences: int = 6,
    max_chars: int = 1200,
) -> str:
    source = " ".join(str(text or "").split()).strip()
    if not source:
        return ""
    chunks = re.split(r"(?<=[.!?])\s+", source)
    picked: List[str] = []
    total = 0
    for chunk in chunks:
        sentence = str(chunk or "").strip()
        if not sentence:
            continue
        if picked and total + 1 + len(sentence) > max_chars:
            break
        if not picked and len(sentence) > max_chars:
            picked.append(sentence[: max_chars - 3].rstrip() + "...")
            break
        picked.append(sentence)
        total += len(sentence) + 1
        if len(picked) >= max_sentences:
            break
    return " ".join(picked).strip()


def topic_tokens(text: str) -> List[str]:
    raw = re.findall(r"[a-zA-Z0-9_]+", str(text or "").lower())
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "to",
        "for",
        "of",
        "in",
        "on",
        "at",
        "is",
        "are",
        "my",
        "me",
        "your",
        "you",
        "check",
        "current",
        "latest",
        "today",
    }
    return [t for t in raw if len(t) > 2 and t not in stop]


def prompt_may_need_mcp(prompt: str) -> bool:
    text = str(prompt or "").lower()
    if not text:
        return False
    hints = (
        "http://",
        "https://",
        "www.",
        "browser",
        "navigate",
        "open website",
        "web page",
        "playwright",
        "mcp",
        "dom",
        "click",
        "scroll",
        "form",
        "search",
        "weather",
        "forecast",
        "temperature",
        "news",
        "price",
        "stock",
        "latest",
        "current",
        "live",
        "today",
    )
    return any(token in text for token in hints)
