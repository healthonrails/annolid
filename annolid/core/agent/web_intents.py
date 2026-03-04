from __future__ import annotations

import re
from typing import FrozenSet, Iterable, Tuple


WEATHER_INTENT_TOKENS: FrozenSet[str] = frozenset(
    {
        "weather",
        "forecast",
        "temperature",
    }
)

LIVE_WEB_INTENT_TOKENS: FrozenSet[str] = frozenset(
    set(WEATHER_INTENT_TOKENS)
    | {
        "news",
        "price",
        "stock",
        "latest",
        "current",
        "live",
        "today",
    }
)

LIVE_WEB_INTENT_HINTS: Tuple[str, ...] = tuple(sorted(LIVE_WEB_INTENT_TOKENS))

FAST_WEB_DATA_TOKENS: FrozenSet[str] = frozenset(
    {
        "stock",
        "stocks",
        "price",
        "prices",
        "quote",
        "quotes",
        "ticker",
        "tickers",
        "market",
        "markets",
        "nasdaq",
        "nyse",
        "crypto",
        "forex",
        "etf",
        "index",
        "indices",
        "trading",
        "shares",
    }
)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9_]+")


def tokenize_intent_text(text: str) -> Tuple[str, ...]:
    return tuple(_TOKEN_RE.findall(str(text or "").lower()))


def has_live_web_intent(tokens_or_text: Iterable[str] | str) -> bool:
    if isinstance(tokens_or_text, str):
        tokens = tokenize_intent_text(tokens_or_text)
    else:
        tokens = tuple(str(tok or "").lower() for tok in tokens_or_text)
    return bool(set(tokens).intersection(LIVE_WEB_INTENT_TOKENS))


def has_fast_web_data_intent(tokens_or_text: Iterable[str] | str) -> bool:
    if isinstance(tokens_or_text, str):
        tokens = tokenize_intent_text(tokens_or_text)
    else:
        tokens = tuple(str(tok or "").lower() for tok in tokens_or_text)
    return bool(set(tokens).intersection(FAST_WEB_DATA_TOKENS))
