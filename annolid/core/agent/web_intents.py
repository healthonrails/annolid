from __future__ import annotations

from typing import FrozenSet, Tuple


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
