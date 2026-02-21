from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable, Optional


ContextCallback = Callable[[], dict[str, Any] | Awaitable[dict[str, Any]]]
PathCallback = Callable[[], str | Awaitable[str]]
ActionCallback = Callable[..., Any | Awaitable[Any]]


async def _run_callback(callback: Optional[ActionCallback], **kwargs: Any) -> str:
    if callback is None:
        return json.dumps({"error": "GUI action callback is not configured."})
    try:
        payload = callback(**kwargs)
        if asyncio.iscoroutine(payload):
            payload = await payload
        if payload is None:
            payload = {"ok": True}
        if isinstance(payload, str):
            return payload
        if isinstance(payload, dict):
            return json.dumps(payload)
        return json.dumps({"ok": True, "result": str(payload)})
    except Exception as exc:
        return json.dumps({"error": str(exc)})
