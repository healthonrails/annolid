from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict


def run_awaitable_sync(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    # A loop is already running in this thread. Run the awaitable in a
    # dedicated worker thread to avoid nested-loop RuntimeError.
    with ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(asyncio.run, awaitable).result()


async def execute_direct_gui_command(
    *,
    prompt: str,
    parse_direct_gui_command: Callable[[str], Dict[str, Any]],
    route_direct_gui_command: Callable[..., Any],
    handlers: Dict[str, Callable[..., Any]],
) -> Dict[str, Any]:
    """Route direct command to the router and get back a result dictionary."""
    command = parse_direct_gui_command(prompt)
    if not command:
        return {"message": "", "payload": {}}
    res = await route_direct_gui_command(command, **handlers)
    if isinstance(res, dict):
        return res
    return {"message": str(res or ""), "payload": {}}
