"""Service wrapper for GUI chat Dreaming actions."""

from __future__ import annotations

from typing import Any, Dict

from annolid.core.agent.dream_memory import DreamMemoryManager
from annolid.infrastructure.agent_config import load_agent_config as load_config
from annolid.infrastructure.agent_workspace import get_agent_workspace_path


def run_chat_dream_action(*, action: str = "run", run_id: str = "") -> Dict[str, Any]:
    workspace = get_agent_workspace_path()
    manager = DreamMemoryManager(workspace)
    normalized_action = str(action or "run").strip().lower()

    if normalized_action == "run":
        cfg = load_config()
        dream_cfg = getattr(getattr(cfg, "agents", None), "defaults", None)
        dream_cfg = getattr(dream_cfg, "dream", None)
        max_batch_entries = int(
            getattr(dream_cfg, "max_batch_entries", 50) if dream_cfg else 50
        )
        initialize_cursor_to_end = bool(
            getattr(dream_cfg, "initialize_cursor_to_end", True) if dream_cfg else True
        )
        result = manager.run(
            max_batch_entries=max_batch_entries,
            initialize_cursor_to_end=initialize_cursor_to_end,
        )
        return {
            "ok": bool(result.ok),
            "result": result.message,
            "payload": result.to_dict(),
        }

    if normalized_action == "status":
        return {"ok": True, "result": manager.format_status()}

    if normalized_action == "help":
        return {"ok": True, "result": manager.format_help()}

    if normalized_action == "log":
        return {
            "ok": True,
            "result": manager.format_run_log(str(run_id or "").strip()),
        }

    if normalized_action == "restore":
        selected = str(run_id or "").strip()
        if not selected:
            return {"ok": True, "result": manager.format_restore_list(limit=10)}
        result = manager.restore(selected)
        return {
            "ok": bool(result.ok),
            "result": result.message,
            "payload": result.to_dict(),
        }

    return {"ok": False, "error": f"Unsupported dream action: {normalized_action}"}


def run_chat_dream_run_for_workspace(
    *,
    workspace: str,
    max_batch_entries: int,
    initialize_cursor_to_end: bool,
) -> str:
    manager = DreamMemoryManager(str(workspace or "").strip())
    result = manager.run(
        max_batch_entries=max(1, int(max_batch_entries)),
        initialize_cursor_to_end=bool(initialize_cursor_to_end),
    )
    return str(result.message)


__all__ = ["run_chat_dream_action", "run_chat_dream_run_for_workspace"]
