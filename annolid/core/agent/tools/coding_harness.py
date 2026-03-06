from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from annolid.core.agent.coding_harness import (
    CodingHarnessManager,
    get_coding_harness_manager,
)

from .function_base import FunctionTool


class CodingSessionStartTool(FunctionTool):
    def __init__(
        self,
        *,
        manager: Optional[CodingHarnessManager] = None,
        workspace: Optional[Path] = None,
    ) -> None:
        self._manager = manager
        self._workspace = Path(workspace) if workspace is not None else None
        self._origin_channel = "cli"
        self._origin_chat_id = "direct"

    def set_context(self, channel: str, chat_id: str) -> None:
        self._origin_channel = channel
        self._origin_chat_id = chat_id

    @property
    def name(self) -> str:
        return "coding_session_start"

    @property
    def description(self) -> str:
        return "Start a long-lived coding harness session backed by Codex CLI."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "label": {"type": "string"},
                "provider": {"type": "string"},
                "model": {"type": "string"},
                "workspace": {"type": "string"},
            },
            "required": ["task"],
        }

    async def execute(
        self,
        task: str,
        label: str = "",
        provider: str = "codex_cli",
        model: str = "codex-cli/gpt-5.1-codex",
        workspace: str = "",
        **kwargs: Any,
    ) -> str:
        del kwargs
        manager = self._manager or get_coding_harness_manager()
        reply = await manager.start(
            task=task,
            label=label or None,
            provider=provider,
            model=model,
            workspace=workspace or str(self._workspace or Path.cwd()),
            origin_channel=self._origin_channel,
            origin_chat_id=self._origin_chat_id,
        )
        return reply


class CodingSessionSendTool(FunctionTool):
    def __init__(self, *, manager: Optional[CodingHarnessManager] = None) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "coding_session_send"

    @property
    def description(self) -> str:
        return "Send another instruction to a long-lived coding harness session."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["session_id", "message"],
        }

    async def execute(self, session_id: str, message: str, **kwargs: Any) -> str:
        del kwargs
        manager = self._manager or get_coding_harness_manager()
        ok = await manager.send_message(session_id, message)
        if not ok:
            return json.dumps(
                {
                    "ok": False,
                    "error": "session_not_found_or_closed",
                    "session_id": session_id,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {"ok": True, "session_id": session_id, "queued": True},
            ensure_ascii=False,
        )


class CodingSessionPollTool(FunctionTool):
    def __init__(self, *, manager: Optional[CodingHarnessManager] = None) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "coding_session_poll"

    @property
    def description(self) -> str:
        return "Inspect status and recent transcript for a coding harness session."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
                "tail_messages": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["session_id"],
        }

    async def execute(
        self, session_id: str, tail_messages: int = 6, **kwargs: Any
    ) -> str:
        del kwargs
        manager = self._manager or get_coding_harness_manager()
        payload = await manager.poll(session_id, tail_messages=int(tail_messages))
        return json.dumps(payload, ensure_ascii=False)


class CodingSessionListTool(FunctionTool):
    def __init__(self, *, manager: Optional[CodingHarnessManager] = None) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "coding_session_list"

    @property
    def description(self) -> str:
        return "List active long-lived coding harness sessions."

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {}}

    async def execute(self, **kwargs: Any) -> str:
        del kwargs
        manager = self._manager or get_coding_harness_manager()
        rows = []
        for session_id, meta in manager.list_sessions().items():
            rows.append(
                {
                    "session_id": session_id,
                    "label": meta.label,
                    "status": meta.status,
                    "provider": meta.provider,
                    "model": meta.model,
                    "workspace": meta.workspace,
                    "turn_count": meta.turn_count,
                    "pending_messages": meta.pending_messages,
                }
            )
        return json.dumps({"ok": True, "sessions": rows}, ensure_ascii=False)


class CodingSessionCloseTool(FunctionTool):
    def __init__(self, *, manager: Optional[CodingHarnessManager] = None) -> None:
        self._manager = manager

    @property
    def name(self) -> str:
        return "coding_session_close"

    @property
    def description(self) -> str:
        return "Close a long-lived coding harness session after the current turn."

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "session_id": {"type": "string"},
            },
            "required": ["session_id"],
        }

    async def execute(self, session_id: str, **kwargs: Any) -> str:
        del kwargs
        manager = self._manager or get_coding_harness_manager()
        ok = await manager.close(session_id)
        return json.dumps(
            {"ok": bool(ok), "session_id": session_id},
            ensure_ascii=False,
        )


__all__ = [
    "CodingSessionStartTool",
    "CodingSessionSendTool",
    "CodingSessionPollTool",
    "CodingSessionListTool",
    "CodingSessionCloseTool",
]
