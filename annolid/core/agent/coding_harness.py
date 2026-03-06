from __future__ import annotations

import asyncio
import inspect
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Event as ThreadEvent
from typing import Any, Awaitable, Callable, Dict, List, Optional

from annolid.core.agent.providers.background_chat import run_codex_cli_chat
from annolid.core.agent.session_manager import AgentSessionManager
from annolid.utils.llm_settings import default_settings

HarnessAnnounceCallback = Callable[[str, str, str, str, str], Awaitable[None] | None]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class CodingHarnessSession:
    session_id: str
    label: str
    provider: str
    model: str
    workspace: str
    origin_channel: str
    origin_chat_id: str
    status: str = "idle"
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_error: str = ""
    last_response: str = ""
    turn_count: int = 0
    pending_messages: int = 0
    inbox: asyncio.Queue[Optional[str]] = field(default_factory=asyncio.Queue)
    worker_task: Optional[asyncio.Task[None]] = None
    close_requested: bool = False
    active_cancel_event: Optional[ThreadEvent] = None


class CodingHarnessManager:
    """ACP-style long-lived coding sessions backed by resumable Codex CLI turns."""

    def __init__(
        self,
        *,
        session_manager: Optional[AgentSessionManager] = None,
        announce_callback: Optional[HarnessAnnounceCallback] = None,
        invoke_turn: Optional[Callable[..., tuple[str, str]]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._session_manager = session_manager or AgentSessionManager()
        self._announce_callback = announce_callback
        self._invoke_turn = invoke_turn or run_codex_cli_chat
        self._settings = settings or default_settings()
        self._sessions: Dict[str, CodingHarnessSession] = {}
        self._lock = asyncio.Lock()

    def set_announce_callback(
        self, callback: Optional[HarnessAnnounceCallback]
    ) -> Optional[HarnessAnnounceCallback]:
        previous = self._announce_callback
        self._announce_callback = callback
        return previous

    async def start_session(
        self,
        *,
        task: str,
        label: Optional[str] = None,
        provider: str = "codex_cli",
        model: str = "codex-cli/gpt-5.1-codex",
        workspace: str = "",
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> CodingHarnessSession:
        prompt = str(task or "").strip()
        if not prompt:
            raise ValueError("task cannot be empty")
        display_label = label or (prompt[:40] + ("..." if len(prompt) > 40 else ""))
        meta = await self.create_session(
            label=display_label,
            provider=provider,
            model=model,
            workspace=workspace,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
        )
        meta.pending_messages += 1
        meta.updated_at = _now_iso()
        await meta.inbox.put(prompt)
        self._persist_meta(meta)
        return meta

    async def create_session(
        self,
        *,
        label: Optional[str] = None,
        provider: str = "codex_cli",
        model: str = "codex-cli/gpt-5.1-codex",
        workspace: str = "",
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> CodingHarnessSession:
        workspace_path = str(Path(workspace or ".").expanduser().resolve())
        session_id = f"code_{uuid.uuid4().hex[:8]}"
        display_label = str(label or "ACP Session").strip() or "ACP Session"
        meta = CodingHarnessSession(
            session_id=session_id,
            label=display_label,
            provider=str(provider or "codex_cli").strip().lower(),
            model=str(model or "codex-cli/gpt-5.1-codex").strip(),
            workspace=workspace_path,
            origin_channel=origin_channel,
            origin_chat_id=origin_chat_id,
            status="idle",
            pending_messages=0,
        )
        meta.worker_task = asyncio.create_task(self._run_session(meta))
        async with self._lock:
            self._sessions[session_id] = meta
        self._persist_meta(meta)
        return meta

    async def start(
        self,
        *,
        task: str,
        label: Optional[str] = None,
        provider: str = "codex_cli",
        model: str = "codex-cli/gpt-5.1-codex",
        workspace: str = "",
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
    ) -> str:
        try:
            meta = await self.start_session(
                task=task,
                label=label,
                provider=provider,
                model=model,
                workspace=workspace,
                origin_channel=origin_channel,
                origin_chat_id=origin_chat_id,
            )
        except ValueError as exc:
            return f"Error: {exc}"
        return (
            f"Coding harness [{meta.label}] started (id: {meta.session_id}, "
            f"provider: {meta.provider})."
        )

    async def send_message(self, session_id: str, message: str) -> bool:
        msg = str(message or "").strip()
        if not msg:
            return False
        async with self._lock:
            meta = self._sessions.get(str(session_id or "").strip())
        if meta is None or meta.close_requested:
            return False
        meta.pending_messages += 1
        meta.updated_at = _now_iso()
        self._persist_meta(meta)
        await meta.inbox.put(msg)
        return True

    def list_sessions(self) -> Dict[str, CodingHarnessSession]:
        return dict(self._sessions)

    def get_session(self, session_id: str) -> Optional[CodingHarnessSession]:
        return self._sessions.get(str(session_id or "").strip())

    def get_session_transcript(self, session_id: str) -> List[Dict[str, Any]]:
        session = self._session_manager.get_or_create(self._transcript_key(session_id))
        return [dict(m) for m in session.messages]

    async def poll(
        self,
        session_id: str,
        *,
        tail_messages: int = 6,
    ) -> Dict[str, Any]:
        meta = self.get_session(session_id)
        if meta is None:
            return {"ok": False, "error": "session_not_found", "session_id": session_id}
        transcript = self.get_session_transcript(session_id)
        tail = transcript[-max(1, int(tail_messages)) :]
        return {
            "ok": True,
            "session_id": meta.session_id,
            "label": meta.label,
            "status": meta.status,
            "provider": meta.provider,
            "model": meta.model,
            "workspace": meta.workspace,
            "turn_count": meta.turn_count,
            "pending_messages": meta.pending_messages,
            "last_response": meta.last_response,
            "last_error": meta.last_error,
            "close_requested": bool(meta.close_requested),
            "tail_messages": tail,
        }

    async def close(self, session_id: str) -> bool:
        meta = self.get_session(session_id)
        if meta is None:
            return False
        meta.close_requested = True
        if meta.active_cancel_event is not None:
            meta.active_cancel_event.set()
        meta.updated_at = _now_iso()
        self._persist_meta(meta)
        await meta.inbox.put(None)
        return True

    async def abort(self, session_id: str) -> bool:
        meta = self.get_session(session_id)
        if meta is None or meta.active_cancel_event is None:
            return False
        meta.active_cancel_event.set()
        meta.updated_at = _now_iso()
        self._persist_meta(meta)
        return True

    async def _run_session(self, meta: CodingHarnessSession) -> None:
        while True:
            message = await meta.inbox.get()
            if message is None:
                meta.status = "closed"
                meta.updated_at = _now_iso()
                self._persist_meta(meta)
                return

            meta.pending_messages = max(0, meta.pending_messages - 1)
            meta.status = "running"
            meta.updated_at = _now_iso()
            self._append_transcript(meta.session_id, role="user", content=message)
            self._persist_meta(meta)
            cancel_event = ThreadEvent()
            meta.active_cancel_event = cancel_event

            try:
                invoke_kwargs = {
                    "prompt": message,
                    "image_path": "",
                    "model": meta.model,
                    "provider_name": meta.provider,
                    "settings": self._settings,
                    "load_history_messages": lambda: self.get_session_transcript(
                        meta.session_id
                    ),
                    "session_id": self._provider_session_id(meta.session_id),
                    "runtime": "acp",
                    "timeout_s": 600.0,
                    "max_tokens": 4096,
                }
                try:
                    signature = inspect.signature(self._invoke_turn)
                except (TypeError, ValueError):
                    signature = None
                accepts_kwargs = bool(
                    signature is not None
                    and any(
                        param.kind == inspect.Parameter.VAR_KEYWORD
                        for param in signature.parameters.values()
                    )
                )
                if (
                    signature is None
                    or "cancel_event" in signature.parameters
                    or accepts_kwargs
                ):
                    invoke_kwargs["cancel_event"] = cancel_event
                _user_prompt, text = await asyncio.to_thread(
                    self._invoke_turn,
                    **invoke_kwargs,
                )
                meta.turn_count += 1
                meta.status = "idle"
                meta.last_error = ""
                meta.last_response = str(text or "").strip()
                self._append_transcript(
                    meta.session_id, role="assistant", content=meta.last_response
                )
                await self._announce(meta, meta.last_response)
            except Exception as exc:
                if cancel_event.is_set():
                    meta.status = "idle"
                    meta.last_error = ""
                    meta.last_response = ""
                else:
                    meta.status = "error"
                    meta.last_error = str(exc)
                    meta.last_response = ""
                    await self._announce(meta, meta.last_error)
            finally:
                meta.active_cancel_event = None
                meta.updated_at = _now_iso()
                self._persist_meta(meta)

    async def _announce(self, meta: CodingHarnessSession, text: str) -> None:
        if self._announce_callback is None:
            return
        result = self._announce_callback(
            meta.session_id,
            meta.status,
            meta.label,
            str(text or ""),
            f"{meta.origin_channel}:{meta.origin_chat_id}",
        )
        if asyncio.iscoroutine(result):
            await result

    def _append_transcript(self, session_id: str, *, role: str, content: str) -> None:
        session = self._session_manager.get_or_create(self._transcript_key(session_id))
        session.add_message({"role": role, "content": str(content or "").strip()})
        self._session_manager.save(session)

    def _persist_meta(self, meta: CodingHarnessSession) -> None:
        self._session_manager.update_session_metadata(
            self._transcript_key(meta.session_id),
            {
                "kind": "acp",
                "runtime": "acp",
                "harness_session_id": meta.session_id,
                "label": meta.label,
                "provider": meta.provider,
                "model": meta.model,
                "workspace": meta.workspace,
                "status": meta.status,
                "turn_count": meta.turn_count,
                "pending_messages": meta.pending_messages,
                "close_requested": bool(meta.close_requested),
                "last_error": meta.last_error,
                "updated_at": meta.updated_at,
            },
        )

    @staticmethod
    def _provider_session_id(session_id: str) -> str:
        return f"acp:{session_id}"

    @staticmethod
    def _transcript_key(session_id: str) -> str:
        return f"acp:{session_id}"


_HARNESS_MANAGER: Optional[CodingHarnessManager] = None
_HARNESS_MANAGER_LOOP_ID: Optional[int] = None


def get_coding_harness_manager() -> CodingHarnessManager:
    global _HARNESS_MANAGER, _HARNESS_MANAGER_LOOP_ID
    loop_id: Optional[int]
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None
    if _HARNESS_MANAGER is None or (
        _HARNESS_MANAGER_LOOP_ID is not None
        and loop_id is not None
        and _HARNESS_MANAGER_LOOP_ID != loop_id
    ):
        _HARNESS_MANAGER = CodingHarnessManager()
        _HARNESS_MANAGER_LOOP_ID = loop_id
    return _HARNESS_MANAGER
