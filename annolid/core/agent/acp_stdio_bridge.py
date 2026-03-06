from __future__ import annotations

import asyncio
import json
import time
import uuid
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, TextIO

from .acp import ACPRuntimeManager, get_acp_runtime_manager

PROTOCOL_NAME = "annolid.acp.stdio"
PROTOCOL_VERSION = "1.0"
OPENCLAW_COMPAT_PROTOCOL = "acp"
_DEFAULT_MAX_CLIENT_SESSIONS = 256
_DEFAULT_IDLE_TTL_SECONDS = 24 * 60 * 60


@dataclass
class ACPClientSession:
    session_id: str
    harness_session_id: str
    cwd: str
    session_key: str
    created_at: float
    last_touched_at: float
    active_request_id: Optional[str] = None
    pending_prompt: Optional[asyncio.Future[Dict[str, Any]]] = None


def _now_monotonic() -> float:
    return time.monotonic()


class ACPBridgeError(Exception):
    def __init__(
        self,
        code: int,
        message: str,
        *,
        data: Optional[Mapping[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.data = dict(data or {})


class ACPStdioBridge:
    """JSON-RPC bridge for persistent ACP sessions over stdio."""

    def __init__(
        self,
        *,
        manager: Optional[ACPRuntimeManager] = None,
        workspace: Optional[str | Path] = None,
        max_client_sessions: int = _DEFAULT_MAX_CLIENT_SESSIONS,
        idle_ttl_seconds: float = _DEFAULT_IDLE_TTL_SECONDS,
        input_stream: Optional[TextIO] = None,
        output_stream: Optional[TextIO] = None,
    ) -> None:
        self._manager = manager or get_acp_runtime_manager()
        self._workspace = str(Path(workspace or Path.cwd()).expanduser().resolve())
        self._input = input_stream or sys.stdin
        self._output = output_stream or sys.stdout
        self._write_lock = asyncio.Lock()
        self._shutdown_requested = False
        self._sessions: Dict[str, ACPClientSession] = {}
        self._max_client_sessions = max(1, int(max_client_sessions))
        self._idle_ttl_seconds = max(1.0, float(idle_ttl_seconds))

    async def serve(self) -> int:
        previous_callback = self._manager.set_announce_callback(
            self._handle_session_announcement
        )
        try:
            while not self._shutdown_requested:
                line = await asyncio.to_thread(self._input.readline)
                if line == "":
                    break
                text = str(line or "").strip()
                if not text:
                    continue
                try:
                    payload = json.loads(text)
                except Exception:
                    await self._emit_json(
                        self._error_response(
                            request_id=None,
                            code=-32700,
                            message="Parse error",
                        )
                    )
                    continue
                response = await self.handle_request(payload)
                if response is not None:
                    await self._emit_json(response)
        finally:
            self._manager.set_announce_callback(previous_callback)
        return 0

    async def handle_request(
        self, payload: Mapping[str, Any]
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(payload, Mapping):
            return self._error_response(
                request_id=None, code=-32600, message="Invalid Request"
            )
        request_id = payload.get("id")
        method = str(payload.get("method") or "").strip()
        if not method:
            return self._error_response(
                request_id=request_id,
                code=-32600,
                message="Invalid Request",
            )
        params = payload.get("params", {})
        if params is None:
            params = {}
        if not isinstance(params, Mapping):
            return self._error_response(
                request_id=request_id,
                code=-32602,
                message="Invalid params",
            )

        try:
            result = await self._dispatch(method, dict(params))
        except ACPBridgeError as exc:
            return self._error_response(
                request_id=request_id,
                code=exc.code,
                message=exc.message,
                data=exc.data,
            )
        except Exception as exc:  # pragma: no cover - defensive
            return self._error_response(
                request_id=request_id,
                code=-32000,
                message="Internal error",
                data={"detail": str(exc)},
            )
        if request_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": result,
        }

    async def _dispatch(self, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        method_name = self._normalize_method_name(method)
        normalized_params = self._normalize_params(method_name, params)
        if method_name == "initialize":
            return {
                "protocol": PROTOCOL_NAME,
                "protocol_name": PROTOCOL_NAME,
                "protocols": [PROTOCOL_NAME, OPENCLAW_COMPAT_PROTOCOL],
                "version": PROTOCOL_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "runtime": "acp",
                "transport": "stdio",
                "workspace": self._workspace,
                "capabilities": {
                    "methods": [
                        "initialize",
                        "newSession",
                        "loadSession",
                        "prompt",
                        "cancel",
                        "listSessions",
                        "health",
                        "sessions.spawn",
                        "sessions_spawn",
                        "sessions.send",
                        "sessions_send",
                        "sessions.poll",
                        "sessions_poll",
                        "sessions.list",
                        "sessions_list",
                        "sessions.close",
                        "sessions_close",
                        "shutdown",
                    ],
                    "notifications": ["sessions.updated", "session.updated"],
                    "runtimes": ["acp"],
                    "session_runtime": "acp",
                    "loadSession": True,
                    "promptCapabilities": {
                        "image": False,
                        "audio": False,
                        "embeddedContext": False,
                    },
                    "sessionCapabilities": {"list": {}},
                },
            }
        if method_name == "newsession":
            session = await self._create_acp_session(
                normalized_params, reuse_session_id=None
            )
            return {"sessionId": session.session_id}
        if method_name == "loadsession":
            requested_session_id = self._required_text(normalized_params, "session_id")
            session = await self._create_acp_session(
                normalized_params,
                reuse_session_id=requested_session_id,
            )
            return {}
        if method_name == "prompt":
            session_id = self._required_text(normalized_params, "session_id")
            session = self._get_client_session(session_id)
            if session is None:
                raise ACPBridgeError(
                    -32004,
                    "Session not found",
                    data={"session_id": session_id},
                )
            prompt_text = self._prompt_text(normalized_params.get("prompt"))
            if session.pending_prompt is not None and not session.pending_prompt.done():
                session.pending_prompt.cancel()
            request_id = uuid.uuid4().hex
            future: asyncio.Future[Dict[str, Any]] = (
                asyncio.get_running_loop().create_future()
            )
            session.pending_prompt = future
            session.active_request_id = request_id
            session.last_touched_at = _now_monotonic()
            ok = await self._manager.send_message(
                session.harness_session_id, prompt_text
            )
            if not ok:
                session.pending_prompt = None
                session.active_request_id = None
                raise ACPBridgeError(
                    -32004,
                    "Session not found or closed",
                    data={"session_id": session_id},
                )
            result = await future
            session.pending_prompt = None
            session.active_request_id = None
            session.last_touched_at = _now_monotonic()
            return result
        if method_name == "cancel":
            session_id = self._required_text(normalized_params, "session_id")
            session = self._get_client_session(session_id)
            if session is None:
                return {}
            await self._manager.abort(session.harness_session_id)
            if session.pending_prompt is not None and not session.pending_prompt.done():
                session.pending_prompt.set_result({"stopReason": "cancelled"})
            return {}
        if method_name == "listsessions":
            rows = []
            for session in self._sessions.values():
                meta = self._manager.get_session(session.harness_session_id)
                title = meta.label if meta is not None else session.session_key
                rows.append(
                    {
                        "sessionId": session.session_id,
                        "cwd": session.cwd,
                        "title": title,
                        "updatedAt": None,
                        "_meta": {
                            "sessionKey": session.session_key,
                            "runtime": "acp",
                            "harnessSessionId": session.harness_session_id,
                        },
                    }
                )
            return {"sessions": rows, "nextCursor": None}
        if method_name == "health":
            return {
                "ok": True,
                "protocol": PROTOCOL_NAME,
                "protocol_name": PROTOCOL_NAME,
                "version": PROTOCOL_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "active_sessions": len(self._manager.list_sessions()),
            }
        if method_name == "sessions.spawn":
            task = str(normalized_params.get("task") or "").strip()
            if not task:
                raise ACPBridgeError(
                    -32602,
                    "Invalid params",
                    data={"field": "task", "reason": "task is required"},
                )
            meta = await self._manager.start_session(
                task=task,
                label=self._optional_text(normalized_params.get("label")),
                provider=self._optional_text(normalized_params.get("provider"))
                or "codex_cli",
                model=self._optional_text(normalized_params.get("model"))
                or "codex-cli/gpt-5.1-codex",
                workspace=self._optional_text(normalized_params.get("workspace"))
                or self._workspace,
                origin_channel=self._optional_text(
                    normalized_params.get("origin_channel")
                )
                or "acp-stdio",
                origin_chat_id=self._optional_text(
                    normalized_params.get("origin_chat_id")
                )
                or "external",
            )
            return self._session_payload(meta)
        if method_name == "sessions.send":
            session_id = self._required_text(normalized_params, "session_id")
            message = self._required_text(normalized_params, "message")
            ok = await self._manager.send_message(session_id, message)
            if not ok:
                raise ACPBridgeError(
                    -32004,
                    "Session not found or closed",
                    data={"session_id": session_id},
                )
            return {"ok": True, "session_id": session_id, "queued": True}
        if method_name == "sessions.poll":
            session_id = self._required_text(normalized_params, "session_id")
            payload = await self._manager.poll(
                session_id,
                tail_messages=int(normalized_params.get("tail_messages") or 6),
            )
            if not payload.get("ok", False):
                raise ACPBridgeError(
                    -32004,
                    "Session not found",
                    data={"session_id": session_id},
                )
            payload["sessionId"] = str(payload.get("session_id") or session_id)
            payload["turnCount"] = int(payload.get("turn_count") or 0)
            payload["pendingMessages"] = int(payload.get("pending_messages") or 0)
            payload["lastError"] = str(payload.get("last_error") or "")
            payload["closeRequested"] = bool(payload.get("close_requested") or False)
            payload["runtime"] = "acp"
            return payload
        if method_name == "sessions.list":
            rows = [
                self._session_payload(meta)
                for meta in self._manager.list_sessions().values()
            ]
            return {"ok": True, "sessions": rows}
        if method_name == "sessions.close":
            session_id = self._required_text(normalized_params, "session_id")
            client_session = self._get_client_session(session_id)
            target_session_id = (
                client_session.harness_session_id
                if client_session is not None
                else session_id
            )
            ok = await self._manager.close(target_session_id)
            if not ok:
                raise ACPBridgeError(
                    -32004,
                    "Session not found",
                    data={"session_id": session_id},
                )
            if client_session is not None:
                self._sessions.pop(client_session.session_id, None)
            return {"ok": True, "session_id": session_id, "closed": True}
        if method_name == "shutdown":
            self._shutdown_requested = True
            return {"ok": True, "shutdown": True}
        raise ACPBridgeError(-32601, "Method not found", data={"method": method})

    async def _handle_session_announcement(
        self,
        session_id: str,
        status: str,
        label: str,
        text: str,
        origin: str,
    ) -> None:
        client_session = self._find_client_session_by_harness_id(session_id)
        if client_session is not None:
            client_session.last_touched_at = _now_monotonic()
            if (
                client_session.pending_prompt is not None
                and not client_session.pending_prompt.done()
            ):
                stop_reason = "end_turn" if status == "idle" else "refusal"
                if status == "closed":
                    stop_reason = "cancelled"
                await self._emit_json(
                    {
                        "jsonrpc": "2.0",
                        "method": "session.update",
                        "params": {
                            "sessionId": client_session.session_id,
                            "update": {
                                "sessionUpdate": "agent_message_chunk",
                                "content": {"type": "text", "text": str(text or "")},
                            },
                        },
                    }
                )
                client_session.pending_prompt.set_result({"stopReason": stop_reason})
        await self._emit_json(
            {
                "jsonrpc": "2.0",
                "method": "sessions.updated",
                "params": {
                    "session_id": session_id,
                    "sessionId": session_id,
                    "status": status,
                    "label": label,
                    "text": str(text or ""),
                    "message": str(text or ""),
                    "origin": origin,
                    "runtime": "acp",
                },
            }
        )

    async def _emit_json(self, payload: Mapping[str, Any]) -> None:
        line = json.dumps(dict(payload), ensure_ascii=False)
        async with self._write_lock:
            self._output.write(line + "\n")
            self._output.flush()

    @staticmethod
    def _required_text(params: Mapping[str, Any], key: str) -> str:
        text = str(params.get(key) or "").strip()
        if text:
            return text
        raise ACPBridgeError(
            -32602,
            "Invalid params",
            data={"field": key, "reason": f"{key} is required"},
        )

    @staticmethod
    def _optional_text(value: Any) -> str:
        return str(value or "").strip()

    @staticmethod
    def _session_payload(meta: Any) -> Dict[str, Any]:
        return {
            "session_id": meta.session_id,
            "sessionId": meta.session_id,
            "label": meta.label,
            "status": meta.status,
            "provider": meta.provider,
            "model": meta.model,
            "workspace": meta.workspace,
            "runtime": "acp",
            "kind": "acp",
            "turn_count": int(meta.turn_count),
            "turnCount": int(meta.turn_count),
            "pending_messages": int(meta.pending_messages),
            "pendingMessages": int(meta.pending_messages),
            "created_at": meta.created_at,
            "createdAt": meta.created_at,
            "updated_at": meta.updated_at,
            "updatedAt": meta.updated_at,
            "last_error": meta.last_error,
            "lastError": meta.last_error,
            "close_requested": bool(meta.close_requested),
            "closeRequested": bool(meta.close_requested),
        }

    async def _create_acp_session(
        self, params: Mapping[str, Any], *, reuse_session_id: Optional[str]
    ) -> ACPClientSession:
        await self._reap_idle_client_sessions()
        existing = self._get_client_session(reuse_session_id or "")
        if existing is not None:
            return existing
        cwd = self._optional_text(params.get("cwd")) or self._workspace
        meta = self._meta_from_params(params)
        requested_key = self._optional_text(meta.get("sessionKey"))
        requested_label = self._optional_text(meta.get("sessionLabel"))
        label = (
            requested_label or self._optional_text(params.get("label")) or "ACP Session"
        )
        harness = await self._manager.create_session(
            label=label,
            provider=self._optional_text(params.get("provider")) or "codex_cli",
            model=self._optional_text(params.get("model")) or "codex-cli/gpt-5.1-codex",
            workspace=cwd,
            origin_channel="acp-stdio",
            origin_chat_id=reuse_session_id or "external",
        )
        session_id = reuse_session_id or uuid.uuid4().hex
        now = _now_monotonic()
        session = ACPClientSession(
            session_id=session_id,
            harness_session_id=harness.session_id,
            cwd=cwd,
            session_key=requested_key or f"acp:{session_id}",
            created_at=now,
            last_touched_at=now,
        )
        self._sessions[session.session_id] = session
        return session

    async def _reap_idle_client_sessions(self) -> None:
        if not self._sessions:
            return
        now = _now_monotonic()
        idle_before = now - self._idle_ttl_seconds
        stale_ids = [
            session_id
            for session_id, session in self._sessions.items()
            if self._is_client_session_idle(session)
            and session.last_touched_at <= idle_before
        ]
        for session_id in stale_ids:
            await self._drop_client_session(session_id)
        while len(self._sessions) >= self._max_client_sessions:
            oldest = self._oldest_idle_client_session_id()
            if oldest is None:
                raise ACPBridgeError(
                    -32005,
                    f"ACP session limit reached (max {self._max_client_sessions}).",
                )
            await self._drop_client_session(oldest)

    @staticmethod
    def _is_client_session_idle(session: ACPClientSession) -> bool:
        if session.active_request_id:
            return False
        return session.pending_prompt is None or session.pending_prompt.done()

    def _oldest_idle_client_session_id(self) -> Optional[str]:
        oldest: Optional[ACPClientSession] = None
        for session in self._sessions.values():
            if not self._is_client_session_idle(session):
                continue
            if oldest is None or session.last_touched_at < oldest.last_touched_at:
                oldest = session
        if oldest is None:
            return None
        return oldest.session_id

    async def _drop_client_session(self, session_id: str) -> None:
        session = self._sessions.pop(str(session_id or "").strip(), None)
        if session is None:
            return
        await self._manager.close(session.harness_session_id)

    @classmethod
    def _prompt_text(cls, value: Any) -> str:
        if isinstance(value, str):
            text = value.strip()
            if text:
                return text
        if isinstance(value, list):
            parts = []
            for item in value:
                if not isinstance(item, Mapping):
                    continue
                if str(item.get("type") or "").strip().lower() != "text":
                    continue
                text = cls._optional_text(item.get("text"))
                if text:
                    parts.append(text)
            if parts:
                return "\n".join(parts)
        raise ACPBridgeError(
            -32602,
            "Invalid params",
            data={"field": "prompt", "reason": "prompt text is required"},
        )

    @staticmethod
    def _meta_from_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
        meta = params.get("_meta", {})
        if isinstance(meta, Mapping):
            return meta
        return {}

    def _get_client_session(self, session_id: str) -> Optional[ACPClientSession]:
        session = self._sessions.get(str(session_id or "").strip())
        if session is not None:
            session.last_touched_at = _now_monotonic()
        return session

    def _find_client_session_by_harness_id(
        self, harness_session_id: str
    ) -> Optional[ACPClientSession]:
        for session in self._sessions.values():
            if session.harness_session_id == harness_session_id:
                return session
        return None

    @staticmethod
    def _normalize_method_name(method: str) -> str:
        raw = str(method or "").strip().lower()
        aliases = {
            "new_session": "newsession",
            "load_session": "loadsession",
            "list_sessions": "listsessions",
            "sessions_spawn": "sessions.spawn",
            "session.spawn": "sessions.spawn",
            "session_spawn": "sessions.spawn",
            "sessions_send": "sessions.send",
            "session.send": "sessions.send",
            "session_send": "sessions.send",
            "sessions_poll": "sessions.poll",
            "session.poll": "sessions.poll",
            "session_poll": "sessions.poll",
            "sessions_list": "sessions.list",
            "session.list": "sessions.list",
            "session_list": "sessions.list",
            "sessions_close": "sessions.close",
            "session.close": "sessions.close",
            "session_close": "sessions.close",
        }
        return aliases.get(raw, raw)

    @classmethod
    def _normalize_params(
        cls, method_name: str, params: Mapping[str, Any]
    ) -> Dict[str, Any]:
        out = dict(params)
        if method_name == "sessions.spawn":
            if "task" not in out:
                for key in ("prompt", "message", "instruction"):
                    text = cls._optional_text(out.get(key))
                    if text:
                        out["task"] = text
                        break
            if "origin_channel" not in out and "channel" in out:
                out["origin_channel"] = out.get("channel")
            if "origin_chat_id" not in out:
                for key in ("chat_id", "thread_id", "threadId"):
                    text = cls._optional_text(out.get(key))
                    if text:
                        out["origin_chat_id"] = text
                        break
        if method_name in {"sessions.send", "sessions.poll", "sessions.close"}:
            if "session_id" not in out:
                for key in ("sessionId", "id", "thread_id", "threadId"):
                    text = cls._optional_text(out.get(key))
                    if text:
                        out["session_id"] = text
                        break
        if method_name == "sessions.send" and "message" not in out:
            for key in ("prompt", "task", "instruction"):
                text = cls._optional_text(out.get(key))
                if text:
                    out["message"] = text
                    break
        if method_name == "sessions.poll" and "tail_messages" not in out:
            tail = out.get("tailMessages")
            if tail is not None:
                out["tail_messages"] = tail
        if (
            method_name in {"loadsession", "prompt", "cancel"}
            and "session_id" not in out
        ):
            for key in ("sessionId", "id"):
                text = cls._optional_text(out.get(key))
                if text:
                    out["session_id"] = text
                    break
        if method_name == "prompt" and "prompt" not in out:
            for key in ("message", "task", "instruction"):
                text = cls._optional_text(out.get(key))
                if text:
                    out["prompt"] = text
                    break
        return out

    @staticmethod
    def _error_response(
        *,
        request_id: Any,
        code: int,
        message: str,
        data: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        error: Dict[str, Any] = {"code": int(code), "message": str(message)}
        if data:
            error["data"] = dict(data)
        return {"jsonrpc": "2.0", "id": request_id, "error": error}


def run_stdio_acp_bridge(*, workspace: Optional[str | Path] = None) -> int:
    bridge = ACPStdioBridge(workspace=workspace)
    return asyncio.run(bridge.serve())
