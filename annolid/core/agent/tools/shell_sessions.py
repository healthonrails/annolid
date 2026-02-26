from __future__ import annotations

import asyncio
import contextlib
import json
import re
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from annolid.core.agent.observability import emit_governance_event

from .function_base import FunctionTool


@dataclass
class _ShellSession:
    session_id: str
    command: str
    cwd: str
    created_at: float
    process: asyncio.subprocess.Process
    status: str = "running"
    return_code: Optional[int] = None
    ended_at: Optional[float] = None
    output_lines: List[str] = field(default_factory=list)
    reader_tasks: List[asyncio.Task[Any]] = field(default_factory=list)
    waiter_task: Optional[asyncio.Task[Any]] = None
    timed_out: bool = False


class ShellSessionManager:
    """In-memory shell session registry for background process control."""

    def __init__(
        self, *, ttl_seconds: int = 3600, max_output_lines: int = 4000
    ) -> None:
        self.ttl_seconds = max(60, int(ttl_seconds))
        self.max_output_lines = max(200, int(max_output_lines))
        self._sessions: Dict[str, _ShellSession] = {}
        self._lock = asyncio.Lock()
        self._deny_patterns: List[str] = [
            r"\brm\s+-[rf]{1,2}\b",
            r"\bdel\s+/[fq]\b",
            r"\brmdir\s+/s\b",
            r"\b(format|mkfs|diskpart)\b",
            r"\bdd\s+if=",
            r">\s*/dev/sd",
            r"\b(shutdown|reboot|poweroff)\b",
            r":\(\)\s*\{.*\};\s*:",
        ]

    def _guard_command(self, command: str) -> str | None:
        cmd = str(command or "").strip().lower()
        if not cmd:
            return "empty_command"
        for pattern in self._deny_patterns:
            if re.search(pattern, cmd):
                return "dangerous_command"
        return None

    async def _append_output(self, session_id: str, line: str) -> None:
        async with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return
            sess.output_lines.append(str(line or ""))
            if len(sess.output_lines) > self.max_output_lines:
                overflow = len(sess.output_lines) - self.max_output_lines
                del sess.output_lines[:overflow]

    async def _read_stream(
        self,
        session_id: str,
        stream: Optional[asyncio.StreamReader],
        *,
        prefix: str = "",
    ) -> None:
        if stream is None:
            return
        try:
            while True:
                raw = await stream.readline()
                if not raw:
                    break
                text = raw.decode("utf-8", errors="replace").rstrip("\n")
                line = f"{prefix}{text}" if prefix else text
                await self._append_output(session_id, line)
        except Exception as exc:
            await self._append_output(session_id, f"[stream-error] {exc}")

    async def _wait_for_exit(
        self, session_id: str, process: asyncio.subprocess.Process, timeout_s: float
    ) -> None:
        try:
            if timeout_s > 0:
                rc = await asyncio.wait_for(process.wait(), timeout=timeout_s)
            else:
                rc = await process.wait()
            async with self._lock:
                sess = self._sessions.get(session_id)
                if sess is None:
                    return
                if sess.status == "killed":
                    sess.return_code = int(rc or 0)
                    sess.ended_at = time.time()
                    return
                sess.return_code = int(rc or 0)
                sess.status = "completed" if int(rc or 0) == 0 else "failed"
                sess.ended_at = time.time()
        except asyncio.TimeoutError:
            try:
                process.kill()
            except Exception:
                pass
            async with self._lock:
                sess = self._sessions.get(session_id)
                if sess is None:
                    return
                sess.timed_out = True
                sess.status = "timeout"
                sess.return_code = -9
                sess.ended_at = time.time()
        finally:
            async with self._lock:
                sess = self._sessions.get(session_id)
                if sess is not None:
                    tasks = list(sess.reader_tasks)
                else:
                    tasks = []
            for t in tasks:
                with contextlib.suppress(Exception):
                    await t

    async def start(
        self,
        *,
        command: str,
        cwd: str,
        timeout_s: float = 0.0,
    ) -> _ShellSession:
        blocked = self._guard_command(command)
        if blocked:
            raise ValueError(blocked)
        proc = await asyncio.create_subprocess_shell(
            command,
            cwd=cwd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        session_id = f"sh_{uuid.uuid4().hex[:12]}"
        now = time.time()
        session = _ShellSession(
            session_id=session_id,
            command=command,
            cwd=cwd,
            created_at=now,
            process=proc,
        )
        stdout_task = asyncio.create_task(self._read_stream(session_id, proc.stdout))
        stderr_task = asyncio.create_task(
            self._read_stream(session_id, proc.stderr, prefix="STDERR: ")
        )
        session.reader_tasks = [stdout_task, stderr_task]
        session.waiter_task = asyncio.create_task(
            self._wait_for_exit(session_id, proc, float(timeout_s))
        )
        async with self._lock:
            self._sessions[session_id] = session
        emit_governance_event(
            event_type="shell",
            action="start",
            outcome="ok",
            actor="operator",
            details={
                "session_id": session_id,
                "cwd": cwd,
                "timeout_s": float(timeout_s),
            },
        )
        return session

    async def cleanup_expired(self) -> int:
        now = time.time()
        removed = 0
        async with self._lock:
            stale_ids = []
            for sid, sess in self._sessions.items():
                if sess.status == "running":
                    continue
                ended_at = float(sess.ended_at or sess.created_at)
                if now - ended_at >= self.ttl_seconds:
                    stale_ids.append(sid)
            for sid in stale_ids:
                self._sessions.pop(sid, None)
                removed += 1
        return removed

    async def list_sessions(self) -> List[Dict[str, Any]]:
        await self.cleanup_expired()
        async with self._lock:
            rows = []
            for sid, sess in self._sessions.items():
                rows.append(
                    {
                        "session_id": sid,
                        "status": sess.status,
                        "return_code": sess.return_code,
                        "created_at": sess.created_at,
                        "ended_at": sess.ended_at,
                        "command": sess.command,
                        "cwd": sess.cwd,
                        "output_lines": len(sess.output_lines),
                    }
                )
            return sorted(rows, key=lambda r: float(r["created_at"]), reverse=True)

    async def poll(self, session_id: str, *, wait_ms: int = 0) -> Dict[str, Any]:
        await self.cleanup_expired()
        async with self._lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            return {"ok": False, "error": "session_not_found", "session_id": session_id}
        wait_s = max(0.0, float(wait_ms) / 1000.0)
        if wait_s > 0 and sess.waiter_task is not None and not sess.waiter_task.done():
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(asyncio.shield(sess.waiter_task), timeout=wait_s)
        async with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return {
                    "ok": False,
                    "error": "session_expired",
                    "session_id": session_id,
                }
            return {
                "ok": True,
                "session_id": session_id,
                "status": sess.status,
                "running": sess.status == "running",
                "return_code": sess.return_code,
                "timed_out": bool(sess.timed_out),
                "created_at": sess.created_at,
                "ended_at": sess.ended_at,
            }

    async def log(self, session_id: str, *, tail_lines: int = 200) -> Dict[str, Any]:
        await self.cleanup_expired()
        async with self._lock:
            sess = self._sessions.get(session_id)
            if sess is None:
                return {
                    "ok": False,
                    "error": "session_not_found",
                    "session_id": session_id,
                }
            tail = max(1, int(tail_lines))
            lines = list(sess.output_lines[-tail:])
            return {
                "ok": True,
                "session_id": session_id,
                "status": sess.status,
                "tail_lines": tail,
                "lines": lines,
                "text": "\n".join(lines),
            }

    async def write(
        self,
        session_id: str,
        *,
        text: str,
        submit: bool = False,
    ) -> Dict[str, Any]:
        async with self._lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            return {"ok": False, "error": "session_not_found", "session_id": session_id}
        if sess.status != "running":
            return {
                "ok": False,
                "error": "session_not_running",
                "session_id": session_id,
                "status": sess.status,
            }
        if sess.process.stdin is None:
            return {"ok": False, "error": "stdin_unavailable", "session_id": session_id}
        payload = str(text or "")
        if submit:
            payload += "\n"
        try:
            sess.process.stdin.write(payload.encode("utf-8"))
            await sess.process.stdin.drain()
            return {
                "ok": True,
                "session_id": session_id,
                "bytes": len(payload.encode("utf-8")),
            }
        except Exception as exc:
            return {"ok": False, "error": str(exc), "session_id": session_id}

    async def kill(self, session_id: str) -> Dict[str, Any]:
        async with self._lock:
            sess = self._sessions.get(session_id)
        if sess is None:
            return {"ok": False, "error": "session_not_found", "session_id": session_id}
        if sess.status != "running":
            return {
                "ok": True,
                "session_id": session_id,
                "status": sess.status,
                "already_stopped": True,
            }
        try:
            sess.process.terminate()
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(sess.process.wait(), timeout=1.5)
            if sess.process.returncode is None:
                sess.process.kill()
            async with self._lock:
                live = self._sessions.get(session_id)
                if live is not None:
                    live.status = "killed"
                    live.return_code = (
                        int(sess.process.returncode)
                        if sess.process.returncode is not None
                        else -9
                    )
                    live.ended_at = time.time()
            emit_governance_event(
                event_type="shell",
                action="kill",
                outcome="ok",
                actor="operator",
                details={"session_id": session_id},
            )
            return {"ok": True, "session_id": session_id, "status": "killed"}
        except Exception as exc:
            return {"ok": False, "error": str(exc), "session_id": session_id}


_SESSION_MANAGER: Optional[ShellSessionManager] = None
_SESSION_MANAGER_LOOP_ID: Optional[int] = None


def get_shell_session_manager() -> ShellSessionManager:
    global _SESSION_MANAGER, _SESSION_MANAGER_LOOP_ID
    loop_id: Optional[int]
    try:
        loop_id = id(asyncio.get_running_loop())
    except RuntimeError:
        loop_id = None
    if _SESSION_MANAGER is None or (
        _SESSION_MANAGER_LOOP_ID is not None
        and loop_id is not None
        and _SESSION_MANAGER_LOOP_ID != loop_id
    ):
        _SESSION_MANAGER = ShellSessionManager()
        _SESSION_MANAGER_LOOP_ID = loop_id
    return _SESSION_MANAGER


class ExecStartTool(FunctionTool):
    @property
    def name(self) -> str:
        return "exec_start"

    @property
    def description(self) -> str:
        return (
            "Start a shell command as a managed session. Supports background mode "
            "with session_id for later poll/log/write/kill operations."
        )

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string"},
                "working_dir": {"type": "string"},
                "background": {"type": "boolean"},
                "timeout_s": {"type": "number", "minimum": 0.0, "maximum": 86400.0},
                "pty": {"type": "boolean"},
            },
            "required": ["command"],
        }

    async def execute(
        self,
        command: str,
        working_dir: str = "",
        background: bool = True,
        timeout_s: float = 0.0,
        pty: bool = False,
    ) -> str:
        manager = get_shell_session_manager()
        cmd = str(command or "").strip()
        if not cmd:
            return json.dumps(
                {"ok": False, "error": "empty_command"},
                ensure_ascii=False,
            )
        cwd = str(working_dir or "").strip() or str(Path.cwd())
        cwd_path = Path(cwd).expanduser().resolve()
        if not cwd_path.exists() or not cwd_path.is_dir():
            return json.dumps(
                {
                    "ok": False,
                    "error": "invalid_working_dir",
                    "working_dir": str(cwd_path),
                },
                ensure_ascii=False,
            )
        try:
            session = await manager.start(
                command=cmd,
                cwd=str(cwd_path),
                timeout_s=max(0.0, float(timeout_s)),
            )
        except ValueError as exc:
            return json.dumps(
                {"ok": False, "error": str(exc)},
                ensure_ascii=False,
            )
        payload: Dict[str, Any] = {
            "ok": True,
            "session_id": session.session_id,
            "background": bool(background),
            "pty_requested": bool(pty),
            "pty_supported": False,
            "cwd": str(cwd_path),
        }
        if not background:
            # Wait for completion and return output in one response.
            while True:
                polled = await manager.poll(session.session_id, wait_ms=50)
                if not bool(polled.get("ok", False)):
                    break
                if not bool(polled.get("running", False)):
                    break
            log = await manager.log(session.session_id, tail_lines=400)
            polled = await manager.poll(session.session_id)
            payload.update(
                {
                    "status": polled.get("status"),
                    "return_code": polled.get("return_code"),
                    "output": log.get("text", ""),
                }
            )
        return json.dumps(payload, ensure_ascii=False)


class ExecProcessTool(FunctionTool):
    @property
    def name(self) -> str:
        return "exec_process"

    @property
    def description(self) -> str:
        return "Manage shell execution sessions: list, poll, log, write, and kill."

    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "poll", "log", "write", "submit", "kill"],
                },
                "session_id": {"type": "string"},
                "wait_ms": {"type": "integer", "minimum": 0, "maximum": 60000},
                "tail_lines": {"type": "integer", "minimum": 1, "maximum": 2000},
                "text": {"type": "string"},
                "submit": {"type": "boolean"},
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        session_id: str = "",
        wait_ms: int = 0,
        tail_lines: int = 200,
        text: str = "",
        submit: bool = False,
    ) -> str:
        manager = get_shell_session_manager()
        act = str(action or "").strip().lower()
        if act == "list":
            rows = await manager.list_sessions()
            return json.dumps({"ok": True, "sessions": rows}, ensure_ascii=False)
        if act == "poll":
            payload = await manager.poll(
                str(session_id or "").strip(),
                wait_ms=max(0, int(wait_ms)),
            )
            return json.dumps(payload, ensure_ascii=False)
        if act == "log":
            payload = await manager.log(
                str(session_id or "").strip(),
                tail_lines=max(1, int(tail_lines)),
            )
            return json.dumps(payload, ensure_ascii=False)
        if act == "write":
            payload = await manager.write(
                str(session_id or "").strip(),
                text=str(text or ""),
                submit=bool(submit),
            )
            return json.dumps(payload, ensure_ascii=False)
        if act == "submit":
            payload = await manager.write(
                str(session_id or "").strip(),
                text=str(text or ""),
                submit=True,
            )
            return json.dumps(payload, ensure_ascii=False)
        if act == "kill":
            payload = await manager.kill(str(session_id or "").strip())
            return json.dumps(payload, ensure_ascii=False)
        return json.dumps({"ok": False, "error": "unsupported_action"})


__all__ = [
    "ShellSessionManager",
    "get_shell_session_manager",
    "ExecStartTool",
    "ExecProcessTool",
]
