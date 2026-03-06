from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Event as ThreadEvent, Lock
from typing import Any, Callable, Dict, List, Optional, Sequence

from .base import LLMProvider, LLMResponse

DEFAULT_CODEX_CLI = "codex"
DEFAULT_CODEX_CLI_MODEL = "codex-cli/gpt-5.3-codex"
_ANNOLID_DIR = Path.home() / ".annolid"
_CODEX_CLI_SESSION_FILE = _ANNOLID_DIR / "codex_cli_sessions.json"
_CODEX_CLI_SESSION_LOCK = Lock()


@dataclass(frozen=True)
class CodexCLIResolved:
    model: str
    cli_path: str
    workdir: str
    session_id: str
    runtime: str


def resolve_codex_cli(config: Any) -> CodexCLIResolved:
    params = dict(getattr(config, "params", {}) or {})
    model = str(getattr(config, "model", "") or "").strip() or DEFAULT_CODEX_CLI_MODEL
    cli_path = str(params.get("cli_path") or "").strip() or DEFAULT_CODEX_CLI
    workdir = str(params.get("workdir") or "").strip() or os.getcwd()
    session_id = str(params.get("session_id") or "").strip()
    runtime = str(params.get("runtime") or "").strip().lower()
    return CodexCLIResolved(
        model=model,
        cli_path=cli_path,
        workdir=workdir,
        session_id=session_id,
        runtime=runtime,
    )


class CodexCLIProvider(LLMProvider):
    """Conservative text-only Codex CLI runtime for fallback/local execution."""

    def __init__(
        self,
        *,
        resolved: CodexCLIResolved,
        runner: Optional[Callable[..., str]] = None,
    ) -> None:
        self._resolved = resolved
        self._runner = runner or _run_codex_cli

    def get_default_model(self) -> str:
        return self._resolved.model

    async def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = 0.7,
        timeout_seconds: Optional[float] = None,
        on_token: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[ThreadEvent] = None,
    ) -> LLMResponse:
        del max_tokens, temperature
        cli_path = str(self._resolved.cli_path or DEFAULT_CODEX_CLI).strip()
        if not shutil.which(cli_path):
            return LLMResponse(
                content=(
                    "Codex CLI provider requires the `codex` executable. "
                    "Install Codex and ensure `codex` is available on PATH."
                ),
                finish_reason="error",
            )

        prompt = _render_prompt(messages, tools=tools)
        timeout_s = float(timeout_seconds) if timeout_seconds else 180.0
        try:
            runner_kwargs = {
                "cli_path": cli_path,
                "prompt": prompt,
                "model": _strip_model_prefix(model or self._resolved.model),
                "workdir": self._resolved.workdir,
                "timeout_seconds": timeout_s,
                "images": _extract_latest_user_images(messages),
                "session_id": self._resolved.session_id,
                "runtime": self._resolved.runtime,
            }
            if cancel_event is not None:
                runner_kwargs["cancel_event"] = cancel_event
            content = await asyncio.to_thread(self._runner, **runner_kwargs)
        except subprocess.TimeoutExpired:
            return LLMResponse(
                content=f"Codex CLI request timed out after {timeout_s:.0f}s.",
                finish_reason="error",
            )
        except Exception as exc:
            return LLMResponse(
                content=f"Error calling Codex CLI: {exc}",
                finish_reason="error",
            )

        if on_token is not None and content:
            on_token(content)
        return LLMResponse(content=content, finish_reason="stop")


def _run_codex_cli(
    *,
    cli_path: str,
    prompt: str,
    model: str,
    workdir: str,
    timeout_seconds: float,
    images: Sequence[str],
    session_id: str,
    runtime: str,
    cancel_event: Optional[ThreadEvent] = None,
) -> str:
    with tempfile.TemporaryDirectory(prefix="annolid-codex-cli-") as tmpdir:
        output_path = Path(tmpdir) / "last_message.txt"
        mapped_thread_id = _load_codex_thread_id(
            session_id=session_id,
            model=model,
            workdir=workdir,
        )
        completed = _invoke_codex_cli(
            cli_path=cli_path,
            prompt=prompt,
            model=model,
            workdir=workdir,
            timeout_seconds=timeout_seconds,
            images=images,
            output_path=output_path,
            thread_id=mapped_thread_id,
            runtime=runtime,
            session_id=session_id,
            cancel_event=cancel_event,
        )
        if completed.returncode != 0 and mapped_thread_id:
            _delete_codex_thread_id(
                session_id=session_id,
                model=model,
                workdir=workdir,
            )
            completed = _invoke_codex_cli(
                cli_path=cli_path,
                prompt=prompt,
                model=model,
                workdir=workdir,
                timeout_seconds=timeout_seconds,
                images=images,
                output_path=output_path,
                thread_id="",
                runtime=runtime,
                session_id=session_id,
                cancel_event=cancel_event,
            )
        output_text = ""
        if output_path.exists():
            output_text = output_path.read_text(encoding="utf-8").strip()
        thread_id = _parse_thread_id(completed.stdout)
        if session_id and thread_id:
            _save_codex_thread_id(
                session_id=session_id,
                model=model,
                workdir=workdir,
                thread_id=thread_id,
            )
        if completed.returncode != 0:
            detail = output_text or completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                detail or f"Codex CLI exited with code {completed.returncode}."
            )
        if output_text:
            return output_text
        fallback = completed.stdout.strip()
        if fallback:
            return fallback
        raise RuntimeError("Codex CLI returned an empty response.")


def _invoke_codex_cli(
    *,
    cli_path: str,
    prompt: str,
    model: str,
    workdir: str,
    timeout_seconds: float,
    images: Sequence[str],
    output_path: Path,
    thread_id: str,
    runtime: str,
    session_id: str,
    cancel_event: Optional[ThreadEvent],
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    if runtime == "acp":
        env["ANNOLID_AGENT_RUNTIME"] = "acp"
        env["ANNOLID_SHELL"] = "acp"
        env["OPENCLAW_SHELL"] = "acp"
        if session_id:
            env["ANNOLID_ACP_SESSION_ID"] = session_id
    if thread_id:
        cmd = [
            cli_path,
            "exec",
            "resume",
            thread_id,
            "-",
            "--model",
            model,
            "--skip-git-repo-check",
            "--json",
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
        ]
    else:
        cmd = [
            cli_path,
            "exec",
            "-",
            "--model",
            model,
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--json",
            "--color",
            "never",
            "--output-last-message",
            str(output_path),
        ]
    for image_path in images:
        cmd.extend(["--image", image_path])
    if cancel_event is None:
        return subprocess.run(
            cmd,
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
            cwd=workdir,
            env=env,
        )
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=workdir,
        env=env,
    )
    try:
        stdout, stderr = _communicate_with_cancel(
            proc,
            prompt=prompt,
            timeout_seconds=timeout_seconds,
            cancel_event=cancel_event,
        )
    except Exception:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=1.0)
        raise
    return subprocess.CompletedProcess(
        cmd, proc.returncode, stdout=stdout, stderr=stderr
    )


def _communicate_with_cancel(
    proc: subprocess.Popen[str],
    *,
    prompt: str,
    timeout_seconds: float,
    cancel_event: Optional[ThreadEvent],
) -> tuple[str, str]:
    deadline = timeout_seconds if timeout_seconds > 0 else None
    while True:
        if cancel_event is not None and cancel_event.is_set():
            proc.terminate()
            try:
                stdout, stderr = proc.communicate(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
            raise RuntimeError("Codex CLI request cancelled.")
        try:
            stdout, stderr = proc.communicate(input=prompt, timeout=0.2)
            return stdout, stderr
        except subprocess.TimeoutExpired:
            prompt = ""
            if deadline is not None:
                deadline -= 0.2
                if deadline <= 0:
                    proc.kill()
                    stdout, stderr = proc.communicate()
                    raise subprocess.TimeoutExpired(
                        proc.args, timeout_seconds, stdout, stderr
                    )


def _strip_model_prefix(model: str) -> str:
    raw = str(model or "").strip()
    lowered = raw.lower()
    for prefix in ("codex-cli/", "codex_cli/"):
        if lowered.startswith(prefix):
            return raw[len(prefix) :]
    return raw


def _render_prompt(
    messages: Sequence[Dict[str, Any]],
    *,
    tools: Optional[Sequence[Dict[str, Any]]] = None,
) -> str:
    sections: List[str] = [
        "You are running inside Annolid's Codex CLI runtime.",
        "Respond with the final answer only.",
        "Do not run shell commands, edit files, or depend on external tools in this mode.",
    ]
    if tools:
        sections.append(
            "Annolid tools are unavailable in this runtime; answer from the provided context only."
        )

    for message in messages:
        role = str(message.get("role") or "user").strip().upper()
        text = _message_text(message.get("content"))
        if not text:
            continue
        sections.append(f"{role}:\n{text}")
    return "\n\n".join(sections).strip()


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type") or "").strip().lower()
            if item_type in {"text", "input_text", "output_text"}:
                text = str(item.get("text") or "").strip()
                if text:
                    parts.append(text)
        return "\n".join(parts).strip()
    return ""


def _extract_latest_user_images(messages: Sequence[Dict[str, Any]]) -> List[str]:
    images: List[str] = []
    for message in messages:
        if str(message.get("role") or "").strip().lower() != "user":
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict):
                continue
            image_path = str(item.get("image_path") or "").strip()
            if image_path and os.path.exists(image_path):
                images.append(image_path)
    return images


def _session_map_key(*, session_id: str, model: str, workdir: str) -> str:
    return f"{session_id}::{Path(workdir).resolve()}::{model}"


def _load_codex_thread_id(*, session_id: str, model: str, workdir: str) -> str:
    if not session_id:
        return ""
    data = _read_session_map()
    value = data.get(
        _session_map_key(session_id=session_id, model=model, workdir=workdir)
    )
    return str(value or "").strip()


def _save_codex_thread_id(
    *, session_id: str, model: str, workdir: str, thread_id: str
) -> None:
    if not session_id or not thread_id:
        return
    with _CODEX_CLI_SESSION_LOCK:
        data = _read_session_map_unlocked()
        data[_session_map_key(session_id=session_id, model=model, workdir=workdir)] = (
            thread_id
        )
        _write_session_map_unlocked(data)


def _delete_codex_thread_id(*, session_id: str, model: str, workdir: str) -> None:
    if not session_id:
        return
    with _CODEX_CLI_SESSION_LOCK:
        data = _read_session_map_unlocked()
        data.pop(
            _session_map_key(session_id=session_id, model=model, workdir=workdir), None
        )
        _write_session_map_unlocked(data)


def _read_session_map() -> Dict[str, str]:
    with _CODEX_CLI_SESSION_LOCK:
        return _read_session_map_unlocked()


def _read_session_map_unlocked() -> Dict[str, str]:
    try:
        raw = _CODEX_CLI_SESSION_FILE.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, str] = {}
    for key, value in data.items():
        text_key = str(key or "").strip()
        text_value = str(value or "").strip()
        if text_key and text_value:
            out[text_key] = text_value
    return out


def _write_session_map_unlocked(data: Dict[str, str]) -> None:
    _ANNOLID_DIR.mkdir(parents=True, exist_ok=True)
    _CODEX_CLI_SESSION_FILE.write_text(
        json.dumps(data, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _parse_thread_id(stdout: str) -> str:
    for raw_line in str(stdout or "").splitlines():
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except Exception:
            continue
        if str(payload.get("type") or "").strip() != "thread.started":
            continue
        thread_id = str(payload.get("thread_id") or "").strip()
        if thread_id:
            return thread_id
    return ""
