from __future__ import annotations

import asyncio
import base64
import inspect
import json
import re
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from .auth import ColabAuthManager, ColabAuthRequiredError
from .models import (
    ColabRuntimeInfo,
    ColabRuntimeSpec,
    accelerator_to_api_spec,
    normalize_accelerator,
)
from .notebooks import build_colab_drive_url
from annolid.utils.safe_http import build_http_request, read_bounded, safe_urlopen

LogCallback = Optional[Callable[[str], None]]

COLAB_API_BASE = "https://colaboratory.googleapis.com/v1beta"


def _emit_log(callback: LogCallback, message: str) -> None:
    if callback is not None:
        try:
            callback(message)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Backend 1: Official Google Colab CLI (`google-colab-cli`)
# ---------------------------------------------------------------------------


class ColabCliBackend:
    """Wrapper around the official `google-colab-cli` (`colab`) binary."""

    def __init__(self, auth_manager: ColabAuthManager) -> None:
        self._auth = auth_manager

    @property
    def name(self) -> str:
        return "cli"

    def is_available(self) -> bool:
        return bool(self._auth.find_colab_cli())

    def _binary(self) -> str:
        cli = self._auth.find_colab_cli()
        if not cli:
            raise FileNotFoundError(
                "Official `google-colab-cli` (`colab`) not found on PATH. "
                "Install it with `uv tool install google-colab-cli` "
                "(or `pip install google-colab-cli`)."
            )
        return cli

    def _base_cmd(self) -> list[str]:
        return [
            self._binary(),
            f"--auth={self._auth.resolve_colab_cli_auth_mode()}",
        ]

    def _run_capture(
        self,
        args: list[str],
        *,
        timeout: int = 300,
    ) -> subprocess.CompletedProcess[str]:
        cmd = [*self._base_cmd(), *args]
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            low = detail.lower()
            if any(
                marker in low
                for marker in (
                    "unauthenticated",
                    "unauthorized",
                    "invalid_grant",
                    "not authenticated",
                    "re-authenticate",
                    "token expired",
                    "login required",
                    "http 401",
                    "http 403",
                    "401 ",
                    "403 ",
                )
            ):
                raise ColabAuthRequiredError(
                    f"Colab CLI authentication required (`{' '.join(args)}`, exit {proc.returncode}): {detail}",
                    backend="cli",
                    auth_status=self._auth.get_auth_status(),
                )
            if any(
                marker in low
                for marker in (
                    "allocation refused",
                    "too many active sessions",
                    "precondition failed",
                )
            ):
                raise RuntimeError(
                    f"Colab allocation refused (`{' '.join(args)}`): {detail}\n"
                    "Tip: check running VMs with `annolid-run colab sessions`, stop idle ones "
                    "with `annolid-run colab stop <session>`, or reuse an active one with `--session <name>`."
                )
            raise RuntimeError(
                f"Colab CLI command failed (`{' '.join(args)}`, exit {proc.returncode}): {detail}"
            )
        return proc

    def list_sessions(self) -> list[ColabRuntimeInfo]:
        proc = self._run_capture(["sessions"], timeout=60)
        text = (proc.stdout or "").strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return [
                    ColabRuntimeInfo(
                        name=str(item.get("name") or item.get("session") or ""),
                        session_name=str(item.get("session") or item.get("name") or ""),
                        backend="cli",
                        accelerator=str(
                            item.get("accelerator") or item.get("gpu") or ""
                        ),
                        variant=str(item.get("variant") or ""),
                        shape=str(item.get("shape") or ""),
                        state=str(item.get("state") or "ACTIVE"),
                        browser_url=str(item.get("url") or ""),
                    )
                    for item in parsed
                    if isinstance(item, dict)
                ]
        except Exception:
            pass

        runtimes: list[ColabRuntimeInfo] = []
        session_line_re = re.compile(
            r"^\[(?P<name>[^\]]+)\]\s+(?P<endpoint>\S+)(?P<rest>.*)$"
        )
        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line or line.lower().startswith(
                ("session", "name", "---", "no ", "[colab]")
            ):
                continue
            m = session_line_re.match(line)
            if m:
                sess_name = m.group("name").strip()
                endpoint = m.group("endpoint").strip()
                rest = m.group("rest") or ""
                kv: dict[str, str] = {}
                for seg in rest.split("|"):
                    if ":" in seg:
                        k, v = seg.split(":", 1)
                        kv[k.strip().lower()] = v.strip()
                runtimes.append(
                    ColabRuntimeInfo(
                        name=endpoint if sess_name == "?" else sess_name,
                        session_name=sess_name,
                        backend="cli",
                        accelerator=kv.get("hardware", ""),
                        shape=kv.get("shape", ""),
                        variant=kv.get("variant", ""),
                        state=kv.get("status", "ACTIVE"),
                    )
                )
                continue
            parts = line.split()
            if not parts:
                continue
            session_id = parts[0]
            state = parts[-1] if len(parts) > 1 else "ACTIVE"
            accel = parts[1] if len(parts) > 2 else ""
            runtimes.append(
                ColabRuntimeInfo(
                    name=session_id,
                    session_name=session_id,
                    backend="cli",
                    accelerator=accel,
                    state=state,
                )
            )
        return runtimes

    def _colab_cli_python(self) -> Optional[str]:
        cli = self._auth.find_colab_cli()
        if not cli:
            return None
        resolved = Path(cli).resolve()
        for candidate in (
            resolved.parent / "python",
            resolved.parent / "python3",
        ):
            if candidate.is_file():
                return str(candidate)
        return None

    def _unassign_orphan_endpoint(self, endpoint: str) -> bool:
        """Unassign a server-side Colab endpoint that has no local `colab-cli` session entry."""
        py_bin = self._colab_cli_python()
        if not py_bin:
            return False
        auth_mode = self._auth.resolve_colab_cli_auth_mode()
        script = (
            "import sys\n"
            "from colab_cli.auth import AuthProvider, get_credentials\n"
            "from colab_cli.client import Client, Prod\n"
            "provider = AuthProvider.ADC if sys.argv[2] == 'adc' else AuthProvider.OAUTH2\n"
            "Client(Prod(), get_credentials(provider=provider)).unassign(sys.argv[1])\n"
        )
        proc = subprocess.run(
            [py_bin, "-c", script, str(endpoint), auth_mode],
            capture_output=True,
            text=True,
            timeout=60,
        )
        return proc.returncode == 0

    def create_runtime(
        self,
        *,
        session_name: str,
        accelerator: str = "T4",
        high_mem: bool = False,
        log_callback: LogCallback = None,
    ) -> ColabRuntimeInfo:
        acc = normalize_accelerator(accelerator)
        args = ["new", "-s", session_name]
        if acc.startswith("TPU_"):
            args.extend(["--tpu", acc.removeprefix("TPU_").lower()])
        elif acc != "CPU":
            args.extend(["--gpu", acc])
        if high_mem:
            args.append("--high-mem")
        before_endpoints: set[str] = set()
        try:
            before_endpoints = {
                s.name for s in self.list_sessions() if s.session_name == "?"
            }
        except Exception:
            before_endpoints = set()
        _emit_log(
            log_callback,
            f"Provisioning Colab CLI session '{session_name}' ({acc})...",
        )
        try:
            proc = self._run_capture(args, timeout=300)
        except subprocess.TimeoutExpired as exc:
            # Best-effort cleanup if Google assigned a VM right before timeout
            # so we do not leak an orphaned billable GPU/TPU runtime.
            try:
                after_orphans = [
                    s.name
                    for s in self.list_sessions()
                    if s.session_name == "?" and s.name not in before_endpoints
                ]
                for ep in after_orphans:
                    if self._unassign_orphan_endpoint(ep):
                        _emit_log(
                            log_callback,
                            f"Cleaned up orphaned Colab runtime '{ep}' after provisioning timeout.",
                        )
            except Exception:
                pass
            raise RuntimeError(
                f"Provisioning Colab session '{session_name}' ({acc}) timed out after 300s. "
                "Retry or check `annolid-run colab sessions`."
            ) from exc
        if proc.stdout.strip():
            _emit_log(log_callback, proc.stdout.strip())
        browser_url = ""
        try:
            url_proc = self._run_capture(["url", "-s", session_name], timeout=30)
            for token in (url_proc.stdout or "").split():
                if token.startswith("https://"):
                    browser_url = token
                    break
        except Exception:
            browser_url = ""
        return ColabRuntimeInfo(
            name=session_name,
            session_name=session_name,
            backend="cli",
            accelerator=acc,
            state="ACTIVE",
            browser_url=browser_url,
        )

    def upload_file(
        self,
        *,
        session_name: str,
        local_path: Path,
        remote_path: str,
        log_callback: LogCallback = None,
    ) -> None:
        _emit_log(
            log_callback,
            f"Uploading {local_path.name} -> {remote_path} via Colab CLI...",
        )
        self._run_capture(
            ["upload", "-s", session_name, str(local_path), remote_path],
            timeout=1800,
        )

    def download_file(
        self,
        *,
        session_name: str,
        remote_path: str,
        local_path: Path,
        log_callback: LogCallback = None,
    ) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        _emit_log(
            log_callback,
            f"Downloading {remote_path} -> {local_path} via Colab CLI...",
        )
        self._run_capture(
            ["download", "-s", session_name, remote_path, str(local_path)],
            timeout=1800,
        )

    def exec_script(
        self,
        *,
        session_name: str,
        script_path: Path,
        log_callback: LogCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        cmd = [
            *self._base_cmd(),
            "exec",
            "-s",
            session_name,
            "--timeout",
            "0",
            "-f",
            str(script_path),
        ]
        _emit_log(
            log_callback,
            f"Executing remote Annolid job on Colab session '{session_name}'...",
        )
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        lines: list[str] = []
        try:
            for line in proc.stdout:
                lines.append(line)
                _emit_log(log_callback, line.rstrip("\n"))
                if cancel_event is not None and cancel_event.is_set():
                    proc.terminate()
                    raise InterruptedError("Colab CLI execution cancelled by user.")
            rc = int(proc.wait())
        finally:
            if proc.poll() is None:
                proc.terminate()
        output = "".join(lines)
        if rc != 0:
            raise RuntimeError(f"Colab remote script failed with exit code {rc}.")
        # `colab exec` executes inside a Jupyter kernel and may exit 0 even when
        # the remote Python script raised an exception; detect explicit error marker.
        err_marker = "ANNOLID_COLAB_ERROR_JSON="
        for raw_line in reversed(output.splitlines()):
            idx = raw_line.find(err_marker)
            if idx != -1:
                raw_json = raw_line[idx + len(err_marker) :].strip()
                err_msg = raw_json
                try:
                    parsed_err = json.loads(raw_json)
                    if isinstance(parsed_err, dict) and parsed_err.get("error"):
                        err_msg = str(parsed_err["error"])
                except Exception:
                    pass
                raise RuntimeError(f"Colab remote script failed: {err_msg}")
        return output

    def stop_runtime(self, session_name: str) -> None:
        proc = self._run_capture(["stop", "-s", session_name], timeout=60)
        combined = f"{proc.stdout or ''}\n{proc.stderr or ''}".lower()
        if "not found" in combined and session_name.startswith(
            ("gpu-", "tpu-", "cpu-", "m-")
        ):
            if not self._unassign_orphan_endpoint(session_name):
                raise RuntimeError(
                    f"Colab session '{session_name}' was not found locally and could not "
                    "be unassigned on the Colab server."
                )


# ---------------------------------------------------------------------------
# Backend 2: Native Google Colab REST API (v1beta) + Jupyter Proxy
# ---------------------------------------------------------------------------


class ColabApiBackend:
    """Cross-platform native client for `colaboratory.googleapis.com/v1beta` + Jupyter API."""

    _CHUNK_BYTES = 8 * 1024 * 1024  # 8 MiB chunks for Contents API uploads
    _STREAM_DOWNLOAD_CHUNK_BYTES = 1024 * 1024  # 1 MiB streaming binary reads
    _LARGE_UPLOAD_WARN_BYTES = 100 * 1024 * 1024  # 100 MiB advisory threshold
    _TRANSIENT_HTTP_CODES = frozenset({429, 500, 502, 503, 504})
    _MAX_RETRIES = 4
    _RETRY_BASE_SECONDS = 1.0

    def __init__(
        self,
        auth_manager: ColabAuthManager,
        *,
        api_base: str = COLAB_API_BASE,
        auth_mode: str = "auto",
        timeout_s: float = 60.0,
    ) -> None:
        self._auth = auth_manager
        self._api_base = str(api_base or COLAB_API_BASE).rstrip("/")
        self._auth_mode = str(auth_mode or "auto").strip().lower()
        self._timeout_s = float(timeout_s)

    @property
    def name(self) -> str:
        return "api"

    def _bearer_headers(self) -> dict[str, str]:
        token = self._auth.get_colab_api_access_token(auth_mode=self._auth_mode)
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _sleep_backoff(self, attempt: int) -> None:
        delay = min(30.0, float(self._RETRY_BASE_SECONDS) * (2**attempt))
        if delay > 0:
            time.sleep(delay)

    @staticmethod
    def _summarize_error_body(err_body: str) -> str:
        raw = str(err_body or "").strip()
        if not raw:
            return ""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                err = parsed.get("error")
                if isinstance(err, dict):
                    status = str(err.get("status") or "").strip()
                    msg = str(err.get("message") or "").strip()
                    if status and msg:
                        return f"{status}: {msg}"
                    if msg:
                        return msg
                elif isinstance(err, str) and err.strip():
                    return err.strip()
        except Exception:
            pass
        first_line = raw.splitlines()[0].strip()
        return first_line[:300]

    def _request_json(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        body: Any = None,
        timeout: Optional[float] = None,
    ) -> dict[str, Any]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        last_exc: Optional[Exception] = None
        for attempt in range(self._MAX_RETRIES + 1):
            req = build_http_request(
                url,
                data=data,
                headers=headers,
                method=method.upper(),
                allow_private=False,
                allow_loopback=True,
                allow_http_only_for_loopback=True,
            )
            try:
                with safe_urlopen(
                    req,
                    timeout=float(timeout or self._timeout_s),
                    allow_private=False,
                    allow_loopback=True,
                    allow_http_only_for_loopback=True,
                    urlopen_func=urllib.request.urlopen,
                ) as resp:
                    raw = (
                        read_bounded(resp, 20 * 1024 * 1024, label="Colab API response")
                        .decode("utf-8", errors="replace")
                        .strip()
                    )
                    if not raw:
                        return {}
                    parsed = json.loads(raw)
                    return parsed if isinstance(parsed, dict) else {"data": parsed}
            except urllib.error.HTTPError as exc:
                err_body = ""
                try:
                    err_body = read_bounded(
                        exc, 256 * 1024, label="Colab API error body"
                    ).decode("utf-8", errors="replace")
                except Exception:
                    pass
                if (
                    exc.code in self._TRANSIENT_HTTP_CODES
                    and attempt < self._MAX_RETRIES
                ):
                    self._sleep_backoff(attempt)
                    last_exc = exc
                    continue
                summary = self._summarize_error_body(err_body)
                if exc.code in {401, 403}:
                    raise ColabAuthRequiredError(
                        f"Google Colab API authorization failed (HTTP {exc.code}): {summary}. "
                        "Ensure your token includes the `https://www.googleapis.com/auth/colaboratory` scope "
                        "and run `annolid-run colab auth login`.",
                        backend="api",
                        auth_status=self._auth.get_auth_status(),
                    ) from exc
                raise RuntimeError(
                    f"Google Colab API request failed ({method} {url}, HTTP {exc.code}): {summary}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < self._MAX_RETRIES:
                    self._sleep_backoff(attempt)
                    last_exc = exc
                    continue
                raise RuntimeError(
                    f"Network error contacting Google Colab API ({url}): {exc.reason}"
                ) from exc
        raise RuntimeError(
            f"Google Colab API request failed after retries ({method} {url}): {last_exc}"
        )

    @staticmethod
    def is_proxy_token_expiring(
        runtime: ColabRuntimeInfo, *, min_ttl_seconds: int = 300
    ) -> bool:
        raw = str(runtime.expire_time or "").strip()
        if not raw:
            return False
        try:
            iso = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
            expiry = datetime.fromisoformat(iso)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=timezone.utc)
            remaining = (expiry - datetime.now(timezone.utc)).total_seconds()
            return remaining <= float(min_ttl_seconds)
        except Exception:
            return False

    def refresh_runtime_connection(
        self,
        runtime: ColabRuntimeInfo,
        *,
        force: bool = False,
        min_ttl_seconds: int = 300,
        log_callback: LogCallback = None,
    ) -> ColabRuntimeInfo:
        """Refresh a short-lived Colab Jupyter proxy token before or on expiry."""
        if not runtime.name:
            return runtime
        if not force and not self.is_proxy_token_expiring(
            runtime, min_ttl_seconds=min_ttl_seconds
        ):
            return runtime
        _emit_log(
            log_callback,
            f"Refreshing Colab runtime proxy token for '{runtime.name}'...",
        )
        refreshed = self.get_runtime(runtime.name)
        if refreshed.proxy_url and refreshed.proxy_token:
            return refreshed
        return runtime

    def list_runtime_specs(self) -> list[ColabRuntimeSpec]:
        payload = self._request_json(
            "GET",
            f"{self._api_base}/runtimespecs",
            headers=self._bearer_headers(),
        )
        specs: list[ColabRuntimeSpec] = []
        for item in list(payload.get("runtimeSpecs") or []):
            if not isinstance(item, dict):
                continue
            key = item.get("key") or {}
            if not isinstance(key, dict):
                continue
            specs.append(
                ColabRuntimeSpec(
                    variant=str(key.get("variant") or ""),
                    accelerator=str(key.get("accelerator") or "NONE"),
                    shape=str(key.get("shape") or "SHAPE_STANDARD"),
                    eligible=bool(item.get("eligible", False)),
                )
            )
        return specs

    def list_runtimes(self) -> list[ColabRuntimeInfo]:
        payload = self._request_json(
            "GET",
            f"{self._api_base}/runtimes",
            headers=self._bearer_headers(),
        )
        return [
            self._parse_runtime(item)
            for item in list(payload.get("runtimes") or [])
            if isinstance(item, dict)
        ]

    def _parse_runtime(self, payload: dict[str, Any]) -> ColabRuntimeInfo:
        spec = payload.get("runtimeSpec") or {}
        conn = payload.get("connectionInfo") or {}
        return ColabRuntimeInfo(
            name=str(payload.get("name") or ""),
            backend="api",
            session_name=str(payload.get("name") or "").split("/")[-1],
            accelerator=str(spec.get("accelerator") or ""),
            variant=str(spec.get("variant") or ""),
            shape=str(spec.get("shape") or ""),
            state=str(payload.get("state") or "ACTIVE"),
            proxy_url=str(conn.get("url") or "").rstrip("/"),
            proxy_token=str(conn.get("token") or ""),
            expire_time=str(conn.get("expireTime") or ""),
        )

    def _select_runtime_spec(
        self, accelerator: str, *, high_mem: bool = False
    ) -> dict[str, str]:
        desired = accelerator_to_api_spec(accelerator, high_mem=high_mem)
        try:
            specs = self.list_runtime_specs()
        except Exception:
            return desired

        eligible = [s for s in specs if s.eligible]
        for spec in eligible:
            if (
                spec.variant == desired["variant"]
                and spec.accelerator.upper() == desired["accelerator"].upper()
                and spec.shape == desired["shape"]
            ):
                return {
                    "variant": spec.variant,
                    "accelerator": spec.accelerator,
                    "shape": spec.shape,
                }
        for spec in eligible:
            if (
                spec.variant == desired["variant"]
                and spec.accelerator.upper() == desired["accelerator"].upper()
            ):
                return {
                    "variant": spec.variant,
                    "accelerator": spec.accelerator,
                    "shape": spec.shape,
                }
        return desired

    def create_runtime(
        self,
        *,
        accelerator: str = "T4",
        high_mem: bool = False,
        log_callback: LogCallback = None,
    ) -> ColabRuntimeInfo:
        spec = self._select_runtime_spec(accelerator, high_mem=high_mem)
        request_id = str(uuid.uuid4())
        _emit_log(
            log_callback,
            f"Provisioning Colab API runtime ({spec['variant']} / {spec['accelerator']} / {spec['shape']})...",
        )
        created = self._request_json(
            "POST",
            f"{self._api_base}/runtimes?requestId={urllib.parse.quote(request_id)}",
            headers=self._bearer_headers(),
            body={"runtimeSpec": spec},
            timeout=180.0,
        )
        info = self._parse_runtime(created)
        if info.name and (not info.proxy_url or not info.proxy_token):
            info = self.get_runtime(info.name)
        if not info.proxy_url or not info.proxy_token:
            raise RuntimeError(
                f"Provisioned Colab runtime {info.name!r} did not return Jupyter connectionInfo."
            )
        return info

    def get_runtime(self, runtime_name: str) -> ColabRuntimeInfo:
        clean = str(runtime_name or "").strip().lstrip("/")
        if not clean.startswith("runtimes/"):
            clean = f"runtimes/{clean}"
        payload = self._request_json(
            "GET",
            f"{self._api_base}/{clean}",
            headers=self._bearer_headers(),
        )
        return self._parse_runtime(payload)

    def delete_runtime(self, runtime_name: str) -> None:
        clean = str(runtime_name or "").strip().lstrip("/")
        if not clean.startswith("runtimes/"):
            clean = f"runtimes/{clean}"
        self._request_json(
            "DELETE",
            f"{self._api_base}/{clean}",
            headers=self._bearer_headers(),
        )

    @staticmethod
    def _proxy_headers(runtime: ColabRuntimeInfo) -> dict[str, str]:
        return {
            "X-Colab-Runtime-Proxy-Token": runtime.proxy_token,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

    def _ensure_remote_directory(
        self, runtime: ColabRuntimeInfo, remote_dir: str
    ) -> None:
        parts = [p for p in remote_dir.strip("/").split("/") if p]
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else part
            encoded = urllib.parse.quote(current, safe="/")
            url = f"{runtime.proxy_url}/api/contents/{encoded}"
            try:
                self._request_json(
                    "PUT",
                    url,
                    headers=self._proxy_headers(runtime),
                    body={"type": "directory"},
                    timeout=30.0,
                )
            except RuntimeError:
                # Directory may already exist on the remote runtime.
                pass

    def _proxy_request_json(
        self,
        runtime: ColabRuntimeInfo,
        method: str,
        path_and_query: str,
        *,
        body: Any = None,
        timeout: Optional[float] = None,
        log_callback: LogCallback = None,
    ) -> tuple[dict[str, Any], ColabRuntimeInfo]:
        active = self.refresh_runtime_connection(runtime, log_callback=log_callback)
        url = f"{active.proxy_url}{path_and_query}"
        try:
            res = self._request_json(
                method,
                url,
                headers=self._proxy_headers(active),
                body=body,
                timeout=timeout,
            )
            return res, active
        except RuntimeError as exc:
            text = str(exc)
            if active.name and ("HTTP 401" in text or "HTTP 403" in text):
                refreshed = self.refresh_runtime_connection(
                    active, force=True, log_callback=log_callback
                )
                if refreshed.proxy_token != active.proxy_token:
                    res = self._request_json(
                        method,
                        f"{refreshed.proxy_url}{path_and_query}",
                        headers=self._proxy_headers(refreshed),
                        body=body,
                        timeout=timeout,
                    )
                    return res, refreshed
            raise

    def upload_file(
        self,
        *,
        runtime: ColabRuntimeInfo,
        local_path: Path,
        remote_path: str,
        log_callback: LogCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> ColabRuntimeInfo:
        active = self.refresh_runtime_connection(runtime, log_callback=log_callback)
        rel = remote_path.lstrip("/")
        if rel.startswith("content/"):
            rel = rel[len("content/") :]
        parent_rel = str(Path(rel).parent.as_posix())
        if parent_rel and parent_rel != ".":
            self._ensure_remote_directory(active, parent_rel)

        encoded = urllib.parse.quote(rel, safe="/")
        path_q = f"/api/contents/{encoded}"
        total_bytes = int(local_path.stat().st_size)
        total_mb = total_bytes / (1024 * 1024)
        _emit_log(
            log_callback,
            f"Uploading {local_path.name} ({total_mb:.1f} MB) to Colab runtime...",
        )
        if total_bytes >= self._LARGE_UPLOAD_WARN_BYTES:
            _emit_log(
                log_callback,
                f"[INFO] Large bundle ({total_mb:.1f} MB) via Jupyter Contents API: "
                "using 8 MiB retryable chunks. For multi-GB video sets, `--backend cli` "
                "or `--backend notebook` (Google Drive resumable upload) can be faster.",
            )

        if total_bytes <= self._CHUNK_BYTES:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError("Colab API upload cancelled by user.")
            b64 = base64.b64encode(local_path.read_bytes()).decode("ascii")
            _, active = self._proxy_request_json(
                active,
                "PUT",
                path_q,
                body={"type": "file", "format": "base64", "content": b64},
                timeout=300.0,
                log_callback=log_callback,
            )
            return active

        with open(local_path, "rb") as fh:
            chunk_index = 1
            sent = 0
            while True:
                if cancel_event is not None and cancel_event.is_set():
                    raise InterruptedError("Colab API upload cancelled by user.")
                data = fh.read(self._CHUNK_BYTES)
                if not data:
                    break
                sent += len(data)
                is_last = sent >= total_bytes
                payload = {
                    "type": "file",
                    "format": "base64",
                    "content": base64.b64encode(data).decode("ascii"),
                    "chunk": -1 if is_last else chunk_index,
                }
                _, active = self._proxy_request_json(
                    active,
                    "PUT",
                    path_q,
                    body=payload,
                    timeout=300.0,
                    log_callback=log_callback,
                )
                pct = int(round((sent / max(1, total_bytes)) * 100))
                _emit_log(
                    log_callback,
                    f"  uploaded {sent / (1024 * 1024):.1f} / {total_mb:.1f} MB ({pct}%)",
                )
                chunk_index += 1
        return active

    def _stream_download_binary(
        self,
        runtime: ColabRuntimeInfo,
        encoded_rel: str,
        local_path: Path,
        *,
        log_callback: LogCallback = None,
    ) -> tuple[bool, ColabRuntimeInfo]:
        """Stream binary file via Jupyter `/files/` endpoint with O(1) memory."""
        active = self.refresh_runtime_connection(runtime, log_callback=log_callback)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_name(f".{local_path.name}.part")
        refreshed_once = False
        try:
            for attempt in range(self._MAX_RETRIES + 1):
                url = f"{active.proxy_url}/files/{encoded_rel}"
                req = build_http_request(
                    url,
                    headers={"X-Colab-Runtime-Proxy-Token": active.proxy_token},
                    method="GET",
                    allow_private=False,
                    allow_loopback=True,
                    allow_http_only_for_loopback=True,
                )
                try:
                    with safe_urlopen(
                        req,
                        timeout=600.0,
                        allow_private=False,
                        allow_loopback=True,
                        allow_http_only_for_loopback=True,
                        urlopen_func=urllib.request.urlopen,
                    ) as resp:
                        with open(tmp, "wb") as dst:
                            while True:
                                chunk = resp.read(self._STREAM_DOWNLOAD_CHUNK_BYTES)
                                if not chunk:
                                    break
                                dst.write(chunk)
                    tmp.replace(local_path)
                    return True, active
                except urllib.error.HTTPError as exc:
                    if exc.code in {404, 405, 501}:
                        return False, active
                    if exc.code in {401, 403} and active.name and not refreshed_once:
                        refreshed_once = True
                        active = self.refresh_runtime_connection(
                            active, force=True, log_callback=log_callback
                        )
                        continue
                    if (
                        exc.code in self._TRANSIENT_HTTP_CODES
                        and attempt < self._MAX_RETRIES
                    ):
                        self._sleep_backoff(attempt)
                        continue
                    return False, active
                except urllib.error.URLError:
                    if attempt < self._MAX_RETRIES:
                        self._sleep_backoff(attempt)
                        continue
                    return False, active
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        return False, active

    def download_file(
        self,
        *,
        runtime: ColabRuntimeInfo,
        remote_path: str,
        local_path: Path,
        log_callback: LogCallback = None,
    ) -> ColabRuntimeInfo:
        active = self.refresh_runtime_connection(runtime, log_callback=log_callback)
        rel = remote_path.lstrip("/")
        if rel.startswith("content/"):
            rel = rel[len("content/") :]
        encoded = urllib.parse.quote(rel, safe="/")
        _emit_log(log_callback, f"Downloading {remote_path} from Colab runtime...")

        # Prefer O(1)-memory binary streaming via Jupyter /files/ endpoint.
        streamed, active = self._stream_download_binary(
            active,
            encoded,
            local_path,
            log_callback=log_callback,
        )
        if streamed:
            return active

        # Fallback: Jupyter Contents API JSON/base64 response.
        payload, active = self._proxy_request_json(
            active,
            "GET",
            f"/api/contents/{encoded}?content=1&type=file",
            timeout=600.0,
            log_callback=log_callback,
        )
        content = payload.get("content")
        fmt = str(payload.get("format") or "").lower()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_name(f".{local_path.name}.part")
        try:
            if fmt == "base64" and isinstance(content, str):
                tmp.write_bytes(base64.b64decode(content))
            elif isinstance(content, str):
                tmp.write_text(content, encoding="utf-8")
            else:
                raise RuntimeError(
                    f"Unexpected Jupyter contents response format {fmt!r} for {remote_path}"
                )
            tmp.replace(local_path)
        finally:
            try:
                if tmp.exists():
                    tmp.unlink()
            except OSError:
                pass
        return active

    def interrupt_kernel(self, runtime: ColabRuntimeInfo, kernel_id: str) -> None:
        try:
            self._proxy_request_json(
                runtime,
                "POST",
                f"/api/kernels/{urllib.parse.quote(kernel_id)}/interrupt",
                timeout=15.0,
            )
        except Exception:
            pass

    def execute_python(
        self,
        *,
        runtime: ColabRuntimeInfo,
        code: str,
        log_callback: LogCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        """Start an ephemeral Jupyter kernel on the runtime and stream execution output."""
        kernel, active = self._proxy_request_json(
            runtime,
            "POST",
            "/api/kernels",
            body={"name": "python3"},
            timeout=60.0,
            log_callback=log_callback,
        )
        kernel_id = str(kernel.get("id") or "").strip()
        if not kernel_id:
            raise RuntimeError("Failed to start Jupyter kernel on Colab runtime.")
        try:
            return asyncio.run(
                self._execute_on_kernel_ws(
                    runtime=active,
                    kernel_id=kernel_id,
                    code=code,
                    log_callback=log_callback,
                    cancel_event=cancel_event,
                )
            )
        finally:
            try:
                self._proxy_request_json(
                    active,
                    "DELETE",
                    f"/api/kernels/{urllib.parse.quote(kernel_id)}",
                    timeout=30.0,
                )
            except Exception:
                pass

    async def _execute_on_kernel_ws(
        self,
        *,
        runtime: ColabRuntimeInfo,
        kernel_id: str,
        code: str,
        log_callback: LogCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        try:
            import websockets
        except ImportError as exc:
            raise ImportError(
                "The `websockets` package is required for direct Colab API execution. "
                'Install with `pip install "annolid[annolid_bot]"` (or `pip install websockets`).'
            ) from exc

        ws_base = runtime.proxy_url.replace("https://", "wss://").replace(
            "http://", "ws://"
        )
        ws_url = f"{ws_base}/api/kernels/{urllib.parse.quote(kernel_id)}/channels"
        headers = {"X-Colab-Runtime-Proxy-Token": runtime.proxy_token}

        connect_kwargs: dict[str, Any] = {"max_size": 16 * 1024 * 1024}
        sig = inspect.signature(websockets.connect)
        if "additional_headers" in sig.parameters:
            connect_kwargs["additional_headers"] = headers
        else:
            connect_kwargs["extra_headers"] = headers
        if "ping_interval" in sig.parameters:
            connect_kwargs["ping_interval"] = 30
        if "ping_timeout" in sig.parameters:
            connect_kwargs["ping_timeout"] = 30

        msg_id = uuid.uuid4().hex
        session_id = uuid.uuid4().hex
        execute_msg = {
            "header": {
                "msg_id": msg_id,
                "username": "annolid",
                "session": session_id,
                "msg_type": "execute_request",
                "version": "5.3",
            },
            "parent_header": {},
            "metadata": {},
            "content": {
                "code": code,
                "silent": False,
                "store_history": False,
                "user_expressions": {},
                "allow_stdin": False,
                "stop_on_error": True,
            },
            "channel": "shell",
            "buffers": [],
        }

        collected: list[str] = []
        error_msg: Optional[str] = None
        reply_received = False
        idle_received = False

        async with websockets.connect(ws_url, **connect_kwargs) as ws:
            await ws.send(json.dumps(execute_msg))
            while not (reply_received and idle_received):
                if cancel_event is not None and cancel_event.is_set():
                    self.interrupt_kernel(runtime, kernel_id)
                    raise InterruptedError("Colab API execution cancelled by user.")
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", errors="replace")
                msg = json.loads(raw)
                parent_id = str((msg.get("parent_header") or {}).get("msg_id") or "")
                if parent_id != msg_id:
                    continue
                msg_type = str(
                    msg.get("msg_type")
                    or (msg.get("header") or {}).get("msg_type")
                    or ""
                )
                content = msg.get("content") or {}
                if msg_type == "stream":
                    text = str(content.get("text") or "")
                    if text:
                        collected.append(text)
                        for line in text.splitlines():
                            _emit_log(log_callback, line)
                elif msg_type == "error":
                    ename = str(content.get("ename") or "Error")
                    evalue = str(content.get("evalue") or "")
                    tb_lines = [str(x) for x in (content.get("traceback") or [])]
                    error_msg = f"{ename}: {evalue}".strip(": ")
                    for line in tb_lines:
                        _emit_log(log_callback, line)
                elif msg_type == "execute_reply":
                    reply_received = True
                    if (
                        str(content.get("status") or "").lower() == "error"
                        and not error_msg
                    ):
                        error_msg = (
                            f"{content.get('ename', 'Error')}: {content.get('evalue', '')}"
                        ).strip(": ")
                elif msg_type == "status":
                    if str(content.get("execution_state") or "") == "idle":
                        idle_received = True

        if error_msg:
            raise RuntimeError(f"Colab remote execution error: {error_msg}")
        return "".join(collected)


# ---------------------------------------------------------------------------
# Backend 3: Google Drive + Generated Notebook Backend (Universal)
# ---------------------------------------------------------------------------


class ColabDriveNotebookBackend:
    """Universal Colab backend that syncs bundles & .ipynb notebooks via Google Drive."""

    def __init__(self, auth_manager: ColabAuthManager) -> None:
        self._auth = auth_manager

    @property
    def name(self) -> str:
        return "notebook"

    def is_drive_ready(self) -> bool:
        status = self._auth.get_auth_status()
        return bool(status.get("drive_ready", False))

    def probe_drive_account(self) -> dict[str, Any]:
        """Run a lightweight read-only Drive API probe (`about.get`) for diagnostics."""
        drive_tool = self._auth.create_drive_tool(allow_interactive_auth=False)
        service = drive_tool._get_service()
        payload = (
            service.about()
            .get(fields="user(displayName,emailAddress),storageQuota(limit,usage)")
            .execute()
        )
        user = payload.get("user") or {}
        quota = payload.get("storageQuota") or {}
        return {
            "display_name": str(user.get("displayName") or ""),
            "email": str(user.get("emailAddress") or ""),
            "storage_limit_bytes": int(quota.get("limit") or 0),
            "storage_usage_bytes": int(quota.get("usage") or 0),
        }

    def upload_job_bundle(
        self,
        *,
        drive_job_rel_path: str,
        notebook_path: Path,
        input_archive_path: Path,
        log_callback: LogCallback = None,
    ) -> dict[str, Any]:
        drive_tool = self._auth.create_drive_tool(
            allow_interactive_auth=self._auth.allow_interactive_auth,
        )
        _emit_log(
            log_callback,
            f"Ensuring Google Drive folder '{drive_job_rel_path}'...",
        )
        folder_id = drive_tool.ensure_folder_path_sync(drive_job_rel_path)

        _emit_log(
            log_callback,
            f"Uploading input bundle {input_archive_path.name} to Google Drive...",
        )
        archive_res = drive_tool.upload_local_file_sync(
            input_archive_path,
            remote_folder_id=folder_id,
            skip_if_exists=False,
        )

        _emit_log(
            log_callback,
            f"Uploading Colab notebook {notebook_path.name} to Google Drive...",
        )
        nb_res = drive_tool.upload_local_file_sync(
            notebook_path,
            remote_folder_id=folder_id,
            skip_if_exists=False,
            mimetype="application/x-ipynb+json",
        )
        nb_file = nb_res.get("file") or {}
        nb_id = str(nb_file.get("id") or "").strip()
        colab_url = build_colab_drive_url(nb_id) if nb_id else ""
        return {
            "drive_folder": drive_job_rel_path,
            "drive_folder_id": folder_id,
            "drive_notebook_id": nb_id,
            "drive_archive_id": str((archive_res.get("file") or {}).get("id") or ""),
            "colab_url": colab_url,
        }

    def pull_job_outputs(
        self,
        *,
        drive_folder_id: str = "",
        drive_folder_path: str = "",
        dest_dir: Path,
        log_callback: LogCallback = None,
    ) -> dict[str, Any]:
        drive_tool = self._auth.create_drive_tool(
            allow_interactive_auth=self._auth.allow_interactive_auth,
        )
        folder_id = str(drive_folder_id or "").strip()
        if not folder_id and drive_folder_path:
            folder_id = drive_tool.ensure_folder_path_sync(drive_folder_path)
        if not folder_id:
            raise ValueError(
                "Either drive_folder_id or drive_folder_path is required to pull outputs."
            )

        files = drive_tool.list_folder_files_sync(folder_id)
        by_name = {str(item.get("name") or ""): item for item in files}
        dest_dir.mkdir(parents=True, exist_ok=True)
        downloaded: list[Path] = []
        result_summary: dict[str, Any] = {}

        if "job_result.json" in by_name:
            manifest_local = dest_dir / "job_result.json"
            _emit_log(log_callback, "Downloading job_result.json from Google Drive...")
            drive_tool.download_remote_file_sync(
                str(by_name["job_result.json"]["id"]),
                manifest_local,
            )
            downloaded.append(manifest_local)
            try:
                result_summary = json.loads(manifest_local.read_text(encoding="utf-8"))
            except Exception:
                result_summary = {}

        if "output_bundle.zip" in by_name:
            bundle_local = dest_dir / "output_bundle.zip"
            _emit_log(
                log_callback, "Downloading output_bundle.zip from Google Drive..."
            )
            drive_tool.download_remote_file_sync(
                str(by_name["output_bundle.zip"]["id"]),
                bundle_local,
            )
            downloaded.append(bundle_local)
        else:
            # Download any model weights / CSV / JSON artifacts found directly in folder
            dest_resolved = dest_dir.resolve()
            for name, meta in by_name.items():
                if name.endswith(
                    (".pt", ".pth", ".onnx", ".csv", ".json")
                ) and not name.endswith(".ipynb"):
                    raw_name = str(name or "")
                    if (
                        not raw_name
                        or "\x00" in raw_name
                        or "/" in raw_name
                        or "\\" in raw_name
                        or raw_name in {".", ".."}
                        or ".." in Path(raw_name).parts
                    ):
                        raise ValueError(
                            f"Refusing unsafe Google Drive filename during download: {raw_name!r}"
                        )
                    local_target = (dest_dir / raw_name).resolve()
                    if not local_target.is_relative_to(dest_resolved):
                        raise ValueError(
                            f"Refusing Google Drive path outside destination directory: {raw_name!r}"
                        )
                    _emit_log(
                        log_callback, f"Downloading {raw_name} from Google Drive..."
                    )
                    drive_tool.download_remote_file_sync(
                        str(meta["id"]),
                        local_target,
                    )
                    downloaded.append(local_target)

        return {
            "drive_folder_id": folder_id,
            "downloaded": downloaded,
            "result_summary": result_summary,
            "remote_files": [str(item.get("name") or "") for item in files],
        }
