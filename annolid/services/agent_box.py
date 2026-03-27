"""Service-layer helpers for Box OAuth bootstrap and token refresh."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

DEFAULT_BOX_AUTHORIZE_BASE_URL = "https://account.box.com"
DEFAULT_BOX_TOKEN_URL = "https://api.box.com/oauth2/token"


def _resolved_config_path(config_path: str | Path | None = None) -> Path:
    from annolid.core.agent.config import get_config_path

    return Path(config_path).expanduser() if config_path else get_config_path()


def _is_loopback_host(hostname: str) -> bool:
    host = str(hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def _default_box_authorize_base_url(value: str | None = None) -> str:
    resolved = str(value or "").strip()
    return resolved or DEFAULT_BOX_AUTHORIZE_BASE_URL


def _persist_box_oauth_config(
    cfg: Any,
    cfg_path: Path,
    *,
    enabled: bool | None = None,
    client_id: str = "",
    client_secret: str = "",
    authorize_base_url: str = "",
    redirect_uri: str = "",
    token_url: str = "",
    access_token: str = "",
    refresh_token: str = "",
) -> None:
    if enabled is not None:
        cfg.tools.box.enabled = bool(enabled)
    cfg.tools.box.client_id = str(client_id or "").strip()
    cfg.tools.box.client_secret = str(client_secret or "").strip()
    cfg.tools.box.authorize_base_url = _default_box_authorize_base_url(
        authorize_base_url
    )
    cfg.tools.box.redirect_uri = str(redirect_uri or "").strip()
    cfg.tools.box.token_url = str(token_url or "").strip() or DEFAULT_BOX_TOKEN_URL
    cfg.tools.box.access_token = str(access_token or "").strip()
    cfg.tools.box.refresh_token = str(refresh_token or "").strip()
    from annolid.core.agent.config import save_config

    save_config(cfg, cfg_path)


def _validate_loopback_redirect_uri(redirect_uri: str) -> tuple[str, int, str]:
    parsed = urlparse(str(redirect_uri or "").strip())
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("redirect_uri must be an absolute local URL.")
    if parsed.scheme.lower() != "http":
        raise ValueError(
            "redirect_uri for the local Box callback must use http://, not https://."
        )
    if not _is_loopback_host(parsed.hostname or ""):
        raise ValueError("redirect_uri must use a localhost or loopback host.")
    path = str(parsed.path or "/") or "/"
    if not path.startswith("/"):
        path = f"/{path}"
    return str(parsed.hostname or "127.0.0.1"), int(parsed.port or 0), path


class BoxOAuthCallbackServer:
    """Temporary local HTTP server that captures a Box OAuth callback code."""

    def __init__(self, *, redirect_uri: str) -> None:
        host, port, path = _validate_loopback_redirect_uri(redirect_uri)

        self.redirect_uri = str(redirect_uri).strip()
        self.host = host
        self.port = int(port or 0)
        self.path = path
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._done = threading.Event()
        self.result: dict[str, Any] | None = None

    def start(self) -> str:
        with self._lock:
            if self._httpd is not None:
                return self.callback_url

            owner = self

            class _Handler(BaseHTTPRequestHandler):
                server_version = "AnnolidBoxOAuthCallback/1.0"

                def log_message(self, fmt: str, *args: object) -> None:
                    return

                def do_GET(self) -> None:  # noqa: N802
                    parsed = urlparse(self.path)
                    if parsed.path != owner.path:
                        self.send_error(404)
                        return
                    query = parse_qs(parsed.query or "")
                    error_value = str((query.get("error") or [""])[0] or "").strip()
                    error_description = str(
                        (query.get("error_description") or [""])[0] or ""
                    ).strip()
                    code = str((query.get("code") or [""])[0] or "").strip()
                    state = str((query.get("state") or [""])[0] or "").strip()
                    if error_value:
                        owner.result = {
                            "ok": False,
                            "error": error_value,
                            "error_description": error_description,
                            "state": state,
                        }
                    else:
                        owner.result = {
                            "ok": True,
                            "code": code,
                            "state": state,
                        }
                    body = (
                        "<html><body><h3>Annolid received the Box authorization."
                        "</h3><p>You can close this tab and return to Annolid.</p>"
                        "</body></html>"
                    ).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                    owner._done.set()

            self._httpd = ThreadingHTTPServer((self.host, self.port), _Handler)
            self.port = int(getattr(self._httpd, "server_port", self.port) or self.port)
            self._thread = threading.Thread(
                target=self._httpd.serve_forever,
                name="BoxOAuthCallbackServer",
                daemon=True,
            )
            self._thread.start()
            return self.callback_url

    def wait_for_result(self, timeout_s: float = 300.0) -> dict[str, Any]:
        self._done.wait(timeout=max(1.0, float(timeout_s or 300.0)))
        return dict(self.result or {})

    def stop(self) -> None:
        with self._lock:
            httpd = self._httpd
            thread = self._thread
            self._httpd = None
            self._thread = None
        self._done.set()
        if httpd is not None:
            try:
                httpd.shutdown()
            finally:
                httpd.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    @property
    def callback_url(self) -> str:
        host = self.host
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        return f"http://{host}:{self.port}{self.path}"


def get_box_oauth_authorize_url(
    *,
    config_path: str | Path | None = None,
    client_id: str | None = None,
    redirect_uri: str | None = None,
    state: str | None = None,
    authorize_base_url: str | None = None,
) -> dict[str, Any]:
    from annolid.core.agent.config import load_config
    from annolid.utils.box import build_box_authorize_url

    cfg_path = _resolved_config_path(config_path)
    cfg = load_config(cfg_path)
    resolved_client_id = str(client_id or cfg.tools.box.client_id or "").strip()
    resolved_redirect_uri = str(
        redirect_uri or cfg.tools.box.redirect_uri or ""
    ).strip()
    resolved_authorize_base_url = _default_box_authorize_base_url(
        authorize_base_url or cfg.tools.box.authorize_base_url
    )
    if not resolved_client_id:
        raise SystemExit(
            "Provide --client-id or set tools.box.client_id in agent config."
        )
    if not resolved_redirect_uri:
        raise SystemExit(
            "Provide --redirect-uri or set tools.box.redirect_uri in agent config."
        )

    url = build_box_authorize_url(
        client_id=resolved_client_id,
        redirect_uri=resolved_redirect_uri,
        state=state,
        authorize_base_url=resolved_authorize_base_url,
    )
    return {
        "ok": True,
        "config_path": str(cfg_path),
        "authorize_url": url,
        "authorize_base_url": resolved_authorize_base_url,
        "client_id": resolved_client_id,
        "redirect_uri": resolved_redirect_uri,
        "state": str(state or ""),
    }


def exchange_box_oauth_code(
    *,
    code: str,
    redirect_uri: str,
    config_path: str | Path | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    token_url: str | None = None,
    authorize_base_url: str | None = None,
    persist: bool = False,
    enable_box: bool = True,
) -> tuple[dict[str, Any], int]:
    from annolid.utils.box import BOX_TOKEN_URL, exchange_box_authorization_code

    cfg_path = _resolved_config_path(config_path)
    from annolid.core.agent.config import load_config

    cfg = load_config(cfg_path)
    resolved_client_id = str(client_id or cfg.tools.box.client_id or "").strip()
    resolved_client_secret = str(
        client_secret or cfg.tools.box.client_secret or ""
    ).strip()
    resolved_authorize_base_url = _default_box_authorize_base_url(
        authorize_base_url or cfg.tools.box.authorize_base_url
    )
    resolved_token_url = str(
        token_url or cfg.tools.box.token_url or BOX_TOKEN_URL
    ).strip()
    resolved_code = str(code or "").strip()
    resolved_redirect = str(redirect_uri or cfg.tools.box.redirect_uri or "").strip()

    if not resolved_code:
        return ({"ok": False, "error": "Missing OAuth authorization code."}, 1)
    if not resolved_redirect:
        return ({"ok": False, "error": "Missing redirect URI."}, 1)
    if not resolved_client_id:
        return ({"ok": False, "error": "Missing Box client_id."}, 1)
    if not resolved_client_secret:
        return ({"ok": False, "error": "Missing Box client_secret."}, 1)

    try:
        tokens = exchange_box_authorization_code(
            client_id=resolved_client_id,
            client_secret=resolved_client_secret,
            code=resolved_code,
            redirect_uri=resolved_redirect,
            token_url=resolved_token_url,
        )
    except Exception as exc:
        return ({"ok": False, "error": f"Failed OAuth exchange: {exc}"}, 1)

    payload: dict[str, Any] = {
        "ok": True,
        "config_path": str(cfg_path),
        "authorize_base_url": resolved_authorize_base_url,
        "token_url": resolved_token_url,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_in": tokens.expires_in,
        "token_type": tokens.token_type,
        "persisted": False,
    }

    if persist:
        _persist_box_oauth_config(
            cfg,
            cfg_path,
            enabled=enable_box,
            client_id=resolved_client_id,
            client_secret=resolved_client_secret,
            authorize_base_url=resolved_authorize_base_url,
            redirect_uri=resolved_redirect,
            token_url=resolved_token_url,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
        )
        payload["persisted"] = True

    return payload, 0


def refresh_box_oauth_token(
    *,
    config_path: str | Path | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    refresh_token: str | None = None,
    token_url: str | None = None,
    persist: bool = False,
) -> tuple[dict[str, Any], int]:
    from annolid.utils.box import BOX_TOKEN_URL, refresh_box_access_token

    cfg_path = _resolved_config_path(config_path)
    from annolid.core.agent.config import load_config

    cfg = load_config(cfg_path)
    resolved_client_id = str(client_id or cfg.tools.box.client_id or "").strip()
    resolved_client_secret = str(
        client_secret or cfg.tools.box.client_secret or ""
    ).strip()
    resolved_refresh = str(refresh_token or cfg.tools.box.refresh_token or "").strip()
    resolved_token_url = str(
        token_url or cfg.tools.box.token_url or BOX_TOKEN_URL
    ).strip()
    resolved_authorize_base_url = _default_box_authorize_base_url(
        cfg.tools.box.authorize_base_url
    )

    if not resolved_client_id:
        return ({"ok": False, "error": "Missing Box client_id."}, 1)
    if not resolved_client_secret:
        return ({"ok": False, "error": "Missing Box client_secret."}, 1)
    if not resolved_refresh:
        return ({"ok": False, "error": "Missing Box refresh_token."}, 1)

    try:
        tokens = refresh_box_access_token(
            client_id=resolved_client_id,
            client_secret=resolved_client_secret,
            refresh_token=resolved_refresh,
            token_url=resolved_token_url,
        )
    except Exception as exc:
        return ({"ok": False, "error": f"Failed token refresh: {exc}"}, 1)

    payload: dict[str, Any] = {
        "ok": True,
        "config_path": str(cfg_path),
        "authorize_base_url": resolved_authorize_base_url,
        "token_url": resolved_token_url,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_in": tokens.expires_in,
        "token_type": tokens.token_type,
        "persisted": False,
    }

    if persist:
        _persist_box_oauth_config(
            cfg,
            cfg_path,
            client_id=resolved_client_id,
            client_secret=resolved_client_secret,
            authorize_base_url=resolved_authorize_base_url,
            token_url=resolved_token_url,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
        )
        payload["persisted"] = True

    return payload, 0


def complete_box_oauth_browser_flow(
    *,
    config_path: str | Path | None = None,
    client_id: str | None = None,
    client_secret: str | None = None,
    redirect_uri: str | None = None,
    authorize_base_url: str | None = None,
    token_url: str | None = None,
    state: str | None = None,
    persist: bool = True,
    timeout_s: float = 300.0,
    open_browser: bool = True,
) -> tuple[dict[str, Any], int]:
    from annolid.utils.box import (
        BOX_TOKEN_URL,
        build_box_authorize_url,
        exchange_box_authorization_code,
    )
    import webbrowser

    cfg_path = _resolved_config_path(config_path)
    from annolid.core.agent.config import load_config

    cfg = load_config(cfg_path)
    resolved_client_id = str(client_id or cfg.tools.box.client_id or "").strip()
    resolved_client_secret = str(
        client_secret or cfg.tools.box.client_secret or ""
    ).strip()
    resolved_redirect_uri = str(
        redirect_uri or cfg.tools.box.redirect_uri or ""
    ).strip()
    resolved_authorize_base_url = _default_box_authorize_base_url(
        authorize_base_url or cfg.tools.box.authorize_base_url
    )
    resolved_token_url = str(
        token_url or cfg.tools.box.token_url or BOX_TOKEN_URL
    ).strip()

    if not resolved_client_id:
        return ({"ok": False, "error": "Missing Box client_id."}, 1)
    if not resolved_client_secret:
        return ({"ok": False, "error": "Missing Box client_secret."}, 1)
    if not resolved_redirect_uri:
        return ({"ok": False, "error": "Missing Box redirect_uri."}, 1)

    try:
        listener = BoxOAuthCallbackServer(redirect_uri=resolved_redirect_uri)
        callback_url = listener.start()
    except Exception as exc:
        return (
            {
                "ok": False,
                "error": f"Failed to start local OAuth callback listener: {exc}",
            },
            1,
        )

    auth_url = build_box_authorize_url(
        client_id=resolved_client_id,
        redirect_uri=resolved_redirect_uri,
        state=state,
        authorize_base_url=resolved_authorize_base_url,
    )
    opened = False
    try:
        if open_browser:
            opened = bool(webbrowser.open_new_tab(auth_url))
    except Exception:
        opened = False

    result: dict[str, Any] = {"ok": False, "persisted": False}
    try:
        callback = listener.wait_for_result(timeout_s=timeout_s)
        if not callback:
            result.update(
                {
                    "error": (
                        "Timed out waiting for the Box redirect. "
                        f"Open this URL manually if needed: {auth_url}"
                    ),
                    "authorize_url": auth_url,
                    "callback_url": callback_url,
                    "opened_in_browser": opened,
                }
            )
            return (result, 1)

        if callback.get("error"):
            result.update(
                {
                    "error": str(callback.get("error") or "Box authorization failed."),
                    "error_description": str(callback.get("error_description") or ""),
                    "authorize_url": auth_url,
                    "callback_url": callback_url,
                    "opened_in_browser": opened,
                }
            )
            return (result, 1)

        code = str(callback.get("code") or "").strip()
        if not code:
            result.update(
                {
                    "error": "Box redirect did not include an authorization code.",
                    "authorize_url": auth_url,
                    "callback_url": callback_url,
                    "opened_in_browser": opened,
                }
            )
            return (result, 1)

        tokens = exchange_box_authorization_code(
            client_id=resolved_client_id,
            client_secret=resolved_client_secret,
            code=code,
            redirect_uri=resolved_redirect_uri,
            token_url=resolved_token_url,
        )
    except Exception as exc:
        result.update(
            {
                "error": f"Failed Box OAuth flow: {exc}",
                "authorize_url": auth_url,
                "callback_url": callback_url,
                "opened_in_browser": opened,
            }
        )
        return (result, 1)
    finally:
        listener.stop()

    result = {
        "ok": True,
        "config_path": str(cfg_path),
        "authorize_base_url": resolved_authorize_base_url,
        "authorize_url": auth_url,
        "callback_url": callback_url,
        "opened_in_browser": opened,
        "token_url": resolved_token_url,
        "access_token": tokens.access_token,
        "refresh_token": tokens.refresh_token,
        "expires_in": tokens.expires_in,
        "token_type": tokens.token_type,
        "persisted": False,
    }

    if persist:
        _persist_box_oauth_config(
            cfg,
            cfg_path,
            enabled=True,
            client_id=resolved_client_id,
            client_secret=resolved_client_secret,
            authorize_base_url=resolved_authorize_base_url,
            redirect_uri=resolved_redirect_uri,
            token_url=resolved_token_url,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
        )
        result["persisted"] = True

    return result, 0


__all__ = [
    "exchange_box_oauth_code",
    "complete_box_oauth_browser_flow",
    "get_box_oauth_authorize_url",
    "BoxOAuthCallbackServer",
    "refresh_box_oauth_token",
]
