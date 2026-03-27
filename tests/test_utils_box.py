from __future__ import annotations

import io
import json
from urllib import error

import pytest

from annolid.utils.box import (
    BoxApiClient,
    build_box_authorize_url,
    exchange_box_authorization_code,
    refresh_box_access_token,
)
from annolid.core.agent.config import AgentConfig, save_config
from annolid.services.agent_box import (
    BoxOAuthCallbackServer,
    complete_box_oauth_browser_flow,
    get_box_oauth_authorize_url,
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def test_build_box_authorize_url() -> None:
    url = build_box_authorize_url(
        client_id="cid",
        redirect_uri="https://example.com/callback",
        state="xyz",
    )
    assert "client_id=cid" in url
    assert "redirect_uri=https%3A%2F%2Fexample.com%2Fcallback" in url
    assert "state=xyz" in url


def test_build_box_authorize_url_uses_custom_host() -> None:
    url = build_box_authorize_url(
        client_id="cid",
        redirect_uri="https://example.com/callback",
        authorize_base_url="https://my_org_xxx.account.box.com",
    )
    assert url.startswith("https://my_org_xxx.account.box.com/api/oauth2/authorize?")


def test_exchange_and_refresh_token_requests(monkeypatch) -> None:
    calls: list[dict[str, str]] = []

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        body = req.data.decode("utf-8") if getattr(req, "data", None) else ""
        calls.append({"url": req.full_url, "body": body})
        if "grant_type=authorization_code" in body:
            return _FakeResponse(
                json.dumps(
                    {
                        "access_token": "a1",
                        "refresh_token": "r1",
                        "expires_in": 3600,
                        "token_type": "bearer",
                    }
                ).encode("utf-8")
            )
        return _FakeResponse(
            json.dumps(
                {
                    "access_token": "a2",
                    "refresh_token": "r2",
                    "expires_in": 3600,
                    "token_type": "bearer",
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("annolid.utils.box.request.urlopen", _fake_urlopen)

    exchanged = exchange_box_authorization_code(
        client_id="cid",
        client_secret="sec",
        code="code123",
        redirect_uri="https://example.com/callback",
    )
    refreshed = refresh_box_access_token(
        client_id="cid",
        client_secret="sec",
        refresh_token="r1",
    )

    assert exchanged.access_token == "a1"
    assert exchanged.refresh_token == "r1"
    assert refreshed.access_token == "a2"
    assert refreshed.refresh_token == "r2"
    assert len(calls) == 2


def test_get_box_oauth_authorize_url_uses_saved_redirect_uri(tmp_path) -> None:
    cfg = AgentConfig()
    cfg.tools.box.client_id = "cid"
    cfg.tools.box.redirect_uri = "https://localhost:8765/oauth/callback"
    cfg.tools.box.authorize_base_url = "https://my_org_xxx.account.box.com"
    cfg_path = tmp_path / "config.json"
    save_config(cfg, cfg_path)

    payload = get_box_oauth_authorize_url(config_path=cfg_path)
    assert payload["redirect_uri"] == "https://localhost:8765/oauth/callback"
    assert payload["authorize_url"].startswith(
        "https://my_org_xxx.account.box.com/api/oauth2/authorize?"
    )


def test_box_api_client_auto_refresh_on_401(monkeypatch) -> None:
    state = {"api_calls": 0, "token_calls": 0}

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        body = req.data.decode("utf-8") if getattr(req, "data", None) else ""
        if "grant_type=refresh_token" in body:
            state["token_calls"] += 1
            return _FakeResponse(
                json.dumps(
                    {
                        "access_token": "new-access",
                        "refresh_token": "new-refresh",
                        "expires_in": 3600,
                        "token_type": "bearer",
                    }
                ).encode("utf-8")
            )

        state["api_calls"] += 1
        if state["api_calls"] == 1:
            payload = json.dumps({"message": "Unauthorized"}).encode("utf-8")
            raise error.HTTPError(
                url=req.full_url,
                code=401,
                msg="Unauthorized",
                hdrs=None,
                fp=io.BytesIO(payload),
            )
        return _FakeResponse(json.dumps({"ok": True}).encode("utf-8"))

    monkeypatch.setattr("annolid.utils.box.request.urlopen", _fake_urlopen)

    client = BoxApiClient(
        access_token="old-access",
        client_id="cid",
        client_secret="sec",
        refresh_token="old-refresh",
    )
    payload = client.request_json("GET", "https://api.box.com/2.0/folders/0/items")
    assert payload["ok"] is True
    assert state["api_calls"] == 2
    assert state["token_calls"] == 1
    assert client.access_token == "new-access"
    assert client.refresh_token == "new-refresh"


def test_box_oauth_callback_server_captures_code() -> None:
    server = BoxOAuthCallbackServer(redirect_uri="http://127.0.0.1:0/oauth/callback")
    assert server.host == "127.0.0.1"
    assert server.path == "/oauth/callback"
    server.port = 8765
    assert server.callback_url == "http://127.0.0.1:8765/oauth/callback"


def test_box_oauth_callback_server_rejects_https_loopback() -> None:
    with pytest.raises(ValueError, match="must use http://"):
        BoxOAuthCallbackServer(redirect_uri="https://localhost:8765/oauth/callback")


def test_complete_box_oauth_browser_flow(monkeypatch) -> None:
    opened: list[str] = []

    def _fake_open_new_tab(url: str) -> bool:
        opened.append(url)
        return True

    def _fake_exchange_box_authorization_code(**kwargs):
        assert kwargs["code"] == "abc123"
        return type(
            "_Tokens",
            (),
            {
                "access_token": "token-a",
                "refresh_token": "token-r",
                "expires_in": 3600,
                "token_type": "bearer",
            },
        )()

    monkeypatch.setattr("webbrowser.open_new_tab", _fake_open_new_tab)
    monkeypatch.setattr(
        "annolid.utils.box.exchange_box_authorization_code",
        _fake_exchange_box_authorization_code,
    )

    class _FakeListener:
        def __init__(self, *, redirect_uri: str) -> None:
            self.redirect_uri = redirect_uri
            self.callback_url = "http://127.0.0.1:8765/oauth/callback"

        def start(self) -> str:
            return self.callback_url

        def wait_for_result(self, timeout_s: float = 300.0) -> dict[str, str]:
            return {"ok": True, "code": "abc123", "state": ""}

        def stop(self) -> None:
            return None

    monkeypatch.setattr(
        "annolid.services.agent_box.BoxOAuthCallbackServer",
        _FakeListener,
    )

    result, exit_code = complete_box_oauth_browser_flow(
        client_id="cid",
        client_secret="sec",
        redirect_uri="http://127.0.0.1:0/oauth/callback",
        authorize_base_url="https://my_org_xxx.account.box.com",
        persist=False,
        timeout_s=5.0,
        open_browser=True,
    )
    assert exit_code == 0
    assert result["ok"] is True
    assert opened
    assert result["authorize_url"].startswith(
        "https://my_org_xxx.account.box.com/api/oauth2/authorize?"
    )


def test_complete_box_oauth_browser_flow_persists_tokens(monkeypatch) -> None:
    saved: dict[str, object] = {}

    def _fake_open_new_tab(url: str) -> bool:
        return True

    def _fake_exchange_box_authorization_code(**kwargs):
        return type(
            "_Tokens",
            (),
            {
                "access_token": "token-a",
                "refresh_token": "token-r",
                "expires_in": 3600,
                "token_type": "bearer",
            },
        )()

    class _FakeListener:
        def __init__(self, *, redirect_uri: str) -> None:
            self.redirect_uri = redirect_uri
            self.callback_url = "http://127.0.0.1:8765/oauth/callback"

        def start(self) -> str:
            return self.callback_url

        def wait_for_result(self, timeout_s: float = 300.0) -> dict[str, str]:
            return {"ok": True, "code": "abc123", "state": ""}

        def stop(self) -> None:
            return None

    def _fake_save_config(cfg, cfg_path) -> None:
        saved["cfg_path"] = str(cfg_path)
        saved["client_id"] = cfg.tools.box.client_id
        saved["client_secret"] = cfg.tools.box.client_secret
        saved["redirect_uri"] = cfg.tools.box.redirect_uri
        saved["access_token"] = cfg.tools.box.access_token
        saved["refresh_token"] = cfg.tools.box.refresh_token

    monkeypatch.setattr("webbrowser.open_new_tab", _fake_open_new_tab)
    monkeypatch.setattr(
        "annolid.utils.box.exchange_box_authorization_code",
        _fake_exchange_box_authorization_code,
    )
    monkeypatch.setattr(
        "annolid.services.agent_box.BoxOAuthCallbackServer",
        _FakeListener,
    )
    monkeypatch.setattr("annolid.core.agent.config.save_config", _fake_save_config)

    result, exit_code = complete_box_oauth_browser_flow(
        client_id="cid",
        client_secret="sec",
        redirect_uri="http://127.0.0.1:0/oauth/callback",
        authorize_base_url="https://my_org_xxx.account.box.com",
        persist=True,
        timeout_s=5.0,
        open_browser=True,
    )
    assert exit_code == 0
    assert result["ok"] is True
    assert result["persisted"] is True
    assert saved["client_id"] == "cid"
    assert saved["client_secret"] == "sec"
    assert saved["redirect_uri"] == "http://127.0.0.1:0/oauth/callback"
    assert saved["access_token"] == "token-a"
    assert saved["refresh_token"] == "token-r"
