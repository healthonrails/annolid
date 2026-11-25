"""Tests for shared Google OAuth refactoring, Google Drive enhancements, and ColabAuthManager."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from annolid.colab.auth import ColabAuthManager, ColabAuthRequiredError
from annolid.core.agent.tools.google_auth import GoogleOAuthHelper
from annolid.core.agent.tools.google_drive import GoogleDriveTool


def test_google_oauth_merge_scopes_preserves_order_and_deduplicates():
    merged = GoogleOAuthHelper.merge_scopes(
        required_scopes=[
            GoogleOAuthHelper.COLAB_SCOPE,
            GoogleOAuthHelper.DRIVE_SCOPE,
        ],
        existing_scopes=[
            GoogleOAuthHelper.CALENDAR_SCOPE,
            GoogleOAuthHelper.DRIVE_SCOPE,
        ],
    )
    assert merged == [
        GoogleOAuthHelper.CALENDAR_SCOPE,
        GoogleOAuthHelper.DRIVE_SCOPE,
        GoogleOAuthHelper.COLAB_SCOPE,
    ]


def test_google_oauth_inspect_status_missing_and_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    creds_file = tmp_path / "client.json"
    token_file = tmp_path / "token.json"

    monkeypatch.setattr(
        GoogleOAuthHelper, "is_oauth_available", staticmethod(lambda: True)
    )

    status_missing = GoogleOAuthHelper.inspect_auth_status(
        credentials_file=str(creds_file),
        token_file=str(token_file),
        required_scopes=GoogleOAuthHelper.COLAB_SCOPES,
    )
    assert status_missing["ready"] is False
    assert status_missing["has_credentials_file"] is False
    assert status_missing["has_token_file"] is False
    assert GoogleOAuthHelper.COLAB_SCOPE in status_missing["missing_scopes"]

    creds_file.write_text('{"installed": {}}', encoding="utf-8")
    token_file.write_text(
        json.dumps(
            {
                "token": "fake-access",
                "refresh_token": "fake-refresh",
                "scopes": list(GoogleOAuthHelper.COLAB_AND_DRIVE_SCOPES),
            }
        ),
        encoding="utf-8",
    )

    status_ready = GoogleOAuthHelper.inspect_auth_status(
        credentials_file=str(creds_file),
        token_file=str(token_file),
        required_scopes=GoogleOAuthHelper.COLAB_SCOPES,
    )
    assert status_ready["ready"] is True
    assert status_ready["has_refresh_token"] is True
    assert status_ready["token_expired"] is False
    assert status_ready["missing_scopes"] == []

    # Missing optional google-auth dependencies must mark status not ready clearly
    monkeypatch.setattr(
        GoogleOAuthHelper, "is_oauth_available", staticmethod(lambda: False)
    )
    status_no_deps = GoogleOAuthHelper.inspect_auth_status(
        credentials_file=str(creds_file),
        token_file=str(token_file),
        required_scopes=GoogleOAuthHelper.COLAB_SCOPES,
    )
    assert status_no_deps["ready"] is False
    assert "not installed" in status_no_deps["reason"].lower()

    # Expired token without refresh_token is flagged as not ready
    monkeypatch.setattr(
        GoogleOAuthHelper, "is_oauth_available", staticmethod(lambda: True)
    )
    token_file.write_text(
        json.dumps(
            {
                "token": "old-access",
                "expiry": "2020-01-01T00:00:00Z",
                "scopes": list(GoogleOAuthHelper.COLAB_AND_DRIVE_SCOPES),
            }
        ),
        encoding="utf-8",
    )
    status_expired = GoogleOAuthHelper.inspect_auth_status(
        credentials_file=str(creds_file),
        token_file=str(token_file),
        required_scopes=GoogleOAuthHelper.COLAB_SCOPES,
    )
    assert status_expired["ready"] is False
    assert status_expired["token_expired"] is True
    assert "expired" in status_expired["reason"].lower()


def test_google_oauth_login_interactive_merges_existing_scopes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    creds_file = tmp_path / "client.json"
    token_file = tmp_path / "token.json"
    creds_file.write_text('{"installed": {}}', encoding="utf-8")
    token_file.write_text(
        json.dumps({"scopes": [GoogleOAuthHelper.CALENDAR_SCOPE]}),
        encoding="utf-8",
    )

    requested_scopes: list[str] = []

    class _FakeCreds:
        def to_json(self) -> str:
            return json.dumps(
                {
                    "token": "new-tok",
                    "refresh_token": "new-ref",
                    "scopes": requested_scopes,
                }
            )

    def _fake_flow(credentials_path: Path, scopes, *, open_browser: bool = True):
        assert credentials_path == creds_file
        requested_scopes.extend(list(scopes))
        return _FakeCreds()

    monkeypatch.setattr(
        GoogleOAuthHelper,
        "_run_installed_app_flow",
        classmethod(lambda cls, *a, **k: _fake_flow(*a, **k)),
    )

    GoogleOAuthHelper.login_interactive(
        credentials_file=str(creds_file),
        token_file=str(token_file),
        scopes=GoogleOAuthHelper.COLAB_AND_DRIVE_SCOPES,
        merge_existing_scopes=True,
    )

    assert GoogleOAuthHelper.CALENDAR_SCOPE in requested_scopes
    assert GoogleOAuthHelper.COLAB_SCOPE in requested_scopes
    assert GoogleOAuthHelper.DRIVE_SCOPE in requested_scopes
    saved = json.loads(token_file.read_text(encoding="utf-8"))
    assert GoogleOAuthHelper.CALENDAR_SCOPE in saved["scopes"]
    assert GoogleOAuthHelper.clear_token(str(token_file)) is True
    assert GoogleOAuthHelper.clear_token(str(token_file)) is False


def test_google_oauth_get_credentials_allow_browser_auth_without_tty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    creds_file = tmp_path / "client.json"
    token_file = tmp_path / "token.json"
    creds_file.write_text('{"installed": {}}', encoding="utf-8")

    monkeypatch.setattr(
        GoogleOAuthHelper, "interactive_auth_possible", staticmethod(lambda: False)
    )

    # Default allow_browser_auth=False still refuses without TTY
    with pytest.raises(RuntimeError, match="no TTY"):
        GoogleOAuthHelper.get_credentials(
            credentials_file=str(creds_file),
            token_file=str(token_file),
            scopes=[GoogleOAuthHelper.DRIVE_SCOPE],
            allow_interactive_auth=True,
            allow_browser_auth=False,
        )

    class _FakeCreds:
        valid = True
        token = "browser-token"

        def to_json(self) -> str:
            return json.dumps(
                {"token": self.token, "scopes": [GoogleOAuthHelper.DRIVE_SCOPE]}
            )

    monkeypatch.setattr(
        GoogleOAuthHelper,
        "_run_installed_app_flow",
        classmethod(lambda cls, *_a, **_k: _FakeCreds()),
    )

    creds = GoogleOAuthHelper.get_credentials(
        credentials_file=str(creds_file),
        token_file=str(token_file),
        scopes=[GoogleOAuthHelper.DRIVE_SCOPE],
        allow_interactive_auth=True,
        allow_browser_auth=True,
    )
    assert creds.token == "browser-token"
    assert token_file.is_file()


def test_google_drive_mimetype_detection_and_download_action(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    assert (
        GoogleDriveTool._guess_upload_mimetype(Path("run.ipynb"))
        == "application/x-ipynb+json"
    )
    assert (
        GoogleDriveTool._guess_upload_mimetype(Path("bundle.zip")) == "application/zip"
    )
    assert GoogleDriveTool._guess_upload_mimetype(Path("clip.mp4")) == "video/mp4"
    assert (
        GoogleDriveTool._guess_upload_mimetype(Path("best.pt"))
        == "application/octet-stream"
    )

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "outside.zip"
    tool = GoogleDriveTool(allowed_local_roots=[allowed])
    monkeypatch.setattr(tool, "_get_service", lambda: object())

    # Rejects missing required args
    assert "requires `file_id`" in asyncio.run(tool.execute(action="download_file"))
    assert "requires `local_path`" in asyncio.run(
        tool.execute(action="download_file", file_id="fid-1")
    )

    # Enforces allowed roots
    err = asyncio.run(
        tool.execute(
            action="download_file",
            file_id="fid-1",
            local_path=str(outside),
        )
    )
    assert "outside allowed roots" in err.lower()

    # Happy path
    target = allowed / "out.zip"
    monkeypatch.setattr(
        tool,
        "_download_file_resumable",
        lambda _svc, *, file_id, local_path, **_kw: {
            "file_id": file_id,
            "local_path": str(local_path),
            "size_bytes": 123,
        },
    )
    ok = json.loads(
        asyncio.run(
            tool.execute(
                action="download_file",
                file_id="fid-1",
                local_path=str(target),
            )
        )
    )
    assert ok["status"] == "downloaded"
    assert ok["file_id"] == "fid-1"
    assert ok["size_bytes"] == 123


def test_colab_auth_manager_recommendations_and_env_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    creds = tmp_path / "client.json"
    token = tmp_path / "token.json"
    adc_file = tmp_path / "adc.json"
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(adc_file))
    monkeypatch.setattr(
        GoogleOAuthHelper, "is_oauth_available", staticmethod(lambda: True)
    )

    mgr = ColabAuthManager(credentials_file=str(creds), token_file=str(token))
    monkeypatch.setattr(mgr, "find_colab_cli", lambda: None)
    monkeypatch.setattr(mgr, "find_gcloud_cli", lambda: None)

    st = mgr.get_auth_status()
    assert st["recommended_backend"] == "notebook"
    assert st["api_ready"] is False
    assert st["drive_ready"] is False

    # Drive OAuth ready -> still notebook backend, with drive_ready=True
    creds.write_text('{"installed": {}}', encoding="utf-8")
    token.write_text(
        json.dumps({"token": "t", "scopes": [GoogleOAuthHelper.DRIVE_SCOPE]}),
        encoding="utf-8",
    )
    st_drive = mgr.get_auth_status()
    assert st_drive["recommended_backend"] == "notebook"
    assert st_drive["drive_ready"] is True
    assert st_drive["api_ready"] is False

    # Colab OAuth ready -> recommends api
    token.write_text(
        json.dumps(
            {"token": "t", "scopes": list(GoogleOAuthHelper.COLAB_AND_DRIVE_SCOPES)}
        ),
        encoding="utf-8",
    )
    st_api = mgr.get_auth_status()
    assert st_api["recommended_backend"] == "api"
    assert st_api["api_ready"] is True

    # Colab CLI installed -> recommends cli
    monkeypatch.setattr(mgr, "find_colab_cli", lambda: "/usr/local/bin/colab")
    monkeypatch.setattr(
        "annolid.colab.auth.subprocess.run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout="colab 0.2.0\n", stderr=""
        ),
    )
    st_cli = mgr.get_auth_status()
    assert st_cli["recommended_backend"] == "cli"
    assert st_cli["cli_ready"] is True

    # Explicit env token fallback for API token retrieval
    monkeypatch.setenv("ANNOLID_COLAB_ACCESS_TOKEN", "env-secret-token")
    monkeypatch.setattr(
        GoogleOAuthHelper,
        "get_credentials",
        classmethod(
            lambda cls, *a, **k: (_ for _ in ()).throw(RuntimeError("no oauth"))
        ),
    )
    monkeypatch.setattr(
        GoogleOAuthHelper,
        "get_adc_credentials",
        classmethod(lambda cls, *a, **k: (_ for _ in ()).throw(RuntimeError("no adc"))),
    )
    assert mgr.get_colab_api_access_token() == "env-secret-token"

    # Without env token or valid OAuth/ADC, raises ColabAuthRequiredError
    monkeypatch.delenv("ANNOLID_COLAB_ACCESS_TOKEN", raising=False)
    with pytest.raises(ColabAuthRequiredError):
        mgr.get_colab_api_access_token()


def test_colab_auth_manager_install_colab_cli_and_local_bin_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_home = tmp_path / "home"
    local_bin = fake_home / ".local" / "bin"
    local_bin.mkdir(parents=True)
    uv_bin = local_bin / "uv"
    uv_bin.write_text("#!/bin/sh\n", encoding="utf-8")
    uv_bin.chmod(0o755)
    colab_bin = local_bin / "colab"

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    monkeypatch.setattr("annolid.colab.auth.shutil.which", lambda _name: None)

    mgr = ColabAuthManager(
        credentials_file=str(tmp_path / "missing_c.json"),
        token_file=str(tmp_path / "missing_t.json"),
    )
    assert mgr.find_colab_cli() is None

    recorded: list[list[str]] = []

    def _fake_install(cmd, capture_output=True, text=True, timeout=300):
        recorded.append(list(cmd))
        colab_bin.write_text("#!/bin/sh\necho colab 0.2.0\n", encoding="utf-8")
        colab_bin.chmod(0o755)
        return SimpleNamespace(
            returncode=0, stdout="Installed google-colab-cli\n", stderr=""
        )

    monkeypatch.setattr("annolid.colab.auth.subprocess.run", _fake_install)
    logs: list[str] = []
    installed = mgr.install_colab_cli(log_callback=logs.append)
    assert installed == str(colab_bin)
    assert recorded == [[str(uv_bin), "tool", "install", "google-colab-cli"]]
    assert mgr.find_colab_cli() == str(colab_bin)
    assert any("Installed google-colab-cli" in line for line in logs)


def test_colab_auth_manager_account_info_and_switch_account_logout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    prefs = tmp_path / "auth_prefs.json"
    token = tmp_path / "token.json"
    token.write_text('{"token": "old"}', encoding="utf-8")
    mgr = ColabAuthManager(
        credentials_file=str(tmp_path / "creds.json"),
        token_file=str(token),
        prefs_file=prefs,
    )
    mgr._save_auth_prefs({"colab_cli_auth_mode": "adc"})
    monkeypatch.setattr(mgr, "find_colab_cli", lambda: "/usr/local/bin/colab")
    monkeypatch.setattr(mgr, "find_gcloud_cli", lambda: "/opt/homebrew/bin/gcloud")
    monkeypatch.setattr(mgr, "_colab_cli_python", lambda: "/usr/local/bin/python3")

    def _fake_run(cmd, **_kwargs):
        if "whoami" in cmd:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Auth provider: adc\n"
                    "Email:         researcher@gmail.com\n"
                    "Expires in:    58m\n"
                ),
                stderr="",
            )
        if "usage" in cmd:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Current balance: 15.50 compute units\n"
                    "Usage rate: 1.96/hr\n"
                    "Active assignments: 1\n"
                ),
                stderr="",
            )
        if "revoke" in cmd:
            return SimpleNamespace(returncode=0, stdout="revoked", stderr="")
        if any("keyring.delete_password" in str(a) for a in cmd):
            return SimpleNamespace(returncode=0, stdout="DELETED\n", stderr="")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("annolid.colab.auth.subprocess.run", _fake_run)
    info = mgr.get_account_info()
    assert info["signed_in"] is True
    assert info["email"] == "researcher@gmail.com"
    assert info["provider"] == "colab-cli (adc)"
    assert info["compute_balance"] == "15.50 compute units"
    assert info["usage_rate"] == "1.96/hr"
    assert info["active_assignments"] == 1

    out = mgr.logout(include_colab_cli=True, clear_adc=True)
    assert out["removed_token"] is True
    assert out["cleared_colab_cli"] is True
    assert out["cleared_adc"] is True
    assert not prefs.exists()
