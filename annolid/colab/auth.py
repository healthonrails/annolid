from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path
from typing import Any, Callable, Optional

from annolid.core.agent.tools.google_auth import GoogleOAuthHelper
from annolid.core.agent.tools.google_drive import GoogleDriveTool

DEFAULT_GOOGLE_CREDENTIALS_FILE = "~/.annolid/agent/google_oauth_credentials.json"
DEFAULT_GOOGLE_TOKEN_FILE = "~/.annolid/agent/google_oauth_token.json"


class ColabAuthRequiredError(RuntimeError):
    """Raised when Google Colab/Drive auth is missing, expired, or lacks scopes."""

    def __init__(
        self,
        message: str,
        *,
        backend: str = "auto",
        auth_status: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.backend = str(backend or "auto")
        self.auth_status = dict(auth_status or {})


def _load_configured_google_auth_paths() -> tuple[str, str, bool]:
    """Read Google OAuth paths from ~/.annolid/config.json if available."""
    try:
        from annolid.core.agent.config.loader import load_config

        cfg = load_config()
        gauth = getattr(getattr(cfg, "tools", None), "google_auth", None)
        if gauth is not None:
            creds = str(
                getattr(gauth, "credentials_file", "")
                or DEFAULT_GOOGLE_CREDENTIALS_FILE
            ).strip()
            token = str(
                getattr(gauth, "token_file", "") or DEFAULT_GOOGLE_TOKEN_FILE
            ).strip()
            allow = bool(getattr(gauth, "allow_interactive_auth", False))
            return creds, token, allow
    except Exception:
        pass
    return DEFAULT_GOOGLE_CREDENTIALS_FILE, DEFAULT_GOOGLE_TOKEN_FILE, False


class ColabAuthManager:
    """Coordinate Google Colab and Google Drive authentication across official flows.

    Supported authentication paths:
    1. Official `google-colab-cli` (`colab` binary; personal Google account OAuth2 or ADC).
    2. Shared Annolid Google OAuth2 (`google_oauth_credentials.json` + `google_oauth_token.json`)
       for native Colab REST API (`colaboratory.googleapis.com/v1beta`) and/or Google Drive sync.
    3. Google Cloud Application Default Credentials (`gcloud auth application-default login`
       with `colaboratory` scope).
    """

    def __init__(
        self,
        *,
        credentials_file: Optional[str] = None,
        token_file: Optional[str] = None,
        allow_interactive_auth: Optional[bool] = None,
        colab_cli_path: Optional[str] = None,
        colab_cli_auth_mode: str = "auto",
        prefs_file: Optional[Path] = None,
    ) -> None:
        cfg_creds, cfg_token, cfg_allow = _load_configured_google_auth_paths()
        self.credentials_file = str(credentials_file or cfg_creds).strip()
        self.token_file = str(token_file or cfg_token).strip()
        self.allow_interactive_auth = (
            bool(allow_interactive_auth)
            if allow_interactive_auth is not None
            else bool(cfg_allow)
        )
        self._colab_cli_override = str(colab_cli_path or "").strip() or None
        self.colab_cli_auth_mode = (
            str(colab_cli_auth_mode or "auto").strip().lower() or "auto"
        )
        self._prefs_file = (
            Path(prefs_file).expanduser()
            if prefs_file is not None
            else Path.home() / ".annolid" / "colab" / "auth_prefs.json"
        )

    def _read_auth_prefs(self) -> dict[str, Any]:
        try:
            if self._prefs_file.is_file():
                data = json.loads(self._prefs_file.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return data
        except Exception:
            pass
        return {}

    def _save_auth_prefs(self, updates: dict[str, Any]) -> None:
        try:
            data = self._read_auth_prefs()
            data.update(updates)
            self._prefs_file.parent.mkdir(parents=True, exist_ok=True)
            self._prefs_file.write_text(
                json.dumps(data, indent=2) + "\n", encoding="utf-8"
            )
        except Exception:
            pass

    def resolve_colab_cli_auth_mode(self) -> str:
        """Resolve which `--auth` mode (`oauth2` or `adc`) to pass to `colab`."""
        explicit = str(self.colab_cli_auth_mode or "auto").strip().lower()
        if explicit in {"oauth2", "adc"}:
            return explicit
        saved = (
            str(self._read_auth_prefs().get("colab_cli_auth_mode") or "")
            .strip()
            .lower()
        )
        if saved in {"oauth2", "adc"}:
            return saved
        return "oauth2"

    def find_colab_cli(self) -> Optional[str]:
        if self._colab_cli_override:
            candidate = Path(self._colab_cli_override).expanduser()
            if candidate.is_file():
                return str(candidate)
            return shutil.which(self._colab_cli_override)
        found = shutil.which("colab")
        if found:
            return found
        # GUI launches on macOS/Linux often omit ~/.local/bin from PATH,
        # which is where `uv tool install google-colab-cli` installs `colab`.
        for fallback in (
            Path.home() / ".local" / "bin" / "colab",
            Path(sys.executable).resolve().parent / "colab",
        ):
            if fallback.is_file() and os.access(fallback, os.X_OK):
                return str(fallback)
        return None

    def find_gcloud_cli(self) -> Optional[str]:
        found = shutil.which("gcloud")
        if found:
            return found
        for fallback in (
            Path("/opt/homebrew/bin/gcloud"),
            Path("/usr/local/bin/gcloud"),
        ):
            if fallback.is_file() and os.access(fallback, os.X_OK):
                return str(fallback)
        return None

    def install_colab_cli(
        self,
        *,
        log_callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Install the official `google-colab-cli` package via `uv` or `pip`."""
        existing = self.find_colab_cli()
        if existing:
            return existing
        uv_bin = shutil.which("uv") or (
            str(Path.home() / ".local" / "bin" / "uv")
            if (Path.home() / ".local" / "bin" / "uv").is_file()
            else None
        )
        if uv_bin:
            cmd = [uv_bin, "tool", "install", "google-colab-cli"]
        else:
            cmd = [sys.executable, "-m", "pip", "install", "google-colab-cli"]
        if log_callback is not None:
            log_callback("Installing official google-colab-cli: " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise RuntimeError(
                f"Failed to install google-colab-cli (exit {proc.returncode}): {detail}"
            )
        installed = self.find_colab_cli()
        if not installed:
            raise RuntimeError(
                "`google-colab-cli` installed, but the `colab` binary was not found on PATH "
                "or in ~/.local/bin. Add ~/.local/bin to PATH and retry."
            )
        if log_callback is not None:
            log_callback(f"Installed google-colab-cli at {installed}")
        return installed

    def inspect_colab_cli(self) -> dict[str, Any]:
        cli_path = self.find_colab_cli()
        system_name = platform.system().lower()
        native_os_supported = system_name in {"darwin", "linux"}
        version_text = ""
        if cli_path:
            try:
                proc = subprocess.run(
                    [cli_path, "version"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                lines = (proc.stdout or proc.stderr or "").strip().splitlines()
                version_text = lines[0].strip() if lines else ""
            except Exception:
                version_text = ""
        sessions_file = Path.home() / ".config" / "colab-cli" / "sessions.json"
        token_file = Path.home() / ".config" / "colab-cli" / "token.json"
        try:
            has_local_state = bool(sessions_file.exists() or token_file.exists())
        except OSError:
            has_local_state = False
        return {
            "installed": bool(cli_path),
            "binary": cli_path or "",
            "version": version_text,
            "native_os_supported": native_os_supported,
            "auth_mode": self.resolve_colab_cli_auth_mode(),
            "has_local_state": has_local_state,
            "install_command": "uv tool install google-colab-cli",
            "pip_install_command": "pip install google-colab-cli",
        }

    def get_auth_status(self) -> dict[str, Any]:
        cli_info = self.inspect_colab_cli()
        oauth_colab = GoogleOAuthHelper.inspect_auth_status(
            credentials_file=self.credentials_file,
            token_file=self.token_file,
            required_scopes=GoogleOAuthHelper.COLAB_SCOPES,
        )
        oauth_drive = GoogleOAuthHelper.inspect_auth_status(
            credentials_file=self.credentials_file,
            token_file=self.token_file,
            required_scopes=[GoogleOAuthHelper.DRIVE_SCOPE],
        )
        adc_info = GoogleOAuthHelper.inspect_adc_status()
        adc_info["gcloud_installed"] = bool(self.find_gcloud_cli())

        api_ready = bool(oauth_colab["ready"] or adc_info["ready"])
        cli_ready = bool(cli_info["installed"])
        drive_ready = bool(oauth_drive["ready"])

        if cli_ready:
            recommended_backend = "cli"
            summary = "Google Colab CLI (`colab`) is available for direct GPU/TPU runs."
        elif api_ready:
            recommended_backend = "api"
            summary = (
                "Google Colab REST API credentials are ready for direct remote runs."
            )
        elif drive_ready:
            recommended_backend = "notebook"
            summary = (
                "Google Drive OAuth is ready: Annolid can upload datasets & notebooks "
                "directly to Drive and open them in Colab."
            )
        else:
            recommended_backend = "notebook"
            summary = (
                "No direct Colab runtime auth configured yet. Install `google-colab-cli`, "
                "sign in with Google OAuth / gcloud ADC, or generate a local .ipynb "
                "notebook for Colab browser upload."
            )

        return {
            "recommended_backend": recommended_backend,
            "summary": summary,
            "colab_cli": cli_info,
            "oauth_colab": oauth_colab,
            "oauth_drive": oauth_drive,
            "adc": adc_info,
            "drive_ready": drive_ready,
            "api_ready": api_ready,
            "cli_ready": cli_ready,
            "credentials_file": str(Path(self.credentials_file).expanduser()),
            "token_file": str(Path(self.token_file).expanduser()),
        }

    def login(
        self,
        *,
        method: str = "auto",
        include_drive: bool = True,
        open_browser: bool = True,
        install_cli_if_missing: bool = False,
        log_callback: Optional[Callable[[str], None]] = None,
        auth_code_callback: Optional[Callable[[str], Optional[str]]] = None,
    ) -> dict[str, Any]:
        selected = str(method or "auto").strip().lower()
        if selected == "personal":
            if (
                platform.system().lower() in {"darwin", "linux"}
                and not self.find_colab_cli()
            ):
                self.install_colab_cli(log_callback=log_callback)
            if self.find_gcloud_cli():
                selected = "adc"
            elif self.find_colab_cli():
                selected = "colab-cli"
            elif Path(self.credentials_file).expanduser().is_file():
                selected = "oauth2"
            else:
                selected = "colab-cli"
        if selected == "auto":
            if self.find_colab_cli():
                selected = "colab-cli"
            elif install_cli_if_missing and platform.system().lower() in {
                "darwin",
                "linux",
            }:
                selected = "colab-cli"
            elif Path(self.credentials_file).expanduser().is_file():
                selected = "oauth2"
            elif self.find_gcloud_cli():
                selected = "adc"
            else:
                raise ColabAuthRequiredError(
                    "No Colab authentication tool or OAuth client file found.\n"
                    "Choose one of:\n"
                    "  1. Install the official Google Colab CLI (personal Google account, no GCP project needed):\n"
                    "     uv tool install google-colab-cli\n"
                    "  2. Or configure a Google Cloud OAuth client JSON at:\n"
                    f"     {Path(self.credentials_file).expanduser()}\n"
                    "  3. Or install `gcloud` CLI and run:\n"
                    f"     {GoogleOAuthHelper.inspect_adc_status()['setup_command']}",
                    backend="auto",
                    auth_status=self.get_auth_status(),
                )

        if selected == "colab-cli":
            cli_path = self.find_colab_cli()
            if not cli_path and install_cli_if_missing:
                cli_path = self.install_colab_cli(log_callback=log_callback)
            if not cli_path:
                raise ColabAuthRequiredError(
                    "Official `google-colab-cli` (`colab`) not found on PATH. "
                    "Install with `uv tool install google-colab-cli` (or pass `--install-cli`).",
                    backend="cli",
                    auth_status=self.get_auth_status(),
                )
            cli_mode = (
                self.colab_cli_auth_mode
                if self.colab_cli_auth_mode in {"oauth2", "adc"}
                else "oauth2"
            )
            cmd = [cli_path, f"--auth={cli_mode}", "sessions"]
            if log_callback is not None:
                log_callback(
                    f"Launching Google Colab CLI browser sign-in (`colab --auth={cli_mode} sessions`)..."
                )
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE if auth_code_callback is not None else None,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            lines: list[str] = []
            auth_url = ""
            url_opened = False
            code_submitted = False
            url_re = re.compile(r"(https://accounts\.google\.com/\S+)")
            try:
                for line in proc.stdout:
                    lines.append(line)
                    stripped = line.rstrip("\n")
                    if log_callback is not None:
                        log_callback(stripped)
                    m = url_re.search(stripped)
                    if m and not url_opened:
                        auth_url = m.group(1)
                        if open_browser:
                            try:
                                webbrowser.open(auth_url)
                                url_opened = True
                                if log_callback is not None:
                                    log_callback(
                                        "Opened the current Google sign-in URL in your browser. "
                                        "Approve access in THAT tab and paste the displayed code."
                                    )
                            except Exception:
                                pass
                    if not code_submitted and "authorization code" in stripped.lower():
                        if auth_code_callback is not None and proc.stdin is not None:
                            code_submitted = True
                            pasted = auth_code_callback(auth_url)
                            if not pasted or not str(pasted).strip():
                                proc.terminate()
                                raise ColabAuthRequiredError(
                                    "Google Colab sign-in was cancelled before entering the authorization code.",
                                    backend="cli",
                                    auth_status=self.get_auth_status(),
                                )
                            proc.stdin.write(str(pasted).strip() + "\n")
                            proc.stdin.flush()
                        elif log_callback is not None:
                            log_callback(
                                "Enter the authorization code from your browser and press Enter:"
                            )
                rc = int(proc.wait(timeout=300))
            finally:
                if proc.poll() is None:
                    proc.terminate()
            output_text = "".join(lines).strip()
            if rc != 0:
                if "invalid_grant" in output_text.lower():
                    raise ColabAuthRequiredError(
                        "Google rejected the authorization code (`invalid_grant`). "
                        "Make sure you approve the exact URL printed by this run (each run generates a unique PKCE verifier) "
                        "and paste the full code promptly. If `gcloud` is installed, you can also use "
                        "`annolid-run colab auth login --method adc` for automatic browser sign-in without copying codes.",
                        backend="cli",
                        auth_status=self.get_auth_status(),
                    )
                raise ColabAuthRequiredError(
                    f"Colab CLI authentication failed (exit {rc}): {output_text}",
                    backend="cli",
                    auth_status=self.get_auth_status(),
                )
            self._save_auth_prefs({"colab_cli_auth_mode": cli_mode})
            return {
                "ok": True,
                "method": "colab-cli",
                "cli_auth_mode": cli_mode,
                "output": output_text,
                "status": self.get_auth_status(),
            }

        if selected in {"oauth2", "drive"}:
            if not GoogleOAuthHelper.is_oauth_available():
                raise ImportError(
                    "Google OAuth Python packages are not installed. "
                    'Install them with `pip install "annolid[annolid_bot]"` '
                    "(or `pip install google-api-python-client google-auth-oauthlib`)."
                )
            scopes = (
                list(GoogleOAuthHelper.COLAB_AND_DRIVE_SCOPES)
                if selected == "oauth2" and include_drive
                else (
                    list(GoogleOAuthHelper.COLAB_SCOPES)
                    if selected == "oauth2"
                    else [GoogleOAuthHelper.DRIVE_SCOPE]
                )
            )
            GoogleOAuthHelper.login_interactive(
                credentials_file=self.credentials_file,
                token_file=self.token_file,
                scopes=scopes,
                merge_existing_scopes=True,
                open_browser=open_browser,
            )
            return {
                "ok": True,
                "method": selected,
                "token_file": str(Path(self.token_file).expanduser()),
                "scopes": scopes,
                "status": self.get_auth_status(),
            }

        if selected == "adc":
            gcloud_bin = self.find_gcloud_cli()
            if not gcloud_bin:
                raise FileNotFoundError(
                    "`gcloud` CLI not found on PATH. Install Google Cloud SDK or "
                    "use `--method oauth2` / `--method colab-cli`."
                )
            scopes_csv = ",".join(GoogleOAuthHelper.ADC_COLAB_SCOPES)
            cmd = [
                gcloud_bin,
                "auth",
                "application-default",
                "login",
                f"--scopes={scopes_csv}",
            ]
            if not open_browser:
                cmd.append("--no-launch-browser")
            if log_callback is not None:
                log_callback(
                    "Launching Google Cloud ADC browser sign-in (`gcloud auth application-default login`)..."
                )
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            lines_adc: list[str] = []
            try:
                for line in proc.stdout:
                    lines_adc.append(line)
                    if log_callback is not None:
                        log_callback(line.rstrip("\n"))
                rc = int(proc.wait(timeout=300))
            finally:
                if proc.poll() is None:
                    proc.terminate()
            detail = "".join(lines_adc).strip()
            if rc != 0:
                hint = ""
                if "scope is required but not consented" in detail.lower():
                    hint = (
                        " On Google's consent screen, make sure all requested permission "
                        "checkboxes (including Google Cloud and Google Colab) are checked "
                        "before clicking Continue."
                    )
                raise ColabAuthRequiredError(
                    f"`gcloud auth application-default login` failed (exit {rc}): {detail}{hint}",
                    backend="api",
                    auth_status=self.get_auth_status(),
                )
            self._save_auth_prefs({"colab_cli_auth_mode": "adc"})
            return {
                "ok": True,
                "method": "adc",
                "cli_auth_mode": "adc",
                "scopes": list(GoogleOAuthHelper.ADC_COLAB_SCOPES),
                "status": self.get_auth_status(),
            }

        raise ValueError(
            f"Unsupported auth method {method!r}. "
            "Use one of: auto, personal, colab-cli, oauth2, drive, adc."
        )

    def _colab_cli_python(self) -> Optional[str]:
        cli = self.find_colab_cli()
        if not cli:
            return None
        resolved = Path(cli).resolve()
        for candidate in (resolved.parent / "python", resolved.parent / "python3"):
            if candidate.is_file():
                return str(candidate)
        return None

    def get_account_info(self) -> dict[str, Any]:
        """Inspect active signed-in personal Google account email and Colab usage.

        Read-only and strictly non-interactive (never blocks on an OAuth stdin prompt).
        """
        info: dict[str, Any] = {
            "signed_in": False,
            "email": "",
            "provider": "",
            "expires_in": "",
            "compute_balance": "",
            "usage_rate": "",
            "active_assignments": None,
            "detail": "Not signed in to a Google Colab account.",
        }
        cli_path = self.find_colab_cli()
        if cli_path:
            mode = self.resolve_colab_cli_auth_mode()
            # Never invoke `colab --auth=oauth2 whoami` unless a token already exists
            # or stdin is closed so it fails fast instead of waiting for user input.
            try:
                proc = subprocess.run(
                    [cli_path, f"--auth={mode}", "whoami"],
                    stdin=subprocess.DEVNULL,
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                out = (proc.stdout or "") + "\n" + (proc.stderr or "")
                if proc.returncode == 0 and "Email:" in out:
                    email = ""
                    provider = mode
                    expires_in = ""
                    for raw_line in out.splitlines():
                        line = raw_line.strip()
                        if line.startswith("Email:"):
                            email = line.split(":", 1)[1].strip()
                        elif line.startswith("Auth provider:"):
                            provider = line.split(":", 1)[1].strip() or mode
                        elif line.startswith("Expires in:"):
                            expires_in = line.split(":", 1)[1].strip()
                    if email and email.lower() != "none":
                        info["signed_in"] = True
                        info["email"] = email
                        info["provider"] = f"colab-cli ({provider})"
                        info["expires_in"] = expires_in
                        info["detail"] = (
                            f"Signed in as {email} via Colab CLI ({provider})"
                        )
                        # Fetch compute balance & active assignments (best effort)
                        try:
                            u_proc = subprocess.run(
                                [cli_path, f"--auth={mode}", "usage"],
                                stdin=subprocess.DEVNULL,
                                capture_output=True,
                                text=True,
                                timeout=15,
                            )
                            if u_proc.returncode == 0:
                                for u_line in (u_proc.stdout or "").splitlines():
                                    s = u_line.strip()
                                    if s.startswith("Current balance:"):
                                        info["compute_balance"] = s.split(":", 1)[
                                            1
                                        ].strip()
                                    elif s.startswith("Usage rate:"):
                                        info["usage_rate"] = s.split(":", 1)[1].strip()
                                    elif s.startswith("Active assignments:"):
                                        raw_n = s.split(":", 1)[1].strip()
                                        try:
                                            info["active_assignments"] = int(raw_n)
                                        except ValueError:
                                            pass
                        except Exception:
                            pass
                        return info
            except Exception as exc:
                info["detail"] = f"Could not query Colab CLI account ({exc})"

        # Fallback: report shared Annolid OAuth / ADC readiness without interactive prompts
        st = self.get_auth_status()
        if st["oauth_colab"]["ready"]:
            info["signed_in"] = True
            info["provider"] = "oauth2"
            info["detail"] = "Signed in via Annolid Google OAuth2 (Colab API)"
        elif st["oauth_drive"]["ready"]:
            info["signed_in"] = True
            info["provider"] = "drive"
            info["detail"] = "Signed in via Google Drive OAuth (Notebook mode)"
        elif st["adc"]["ready"]:
            info["signed_in"] = False
            info["provider"] = "adc"
            info["detail"] = (
                "Google Cloud ADC file detected (click Test Live Connection to verify)"
            )
        return info

    def logout(
        self,
        *,
        include_colab_cli: bool = False,
        clear_adc: bool = False,
    ) -> dict[str, Any]:
        removed = GoogleOAuthHelper.clear_token(self.token_file)
        cleared_cli_keyring = False
        cleared_adc = False
        if include_colab_cli:
            # Clear saved Colab CLI keyring token + legacy plaintext token + mode pref
            py_bin = self._colab_cli_python()
            if py_bin:
                try:
                    proc = subprocess.run(
                        [
                            py_bin,
                            "-c",
                            "import keyring\n"
                            "try:\n"
                            "    keyring.delete_password('google-colab-cli', 'oauth_token')\n"
                            "    print('DELETED')\n"
                            "except Exception:\n"
                            "    pass\n",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    if "DELETED" in (proc.stdout or ""):
                        cleared_cli_keyring = True
                except Exception:
                    pass
            legacy_tok = Path.home() / ".config" / "colab-cli" / "token.json"
            try:
                if legacy_tok.is_file():
                    legacy_tok.unlink()
                    cleared_cli_keyring = True
            except OSError:
                pass
            try:
                if self._prefs_file.is_file():
                    self._prefs_file.unlink()
            except OSError:
                pass
        if clear_adc:
            gcloud_bin = self.find_gcloud_cli()
            if gcloud_bin:
                try:
                    proc = subprocess.run(
                        [
                            gcloud_bin,
                            "auth",
                            "application-default",
                            "revoke",
                            "--quiet",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    cleared_adc = proc.returncode == 0
                except Exception:
                    cleared_adc = False
            adc_path = (
                GoogleOAuthHelper.default_adc_path()
                if hasattr(GoogleOAuthHelper, "default_adc_path")
                else (
                    Path.home()
                    / ".config"
                    / "gcloud"
                    / "application_default_credentials.json"
                )
            )
            try:
                if not cleared_adc and adc_path.is_file():
                    adc_path.unlink()
                    cleared_adc = True
            except OSError:
                pass
        return {
            "ok": True,
            "removed_token": removed,
            "cleared_colab_cli": cleared_cli_keyring,
            "cleared_adc": cleared_adc,
            "token_file": str(Path(self.token_file).expanduser()),
            "status": self.get_auth_status(),
        }

    def get_colab_api_access_token(
        self,
        *,
        auth_mode: str = "auto",
        allow_browser_auth: bool = False,
    ) -> str:
        """Obtain a fresh Bearer access token for `colaboratory.googleapis.com/v1beta`."""
        mode = str(auth_mode or "auto").strip().lower()
        errors: list[str] = []

        if mode in {"auto", "oauth2"}:
            oauth_status = GoogleOAuthHelper.inspect_auth_status(
                credentials_file=self.credentials_file,
                token_file=self.token_file,
                required_scopes=GoogleOAuthHelper.COLAB_SCOPES,
            )
            if oauth_status["ready"] or (
                mode == "oauth2" and (self.allow_interactive_auth or allow_browser_auth)
            ):
                try:
                    creds = GoogleOAuthHelper.get_credentials(
                        credentials_file=self.credentials_file,
                        token_file=self.token_file,
                        scopes=list(GoogleOAuthHelper.COLAB_SCOPES),
                        allow_interactive_auth=bool(
                            self.allow_interactive_auth or allow_browser_auth
                        ),
                        service_label="Google Colab",
                        allow_browser_auth=bool(allow_browser_auth),
                        merge_existing_scopes=True,
                    )
                    token = str(getattr(creds, "token", "") or "").strip()
                    if token:
                        return token
                except Exception as exc:
                    errors.append(f"OAuth2: {exc}")
                    if mode == "oauth2":
                        raise

        if mode in {"auto", "adc"}:
            try:
                creds, _project = GoogleOAuthHelper.get_adc_credentials(
                    scopes=GoogleOAuthHelper.ADC_COLAB_SCOPES
                )
                token = str(getattr(creds, "token", "") or "").strip()
                if token:
                    return token
            except Exception as exc:
                errors.append(f"ADC: {exc}")
                if mode == "adc":
                    raise

        env_token = str(os.environ.get("ANNOLID_COLAB_ACCESS_TOKEN") or "").strip()
        if env_token:
            return env_token

        detail = " | ".join(errors) if errors else "No valid Colab credentials found."
        raise ColabAuthRequiredError(
            f"Unable to acquire Google Colab API access token ({detail}). "
            "Run `annolid-run colab auth login` to authenticate.",
            backend="api",
            auth_status=self.get_auth_status(),
        )

    def create_drive_tool(
        self,
        *,
        allow_interactive_auth: Optional[bool] = None,
        allowed_local_roots: Optional[list[str | Path]] = None,
    ) -> GoogleDriveTool:
        return GoogleDriveTool(
            credentials_file=self.credentials_file,
            token_file=self.token_file,
            allow_interactive_auth=(
                bool(allow_interactive_auth)
                if allow_interactive_auth is not None
                else self.allow_interactive_auth
            ),
            allowed_local_roots=allowed_local_roots,
        )
