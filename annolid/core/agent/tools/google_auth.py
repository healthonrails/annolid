from __future__ import annotations

import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


class GoogleOAuthHelper:
    """Shared OAuth/ADC token bootstrap for Google API tools (Drive, Calendar, Colab)."""

    DRIVE_SCOPE = "https://www.googleapis.com/auth/drive"
    CALENDAR_SCOPE = "https://www.googleapis.com/auth/calendar"
    COLAB_SCOPE = "https://www.googleapis.com/auth/colaboratory"
    COLAB_READONLY_SCOPE = "https://www.googleapis.com/auth/colaboratory.readonly"
    USERINFO_EMAIL_SCOPE = "https://www.googleapis.com/auth/userinfo.email"
    OPENID_SCOPE = "openid"
    CLOUD_PLATFORM_SCOPE = "https://www.googleapis.com/auth/cloud-platform"

    COLAB_SCOPES: tuple[str, ...] = (
        COLAB_SCOPE,
        USERINFO_EMAIL_SCOPE,
        OPENID_SCOPE,
    )
    COLAB_AND_DRIVE_SCOPES: tuple[str, ...] = (
        COLAB_SCOPE,
        DRIVE_SCOPE,
        USERINFO_EMAIL_SCOPE,
        OPENID_SCOPE,
    )
    ADC_COLAB_SCOPES: tuple[str, ...] = (
        OPENID_SCOPE,
        CLOUD_PLATFORM_SCOPE,
        USERINFO_EMAIL_SCOPE,
        COLAB_SCOPE,
    )

    @staticmethod
    def ensure_private_parent(path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.parent.chmod(0o700)
        except OSError:
            pass

    @staticmethod
    def ensure_private_file(path: Path) -> None:
        try:
            path.chmod(0o600)
        except OSError:
            pass

    @staticmethod
    def is_oauth_available() -> bool:
        required = (
            "google.auth.transport.requests",
            "google.oauth2.credentials",
            "google_auth_oauthlib.flow",
        )
        for name in required:
            try:
                if importlib.util.find_spec(name) is None:
                    return False
            except (ImportError, ModuleNotFoundError):
                return False
        return True

    @staticmethod
    def is_adc_library_available() -> bool:
        required = (
            "google.auth",
            "google.auth.transport.requests",
        )
        for name in required:
            try:
                if importlib.util.find_spec(name) is None:
                    return False
            except (ImportError, ModuleNotFoundError):
                return False
        return True

    @staticmethod
    def interactive_auth_possible() -> bool:
        stdin = getattr(sys, "stdin", None)
        stdout = getattr(sys, "stdout", None)
        return bool(
            stdin is not None
            and stdout is not None
            and hasattr(stdin, "isatty")
            and hasattr(stdout, "isatty")
            and stdin.isatty()
            and stdout.isatty()
        )

    @staticmethod
    def merge_scopes(
        required_scopes: Sequence[str],
        existing_scopes: Sequence[str] | set[str] | None = None,
    ) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for source in (existing_scopes or (), required_scopes or ()):
            for raw in source:
                scope = str(raw or "").strip()
                if scope and scope not in seen:
                    seen.add(scope)
                    merged.append(scope)
        return merged

    @staticmethod
    def preflight(
        *,
        credentials_file: str,
        token_file: str,
        allow_interactive_auth: bool = False,
    ) -> tuple[bool, str]:
        token_path = Path(str(token_file or "").strip()).expanduser()
        credentials_path = Path(str(credentials_file or "").strip()).expanduser()
        if token_path.exists():
            return True, ""
        if credentials_path.exists():
            if bool(allow_interactive_auth):
                return True, ""
            return (
                False,
                "Google credentials exist but interactive auth is disabled and "
                "no cached token is available.",
            )
        return (False, "Google token and credentials files are both missing.")

    @staticmethod
    def _looks_like_deleted_client_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return any(
            token in text
            for token in (
                "deleted_client",
                "oauth client was deleted",
                "oauth client is disabled",
                "invalid_client",
            )
        )

    @staticmethod
    def _looks_like_reauth_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return any(
            token in text
            for token in (
                "invalid_grant",
                "token has been expired or revoked",
                "reauth",
                "revoked",
                "unauthorized_client",
            )
        )

    @staticmethod
    def _read_token_payload(token_path: Path) -> dict[str, Any]:
        try:
            payload = json.loads(token_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @classmethod
    def _read_token_scopes(cls, token_path: Path) -> set[str]:
        payload = cls._read_token_payload(token_path)
        raw = payload.get("scopes")
        if isinstance(raw, str):
            return {item.strip() for item in raw.split() if item.strip()}
        if isinstance(raw, list):
            return {str(item).strip() for item in raw if str(item).strip()}
        return set()

    @staticmethod
    def _missing_scopes(
        required_scopes: Sequence[str], granted_scopes: set[str]
    ) -> list[str]:
        required = [
            str(scope).strip()
            for scope in list(required_scopes or [])
            if str(scope).strip()
        ]
        if not required:
            return []
        if not granted_scopes:
            return required
        return [scope for scope in required if scope not in granted_scopes]

    @classmethod
    def _save_token(cls, token_path: Path, creds: Any) -> None:
        import tempfile

        cls.ensure_private_parent(token_path)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{token_path.name}.",
            suffix=".tmp",
            dir=str(token_path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(creds.to_json())
            cls.ensure_private_file(tmp_path)
            os.replace(str(tmp_path), str(token_path))
            cls.ensure_private_file(token_path)
        except Exception:
            try:
                if tmp_path.exists():
                    tmp_path.unlink()
            except OSError:
                pass
            raise

    @classmethod
    def _run_installed_app_flow(
        cls,
        credentials_path: Path,
        scopes: Sequence[str],
        *,
        open_browser: bool = True,
    ) -> Any:
        try:
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise RuntimeError(
                "Google OAuth dependencies are not installed "
                '(install `pip install "annolid[annolid_bot]"`).'
            ) from exc

        if not credentials_path.exists():
            raise FileNotFoundError(
                f"Google credentials file not found at {credentials_path}"
            )
        cls.ensure_private_file(credentials_path)
        flow = InstalledAppFlow.from_client_secrets_file(
            str(credentials_path), list(scopes)
        )
        return flow.run_local_server(port=0, open_browser=bool(open_browser))

    @classmethod
    def inspect_auth_status(
        cls,
        *,
        credentials_file: str,
        token_file: str,
        required_scopes: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        credentials_path = Path(str(credentials_file or "").strip()).expanduser()
        token_path = Path(str(token_file or "").strip()).expanduser()
        has_credentials = (
            bool(str(credentials_file or "").strip()) and credentials_path.is_file()
        )
        has_token = bool(str(token_file or "").strip()) and token_path.is_file()
        oauth_deps = cls.is_oauth_available()

        granted_scopes: list[str] = []
        missing_scopes: list[str] = []
        has_refresh_token = False
        expiry = ""
        token_expired = False
        token_parse_error = ""

        if has_token:
            payload = cls._read_token_payload(token_path)
            if not payload:
                token_parse_error = (
                    "Cached Google OAuth token file could not be parsed."
                )
            else:
                granted_set = cls._read_token_scopes(token_path)
                granted_scopes = sorted(granted_set)
                has_refresh_token = bool(
                    str(payload.get("refresh_token") or "").strip()
                )
                expiry = str(payload.get("expiry") or "").strip()
                if expiry:
                    try:
                        iso = expiry[:-1] + "+00:00" if expiry.endswith("Z") else expiry
                        exp_dt = datetime.fromisoformat(iso)
                        if exp_dt.tzinfo is None:
                            exp_dt = exp_dt.replace(tzinfo=timezone.utc)
                        token_expired = exp_dt <= datetime.now(timezone.utc)
                    except Exception:
                        token_expired = False
                if required_scopes:
                    missing_scopes = cls._missing_scopes(required_scopes, granted_set)
        elif required_scopes:
            missing_scopes = [str(s).strip() for s in required_scopes if str(s).strip()]

        ready = (
            oauth_deps
            and has_token
            and not token_parse_error
            and not missing_scopes
            and (not token_expired or has_refresh_token)
        )
        if not oauth_deps:
            reason = (
                "Google OAuth dependencies are not installed "
                '(install `pip install "annolid[annolid_bot]"`).'
            )
        elif token_parse_error:
            reason = token_parse_error
        elif not has_token and not has_credentials:
            reason = (
                "Google OAuth client credentials and cached token are both missing."
            )
        elif not has_token:
            reason = "OAuth client credentials found, but user has not signed in yet."
        elif missing_scopes:
            reason = "Cached token is missing required scopes: " + ", ".join(
                missing_scopes
            )
        elif token_expired and not has_refresh_token:
            reason = (
                "Cached OAuth token is expired and has no refresh token; sign in again."
            )
        else:
            reason = ""

        return {
            "ready": ready,
            "oauth_dependencies_installed": oauth_deps,
            "credentials_file": str(credentials_path)
            if str(credentials_file or "").strip()
            else "",
            "token_file": str(token_path) if str(token_file or "").strip() else "",
            "has_credentials_file": has_credentials,
            "has_token_file": has_token,
            "has_refresh_token": has_refresh_token,
            "token_expired": token_expired,
            "expiry": expiry,
            "granted_scopes": granted_scopes,
            "missing_scopes": missing_scopes,
            "reason": reason,
        }

    @classmethod
    def default_adc_path(cls) -> Path:
        env_path = str(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
        if env_path:
            return Path(env_path).expanduser()
        if os.name == "nt":
            appdata = str(os.environ.get("APPDATA") or "").strip()
            if appdata:
                return Path(appdata) / "gcloud" / "application_default_credentials.json"
        return (
            Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
        )

    @classmethod
    def inspect_adc_status(cls) -> dict[str, Any]:
        adc_path = cls.default_adc_path()
        exists = adc_path.is_file()
        lib_available = cls.is_adc_library_available()
        cred_type = ""
        quota_project = ""
        if exists:
            payload = cls._read_token_payload(adc_path)
            cred_type = str(payload.get("type") or "").strip()
            quota_project = str(payload.get("quota_project_id") or "").strip()
        return {
            "ready": bool(exists and lib_available),
            "library_available": lib_available,
            "adc_file": str(adc_path),
            "exists": exists,
            "credential_type": cred_type,
            "quota_project_id": quota_project,
            "setup_command": (
                "gcloud auth application-default login "
                "--scopes=" + ",".join(cls.ADC_COLAB_SCOPES)
            ),
        }

    @classmethod
    def get_adc_credentials(
        cls,
        *,
        scopes: Sequence[str] | None = None,
        quota_project_id: str | None = None,
    ) -> tuple[Any, str | None]:
        import contextlib
        import warnings
        from unittest.mock import patch as _patch

        import google.auth
        from google.auth.transport.requests import Request

        req_scopes = list(scopes) if scopes else list(cls.ADC_COLAB_SCOPES)
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(warnings.catch_warnings())
                warnings.filterwarnings(
                    "ignore",
                    message=r"Your application has authenticated using end user credentials.*",
                    category=UserWarning,
                )
                if quota_project_id is None:
                    try:
                        from google.auth import _cloud_sdk

                        # Avoid spawning `gcloud config get-value project` (2-13s latency)
                        # when caller only needs the ADC access token.
                        stack.enter_context(
                            _patch.object(
                                _cloud_sdk, "get_project_id", return_value=None
                            )
                        )
                    except Exception:
                        pass
                creds, project = google.auth.default(
                    scopes=req_scopes,
                    quota_project_id=quota_project_id,
                )
            if not getattr(creds, "valid", False):
                creds.refresh(Request())
            return creds, project
        except Exception as exc:
            setup_cmd = "gcloud auth application-default login --scopes=" + ",".join(
                req_scopes
            )
            raise RuntimeError(
                f"Google Application Default Credentials (ADC) failed: {exc}. "
                f"Run `{setup_cmd}` and retry."
            ) from exc

    @classmethod
    def login_interactive(
        cls,
        *,
        credentials_file: str,
        token_file: str,
        scopes: Sequence[str],
        merge_existing_scopes: bool = True,
        open_browser: bool = True,
    ) -> Any:
        token_path = Path(str(token_file).strip()).expanduser()
        credentials_path = Path(str(credentials_file).strip()).expanduser()
        existing_scopes = (
            cls._read_token_scopes(token_path)
            if merge_existing_scopes and token_path.exists()
            else set()
        )
        effective_scopes = cls.merge_scopes(scopes, existing_scopes)
        creds = cls._run_installed_app_flow(
            credentials_path,
            effective_scopes,
            open_browser=open_browser,
        )
        cls._save_token(token_path, creds)
        return creds

    @classmethod
    def clear_token(cls, token_file: str) -> bool:
        raw = str(token_file or "").strip()
        if not raw:
            return False
        token_path = Path(raw).expanduser()
        if not token_path.exists():
            return False
        token_path.unlink()
        return True

    @classmethod
    def get_credentials(
        cls,
        *,
        credentials_file: str,
        token_file: str,
        scopes: list[str],
        allow_interactive_auth: bool,
        service_label: str = "Google API",
        allow_browser_auth: bool = False,
        merge_existing_scopes: bool = True,
    ) -> Any:
        token_path = Path(str(token_file).strip()).expanduser()
        credentials_path = Path(str(credentials_file).strip()).expanduser()
        can_prompt = bool(allow_browser_auth or cls.interactive_auth_possible())

        cls.ensure_private_parent(token_path)
        if credentials_path.exists():
            cls.ensure_private_file(credentials_path)

        creds = None
        existing_scopes: set[str] = set()
        if token_path.exists():
            try:
                from google.oauth2.credentials import Credentials
            except ImportError as exc:
                raise RuntimeError(
                    "Google OAuth dependencies are not installed "
                    '(install `pip install "annolid[annolid_bot]"`).'
                ) from exc
            existing_scopes = cls._read_token_scopes(token_path)
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), scopes)
            except Exception:
                creds = None
            missing_scopes = cls._missing_scopes(scopes, existing_scopes)
            if missing_scopes:
                if allow_interactive_auth and credentials_path.exists() and can_prompt:
                    effective_scopes = (
                        cls.merge_scopes(scopes, existing_scopes)
                        if merge_existing_scopes
                        else list(scopes)
                    )
                    creds = cls._run_installed_app_flow(
                        credentials_path, effective_scopes
                    )
                    cls._save_token(token_path, creds)
                else:
                    raise RuntimeError(
                        f"{service_label} token is missing required permissions. "
                        f"Missing scopes: {', '.join(missing_scopes)}. "
                        "Enable `tools.googleAuth.allowInteractiveAuth=true` and "
                        "run the action from an interactive session (or use "
                        "`annolid-run colab auth login`) to re-authorize."
                    )

        if not creds or not creds.valid:
            effective_scopes = (
                cls.merge_scopes(scopes, existing_scopes)
                if merge_existing_scopes
                else list(scopes)
            )
            if creds and creds.expired and creds.refresh_token:
                try:
                    from google.auth.transport.requests import Request
                except ImportError as exc:
                    raise RuntimeError(
                        "Google OAuth dependencies are not installed "
                        '(install `pip install "annolid[annolid_bot]"`).'
                    ) from exc
                try:
                    creds.refresh(Request())
                except Exception as exc:
                    if cls._looks_like_deleted_client_error(exc):
                        if (
                            allow_interactive_auth
                            and credentials_path.exists()
                            and can_prompt
                        ):
                            creds = cls._run_installed_app_flow(
                                credentials_path, effective_scopes
                            )
                        else:
                            raise RuntimeError(
                                f"The {service_label} service is unavailable due to a "
                                "deleted OAuth client. Please login again and save "
                                "new credentials: update "
                                "`tools.googleAuth.credentialsFile`, set "
                                "`tools.googleAuth.allowInteractiveAuth=true`, and "
                                "run a Google action from an interactive Annolid "
                                "session."
                            ) from exc
                    elif cls._looks_like_reauth_error(exc):
                        if (
                            allow_interactive_auth
                            and credentials_path.exists()
                            and can_prompt
                        ):
                            creds = cls._run_installed_app_flow(
                                credentials_path, effective_scopes
                            )
                        else:
                            raise RuntimeError(
                                "Google authorization is no longer valid. "
                                "Sign in again via `annolid-run colab auth login` "
                                "(or Google Colab Studio), or set "
                                "`tools.googleAuth.allowInteractiveAuth=true` and "
                                "run a Google action from an interactive Annolid "
                                "session."
                            ) from exc
                    else:
                        raise RuntimeError(
                            "Google token refresh failed. Please login again and save "
                            "a fresh token."
                        ) from exc
            else:
                if not allow_interactive_auth:
                    raise RuntimeError(
                        "Google OAuth token is unavailable. Enable "
                        "`allow_interactive_auth` to authorize once from an "
                        "interactive session, or provide a valid cached token file."
                    )
                if not can_prompt:
                    raise RuntimeError(
                        "Google OAuth requires interactive auth, but no TTY is "
                        "available for local authorization."
                    )
                if not credentials_path.exists():
                    raise FileNotFoundError(
                        f"Google credentials file not found at {credentials_path}"
                    )
                creds = cls._run_installed_app_flow(credentials_path, effective_scopes)

            cls._save_token(token_path, creds)

        return creds
