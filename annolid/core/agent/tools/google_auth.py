from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any


class GoogleOAuthHelper:
    """Shared OAuth/token bootstrap for Google API tools."""

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
    def _read_token_scopes(token_path: Path) -> set[str]:
        try:
            payload = json.loads(token_path.read_text(encoding="utf-8"))
        except Exception:
            return set()
        raw = payload.get("scopes")
        if isinstance(raw, str):
            return {item.strip() for item in raw.split() if item.strip()}
        if isinstance(raw, list):
            return {str(item).strip() for item in raw if str(item).strip()}
        return set()

    @staticmethod
    def _missing_scopes(
        required_scopes: list[str], granted_scopes: set[str]
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
    def get_credentials(
        cls,
        *,
        credentials_file: str,
        token_file: str,
        scopes: list[str],
        allow_interactive_auth: bool,
        service_label: str = "Google API",
    ) -> Any:
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials
        from google_auth_oauthlib.flow import InstalledAppFlow

        token_path = Path(str(token_file).strip()).expanduser()
        credentials_path = Path(str(credentials_file).strip()).expanduser()

        cls.ensure_private_parent(token_path)
        if credentials_path.exists():
            cls.ensure_private_file(credentials_path)

        creds = None
        if token_path.exists():
            try:
                creds = Credentials.from_authorized_user_file(str(token_path), scopes)
            except Exception:
                creds = None
            token_scopes = cls._read_token_scopes(token_path)
            missing_scopes = cls._missing_scopes(scopes, token_scopes)
            if missing_scopes:
                if (
                    allow_interactive_auth
                    and credentials_path.exists()
                    and cls.interactive_auth_possible()
                ):
                    flow = InstalledAppFlow.from_client_secrets_file(
                        str(credentials_path), scopes
                    )
                    creds = flow.run_local_server(port=0)
                    token_path.write_text(creds.to_json(), encoding="utf-8")
                    cls.ensure_private_file(token_path)
                else:
                    raise RuntimeError(
                        f"{service_label} token is missing required permissions. "
                        f"Missing scopes: {', '.join(missing_scopes)}. "
                        "Enable `tools.googleAuth.allowInteractiveAuth=true` and "
                        "run the action from an interactive session to re-authorize."
                    )

        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                except Exception as exc:
                    if cls._looks_like_deleted_client_error(exc):
                        if (
                            allow_interactive_auth
                            and credentials_path.exists()
                            and cls.interactive_auth_possible()
                        ):
                            flow = InstalledAppFlow.from_client_secrets_file(
                                str(credentials_path), scopes
                            )
                            creds = flow.run_local_server(port=0)
                        else:
                            raise RuntimeError(
                                "The Google Drive service is unavailable due to a "
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
                            and cls.interactive_auth_possible()
                        ):
                            flow = InstalledAppFlow.from_client_secrets_file(
                                str(credentials_path), scopes
                            )
                            creds = flow.run_local_server(port=0)
                        else:
                            raise RuntimeError(
                                "Google authorization is no longer valid. "
                                "Please login again and save a new token by setting "
                                "`tools.googleAuth.allowInteractiveAuth=true` and "
                                "running a Google action from an interactive Annolid "
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
                if not cls.interactive_auth_possible():
                    raise RuntimeError(
                        "Google OAuth requires interactive auth, but no TTY is "
                        "available for local authorization."
                    )
                if not credentials_path.exists():
                    raise FileNotFoundError(
                        f"Google credentials file not found at {credentials_path}"
                    )
                flow = InstalledAppFlow.from_client_secrets_file(
                    str(credentials_path), scopes
                )
                creds = flow.run_local_server(port=0)

            token_path.write_text(creds.to_json(), encoding="utf-8")
            cls.ensure_private_file(token_path)

        return creds
