from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path
from typing import Any

from .function_base import FunctionTool
from .google_auth import GoogleOAuthHelper


class GoogleDriveTool(FunctionTool):
    """Google Drive tool using shared Google OAuth credentials/token."""

    _SCOPES = ["https://www.googleapis.com/auth/drive"]
    _ACTIONS = {"list_files", "get_file", "create_folder", "delete_file"}

    def __init__(
        self,
        *,
        credentials_file: str = "~/.annolid/agent/google_oauth_credentials.json",
        token_file: str = "~/.annolid/agent/google_oauth_token.json",
        allow_interactive_auth: bool = False,
    ) -> None:
        self._credentials_file = str(credentials_file or "").strip()
        self._token_file = str(token_file or "").strip()
        self._allow_interactive_auth = bool(allow_interactive_auth)
        self._service_cache: Any = None
        self._service_cache_key: tuple[Any, ...] | None = None

    @property
    def name(self) -> str:
        return "google_drive"

    @property
    def description(self) -> str:
        return (
            "Manage Google Drive files and folders (list/get/create_folder/delete) "
            "using Google API OAuth credentials."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": sorted(self._ACTIONS)},
                "query": {
                    "type": "string",
                    "description": "Drive query for list_files, e.g. trashed=false",
                },
                "max_results": {"type": "integer", "minimum": 1, "maximum": 100},
                "file_id": {"type": "string"},
                "folder_name": {"type": "string"},
                "parent_id": {"type": "string"},
                "supports_all_drives": {"type": "boolean"},
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        action_name = str(action or "").strip().lower()
        if action_name not in self._ACTIONS:
            return "Error: Unsupported action. Use one of: list_files, get_file, create_folder, delete_file."

        if (
            action_name in {"get_file", "delete_file"}
            and not str(kwargs.get("file_id", "") or "").strip()
        ):
            return f"Error: {action_name} requires `file_id`."
        if (
            action_name == "create_folder"
            and not str(kwargs.get("folder_name", "") or "").strip()
        ):
            return "Error: create_folder requires `folder_name`."

        try:
            service = await asyncio.to_thread(self._get_service)
        except ImportError as exc:
            return (
                "Error: Google Drive dependencies are not installed. "
                'Install optional extras with `pip install "annolid[google_calendar]"`. '
                f"Details: {exc}"
            )
        except FileNotFoundError as exc:
            return f"Error: {exc}"
        except PermissionError as exc:
            return f"Error: Google Drive file permissions prevent access: {exc}"
        except RuntimeError as exc:
            return f"Error: {exc}"
        except Exception as exc:
            return f"Error: Failed to initialize Google Drive service: {exc}"

        try:
            if action_name == "list_files":
                return await asyncio.to_thread(
                    self._list_files,
                    service,
                    str(kwargs.get("query", "") or "").strip(),
                    int(kwargs.get("max_results", 20) or 20),
                )
            if action_name == "get_file":
                return await asyncio.to_thread(
                    self._get_file,
                    service,
                    str(kwargs.get("file_id", "") or "").strip(),
                    bool(kwargs.get("supports_all_drives", True)),
                )
            if action_name == "create_folder":
                return await asyncio.to_thread(
                    self._create_folder,
                    service,
                    str(kwargs.get("folder_name", "") or "").strip(),
                    str(kwargs.get("parent_id", "") or "").strip(),
                    bool(kwargs.get("supports_all_drives", True)),
                )
            if action_name == "delete_file":
                return await asyncio.to_thread(
                    self._delete_file,
                    service,
                    str(kwargs.get("file_id", "") or "").strip(),
                    bool(kwargs.get("supports_all_drives", True)),
                )
        except Exception as exc:
            message = str(exc or "")
            if "insufficientpermissions" in message.lower() or "403" in message:
                return (
                    "Error: Google Drive returned 403 insufficient permissions. "
                    "Re-authorize Google OAuth with drive scope by setting "
                    "`tools.googleAuth.allowInteractiveAuth=true` and running a "
                    "drive action from an interactive Annolid session."
                )
            return f"Error: Google Drive request failed: {exc}"
        return "Error: Unsupported action."

    @classmethod
    def is_available(cls) -> bool:
        required_modules = (
            "google.auth.transport.requests",
            "google.oauth2.credentials",
            "google_auth_oauthlib.flow",
            "googleapiclient.discovery",
        )
        for name in required_modules:
            try:
                if importlib.util.find_spec(name) is None:
                    return False
            except (ImportError, ModuleNotFoundError):
                return False
        return True

    @classmethod
    def preflight(
        cls,
        *,
        credentials_file: str,
        token_file: str,
        allow_interactive_auth: bool = False,
    ) -> tuple[bool, str]:
        return GoogleOAuthHelper.preflight(
            credentials_file=credentials_file,
            token_file=token_file,
            allow_interactive_auth=allow_interactive_auth,
        )

    def _resolve_credentials_path(self) -> Path:
        return Path(self._credentials_file).expanduser()

    def _resolve_token_path(self) -> Path:
        return Path(self._token_file).expanduser()

    def _service_files_key(self) -> tuple[Any, ...]:
        token_path = self._resolve_token_path()
        credentials_path = self._resolve_credentials_path()
        return (
            self._path_version(token_path),
            self._path_version(credentials_path),
            self._allow_interactive_auth,
        )

    @staticmethod
    def _path_version(path: Path) -> tuple[bool, int | None]:
        if not path.exists():
            return (False, None)
        try:
            return (True, path.stat().st_mtime_ns)
        except OSError:
            return (True, None)

    def _get_service(self) -> Any:
        cache_key = self._service_files_key()
        if self._service_cache is not None and self._service_cache_key == cache_key:
            return self._service_cache

        from googleapiclient.discovery import build

        creds = GoogleOAuthHelper.get_credentials(
            credentials_file=self._credentials_file,
            token_file=self._token_file,
            scopes=self._SCOPES,
            allow_interactive_auth=self._allow_interactive_auth,
            service_label="Google Drive",
        )
        self._service_cache = build(
            "drive", "v3", credentials=creds, cache_discovery=False
        )
        self._service_cache_key = cache_key
        return self._service_cache

    def _list_files(self, service: Any, query: str, max_results: int) -> str:
        payload = (
            service.files()
            .list(
                q=query or None,
                pageSize=max(1, min(100, int(max_results))),
                fields=(
                    "files(id,name,mimeType,modifiedTime,owners(displayName),webViewLink),"
                    "nextPageToken"
                ),
                includeItemsFromAllDrives=True,
                supportsAllDrives=True,
            )
            .execute()
        )
        return json.dumps(payload, ensure_ascii=True)

    def _get_file(self, service: Any, file_id: str, supports_all_drives: bool) -> str:
        payload = (
            service.files()
            .get(
                fileId=file_id,
                fields=(
                    "id,name,mimeType,description,createdTime,modifiedTime,"
                    "owners(displayName,emailAddress),parents,webViewLink,size"
                ),
                supportsAllDrives=bool(supports_all_drives),
            )
            .execute()
        )
        return json.dumps(payload, ensure_ascii=True)

    def _create_folder(
        self,
        service: Any,
        folder_name: str,
        parent_id: str,
        supports_all_drives: bool,
    ) -> str:
        body: dict[str, Any] = {
            "name": folder_name,
            "mimeType": "application/vnd.google-apps.folder",
        }
        if parent_id:
            body["parents"] = [parent_id]
        payload = (
            service.files()
            .create(
                body=body,
                fields="id,name,webViewLink",
                supportsAllDrives=bool(supports_all_drives),
            )
            .execute()
        )
        return json.dumps({"created": payload}, ensure_ascii=True)

    def _delete_file(
        self, service: Any, file_id: str, supports_all_drives: bool
    ) -> str:
        service.files().delete(
            fileId=file_id,
            supportsAllDrives=bool(supports_all_drives),
        ).execute()
        return json.dumps({"deleted_file_id": file_id}, ensure_ascii=True)


__all__ = ["GoogleDriveTool"]
