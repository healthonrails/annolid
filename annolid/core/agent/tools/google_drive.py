from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .function_base import FunctionTool
from .google_auth import GoogleOAuthHelper


class GoogleDriveTool(FunctionTool):
    """Google Drive tool using shared Google OAuth credentials/token."""

    _SCOPES = ["https://www.googleapis.com/auth/drive"]
    _ACTIONS = {
        "list_files",
        "get_file",
        "create_folder",
        "delete_file",
        "upload_file",
        "upload_saved_videos",
        "upload_realtime_videos",
    }
    _VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}

    def __init__(
        self,
        *,
        credentials_file: str = "~/.annolid/agent/google_oauth_credentials.json",
        token_file: str = "~/.annolid/agent/google_oauth_token.json",
        allow_interactive_auth: bool = False,
        allowed_local_roots: list[str | Path] | None = None,
    ) -> None:
        self._credentials_file = str(credentials_file or "").strip()
        self._token_file = str(token_file or "").strip()
        self._allow_interactive_auth = bool(allow_interactive_auth)
        self._allowed_local_roots = [
            Path(root).expanduser().resolve()
            for root in (allowed_local_roots or [])
            if str(root or "").strip()
        ]
        self._service_cache: Any = None
        self._service_cache_key: tuple[Any, ...] | None = None

    @property
    def name(self) -> str:
        return "google_drive"

    @property
    def description(self) -> str:
        return (
            "Manage Google Drive files and folders (list/get/create/delete/upload) "
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
                "local_path": {
                    "type": "string",
                    "description": "Local file path for upload_file.",
                },
                "source_dir": {
                    "type": "string",
                    "description": "Local directory for batch upload actions.",
                },
                "remote_folder_id": {
                    "type": "string",
                    "description": "Drive folder id where uploads should be placed.",
                },
                "remote_folder_path": {
                    "type": "string",
                    "description": "Drive folder path to create/use, e.g. annolid/realtime_detect.",
                },
                "chunk_size_mb": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 64,
                    "description": "Resumable upload chunk size in MB.",
                },
                "max_files": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2000,
                    "description": "Max files to upload in batch actions.",
                },
                "modified_within_hours": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 720,
                    "description": "Only upload files modified within this many hours.",
                },
                "skip_if_exists": {
                    "type": "boolean",
                    "description": "Skip upload if same file name+size already exists in target folder.",
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        action_name = str(action or "").strip().lower()
        if action_name not in self._ACTIONS:
            return (
                "Error: Unsupported action. Use one of: "
                "list_files, get_file, create_folder, delete_file, "
                "upload_file, upload_saved_videos, upload_realtime_videos."
            )

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
        if (
            action_name == "upload_file"
            and not str(kwargs.get("local_path", "") or "").strip()
        ):
            return "Error: upload_file requires `local_path`."

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
            if action_name == "upload_file":
                return await asyncio.to_thread(
                    self._upload_file_action,
                    service,
                    local_path=str(kwargs.get("local_path", "") or "").strip(),
                    remote_folder_id=str(
                        kwargs.get("remote_folder_id", "") or ""
                    ).strip(),
                    remote_folder_path=str(
                        kwargs.get("remote_folder_path", "") or ""
                    ).strip(),
                    supports_all_drives=bool(kwargs.get("supports_all_drives", True)),
                    chunk_size_mb=int(kwargs.get("chunk_size_mb", 8) or 8),
                    skip_if_exists=bool(kwargs.get("skip_if_exists", True)),
                )
            if action_name in {"upload_saved_videos", "upload_realtime_videos"}:
                default_dir = (
                    "~/.annolid/realtime"
                    if action_name == "upload_realtime_videos"
                    else ""
                )
                return await asyncio.to_thread(
                    self._upload_video_batch_action,
                    service,
                    action_name=action_name,
                    source_dir=str(kwargs.get("source_dir", "") or default_dir).strip(),
                    remote_folder_id=str(
                        kwargs.get("remote_folder_id", "") or ""
                    ).strip(),
                    remote_folder_path=str(
                        kwargs.get("remote_folder_path", "") or ""
                    ).strip(),
                    supports_all_drives=bool(kwargs.get("supports_all_drives", True)),
                    chunk_size_mb=int(kwargs.get("chunk_size_mb", 8) or 8),
                    max_files=int(kwargs.get("max_files", 100) or 100),
                    modified_within_hours=int(
                        kwargs.get("modified_within_hours", 72) or 72
                    ),
                    skip_if_exists=bool(kwargs.get("skip_if_exists", True)),
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

    def _resolve_local_path(self, raw_path: str) -> Path:
        path = Path(str(raw_path or "").strip()).expanduser()
        resolved = path.resolve()
        if self._allowed_local_roots:
            allowed = any(
                resolved == root or root in resolved.parents
                for root in self._allowed_local_roots
            )
            if not allowed:
                raise PermissionError(
                    f"Local path is outside allowed roots: {resolved}"
                )
        return resolved

    def _ensure_drive_folder_path(
        self,
        service: Any,
        folder_path: str,
        *,
        supports_all_drives: bool,
    ) -> str:
        cleaned = str(folder_path or "").strip().strip("/")
        if not cleaned:
            return ""
        parent_id = ""
        for part in [p.strip() for p in cleaned.split("/") if p.strip()]:
            escaped = part.replace("'", "\\'")
            parent_clause = (
                f" and '{parent_id}' in parents"
                if parent_id
                else " and 'root' in parents"
            )
            q = (
                "mimeType='application/vnd.google-apps.folder' "
                f"and name='{escaped}' and trashed=false{parent_clause}"
            )
            res = (
                service.files()
                .list(
                    q=q,
                    pageSize=1,
                    fields="files(id,name)",
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=bool(supports_all_drives),
                )
                .execute()
            )
            files = list(res.get("files") or [])
            if files:
                parent_id = str(files[0].get("id") or "")
                continue
            body: dict[str, Any] = {
                "name": part,
                "mimeType": "application/vnd.google-apps.folder",
            }
            if parent_id:
                body["parents"] = [parent_id]
            created = (
                service.files()
                .create(
                    body=body,
                    fields="id,name",
                    supportsAllDrives=bool(supports_all_drives),
                )
                .execute()
            )
            parent_id = str(created.get("id") or "")
        return parent_id

    def _find_existing_file_by_name_and_size(
        self,
        service: Any,
        *,
        file_name: str,
        size_bytes: int,
        folder_id: str,
        supports_all_drives: bool,
    ) -> dict[str, Any] | None:
        escaped_name = str(file_name).replace("'", "\\'")
        parent_clause = (
            f" and '{folder_id}' in parents" if folder_id else " and 'root' in parents"
        )
        q = f"name='{escaped_name}' and trashed=false{parent_clause}"
        res = (
            service.files()
            .list(
                q=q,
                pageSize=10,
                fields="files(id,name,size,modifiedTime,webViewLink)",
                includeItemsFromAllDrives=True,
                supportsAllDrives=bool(supports_all_drives),
            )
            .execute()
        )
        for row in list(res.get("files") or []):
            try:
                if int(row.get("size") or -1) == int(size_bytes):
                    return row
            except Exception:
                continue
        return None

    def _upload_file_resumable(
        self,
        service: Any,
        *,
        local_path: Path,
        folder_id: str,
        supports_all_drives: bool,
        chunk_size_mb: int,
    ) -> dict[str, Any]:
        from googleapiclient.errors import HttpError
        from googleapiclient.http import MediaFileUpload

        chunk_size = max(1, min(64, int(chunk_size_mb))) * 1024 * 1024
        media = MediaFileUpload(
            str(local_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=chunk_size,
        )
        body: dict[str, Any] = {"name": local_path.name}
        if folder_id:
            body["parents"] = [folder_id]
        request = service.files().create(
            body=body,
            media_body=media,
            fields="id,name,size,md5Checksum,webViewLink,modifiedTime",
            supportsAllDrives=bool(supports_all_drives),
        )
        retries = 0
        result = None
        while result is None:
            try:
                _status, result = request.next_chunk()
            except HttpError as exc:
                status = int(getattr(getattr(exc, "resp", None), "status", 0) or 0)
                if status in {429, 500, 502, 503, 504} and retries < 5:
                    retries += 1
                    time.sleep(min(30, 2**retries))
                    continue
                raise
        return dict(result or {})

    def _is_realtime_detect_video(self, path: Path) -> bool:
        parts = [p.lower() for p in path.parts]
        stem = path.stem.lower()
        return any(
            token in stem
            for token in ("realtime", "detect", "detection", "inference", "live")
        ) or any(
            token in part
            for part in parts
            for token in ("realtime", "detect", "detection", "live")
        )

    def _scan_video_files(
        self,
        source_dir: Path,
        *,
        realtime_only: bool,
        modified_within_hours: int,
        max_files: int,
    ) -> list[Path]:
        cutoff = datetime.now(timezone.utc).timestamp() - (
            max(1, int(modified_within_hours)) * 3600
        )
        found: list[Path] = []
        for root, _dirs, files in os.walk(source_dir):
            for name in files:
                path = Path(root) / name
                if path.suffix.lower() not in self._VIDEO_EXTENSIONS:
                    continue
                try:
                    stat = path.stat()
                except OSError:
                    continue
                if stat.st_mtime < cutoff:
                    continue
                is_realtime = self._is_realtime_detect_video(path)
                if realtime_only and not is_realtime:
                    continue
                if (not realtime_only) and is_realtime:
                    continue
                found.append(path)
        found.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return found[: max(1, int(max_files))]

    def _upload_file_action(
        self,
        service: Any,
        *,
        local_path: str,
        remote_folder_id: str,
        remote_folder_path: str,
        supports_all_drives: bool,
        chunk_size_mb: int,
        skip_if_exists: bool,
    ) -> str:
        file_path = self._resolve_local_path(local_path)
        if not file_path.exists() or not file_path.is_file():
            return f"Error: local file not found: {file_path}"
        folder_id = str(remote_folder_id or "").strip()
        if remote_folder_path and not folder_id:
            folder_id = self._ensure_drive_folder_path(
                service,
                remote_folder_path,
                supports_all_drives=supports_all_drives,
            )
        size_bytes = int(file_path.stat().st_size)
        if skip_if_exists:
            existing = self._find_existing_file_by_name_and_size(
                service,
                file_name=file_path.name,
                size_bytes=size_bytes,
                folder_id=folder_id,
                supports_all_drives=supports_all_drives,
            )
            if existing is not None:
                return json.dumps(
                    {
                        "status": "skipped_existing",
                        "file": existing,
                        "local_path": str(file_path),
                    },
                    ensure_ascii=True,
                )
        uploaded = self._upload_file_resumable(
            service,
            local_path=file_path,
            folder_id=folder_id,
            supports_all_drives=supports_all_drives,
            chunk_size_mb=chunk_size_mb,
        )
        return json.dumps(
            {
                "status": "uploaded",
                "file": uploaded,
                "local_path": str(file_path),
            },
            ensure_ascii=True,
        )

    def _upload_video_batch_action(
        self,
        service: Any,
        *,
        action_name: str,
        source_dir: str,
        remote_folder_id: str,
        remote_folder_path: str,
        supports_all_drives: bool,
        chunk_size_mb: int,
        max_files: int,
        modified_within_hours: int,
        skip_if_exists: bool,
    ) -> str:
        if not source_dir:
            return "Error: source_dir is required for batch upload actions."
        root = self._resolve_local_path(source_dir)
        if not root.exists() or not root.is_dir():
            return f"Error: source_dir not found: {root}"

        folder_id = str(remote_folder_id or "").strip()
        if remote_folder_path and not folder_id:
            folder_id = self._ensure_drive_folder_path(
                service,
                remote_folder_path,
                supports_all_drives=supports_all_drives,
            )

        realtime_only = action_name == "upload_realtime_videos"
        candidates = self._scan_video_files(
            root,
            realtime_only=realtime_only,
            modified_within_hours=modified_within_hours,
            max_files=max_files,
        )

        uploaded: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for file_path in candidates:
            try:
                size_bytes = int(file_path.stat().st_size)
                if skip_if_exists:
                    existing = self._find_existing_file_by_name_and_size(
                        service,
                        file_name=file_path.name,
                        size_bytes=size_bytes,
                        folder_id=folder_id,
                        supports_all_drives=supports_all_drives,
                    )
                    if existing is not None:
                        skipped.append(
                            {
                                "local_path": str(file_path),
                                "existing_id": existing.get("id"),
                            }
                        )
                        continue
                payload = self._upload_file_resumable(
                    service,
                    local_path=file_path,
                    folder_id=folder_id,
                    supports_all_drives=supports_all_drives,
                    chunk_size_mb=chunk_size_mb,
                )
                uploaded.append({"local_path": str(file_path), "file": payload})
            except Exception as exc:
                failed.append({"local_path": str(file_path), "error": str(exc)})

        return json.dumps(
            {
                "action": action_name,
                "source_dir": str(root),
                "candidate_count": len(candidates),
                "uploaded_count": len(uploaded),
                "skipped_count": len(skipped),
                "failed_count": len(failed),
                "uploaded": uploaded,
                "skipped": skipped,
                "failed": failed,
            },
            ensure_ascii=True,
        )

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
