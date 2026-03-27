from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Sequence

from annolid.utils.box import BoxApiClient, BoxApiError, BoxOAuthTokens

from .common import _resolve_read_path, _resolve_write_path
from .function_base import FunctionTool


class BoxTool(FunctionTool):
    """Tool for core Box file operations used by Annolid agent workflows."""

    _ACTIONS = {
        "list_folder_items",
        "search",
        "get_file_info",
        "download_file",
        "upload_file",
    }

    def __init__(
        self,
        *,
        access_token: str = "",
        client_id: str = "",
        client_secret: str = "",
        refresh_token: str = "",
        token_url: str = "https://api.box.com/oauth2/token",
        api_base: str = "https://api.box.com/2.0",
        upload_api_base: str = "https://upload.box.com/api/2.0",
        enterprise_id: str = "",
        auto_refresh: bool = True,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = list(allowed_read_roots or [])
        self._enterprise_id = str(enterprise_id or "").strip()
        self._last_refresh: BoxOAuthTokens | None = None

        self._client = BoxApiClient(
            access_token=access_token,
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
            token_url=token_url,
            api_base=api_base,
            upload_api_base=upload_api_base,
            auto_refresh=bool(auto_refresh),
            on_token_refresh=self._on_token_refresh,
        )

    def _on_token_refresh(self, tokens: BoxOAuthTokens) -> None:
        self._last_refresh = tokens

    @property
    def name(self) -> str:
        return "box"

    @property
    def description(self) -> str:
        return (
            "Access Box content for Annolid workflows and Cornell Box requests: "
            "list folder items, search, get file metadata, download files, and "
            "upload files."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": sorted(self._ACTIONS),
                },
                "folder_id": {"type": "string", "description": "Box folder id."},
                "file_id": {"type": "string", "description": "Box file id."},
                "query": {"type": "string", "description": "Search query text."},
                "destination_path": {
                    "type": "string",
                    "description": (
                        "Local destination path for download. Can be a file path "
                        "or an existing directory."
                    ),
                },
                "file_path": {
                    "type": "string",
                    "description": "Local file path to upload.",
                },
                "file_name": {
                    "type": "string",
                    "description": (
                        "Optional override name for upload/download target file name."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 1000,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                },
                "item_type": {
                    "type": "string",
                    "enum": ["", "file", "folder", "web_link"],
                    "description": "Optional Box search type filter.",
                },
                "fields": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of Box fields to request.",
                },
                "overwrite": {
                    "type": "boolean",
                    "description": (
                        "For download/upload, allow replacing an existing local "
                        "or remote file version."
                    ),
                },
            },
            "required": ["action"],
        }

    async def execute(self, action: str, **kwargs: Any) -> str:
        action_name = str(action or "").strip().lower()
        if action_name not in self._ACTIONS:
            return self._error(
                "Unsupported action.",
                allowed_actions=sorted(self._ACTIONS),
            )

        try:
            self._client.ensure_access_token()
        except (BoxApiError, ValueError) as exc:
            return self._error(
                str(exc),
                hint=(
                    "Configure tools.box.access_token, or set tools.box.client_id, "
                    "tools.box.client_secret, and tools.box.refresh_token to enable "
                    "OAuth refresh. If your Box tenant uses an org-specific login "
                    "host, set tools.box.authorize_base_url and regenerate the auth "
                    "URL with `annolid-run agent-box-auth-url`."
                ),
            )

        try:
            if action_name == "list_folder_items":
                return await asyncio.to_thread(
                    self._list_folder_items,
                    folder_id=str(kwargs.get("folder_id") or "0").strip() or "0",
                    limit=int(kwargs.get("limit") or 100),
                    offset=int(kwargs.get("offset") or 0),
                    fields=self._normalize_fields(kwargs.get("fields")),
                )

            if action_name == "search":
                query = str(kwargs.get("query") or "").strip()
                if not query:
                    return self._error("search requires `query`.")
                return await asyncio.to_thread(
                    self._search,
                    query=query,
                    limit=int(kwargs.get("limit") or 20),
                    offset=int(kwargs.get("offset") or 0),
                    item_type=str(kwargs.get("item_type") or "").strip().lower(),
                    fields=self._normalize_fields(kwargs.get("fields")),
                )

            if action_name == "get_file_info":
                file_id = str(kwargs.get("file_id") or "").strip()
                if not file_id:
                    return self._error("get_file_info requires `file_id`.")
                return await asyncio.to_thread(
                    self._get_file_info,
                    file_id=file_id,
                    fields=self._normalize_fields(kwargs.get("fields")),
                )

            if action_name == "download_file":
                file_id = str(kwargs.get("file_id") or "").strip()
                destination_path = str(kwargs.get("destination_path") or "").strip()
                if not file_id or not destination_path:
                    return self._error(
                        "download_file requires `file_id` and `destination_path`."
                    )
                return await asyncio.to_thread(
                    self._download_file,
                    file_id=file_id,
                    destination_path=destination_path,
                    file_name=str(kwargs.get("file_name") or "").strip(),
                    overwrite=bool(kwargs.get("overwrite", False)),
                )

            if action_name == "upload_file":
                folder_id = str(kwargs.get("folder_id") or "0").strip() or "0"
                file_path = str(kwargs.get("file_path") or "").strip()
                if not file_path:
                    return self._error("upload_file requires `file_path`.")
                return await asyncio.to_thread(
                    self._upload_file,
                    folder_id=folder_id,
                    file_path=file_path,
                    file_name=str(kwargs.get("file_name") or "").strip(),
                    overwrite=bool(kwargs.get("overwrite", False)),
                )
        except ValueError as exc:
            return self._error(str(exc))
        except PermissionError as exc:
            return self._error(str(exc))
        except FileNotFoundError as exc:
            return self._error(str(exc))
        except BoxApiError as exc:
            return self._error(
                exc.message,
                status=exc.status,
                details=exc.payload,
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            return self._error("Box tool failed unexpectedly.", details=str(exc))

        return self._error("Unsupported action.")

    @staticmethod
    def _normalize_fields(value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        fields: list[str] = []
        for item in value:
            text = str(item or "").strip()
            if text:
                fields.append(text)
        return fields

    def _list_folder_items(
        self,
        *,
        folder_id: str,
        limit: int,
        offset: int,
        fields: list[str],
    ) -> str:
        if limit < 1 or limit > 1000:
            raise ValueError("`limit` must be between 1 and 1000.")
        if offset < 0:
            raise ValueError("`offset` must be >= 0.")
        payload = self._client.list_folder_items(
            folder_id=folder_id,
            limit=limit,
            offset=offset,
            fields=fields,
        )
        return self._ok(
            action="list_folder_items",
            folder_id=folder_id,
            total_count=int(payload.get("total_count", 0) or 0),
            entries=payload.get("entries", []),
            limit=payload.get("limit", limit),
            offset=payload.get("offset", offset),
            token_refreshed=bool(self._last_refresh is not None),
        )

    def _search(
        self,
        *,
        query: str,
        limit: int,
        offset: int,
        item_type: str,
        fields: list[str],
    ) -> str:
        if limit < 1 or limit > 1000:
            raise ValueError("`limit` must be between 1 and 1000.")
        if offset < 0:
            raise ValueError("`offset` must be >= 0.")
        payload = self._client.search(
            query=query,
            limit=limit,
            offset=offset,
            item_type=item_type,
            fields=fields,
        )
        return self._ok(
            action="search",
            query=query,
            total_count=int(payload.get("total_count", 0) or 0),
            entries=payload.get("entries", []),
            limit=payload.get("limit", limit),
            offset=payload.get("offset", offset),
            token_refreshed=bool(self._last_refresh is not None),
        )

    def _get_file_info(self, *, file_id: str, fields: list[str]) -> str:
        payload = self._client.get_file_info(file_id=file_id, fields=fields)
        return self._ok(
            action="get_file_info",
            file=payload,
            token_refreshed=bool(self._last_refresh is not None),
        )

    def _download_file(
        self,
        *,
        file_id: str,
        destination_path: str,
        file_name: str,
        overwrite: bool,
    ) -> str:
        requested_path = _resolve_write_path(
            destination_path, allowed_dir=self._allowed_dir
        )
        if requested_path.exists() and requested_path.is_dir():
            info = self._client.get_file_info(
                file_id=file_id, fields=["id", "name", "size", "sha1"]
            )
            resolved_name = file_name or str(info.get("name") or "").strip()
            if not resolved_name:
                raise ValueError("Unable to resolve file name for download target.")
            target_path = requested_path / resolved_name
            file_meta = info
        else:
            target_path = requested_path
            file_meta = {}

        if target_path.exists() and not overwrite:
            raise ValueError(
                f"Target path already exists: {target_path}. Set overwrite=true to replace it."
            )

        binary_payload = self._client.download_file_content(file_id=file_id)
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_bytes(binary_payload)

        return self._ok(
            action="download_file",
            file_id=file_id,
            path=str(target_path),
            bytes_written=len(binary_payload),
            file=file_meta,
            token_refreshed=bool(self._last_refresh is not None),
        )

    def _upload_file(
        self,
        *,
        folder_id: str,
        file_path: str,
        file_name: str,
        overwrite: bool,
    ) -> str:
        source_path = _resolve_read_path(
            file_path,
            allowed_dir=self._allowed_dir,
            allowed_read_roots=self._allowed_read_roots,
        )
        if not source_path.exists() or not source_path.is_file():
            raise FileNotFoundError(f"Upload file does not exist: {source_path}")

        target_name = file_name or source_path.name
        try:
            upload_response = self._client.upload_file_content(
                folder_id=folder_id,
                source_path=source_path,
                target_name=target_name,
            )
        except BoxApiError as exc:
            if exc.status == 409:
                conflict_payload = exc.payload if isinstance(exc.payload, dict) else {}
                if not overwrite:
                    return self._error(
                        "A file with the same name already exists in the target folder.",
                        conflict=conflict_payload,
                    )
                existing_id = (
                    conflict_payload.get("context_info", {})
                    .get("conflicts", {})
                    .get("id")
                )
                if not existing_id:
                    return self._error(
                        "Box reported a conflict but no existing file id was provided.",
                        conflict=conflict_payload,
                    )
                upload_response = self._client.upload_file_version(
                    file_id=str(existing_id),
                    source_path=source_path,
                )
            else:
                raise

        if isinstance(upload_response, dict) and upload_response.get("entries"):
            return self._ok(
                action="upload_file",
                folder_id=folder_id,
                source_path=str(source_path),
                uploaded=upload_response.get("entries", [])[0],
                token_refreshed=bool(self._last_refresh is not None),
            )

        return self._error(
            "Upload did not return a valid Box response.",
            response=upload_response,
        )

    @staticmethod
    def _ok(**payload: Any) -> str:
        body = {"ok": True}
        body.update(payload)
        return json.dumps(body, ensure_ascii=False)

    @staticmethod
    def _error(message: str, **payload: Any) -> str:
        body = {"ok": False, "error": str(message)}
        body.update(payload)
        return json.dumps(body, ensure_ascii=False)


__all__ = ["BoxTool"]
