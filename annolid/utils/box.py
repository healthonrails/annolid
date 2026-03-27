"""Utilities for Box OAuth and REST API access.

This module provides a small stdlib-only client for Box OAuth2 and common file
operations used by Annolid runtime surfaces.
"""

from __future__ import annotations

import json
import mimetypes
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib import error, parse, request

BOX_AUTHORIZE_BASE_URL = "https://account.box.com"
BOX_AUTHORIZE_URL = f"{BOX_AUTHORIZE_BASE_URL}/api/oauth2/authorize"
BOX_TOKEN_URL = "https://api.box.com/oauth2/token"
BOX_API_BASE = "https://api.box.com/2.0"
BOX_UPLOAD_API_BASE = "https://upload.box.com/api/2.0"


class BoxApiError(RuntimeError):
    """Raised when Box OAuth/API requests fail."""

    def __init__(self, *, status: int, message: str, payload: Any = None) -> None:
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)
        self.payload = payload


@dataclass
class BoxOAuthTokens:
    access_token: str
    refresh_token: str = ""
    expires_in: int | None = None
    token_type: str = "bearer"


def build_box_authorize_url(
    *,
    client_id: str,
    redirect_uri: str,
    state: str | None = None,
    response_type: str = "code",
    authorize_base_url: str = BOX_AUTHORIZE_BASE_URL,
) -> str:
    """Build a Box OAuth authorization URL."""
    auth_url = _resolve_box_authorize_url(authorize_base_url)
    params: dict[str, str] = {
        "response_type": str(response_type or "code"),
        "client_id": str(client_id or "").strip(),
        "redirect_uri": str(redirect_uri or "").strip(),
    }
    if not params["client_id"]:
        raise ValueError("client_id is required.")
    if not params["redirect_uri"]:
        raise ValueError("redirect_uri is required.")
    if state is not None and str(state).strip():
        params["state"] = str(state).strip()
    return f"{auth_url}?{parse.urlencode(params)}"


def _resolve_box_authorize_url(authorize_base_url: str) -> str:
    """Normalize a Box organization auth host into a full authorize URL."""
    raw_value = str(authorize_base_url or "").strip()
    if not raw_value:
        return BOX_AUTHORIZE_URL

    candidate = raw_value
    if "://" not in candidate:
        candidate = f"https://{candidate.lstrip('/')}"

    parsed = parse.urlsplit(candidate)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError("authorize_base_url must include a valid Box host.")

    path = str(parsed.path or "").rstrip("/")
    if path.endswith("/api/oauth2/authorize"):
        return parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))

    if not path:
        path = "/api/oauth2/authorize"
    elif not path.endswith("/api/oauth2/authorize"):
        path = f"{path}/api/oauth2/authorize"

    return parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))


def exchange_box_authorization_code(
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
    token_url: str = BOX_TOKEN_URL,
    timeout_s: float = 30.0,
) -> BoxOAuthTokens:
    """Exchange a Box OAuth authorization code for access+refresh tokens."""
    payload = {
        "grant_type": "authorization_code",
        "code": str(code or "").strip(),
        "client_id": str(client_id or "").strip(),
        "client_secret": str(client_secret or "").strip(),
        "redirect_uri": str(redirect_uri or "").strip(),
    }
    if not payload["code"]:
        raise ValueError("code is required.")
    if not payload["client_id"]:
        raise ValueError("client_id is required.")
    if not payload["client_secret"]:
        raise ValueError("client_secret is required.")
    if not payload["redirect_uri"]:
        raise ValueError("redirect_uri is required.")
    data = _oauth_token_request(
        token_url=token_url, form_payload=payload, timeout_s=timeout_s
    )
    return BoxOAuthTokens(
        access_token=str(data.get("access_token") or "").strip(),
        refresh_token=str(data.get("refresh_token") or "").strip(),
        expires_in=int(data.get("expires_in"))
        if data.get("expires_in") is not None
        else None,
        token_type=str(data.get("token_type") or "bearer").strip() or "bearer",
    )


def refresh_box_access_token(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
    token_url: str = BOX_TOKEN_URL,
    timeout_s: float = 30.0,
) -> BoxOAuthTokens:
    """Refresh a Box access token using OAuth refresh_token grant."""
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": str(refresh_token or "").strip(),
        "client_id": str(client_id or "").strip(),
        "client_secret": str(client_secret or "").strip(),
    }
    if not payload["refresh_token"]:
        raise ValueError("refresh_token is required.")
    if not payload["client_id"]:
        raise ValueError("client_id is required.")
    if not payload["client_secret"]:
        raise ValueError("client_secret is required.")
    data = _oauth_token_request(
        token_url=token_url, form_payload=payload, timeout_s=timeout_s
    )
    return BoxOAuthTokens(
        access_token=str(data.get("access_token") or "").strip(),
        refresh_token=str(
            data.get("refresh_token") or payload["refresh_token"]
        ).strip(),
        expires_in=int(data.get("expires_in"))
        if data.get("expires_in") is not None
        else None,
        token_type=str(data.get("token_type") or "bearer").strip() or "bearer",
    )


def _oauth_token_request(
    *,
    token_url: str,
    form_payload: dict[str, str],
    timeout_s: float,
) -> dict[str, Any]:
    encoded = parse.urlencode(form_payload).encode("utf-8")
    req = request.Request(
        str(token_url or BOX_TOKEN_URL),
        method="POST",
        data=encoded,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        },
    )
    raw = _send_request(req=req, timeout_s=timeout_s)
    if not isinstance(raw, dict):
        raise BoxApiError(
            status=0,
            message="Box OAuth endpoint returned non-object JSON payload.",
            payload=raw,
        )
    if not str(raw.get("access_token") or "").strip():
        raise BoxApiError(
            status=0,
            message="Box OAuth endpoint did not return access_token.",
            payload=raw,
        )
    return raw


def _send_request(*, req: request.Request, timeout_s: float) -> Any:
    try:
        with request.urlopen(req, timeout=max(1.0, float(timeout_s or 30.0))) as resp:  # noqa: S310
            body = resp.read()
    except error.HTTPError as exc:
        body = exc.read()
        parsed_payload: Any = None
        message = str(exc)
        if body:
            with _suppress_json_decode_errors():
                parsed_payload = json.loads(body.decode("utf-8"))
                if isinstance(parsed_payload, dict):
                    message = str(parsed_payload.get("message") or message)
        raise BoxApiError(
            status=int(getattr(exc, "code", 0) or 0),
            message=message,
            payload=parsed_payload,
        ) from exc
    except error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise BoxApiError(
            status=0, message=f"Failed to reach Box API: {reason}"
        ) from exc

    if not body:
        return {}
    try:
        return json.loads(body.decode("utf-8"))
    except json.JSONDecodeError:
        return body.decode("utf-8", errors="replace")


class BoxApiClient:
    """Small Box REST client with optional OAuth refresh support."""

    def __init__(
        self,
        *,
        access_token: str = "",
        client_id: str = "",
        client_secret: str = "",
        refresh_token: str = "",
        token_url: str = BOX_TOKEN_URL,
        api_base: str = BOX_API_BASE,
        upload_api_base: str = BOX_UPLOAD_API_BASE,
        timeout_s: float = 60.0,
        auto_refresh: bool = True,
        on_token_refresh: Callable[[BoxOAuthTokens], None] | None = None,
    ) -> None:
        self.access_token = str(access_token or "").strip()
        self.client_id = str(client_id or "").strip()
        self.client_secret = str(client_secret or "").strip()
        self.refresh_token = str(refresh_token or "").strip()
        self.token_url = str(token_url or BOX_TOKEN_URL).strip() or BOX_TOKEN_URL
        self.api_base = str(api_base or BOX_API_BASE).rstrip("/")
        self.upload_api_base = str(upload_api_base or BOX_UPLOAD_API_BASE).rstrip("/")
        self.timeout_s = max(1.0, float(timeout_s or 60.0))
        self.auto_refresh = bool(auto_refresh)
        self.on_token_refresh = on_token_refresh

    def can_refresh(self) -> bool:
        return bool(self.client_id and self.client_secret and self.refresh_token)

    def ensure_access_token(self) -> None:
        if self.access_token:
            return
        if self.auto_refresh and self.can_refresh():
            self.refresh_access_token()
            return
        raise BoxApiError(
            status=401,
            message="Box access token is missing and refresh credentials are unavailable.",
        )

    def refresh_access_token(self) -> BoxOAuthTokens:
        tokens = refresh_box_access_token(
            client_id=self.client_id,
            client_secret=self.client_secret,
            refresh_token=self.refresh_token,
            token_url=self.token_url,
            timeout_s=self.timeout_s,
        )
        self.access_token = tokens.access_token
        if tokens.refresh_token:
            self.refresh_token = tokens.refresh_token
        if self.on_token_refresh is not None:
            self.on_token_refresh(tokens)
        return tokens

    def list_folder_items(
        self,
        *,
        folder_id: str,
        limit: int = 100,
        offset: int = 0,
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": int(limit), "offset": int(offset)}
        if fields:
            params["fields"] = ",".join(fields)
        return self.request_json(
            "GET",
            f"{self.api_base}/folders/{parse.quote(str(folder_id))}/items",
            params=params,
        )

    def search(
        self,
        *,
        query: str,
        limit: int = 20,
        offset: int = 0,
        item_type: str = "",
        fields: list[str] | None = None,
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "query": str(query),
            "limit": int(limit),
            "offset": int(offset),
        }
        if item_type in {"file", "folder", "web_link"}:
            params["type"] = item_type
        if fields:
            params["fields"] = ",".join(fields)
        return self.request_json("GET", f"{self.api_base}/search", params=params)

    def get_file_info(
        self, *, file_id: str, fields: list[str] | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {}
        if fields:
            params["fields"] = ",".join(fields)
        return self.request_json(
            "GET",
            f"{self.api_base}/files/{parse.quote(str(file_id))}",
            params=params,
        )

    def download_file_content(self, *, file_id: str) -> bytes:
        return self.request_bytes(
            "GET",
            f"{self.api_base}/files/{parse.quote(str(file_id))}/content",
        )

    def upload_file_content(
        self,
        *,
        folder_id: str,
        source_path: Path,
        target_name: str,
    ) -> dict[str, Any]:
        attributes = {"name": target_name, "parent": {"id": str(folder_id)}}
        content_type, payload = build_multipart_upload_body(
            source_path=source_path,
            attributes=attributes,
        )
        return self.request_json(
            "POST",
            f"{self.upload_api_base}/files/content",
            data=payload,
            headers={"Content-Type": content_type},
        )

    def upload_file_version(
        self,
        *,
        file_id: str,
        source_path: Path,
    ) -> dict[str, Any]:
        attributes = {"name": source_path.name}
        content_type, payload = build_multipart_upload_body(
            source_path=source_path,
            attributes=attributes,
        )
        return self.request_json(
            "POST",
            f"{self.upload_api_base}/files/{parse.quote(str(file_id))}/content",
            data=payload,
            headers={"Content-Type": content_type},
        )

    def request_json(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        raw = self.request_bytes(
            method,
            url,
            params=params,
            data=data,
            headers=headers,
        )
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except json.JSONDecodeError as exc:
            raise BoxApiError(
                status=0,
                message=f"Box API returned non-JSON response: {exc}",
            ) from exc
        if isinstance(payload, dict):
            return payload
        raise BoxApiError(
            status=0,
            message="Box API returned unexpected JSON payload type.",
            payload=payload,
        )

    def request_bytes(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        data: bytes | None = None,
        headers: dict[str, str] | None = None,
    ) -> bytes:
        self.ensure_access_token()
        return self._request_bytes_with_retry(
            method=method,
            url=url,
            params=params,
            data=data,
            headers=headers,
            allow_refresh_retry=bool(self.auto_refresh and self.can_refresh()),
        )

    def _request_bytes_with_retry(
        self,
        *,
        method: str,
        url: str,
        params: dict[str, Any] | None,
        data: bytes | None,
        headers: dict[str, str] | None,
        allow_refresh_retry: bool,
    ) -> bytes:
        final_url = str(url)
        if params:
            query = parse.urlencode(
                {key: value for key, value in params.items() if value is not None}
            )
            if query:
                final_url = f"{final_url}?{query}"

        req_headers = {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }
        if headers:
            req_headers.update(headers)
        req = request.Request(
            final_url,
            method=str(method or "GET").upper(),
            data=data,
            headers=req_headers,
        )
        try:
            with request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310
                return resp.read()
        except error.HTTPError as exc:
            body = exc.read()
            parsed_payload: Any = None
            message = str(exc)
            if body:
                with _suppress_json_decode_errors():
                    parsed_payload = json.loads(body.decode("utf-8"))
                    if isinstance(parsed_payload, dict):
                        message = str(parsed_payload.get("message") or message)

            status = int(getattr(exc, "code", 0) or 0)
            if status == 401 and allow_refresh_retry:
                self.refresh_access_token()
                return self._request_bytes_with_retry(
                    method=method,
                    url=url,
                    params=params,
                    data=data,
                    headers=headers,
                    allow_refresh_retry=False,
                )
            raise BoxApiError(
                status=status, message=message, payload=parsed_payload
            ) from exc
        except error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            raise BoxApiError(
                status=0, message=f"Failed to reach Box API: {reason}"
            ) from exc


class _suppress_json_decode_errors:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, tb) -> bool:
        from json import JSONDecodeError

        return bool(exc_type and issubclass(exc_type, JSONDecodeError))


def build_multipart_upload_body(
    *,
    source_path: Path,
    attributes: dict[str, Any],
) -> tuple[str, bytes]:
    """Build multipart/form-data payload for Box file upload endpoints."""
    boundary = f"annolid-box-{uuid.uuid4().hex}"
    content_type = f"multipart/form-data; boundary={boundary}"

    mime_type = mimetypes.guess_type(source_path.name)[0] or "application/octet-stream"
    file_bytes = source_path.read_bytes()

    parts: list[bytes] = []
    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        b'Content-Disposition: form-data; name="attributes"\r\n'
        b"Content-Type: application/json\r\n\r\n"
    )
    parts.append(json.dumps(attributes).encode("utf-8"))
    parts.append(b"\r\n")

    parts.append(f"--{boundary}\r\n".encode("utf-8"))
    parts.append(
        (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{source_path.name}"\r\n'
        ).encode("utf-8")
    )
    parts.append(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
    parts.append(file_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    return content_type, b"".join(parts)


def _load_box_config_file(config_file: str | Path) -> dict[str, Any]:
    path = Path(config_file).expanduser()
    return json.loads(path.read_text(encoding="utf-8"))


def get_box_client(
    config_file: str | Path = "../config/box_config.json",
) -> BoxApiClient:
    """Backwards-compatible helper that builds a BoxApiClient from config JSON.

    Expected keys include either legacy Box app config (`boxAppSettings`) or modern
    flat fields under `tools.box`-style payloads.
    """
    cfg = _load_box_config_file(config_file)
    box_app = dict(cfg.get("boxAppSettings") or {})
    return BoxApiClient(
        access_token=str(
            cfg.get("developer_token") or cfg.get("access_token") or ""
        ).strip(),
        client_id=str(cfg.get("client_id") or box_app.get("clientID") or "").strip(),
        client_secret=str(
            cfg.get("client_secret") or box_app.get("clientSecret") or ""
        ).strip(),
        refresh_token=str(cfg.get("refresh_token") or "").strip(),
        token_url=str(cfg.get("token_url") or BOX_TOKEN_URL).strip() or BOX_TOKEN_URL,
        api_base=str(cfg.get("api_base") or BOX_API_BASE).strip() or BOX_API_BASE,
        upload_api_base=str(cfg.get("upload_api_base") or BOX_UPLOAD_API_BASE).strip()
        or BOX_UPLOAD_API_BASE,
    )


def get_box_folder_items(
    client: BoxApiClient, folder_id: str = "0"
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Backwards-compatible helper returning folder metadata and entries."""
    payload = client.list_folder_items(folder_id=str(folder_id), limit=1000, offset=0)
    folder = {"id": str(folder_id)}
    entries = list(payload.get("entries") or [])
    return folder, entries


def upload_file(
    client: BoxApiClient, folder_id: str, local_file_path: str | Path
) -> dict[str, Any]:
    """Upload a local file to a Box folder."""
    src = Path(local_file_path).expanduser().resolve()
    return client.upload_file_content(
        folder_id=str(folder_id),
        source_path=src,
        target_name=src.name,
    )


def download_file(
    client: BoxApiClient, file_id: str, local_file_path: str | Path
) -> Path:
    """Download a Box file to local disk and return written path."""
    out_path = Path(local_file_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(client.download_file_content(file_id=str(file_id)))
    return out_path


def is_results_complete(
    box_folder_entries: list[dict[str, Any]],
    result_file_pattern: str = "_motion.csv",
    num_expected_results: int = 0,
) -> bool:
    """Check whether folder entries contain the expected number of result files."""
    num_results = 0
    for item in box_folder_entries:
        name = str(item.get("name") or "")
        if result_file_pattern in name:
            num_results += 1
    return num_results == int(num_expected_results)


def upload_results(
    client: BoxApiClient,
    *,
    folder_id: str,
    tracking_results: dict[str, str | Path],
    csv_pattern: str = "motion.csv",
) -> bool:
    """Upload Annolid tracking results for folder name keyed in tracking_results."""
    folder_payload = client.list_folder_items(
        folder_id=str(folder_id), limit=1000, offset=0
    )
    folder_entries = list(folder_payload.get("entries") or [])
    folder_name = str(folder_payload.get("name") or folder_id)

    if any(csv_pattern in str(item.get("name") or "") for item in folder_entries):
        return True

    local_file = tracking_results.get(folder_name)
    if local_file is None:
        return False

    source = Path(local_file).expanduser().resolve()
    if not source.exists():
        return False

    client.upload_file_content(
        folder_id=str(folder_id),
        source_path=source,
        target_name=source.name,
    )
    return True


__all__ = [
    "BOX_API_BASE",
    "BOX_AUTHORIZE_BASE_URL",
    "BOX_AUTHORIZE_URL",
    "BOX_TOKEN_URL",
    "BOX_UPLOAD_API_BASE",
    "BoxApiClient",
    "BoxApiError",
    "BoxOAuthTokens",
    "build_box_authorize_url",
    "build_multipart_upload_body",
    "download_file",
    "exchange_box_authorization_code",
    "get_box_client",
    "get_box_folder_items",
    "is_results_complete",
    "refresh_box_access_token",
    "upload_file",
    "upload_results",
]
