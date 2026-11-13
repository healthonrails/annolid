from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Optional
import urllib.request
import uuid


def _encode_multipart_formdata(
    *,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    content_type: str,
    file_bytes: bytes,
) -> tuple[bytes, str]:
    boundary = f"----AnnolidFormBoundary{uuid.uuid4().hex}"
    safe_filename = filename.replace('"', "_").replace("\r", "_").replace("\n", "_")
    chunks: list[bytes] = []
    for name, value in fields.items():
        chunks.append(f"--{boundary}\r\n".encode("ascii"))
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8")
        )
        chunks.append(f"{value}\r\n".encode("utf-8"))
    chunks.append(f"--{boundary}\r\n".encode("ascii"))
    chunks.append(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{safe_filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8")
    )
    chunks.append(file_bytes)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("ascii"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _transcribe_via_urllib(
    *,
    url: str,
    headers: dict[str, str],
    path: Path,
    model: str,
) -> str:
    body, content_type = _encode_multipart_formdata(
        fields={"model": model},
        file_field="file",
        filename=path.name,
        content_type="audio/wav",
        file_bytes=path.read_bytes(),
    )
    request_headers = dict(headers)
    request_headers["Content-Type"] = content_type
    request = urllib.request.Request(
        url,
        data=body,
        headers=request_headers,
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60.0) as response:
        raw = response.read()
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict):
        return ""
    return str(payload.get("text") or "")


class OpenAICompatTranscriptionProvider:
    """Audio transcription via an OpenAI-compatible transcription endpoint."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        model: str = "whisper-1",
    ) -> None:
        self.api_key = str(api_key or os.environ.get("OPENAI_API_KEY") or "").strip()
        self.api_base = str(
            api_base or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1"
        ).rstrip("/")
        self.model = str(model)

    async def transcribe(self, file_path: str | Path) -> str:
        if not self.api_key:
            return ""
        path = Path(file_path).expanduser()
        if not path.exists():
            return ""

        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.api_base}/audio/transcriptions"

        try:
            try:
                import httpx
            except ImportError:
                httpx = None  # type: ignore[assignment]

            if httpx is not None:
                async with httpx.AsyncClient(timeout=60.0) as client:
                    with path.open("rb") as handle:
                        files = {
                            "file": (path.name, handle, "audio/wav"),
                            "model": (None, self.model),
                        }
                        response = await client.post(url, headers=headers, files=files)
                        response.raise_for_status()
                        payload = response.json()
                        return str(payload.get("text") or "")

            return await asyncio.to_thread(
                _transcribe_via_urllib,
                url=url,
                headers=headers,
                path=path,
                model=self.model,
            )
        except Exception:
            return ""
