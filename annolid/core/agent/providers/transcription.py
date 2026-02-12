from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import httpx


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
        except Exception:
            return ""
