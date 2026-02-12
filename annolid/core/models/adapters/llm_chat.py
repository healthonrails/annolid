from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from annolid.core.agent.providers import resolve_openai_compat
from annolid.utils.llm_settings import resolve_llm_config

from ..base import ModelCapabilities, ModelRequest, ModelResponse, RuntimeModel


def _guess_mime_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if ext == ".png":
        return "image/png"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    if ext == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def _encode_image_data_uri(path: Path) -> str:
    mime = _guess_mime_type(path)
    data = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{data}"


@dataclass(frozen=True)
class _OpenAICompatConfig:
    provider: str
    model: str
    api_key: str
    base_url: str


class LLMChatAdapter(RuntimeModel):
    """OpenAI-compatible chat adapter backed by Annolid LLM settings."""

    def __init__(
        self,
        *,
        profile: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        persist: bool = False,
        client_factory: Optional[Callable[[_OpenAICompatConfig], Any]] = None,
    ) -> None:
        self._profile = profile
        self._provider_override = provider
        self._model_override = model
        self._persist = bool(persist)
        self._client_factory = client_factory

        self._client: Any = None
        self._config: Optional[_OpenAICompatConfig] = None

    @property
    def model_id(self) -> str:
        if self._config is None:
            return "llm:unloaded"
        return f"{self._config.provider}:{self._config.model}"

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            tasks=("chat", "caption"),
            input_modalities=("text", "image"),
            output_modalities=("text",),
            streaming=False,
        )

    def load(self) -> None:
        if self._client is not None:
            return

        cfg = resolve_llm_config(
            profile=self._profile,
            provider=self._provider_override,
            model=self._model_override,
            persist=self._persist,
        )
        resolved_cfg = resolve_openai_compat(cfg)

        resolved = _OpenAICompatConfig(
            provider=resolved_cfg.provider,
            model=resolved_cfg.model,
            api_key=resolved_cfg.api_key,
            base_url=resolved_cfg.base_url,
        )
        self._config = resolved

        if self._client_factory is not None:
            self._client = self._client_factory(resolved)
            return

        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OpenAI client is required for LLMChatAdapter. Install with `pip install openai`."
            ) from exc

        self._client = OpenAI(api_key=resolved.api_key, base_url=resolved.base_url)

    def predict(self, request: ModelRequest) -> ModelResponse:
        self.load()
        if self._client is None or self._config is None:
            raise RuntimeError("LLMChatAdapter is not loaded.")

        task = str(request.task).strip().lower()
        if task not in {"chat", "caption"}:
            raise ValueError(f"Unsupported task for LLMChatAdapter: {request.task!r}")

        temperature = float(request.params.get("temperature", 0.2))
        max_tokens = request.params.get("max_tokens")
        max_tokens_value = int(max_tokens) if max_tokens is not None else None

        if request.messages is not None:
            messages = list(request.messages)
        else:
            system_prompt = request.params.get("system_prompt")
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": str(system_prompt)})

            user_text = str(request.text or "").strip() or "Describe the input."
            if task == "caption" and request.image_path:
                image_uri = _encode_image_data_uri(Path(request.image_path))
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": user_text},
                            {"type": "image_url", "image_url": {"url": image_uri}},
                        ],
                    }
                )
            else:
                messages.append({"role": "user", "content": user_text})

        kwargs: Dict[str, Any] = {
            "model": self._config.model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens_value is not None:
            kwargs["max_tokens"] = max_tokens_value

        result = self._client.chat.completions.create(**kwargs)
        try:
            text = result.choices[0].message.content
        except Exception:
            text = None

        return ModelResponse(
            task=request.task,
            output={"text": text},
            text=text,
            raw=result,
            meta={"provider": self._config.provider, "model": self._config.model},
        )

    def close(self) -> None:
        self._client = None
        self._config = None
