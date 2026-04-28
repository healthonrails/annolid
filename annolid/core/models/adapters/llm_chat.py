from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional

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


def _get_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _content_to_text(content: Any) -> Optional[str]:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            value = _content_to_text(item)
            if value:
                parts.append(value)
        return "".join(parts) if parts else None
    if isinstance(content, Mapping):
        value = content.get("text") or content.get("content")
        if value is not None:
            return _content_to_text(value)
        if content.get("type") in {"output_text", "text"} and content.get("value"):
            return str(content.get("value"))
        return json.dumps(dict(content), ensure_ascii=False)
    return str(content) if content else None


def _extract_completion_text(result: Any) -> Optional[str]:
    direct_text = _content_to_text(_get_value(result, "output_text", None))
    if direct_text:
        return direct_text
    direct_text = _content_to_text(_get_value(result, "text", None))
    if direct_text:
        return direct_text

    choices = _get_value(result, "choices", None) or []
    if not choices:
        return None
    choice = choices[0]
    message = _get_value(choice, "message", None)
    if message is not None:
        for key in ("content", "parsed", "reasoning_content", "refusal"):
            text = _content_to_text(_get_value(message, key, None))
            if text:
                return text
    return _content_to_text(_get_value(choice, "text", None))


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
        use_annolid_bot_system: bool = True,
        workspace: Optional[str] = None,
        client_factory: Optional[Callable[[_OpenAICompatConfig], Any]] = None,
    ) -> None:
        self._profile = profile
        self._provider_override = provider
        self._model_override = model
        self._persist = bool(persist)
        self._use_annolid_bot_system = bool(use_annolid_bot_system)
        self._workspace_override = str(workspace or "").strip() or None
        self._client_factory = client_factory

        self._client: Any = None
        self._config: Optional[_OpenAICompatConfig] = None
        self._annolid_system_prompt_cache: Optional[str] = None

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
            use_annolid_system = bool(
                request.params.get(
                    "use_annolid_bot_system", self._use_annolid_bot_system
                )
            )
            if not system_prompt and use_annolid_system:
                system_prompt = self._resolve_annolid_system_prompt(
                    task_hint=str(request.text or "").strip()
                )
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
        response_format = request.params.get("response_format")
        if isinstance(response_format, Mapping):
            kwargs["response_format"] = dict(response_format)

        result = self._client.chat.completions.create(**kwargs)
        text = _extract_completion_text(result)
        choices = _get_value(result, "choices", None) or []
        finish_reason = ""
        if choices:
            finish_reason = str(_get_value(choices[0], "finish_reason", "") or "")

        return ModelResponse(
            task=request.task,
            output={"text": text},
            text=text,
            raw=result,
            meta={
                "provider": self._config.provider,
                "model": self._config.model,
                "finish_reason": finish_reason,
                "empty_text": not bool(str(text or "").strip()),
            },
        )

    def _resolve_annolid_system_prompt(self, *, task_hint: str = "") -> str:
        cached = str(self._annolid_system_prompt_cache or "").strip()
        if cached:
            return cached
        try:
            from annolid.core.agent.context import AgentContextBuilder
            from annolid.core.agent.utils import get_agent_workspace_path

            workspace = get_agent_workspace_path(self._workspace_override)
            builder = AgentContextBuilder(Path(workspace))
            prompt = str(
                builder.build_system_prompt(
                    task_hint=task_hint or "Vision-language classification request.",
                )
                or ""
            ).strip()
            if prompt:
                self._annolid_system_prompt_cache = prompt
            return prompt
        except Exception:
            return ""

    def close(self) -> None:
        self._client = None
        self._config = None
        self._annolid_system_prompt_cache = None
