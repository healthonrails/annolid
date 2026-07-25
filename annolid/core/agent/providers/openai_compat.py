from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from annolid.utils.llm_settings import LLMConfig
from annolid.utils.logger import logger

from .base import (
    LLMProvider,
    LLMResponse,
    ToolCallRequest,
    error_response_from_exception,
)
from .call_runtime import (
    sanitize_openai_messages,
    sanitize_provider_error,
    sanitize_tool_schemas,
)
from .model_capabilities import is_known_text_only_model
from .registry import find_by_model, find_by_name, find_gateway


_MULTIMODAL_CONTENT_TYPES = frozenset(
    {
        "audio",
        "file",
        "image",
        "image_url",
        "input_audio",
        "input_file",
        "input_image",
        "input_video",
        "video",
        "video_url",
    }
)
_MULTIMODAL_DISABLED_ERROR_MARKERS = (
    "not a multimodal model",
    "does not support image input",
    "image input is not supported",
    "image inputs are not supported",
)
_TEXT_ONLY_ATTACHMENT_NOTE = (
    "[Visual or file attachment omitted because the current model accepts "
    "text-only input.]"
)


def _is_multimodal_content_block(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    block_type = str(value.get("type") or "").strip().lower()
    if block_type in _MULTIMODAL_CONTENT_TYPES:
        return True
    return any(
        key in value
        for key in (
            "audio_url",
            "file_data",
            "file_id",
            "file_url",
            "image_path",
            "image_url",
            "video_url",
        )
    )


def _messages_contain_multimodal_content(
    messages: List[Dict[str, Any]],
) -> bool:
    for message in messages:
        content = message.get("content")
        if isinstance(content, list) and any(
            _is_multimodal_content_block(item) for item in content
        ):
            return True
        if _is_multimodal_content_block(content):
            return True
    return False


def _text_only_message_content(content: Any) -> Any:
    if not isinstance(content, (dict, list)):
        return content
    blocks = content if isinstance(content, list) else [content]
    text_parts: List[str] = []
    omitted_attachment = False
    for block in blocks:
        if isinstance(block, str):
            if block:
                text_parts.append(block)
            continue
        if not isinstance(block, dict):
            text_parts.append(json.dumps(block, ensure_ascii=False))
            continue
        if _is_multimodal_content_block(block):
            omitted_attachment = True
            continue
        block_type = str(block.get("type") or "").strip().lower()
        if block_type in {"text", "input_text", "output_text"}:
            text = block.get("text")
            if isinstance(text, str) and text:
                text_parts.append(text)
            continue
        text_parts.append(json.dumps(block, ensure_ascii=False))
    if omitted_attachment:
        text_parts.append(_TEXT_ONLY_ATTACHMENT_NOTE)
    return "\n".join(text_parts)


def _downgrade_messages_to_text_only(
    messages: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    downgraded: List[Dict[str, Any]] = []
    for message in messages:
        normalized = dict(message)
        normalized["content"] = _text_only_message_content(message.get("content"))
        downgraded.append(normalized)
    return downgraded


def _should_retry_without_multimodal_content(
    exc: Exception,
    *,
    provider: str,
    model: str,
) -> bool:
    text = str(exc or "").strip().lower()
    if any(marker in text for marker in _MULTIMODAL_DISABLED_ERROR_MARKERS):
        return True
    if (
        "received multimodal data but multimodal processing is not enabled" in text
        and is_known_text_only_model(provider=provider, model=model)
    ):
        return True
    return False


@dataclass(frozen=True)
class OpenAICompatResolved:
    provider: str
    model: str
    api_key: str
    base_url: str


def resolve_openai_compat(config: LLMConfig) -> OpenAICompatResolved:
    provider_name = str(config.provider or "").strip().lower()
    model = str(config.model or "").strip()
    params = dict(config.params or {})
    api_key = str(params.get("api_key") or "").strip()
    base_url = str(params.get("base_url") or params.get("host") or "").strip()

    gateway = find_gateway(
        provider_name=provider_name,
        api_key=api_key or None,
        api_base=base_url or None,
    )
    spec = gateway or find_by_name(provider_name) or find_by_model(model)
    if spec is None:
        if provider_name in {"openai", "ollama"}:
            spec = find_by_name(provider_name)
        else:
            raise ValueError(
                f"Unsupported provider/model for agent loop: {provider_name}:{model}"
            )

    if spec.name == "ollama":
        if not base_url:
            base_url = spec.default_api_base
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        if not api_key:
            api_key = "ollama"
    elif spec.name == "vllm":
        if not base_url:
            base_url = spec.default_api_base
        if not base_url:
            raise ValueError(
                f"{spec.name} provider requires base_url/host in LLM settings."
            )
        if not base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/") + "/v1"
        if not api_key:
            api_key = "dummy"
    elif spec.name == "openai":
        if not base_url:
            base_url = spec.default_api_base
        if not api_key:
            raise ValueError("OpenAI provider requires API key for tool-calling chat.")
    else:
        # OpenRouter/other OpenAI-compatible gateways.
        if not base_url:
            base_url = spec.default_api_base
        if not api_key:
            raise ValueError(f"{spec.name} requires API key.")

    return OpenAICompatResolved(
        provider=spec.name,
        model=model,
        api_key=api_key,
        base_url=base_url,
    )


class OpenAICompatProvider(LLMProvider):
    """Provider that normalizes OpenAI-compatible APIs (OpenAI/Ollama/OpenRouter)."""

    def __init__(
        self,
        *,
        resolved: OpenAICompatResolved,
        client_factory: Optional[Callable[[OpenAICompatResolved], Any]] = None,
    ) -> None:
        self._resolved = resolved
        self._client_factory = client_factory
        self._client: Any = None
        self._text_only_models: set[str] = set()

    def get_default_model(self) -> str:
        return self._resolved.model

    def _ensure_client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory(self._resolved)
            return self._client
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai package is required for OpenAICompatProvider."
            ) from exc
        self._client = AsyncOpenAI(
            api_key=self._resolved.api_key,
            base_url=self._resolved.base_url,
            # Retries are classified in LLMProvider.chat_with_retry so callers
            # get consistent bounds and Retry-After behavior across SDKs.
            max_retries=0,
        )
        return self._client

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    async def _close_client(self, client: Any) -> None:
        close_fn = getattr(client, "aclose", None)
        if callable(close_fn):
            try:
                result = close_fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                logger.debug(
                    "Failed to close OpenAI-compatible async client cleanly: %s",
                    sanitize_provider_error(exc),
                )
            finally:
                if client is self._client:
                    self._client = None
            return
        close_fn = getattr(client, "close", None)
        if callable(close_fn):
            try:
                result = close_fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                logger.debug(
                    "Failed to close OpenAI-compatible client cleanly: %s",
                    sanitize_provider_error(exc),
                )
            finally:
                if client is self._client:
                    self._client = None

    async def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = 0.7,
        timeout_seconds: Optional[float] = None,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        client = self._ensure_client()
        try:
            payload: Dict[str, Any] = {
                "model": model or self._resolved.model,
                "messages": sanitize_openai_messages(list(messages)),
                "max_tokens": int(max_tokens),
            }
            if temperature is not None:
                payload["temperature"] = float(temperature)
            if timeout_seconds is not None:
                payload["timeout"] = float(timeout_seconds)

            should_stream = bool(on_token) and not tools

            if tools:
                payload["tools"] = sanitize_tool_schemas(list(tools))
                payload["tool_choice"] = "auto"
                should_stream = False

            request_model = str(model or self._resolved.model)
            request_messages = list(payload.get("messages") or [])
            model_cache_key = request_model.strip().lower()
            model_is_text_only = (
                model_cache_key in self._text_only_models
                or is_known_text_only_model(
                    provider=self._resolved.provider,
                    model=request_model,
                )
            )
            if model_is_text_only and _messages_contain_multimodal_content(
                request_messages
            ):
                logger.info(
                    "OpenAI-compatible provider %s model %s is text-only; "
                    "omitting multimodal message blocks before the request.",
                    self._resolved.provider,
                    request_model,
                )
                payload["messages"] = _downgrade_messages_to_text_only(request_messages)

            try:
                return await self._send_payload(
                    client=client,
                    payload=payload,
                    should_stream=should_stream,
                    on_token=on_token,
                )
            except Exception as exc:
                request_messages = list(payload.get("messages") or [])
                if not (
                    _should_retry_without_multimodal_content(
                        exc,
                        provider=self._resolved.provider,
                        model=request_model,
                    )
                    and _messages_contain_multimodal_content(request_messages)
                ):
                    raise
                self._text_only_models.add(model_cache_key)
                logger.warning(
                    "OpenAI-compatible provider %s rejected multimodal input for "
                    "model %s; retrying once with text-only message content.",
                    self._resolved.provider,
                    model or self._resolved.model,
                )
                retry_payload = dict(payload)
                retry_payload["messages"] = _downgrade_messages_to_text_only(
                    request_messages
                )
                return await self._send_payload(
                    client=client,
                    payload=retry_payload,
                    should_stream=should_stream,
                    on_token=on_token,
                )
        except Exception as exc:
            return error_response_from_exception(
                exc,
                prefix=(
                    "Error calling "
                    f"{self._resolved.provider}:{model or self._resolved.model}"
                ),
            )

    async def _send_payload(
        self,
        *,
        client: Any,
        payload: Dict[str, Any],
        should_stream: bool,
        on_token: Optional[Callable[[str], None]],
    ) -> LLMResponse:
        if not should_stream:
            completion = await client.chat.completions.create(**payload)
            return self._parse_response(completion)

        stream_payload = dict(payload)
        stream_payload["stream"] = True
        response_content = []
        reasoning_chunks = []
        finish_reason = "stop"
        usage = {}

        stream = await client.chat.completions.create(**stream_payload)
        async for chunk in stream:
            if not chunk.choices:
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = {
                        "prompt_tokens": chunk.usage.prompt_tokens,
                        "completion_tokens": chunk.usage.completion_tokens,
                        "total_tokens": chunk.usage.total_tokens,
                    }
                continue

            delta = chunk.choices[0].delta
            content = getattr(delta, "content", "") or ""
            # reasoning_content is sometimes in the delta for O1/O3 or deepseek
            reasoning = getattr(delta, "reasoning_content", None)

            if reasoning and on_token is not None:
                on_token(f"<think>{reasoning}</think>")
                reasoning_chunks.append(reasoning)

            if content:
                if on_token is not None:
                    on_token(content)
                response_content.append(content)

            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        content = "".join(response_content)
        reasoning_content = "".join(reasoning_chunks) if reasoning_chunks else None
        if not content and not reasoning_content and finish_reason == "stop":
            return LLMResponse(
                content="Model provider returned an empty streamed response.",
                finish_reason="error",
                usage=usage,
                error_kind="empty",
                error_should_retry=True,
            )
        return LLMResponse(
            content=content,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=reasoning_content,
        )

    async def close(self) -> None:
        client = self._client
        if client is None:
            return
        await self._close_client(client)

    def _parse_response(self, completion: Any) -> LLMResponse:
        choices = self._get_value(completion, "choices", None)
        if not choices:
            return LLMResponse(
                content="Model provider returned no response choices.",
                finish_reason="error",
                usage={},
                error_kind="empty",
                error_should_retry=True,
            )
        choice = choices[0]
        message = self._get_value(choice, "message", None)
        if message is None:
            return LLMResponse(
                content="Model provider returned a choice without a message.",
                finish_reason="error",
                usage={},
                error_kind="empty",
                error_should_retry=True,
            )

        tool_calls: List[ToolCallRequest] = []
        for tc in list(self._get_value(message, "tool_calls", None) or []):
            fn = self._get_value(tc, "function", None)
            raw_args = self._get_value(fn, "arguments", "{}")
            args: Dict[str, Any]
            if isinstance(raw_args, str):
                try:
                    parsed = json.loads(raw_args)
                    args = parsed if isinstance(parsed, dict) else {"_raw": raw_args}
                except json.JSONDecodeError:
                    args = {"_raw": raw_args}
            elif isinstance(raw_args, dict):
                args = dict(raw_args)
            else:
                args = {"_raw": raw_args}

            tool_calls.append(
                ToolCallRequest(
                    id=str(self._get_value(tc, "id", "")),
                    name=str(self._get_value(fn, "name", "")),
                    arguments=args,
                )
            )

        usage: Dict[str, int] = {}
        usage_obj = self._get_value(completion, "usage", None)
        if usage_obj is not None:
            usage = {
                "prompt_tokens": int(
                    self._get_value(usage_obj, "prompt_tokens", 0) or 0
                ),
                "completion_tokens": int(
                    self._get_value(usage_obj, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(self._get_value(usage_obj, "total_tokens", 0) or 0),
            }

        content = self._get_value(message, "content", "")
        if content is None:
            content = ""
        finish_reason = str(self._get_value(choice, "finish_reason", "stop") or "stop")
        reasoning_content = self._get_value(message, "reasoning_content", None)
        if (
            not str(content)
            and not tool_calls
            and not reasoning_content
            and finish_reason == "stop"
        ):
            return LLMResponse(
                content="Model provider returned an empty response.",
                finish_reason="error",
                usage=usage,
                error_kind="empty",
                error_should_retry=True,
            )

        return LLMResponse(
            content=str(content),
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=usage,
            reasoning_content=reasoning_content,
        )
