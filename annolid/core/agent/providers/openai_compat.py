from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from annolid.utils.llm_settings import LLMConfig

from .base import LLMProvider, LLMResponse, ToolCallRequest
from .registry import find_by_model, find_by_name, find_gateway


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

    # Keep third-party SDK env behavior aligned.
    if api_key and spec.env_key:
        os.environ[spec.env_key] = api_key
    if base_url and spec.name == "openai":
        os.environ["OPENAI_BASE_URL"] = base_url

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
            except Exception:
                pass
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
            except Exception:
                pass
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
    ) -> LLMResponse:
        client = self._ensure_client()
        try:
            payload: Dict[str, Any] = {
                "model": model or self._resolved.model,
                "messages": list(messages),
                "max_tokens": int(max_tokens),
            }
            if temperature is not None:
                payload["temperature"] = float(temperature)
            if timeout_seconds is not None:
                payload["timeout"] = float(timeout_seconds)
            if tools:
                payload["tools"] = list(tools)
                payload["tool_choice"] = "auto"
            completion = await client.chat.completions.create(**payload)
        except Exception as exc:
            return LLMResponse(
                content=f"Error calling LLM: {exc}", finish_reason="error"
            )
        return self._parse_response(completion)

    async def close(self) -> None:
        client = self._client
        if client is None:
            return
        await self._close_client(client)

    def _parse_response(self, completion: Any) -> LLMResponse:
        choices = self._get_value(completion, "choices", None)
        if not choices:
            return LLMResponse(
                content="",
                tool_calls=(),
                finish_reason=str(
                    self._get_value(completion, "finish_reason", "stop") or "stop"
                ),
                usage={},
                reasoning_content=None,
            )
        choice = choices[0]
        message = self._get_value(choice, "message", None)
        if message is None:
            return LLMResponse(
                content="",
                tool_calls=(),
                finish_reason=str(
                    self._get_value(choice, "finish_reason", "stop") or "stop"
                ),
                usage={},
                reasoning_content=None,
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

        return LLMResponse(
            content=str(content),
            tool_calls=tool_calls,
            finish_reason=str(
                self._get_value(choice, "finish_reason", "stop") or "stop"
            ),
            usage=usage,
            reasoning_content=self._get_value(message, "reasoning_content", None),
        )
