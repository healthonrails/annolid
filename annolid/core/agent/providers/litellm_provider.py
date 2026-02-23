from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from .base import LLMProvider, LLMResponse, ToolCallRequest
from .registry import find_by_model, find_by_name, find_gateway


class LiteLLMProvider(LLMProvider):
    """LiteLLM-backed multi-provider adapter with graceful optional import."""

    def __init__(
        self,
        *,
        provider_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        default_model: str,
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.provider_name = str(provider_name or "").strip().lower() or None
        self.api_key = str(api_key or "").strip() or None
        self.api_base = str(api_base or "").strip() or None
        self.default_model = str(default_model)
        self.extra_headers = dict(extra_headers or {})

        self._gateway = find_gateway(
            provider_name=self.provider_name,
            api_key=self.api_key,
            api_base=self.api_base,
        )
        self._configure_env()

    def get_default_model(self) -> str:
        return self.default_model

    def _provider_spec(self):
        return (
            self._gateway
            or find_by_name(self.provider_name or "")
            or find_by_model(self.default_model)
        )

    def _configure_env(self) -> None:
        spec = self._provider_spec()
        if spec is None:
            return
        if self.api_key:
            if self._gateway:
                os.environ[spec.env_key] = self.api_key
            else:
                os.environ.setdefault(spec.env_key, self.api_key)

        effective_base = self.api_base or spec.default_api_base
        for env_name, env_val in spec.env_extras:
            resolved = env_val.replace("{api_key}", self.api_key or "")
            resolved = resolved.replace("{api_base}", effective_base)
            os.environ.setdefault(env_name, resolved)

    def _resolve_model(self, model: str) -> str:
        model_name = str(model or self.default_model)
        spec = self._provider_spec() or find_by_model(model_name)
        if spec is None:
            return model_name

        if self._gateway and spec.strip_model_prefix and "/" in model_name:
            model_name = model_name.split("/")[-1]

        if spec.litellm_prefix:
            if not any(model_name.startswith(prefix) for prefix in spec.skip_prefixes):
                if not model_name.startswith(f"{spec.litellm_prefix}/"):
                    model_name = f"{spec.litellm_prefix}/{model_name}"
        return model_name

    def _apply_model_overrides(self, model: str, payload: Dict[str, Any]) -> None:
        spec = self._provider_spec() or find_by_model(model)
        if spec is None:
            return
        lowered = model.lower()
        for pattern, overrides in spec.model_overrides:
            if pattern in lowered:
                payload.update(dict(overrides))
                return

    async def chat(
        self,
        *,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        model: Optional[str] = None,
        max_tokens: int = 4096,
        temperature: Optional[float] = 0.7,
        on_token: Optional[Callable[[str], None]] = None,
    ) -> LLMResponse:
        try:
            import litellm  # type: ignore
            from litellm import acompletion  # type: ignore
        except ImportError:
            return LLMResponse(
                content=(
                    "LiteLLM provider requires `litellm`. "
                    "Install with `pip install litellm`."
                ),
                finish_reason="error",
            )

        litellm.suppress_debug_info = True
        litellm.drop_params = True

        resolved_model = self._resolve_model(model or self.default_model)
        payload: Dict[str, Any] = {
            "model": resolved_model,
            "messages": list(messages),
            "max_tokens": int(max_tokens),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        self._apply_model_overrides(resolved_model, payload)

        if self.api_key:
            payload["api_key"] = self.api_key
        if self.api_base:
            payload["api_base"] = self.api_base
        elif self._provider_spec() and self._provider_spec().default_api_base:
            payload["api_base"] = self._provider_spec().default_api_base

        if self.extra_headers:
            payload["extra_headers"] = dict(self.extra_headers)

        # Tools do not support streaming well in many LiteLLM integrations if mixed with content.
        # But we can try streaming if no tools are provided.
        should_stream = bool(on_token) and not tools

        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
            should_stream = (
                False  # Disable streaming when tools are present for stability
            )

        try:
            if should_stream:
                payload["stream"] = True
                response_content = []
                reasoning_chunks = []
                finish_reason = "stop"
                usage = {}

                async for chunk in await acompletion(**payload):
                    delta = chunk.choices[0].delta
                    content = getattr(delta, "content", "") or ""
                    reasoning = getattr(delta, "reasoning_content", None)

                    if reasoning:
                        on_token(f"<think>{reasoning}</think>")
                        reasoning_chunks.append(reasoning)

                    if content:
                        on_token(content)
                        response_content.append(content)

                    if chunk.choices[0].finish_reason:
                        finish_reason = chunk.choices[0].finish_reason

                    if hasattr(chunk, "usage") and chunk.usage:
                        usage = {
                            "prompt_tokens": chunk.usage.prompt_tokens,
                            "completion_tokens": chunk.usage.completion_tokens,
                            "total_tokens": chunk.usage.total_tokens,
                        }

                return LLMResponse(
                    content="".join(response_content),
                    finish_reason=finish_reason,
                    usage=usage,
                    reasoning_content="".join(reasoning_chunks)
                    if reasoning_chunks
                    else None,
                )
            else:
                completion = await acompletion(**payload)
                return self._parse_response(completion)
        except Exception as exc:
            return LLMResponse(
                content=f"Error calling LLM: {exc}", finish_reason="error"
            )

    def _parse_response(self, completion: Any) -> LLMResponse:
        choice = completion.choices[0]
        message = choice.message

        tool_calls: List[ToolCallRequest] = []
        for tc in list(getattr(message, "tool_calls", None) or []):
            raw_args = getattr(getattr(tc, "function", None), "arguments", "{}")
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
                    id=str(getattr(tc, "id", "")),
                    name=str(getattr(getattr(tc, "function", None), "name", "")),
                    arguments=args,
                )
            )

        usage_obj = getattr(completion, "usage", None)
        usage = {}
        if usage_obj is not None:
            usage = {
                "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    getattr(usage_obj, "completion_tokens", 0) or 0
                ),
                "total_tokens": int(getattr(usage_obj, "total_tokens", 0) or 0),
            }

        return LLMResponse(
            content=str(getattr(message, "content", "") or ""),
            tool_calls=tool_calls,
            finish_reason=str(getattr(choice, "finish_reason", "stop") or "stop"),
            usage=usage,
            reasoning_content=getattr(message, "reasoning_content", None),
        )
