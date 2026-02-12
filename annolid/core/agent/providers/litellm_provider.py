from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

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
            "messages": [dict(m) for m in messages],
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
        if tools:
            payload["tools"] = [dict(t) for t in tools]
            payload["tool_choice"] = "auto"

        try:
            completion = await acompletion(**payload)
        except Exception as exc:
            return LLMResponse(
                content=f"Error calling LLM: {exc}", finish_reason="error"
            )
        return self._parse_response(completion)

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
