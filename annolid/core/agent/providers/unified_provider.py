from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Optional

from .base import LLMProvider, LLMResponse, ToolCallRequest
from .registry import find_by_model, find_by_name, find_gateway


class UnifiedLLMProvider(LLMProvider):
    """Compatibility adapter backed by OpenAI-compatible and Anthropic SDKs."""

    _runtime_logging_configured = False

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
        self._openai_client: Any = None
        self._anthropic_client: Any = None

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

    @staticmethod
    def _get_value(obj: Any, key: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

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

        # If the model looks like it has an explicit prefix, try to find that specific spec first
        model_spec = find_by_model(model_name) if "/" in model_name else None
        spec = model_spec or self._provider_spec() or find_by_model(model_name)
        if spec is None:
            return model_name

        if spec.strip_model_prefix and "/" in model_name:
            # Only strip the specific provider prefix if it matches
            prefixes_to_strip = [f"{spec.name}/", f"{spec.model_prefix}/"]
            for pfx in prefixes_to_strip:
                if model_name.startswith(pfx):
                    model_name = model_name[len(pfx) :]
                    break

        if spec.model_prefix:
            if not any(model_name.startswith(prefix) for prefix in spec.skip_prefixes):
                if not model_name.startswith(f"{spec.model_prefix}/"):
                    model_name = f"{spec.model_prefix}/{model_name}"
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

    @staticmethod
    def _sanitize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        allowed = {
            "role",
            "content",
            "tool_calls",
            "tool_call_id",
            "name",
            "reasoning_content",
        }
        sanitized: List[Dict[str, Any]] = []
        for msg in messages:
            clean = {k: v for k, v in dict(msg).items() if k in allowed}
            if (
                str(clean.get("role") or "") == "assistant"
                and clean.get("tool_calls")
                and ("content" not in clean or clean.get("content") == "")
            ):
                clean["content"] = None
            sanitized.append(clean)
        return sanitized

    @classmethod
    def _to_openai_messages(
        cls, messages: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Normalize messages to OpenAI-compatible shape."""
        sanitized = cls._sanitize_messages(messages)
        normalized: List[Dict[str, Any]] = []
        for msg in sanitized:
            clean = dict(msg)
            clean.pop("reasoning_content", None)
            normalized.append(clean)
        return normalized

    @staticmethod
    def _parse_tool_call_arguments(raw_args: Any) -> Dict[str, Any]:
        if isinstance(raw_args, dict):
            return dict(raw_args)
        if isinstance(raw_args, str):
            text = raw_args.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    return dict(parsed)
            except json.JSONDecodeError:
                try:
                    import json_repair  # type: ignore

                    repaired = json_repair.loads(text)
                    if isinstance(repaired, dict):
                        return dict(repaired)
                except Exception:
                    pass
            return {"_raw": raw_args}
        return {"_raw": raw_args}

    def _resolved_provider_name(self, model: Optional[str] = None) -> str:
        model_name = str(model or self.default_model or "")
        explicit = model_name.split("/", 1)[0].lower() if "/" in model_name else ""
        if explicit in {
            "anthropic",
            "claude",
        }:
            return "anthropic"
        spec = self._provider_spec() or find_by_model(model_name)
        return str(getattr(spec, "name", "") or "")

    def _ensure_openai_client(self) -> Any:
        if self._openai_client is not None:
            return self._openai_client
        try:
            from openai import AsyncOpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "openai package is required for UnifiedLLMProvider."
            ) from exc
        self._openai_client = AsyncOpenAI(
            api_key=self.api_key or "no-key",
            base_url=self.api_base or None,
            default_headers=(dict(self.extra_headers) if self.extra_headers else None),
        )
        return self._openai_client

    def _ensure_anthropic_client(self) -> Any:
        if self._anthropic_client is not None:
            return self._anthropic_client
        try:
            from anthropic import AsyncAnthropic  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "anthropic package is required for Anthropic models. "
                "Install with `pip install anthropic`."
            ) from exc
        kwargs: Dict[str, Any] = {}
        if self.api_key:
            kwargs["api_key"] = self.api_key
        if self.api_base:
            kwargs["base_url"] = self.api_base
        if self.extra_headers:
            kwargs["default_headers"] = dict(self.extra_headers)
        self._anthropic_client = AsyncAnthropic(**kwargs)
        return self._anthropic_client

    @staticmethod
    def _strip_known_model_prefix(model_name: str, provider_name: str) -> str:
        prefix = f"{provider_name}/"
        if model_name.startswith(prefix):
            return model_name[len(prefix) :]
        return model_name

    def _convert_tools_for_anthropic(
        self, tools: Optional[List[Dict[str, Any]]]
    ) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        converted: List[Dict[str, Any]] = []
        for tool in tools:
            fn = dict(tool.get("function") or tool)
            entry = {
                "name": str(fn.get("name") or ""),
                "input_schema": fn.get(
                    "parameters",
                    {"type": "object", "properties": {}},
                ),
            }
            description = fn.get("description")
            if description:
                entry["description"] = str(description)
            converted.append(entry)
        return converted

    def _convert_messages_for_anthropic(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        system_parts: List[str] = []
        converted: List[Dict[str, Any]] = []

        for msg in self._sanitize_messages(messages):
            role = str(msg.get("role") or "")
            content = msg.get("content")
            if role == "system":
                if isinstance(content, str):
                    system_parts.append(content)
                elif content is not None:
                    system_parts.append(str(content))
                continue

            if role == "tool":
                converted.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": str(msg.get("tool_call_id") or ""),
                                "content": content if content is not None else "",
                            }
                        ],
                    }
                )
                continue

            if role == "assistant":
                blocks: List[Dict[str, Any]] = []
                if isinstance(content, str) and content:
                    blocks.append({"type": "text", "text": content})
                for idx, tc in enumerate(list(msg.get("tool_calls") or [])):
                    fn = dict(tc.get("function") or {})
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": str(tc.get("id") or f"call_{idx}"),
                            "name": str(fn.get("name") or ""),
                            "input": self._parse_tool_call_arguments(
                                fn.get("arguments", "{}")
                            ),
                        }
                    )
                converted.append(
                    {
                        "role": "assistant",
                        "content": blocks or [{"type": "text", "text": ""}],
                    }
                )
                continue

            converted.append(
                {
                    "role": "user",
                    "content": content if content is not None else "",
                }
            )

        merged: List[Dict[str, Any]] = []
        for msg in converted:
            if merged and merged[-1]["role"] == msg["role"]:
                prev = merged[-1]["content"]
                curr = msg["content"]
                if isinstance(prev, str):
                    prev = [{"type": "text", "text": prev}]
                if isinstance(curr, str):
                    curr = [{"type": "text", "text": curr}]
                if isinstance(curr, list):
                    prev.extend(curr)
                merged[-1]["content"] = prev
            else:
                merged.append(msg)

        return "\n\n".join([text for text in system_parts if text]).strip(), merged

    def _parse_anthropic_response(self, completion: Any) -> LLMResponse:
        content_blocks = list(self._get_value(completion, "content", []) or [])
        text_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        tool_calls: List[ToolCallRequest] = []
        for idx, block in enumerate(content_blocks):
            b_type = str(self._get_value(block, "type", "") or "")
            if b_type == "text":
                text = str(self._get_value(block, "text", "") or "")
                if text:
                    text_chunks.append(text)
            elif b_type == "thinking":
                reasoning = str(self._get_value(block, "thinking", "") or "")
                if reasoning:
                    reasoning_chunks.append(reasoning)
            elif b_type == "tool_use":
                tool_calls.append(
                    ToolCallRequest(
                        id=str(self._get_value(block, "id", "") or f"call_{idx}"),
                        name=str(self._get_value(block, "name", "")),
                        arguments=dict(self._get_value(block, "input", {}) or {}),
                    )
                )

        usage_obj = self._get_value(completion, "usage", None)
        usage: Dict[str, int] = {}
        if usage_obj is not None:
            input_tokens = int(self._get_value(usage_obj, "input_tokens", 0) or 0)
            output_tokens = int(self._get_value(usage_obj, "output_tokens", 0) or 0)
            usage = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

        return LLMResponse(
            content="".join(text_chunks),
            tool_calls=tool_calls,
            finish_reason=str(
                self._get_value(completion, "stop_reason", "stop") or "stop"
            ),
            usage=usage,
            reasoning_content="".join(reasoning_chunks) if reasoning_chunks else None,
        )

    async def _chat_openai_compat(
        self,
        *,
        resolved_model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        temperature: Optional[float],
        on_token: Optional[Callable[[str], None]],
    ) -> LLMResponse:
        client = self._ensure_openai_client()
        payload: Dict[str, Any] = {
            "model": resolved_model,
            "messages": self._to_openai_messages(list(messages)),
            "max_tokens": int(max_tokens),
        }
        if temperature is not None:
            payload["temperature"] = float(temperature)
        self._apply_model_overrides(resolved_model, payload)
        should_stream = bool(on_token) and not tools
        if tools:
            payload["tools"] = list(tools)
            payload["tool_choice"] = "auto"
            should_stream = False

        if should_stream:
            payload["stream"] = True
            response_content: List[str] = []
            reasoning_chunks: List[str] = []
            finish_reason = "stop"
            usage: Dict[str, int] = {}

            stream = await client.chat.completions.create(**payload)
            async for chunk in stream:
                choices = self._get_value(chunk, "choices", None)
                if not choices:
                    chunk_usage = self._get_value(chunk, "usage", None)
                    if chunk_usage:
                        usage = {
                            "prompt_tokens": int(
                                self._get_value(chunk_usage, "prompt_tokens", 0) or 0
                            ),
                            "completion_tokens": int(
                                self._get_value(chunk_usage, "completion_tokens", 0)
                                or 0
                            ),
                            "total_tokens": int(
                                self._get_value(chunk_usage, "total_tokens", 0) or 0
                            ),
                        }
                    continue
                delta = self._get_value(choices[0], "delta", None)
                content = str(self._get_value(delta, "content", "") or "")
                reasoning = self._get_value(delta, "reasoning_content", None)
                if reasoning and on_token:
                    on_token(f"<think>{reasoning}</think>")
                    reasoning_chunks.append(str(reasoning))
                if content and on_token:
                    on_token(content)
                    response_content.append(content)
                chunk_finish = self._get_value(choices[0], "finish_reason", None)
                if chunk_finish:
                    finish_reason = str(chunk_finish)
            return LLMResponse(
                content="".join(response_content),
                finish_reason=finish_reason,
                usage=usage,
                reasoning_content="".join(reasoning_chunks)
                if reasoning_chunks
                else None,
            )

        completion = await client.chat.completions.create(**payload)
        return self._parse_response(completion)

    async def _chat_anthropic(
        self,
        *,
        resolved_model: str,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]],
        max_tokens: int,
        temperature: Optional[float],
        on_token: Optional[Callable[[str], None]],
    ) -> LLMResponse:
        client = self._ensure_anthropic_client()
        system_text, anthropic_messages = self._convert_messages_for_anthropic(messages)
        stripped_model = self._strip_known_model_prefix(resolved_model, "anthropic")
        stripped_model = self._strip_known_model_prefix(stripped_model, "claude")
        payload: Dict[str, Any] = {
            "model": stripped_model,
            "messages": anthropic_messages,
            "max_tokens": int(max_tokens),
        }
        if system_text:
            payload["system"] = system_text
        if temperature is not None:
            payload["temperature"] = float(temperature)
        converted_tools = self._convert_tools_for_anthropic(tools)
        if converted_tools:
            payload["tools"] = converted_tools
            payload["tool_choice"] = {"type": "auto"}
        completion = await client.messages.create(**payload)
        response = self._parse_anthropic_response(completion)
        if on_token and response.content:
            on_token(response.content)
        return response

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
        resolved_model = self._resolve_model(model or self.default_model)
        try:
            provider_name = self._resolved_provider_name(resolved_model)
            if provider_name == "anthropic":
                return await self._chat_anthropic(
                    resolved_model=resolved_model,
                    messages=messages,
                    tools=tools,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    on_token=on_token,
                )
            return await self._chat_openai_compat(
                resolved_model=resolved_model,
                messages=messages,
                tools=tools,
                max_tokens=max_tokens,
                temperature=temperature,
                on_token=on_token,
            )
        except Exception as exc:
            return LLMResponse(
                content=f"Error calling LLM: {exc}", finish_reason="error"
            )

    @classmethod
    def _configure_runtime_logging(cls, module: Any) -> None:
        """
        Compatibility shim kept for existing tests and legacy call sites.
        """
        module.suppress_debug_info = True
        module.drop_params = True
        if hasattr(module, "set_verbose"):
            try:
                module.set_verbose = False
            except Exception:
                pass
        if cls._runtime_logging_configured:
            return
        cls._runtime_logging_configured = True

    def _parse_response(self, completion: Any) -> LLMResponse:
        choices = self._get_value(completion, "choices", None)
        if not choices:
            return LLMResponse(content="", finish_reason="stop", usage={})
        choice = choices[0]
        message = self._get_value(choice, "message", None)
        if message is None:
            return LLMResponse(
                content="",
                finish_reason=str(
                    self._get_value(choice, "finish_reason", "stop") or "stop"
                ),
                usage={},
            )

        tool_calls: List[ToolCallRequest] = []
        for idx, tc in enumerate(
            list(self._get_value(message, "tool_calls", None) or [])
        ):
            fn = self._get_value(tc, "function", None)
            raw_args = self._get_value(fn, "arguments", "{}")
            args = self._parse_tool_call_arguments(raw_args)
            tool_calls.append(
                ToolCallRequest(
                    id=str(self._get_value(tc, "id", "") or f"call_{idx}"),
                    name=str(self._get_value(fn, "name", "")),
                    arguments=args,
                )
            )

        usage_obj = self._get_value(completion, "usage", None)
        usage = {}
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

        return LLMResponse(
            content=str(self._get_value(message, "content", "") or ""),
            tool_calls=tool_calls,
            finish_reason=str(
                self._get_value(choice, "finish_reason", "stop") or "stop"
            ),
            usage=usage,
            reasoning_content=self._get_value(message, "reasoning_content", None),
        )

    async def close(self) -> None:
        """Best-effort shutdown for underlying async clients."""
        clients = [self._openai_client, self._anthropic_client]
        self._openai_client = None
        self._anthropic_client = None
        for client in clients:
            if client is None:
                continue
            close_fn = getattr(client, "aclose", None)
            if callable(close_fn):
                try:
                    result = close_fn()
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    continue
                continue
            close_fn = getattr(client, "close", None)
            if callable(close_fn):
                try:
                    result = close_fn()
                    if hasattr(result, "__await__"):
                        await result
                except Exception:
                    continue
