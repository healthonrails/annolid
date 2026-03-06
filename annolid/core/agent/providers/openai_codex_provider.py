from __future__ import annotations

import asyncio
import hashlib
import json
from urllib.parse import urlparse, urlunparse
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional, Sequence, Tuple

from .base import LLMProvider, LLMResponse, ToolCallRequest

DEFAULT_CODEX_URL = "https://chatgpt.com/backend-api/codex/responses"
DEFAULT_ORIGINATOR = "annolid"


@dataclass(frozen=True)
class OpenAICodexResolved:
    model: str
    base_url: str
    transport: str
    websocket_url: str


def resolve_openai_codex(config: Any) -> OpenAICodexResolved:
    params = dict(getattr(config, "params", {}) or {})
    base_url = str(params.get("base_url") or "").strip() or DEFAULT_CODEX_URL
    model = str(getattr(config, "model", "") or "").strip()
    transport = _normalize_transport(params.get("transport"))
    websocket_url = _resolve_websocket_url(
        params.get("websocket_url"), base_url=base_url
    )
    return OpenAICodexResolved(
        model=model or "openai-codex/gpt-5.4",
        base_url=base_url,
        transport=transport,
        websocket_url=websocket_url,
    )


class OpenAICodexProvider(LLMProvider):
    """OpenAI Codex Responses API provider using OAuth CLI credentials."""

    def __init__(
        self,
        *,
        resolved: OpenAICodexResolved,
        token_getter: Optional[Callable[[], Any]] = None,
        request_callable: Optional[Callable[..., Any]] = None,
        websocket_request_callable: Optional[Callable[..., Any]] = None,
    ) -> None:
        self._resolved = resolved
        self._token_getter = token_getter
        self._request_callable = request_callable or _request_codex
        self._websocket_request_callable = (
            websocket_request_callable or _request_codex_websocket
        )

    def get_default_model(self) -> str:
        return self._resolved.model

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
        del max_tokens, temperature
        token_getter = self._token_getter
        if token_getter is None:
            try:
                from oauth_cli_kit import get_token as get_codex_token  # type: ignore
            except ImportError:
                return LLMResponse(
                    content=(
                        "OpenAI Codex provider requires `oauth_cli_kit`. "
                        "Install it in your Annolid environment, for example: "
                        "`.venv/bin/pip install oauth-cli-kit`."
                    ),
                    finish_reason="error",
                )
            token_getter = get_codex_token

        try:
            token = await asyncio.to_thread(token_getter)
        except Exception as exc:
            return LLMResponse(
                content=f"Error getting Codex OAuth token: {exc}",
                finish_reason="error",
            )

        system_prompt, input_items = _convert_messages(messages)
        body: Dict[str, Any] = {
            "model": _strip_model_prefix(model or self._resolved.model),
            "store": False,
            "stream": True,
            "instructions": system_prompt,
            "input": input_items,
            "text": {"verbosity": "medium"},
            "include": ["reasoning.encrypted_content"],
            "prompt_cache_key": _prompt_cache_key(messages),
            "tool_choice": "auto",
            "parallel_tool_calls": True,
        }
        if tools:
            body["tools"] = _convert_tools(tools)

        headers = _build_headers(
            account_id=str(getattr(token, "account_id", "") or ""),
            access_token=str(getattr(token, "access", "") or ""),
        )
        url = str(self._resolved.base_url or DEFAULT_CODEX_URL).strip()
        timeout_s = float(timeout_seconds) if timeout_seconds else 60.0

        try:
            (
                content,
                tool_calls,
                finish_reason,
                reasoning_content,
            ) = await _request_codex_with_transport(
                url=url,
                websocket_url=self._resolved.websocket_url,
                headers=headers,
                body=body,
                transport=self._resolved.transport,
                timeout_seconds=timeout_s,
                on_token=on_token,
                request_callable=self._request_callable,
                websocket_request_callable=self._websocket_request_callable,
            )
        except Exception as exc:
            return LLMResponse(
                content=f"Error calling Codex: {exc}",
                finish_reason="error",
            )
        return LLMResponse(
            content=content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            reasoning_content=reasoning_content or None,
        )


def _strip_model_prefix(model: str) -> str:
    raw = str(model or "").strip()
    lowered = raw.lower()
    for prefix in ("openai-codex/", "openai_codex/"):
        if lowered.startswith(prefix):
            return raw[len(prefix) :]
    return raw


def _build_headers(*, account_id: str, access_token: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {access_token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": "responses=experimental",
        "originator": DEFAULT_ORIGINATOR,
        "User-Agent": "annolid (python)",
        "accept": "text/event-stream",
        "content-type": "application/json",
    }


async def _request_codex(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    *,
    transport: str,
    timeout_seconds: float,
    on_token: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[ToolCallRequest], str, str]:
    try:
        import httpx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenAI Codex provider requires `httpx`. "
            "Install it in your Annolid environment, for example: "
            "`.venv/bin/pip install httpx`."
        ) from exc

    async with httpx.AsyncClient(timeout=timeout_seconds, verify=True) as client:
        async with client.stream("POST", url, headers=headers, json=body) as response:
            if response.status_code != 200:
                text = await response.aread()
                raise RuntimeError(
                    _friendly_error(
                        response.status_code, text.decode("utf-8", "ignore")
                    )
                )
            return await _consume_sse(response, on_token=on_token)


async def _request_codex_with_transport(
    *,
    url: str,
    websocket_url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    transport: str,
    timeout_seconds: float,
    on_token: Optional[Callable[[str], None]],
    request_callable: Callable[..., Any],
    websocket_request_callable: Callable[..., Any],
) -> Tuple[str, List[ToolCallRequest], str, str]:
    resolved_transport = _normalize_transport(transport)
    if resolved_transport == "sse":
        return await request_callable(
            url,
            headers,
            body,
            transport="sse",
            timeout_seconds=timeout_seconds,
            on_token=on_token,
        )
    if resolved_transport == "websocket":
        return await websocket_request_callable(
            websocket_url,
            headers,
            body,
            timeout_seconds=timeout_seconds,
            on_token=on_token,
        )
    try:
        return await websocket_request_callable(
            websocket_url,
            headers,
            body,
            timeout_seconds=timeout_seconds,
            on_token=on_token,
        )
    except Exception:
        return await request_callable(
            url,
            headers,
            body,
            transport="sse",
            timeout_seconds=timeout_seconds,
            on_token=on_token,
        )


async def _request_codex_websocket(
    url: str,
    headers: Dict[str, str],
    body: Dict[str, Any],
    *,
    timeout_seconds: float,
    on_token: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[ToolCallRequest], str, str]:
    try:
        import websockets  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "OpenAI Codex websocket transport requires `websockets`. "
            "Install it in your Annolid environment, for example: "
            "`.venv/bin/pip install websockets`."
        ) from exc

    async with asyncio.timeout(timeout_seconds):
        async with websockets.connect(
            url,
            additional_headers=headers,
            max_size=None,
        ) as websocket:
            await websocket.send(json.dumps(body))

            async def _iter_events() -> AsyncGenerator[Dict[str, Any], None]:
                while True:
                    message = await websocket.recv()
                    if isinstance(message, bytes):
                        text = message.decode("utf-8", "ignore")
                    else:
                        text = str(message or "")
                    for event in _decode_ws_events(text):
                        yield event
                        if event.get("type") in {
                            "response.completed",
                            "response.failed",
                            "error",
                        }:
                            return

            return await _consume_event_stream(_iter_events(), on_token=on_token)


def _convert_tools(tools: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for tool in tools:
        fn = (tool.get("function") or {}) if tool.get("type") == "function" else tool
        name = fn.get("name")
        if not name:
            continue
        params = fn.get("parameters") or {}
        converted.append(
            {
                "type": "function",
                "name": name,
                "description": fn.get("description") or "",
                "parameters": params if isinstance(params, dict) else {},
            }
        )
    return converted


def _convert_messages(
    messages: Sequence[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    system_prompt = ""
    input_items: List[Dict[str, Any]] = []
    for idx, msg in enumerate(messages):
        role = msg.get("role")
        content = msg.get("content")
        if role == "system":
            system_prompt = content if isinstance(content, str) else ""
            continue
        if role == "user":
            input_items.append(_convert_user_message(content))
            continue
        if role == "assistant":
            if isinstance(content, str) and content:
                input_items.append(
                    {
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": content}],
                        "status": "completed",
                        "id": f"msg_{idx}",
                    }
                )
            for tool_call in msg.get("tool_calls", []) or []:
                fn = tool_call.get("function") or {}
                call_id, item_id = _split_tool_call_id(tool_call.get("id"))
                input_items.append(
                    {
                        "type": "function_call",
                        "id": item_id or f"fc_{idx}",
                        "call_id": call_id or f"call_{idx}",
                        "name": fn.get("name"),
                        "arguments": fn.get("arguments") or "{}",
                    }
                )
            continue
        if role == "tool":
            call_id, _ = _split_tool_call_id(msg.get("tool_call_id"))
            if isinstance(content, str):
                output_text = content
            else:
                output_text = json.dumps(content, ensure_ascii=False)
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": call_id,
                    "output": output_text,
                }
            )
    return system_prompt, input_items


def _convert_user_message(content: Any) -> Dict[str, Any]:
    if isinstance(content, str):
        return {"role": "user", "content": [{"type": "input_text", "text": content}]}
    if isinstance(content, list):
        converted: List[Dict[str, Any]] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            if item.get("type") == "text":
                converted.append({"type": "input_text", "text": item.get("text", "")})
            elif item.get("type") == "image_url":
                url = (item.get("image_url") or {}).get("url")
                if url:
                    converted.append(
                        {"type": "input_image", "image_url": url, "detail": "auto"}
                    )
        if converted:
            return {"role": "user", "content": converted}
    return {"role": "user", "content": [{"type": "input_text", "text": ""}]}


def _split_tool_call_id(tool_call_id: Any) -> Tuple[str, Optional[str]]:
    if isinstance(tool_call_id, str) and tool_call_id:
        if "|" in tool_call_id:
            call_id, item_id = tool_call_id.split("|", 1)
            return call_id, item_id or None
        return tool_call_id, None
    return "call_0", None


def _prompt_cache_key(messages: Sequence[Dict[str, Any]]) -> str:
    raw = json.dumps(list(messages), ensure_ascii=True, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


async def _iter_sse(response: Any) -> AsyncGenerator[Dict[str, Any], None]:
    buffer: List[str] = []
    async for line in response.aiter_lines():
        if line == "":
            if buffer:
                data_lines = [
                    data_line[5:].strip()
                    for data_line in buffer
                    if data_line.startswith("data:")
                ]
                buffer = []
                if not data_lines:
                    continue
                data = "\n".join(data_lines).strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    yield json.loads(data)
                except Exception:
                    continue
            continue
        buffer.append(line)


async def _consume_sse(
    response: Any,
    *,
    on_token: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[ToolCallRequest], str, str]:
    return await _consume_event_stream(_iter_sse(response), on_token=on_token)


async def _consume_event_stream(
    events: AsyncGenerator[Dict[str, Any], None],
    *,
    on_token: Optional[Callable[[str], None]] = None,
) -> Tuple[str, List[ToolCallRequest], str, str]:
    content = ""
    reasoning_content = ""
    tool_calls: List[ToolCallRequest] = []
    tool_call_buffers: Dict[str, Dict[str, Any]] = {}
    finish_reason = "stop"

    async for event in events:
        event_type = event.get("type")
        if event_type == "response.output_item.added":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if call_id:
                    tool_call_buffers[call_id] = {
                        "id": item.get("id") or "fc_0",
                        "name": item.get("name"),
                        "arguments": item.get("arguments") or "",
                    }
        elif event_type == "response.output_text.delta":
            delta = str(event.get("delta") or "")
            content += delta
            if on_token and delta:
                on_token(delta)
        elif event_type == "response.reasoning_summary_text.delta":
            delta = str(event.get("delta") or "")
            reasoning_content += delta
        elif event_type == "response.function_call_arguments.delta":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] += event.get("delta") or ""
        elif event_type == "response.function_call_arguments.done":
            call_id = event.get("call_id")
            if call_id and call_id in tool_call_buffers:
                tool_call_buffers[call_id]["arguments"] = event.get("arguments") or ""
        elif event_type == "response.output_item.done":
            item = event.get("item") or {}
            if item.get("type") == "function_call":
                call_id = item.get("call_id")
                if not call_id:
                    continue
                buf = tool_call_buffers.get(call_id) or {}
                args_raw = buf.get("arguments") or item.get("arguments") or "{}"
                try:
                    args = json.loads(args_raw)
                except Exception:
                    args = {"_raw": args_raw}
                tool_calls.append(
                    ToolCallRequest(
                        id=f"{call_id}|{buf.get('id') or item.get('id') or 'fc_0'}",
                        name=str(buf.get("name") or item.get("name") or ""),
                        arguments=args if isinstance(args, dict) else {"_raw": args},
                    )
                )
        elif event_type == "response.completed":
            status = (event.get("response") or {}).get("status")
            finish_reason = _map_finish_reason(status)
        elif event_type in {"error", "response.failed"}:
            raise RuntimeError("Codex response failed")

    return content, tool_calls, finish_reason, reasoning_content


_FINISH_REASON_MAP = {
    "completed": "stop",
    "incomplete": "length",
    "failed": "error",
    "cancelled": "error",
}


def _normalize_transport(value: Any) -> str:
    transport = str(value or "").strip().lower()
    if transport in {"auto", "sse", "websocket"}:
        return transport
    return "auto"


def _resolve_websocket_url(value: Any, *, base_url: str) -> str:
    explicit = str(value or "").strip()
    if explicit:
        return explicit
    parsed = urlparse(str(base_url or "").strip() or DEFAULT_CODEX_URL)
    if parsed.scheme == "https":
        scheme = "wss"
    elif parsed.scheme == "http":
        scheme = "ws"
    elif parsed.scheme in {"ws", "wss"}:
        scheme = parsed.scheme
    else:
        scheme = "wss"
    return urlunparse(parsed._replace(scheme=scheme))


def _decode_ws_events(message: str) -> List[Dict[str, Any]]:
    text = str(message or "").strip()
    if not text:
        return []
    if text == "[DONE]":
        return []
    try:
        payload = json.loads(text)
    except Exception:
        return []
    if isinstance(payload, dict):
        return [payload]
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    return []


def _map_finish_reason(status: Optional[str]) -> str:
    return _FINISH_REASON_MAP.get(status or "completed", "stop")


def _friendly_error(status_code: int, raw: str) -> str:
    if status_code == 429:
        return "ChatGPT usage quota exceeded or rate limit triggered. Please try again later."
    return f"HTTP {status_code}: {raw}"
