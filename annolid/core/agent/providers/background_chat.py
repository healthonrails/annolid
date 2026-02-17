from __future__ import annotations

import asyncio
import base64
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import importlib
import mimetypes
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from annolid.utils.llm_settings import LLMConfig
from annolid.utils.llm_settings import provider_definitions, provider_kind

from .openai_compat import OpenAICompatProvider, resolve_openai_compat

OLLAMA_PLAIN_MODE_COOLDOWN_TURNS = 2
_OLLAMA_TOOL_SUPPORT_CACHE: Dict[str, bool] = {}
_OLLAMA_FORCE_PLAIN_CACHE: Dict[str, int] = {}


def dependency_error_for_kind(kind: str) -> Optional[str]:
    if kind == "openai_compat":
        if importlib.util.find_spec("openai") is None:
            return (
                "OpenAI-compatible provider requires the `openai` package. "
                "Install it in your Annolid environment, for example: "
                "`.venv/bin/pip install openai`."
            )
    if kind == "gemini":
        if importlib.util.find_spec("google.generativeai") is None:
            return (
                "Gemini provider requires `google-generativeai`. "
                "Install it in your Annolid environment, for example: "
                "`.venv/bin/pip install google-generativeai`."
            )
    return None


def _inject_openai_compat_env_defaults(
    *, settings: Dict[str, Any], provider_name: str, params: Dict[str, Any]
) -> Dict[str, Any]:
    out = dict(params or {})
    provider_key = str(provider_name or "").strip().lower()
    p_kind = provider_kind(settings, provider_key)
    if p_kind != "openai_compat":
        return out
    defs = provider_definitions(settings)
    spec = defs.get(provider_key, {})
    env_keys = spec.get("env_keys", [])
    if isinstance(env_keys, str):
        env_keys = [env_keys]
    if not isinstance(env_keys, list):
        env_keys = []

    if not str(out.get("api_key") or "").strip():
        for env_name in env_keys:
            value = str(os.getenv(str(env_name).strip()) or "").strip()
            if value:
                out["api_key"] = value
                break

    # Fallbacks for aliases/custom providers that may not define env_keys.
    if not str(out.get("api_key") or "").strip():
        for env_name in (
            "OPENAI_API_KEY",
            "OPENROUTER_API_KEY",
            "NVIDIA_API_KEY",
            "MOONSHOT_API_KEY",
        ):
            value = str(os.getenv(env_name) or "").strip()
            if value:
                out["api_key"] = value
                break

    if not str(out.get("base_url") or "").strip():
        env_base = str(spec.get("base_url_env") or "").strip()
        if env_base:
            value = str(os.getenv(env_base) or "").strip()
            if value:
                out["base_url"] = value
    if not str(out.get("base_url") or "").strip():
        default_base = str(spec.get("base_url_default") or "").strip()
        if default_base:
            out["base_url"] = default_base
    return out


def run_ollama_streaming_chat(
    *,
    prompt: str,
    image_path: str,
    model: str,
    settings: Dict[str, Any],
    load_history_messages: Callable[[], List[Dict[str, Any]]],
    emit_chunk: Callable[[str], None],
    emit_final: Callable[[str, bool], None],
    persist_turn: Callable[[str, str], None],
) -> None:
    ollama_module = importlib.import_module("ollama")

    host = str(settings.get("ollama", {}).get("host") or "").strip()
    prev_host_present = "OLLAMA_HOST" in os.environ
    prev_host_value = os.environ.get("OLLAMA_HOST")
    try:
        if host:
            os.environ["OLLAMA_HOST"] = host
        else:
            os.environ.pop("OLLAMA_HOST", None)

        messages = load_history_messages()
        user_message: Dict[str, Any] = {"role": "user", "content": prompt}
        if image_path and os.path.exists(image_path):
            user_message["images"] = [image_path]
        messages.append(user_message)

        stream = ollama_module.chat(
            model=model,
            messages=messages,
            stream=True,
        )
        full_response = ""
        for part in stream:
            if "message" in part and "content" in part["message"]:
                chunk = str(part["message"]["content"] or "")
                full_response += chunk
                emit_chunk(chunk)
            elif "error" in part:
                emit_final(f"Stream error: {part['error']}", True)
                return

        if not full_response.strip():
            emit_final("No response from Ollama.", True)
            return
        persist_turn(prompt, full_response)
        emit_final("", False)
    finally:
        if prev_host_present and prev_host_value is not None:
            os.environ["OLLAMA_HOST"] = prev_host_value
        else:
            os.environ.pop("OLLAMA_HOST", None)


def run_openai_compat_chat(
    *,
    prompt: str,
    image_path: str,
    model: str,
    provider_name: str,
    settings: Dict[str, Any],
    load_history_messages: Callable[[], List[Dict[str, Any]]],
    max_tokens: int = 4096,
    timeout_s: Optional[float] = None,
) -> Tuple[str, str]:
    provider_key = str(provider_name or "openai").strip().lower()
    provider_block = dict(settings.get(provider_key, {}) or {})
    provider_block = _inject_openai_compat_env_defaults(
        settings=settings,
        provider_name=provider_key,
        params=provider_block,
    )
    cfg = LLMConfig(
        provider=provider_key,
        model=model,
        params=provider_block,
    )
    resolved = resolve_openai_compat(cfg)
    provider = OpenAICompatProvider(resolved=resolved)

    user_prompt = str(prompt or "")
    messages = load_history_messages()
    user_content: Any = user_prompt
    if image_path and os.path.exists(image_path):
        mime, _ = mimetypes.guess_type(image_path)
        mime = str(mime or "").strip().lower()
        if mime.startswith("image/"):
            with open(image_path, "rb") as f:
                raw = base64.b64encode(f.read()).decode("utf-8")
            user_content = [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{raw}"},
                },
            ]
    messages.append({"role": "user", "content": user_content})

    async def _chat_once() -> str:
        model_lower = (model or "").lower()
        temperature = 0.7 if "gpt-5" not in model_lower else None
        coro = provider.chat(
            messages=messages,
            model=model,
            max_tokens=int(max_tokens),
            temperature=temperature,
            timeout_seconds=timeout_s,
        )
        if timeout_s is not None and float(timeout_s) > 0:
            resp = await asyncio.wait_for(coro, timeout=float(timeout_s))
        else:
            resp = await coro
        return str(resp.content or "")

    try:
        text = asyncio.run(_chat_once())
    except asyncio.TimeoutError as exc:
        limit = float(timeout_s) if timeout_s is not None else 0.0
        raise TimeoutError(
            f"Provider request timed out after {limit:.0f}s for {provider_key}:{model}."
        ) from exc
    return user_prompt, text


def run_gemini_chat(
    *,
    prompt: str,
    image_path: str,
    model: str,
    provider_name: str,
    settings: Dict[str, Any],
) -> Tuple[str, str]:
    try:
        import google.generativeai as genai  # type: ignore
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The 'google-generativeai' package is required for Gemini providers."
        ) from exc

    config = settings.get(provider_name, {})
    api_key = config.get("api_key")
    if not api_key:
        raise ValueError("Gemini API key is missing. Configure it in settings.")

    genai.configure(api_key=api_key)
    model_name = model or "gemini-1.5-flash"
    gm = genai.GenerativeModel(model_name)

    user_prompt = str(prompt or "")
    if image_path and os.path.exists(image_path):
        user_prompt += (
            f"\n\n[Note: Image context available at {image_path}. "
            "Use this visual context in your response.]"
        )

    result = gm.generate_content(user_prompt)
    text = str(getattr(result, "text", "") or "")
    return user_prompt, text


def ollama_plain_mode_remaining(model: str) -> int:
    return int(_OLLAMA_FORCE_PLAIN_CACHE.get(str(model or ""), 0) or 0)


def ollama_plain_mode_decrement(model: str) -> int:
    key = str(model or "")
    remaining = max(0, ollama_plain_mode_remaining(key) - 1)
    _OLLAMA_FORCE_PLAIN_CACHE[key] = remaining
    return remaining


def ollama_mark_plain_mode(
    model: str, turns: int = OLLAMA_PLAIN_MODE_COOLDOWN_TURNS
) -> None:
    _OLLAMA_FORCE_PLAIN_CACHE[str(model or "")] = max(0, int(turns))


def ollama_clear_plain_mode(model: str) -> None:
    _OLLAMA_FORCE_PLAIN_CACHE.pop(str(model or ""), None)


def recover_with_plain_ollama_reply(
    *,
    prompt: str,
    image_path: str,
    model: str,
    settings: Dict[str, Any],
    logger: Any,
    import_module: Optional[Callable[[str], Any]] = None,
    first_timeout_s: float = 12.0,
    nudge_timeout_s: float = 8.0,
) -> str:
    host = str(settings.get("ollama", {}).get("host") or "").strip()
    import_fn = import_module or importlib.import_module
    try:
        ollama_module = import_fn("ollama")
    except ImportError:
        return ""

    user_message: Dict[str, Any] = {
        "role": "user",
        "content": str(prompt or ""),
    }
    if image_path and os.path.exists(image_path):
        user_message["images"] = [image_path]

    def _run_stream_once(extra_nudge: bool = False) -> str:
        prev_host_present = "OLLAMA_HOST" in os.environ
        prev_host_value = os.environ.get("OLLAMA_HOST")
        try:
            if host:
                os.environ["OLLAMA_HOST"] = host
            else:
                os.environ.pop("OLLAMA_HOST", None)
            msgs = [user_message]
            if extra_nudge:
                msgs.append(
                    {
                        "role": "user",
                        "content": "Reply with plain text in one short paragraph.",
                    }
                )
            stream_iter = ollama_module.chat(
                model=model,
                messages=msgs,
                stream=True,
            )
            chunks: List[str] = []
            for part in stream_iter:
                if "message" in part and "content" in part["message"]:
                    chunk = str(part["message"]["content"] or "")
                    if chunk:
                        chunks.append(chunk)
            return "".join(chunks).strip()
        finally:
            if prev_host_present and prev_host_value is not None:
                os.environ["OLLAMA_HOST"] = prev_host_value
            else:
                os.environ.pop("OLLAMA_HOST", None)

    try:
        first_timeout = max(2.0, float(first_timeout_s))
        nudge_timeout = max(2.0, float(nudge_timeout_s))
        with ThreadPoolExecutor(max_workers=1) as executor:
            text_stream = executor.submit(_run_stream_once, False).result(
                timeout=first_timeout
            )
        logger.info(
            "annolid-bot plain ollama stream recovery model=%s content_chars=%d",
            model,
            len(text_stream),
        )
        if text_stream:
            return text_stream
        with ThreadPoolExecutor(max_workers=1) as executor:
            text_stream_nudge = executor.submit(_run_stream_once, True).result(
                timeout=nudge_timeout
            )
        logger.info(
            "annolid-bot plain ollama stream-nudge recovery model=%s content_chars=%d",
            model,
            len(text_stream_nudge),
        )
        return text_stream_nudge
    except FutureTimeoutError:
        logger.warning(
            "annolid-bot plain ollama recovery timed out model=%s",
            model,
        )
        return ""
    except Exception as exc:
        logger.warning(
            "annolid-bot plain ollama recovery failed model=%s error=%s",
            model,
            exc,
        )
        return ""


def build_ollama_llm_callable(
    *,
    prompt: str,
    settings: Dict[str, Any],
    parse_tool_calls: Callable[[Any], List[Dict[str, Any]]],
    normalize_messages: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
    extract_text: Callable[[Dict[str, Any]], str],
    prompt_may_need_tools: Callable[[str], bool],
    logger: Any,
    import_module: Optional[Callable[[str], Any]] = None,
    tool_request_timeout_s: float = 45.0,
    plain_request_timeout_s: float = 25.0,
) -> Callable[
    [List[Dict[str, Any]], List[Dict[str, Any]], str], Awaitable[Dict[str, Any]]
]:
    host = str(settings.get("ollama", {}).get("host") or "").strip()
    import_fn = import_module or importlib.import_module
    try:
        ollama_module = import_fn("ollama")
    except ImportError as exc:
        raise ImportError(
            "The python 'ollama' package is required for Ollama agent mode."
        ) from exc

    def _coerce_tool_calls(tool_calls_payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(tool_calls_payload, list):
            return []
        if tool_calls_payload and all(
            isinstance(item, dict) and "name" in item for item in tool_calls_payload
        ):
            return [dict(item) for item in tool_calls_payload if isinstance(item, dict)]
        return parse_tool_calls(tool_calls_payload)

    async def _call(
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        model_id: str,
    ) -> Dict[str, Any]:
        tool_timeout = max(5.0, float(tool_request_timeout_s))
        plain_timeout = max(5.0, float(plain_request_timeout_s))
        prepared = normalize_messages(messages)
        supports_tools = _OLLAMA_TOOL_SUPPORT_CACHE.get(model_id, True)
        if tools and not supports_tools and prompt_may_need_tools(prompt):
            supports_tools = True
            logger.info(
                "annolid-bot forcing tool reprobe for tool-intent prompt model=%s",
                model_id,
            )
        effective_tools = (
            [dict(t) for t in tools] if (tools and supports_tools) else None
        )
        logger.info(
            "annolid-bot ollama request model=%s effective_tools_sent=%d supports_tools=%s",
            model_id,
            len(effective_tools or []),
            supports_tools,
        )

        def _invoke_chat_stream(
            tools_payload: Optional[List[Dict[str, Any]]],
        ) -> Dict[str, Any]:
            prev_host_present = "OLLAMA_HOST" in os.environ
            prev_host_value = os.environ.get("OLLAMA_HOST")
            try:
                if host:
                    os.environ["OLLAMA_HOST"] = host
                else:
                    os.environ.pop("OLLAMA_HOST", None)
                stream_iter = ollama_module.chat(
                    model=model_id,
                    messages=prepared,
                    tools=tools_payload,
                    stream=True,
                )
                chunks: List[str] = []
                tool_calls_by_id: Dict[str, Dict[str, Any]] = {}
                done_reason = "stop"
                for part in stream_iter:
                    if not isinstance(part, dict):
                        continue
                    done_reason = str(part.get("done_reason") or done_reason)
                    msg = part.get("message")
                    if isinstance(msg, dict):
                        content = msg.get("content")
                        if isinstance(content, str) and content:
                            chunks.append(content)
                        raw_tool_calls = msg.get("tool_calls")
                        if raw_tool_calls:
                            for call in parse_tool_calls(raw_tool_calls):
                                call_id = str(
                                    call.get("id") or f"call_{len(tool_calls_by_id)}"
                                )
                                tool_calls_by_id[call_id] = call
                content = "".join(chunks).strip()
                tool_calls = list(tool_calls_by_id.values())
                return {
                    "done_reason": done_reason,
                    "message": {"content": content, "tool_calls": tool_calls},
                }
            finally:
                if prev_host_present and prev_host_value is not None:
                    os.environ["OLLAMA_HOST"] = prev_host_value
                else:
                    os.environ.pop("OLLAMA_HOST", None)

        try:
            response = await asyncio.wait_for(
                asyncio.to_thread(_invoke_chat_stream, effective_tools),
                tool_timeout if effective_tools is not None else plain_timeout,
            )
        except Exception as exc:
            msg = str(exc)
            if "400" in msg and effective_tools:
                logger.warning(
                    "annolid-bot ollama tool-call request rejected; retrying without tools model=%s error=%s",
                    model_id,
                    exc,
                )
                _OLLAMA_TOOL_SUPPORT_CACHE[model_id] = False
                response = await asyncio.wait_for(
                    asyncio.to_thread(_invoke_chat_stream, None), plain_timeout
                )
            else:
                raise
        msg = dict(response.get("message") or {})
        tool_calls = _coerce_tool_calls(msg.get("tool_calls"))
        content = extract_text(response)
        logger.info(
            "annolid-bot ollama raw response model=%s done_reason=%s content_chars=%d tool_calls=%d",
            model_id,
            str(response.get("done_reason") or ""),
            len(content),
            len(tool_calls),
        )
        if tool_calls:
            _OLLAMA_TOOL_SUPPORT_CACHE[model_id] = True
            ollama_clear_plain_mode(model_id)
        if not content.strip() and not tool_calls:
            if effective_tools is not None:
                logger.warning(
                    "annolid-bot ollama returned empty content with tools; retrying once without tools model=%s",
                    model_id,
                )
                _OLLAMA_TOOL_SUPPORT_CACHE[model_id] = False
                response2 = await asyncio.wait_for(
                    asyncio.to_thread(_invoke_chat_stream, None), plain_timeout
                )
                msg2 = dict(response2.get("message") or {})
                tool_calls2 = _coerce_tool_calls(msg2.get("tool_calls"))
                content2 = extract_text(response2)
                logger.info(
                    "annolid-bot ollama no-tools retry model=%s done_reason=%s content_chars=%d tool_calls=%d",
                    model_id,
                    str(response2.get("done_reason") or ""),
                    len(content2),
                    len(tool_calls2),
                )
                if content2.strip() or tool_calls2:
                    response = response2
                    tool_calls = tool_calls2
                    content = content2
            if not content.strip() and not tool_calls:
                logger.warning(
                    "annolid-bot ollama returned empty content (tools=%s) model=%s",
                    bool(effective_tools),
                    model_id,
                )
                if effective_tools is None:
                    ollama_mark_plain_mode(model_id)
        return {
            "content": content,
            "tool_calls": tool_calls,
            "finish_reason": str(response.get("done_reason") or "stop"),
            "usage": {},
            "reasoning_content": None,
        }

    return _call
