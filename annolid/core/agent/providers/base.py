from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from annolid.utils.logger import logger

from .call_runtime import (
    NON_RETRYABLE_QUOTA_MARKERS,
    RETRYABLE_ERROR_KINDS,
    RETRYABLE_STATUS_CODES,
    TRANSIENT_ERROR_MARKERS,
    classify_provider_exception,
    sanitize_provider_error,
)


@dataclass(frozen=True)
class ToolCallRequest:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class LLMResponse:
    content: Optional[str]
    tool_calls: List[ToolCallRequest] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: Dict[str, int] = field(default_factory=dict)
    reasoning_content: Optional[str] = None
    error_status_code: Optional[int] = None
    error_kind: Optional[str] = None
    error_type: Optional[str] = None
    error_code: Optional[str] = None
    error_retry_after_s: Optional[float] = None
    error_should_retry: Optional[bool] = None

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def error_response_from_exception(
    exc: Exception,
    *,
    prefix: str = "Error calling LLM",
) -> LLMResponse:
    """Convert an SDK/transport exception into a structured, redacted response."""
    details = classify_provider_exception(exc)
    return LLMResponse(
        content=f"{prefix}: {details.message}",
        finish_reason="error",
        error_status_code=details.status_code,
        error_kind=details.kind,
        error_type=details.error_type,
        error_code=details.error_code,
        error_retry_after_s=details.retry_after_s,
        error_should_retry=details.should_retry,
    )


class ProviderCallError(RuntimeError):
    """A failed provider call that retains retry classification without secrets."""

    def __init__(
        self,
        message: str,
        *,
        provider: str = "",
        model: str = "",
        status_code: Optional[int] = None,
        error_kind: Optional[str] = None,
        retry_after_s: Optional[float] = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(sanitize_provider_error(message))
        self.provider = str(provider or "")
        self.model = str(model or "")
        self.status_code = status_code
        self.error_kind = error_kind
        self.retry_after_s = retry_after_s
        self.retryable = bool(retryable)

    @classmethod
    def from_response(
        cls,
        response: LLMResponse,
        *,
        provider: str = "",
        model: str = "",
    ) -> ProviderCallError:
        detail = (
            str(response.content or "").strip() or "Model provider returned an error."
        )
        return cls(
            detail,
            provider=provider,
            model=model,
            status_code=response.error_status_code,
            error_kind=response.error_kind,
            retry_after_s=response.error_retry_after_s,
            retryable=LLMProvider.is_retryable_response(response),
        )


def raise_for_error_response(
    response: LLMResponse,
    *,
    provider: str = "",
    model: str = "",
) -> None:
    if str(response.finish_reason or "").strip().lower() == "error":
        raise ProviderCallError.from_response(
            response,
            provider=provider,
            model=model,
        )


class LLMProvider(ABC):
    """Abstract provider interface used by agent loops/subagents."""

    @abstractmethod
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
        raise NotImplementedError

    @classmethod
    def is_retryable_response(cls, response: LLMResponse) -> bool:
        """Return whether repeating the same provider request may succeed."""
        if response.error_should_retry is not None:
            return bool(response.error_should_retry)
        status = response.error_status_code
        if status is not None:
            if int(status) == 429 and any(
                marker in str(response.content or "").lower()
                for marker in NON_RETRYABLE_QUOTA_MARKERS
            ):
                return False
            if int(status) in RETRYABLE_STATUS_CODES or int(status) >= 500:
                return True
        kind = str(response.error_kind or "").strip().lower()
        if kind in RETRYABLE_ERROR_KINDS:
            return True
        if kind in {
            "authentication",
            "context_length",
            "invalid_request",
            "quota",
        }:
            return False
        text = str(response.content or "").lower()
        if any(marker in text for marker in NON_RETRYABLE_QUOTA_MARKERS):
            return False
        return any(marker in text for marker in TRANSIENT_ERROR_MARKERS)

    async def chat_with_retry(
        self,
        *,
        max_retries: int = 1,
        retry_delays: tuple[float, ...] = (0.5,),
        max_retry_after_s: float = 30.0,
        **chat_kwargs: Any,
    ) -> LLMResponse:
        """Call ``chat`` with one bounded retry for pre-output transient errors.

        Provider retries stop once streaming has emitted content, preventing
        duplicated partial answers. Long ``Retry-After`` windows are surfaced
        to the caller instead of blocking the GUI or a channel worker.
        """
        retries = max(0, int(max_retries))
        delays = tuple(max(0.0, float(item)) for item in retry_delays) or (0.0,)
        retry_after_cap = max(0.0, float(max_retry_after_s))
        emitted_content = False
        original_on_token = chat_kwargs.get("on_token")

        if callable(original_on_token):

            def _tracked_on_token(token: str) -> None:
                nonlocal emitted_content
                if token:
                    emitted_content = True
                original_on_token(token)

            chat_kwargs["on_token"] = _tracked_on_token

        for attempt in range(retries + 1):
            try:
                response = await self.chat(**chat_kwargs)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                response = error_response_from_exception(exc)

            if str(response.finish_reason or "").strip().lower() != "error":
                return response
            if emitted_content:
                response.error_should_retry = False
                return response
            if attempt >= retries or not self.is_retryable_response(response):
                return response

            retry_after = response.error_retry_after_s
            if retry_after is not None and retry_after > retry_after_cap:
                response.error_should_retry = False
                logger.warning(
                    "Model provider requested Retry-After %.1fs; skipping inline "
                    "retry because the bounded limit is %.1fs.",
                    retry_after,
                    retry_after_cap,
                )
                return response
            delay = (
                float(retry_after)
                if retry_after is not None
                else delays[min(attempt, len(delays) - 1)]
            )
            logger.warning(
                "Transient model call failure; retrying attempt %d/%d in %.1fs "
                "(kind=%s status=%s).",
                attempt + 1,
                retries,
                delay,
                response.error_kind or "unknown",
                response.error_status_code or "unknown",
            )
            if delay > 0:
                await asyncio.sleep(delay)

        return response

    @abstractmethod
    def get_default_model(self) -> str:
        raise NotImplementedError

    async def close(self) -> None:
        """Optional provider cleanup hook for long-lived async clients."""
        return None
