from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


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
    ) -> LLMResponse:
        raise NotImplementedError

    @abstractmethod
    def get_default_model(self) -> str:
        raise NotImplementedError
