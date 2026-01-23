from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


@dataclass(frozen=True)
class ModelCapabilities:
    """Describe what a runtime model supports at inference time."""

    tasks: Tuple[str, ...] = ()
    input_modalities: Tuple[str, ...] = ()
    output_modalities: Tuple[str, ...] = ()
    streaming: bool = False


@dataclass(frozen=True)
class ModelRequest:
    """Generic request payload used by runtime model adapters."""

    task: str
    text: Optional[str] = None
    image: Any = None
    image_path: Optional[str] = None
    messages: Optional[Sequence[Mapping[str, Any]]] = None
    params: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelResponse:
    """Generic response payload returned by runtime model adapters."""

    task: str
    output: Any
    text: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)
    raw: Any = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "task": self.task,
            "output": self.output,
        }
        if self.text is not None:
            payload["text"] = self.text
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


class RuntimeModel(ABC):
    """Minimal runtime interface shared by LLM and local CV models."""

    @property
    @abstractmethod
    def model_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def capabilities(self) -> ModelCapabilities:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def predict(self, request: ModelRequest) -> ModelResponse:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def __enter__(self) -> "RuntimeModel":
        self.load()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.close()
