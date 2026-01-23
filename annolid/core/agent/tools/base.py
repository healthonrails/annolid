from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generic, Iterable, Optional, Protocol, Sequence, TypeVar

from annolid.core.types import FrameRef, Geometry, RLEGeometry


class ToolError(RuntimeError):
    """Raised when a tool fails in a recoverable way."""


class CancellationToken(Protocol):
    """Minimal cancellation primitive shared by GUI/CLI."""

    def is_set(self) -> bool:  # pragma: no cover - protocol
        raise NotImplementedError


class ArtifactStore(Protocol):
    """Phase 3 artifact persistence interface (implemented in later steps)."""

    def resolve(self, *parts: str) -> Path:  # pragma: no cover - protocol
        raise NotImplementedError


@dataclass(frozen=True)
class ToolContext:
    """Execution context passed to each tool.

    The intent is to keep this GUI-free and lightweight while carrying enough
    shared metadata to enable modular pipelines (Phase 4).
    """

    video_path: Path
    results_dir: Path
    run_id: str
    fps: Optional[float] = None
    total_frames: Optional[int] = None
    artifact_store: Optional[ArtifactStore] = None
    cancel_token: Optional[CancellationToken] = None
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger("annolid.agent")
    )
    meta: Dict[str, object] = field(default_factory=dict)

    @property
    def video_name(self) -> str:
        return self.video_path.name

    def cancelled(self) -> bool:
        token = self.cancel_token
        if token is None:
            return False
        try:
            return bool(token.is_set())
        except Exception:
            return False


@dataclass(frozen=True)
class FrameData:
    """A frame reference with optional decoded pixels/paths.

    Tools should prefer using `ref` for stable indexing/timestamps and treat
    `image_rgb`/`image_path` as optional conveniences.
    """

    ref: FrameRef
    image_rgb: Any = None
    image_path: Optional[Path] = None
    meta: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class FrameBatch:
    frames: Sequence[FrameData]

    def __iter__(self) -> Iterable[FrameData]:
        return iter(self.frames)


@dataclass(frozen=True)
class Instance:
    """Generic per-frame instance output.

    This is intentionally flexible enough to represent detections, segmentations,
    or tracked observations. Stable track IDs become first-class in `Track`
    objects (see `annolid.core.types.tracking`), which tools can emit later.
    """

    frame: FrameRef
    geometry: Geometry
    label: Optional[str] = None
    score: Optional[float] = None
    mask: Optional[RLEGeometry] = None
    instance_id: Optional[str] = None
    track_id: Optional[str] = None
    meta: Dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "frame": self.frame.to_dict(),
            "geometry": self.geometry.to_dict(),
        }
        if self.label is not None:
            payload["label"] = str(self.label)
        if self.score is not None:
            payload["score"] = float(self.score)
        if self.mask is not None:
            payload["mask"] = self.mask.to_dict()
        if self.instance_id is not None:
            payload["instance_id"] = str(self.instance_id)
        if self.track_id is not None:
            payload["track_id"] = str(self.track_id)
        if self.meta:
            payload["meta"] = dict(self.meta)
        return payload


@dataclass(frozen=True)
class Instances:
    frame: FrameRef
    instances: Sequence[Instance]
    meta: Dict[str, object] = field(default_factory=dict)


ToolInput = TypeVar("ToolInput")
ToolOutput = TypeVar("ToolOutput")


class Tool(ABC, Generic[ToolInput, ToolOutput]):
    """Base class for pluggable tools."""

    name: str = "tool"

    def __init__(self, *, config: Optional[Dict[str, object]] = None) -> None:
        self.config = dict(config or {})

    @abstractmethod
    def run(self, ctx: ToolContext, payload: ToolInput) -> ToolOutput:
        raise NotImplementedError

    def run_many(
        self, ctx: ToolContext, items: Iterable[ToolInput]
    ) -> Sequence[ToolOutput]:
        outputs: list[ToolOutput] = []
        for item in items:
            if ctx.cancelled():
                break
            outputs.append(self.run(ctx, item))
        return outputs
