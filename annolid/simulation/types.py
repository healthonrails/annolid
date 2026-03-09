from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


Point2D = Tuple[float, float]
Point3D = Tuple[float, float, float]


@dataclass(frozen=True)
class Pose2DFrame:
    frame_index: int
    image_height: int
    image_width: int
    video_name: str
    image_path: str = ""
    timestamp_sec: Optional[float] = None
    points: Dict[str, Point2D] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    instances: Dict[str, str] = field(default_factory=dict)
    source_record: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Pose3DFrame:
    frame_index: int
    video_name: str
    timestamp_sec: Optional[float] = None
    points: Dict[str, Point3D] = field(default_factory=dict)
    scores: Dict[str, float] = field(default_factory=dict)
    source_record: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationFrameResult:
    frame_index: int
    state: Dict[str, Any]
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    timestamp_sec: Optional[float] = None


@dataclass(frozen=True)
class SimulationRunResult:
    frames: list[SimulationFrameResult]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SimulationMapping:
    backend: str
    keypoint_to_site: Dict[str, str] = field(default_factory=dict)
    site_to_joint: Dict[str, str] = field(default_factory=dict)
    coordinate_system: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class SimulationAdapter(ABC):
    name: str = ""

    @abstractmethod
    def configure(self, config: Mapping[str, Any]) -> None:
        """Accept backend-specific configuration."""

    @abstractmethod
    def initialize(self) -> None:
        """Prepare runtime resources."""

    def fit_2d(self, observations: Sequence[Pose2DFrame]) -> SimulationRunResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement 2D fitting"
        )

    def fit_3d(self, observations: Sequence[Pose3DFrame]) -> SimulationRunResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement 3D fitting"
        )

    def rollout(
        self, initial_state: Mapping[str, Any], steps: int
    ) -> SimulationRunResult:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement rollout"
        )
