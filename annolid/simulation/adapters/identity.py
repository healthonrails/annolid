from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from annolid.simulation.types import (
    Pose2DFrame,
    SimulationAdapter,
    SimulationFrameResult,
    SimulationRunResult,
)


class IdentitySimulationAdapter(SimulationAdapter):
    """Lightweight backend for validating simulation IO and mappings."""

    name = "identity"

    def __init__(self) -> None:
        self._config: Dict[str, Any] = {}

    def configure(self, config: Mapping[str, Any]) -> None:
        self._config = dict(config)

    def initialize(self) -> None:
        return None

    def fit_2d(self, observations: Sequence[Pose2DFrame]) -> SimulationRunResult:
        keypoint_to_site = dict(self._config.get("keypoint_to_site") or {})
        frames = []
        for frame in observations:
            mapped_sites = {
                keypoint_to_site.get(label, label): [point[0], point[1], 0.0]
                for label, point in frame.points.items()
            }
            frames.append(
                SimulationFrameResult(
                    frame_index=frame.frame_index,
                    timestamp_sec=frame.timestamp_sec,
                    state={
                        "backend": self.name,
                        "site_targets": mapped_sites,
                    },
                    diagnostics={
                        "input_points": len(frame.points),
                        "mapped_sites": len(mapped_sites),
                    },
                )
            )
        return SimulationRunResult(
            frames=frames,
            metadata={"backend": self.name, "frames": len(frames)},
        )
