"""Infrastructure adapter for behavior-agent artifact persistence."""

from __future__ import annotations

from pathlib import Path

from annolid.services.behavior_agent.artifact_store import BehaviorAgentArtifactStore


class BehaviorAgentStoreAdapter(BehaviorAgentArtifactStore):
    def __init__(self, root_dir: str | Path) -> None:
        super().__init__(root_dir)


__all__ = ["BehaviorAgentStoreAdapter"]
