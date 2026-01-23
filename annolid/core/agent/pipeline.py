from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence


@dataclass(frozen=True)
class AgentPipelineConfig:
    """Pipeline-level configuration for modular agent orchestration."""

    sampler: str = "stride"
    stride: int = 1
    target_fps: Optional[float] = None
    random_count: Optional[int] = None
    random_seed: Optional[int] = None
    random_replace: bool = False
    random_include_ends: bool = False
    motion_threshold: Optional[float] = None
    motion_min_step: int = 1

    tool_sequence: Sequence[str] = field(default_factory=tuple)
    behavior_params: Dict[str, object] = field(default_factory=dict)

    embedding_enabled: bool = False
    embedding_top_k: int = 5
    embedding_threshold: float = 0.0

    write_interval_frames: int = 1
    write_interval_seconds: Optional[float] = None

    fail_fast: bool = False
    skip_on_error: bool = True
