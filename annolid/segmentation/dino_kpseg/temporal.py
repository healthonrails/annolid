from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class DinoKPSEGTemporalFusionConfig:
    """Temporal memory fusion for keypoint probability maps.

    Inspired by encoder-only query propagation designs: reuse previous-frame
    keypoint evidence as a lightweight memory prior, especially for low-confidence
    channels, while keeping current-frame evidence dominant.
    """

    enabled: bool = False
    alpha: float = 0.25
    low_conf_threshold: float = 0.35
    max_instances: int = 64


@dataclass
class _KeypointTrackState:
    keypoints_xy: List[Tuple[float, float]] = field(default_factory=list)
    keypoint_scores: List[float] = field(default_factory=list)


class DinoKPSEGTemporalStateStore:
    """State container for per-instance and global keypoint tracks."""

    def __init__(self) -> None:
        self._global: Optional[_KeypointTrackState] = None
        self._by_instance: Dict[int, _KeypointTrackState] = {}

    def reset(self) -> None:
        self._global = None
        self._by_instance = {}

    def get(
        self,
        *,
        instance_id: Optional[int],
    ) -> Tuple[Optional[List[Tuple[float, float]]], Optional[List[float]]]:
        if instance_id is not None:
            state = self._by_instance.get(int(instance_id))
        else:
            state = self._global
        if state is None:
            return None, None
        return list(state.keypoints_xy), list(state.keypoint_scores)

    def set(
        self,
        *,
        instance_id: Optional[int],
        keypoints_xy: Sequence[Sequence[float]],
        keypoint_scores: Sequence[float],
    ) -> None:
        coords = [(float(x), float(y)) for x, y in keypoints_xy]
        scores = [float(s) for s in keypoint_scores]
        state = _KeypointTrackState(keypoints_xy=coords, keypoint_scores=scores)
        if instance_id is None:
            self._global = state
            return
        self._by_instance[int(instance_id)] = state

    def prune_to_max_instances(self, *, max_instances: int) -> None:
        cap = max(1, int(max_instances))
        if len(self._by_instance) <= cap:
            return
        # Keep most recently inserted entries (dict order is insertion order).
        overflow = len(self._by_instance) - cap
        for key in list(self._by_instance.keys())[:overflow]:
            self._by_instance.pop(key, None)


class DinoKPSEGTemporalMemory:
    """Probability-memory fusion keyed by instance id."""

    def __init__(self, config: Optional[DinoKPSEGTemporalFusionConfig] = None) -> None:
        self.config = config or DinoKPSEGTemporalFusionConfig()
        self._global_probs: Optional[torch.Tensor] = None
        self._by_instance_probs: Dict[int, torch.Tensor] = {}

    def reset(self) -> None:
        self._global_probs = None
        self._by_instance_probs = {}

    def _get_prev(self, instance_id: Optional[int]) -> Optional[torch.Tensor]:
        if instance_id is not None:
            return self._by_instance_probs.get(int(instance_id))
        return self._global_probs

    def _set_prev(self, probs: torch.Tensor, *, instance_id: Optional[int]) -> None:
        cached = probs.detach().to("cpu", copy=True)
        if instance_id is None:
            self._global_probs = cached
            return
        self._by_instance_probs[int(instance_id)] = cached
        cap = max(1, int(self.config.max_instances))
        if len(self._by_instance_probs) > cap:
            overflow = len(self._by_instance_probs) - cap
            for key in list(self._by_instance_probs.keys())[:overflow]:
                self._by_instance_probs.pop(key, None)

    def fuse(
        self,
        probs: torch.Tensor,
        *,
        instance_id: Optional[int],
        current_scores: Sequence[float],
    ) -> torch.Tensor:
        if not bool(self.config.enabled):
            return probs
        prev = self._get_prev(instance_id)
        if prev is None:
            return probs
        if tuple(prev.shape) != tuple(probs.shape):
            return probs

        alpha = float(self.config.alpha)
        if alpha <= 0:
            return probs
        alpha = min(alpha, 0.95)

        low_thr = float(self.config.low_conf_threshold)
        scores = torch.as_tensor(
            [float(v) for v in current_scores],
            dtype=probs.dtype,
            device=probs.device,
        )
        if scores.ndim != 1 or scores.shape[0] != probs.shape[0]:
            return probs

        # Fuse only low-confidence channels; high-confidence channels stay frame-driven.
        gate = (scores < low_thr).to(dtype=probs.dtype)[:, None, None]
        blended = (1.0 - alpha) * probs + alpha * prev.to(
            dtype=probs.dtype, device=probs.device
        )
        return (gate * blended) + ((1.0 - gate) * probs)

    def update(self, probs: torch.Tensor, *, instance_id: Optional[int]) -> None:
        if not bool(self.config.enabled):
            return
        self._set_prev(probs, instance_id=instance_id)
