from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Optional

import torch

try:
    import resource  # type: ignore
except Exception:  # pragma: no cover - platform-specific import
    resource = None  # type: ignore


def _process_rss_mb() -> float:
    if resource is None:
        return 0.0
    rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # macOS reports bytes; Linux reports KiB.
    if platform.system().lower() == "darwin":
        return rss / (1024.0 * 1024.0)
    return rss / 1024.0


def memory_snapshot() -> Dict[str, float]:
    payload: Dict[str, float] = {
        "rss_mb": float(_process_rss_mb()),
    }
    if torch.cuda.is_available():
        try:
            payload["mem_gpu_allocated"] = float(torch.cuda.memory_allocated())
            payload["mem_gpu_reserved"] = float(torch.cuda.memory_reserved())
        except Exception:
            pass
    return payload


def build_window_telemetry_entry(
    *,
    window_index: int,
    window_start_idx: int,
    window_end_idx: int,
    local_mask_counts: Mapping[int, int],
    boundary_empty_skips: int = 0,
    latency_ms: float = 0.0,
    reacquired_frames: int = 0,
) -> Dict[str, object]:
    covered_frames = sorted(int(k) for k in local_mask_counts.keys())
    frames_in_window = len(covered_frames)
    zero_mask_frames = sum(
        1 for f in covered_frames if int(local_mask_counts.get(int(f), 0)) <= 0
    )
    nonzero_frames = max(0, frames_in_window - zero_mask_frames)
    dropped_rate = (
        float(zero_mask_frames) / float(frames_in_window)
        if frames_in_window > 0
        else 0.0
    )
    return {
        "window_index": int(window_index),
        "start": int(window_start_idx),
        "end": int(window_end_idx),
        "frames": int(frames_in_window),
        "nonzero_frames": int(nonzero_frames),
        "zero_mask_frames": int(zero_mask_frames),
        "dropped_mask_rate": float(dropped_rate),
        "boundary_empty_skips": int(boundary_empty_skips),
        "latency_ms": float(latency_ms),
        "reacquired_frames": int(reacquired_frames),
    }


def build_config_snapshot(config: Mapping[str, object]) -> Dict[str, object]:
    keys = (
        "max_num_objects",
        "multiplex_count",
        "sliding_window_size",
        "sliding_window_stride",
        "use_sliding_window_for_text_prompt",
        "use_explicit_window_reseed",
        "allow_private_state_mutation",
        "offload_video_to_cpu",
        "async_loading_frames",
        "compile_model",
        "device",
        "checkpoint_path",
        "propagation_direction",
    )
    return {str(k): config.get(str(k)) for k in keys}


@dataclass
class Sam3TelemetrySink:
    jsonl_path: Optional[Path] = None

    def emit(self, event_type: str, payload: Mapping[str, object]) -> None:
        if self.jsonl_path is None:
            return
        row: Dict[str, object] = {
            "ts": float(time.time()),
            "event": str(event_type),
        }
        row.update(memory_snapshot())
        row.update(dict(payload or {}))
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        with self.jsonl_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
