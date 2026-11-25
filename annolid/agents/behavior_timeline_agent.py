"""Behavior timeline construction via ONNX EmbeddingGemma 2 retrieval or scoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union

import numpy as np

from annolid.agents.frame_search import cv2_to_pil, search_frames
from annolid.core.media.video import CV2Video
from annolid.services.embedding_search import (
    DEFAULT_SEARCH_BACKEND,
    FrameEmbeddingBackend,
    score_video_behaviors,
)
from annolid.utils.logger import logger


@dataclass
class TimelineConfig:
    # Sampling and retrieval
    hz: int = 1  # samples per second
    neighbors: int = 5  # top-K similar frames from index (retrieval mode)

    # Aggregation and smoothing
    max_behaviors_per_second: int = 3
    smooth_window: int = 3  # samples; odd for symmetric window
    min_confidence: float = 0.15  # prune very weak behaviors

    # Optional ONNX EmbeddingGemma 2 zero-shot text behavior scoring
    behaviors: Optional[Tuple[str, ...]] = None
    backend: str = DEFAULT_SEARCH_BACKEND
    quantization: str = "q4"
    matryoshka_dim: Optional[int] = None
    clip_frames: int = 1
    index_path: Optional[str] = None


def aggregate_behaviors(
    neighbor_results: List[dict], max_behaviors: int
) -> List[Tuple[str, float]]:
    """Aggregate flags from nearest neighbors into normalized behavior votes."""
    vote: Dict[str, int] = {}
    total = 0
    for res in neighbor_results:
        flags = res.get("flags") or []
        if not flags:
            continue
        for f in flags:
            vote[f] = vote.get(f, 0) + 1
            total += 1

    if total == 0:
        return []

    scores = [(b, cnt / total) for b, cnt in vote.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:max_behaviors]


def smooth_timeline(
    raw: List[List[Tuple[str, float]]], window: int, min_conf: float
) -> List[List[Tuple[str, float]]]:
    """Temporal sliding-window smoothing of per-sample behavior confidences."""
    if window <= 1:
        return [[(b, c) for (b, c) in sample if c >= min_conf] for sample in raw]

    half = window // 2
    total_steps = len(raw)
    smoothed: List[List[Tuple[str, float]]] = []
    for t in range(total_steps):
        start = max(0, t - half)
        end = min(total_steps, t + half + 1)
        accum: Dict[str, List[float]] = {}
        for u in range(start, end):
            for b, c in raw[u]:
                accum.setdefault(b, []).append(c)
        avg = [(b, float(np.mean(cs))) for b, cs in accum.items()]
        avg = [(b, c) for (b, c) in avg if c >= min_conf]
        avg.sort(key=lambda x: x[1], reverse=True)
        smoothed.append(avg)
    return smoothed


def build_timeline(
    video_path: str,
    config: Optional[TimelineConfig] = None,
    *,
    backend: Optional[Union[str, FrameEmbeddingBackend]] = None,
    backend_params: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Produce a temporally smoothed behavior timeline for ``video_path``.

    Supports two modes:
    1. **Zero-shot ONNX EmbeddingGemma 2 scoring** when ``config.behaviors`` is
       provided (scores sampled frames/clips directly against text descriptions).
    2. **Exemplar retrieval** against an indexed frame store (``index_path`` or
       default index) when ``config.behaviors`` is omitted.
    """
    cfg = config or TimelineConfig()
    hz = max(1, int(cfg.hz))

    video = CV2Video(video_path)
    try:
        fps_raw = float(video.fps() or 0.0)
        fps = fps_raw if fps_raw > 0 else 1.0
        total = int(video.total_frames())
    finally:
        video.release()

    if total <= 0:
        return []

    step = max(1, int(round(fps / float(hz))))
    duration_sec = int(np.ceil(total / fps))
    logger.info(
        f"Video fps={fps:.2f}, frames={total}, seconds≈{duration_sec}, "
        f"sampling every {step} frames (~{hz} Hz)"
    )

    resolved_backend: Union[str, FrameEmbeddingBackend] = (
        backend if backend is not None else cfg.backend
    )
    resolved_params: Dict[str, Any] = dict(backend_params or {})
    if not isinstance(resolved_backend, FrameEmbeddingBackend):
        resolved_params.setdefault("quantization", cfg.quantization)
        if cfg.matryoshka_dim is not None:
            resolved_params.setdefault("matryoshka_dim", int(cfg.matryoshka_dim))

    if cfg.behaviors:
        scored = score_video_behaviors(
            video_path,
            list(cfg.behaviors),
            backend=resolved_backend,
            backend_params=resolved_params,
            stride=step,
            clip_frames=max(1, int(cfg.clip_frames)),
            top_n=max(1, int(cfg.max_behaviors_per_second)),
            min_similarity=-1.0,
        )
        rows = scored.get("frames") or []
        raw_samples: List[List[Tuple[str, float]]] = [
            [
                (str(item["label"]), float(item["similarity"]))
                for item in (row.get("behaviors") or [])
            ]
            for row in rows
        ]
        smoothed = smooth_timeline(raw_samples, cfg.smooth_window, cfg.min_confidence)
        timeline: List[Dict[str, Any]] = []
        for idx, (row, behaviors) in enumerate(zip(rows, smoothed)):
            frame_idx = int(row["frame_index"])
            sec_val: Union[int, float] = idx if hz == 1 else round(frame_idx / fps, 3)
            timeline.append(
                {
                    "second": sec_val,
                    "frame_index": frame_idx,
                    "behaviors": [
                        {"label": b, "confidence": round(float(c), 3)}
                        for b, c in behaviors[: cfg.max_behaviors_per_second]
                    ],
                    "neighbors": [],
                }
            )
        return timeline

    # Exemplar retrieval mode
    sample_indices = list(range(0, total, step))
    raw_per_sample: List[List[Tuple[str, float]]] = []
    neighbor_cache: List[List[dict]] = []

    video = CV2Video(video_path)
    try:
        for frame_idx in sample_indices:
            frame = video.load_frame(frame_idx)
            if frame is None:
                logger.warning(f"Failed to load frame index={frame_idx}")
                raw_per_sample.append([])
                neighbor_cache.append([])
                continue
            pil_img = cv2_to_pil(frame)
            results = search_frames(
                pil_img,
                limit=cfg.neighbors,
                index_path=cfg.index_path,
                backend=resolved_backend,
                backend_params=resolved_params
                if not isinstance(resolved_backend, FrameEmbeddingBackend)
                else None,
            )
            neighbor_cache.append(results)
            agg = aggregate_behaviors(results, cfg.max_behaviors_per_second)
            raw_per_sample.append(agg)
    finally:
        video.release()

    smoothed = smooth_timeline(raw_per_sample, cfg.smooth_window, cfg.min_confidence)
    timeline = []
    for idx, (frame_idx, behaviors, neighbors) in enumerate(
        zip(sample_indices, smoothed, neighbor_cache)
    ):
        sec_val = idx if hz == 1 else round(frame_idx / fps, 3)
        timeline.append(
            {
                "second": sec_val,
                "frame_index": int(frame_idx),
                "behaviors": [
                    {"label": b, "confidence": round(float(c), 3)} for b, c in behaviors
                ],
                "neighbors": neighbors,
            }
        )
    return timeline


def save_timeline_json(timeline: List[Dict[str, Any]], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(timeline, f, indent=2)
    logger.info(f"Saved timeline to {out_path}")


def print_human_summary(timeline: List[Dict[str, Any]], top_n: int = 1) -> None:
    for entry in timeline:
        sec = entry["second"]
        behaviors = entry.get("behaviors", [])[:top_n]
        sec_label = f"{sec:5d}s" if isinstance(sec, int) else f"{float(sec):6.2f}s"
        if not behaviors:
            print(f"{sec_label}: (no confident behavior)")
        else:
            desc = ", ".join(
                [f"{b['label']} ({b['confidence']:.2f})" for b in behaviors]
            )
            print(f"{sec_label}: {desc}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Build a behavior timeline using ONNX EmbeddingGemma 2 zero-shot scoring "
            "or exemplar retrieval + temporal smoothing."
        )
    )
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--out", default="timeline.json", help="Output JSON path")
    parser.add_argument(
        "--hz", type=int, default=1, help="Sampling rate (samples per second)"
    )
    parser.add_argument(
        "--behaviors",
        default=None,
        help="Comma-separated behavior descriptions for zero-shot ONNX scoring.",
    )
    parser.add_argument(
        "--index",
        default=None,
        help="Optional .npz frame index path for exemplar retrieval mode.",
    )
    parser.add_argument(
        "--neighbors", type=int, default=5, help="Top-K nearest frames to aggregate"
    )
    parser.add_argument(
        "--window", type=int, default=3, help="Smoothing window (odd number of samples)"
    )
    parser.add_argument(
        "--max-per-sec", type=int, default=3, help="Max behaviors per sample"
    )
    parser.add_argument(
        "--min-conf",
        type=float,
        default=0.15,
        help="Minimum confidence/similarity to keep behavior",
    )
    parser.add_argument(
        "--clip-frames",
        type=int,
        default=1,
        help="Frames per temporal window (1 for frame-level, 4/8 for clip-level).",
    )
    parser.add_argument(
        "--quantization",
        default="q4",
        choices=("q4", "quantized", "q4f16", "fp32", "fp16"),
    )
    parser.add_argument(
        "--matryoshka-dim",
        type=int,
        default=None,
        choices=(128, 256, 512, 768),
    )
    parser.add_argument(
        "--no-print", action="store_true", help="Do not print human summary to stdout"
    )

    args = parser.parse_args()

    if args.window % 2 == 0:
        logger.warning("--window should be odd; incrementing by 1")
        args.window += 1

    behaviors: Optional[Tuple[str, ...]] = None
    if args.behaviors:
        parsed = tuple(b.strip() for b in str(args.behaviors).split(",") if b.strip())
        if parsed:
            behaviors = parsed

    cfg = TimelineConfig(
        hz=args.hz,
        neighbors=args.neighbors,
        max_behaviors_per_second=args.max_per_sec,
        smooth_window=args.window,
        min_confidence=args.min_conf,
        behaviors=behaviors,
        quantization=args.quantization,
        matryoshka_dim=args.matryoshka_dim,
        clip_frames=args.clip_frames,
        index_path=args.index,
    )

    timeline = build_timeline(args.video, cfg)
    save_timeline_json(timeline, args.out)
    if not args.no_print:
        print_human_summary(timeline)


if __name__ == "__main__":
    main()
