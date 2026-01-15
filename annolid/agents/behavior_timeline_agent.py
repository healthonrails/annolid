import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from annolid.data.videos import CV2Video
from annolid.agents.frame_search import cv2_to_pil, search_frames

from annolid.utils.logger import logger


@dataclass
class TimelineConfig:
    # Sampling and retrieval
    hz: int = 1  # frames per second to sample
    neighbors: int = 5  # top-K similar frames from LanceDB

    # Aggregation and smoothing
    max_behaviors_per_second: int = 3
    smooth_window: int = 3  # seconds; must be odd for symmetric window
    min_confidence: float = 0.15  # prune very weak behaviors


def aggregate_behaviors(neighbor_results: List[dict], max_behaviors: int) -> List[Tuple[str, float]]:
    """
    Aggregate flags from nearest neighbors into behavior scores.

    Returns list of (behavior, confidence) sorted by confidence desc.
    """
    vote: Dict[str, int] = {}
    total = 0
    for res in neighbor_results:
        flags = res.get('flags') or []
        if not flags:
            continue
        # Each neighbor contributes one vote per flag
        for f in flags:
            vote[f] = vote.get(f, 0) + 1
            total += 1

    if total == 0:
        return []

    scores = [(b, cnt / total) for b, cnt in vote.items()]
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:max_behaviors]


def smooth_timeline(raw: List[List[Tuple[str, float]]], window: int, min_conf: float) -> List[List[Tuple[str, float]]]:
    """
    Simple temporal smoothing: within a sliding window, average confidences of same behavior.
    Returns a new list of per-second behavior/confidence lists.
    """
    if window <= 1:
        # Just prune by min_confidence and return
        return [[(b, c) for (b, c) in second if c >= min_conf] for second in raw]

    half = window // 2
    T = len(raw)
    smoothed: List[List[Tuple[str, float]]] = []
    for t in range(T):
        start = max(0, t - half)
        end = min(T, t + half + 1)
        accum: Dict[str, List[float]] = {}
        for u in range(start, end):
            for b, c in raw[u]:
                accum.setdefault(b, []).append(c)
        avg = [(b, float(np.mean(cs))) for b, cs in accum.items()]
        avg = [(b, c) for (b, c) in avg if c >= min_conf]
        avg.sort(key=lambda x: x[1], reverse=True)
        smoothed.append(avg)
    return smoothed


def build_timeline(video_path: str, config: Optional[TimelineConfig] = None) -> List[Dict]:
    """Produce a per-second behavior timeline using LanceDB retrieval and smoothing."""
    cfg = config or TimelineConfig()
    video = CV2Video(video_path)
    fps = max(1, int(round(video.fps())))
    total = video.total_frames()
    duration_sec = int(np.ceil(total / fps))

    step = max(1, int(round(fps / cfg.hz)))
    logger.info(
        f"Video fps={fps}, frames={total}, seconds≈{duration_sec}, sampling every {step} frames (~{cfg.hz} Hz)")

    raw_per_second: List[List[Tuple[str, float]]] = []
    neighbor_cache: List[List[dict]] = []

    # Sample one frame per second using frame index i*fps
    for sec in range(duration_sec):
        frame_idx = min(sec * fps, max(0, total - 1))
        frame = video.load_frame(frame_idx)
        if frame is None:
            logger.warning(
                f"Failed to load frame at second={sec} (index={frame_idx})")
            raw_per_second.append([])
            neighbor_cache.append([])
            continue

        pil_img = cv2_to_pil(frame)
        results = search_frames(pil_img, limit=cfg.neighbors)
        neighbor_cache.append(results)
        agg = aggregate_behaviors(results, cfg.max_behaviors_per_second)
        raw_per_second.append(agg)

    smoothed = smooth_timeline(
        raw_per_second, cfg.smooth_window, cfg.min_confidence)

    # Build final timeline entries
    timeline: List[Dict] = []
    for sec, (behaviors, neighbors) in enumerate(zip(smoothed, neighbor_cache)):
        timeline.append(
            {
                "second": sec,
                "behaviors": [{"label": b, "confidence": round(c, 3)} for b, c in behaviors],
                "neighbors": neighbors,  # includes image_uri/flags/caption
            }
        )
    return timeline


def save_timeline_json(timeline: List[Dict], out_path: str) -> None:
    with open(out_path, 'w') as f:
        json.dump(timeline, f, indent=2)
    logger.info(f"Saved timeline to {out_path}")


def print_human_summary(timeline: List[Dict], top_n: int = 1) -> None:
    for entry in timeline:
        sec = entry['second']
        behaviors = entry.get('behaviors', [])[:top_n]
        if not behaviors:
            print(f"{sec:5d}s: (no confident behavior)")
        else:
            desc = ", ".join(
                [f"{b['label']} ({b['confidence']:.2f})" for b in behaviors])
            print(f"{sec:5d}s: {desc}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Build a per-second behavior timeline using LanceDB retrieval + smoothing.")
    parser.add_argument("video", help="Path to the video file")
    parser.add_argument("--out", default="timeline.json",
                        help="Output JSON path")
    parser.add_argument("--hz", type=int, default=1,
                        help="Sampling rate (frames per second)")
    parser.add_argument("--neighbors", type=int, default=5,
                        help="Top-K nearest frames to aggregate")
    parser.add_argument("--window", type=int, default=3,
                        help="Smoothing window (seconds, odd number)")
    parser.add_argument("--max-per-sec", type=int, default=3,
                        help="Max behaviors per second")
    parser.add_argument("--min-conf", type=float, default=0.15,
                        help="Minimum confidence to keep behavior")
    parser.add_argument("--no-print", action="store_true",
                        help="Do not print human summary to stdout")

    args = parser.parse_args()

    if args.window % 2 == 0:
        logger.warning("--window should be odd; incrementing by 1")
        args.window += 1

    cfg = TimelineConfig(
        hz=args.hz,
        neighbors=args.neighbors,
        max_behaviors_per_second=args.max_per_sec,
        smooth_window=args.window,
        min_confidence=args.min_conf,
    )

    timeline = build_timeline(args.video, cfg)
    save_timeline_json(timeline, args.out)
    if not args.no_print:
        print_human_summary(timeline)


if __name__ == "__main__":
    main()
