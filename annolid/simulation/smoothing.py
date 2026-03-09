from __future__ import annotations

from dataclasses import replace
from typing import Dict, Sequence

from annolid.simulation.types import Pose2DFrame
from annolid.tracking.kpseg_smoothing import KeypointSmoother


def smooth_pose_frames(
    pose_frames: Sequence[Pose2DFrame],
    *,
    mode: str = "none",
    fps: float = 30.0,
    max_gap_frames: int = 0,
    min_score: float = 0.0,
    ema_alpha: float = 0.7,
    one_euro_min_cutoff: float = 1.0,
    one_euro_beta: float = 0.0,
    one_euro_d_cutoff: float = 1.0,
    kalman_process_noise: float = 1e-2,
    kalman_measurement_noise: float = 1e-1,
) -> list[Pose2DFrame]:
    ordered_frames = sorted(pose_frames, key=lambda item: int(item.frame_index))
    if not ordered_frames:
        return []

    mutable = [
        {
            "points": dict(frame.points),
            "scores": dict(frame.scores),
            "instances": dict(frame.instances),
            "frame": frame,
        }
        for frame in ordered_frames
    ]
    frame_index_to_pos = {
        int(item["frame"].frame_index): pos for pos, item in enumerate(mutable)
    }
    labels = _collect_labels(ordered_frames)
    if max_gap_frames > 0:
        for label in labels:
            _fill_label_gaps(
                mutable,
                frame_index_to_pos=frame_index_to_pos,
                label=label,
                max_gap_frames=int(max_gap_frames),
            )

    if str(mode or "none").strip().lower() != "none":
        smoother = KeypointSmoother(
            mode=mode,
            fps=float(fps),
            ema_alpha=float(ema_alpha),
            min_score=float(min_score),
            one_euro_min_cutoff=float(one_euro_min_cutoff),
            one_euro_beta=float(one_euro_beta),
            one_euro_d_cutoff=float(one_euro_d_cutoff),
            kalman_process_noise=float(kalman_process_noise),
            kalman_measurement_noise=float(kalman_measurement_noise),
        )
        for label in labels:
            for item in mutable:
                if label not in item["points"]:
                    continue
                coord = item["points"][label]
                score = float(item["scores"].get(label, 1.0))
                item["points"][label] = smoother.smooth(
                    label,
                    coord,
                    score=score,
                    mask_ok=True,
                )

    return [
        replace(
            item["frame"],
            points=dict(item["points"]),
            scores=dict(item["scores"]),
            instances=dict(item["instances"]),
        )
        for item in mutable
    ]


def _collect_labels(pose_frames: Sequence[Pose2DFrame]) -> list[str]:
    labels: list[str] = []
    seen: set[str] = set()
    for frame in pose_frames:
        for label in frame.points:
            if label not in seen:
                seen.add(label)
                labels.append(label)
    return labels


def _fill_label_gaps(
    frames: list[dict],
    *,
    frame_index_to_pos: Dict[int, int],
    label: str,
    max_gap_frames: int,
) -> None:
    observations = []
    for item in frames:
        frame = item["frame"]
        if label in item["points"]:
            observations.append((int(frame.frame_index), item["points"][label]))
    for current, nxt in zip(observations, observations[1:]):
        start_idx, start_point = current
        end_idx, end_point = nxt
        gap = int(end_idx) - int(start_idx) - 1
        if gap <= 0 or gap > int(max_gap_frames):
            continue
        for missing_frame_index in range(start_idx + 1, end_idx):
            pos = frame_index_to_pos.get(missing_frame_index)
            if pos is None:
                continue
            ratio = float(missing_frame_index - start_idx) / float(end_idx - start_idx)
            interp = (
                (1.0 - ratio) * float(start_point[0]) + ratio * float(end_point[0]),
                (1.0 - ratio) * float(start_point[1]) + ratio * float(end_point[1]),
            )
            if label not in frames[pos]["points"]:
                frames[pos]["points"][label] = interp
                frames[pos]["scores"][label] = min(
                    float(
                        frames[frame_index_to_pos[start_idx]]["scores"].get(label, 1.0)
                    ),
                    float(
                        frames[frame_index_to_pos[end_idx]]["scores"].get(label, 1.0)
                    ),
                )
