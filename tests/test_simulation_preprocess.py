from __future__ import annotations

from annolid.simulation import smooth_pose_frames
from annolid.simulation.types import Pose2DFrame


def _frame(frame_index: int, point: tuple[float, float] | None) -> Pose2DFrame:
    points = {"nose": point} if point is not None else {}
    scores = {"nose": 1.0} if point is not None else {}
    return Pose2DFrame(
        frame_index=frame_index,
        image_height=32,
        image_width=32,
        video_name="demo.mp4",
        points=points,
        scores=scores,
    )


def test_smooth_pose_frames_interpolates_small_gaps() -> None:
    frames = [
        _frame(0, (0.0, 0.0)),
        _frame(1, None),
        _frame(2, (2.0, 2.0)),
    ]

    out = smooth_pose_frames(frames, mode="none", max_gap_frames=1)

    assert out[1].points["nose"] == (1.0, 1.0)


def test_smooth_pose_frames_applies_ema_after_gap_fill() -> None:
    frames = [
        _frame(0, (0.0, 0.0)),
        _frame(1, None),
        _frame(2, (10.0, 0.0)),
    ]

    out = smooth_pose_frames(
        frames,
        mode="ema",
        max_gap_frames=1,
        ema_alpha=0.5,
    )

    assert out[1].points["nose"] == (2.5, 0.0)
    assert out[2].points["nose"] == (6.25, 0.0)
