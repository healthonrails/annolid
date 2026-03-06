from __future__ import annotations

from pathlib import Path

from annolid.services.tracking import (
    has_tracking_completion_artifacts,
    run_video_processor_frames,
    run_tracking_video_frames,
)


class _FakeProcessor:
    def __init__(self) -> None:
        self.calls = []

    def process_video_frames(self, **kwargs):
        self.calls.append(kwargs)
        return "ok"


def test_run_tracking_video_frames_uses_config() -> None:
    processor = _FakeProcessor()
    message = run_tracking_video_frames(
        processor=processor,
        start_frame=3,
        end_frame=7,
        config={
            "mem_every": 9,
            "has_occlusion": False,
            "save_video_with_color_mask": True,
        },
    )

    assert message == "ok"
    assert processor.calls
    call = processor.calls[0]
    assert call["start_frame"] == 3
    assert call["end_frame"] == 7
    assert call["mem_every"] == 9
    assert call["has_occlusion"] is False
    assert call["save_video_with_color_mask"] is True


def test_has_tracking_completion_artifacts_detects_json(tmp_path: Path) -> None:
    video_path = tmp_path / "demo.mp4"
    video_path.write_bytes(b"")
    out = video_path.with_suffix("")
    out.mkdir(parents=True, exist_ok=True)
    (out / "demo_000000004.json").write_text("{}", encoding="utf-8")

    assert has_tracking_completion_artifacts(
        video_path=str(video_path),
        output_folder=str(out),
        total_frames=5,
    )


def test_run_video_processor_frames_forwards_args() -> None:
    processor = _FakeProcessor()
    message = run_video_processor_frames(
        processor=processor,
        start_frame=1,
        end_frame=9,
        is_cutie=True,
        is_new_segment=True,
        extra_kwargs={"mem_every": 4},
    )

    assert message == "ok"
    call = processor.calls[0]
    assert call["start_frame"] == 1
    assert call["end_frame"] == 9
    assert call["is_cutie"] is True
    assert call["is_new_segment"] is True
    assert call["mem_every"] == 4
