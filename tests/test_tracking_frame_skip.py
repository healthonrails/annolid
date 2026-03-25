from __future__ import annotations

from pathlib import Path

from annolid.tracking.frame_skip import (
    build_seeded_frame_index,
    remove_seeded_frame,
    should_skip_finished_frame_between_adjacent_seeded_frames,
)


def test_should_skip_finished_frame_between_adjacent_seeded_frames(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "video"
    folder.mkdir()
    (folder / "video_000000004.json").write_text("{}", encoding="utf-8")

    seeded = build_seeded_frame_index(3, [5, 3])
    cache: dict[int, bool] = {}

    assert should_skip_finished_frame_between_adjacent_seeded_frames(
        frame_number=4,
        seeded_frames=seeded,
        video_result_folder=folder,
        finished_frame_cache=cache,
    )
    assert not should_skip_finished_frame_between_adjacent_seeded_frames(
        frame_number=6,
        seeded_frames=seeded,
        video_result_folder=folder,
        finished_frame_cache=cache,
    )


def test_remove_seeded_frame_updates_skip_boundaries(tmp_path: Path) -> None:
    folder = tmp_path / "video"
    folder.mkdir()
    (folder / "video_000000004.json").write_text("{}", encoding="utf-8")
    seeded = build_seeded_frame_index(3, [5])
    cache: dict[int, bool] = {}

    assert should_skip_finished_frame_between_adjacent_seeded_frames(
        frame_number=4,
        seeded_frames=seeded,
        video_result_folder=folder,
        finished_frame_cache=cache,
    )
    remove_seeded_frame(seeded, 5)
    assert not should_skip_finished_frame_between_adjacent_seeded_frames(
        frame_number=4,
        seeded_frames=seeded,
        video_result_folder=folder,
        finished_frame_cache=cache,
    )
