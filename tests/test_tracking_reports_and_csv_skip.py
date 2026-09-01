from __future__ import annotations

import json
from pathlib import Path

from annolid.annotation.labelme2csv import convert_json_to_csv
from annolid.postprocessing import tracking_reports


def test_find_tracking_gaps_handles_numeric_labels_without_keyerror(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"")
    json_dir = tmp_path / "video"
    json_dir.mkdir()

    (json_dir / "video_000000000.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (json_dir / "video_000000002.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tracking_reports, "OPENCV_AVAILABLE", False)
    gaps = tracking_reports.find_tracking_gaps(str(video_path))
    assert 1 in gaps
    assert gaps[1][0]["start_frame"] == 1
    assert gaps[1][0]["end_frame"] == 1


def test_find_tracking_gaps_merges_duplicate_frame_json_files(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"")
    json_dir = tmp_path / "video"
    json_dir.mkdir()

    (json_dir / "video_000000000.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "fish",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (json_dir / "video_copy_000000000.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "fish",
                    },
                    {
                        "label": "snail",
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    (json_dir / "video_000000002.json").write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "fish",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tracking_reports, "OPENCV_AVAILABLE", False)

    gaps = tracking_reports.find_tracking_gaps(str(video_path))

    assert gaps["fish"][0]["start_frame"] == 1
    assert gaps["fish"][0]["end_frame"] == 1
    assert gaps["snail"][0]["start_frame"] == 1
    assert gaps["snail"][0]["end_frame"] == 2


def test_find_tracking_gaps_prefers_complete_tracking_csv(
    tmp_path: Path, monkeypatch
) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"")
    json_dir = tmp_path / "video"
    json_dir.mkdir()
    (json_dir / "video_000000000.json").write_text(
        json.dumps({"shapes": [{"label": "head"}]}),
        encoding="utf-8",
    )

    tracking_csv = tmp_path / "video_tracking.csv"
    tracking_csv.write_text(
        "\n".join(
            [
                "frame_number,x1,y1,x2,y2,cx,cy,instance_name,class_score,segmentation,tracking_id",
                "0,0,0,1,1,0.5,0.5,head,1.0,,0",
                "0,0,0,1,1,0.5,0.5,thorax,1.0,,0",
                "1,0,0,1,1,0.5,0.5,head,1.0,,0",
                "1,0,0,1,1,0.5,0.5,thorax,1.0,,0",
                "2,0,0,1,1,0.5,0.5,head,1.0,,0",
                "2,0,0,1,1,0.5,0.5,thorax,1.0,,0",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(tracking_reports, "OPENCV_AVAILABLE", False)

    assert tracking_reports.find_tracking_gaps(str(video_path)) == {}


def test_generate_reports_removes_stale_gap_csv_when_no_gaps(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"")
    stale_csv = tmp_path / "video_gaps_report.csv"
    stale_csv.write_text(
        "instance_label,start_frame,end_frame,duration_frames\nhead,1,2,2\n",
        encoding="utf-8",
    )

    tracking_reports.generate_reports({}, str(video_path))

    assert not stale_csv.exists()
    assert (tmp_path / "video_tracking_gaps_report.md").exists()


def test_convert_json_to_csv_skips_when_existing_csv_is_complete(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "video"
    folder.mkdir()

    frame_payload = {
        "imageHeight": 64,
        "imageWidth": 64,
        "shapes": [
            {
                "label": "fish",
                "shape_type": "rectangle",
                "points": [[10, 10], [30, 30]],
                "group_id": 1,
            }
        ],
    }
    (folder / "video_000000000.json").write_text(
        json.dumps(frame_payload), encoding="utf-8"
    )
    (folder / "video_000000001.json").write_text(
        json.dumps(frame_payload), encoding="utf-8"
    )

    out_csv = tmp_path / "video_tracking.csv"
    out_csv.write_text(
        "\n".join(
            [
                "frame_number,x1,y1,x2,y2,cx,cy,instance_name,class_score,segmentation,tracking_id",
                "0,10,10,30,30,20,20,fish,1.0,,1",
                "1,10,10,30,30,20,20,fish,1.0,,1",
            ]
        ),
        encoding="utf-8",
    )
    before = out_csv.read_text(encoding="utf-8")

    result = convert_json_to_csv(str(folder), csv_file=str(out_csv))

    assert result == str(out_csv)
    assert out_csv.read_text(encoding="utf-8") == before


def test_convert_json_to_csv_preserves_timestamp_enriched_tracked_csv(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "video"
    folder.mkdir()
    payload = {
        "imageHeight": 32,
        "imageWidth": 32,
        "shapes": [
            {
                "label": "fish",
                "shape_type": "rectangle",
                "points": [[4, 4], [12, 12]],
            }
        ],
    }
    for frame in range(2):
        (folder / f"video_{frame:09d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    tracking_csv = tmp_path / "video_tracking.csv"
    tracking_csv.write_text(
        "frame_number,x1,y1,x2,y2,cx,cy,instance_name,class_score,segmentation,tracking_id\n"
        "0,4,4,12,12,8,8,fish,1,,0\n"
        "1,4,4,12,12,8,8,fish,1,,0\n",
        encoding="utf-8",
    )
    tracked_csv = tmp_path / "video_tracked.csv"
    tracked_csv.write_text(
        "frame_number,instance_name,cx,cy,motion_index,timestamps,real_timestamp_sec\n"
        "0,fish,8,8,-1,00:00:00,0.0\n"
        "1,fish,8,8,-1,00:00:01,1.0\n",
        encoding="utf-8",
    )
    before = tracked_csv.read_text(encoding="utf-8")

    result = convert_json_to_csv(
        str(folder),
        csv_file=str(tracking_csv),
        tracked_csv_file=str(tracked_csv),
    )

    assert result == str(tracking_csv)
    assert tracked_csv.read_text(encoding="utf-8") == before


def test_stopped_conversion_does_not_publish_partial_csv_files(tmp_path: Path) -> None:
    class _StopAfterFirstCheck:
        def __init__(self) -> None:
            self.calls = 0

        def is_set(self) -> bool:
            self.calls += 1
            return self.calls > 1

    folder = tmp_path / "video"
    folder.mkdir()
    payload = {
        "imageHeight": 32,
        "imageWidth": 32,
        "shapes": [
            {
                "label": "fish",
                "shape_type": "rectangle",
                "points": [[4, 4], [12, 12]],
            }
        ],
    }
    for frame in range(2):
        (folder / f"video_{frame:09d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    tracking_csv = tmp_path / "video_tracking.csv"
    tracked_csv = tmp_path / "video_tracked.csv"
    tracking_csv.write_text("original tracking data\n", encoding="utf-8")
    tracked_csv.write_text("original tracked data\n", encoding="utf-8")

    result = convert_json_to_csv(
        str(folder),
        csv_file=str(tracking_csv),
        tracked_csv_file=str(tracked_csv),
        stop_event=_StopAfterFirstCheck(),
    )

    assert result == "Stopped"
    assert tracking_csv.read_text(encoding="utf-8") == "original tracking data\n"
    assert tracked_csv.read_text(encoding="utf-8") == "original tracked data\n"
    assert list(tmp_path.glob(".*.tmp")) == []
