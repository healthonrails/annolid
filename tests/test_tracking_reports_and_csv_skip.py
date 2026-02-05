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
