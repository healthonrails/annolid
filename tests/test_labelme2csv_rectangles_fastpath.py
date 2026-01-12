from __future__ import annotations

import csv
from pathlib import Path

from annolid.annotation.labelme2csv import convert_json_to_csv


def test_convert_json_to_csv_emits_rectangle_instances_and_skips_yolo_keypoints(tmp_path: Path) -> None:
    folder = tmp_path / "video"
    folder.mkdir()

    frame0 = folder / "video_000000000.json"
    frame0.write_text(
        "\n".join(
            [
                "{",
                '  "version": "5.5.0",',
                '  "imagePath": "",',
                '  "imageHeight": 100,',
                '  "imageWidth": 200,',
                '  "shapes": [',
                # Instance bbox
                '    {',
                '      "label": "resident",',
                '      "shape_type": "rectangle",',
                '      "points": [[10, 10], [50, 30]],',
                '      "group_id": 1,',
                '      "description": "yolo",',
                '      "instance_label": "resident",',
                '      "score": 0.9',
                "    },",
                # YOLO keypoint (should be skipped when rectangles exist)
                '    {',
                '      "label": "nose",',
                '      "shape_type": "point",',
                '      "points": [[20, 20]],',
                '      "group_id": 1,',
                '      "description": "yolo",',
                '      "instance_label": "resident",',
                '      "instance_id": 1,',
                '      "score": 0.8',
                "    }",
                "  ]",
                "}",
            ]
        ),
        encoding="utf-8",
    )

    out_csv = tmp_path / "out.csv"
    result = convert_json_to_csv(str(folder), csv_file=str(out_csv))
    assert Path(result).exists()

    with out_csv.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.reader(fh))

    # Header + 1 rectangle row; keypoint rows should be skipped.
    assert len(rows) == 2
    header, row = rows
    assert header[:3] == ["frame_number", "x1", "y1"]
    assert row[0] == "0"
    assert row[7] == "resident"
    assert float(row[8]) == 0.9
