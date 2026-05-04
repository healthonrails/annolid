from __future__ import annotations

import csv
import json
from pathlib import Path

from annolid.annotation.labelme2csv import convert_json_to_csv
from annolid.postprocessing.zone_schema import build_zone_shape


def test_tracked_csv_contains_zone_columns_and_binary_membership(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "session"
    folder.mkdir()
    (folder / "session_000000000.png").write_bytes(b"")
    (folder / "session_000000001.png").write_bytes(b"")

    zone = build_zone_shape(
        "left_zone",
        [[0, 0], [50, 0], [50, 50], [0, 50]],
        zone_kind="chamber",
    )

    frame0 = folder / "session_000000000.json"
    frame0.write_text(
        json.dumps(
            {
                "imageHeight": 100,
                "imageWidth": 100,
                "shapes": [
                    zone,
                    {
                        "label": "mouse",
                        "instance_label": "mouse",
                        "shape_type": "rectangle",
                        "points": [[10, 10], [20, 20]],
                        "group_id": 1,
                        "description": "motion_index: 0.1",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    frame1 = folder / "session_000000001.json"
    frame1.write_text(
        json.dumps(
            {
                "imageHeight": 100,
                "imageWidth": 100,
                "shapes": [
                    {
                        "label": "mouse",
                        "instance_label": "mouse",
                        "shape_type": "rectangle",
                        "points": [[70, 70], [80, 80]],
                        "group_id": 1,
                        "description": "motion_index: 0.2",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    tracked_csv = tmp_path / "session_tracked.csv"
    convert_json_to_csv(str(folder), tracked_csv_file=str(tracked_csv))

    with tracked_csv.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert "left_zone" in rows[0]
    assert len(rows) == 2
    assert rows[0]["frame_number"] == "0"
    assert rows[0]["left_zone"] == "1"
    assert rows[1]["frame_number"] == "1"
    assert rows[1]["left_zone"] == "0"


def test_tracked_csv_loads_zone_from_non_first_manual_pair(tmp_path: Path) -> None:
    folder = tmp_path / "session"
    folder.mkdir()

    (folder / "session_000000000.png").write_bytes(b"")
    (folder / "session_000000001.png").write_bytes(b"")

    frame0 = folder / "session_000000000.json"
    frame0.write_text(
        json.dumps(
            {
                "imageHeight": 100,
                "imageWidth": 100,
                "shapes": [
                    {
                        "label": "mouse",
                        "instance_label": "mouse",
                        "shape_type": "rectangle",
                        "points": [[10, 10], [20, 20]],
                        "group_id": 1,
                        "description": "motion_index: 0.1",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    frame1 = folder / "session_000000001.json"
    frame1.write_text(
        json.dumps(
            {
                "imageHeight": 100,
                "imageWidth": 100,
                "shapes": [
                    build_zone_shape(
                        "late_zone",
                        [[0, 0], [50, 0], [50, 50], [0, 50]],
                        zone_kind="chamber",
                    ),
                    {
                        "label": "mouse",
                        "instance_label": "mouse",
                        "shape_type": "rectangle",
                        "points": [[70, 70], [80, 80]],
                        "group_id": 1,
                        "description": "motion_index: 0.2",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    tracked_csv = tmp_path / "session_tracked.csv"
    convert_json_to_csv(str(folder), tracked_csv_file=str(tracked_csv))

    with tracked_csv.open("r", newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert "late_zone" in rows[0]
