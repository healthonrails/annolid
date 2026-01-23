from __future__ import annotations

import json
from pathlib import Path

from annolid.engine.cli import main as annolid_run


def test_annolid_run_validate_agent_output_accepts_valid_ndjson(tmp_path: Path) -> None:
    ndjson_path = tmp_path / "out.ndjson"
    record = {
        "version": "Annolid",
        "video_name": "test_video.mp4",
        "frame_index": 0,
        "imagePath": "",
        "imageHeight": 10,
        "imageWidth": 10,
        "flags": {},
        "otherData": {},
        "shapes": [
            {
                "label": "mouse",
                "points": [[0, 0], [10, 10]],
                "shape_type": "rectangle",
                "group_id": 1,
                "flags": {},
            }
        ],
    }
    ndjson_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rc = annolid_run(["validate-agent-output", "--ndjson", str(ndjson_path)])
    assert rc == 0


def test_annolid_run_validate_agent_output_rejects_invalid_ndjson(
    tmp_path: Path,
) -> None:
    ndjson_path = tmp_path / "bad.ndjson"
    record = {
        "version": "Annolid",
        "video_name": "test_video.mp4",
        "frame_index": 0,
        "imagePath": "",
        "imageHeight": 10,
        "imageWidth": 10,
        "flags": {},
        "otherData": {},
        "shapes": [{"label": "mouse", "shape_type": "rectangle"}],
    }
    ndjson_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rc = annolid_run(["validate-agent-output", "--ndjson", str(ndjson_path)])
    assert rc == 1
