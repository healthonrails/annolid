from __future__ import annotations

import json
from pathlib import Path

from annolid.simulation import (
    build_simulation_view_payload,
    export_simulation_view_payload,
)


def test_build_simulation_view_payload_uses_site_targets_and_metadata(
    tmp_path: Path,
) -> None:
    ndjson_path = tmp_path / "flybody.ndjson"
    record = {
        "video_name": "demo",
        "frame_index": 3,
        "timestamp_sec": 0.15,
        "imagePath": "frame_0003.png",
        "imageHeight": 32,
        "imageWidth": 32,
        "shapes": [],
        "otherData": {
            "simulation": {
                "adapter": "flybody",
                "run_metadata": {"fps": 30},
                "mapping_metadata": {
                    "coordinate_system": {"units": "meters"},
                    "metadata": {"viewer_edges": [["head", "thorax"]]},
                },
                "state": {
                    "dry_run": True,
                    "qpos": [1.0, 2.0],
                    "site_targets": {
                        "head": [0.1, 0.2, 0.3],
                        "thorax": [0.3, 0.4, 0.5],
                    },
                },
                "diagnostics": {"residual": 0.02},
            }
        },
    }
    ndjson_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    payload = build_simulation_view_payload(ndjson_path)

    assert payload["kind"] == "annolid-simulation-v1"
    assert payload["adapter"] == "flybody"
    assert payload["edges"] == [["head", "thorax"]]
    assert payload["metadata"]["run_metadata"]["fps"] == 30
    assert payload["frames"][0]["qpos"] == [1.0, 2.0]
    assert payload["frames"][0]["dry_run"] is True
    assert payload["frames"][0]["points"][0]["label"] == "head"


def test_build_simulation_view_payload_falls_back_to_shape_points(
    tmp_path: Path,
) -> None:
    ndjson_path = tmp_path / "simulation.ndjson"
    record = {
        "video_name": "demo",
        "frame_index": 0,
        "imagePath": "frame_0000.png",
        "imageHeight": 32,
        "imageWidth": 32,
        "shapes": [
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[12, 15]],
            }
        ],
        "otherData": {
            "simulation": {
                "adapter": "identity",
                "state": {},
            }
        },
    }
    ndjson_path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    payload = build_simulation_view_payload(ndjson_path)

    assert payload["adapter"] == "identity"
    assert payload["frames"][0]["points"] == [
        {"label": "nose", "x": 12.0, "y": 15.0, "z": 0.0}
    ]


def test_export_simulation_view_payload_writes_json(tmp_path: Path) -> None:
    ndjson_path = tmp_path / "simulation.ndjson"
    ndjson_path.write_text(
        json.dumps(
            {
                "video_name": "demo",
                "frame_index": 0,
                "imagePath": "frame_0000.png",
                "imageHeight": 32,
                "imageWidth": 32,
                "shapes": [],
                "otherData": {"simulation": {"adapter": "flybody", "state": {}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = export_simulation_view_payload(ndjson_path, out_dir=tmp_path)

    assert out_path.suffix == ".json"
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "annolid-simulation-v1"
    assert payload["adapter"] == "flybody"


def test_export_simulation_view_payload_reuses_up_to_date_json(tmp_path: Path) -> None:
    ndjson_path = tmp_path / "simulation.ndjson"
    ndjson_path.write_text(
        json.dumps(
            {
                "video_name": "demo",
                "frame_index": 0,
                "imagePath": "frame_0000.png",
                "imageHeight": 32,
                "imageWidth": 32,
                "shapes": [],
                "otherData": {"simulation": {"adapter": "flybody", "state": {}}},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = export_simulation_view_payload(ndjson_path, out_dir=tmp_path)
    original_mtime = out_path.stat().st_mtime_ns
    original_text = out_path.read_text(encoding="utf-8")

    reused_path = export_simulation_view_payload(ndjson_path, out_dir=tmp_path)

    assert reused_path == out_path
    assert reused_path.stat().st_mtime_ns == original_mtime
    assert reused_path.read_text(encoding="utf-8") == original_text


def test_build_simulation_view_payload_infers_flybody_edges_from_labels(
    tmp_path: Path,
) -> None:
    ndjson_path = tmp_path / "flybody.ndjson"
    ndjson_path.write_text(
        json.dumps(
            {
                "video_name": "demo",
                "frame_index": 0,
                "imagePath": "frame.png",
                "imageHeight": 32,
                "imageWidth": 32,
                "shapes": [],
                "otherData": {
                    "simulation": {
                        "adapter": "flybody",
                        "mapping_metadata": {},
                        "state": {
                            "site_targets": {
                                "head_site": [0, 0.2, 0],
                                "thorax_site": [0, 0, 0],
                                "abdomen_tip_site": [0, -0.2, 0],
                            }
                        },
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_simulation_view_payload(ndjson_path)

    assert ["head_site", "thorax_site"] in payload["edges"]
    assert ["thorax_site", "abdomen_tip_site"] in payload["edges"]
