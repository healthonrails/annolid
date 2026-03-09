from __future__ import annotations

import json
import base64
import io
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from annolid.engine.cli import main as annolid_run


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _depth_png_base64(values) -> str:
    buf = io.BytesIO()
    imageio.imwrite(buf, np.asarray(values, dtype=np.uint16), format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def test_simulation_runner_identity_writes_annolid_ndjson(tmp_path: Path) -> None:
    input_path = tmp_path / "pose.json"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "simulation.ndjson"
    _write_json(
        input_path,
        {
            "version": "5.0",
            "imagePath": "frame.png",
            "imageHeight": 32,
            "imageWidth": 64,
            "shapes": [
                {"label": "nose", "shape_type": "point", "points": [[10, 12]]},
            ],
        },
    )
    _write_json(
        mapping_path,
        {
            "backend": "identity",
            "keypoint_to_site": {"nose": "snout_site"},
        },
    )

    rc = annolid_run(
        [
            "predict",
            "simulation_runner",
            "--backend",
            "identity",
            "--input",
            str(input_path),
            "--mapping",
            str(mapping_path),
            "--out-ndjson",
            str(output_path),
            "--video-name",
            "demo.mp4",
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["video_name"] == "demo.mp4"
    assert payload["otherData"]["simulation"]["adapter"] == "identity"
    assert payload["otherData"]["simulation"]["state"]["site_targets"][
        "snout_site"
    ] == [10.0, 12.0, 0.0]


def test_flybody_plugin_dry_run_emits_site_targets(tmp_path: Path) -> None:
    input_path = tmp_path / "pose.ndjson"
    mapping_path = tmp_path / "flybody.json"
    output_path = tmp_path / "flybody.ndjson"
    input_path.write_text(
        json.dumps(
            {
                "version": "Annolid",
                "video_name": "demo.mp4",
                "frame_index": 1,
                "imagePath": "frame_0001.png",
                "imageHeight": 32,
                "imageWidth": 64,
                "flags": {},
                "otherData": {},
                "shapes": [
                    {"label": "nose", "shape_type": "point", "points": [[4, 8]]},
                    {"label": "tail_base", "shape_type": "point", "points": [[20, 24]]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        mapping_path,
        {
            "backend": "flybody",
            "keypoint_to_site": {
                "nose": "head_site",
                "tail_base": "abdomen_site",
            },
            "coordinate_system": {"units": "meters"},
        },
    )

    rc = annolid_run(
        [
            "predict",
            "flybody",
            "--input",
            str(input_path),
            "--mapping",
            str(mapping_path),
            "--out-ndjson",
            str(output_path),
            "--dry-run",
            "--default-z",
            "0.5",
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    simulation = payload["otherData"]["simulation"]
    assert simulation["adapter"] == "flybody"
    assert simulation["state"]["dry_run"] is True
    assert simulation["state"]["site_targets"]["head_site"] == [4.0, 8.0, 0.5]
    assert simulation["mapping_metadata"]["coordinate_system"]["units"] == "meters"


def test_flybody_plugin_uses_depth_ndjson_for_3d_targets(tmp_path: Path) -> None:
    input_path = tmp_path / "pose.ndjson"
    mapping_path = tmp_path / "flybody.json"
    depth_path = tmp_path / "depth.ndjson"
    output_path = tmp_path / "flybody_depth.ndjson"
    input_path.write_text(
        json.dumps(
            {
                "version": "Annolid",
                "video_name": "demo.mp4",
                "frame_index": 0,
                "imagePath": "frame_0000.png",
                "imageHeight": 2,
                "imageWidth": 2,
                "flags": {},
                "otherData": {},
                "shapes": [
                    {"label": "nose", "shape_type": "point", "points": [[1, 1]]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    depth_path.write_text(
        json.dumps(
            {
                "frame_index": 0,
                "otherData": {
                    "depth_map": {
                        "image_data": _depth_png_base64([[0, 0], [0, 65535]]),
                        "scale": {"min": 0.0, "max": 2.0},
                        "dtype": "uint16",
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        mapping_path,
        {
            "backend": "flybody",
            "keypoint_to_site": {"nose": "head_site"},
            "coordinate_system": {
                "camera_intrinsics": {"fx": 2.0, "fy": 4.0, "cx": 0.0, "cy": 0.0}
            },
        },
    )

    rc = annolid_run(
        [
            "predict",
            "flybody",
            "--input",
            str(input_path),
            "--mapping",
            str(mapping_path),
            "--depth-ndjson",
            str(depth_path),
            "--out-ndjson",
            str(output_path),
            "--dry-run",
        ]
    )

    assert rc == 0
    payload = json.loads(output_path.read_text(encoding="utf-8").splitlines()[0])
    target = payload["otherData"]["simulation"]["state"]["site_targets"]["head_site"]
    assert target == [1.0, 0.5, 2.0]


def test_flybody_plugin_without_dry_run_surfaces_backend_import_errors(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "pose.json"
    mapping_path = tmp_path / "mapping.json"
    output_path = tmp_path / "out.ndjson"
    _write_json(
        input_path,
        {
            "version": "5.0",
            "imagePath": "frame.png",
            "imageHeight": 32,
            "imageWidth": 64,
            "shapes": [{"label": "nose", "shape_type": "point", "points": [[1, 2]]}],
        },
    )
    _write_json(
        mapping_path, {"backend": "flybody", "keypoint_to_site": {"nose": "head"}}
    )

    with pytest.raises(RuntimeError, match="optional FlyBody stack"):
        annolid_run(
            [
                "predict",
                "flybody",
                "--input",
                str(input_path),
                "--mapping",
                str(mapping_path),
                "--out-ndjson",
                str(output_path),
                "--env-factory",
                "missing.module:factory",
            ]
        )


def test_flybody_help_uses_input_examples(capsys) -> None:
    rc = annolid_run(["help", "predict", "flybody"])

    assert rc == 0
    text = capsys.readouterr().out
    assert "annolid-run predict flybody --input pose.ndjson" in text
    assert "--source <video-or-image>" not in text


def test_flybody_plugin_can_write_mapping_template_from_pose_schema(
    tmp_path: Path,
) -> None:
    schema_path = tmp_path / "pose_schema.json"
    template_path = tmp_path / "flybody.yaml"
    _write_json(
        schema_path,
        {
            "keypoints": ["nose", "tail_base"],
        },
    )

    rc = annolid_run(
        [
            "predict",
            "flybody",
            "--pose-schema",
            str(schema_path),
            "--write-mapping-template",
            str(template_path),
        ]
    )

    assert rc == 0
    text = template_path.read_text(encoding="utf-8")
    assert "backend: flybody" in text
    assert "nose: nose_site" in text
    assert "tail_base: tail_base_site" in text
