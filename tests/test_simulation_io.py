from __future__ import annotations

import json
from pathlib import Path

from annolid.annotation.pose_schema import PoseSchema
from annolid.simulation import (
    SimulationFrameResult,
    SimulationRunResult,
    load_simulation_mapping,
    read_pose_frames,
    write_simulation_ndjson,
)


def test_read_pose_frames_from_labelme_uses_pose_schema_prefixes(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / "frame_0001.json"
    annotation.write_text(
        json.dumps(
            {
                "version": "5.0",
                "imagePath": "frame_0001.png",
                "imageHeight": 64,
                "imageWidth": 96,
                "shapes": [
                    {
                        "label": "resident_nose",
                        "shape_type": "point",
                        "points": [[10, 12]],
                        "flags": {"score": 0.95},
                    },
                    {
                        "label": "resident_tail_base",
                        "shape_type": "point",
                        "points": [[20, 22]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    schema = PoseSchema.from_dict(
        {
            "keypoints": ["nose", "tail_base"],
            "instances": ["resident"],
            "instance_separator": "_",
        }
    )

    frames = read_pose_frames(annotation, pose_schema=schema, video_name="demo.mp4")

    assert len(frames) == 1
    frame = frames[0]
    assert frame.video_name == "demo.mp4"
    assert frame.points == {"nose": (10.0, 12.0), "tail_base": (20.0, 22.0)}
    assert frame.instances == {"nose": "resident", "tail_base": "resident"}
    assert frame.scores == {"nose": 0.95}


def test_load_simulation_mapping_from_yaml(tmp_path: Path) -> None:
    config = tmp_path / "flybody.yaml"
    config.write_text(
        "\n".join(
            [
                "backend: flybody",
                "keypoint_to_site:",
                "  nose: head_site",
                "site_to_joint:",
                "  head_site: neck_joint",
                "coordinate_system:",
                "  units: meters",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    mapping = load_simulation_mapping(config)

    assert mapping.backend == "flybody"
    assert mapping.keypoint_to_site["nose"] == "head_site"
    assert mapping.coordinate_system["units"] == "meters"


def test_write_simulation_ndjson_preserves_shapes_and_adds_metadata(
    tmp_path: Path,
) -> None:
    annotation = tmp_path / "track.ndjson"
    annotation.write_text(
        json.dumps(
            {
                "version": "Annolid",
                "video_name": "demo.mp4",
                "frame_index": 3,
                "timestamp_sec": 0.1,
                "imagePath": "frame_0003.png",
                "imageHeight": 64,
                "imageWidth": 96,
                "flags": {},
                "otherData": {"agent": {"source": "unit-test"}},
                "shapes": [
                    {
                        "label": "nose",
                        "shape_type": "point",
                        "points": [[10, 12]],
                        "flags": {"score": 0.9},
                    }
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    frames = read_pose_frames(annotation)
    result = SimulationRunResult(
        frames=[
            SimulationFrameResult(
                frame_index=3,
                timestamp_sec=0.1,
                state={"qpos": [1.0, 2.0]},
                diagnostics={"residual": 0.01},
            )
        ],
        metadata={"backend_version": "0.1"},
    )

    out_path = tmp_path / "simulation.ndjson"
    write_simulation_ndjson(
        out_path,
        pose_frames=frames,
        result=result,
        adapter_name="flybody",
        extra_metadata={"mapping_name": "demo-map"},
    )

    payload = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["frame_index"] == 3
    assert payload["shapes"][0]["label"] == "nose"
    assert payload["otherData"]["agent"]["source"] == "unit-test"
    assert payload["otherData"]["simulation"]["adapter"] == "flybody"
    assert payload["otherData"]["simulation"]["state"]["qpos"] == [1.0, 2.0]
    assert payload["otherData"]["simulation"]["diagnostics"]["residual"] == 0.01
    assert (
        payload["otherData"]["simulation"]["mapping_metadata"]["mapping_name"]
        == "demo-map"
    )
