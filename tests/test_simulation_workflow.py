from __future__ import annotations

import json
from pathlib import Path

from annolid.simulation import SimulationRunRequest, run_simulation_workflow


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def test_run_simulation_workflow_identity_writes_ndjson(tmp_path: Path) -> None:
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

    out_path = run_simulation_workflow(
        SimulationRunRequest(
            backend="identity",
            input_path=str(input_path),
            mapping_path=str(mapping_path),
            out_ndjson=str(output_path),
            video_name="demo.mp4",
        )
    )

    payload = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert payload["video_name"] == "demo.mp4"
    assert payload["otherData"]["simulation"]["adapter"] == "identity"
    assert payload["otherData"]["simulation"]["state"]["site_targets"][
        "snout_site"
    ] == [10.0, 12.0, 0.0]
