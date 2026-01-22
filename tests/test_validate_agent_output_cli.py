from __future__ import annotations

import json
from pathlib import Path

from annolid.core.types import BBoxGeometry, FrameRef
from annolid.engine.cli import main as annolid_run


def test_annolid_run_validate_agent_output_accepts_valid_ndjson(tmp_path: Path) -> None:
    ndjson_path = tmp_path / "out.ndjson"
    record = {
        "schema_version": "annolid.agent_output.1",
        "type": "detection",
        "frame": FrameRef(frame_index=0, timestamp_sec=0.0).to_dict(),
        "objects": [{"geometry": BBoxGeometry("bbox", (0, 0, 10, 10)).to_dict()}],
    }
    ndjson_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rc = annolid_run(["validate-agent-output", "--ndjson", str(ndjson_path)])
    assert rc == 0


def test_annolid_run_validate_agent_output_rejects_invalid_ndjson(
    tmp_path: Path,
) -> None:
    ndjson_path = tmp_path / "bad.ndjson"
    record = {
        "schema_version": "annolid.agent_output.1",
        "type": "detection",
        "frame": FrameRef(frame_index=0).to_dict(),
        "objects": [{"geometry": {"type": "circle", "r": 3}}],
    }
    ndjson_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    rc = annolid_run(["validate-agent-output", "--ndjson", str(ndjson_path)])
    assert rc == 1
