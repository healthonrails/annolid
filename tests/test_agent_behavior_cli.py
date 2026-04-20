from __future__ import annotations

import json
from pathlib import Path

from annolid.engine.cli import main as annolid_run


def test_agent_behavior_cli_with_artifact_ndjson(tmp_path: Path, capsys) -> None:
    artifacts = tmp_path / "artifacts.ndjson"
    artifacts.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "artifact_id": "a1",
                        "frame_index": 10,
                        "track_id": "mouse_1",
                        "label": "slap in the face",
                        "meta": {"count": 2},
                    }
                ),
                json.dumps(
                    {
                        "artifact_id": "a2",
                        "frame_index": 14,
                        "track_id": "mouse_2",
                        "label": "run away",
                        "meta": {"count": 1},
                    }
                ),
                json.dumps(
                    {
                        "artifact_id": "a3",
                        "frame_index": 60,
                        "track_id": "mouse_1",
                        "label": "fight_initiation",
                        "meta": {"count": 1},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    video_path = tmp_path / "mouse.mp4"
    rc = annolid_run(
        [
            "agent-behavior",
            "--video",
            str(video_path),
            "--results-dir",
            str(tmp_path / "results"),
            "--artifacts-ndjson",
            str(artifacts),
            "--run-id",
            "run_cli_001",
            "--context-prompt",
            "analyze aggression bouts",
            "--default-assay",
            "aggression",
        ]
    )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["run_id"] == "run_cli_001"
    assert payload["task_plan_assay"] == "aggression"
    assert payload["artifact_count"] == 3
    assert payload["segment_count"] == 2
    assert payload["validation_errors"] == []
    assert len(payload["bout_counts"]) == 2

    manifest_path = Path(payload["manifest_path"])
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["analysis_run"]["run_id"] == "run_cli_001"
    assert manifest["artifacts"]["tracks"] == "artifacts/tracks.ndjson"
    assert manifest["artifacts"]["segments"] == "artifacts/behaviors.ndjson"


def test_agent_behavior_cli_fail_on_validation_error(tmp_path: Path, capsys) -> None:
    artifacts = tmp_path / "bad_artifacts.ndjson"
    artifacts.write_text(
        json.dumps(
            {
                "artifact_id": "a1",
                "frame_index": 10,
                "label": "slap_face",
                "meta": {"count": -1},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rc = annolid_run(
        [
            "agent-behavior",
            "--video",
            str(tmp_path / "mouse.mp4"),
            "--results-dir",
            str(tmp_path / "results"),
            "--artifacts-ndjson",
            str(artifacts),
            "--run-id",
            "run_cli_002",
            "--fail-on-validation-error",
        ]
    )
    # Negative sub-event count is ignored in aggregation so no validation failure.
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["validation_errors"] == []
