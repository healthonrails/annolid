from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.tools.artifacts import FileArtifactStore, content_hash


def test_artifact_store_writes_json_and_ndjson(tmp_path: Path) -> None:
    store = FileArtifactStore(base_dir=tmp_path, run_id="run1")
    json_path = store.resolve("detector", "summary.json")
    store.write_json(json_path, {"ok": True})
    assert json_path.exists()
    assert json.loads(json_path.read_text(encoding="utf-8"))["ok"] is True

    ndjson_path = store.resolve("events", "events.ndjson")
    store.write_ndjson(ndjson_path, [{"a": 1}, {"b": 2}])
    lines = ndjson_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_content_hash_stable_for_dict_order() -> None:
    a = {"x": 1, "y": 2}
    b = {"y": 2, "x": 1}
    assert content_hash(a) == content_hash(b)
