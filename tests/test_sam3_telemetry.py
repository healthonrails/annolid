from __future__ import annotations

import json
from pathlib import Path

from annolid.segmentation.SAM.sam3.telemetry import (
    Sam3TelemetrySink,
    build_config_snapshot,
    build_window_telemetry_entry,
    memory_snapshot,
)


def test_build_window_telemetry_entry_contract() -> None:
    entry = build_window_telemetry_entry(
        window_index=3,
        window_start_idx=40,
        window_end_idx=50,
        local_mask_counts={40: 2, 41: 0, 42: 1},
        boundary_empty_skips=2,
        latency_ms=8.5,
        reacquired_frames=1,
    )
    assert entry["window_index"] == 3
    assert entry["start"] == 40
    assert entry["end"] == 50
    assert entry["frames"] == 3
    assert entry["nonzero_frames"] == 2
    assert entry["zero_mask_frames"] == 1
    assert entry["boundary_empty_skips"] == 2
    assert entry["reacquired_frames"] == 1


def test_config_snapshot_keeps_expected_keys() -> None:
    snapshot = build_config_snapshot(
        {
            "max_num_objects": 16,
            "multiplex_count": 8,
            "device": "cpu",
            "unknown_key": "ignored",
        }
    )
    assert snapshot["max_num_objects"] == 16
    assert snapshot["multiplex_count"] == 8
    assert snapshot["device"] == "cpu"
    assert "unknown_key" not in snapshot


def test_telemetry_sink_writes_jsonl_rows(tmp_path: Path) -> None:
    out = tmp_path / "sam3_telemetry.jsonl"
    sink = Sam3TelemetrySink(jsonl_path=out)
    sink.emit("run_start", {"mode": "offline", "config_snapshot": {"device": "cpu"}})
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["event"] == "run_start"
    assert rows[0]["mode"] == "offline"
    assert rows[0]["config_snapshot"]["device"] == "cpu"
    assert "rss_mb" in rows[0]


def test_memory_snapshot_handles_missing_resource_module(monkeypatch) -> None:
    monkeypatch.setattr(
        "annolid.segmentation.SAM.sam3.telemetry.resource",
        None,
    )
    payload = memory_snapshot()
    assert "rss_mb" in payload
    assert payload["rss_mb"] == 0.0
