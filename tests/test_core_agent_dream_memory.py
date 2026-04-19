from __future__ import annotations

import json
from pathlib import Path

from annolid.core.agent.dream_memory import DreamMemoryManager


def _write_history(path: Path, lines: list[str]) -> None:
    body = "# History\n\n" + "\n".join(lines) + "\n"
    path.write_text(body, encoding="utf-8")


def test_dream_run_initializes_cursor_without_processing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    history = workspace / "memory" / "HISTORY.md"
    history.parent.mkdir(parents=True, exist_ok=True)
    _write_history(
        history,
        [
            "[2026-04-17 10:00] user asked for tracking help",
            "[2026-04-17 10:01] assistant suggested /track",
        ],
    )
    manager = DreamMemoryManager(workspace)

    result = manager.run()
    assert result.ok is True
    assert result.did_work is False
    assert result.status == "initialized"
    assert result.cursor_start == 2
    assert result.cursor_end == 2


def test_dream_run_writes_phase_artifacts_and_dream_diary(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    history = memory_dir / "HISTORY.md"
    memory_dir.mkdir(parents=True, exist_ok=True)
    _write_history(
        history,
        [
            "[2026-04-17 10:00] baseline entry",
        ],
    )
    manager = DreamMemoryManager(workspace)
    _ = manager.run()

    _write_history(
        history,
        [
            "[2026-04-17 10:00] baseline entry",
            "[2026-04-17 10:02] tracked mouse behavior bouts",
            "[2026-04-17 10:03] tracked mouse behavior bouts",
            "[2026-04-17 10:04] updated aggression labels",
        ],
    )
    result = manager.run(max_batch_entries=10, initialize_cursor_to_end=False)

    assert result.ok is True
    assert result.did_work is True
    assert result.processed_entries == 3
    assert result.cursor_start == 1
    assert result.cursor_end == 4

    dreams = (memory_dir / "DREAMS.md").read_text(encoding="utf-8")
    assert "## Light Sleep (managed)" in dreams
    assert "## REM Sleep (managed)" in dreams
    assert "## Deep Sleep (managed)" in dreams

    ingestion = json.loads((memory_dir / ".dreams" / "ingestion.json").read_text())
    assert ingestion["cursor"] == 4
    recall_store = json.loads(
        (memory_dir / ".dreams" / "recall-store.json").read_text()
    )
    assert "items" in recall_store

    rows = manager.list_runs(limit=5)
    assert rows
    assert rows[0]["run_id"] == result.run_id
    assert rows[0]["phases"]["light"]["staged_count"] == 3


def test_dream_deep_phase_promotes_scored_candidates(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    history = memory_dir / "HISTORY.md"
    memory_dir.mkdir(parents=True, exist_ok=True)
    _write_history(history, ["[2026-04-17 10:00] baseline"])
    manager = DreamMemoryManager(workspace)
    _ = manager.run()

    _write_history(
        history,
        [
            "[2026-04-17 10:00] baseline",
            "[2026-04-17 10:01] aggression slap in face near tunnel",
            "[2026-04-17 10:02] aggression slap in face near tunnel",
            "[2026-04-17 10:03] run away after chase",
        ],
    )
    result = manager.run(max_batch_entries=10, initialize_cursor_to_end=False)
    assert result.ok is True

    memory_text = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "aggression slap in face near tunnel" in memory_text

    run_row = manager.get_run(result.run_id)
    assert run_row is not None
    assert int(run_row["phases"]["deep"]["promoted_count"]) >= 1


def test_dream_does_not_duplicate_existing_promoted_memory(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    history = memory_dir / "HISTORY.md"
    memory_dir.mkdir(parents=True, exist_ok=True)
    _write_history(history, ["[2026-04-17 10:00] baseline"])
    manager = DreamMemoryManager(workspace)
    _ = manager.run()

    _write_history(
        history,
        [
            "[2026-04-17 10:00] baseline",
            "[2026-04-17 10:01] aggression slap in face near tunnel",
            "[2026-04-17 10:02] aggression slap in face near tunnel",
        ],
    )
    first = manager.run(max_batch_entries=10, initialize_cursor_to_end=False)
    assert first.ok is True
    first_text = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert first_text.count("aggression slap in face near tunnel") == 1

    _write_history(
        history,
        [
            "[2026-04-17 10:00] baseline",
            "[2026-04-17 10:01] aggression slap in face near tunnel",
            "[2026-04-17 10:02] aggression slap in face near tunnel",
            "[2026-04-17 10:03] aggression slap in face near tunnel",
            "[2026-04-17 10:04] aggression slap in face near tunnel",
        ],
    )
    second = manager.run(max_batch_entries=10, initialize_cursor_to_end=False)
    assert second.ok is True
    second_text = (memory_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert second_text.count("aggression slap in face near tunnel") == 1


def test_dream_active_lock_blocks_parallel_run(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    history = memory_dir / "HISTORY.md"
    memory_dir.mkdir(parents=True, exist_ok=True)
    _write_history(
        history,
        [
            "[2026-04-17 10:00] base",
            "[2026-04-17 10:01] new item",
        ],
    )
    manager = DreamMemoryManager(workspace)
    _ = manager.run(initialize_cursor_to_end=False)
    _write_history(
        history,
        [
            "[2026-04-17 10:00] base",
            "[2026-04-17 10:01] new item",
            "[2026-04-17 10:02] another item",
        ],
    )
    lock_payload = '{"pid": 123, "acquired_at": "2099-01-01T00:00:00+00:00"}\n'
    (memory_dir / ".dreams" / "lock").write_text(lock_payload, encoding="utf-8")

    result = manager.run(max_batch_entries=10, initialize_cursor_to_end=False)
    assert result.ok is False
    assert result.status == "locked"


def test_dream_restore_reverts_history_snapshot(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    memory_dir = workspace / "memory"
    history = memory_dir / "HISTORY.md"
    memory_dir.mkdir(parents=True, exist_ok=True)
    _write_history(
        history,
        [
            "[2026-04-17 10:00] base",
        ],
    )
    manager = DreamMemoryManager(workspace)
    _ = manager.run()
    _write_history(
        history,
        [
            "[2026-04-17 10:00] base",
            "[2026-04-17 10:01] new item",
            "[2026-04-17 10:02] new item",
        ],
    )
    run_result = manager.run(max_batch_entries=10, initialize_cursor_to_end=False)
    assert run_result.did_work is True
    assert "[DREAM] consolidated" in history.read_text(encoding="utf-8")

    restored = manager.restore(run_result.run_id)
    assert restored.ok is True
    restored_text = history.read_text(encoding="utf-8")
    assert "[DREAM] consolidated" not in restored_text
