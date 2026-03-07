import pytest

pytest.importorskip("lancedb")
pytest.importorskip("pyarrow")

from pathlib import Path
import tempfile

from annolid.domain.memory.models import MemoryRecord
from annolid.infrastructure.memory.lancedb.backend import LanceDBMemoryBackend
from annolid.infrastructure.memory.lancedb.config import LanceDBConfig


def _make_backend() -> LanceDBMemoryBackend:
    temp_dir = tempfile.TemporaryDirectory()
    config = LanceDBConfig(
        enabled=True,
        db_path=Path(temp_dir.name) / "test.lance",
        embedding_provider="none",
        embedding_model="dummy",
        vector_dim=10,
        vector_weight=0.65,
        bm25_weight=0.35,
        rerank_provider="none",
    )
    backend = LanceDBMemoryBackend(config)
    backend._temp_dir = temp_dir  # type: ignore[attr-defined]
    return backend


def test_lancedb_backend_export_rows_and_jsonl(tmp_path) -> None:
    backend = _make_backend()
    backend.add(
        MemoryRecord(
            text="Project note",
            scope="project:1",
            category="project_note",
            metadata={"project_id": "1"},
        )
    )

    rows = backend.export_rows()
    assert len(rows) == 1
    assert rows[0]["text"] == "Project note"
    assert "vector" not in rows[0]

    output = tmp_path / "memory.jsonl"
    assert backend.export_jsonl(output) == 1
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert "Project note" in lines[0]


def test_lancedb_backend_reembed_all_returns_counts() -> None:
    backend = _make_backend()
    backend.add(MemoryRecord(text="Need embedding refresh"))

    result = backend.reembed_all()
    assert result["success"] == 1
    assert result["failed"] == 0
