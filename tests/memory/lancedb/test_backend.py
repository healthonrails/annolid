import pytest

pytest.importorskip("lancedb")
pytest.importorskip("pyarrow")
from pathlib import Path
import tempfile
from typing import Generator
from annolid.domain.memory.models import MemoryRecord
from annolid.infrastructure.memory.lancedb.config import LanceDBConfig
from annolid.infrastructure.memory.lancedb.backend import LanceDBMemoryBackend


@pytest.fixture
def temp_lancedb_config() -> Generator[LanceDBConfig, None, None]:
    with tempfile.TemporaryDirectory() as temp_dir:
        config = LanceDBConfig(
            enabled=True,
            db_path=Path(temp_dir) / "test.lance",
            embedding_provider="none",  # Will use empty vectors
            embedding_model="dummy",
            vector_dim=10,  # small vectors
            vector_weight=0.65,
            bm25_weight=0.35,
            rerank_provider="none",
        )
        yield config


def test_lancedb_backend_lifecycle(temp_lancedb_config: LanceDBConfig):
    backend = LanceDBMemoryBackend(temp_lancedb_config)
    assert backend.config.enabled

    # Health check
    health = backend.health_check()
    assert health["status"] == "healthy"

    # Empty stats
    stats = backend.stats()
    assert stats["count"] == 0

    # Add record
    rec1 = MemoryRecord(
        text="This is a project decision notice",
        scope="project:123",
        category="decision",
        importance=0.9,
    )
    mid = backend.add(rec1)
    assert mid is not None

    stats = backend.stats()
    assert stats["count"] == 1
    assert backend.stats(scope="project:123")["count"] == 1

    # Search (Using vector fallback search since text searching requires tantivy but we skip actual text assertion here if missing)
    # The record should still be retrieved
    hits = backend.search("decision", scope="project:123")
    assert len(hits) == 1
    assert hits[0].text == "This is a project decision notice"

    # Delete
    deleted = backend.delete(mid)
    assert deleted
    assert backend.stats()["count"] == 0
