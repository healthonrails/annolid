from typing import Any, Dict, List, Optional

from annolid.domain.memory.models import MemoryHit, MemoryRecord
from annolid.domain.memory.protocols import MemoryBackend
from annolid.infrastructure.memory.lancedb.migration import import_records


class _DummyBackend(MemoryBackend):
    def __init__(self):
        self.items = []

    def add(self, record: MemoryRecord) -> str:
        self.items.append(record)
        return str(len(self.items))

    def add_many(self, records: List[MemoryRecord]) -> List[str]:
        return [self.add(r) for r in records]

    def list_memories(
        self,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]:
        _ = (top_k, scope, filters)
        return []

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]:
        _ = (query, top_k, scope, filters)
        return []

    def delete(self, memory_id: str) -> bool:
        _ = memory_id
        return True

    def update(self, memory_id: str, patch: Dict[str, Any]) -> bool:
        _ = (memory_id, patch)
        return True

    def stats(self, *, scope: Optional[str] = None) -> Dict[str, Any]:
        _ = scope
        return {"count": len(self.items)}

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok"}


def test_import_records_counts_imported() -> None:
    backend = _DummyBackend()
    result = import_records(
        backend,
        [
            MemoryRecord(text="one"),
            MemoryRecord(text="two"),
        ],
    )
    assert result.imported == 2
    assert result.failed == 0
