from typing import Any, Dict, List, Optional

from annolid.domain.memory.models import MemoryHit, MemoryRecord
from annolid.domain.memory.protocols import MemoryBackend
from annolid.domain.memory.scopes import MemoryCategory, MemoryScope, MemorySource
from annolid.services.memory.memory_service import MemoryService
from annolid.services.memory.persistence_service import PersistenceService


class MockBackend(MemoryBackend):
    def __init__(self):
        self.records: Dict[str, MemoryRecord] = {}

    def add(self, record: MemoryRecord) -> str:
        new_id = f"mem_{len(self.records)}"
        self.records[new_id] = record
        return new_id

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
        return memory_id in self.records

    def update(self, memory_id: str, patch: Dict[str, Any]) -> bool:
        _ = (memory_id, patch)
        return True

    def stats(self, *, scope: Optional[str] = None) -> Dict[str, Any]:
        _ = scope
        return {"count": len(self.records)}

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok"}


def test_persistence_service_save_project_note() -> None:
    service = MemoryService(MockBackend())
    persistence = PersistenceService(service)

    memory_id = persistence.save_project_note("p1", "Use Cutie for tracking")
    assert memory_id == "mem_0"
    record = service._backend.records[memory_id]  # type: ignore[attr-defined]
    assert record.scope == MemoryScope.project("p1")
    assert record.category == MemoryCategory.PROJECT_NOTE
    assert record.source == MemorySource.PROJECT


def test_persistence_service_save_settings_snapshot_context() -> None:
    service = MemoryService(MockBackend())
    persistence = PersistenceService(service)

    memory_id = persistence.save_settings_snapshot(
        scope=MemoryScope.workspace("w1"),
        description="Known-good export setup",
        settings={"fps": 10},
        context="IR videos",
    )
    record = service._backend.records[memory_id]  # type: ignore[attr-defined]
    assert record.category == MemoryCategory.SETTING
    assert record.metadata["context"] == "IR videos"
