from typing import Any, Dict, List, Optional
from annolid.domain.memory.models import MemoryRecord, MemoryHit
from annolid.domain.memory.protocols import MemoryBackend
from annolid.domain.memory.scopes import MemoryScope, MemoryCategory
from annolid.services.memory.memory_service import MemoryService
from annolid.services.memory.retrieval_service import RetrievalService
from annolid.services.memory.context_service import ContextService


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
        hits = self.search(query="", top_k=top_k, scope=scope, filters=filters)
        if hits:
            return hits
        rows: List[MemoryHit] = []
        for rid, rec in self.records.items():
            if scope and rec.scope != scope:
                continue
            if filters:
                match = True
                for k, v in filters.items():
                    if rec.metadata.get(k) != v and getattr(rec, k, None) != v:
                        match = False
                        break
                if not match:
                    continue
            rows.append(
                MemoryHit(
                    id=rid,
                    text=rec.text,
                    score=rec.importance,
                    scope=rec.scope,
                    category=rec.category,
                    source=rec.source,
                    importance=rec.importance,
                    timestamp_ms=rec.timestamp_ms or 0,
                    tags=rec.tags,
                    metadata=rec.metadata,
                )
            )
        return rows[:top_k]

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]:
        hits = []
        for rid, rec in self.records.items():
            if scope and rec.scope != scope:
                continue
            if filters:
                match = True
                for k, v in filters.items():
                    if rec.metadata.get(k) != v and getattr(rec, k, None) != v:
                        match = False
                        break
                if not match:
                    continue
            if query in rec.text:
                hits.append(
                    MemoryHit(
                        id=rid,
                        text=rec.text,
                        score=1.0,
                        scope=rec.scope,
                        category=rec.category,
                        source=rec.source,
                        importance=rec.importance,
                        timestamp_ms=rec.timestamp_ms or 0,
                        tags=rec.tags,
                        metadata=rec.metadata,
                    )
                )
        return hits[:top_k]

    def delete(self, memory_id: str) -> bool:
        if memory_id in self.records:
            del self.records[memory_id]
            return True
        return False

    def update(self, memory_id: str, patch: Dict[str, Any]) -> bool:
        if memory_id in self.records:
            rec = self.records[memory_id]
            for k, v in patch.items():
                if hasattr(rec, k):
                    setattr(rec, k, v)
                else:
                    rec.metadata[k] = v
            return True
        return False

    def stats(self, *, scope: Optional[str] = None) -> Dict[str, Any]:
        return {"count": len(self.records)}

    def health_check(self) -> Dict[str, Any]:
        return {"status": "ok"}


def test_memory_service_store():
    backend = MockBackend()
    service = MemoryService(backend)

    rid = service.store_project_note("proj1", "Test note", importance=0.8)
    assert rid == "mem_0"

    assert len(backend.records) == 1
    rec = backend.records[rid]
    assert rec.text == "Test note"
    assert rec.scope == MemoryScope.project("proj1")
    assert rec.category == MemoryCategory.PROJECT_NOTE


def test_context_service_build():
    backend = MockBackend()
    mem_service = MemoryService(backend)
    ret_service = RetrievalService(backend)
    ctx_service = ContextService(ret_service)

    mem_service.store_annotation_rule("data1", "Use tail_base instead of tailroot")
    mem_service.store_annotation_rule("data1", "Include occlusion flag on paws")
    mem_service.store_annotation_rule("data2", "Only map head area")

    ctx = ctx_service.build_annotation_context("data1", query="tail")

    # We used a basic query "in" in mock
    assert "Use tail_base instead of tailroot" in ctx
    assert "Include occlusion flag on paws" not in ctx  # query didn't match
    assert "Only map head area" not in ctx  # wrong scope


def test_context_service_build_without_query_lists_scoped_hits():
    backend = MockBackend()
    mem_service = MemoryService(backend)
    ret_service = RetrievalService(backend)
    ctx_service = ContextService(ret_service)

    mem_service.store_project_note("proj1", "Use Cutie for tracking")
    mem_service.store_project_note("proj1", "Export at 10 fps")

    ctx = ctx_service.build_project_context("proj1")

    assert "Use Cutie for tracking" in ctx
    assert "Export at 10 fps" in ctx
