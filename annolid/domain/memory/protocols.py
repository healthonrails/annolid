from typing import Any, Dict, List, Optional, Protocol
from .models import MemoryRecord, MemoryHit


class MemoryBackend(Protocol):
    def add(self, record: MemoryRecord) -> str: ...

    def add_many(self, records: List[MemoryRecord]) -> List[str]: ...

    def list_memories(
        self,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]: ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]: ...

    def delete(self, memory_id: str) -> bool: ...

    def update(self, memory_id: str, patch: Dict[str, Any]) -> bool: ...

    def stats(self, *, scope: Optional[str] = None) -> Dict[str, Any]: ...

    def health_check(self) -> Dict[str, Any]: ...
