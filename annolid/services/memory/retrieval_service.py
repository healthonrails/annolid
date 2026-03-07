import logging
from typing import Any, Dict, List, Optional
from annolid.domain.memory.models import MemoryHit
from annolid.domain.memory.protocols import MemoryBackend


logger = logging.getLogger(__name__)


class RetrievalService:
    """Orchestrates search query routing and memory retrieval."""

    def __init__(self, backend: MemoryBackend):
        self._backend = backend

    def search_memory(
        self,
        query: str,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]:
        """Execute a search or fall back to scoped listing for blank queries."""
        if not query.strip():
            return self.list_memories(top_k=top_k, scope=scope, filters=filters)

        return self._backend.search(
            query=query,
            top_k=top_k,
            scope=scope,
            filters=filters,
        )

    def list_memories(
        self,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]:
        return self._backend.list_memories(top_k=top_k, scope=scope, filters=filters)
