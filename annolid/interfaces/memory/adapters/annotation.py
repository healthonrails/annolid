import logging
from typing import Optional
from annolid.interfaces.memory.registry import get_memory_service, get_context_service
from annolid.domain.memory.scopes import MemoryScope, MemoryCategory, MemorySource

logger = logging.getLogger(__name__)


class AnnotationMemoryAdapter:
    """Adapter for Annolid annotation session contextual memory."""

    def store_annotation_rule(
        self, dataset_id: str, text: str, importance: float = 0.9
    ) -> Optional[str]:
        service = get_memory_service()
        if not service:
            return None

        return service.store_memory(
            text=text,
            scope=MemoryScope.dataset(dataset_id),
            category=MemoryCategory.ANNOTATION_RULE,
            source=MemorySource.ANNOTATION,
            importance=importance,
        )

    def get_annotation_context(self, dataset_id: str, top_k: int = 5) -> str:
        service = get_context_service()
        if not service:
            return ""

        return service.build_annotation_context(dataset_id=dataset_id, top_k=top_k)
