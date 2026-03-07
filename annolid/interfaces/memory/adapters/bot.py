import logging
from typing import Optional
from annolid.interfaces.memory.registry import get_memory_service, get_retrieval_service
from annolid.domain.memory.scopes import MemoryScope, MemoryCategory, MemorySource

logger = logging.getLogger(__name__)


class BotMemoryAdapter:
    """Adapter for Annolid Bot memory store/recall integration."""

    def __init__(self, agent_id: str):
        self.agent_id = agent_id

    def store_chat_memory(self, text: str, importance: float = 0.5) -> Optional[str]:
        service = get_memory_service()
        if not service:
            return None

        return service.store_memory(
            text=text,
            scope=MemoryScope.agent(self.agent_id),
            category=MemoryCategory.FACT,
            source=MemorySource.BOT,
            importance=importance,
        )

    def get_chat_context(self, query: str, top_k: int = 5) -> str:
        retrieval = get_retrieval_service()
        if not retrieval:
            return ""

        hits = retrieval.search_memory(
            query=query, top_k=top_k, scope=MemoryScope.agent(self.agent_id)
        )

        if not hits:
            return ""

        context_lines = [f"### Relevant context for Agent {self.agent_id}"]
        for hit in hits:
            context_lines.append(f"- {hit.text}")

        return "\n".join(context_lines)
