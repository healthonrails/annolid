import logging
from annolid.services.memory.retrieval_service import RetrievalService
from annolid.domain.memory.scopes import MemoryScope, MemoryCategory

logger = logging.getLogger(__name__)


class ContextService:
    """Builds contextual memory bundles for callers."""

    def __init__(self, retrieval_service: RetrievalService):
        self._retrieval_service = retrieval_service

    def build_project_context(
        self, project_id: str, query: str = "", top_k: int = 10
    ) -> str:
        """Retrieves project memory and formats it as a readable, categorized context summary."""
        scope = MemoryScope.project(project_id)
        hits = self._retrieve_hits(query=query, top_k=top_k, scope=scope)
        if not hits:
            return ""

        # Group hits by category for a richer bundle
        grouped_hits = {}
        for hit in hits:
            cat = hit.category.upper()
            if cat not in grouped_hits:
                grouped_hits[cat] = []
            grouped_hits[cat].append(hit)

        context_lines = []
        context_lines.append(f"### Memory Context for Project {project_id}")

        # Define a helpful order of rendering
        order_pref = [
            MemoryCategory.SETTING.upper(),
            MemoryCategory.ANNOTATION_RULE.upper(),
            MemoryCategory.WORKFLOW_RECIPE.upper(),
            MemoryCategory.TROUBLESHOOTING.upper(),
            MemoryCategory.PROJECT_NOTE.upper(),
        ]

        for pref_cat in order_pref:
            if pref_cat in grouped_hits:
                context_lines.append(f"\n#### {pref_cat.replace('_', ' ')}")
                for hit in grouped_hits[pref_cat]:
                    context_lines.append(f"- {hit.text}")
                del grouped_hits[pref_cat]

        # Render any remaining categories
        for cat, remaining_hits in grouped_hits.items():
            context_lines.append(f"\n#### {cat.replace('_', ' ')}")
            for hit in remaining_hits:
                context_lines.append(f"- {hit.text}")

        return "\n".join(context_lines)

    def build_annotation_context(
        self, dataset_id: str, query: str = "", top_k: int = 5
    ) -> str:
        """Retrieves annotation rules/conventions for a given dataset."""
        scope = MemoryScope.dataset(dataset_id)
        hits = self._retrieve_hits(
            query=query,
            top_k=top_k,
            scope=scope,
            filters={"category": MemoryCategory.ANNOTATION_RULE},
        )
        if not hits:
            return ""

        context_lines = []
        context_lines.append(f"### Annotation Rules for Dataset {dataset_id}")
        for hit in hits:
            context_lines.append(f"- {hit.text} (Score: {hit.score:.2f})")

        return "\n".join(context_lines)

    def _retrieve_hits(self, *, query: str, top_k: int, scope: str, filters=None):
        if query.strip():
            return self._retrieval_service.search_memory(
                query=query,
                top_k=top_k,
                scope=scope,
                filters=filters,
            )
        return self._retrieval_service.list_memories(
            top_k=top_k,
            scope=scope,
            filters=filters,
        )
