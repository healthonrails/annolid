import os
import requests
import logging
from typing import List
from annolid.domain.memory.models import MemoryHit

logger = logging.getLogger(__name__)


class Reranker:
    """Optional reranking logic for memory search results."""

    def __init__(
        self,
        provider: str = "jina",
        model: str = "jina-reranker-v2-base-multilingual",
        api_key: str = "",
    ):
        self.provider = provider
        self.model = model
        self.api_key = api_key or os.getenv("ANNOLID_MEMORY_RERANK_API_KEY", "")

    def rerank(self, query: str, hits: List[MemoryHit], top_k: int) -> List[MemoryHit]:
        """Reranks the search results based on the query."""
        if not hits or self.provider == "none" or not self.api_key:
            return hits[:top_k]

        if self.provider == "jina":
            return self._rerank_jina(query, hits, top_k)

        logger.warning(
            f"Unsupported rerank provider: {self.provider}. Skipping rerank."
        )
        return hits[:top_k]

    def _rerank_jina(
        self, query: str, hits: List[MemoryHit], top_k: int
    ) -> List[MemoryHit]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }

            documents = [hit.text for hit in hits]

            payload = {
                "model": self.model,
                "query": query,
                "documents": documents,
                "top_n": top_k,
            }

            resp = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers=headers,
                json=payload,
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()

            reranked_results = data.get("results", [])

            # Map back to MemoryHit objects and update scores
            final_hits = []
            for result in reranked_results:
                idx = result["index"]
                score = result["relevance_score"]
                hit = hits[idx]
                hit.score = score
                final_hits.append(hit)

            return final_hits

        except Exception as e:
            logger.error(f"Failed to rerank with Jina: {e}")
            return hits[:top_k]
