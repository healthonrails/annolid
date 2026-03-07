import json
import logging
import time
import math
import os
from typing import Any, Dict, List, Optional
from annolid.domain.memory.models import MemoryHit
from annolid.infrastructure.memory.lancedb.config import LanceDBConfig
from annolid.infrastructure.memory.lancedb.embedder import Embedder

from annolid.infrastructure.memory.lancedb.reranker import Reranker

logger = logging.getLogger(__name__)


class LanceDBRetriever:
    """Handles hybrid search, recency boosting, and scoring in LanceDB."""

    def __init__(self, table: Any, embedder: Embedder, config: LanceDBConfig):
        self._table = table
        self.embedder = embedder
        self.config = config
        self.reranker = Reranker(
            provider=config.rerank_provider,
            api_key=os.getenv("ANNOLID_MEMORY_RERANK_API_KEY", ""),
        )

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]:
        """
        Executes a hybrid search combining vector similarity and full-text search (BM25 if supported).
        """
        query_vector = self.embedder.embed(query)
        filter_str = self._build_filter_str(scope=scope, filters=filters)

        # Fetch more candidates for reranking
        candidate_pool = top_k * 3

        try:
            # 1. Vector Search
            vector_query = self._table.search(query_vector)
            if filter_str:
                vector_query = vector_query.where(filter_str, prefilter=True)
            vector_results_df = vector_query.limit(candidate_pool).to_pandas()
            vector_hits = (
                vector_results_df.to_dict("records")
                if not vector_results_df.empty
                else []
            )

            # 2. Text Search (BM25 fallback)
            text_hits = []
            try:
                # LanceDB v0.6+ FTS
                text_query = self._table.search(query, query_type="fts")
                if filter_str:
                    text_query = text_query.where(filter_str)
                text_results_df = text_query.limit(candidate_pool).to_pandas()
                text_hits = (
                    text_results_df.to_dict("records")
                    if not text_results_df.empty
                    else []
                )
            except Exception:
                pass  # FTS index might not exist yet or no tantivy

            # Combine and Score
            merged = self._fuse_results(vector_hits, text_hits)

            merged = self._apply_post_filters(merged, filters=filters)
            # Apply Recency Boost & Length Normalization
            scored = self._apply_recency_and_length_boost(merged)

            # Sort by score first to prepare for MMR
            scored.sort(key=lambda x: x["_final_score"], reverse=True)

            # Apply MMR (Maximal Marginal Relevance) for diversity
            # We don't have the raw vectors here easily without merging back, so we do a fast Jaccard text overlap MMR
            diversity_penalty = 0.25  # How much to penalize overlapping vocabulary
            mmr_results = []

            for candidate in scored:
                if not mmr_results:
                    mmr_results.append(candidate)
                    continue

                if len(mmr_results) >= top_k:
                    break

                # Calculate simple word overlap penalty against currently selected results
                cand_words = set(candidate["text"].lower().split())
                max_overlap = 0.0
                if cand_words:
                    for selected in mmr_results:
                        sel_words = set(selected["text"].lower().split())
                        overlap = len(cand_words.intersection(sel_words)) / len(
                            cand_words
                        )
                        if overlap > max_overlap:
                            max_overlap = overlap

                # Adjust final score based on highest overlap (penalty)
                candidate["_mmr_score"] = candidate["_final_score"] - (
                    max_overlap * diversity_penalty
                )

                # Re-sort candidates list conceptually to pick next best (greedy MMR)
                # Since candidates are already sorted by base score, subtracting penalty is a good approx.
                mmr_results.append(candidate)

            # Final re-sort by MMR score
            mmr_results.sort(
                key=lambda x: x.get("_mmr_score", x["_final_score"]), reverse=True
            )
            top_results = mmr_results[:top_k]

            hits = [
                self._row_to_hit(row, score=float(row["_final_score"]))
                for row in top_results
            ]

            # Rerank if configured
            if self.config.rerank_provider != "none":
                hits = self.reranker.rerank(query, hits, top_k)

            return list(sorted(hits, key=lambda x: x.score, reverse=True))
        except Exception as e:
            logger.error(f"Memory retrieval failed: {e}")
            return []

    def list_memories(
        self,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]:
        filter_str = self._build_filter_str(scope=scope, filters=filters)
        try:
            query = self._table.search()
            if filter_str:
                query = query.where(filter_str, prefilter=True)
            results_df = query.limit(top_k).to_pandas()
            rows = results_df.to_dict("records") if not results_df.empty else []
            filtered = self._apply_post_filters(rows, filters=filters)
            filtered.sort(
                key=lambda row: (
                    float(row.get("importance", 0.0)),
                    int(row.get("timestamp_ms", 0)),
                ),
                reverse=True,
            )
            return [
                self._row_to_hit(row, score=float(row.get("importance", 0.0)))
                for row in filtered[:top_k]
            ]
        except Exception as e:
            logger.error(f"Memory listing failed: {e}")
            return []

    def _build_filter_str(
        self, *, scope: Optional[str], filters: Optional[Dict[str, Any]]
    ) -> Optional[str]:
        filter_parts = []
        if scope:
            safe_scope = scope.replace("'", "''")
            filter_parts.append(f"scope = '{safe_scope}'")
        if filters:
            for key, value in filters.items():
                # metadata_json cannot be reliably filtered in SQL string form here.
                if key.startswith("metadata.") or key == "metadata":
                    continue
                if isinstance(value, str):
                    safe_value = value.replace("'", "''")
                    filter_parts.append(f"{key} = '{safe_value}'")
                else:
                    filter_parts.append(f"{key} = {value}")
        return " AND ".join(filter_parts) if filter_parts else None

    def _apply_post_filters(
        self, rows: List[Dict], *, filters: Optional[Dict[str, Any]]
    ) -> List[Dict]:
        if not filters:
            return rows
        filtered: List[Dict] = []
        for row in rows:
            metadata: Dict[str, Any] = {}
            try:
                metadata = json.loads(row.get("metadata_json", "{}"))
            except Exception:
                metadata = {}
            match = True
            for key, expected in filters.items():
                if key == "metadata":
                    if not isinstance(expected, dict):
                        match = False
                        break
                    for mk, mv in expected.items():
                        if metadata.get(mk) != mv:
                            match = False
                            break
                    if not match:
                        break
                    continue
                if key.startswith("metadata."):
                    meta_key = key.split(".", 1)[1]
                    if metadata.get(meta_key) != expected:
                        match = False
                        break
                    continue
                # Non-metadata keys are already handled by LanceDB where possible,
                # but we keep this for compatibility and fallback behavior.
                if row.get(key) != expected:
                    match = False
                    break
            if match:
                filtered.append(row)
        return filtered

    def _row_to_hit(self, row: Dict[str, Any], *, score: float) -> MemoryHit:
        metadata: Dict[str, Any] = {}
        try:
            metadata = json.loads(row.get("metadata_json", "{}"))
        except Exception:
            metadata = {}

        return MemoryHit(
            id=row["id"],
            text=row["text"],
            score=score,
            scope=row["scope"],
            category=row["category"],
            source=row["source"],
            importance=float(row["importance"]),
            timestamp_ms=int(row["timestamp_ms"]),
            tags=list(row["tags"]) if row.get("tags") is not None else [],
            metadata=metadata,
        )

    def _fuse_results(
        self, vector_hits: List[Dict], text_hits: List[Dict]
    ) -> List[Dict]:
        """Simple RRF or max-fusion."""
        merged_map = {}

        # Normalize distances to score for vectors
        for row in vector_hits:
            dist = row.get("_distance", 1.0)
            base_score = max(0.0, 1.0 - dist)
            row["_vector_score"] = base_score
            merged_map[row["id"]] = row

        # Add basic score for FTS based on rank
        for i, row in enumerate(text_hits):
            if row["id"] not in merged_map:
                row["_vector_score"] = 0.0
                merged_map[row["id"]] = row
            # basic decaying BM25 weight approximation: 1.0 -> 0.1
            merged_map[row["id"]]["_text_score"] = max(0.1, 1.0 - (i * 0.1))

        # Weights
        v_weight = self.config.vector_weight
        t_weight = self.config.bm25_weight

        for row in merged_map.values():
            v_score = row.get("_vector_score", 0.0)
            t_score = row.get("_text_score", 0.0)
            row["_fused_score"] = (v_weight * v_score) + (t_weight * t_score)

        return list(merged_map.values())

    def _apply_recency_and_length_boost(self, rows: List[Dict]) -> List[Dict]:
        """Applies exponential decay recency boost and length normalization to scores."""
        now_ms = int(time.time() * 1000)
        ms_per_day = 24 * 60 * 60 * 1000
        half_life_days = 14
        recency_weight = 0.08

        for row in rows:
            # Recency
            ts = row.get("timestamp_ms", now_ms)
            age_days = max(0, (now_ms - ts) / ms_per_day)
            boost = recency_weight * math.exp(-age_days / half_life_days)

            # Length penalty (penalize extreme shortness or extreme length)
            text_len = len(row.get("text", ""))
            length_penalty = 0.0
            if text_len < 20:  # Spam or too short
                length_penalty = -0.1
            elif text_len > 2000:  # Too long, dilute score
                length_penalty = -0.05

            row["_final_score"] = row["_fused_score"] + boost + length_penalty

        return rows
