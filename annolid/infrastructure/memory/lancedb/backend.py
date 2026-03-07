import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from annolid.domain.memory.models import MemoryRecord, MemoryHit
from annolid.domain.memory.protocols import MemoryBackend
from annolid.infrastructure.memory.lancedb.schema import get_lancedb_schema
from annolid.infrastructure.memory.lancedb.config import LanceDBConfig
from annolid.infrastructure.memory.lancedb.embedder import Embedder

logger = logging.getLogger(__name__)


class LanceDBMemoryBackend(MemoryBackend):
    """LanceDB implementation of the MemoryBackend protocol."""

    TABLE_NAME = "memory"

    def __init__(self, config: LanceDBConfig):
        self.config = config
        self.embedder = Embedder(
            provider=config.embedding_provider,
            model=config.embedding_model,
            dimensions=config.vector_dim,
        )
        self._db = None
        self._table = None

        if self.config.enabled:
            self._init_db()

    def _init_db(self):
        try:
            import lancedb
        except ImportError:
            logger.warning(
                "LanceDB is not installed. LanceDBMemoryBackend will be disabled."
            )
            self.config.enabled = False
            return

        try:
            self.config.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._db = lancedb.connect(str(self.config.db_path))

            schema = get_lancedb_schema(self.config.vector_dim)
            if self.TABLE_NAME in self._db.table_names():
                self._table = self._db.open_table(self.TABLE_NAME)
            else:
                self._table = self._db.create_table(self.TABLE_NAME, schema=schema)
                # Create FTS index when table is first created
                try:
                    self._table.create_fts_index("text")
                except Exception as e:
                    logger.warning(
                        f"Failed to create FTS index on LanceDB memory table: {e}"
                    )
        except Exception as e:
            logger.error(f"Failed to initialize LanceDB memory backend: {e}")
            self.config.enabled = False

    def add(self, record: MemoryRecord) -> str:
        ids = self.add_many([record])
        if not ids:
            raise RuntimeError("Failed to add memory record to LanceDB backend.")
        return ids[0]

    def add_many(self, records: List[MemoryRecord]) -> List[str]:
        if not self.config.enabled or self._table is None:
            return []
        if not records:
            return []

        rows = []
        ids = []

        for rec in records:
            rec_id = str(uuid.uuid4())
            ids.append(rec_id)
            vector = self.embedder.embed(rec.text)

            rows.append(
                {
                    "id": rec_id,
                    "text": rec.text,
                    "vector": vector,
                    "scope": rec.scope,
                    "category": rec.category,
                    "source": rec.source,
                    "timestamp_ms": rec.timestamp_ms or int(time.time() * 1000),
                    "importance": float(rec.importance),
                    "token_count": rec.token_count or len(rec.text.split()),
                    "tags": rec.tags,
                    "metadata_json": json.dumps(rec.metadata),
                }
            )

        try:
            self._table.add(rows)
        except Exception as e:
            logger.error(f"Failed to add records to LanceDB: {e}")
            return []

        return ids

    def list_memories(
        self,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]:
        if not self.config.enabled or self._table is None:
            return []

        from annolid.infrastructure.memory.lancedb.retriever import LanceDBRetriever

        retriever = LanceDBRetriever(self._table, self.embedder, self.config)
        return retriever.list_memories(top_k=top_k, scope=scope, filters=filters)

    def export_rows(self) -> List[Dict[str, Any]]:
        if not self.config.enabled or self._table is None:
            return []
        rows = self._table.to_pandas().to_dict(orient="records")
        normalized: List[Dict[str, Any]] = []
        for row in rows:
            record: Dict[str, Any] = {}
            for key, value in row.items():
                if key == "vector":
                    continue
                record[key] = self._normalize_export_value(value)
            normalized.append(record)
        return normalized

    def export_jsonl(self, output_file: Path) -> int:
        rows = self.export_rows()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return len(rows)

    def reembed_all(self, embedder: Optional[Embedder] = None) -> Dict[str, int]:
        if not self.config.enabled or self._table is None:
            return {"success": 0, "failed": 0}
        active_embedder = embedder or self.embedder
        rows = self._table.to_pandas().to_dict(orient="records")
        success_count = 0
        fail_count = 0
        for row in rows:
            try:
                new_vector = active_embedder.embed(str(row["text"]))
                safe_id = str(row["id"]).replace("'", "''")
                self._table.update(
                    where=f"id = '{safe_id}'", values={"vector": new_vector}
                )
                success_count += 1
            except Exception as e:
                logger.error("Failed to re-embed memory %s: %s", row.get("id"), e)
                fail_count += 1
        return {"success": success_count, "failed": fail_count}

    def _normalize_export_value(self, value: Any) -> Any:
        if hasattr(value, "tolist"):
            return value.tolist()
        if hasattr(value, "item"):
            return value.item()
        if isinstance(value, list):
            return [self._normalize_export_value(item) for item in value]
        return value

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        scope: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[MemoryHit]:
        if not self.config.enabled or self._table is None:
            return []

        # Use the retriever for search since it has the complex logic
        # for semantic/bm25 fusion and recency ranking.
        # This will be delegated here to avoid complex circular dependencies.
        from annolid.infrastructure.memory.lancedb.retriever import LanceDBRetriever

        retriever = LanceDBRetriever(self._table, self.embedder, self.config)
        return retriever.search(query, top_k=top_k, scope=scope, filters=filters)

    def delete(self, memory_id: str) -> bool:
        if not self.config.enabled or self._table is None:
            return False

        try:
            safe_memory_id = memory_id.replace("'", "''")
            self._table.delete(f"id = '{safe_memory_id}'")
            return True
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id} from LanceDB: {e}")
            return False

    def update(self, memory_id: str, patch: Dict[str, Any]) -> bool:
        if not self.config.enabled or self._table is None:
            return False

        try:
            update_values = dict(patch)
            if "metadata" in update_values:
                update_values["metadata_json"] = json.dumps(
                    update_values.pop("metadata")
                )
            if "tags" in update_values and not isinstance(update_values["tags"], list):
                update_values["tags"] = list(update_values["tags"])
            safe_memory_id = memory_id.replace("'", "''")
            self._table.update(where=f"id = '{safe_memory_id}'", values=update_values)
            return True
        except Exception as e:
            logger.error(f"Failed to update memory {memory_id} in LanceDB: {e}")
            return False

    def stats(self, *, scope: Optional[str] = None) -> Dict[str, Any]:
        if not self.config.enabled or self._table is None:
            return {"enabled": False, "count": 0}

        try:
            if scope:
                safe_scope = scope.replace("'", "''")
                count = len(
                    self._table.search()
                    .where(f"scope = '{safe_scope}'", prefilter=True)
                    .to_list()
                )
            else:
                count = len(self._table)
            return {
                "enabled": True,
                "count": count,
                "scope": scope,
                "db_path": str(self.config.db_path),
            }
        except Exception as e:
            logger.error(f"Failed to get LanceDB stats: {e}")
            return {"enabled": False, "error": str(e)}

    def health_check(self) -> Dict[str, Any]:
        if not self.config.enabled:
            return {"status": "disabled"}

        try:
            # simple version query or select limit 1 to ensure connectivity
            self._table.head(1)
            return {"status": "healthy", "backend": "lancedb"}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}
