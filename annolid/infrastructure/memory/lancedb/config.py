import os
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LanceDBConfig:
    enabled: bool
    db_path: Path
    embedding_provider: str
    embedding_model: str
    vector_dim: int
    vector_weight: float
    bm25_weight: float
    rerank_provider: str

    @classmethod
    def from_env(cls) -> "LanceDBConfig":
        db_path_str = os.getenv(
            "ANNOLID_MEMORY_LANCEDB_PATH", "~/.annolid/memory/lancedb"
        )
        db_path = Path(db_path_str).expanduser()

        return cls(
            enabled=os.getenv("ANNOLID_MEMORY_ENABLED", "true").lower() == "true",
            db_path=db_path,
            embedding_provider=os.getenv("ANNOLID_MEMORY_EMBEDDING_PROVIDER", "jina"),
            embedding_model=os.getenv(
                "ANNOLID_MEMORY_EMBEDDING_MODEL", "jina-embeddings-v3"
            ),
            vector_dim=int(os.getenv("ANNOLID_MEMORY_VECTOR_DIM", "1024")),
            vector_weight=float(os.getenv("ANNOLID_MEMORY_VECTOR_WEIGHT", "0.65")),
            bm25_weight=float(os.getenv("ANNOLID_MEMORY_BM25_WEIGHT", "0.35")),
            rerank_provider=os.getenv("ANNOLID_MEMORY_RERANK_PROVIDER", "none"),
        )
