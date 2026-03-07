from typing import Any


def get_lancedb_schema(vector_dim: int) -> Any:
    """Returns the PyArrow schema for LanceDB memory storage."""
    import pyarrow as pa

    return pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),
            pa.field("scope", pa.string()),
            pa.field("category", pa.string()),
            pa.field("source", pa.string()),
            pa.field("timestamp_ms", pa.int64()),
            pa.field("importance", pa.float32()),
            pa.field("token_count", pa.int32()),
            pa.field("tags", pa.list_(pa.string())),
            pa.field("metadata_json", pa.string()),
        ]
    )
