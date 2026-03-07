import sys
import logging

try:
    from annolid.interfaces.memory.registry import get_memory_backend
    from annolid.infrastructure.memory.lancedb.embedder import Embedder
    from annolid.infrastructure.memory.lancedb.config import LanceDBConfig
except ImportError:
    print(
        "Error: Could not import annolid modules. Ensure you're running from the project root."
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main(argv=None):
    _ = argv
    """
    Utility script to re-embed all LanceDB memory vectors.
    Useful when changing embedding providers (e.g., openai -> jina) or models.
    """
    logger.info("Initializing memory backend...")
    backend = get_memory_backend()
    if not backend:
        logger.error("Memory subsystem is not enabled or backend unavailable.")
        return 1
    if not hasattr(backend, "reembed_all") or not hasattr(backend, "export_rows"):
        logger.error("Backend does not appear to be LanceDB.")
        return 1

    total_rows = len(backend.export_rows())
    logger.info(f"Found {total_rows} memories to re-embed.")

    if total_rows == 0:
        logger.info("Nothing to do.")
        return 0

    # Initialize embedder directly to bypass backend encapsulation for batching
    config = LanceDBConfig.from_env()
    embedder = Embedder(config.embedding_provider, config.embedding_model)

    success_count = 0
    fail_count = 0

    # LanceDB doesn't allow direct vector updates easily in place,
    # so we delete and re-insert or use LanceDB's update semantics if supported.
    # The safest way is to read the data, delete the old records, and re-add them
    # to guarantee the new vector dimension matches the schema if it changed.
    # However, if the dimension changed, the *entire table schema* needs to be recreated.
    # We will assume for this simple script that the schema was already migrated or we
    # just update the vectors in place if dimensions are the same.

    # Actually, the most robust way across dimension changes is to read everything into memory,
    # drop the table, recreate it, and insert.

    logger.warning("This script will reconstruct the entire LanceDB 'memories' table.")
    user_input = input("Are you sure you want to proceed? (y/n): ")
    if user_input.lower() != "y":
        logger.info("Aborted.")
        return 0

    # Build list of MemoryRecords and manually hit the backend.add()
    # Or, faster: use the backend API to extract all data.

    # Note: If dimensions change, the standard backend initialization will fail
    # if the table exists with the old dimension.
    # Therefore, we just re-embed and update if dimension is the same,
    # OR if recreating, one should move the table manually first.

    logger.info("Generating new embeddings...")
    result = backend.reembed_all(embedder)
    success_count = int(result.get("success", 0))
    fail_count = int(result.get("failed", 0))
    logger.info(
        f"Re-embedding complete. Success: {success_count}, Failed: {fail_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
