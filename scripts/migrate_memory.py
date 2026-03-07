import argparse
import json
import logging
from pathlib import Path
import sys

# Try to import annolid to make sure the environment is set up
try:
    from annolid.domain.memory.scopes import MemoryScope, MemoryCategory
    from annolid.infrastructure.memory.lancedb.migration import import_records
    from annolid.domain.memory.models import MemoryRecord
    from annolid.interfaces.memory.registry import get_memory_backend
except ImportError:
    print(
        "Error: Could not import annolid. Ensure you run this from the project root and have the dependencies installed."
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def migrate_legacy_memories(source_dir: Path):
    """
    Scaffolding for migrating legacy Annolid memories into LanceDB.

    This expects a directory of JSON files representing legacy memories, e.g.
    {
      "text": "The user prefers high contrast mode",
      "scope": "workspace:123",
      "importance": 0.8
    }
    """
    if not source_dir.exists() or not source_dir.is_dir():
        logger.error(f"Source directory not found: {source_dir}")
        return 1

    backend = get_memory_backend()
    if not backend:
        logger.error("Memory subsystem is not enabled or could not be initialized.")
        return 1

    records = []
    skipped_count = 0

    for file_path in source_dir.glob("*.json"):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if "text" not in data:
                logger.warning(f"File {file_path.name} missing 'text' field. Skipping.")
                skipped_count += 1
                continue

            # Map legacy fields to new taxonomy where possible
            text = data["text"]
            scope = data.get("scope", MemoryScope.GLOBAL)
            category = data.get("category", MemoryCategory.OTHER)
            importance = float(data.get("importance", 0.5))

            records.append(
                MemoryRecord(
                    text=text,
                    scope=scope,
                    category=category,
                    importance=importance,
                    metadata={"migrated_from_file": file_path.name},
                )
            )
        except Exception as e:
            skipped_count += 1
            logger.error(f"Error processing {file_path.name}: {e}")

    result = import_records(backend, records)
    logger.info(
        "Migration complete. Imported: %s, Failed: %s, Skipped: %s",
        result.imported,
        result.failed,
        skipped_count,
    )
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Migrate legacy Annolid memories to LanceDB."
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to the directory containing legacy JSON memory files.",
    )

    args = parser.parse_args(argv)

    # Run the migration
    return migrate_legacy_memories(Path(args.source_dir))


if __name__ == "__main__":
    raise SystemExit(main())
