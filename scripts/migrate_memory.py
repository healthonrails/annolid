import argparse
import logging
from pathlib import Path
import sys

# Try to import annolid to make sure the environment is set up
try:
    from annolid.infrastructure.memory.lancedb.migration import (
        collect_legacy_records,
        import_records,
    )
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
    Migrate legacy Annolid memory sources into the configured backend.

    Source directory is scanned recursively for:
    - JSON memory records with top-level ``text``
    - ``memory/MEMORY.md`` and ``memory/HISTORY.md``
    - ``project.annolid.json|yaml|yml``
    """
    if not source_dir.exists() or not source_dir.is_dir():
        logger.error(f"Source directory not found: {source_dir}")
        return 1

    backend = get_memory_backend()
    if not backend:
        logger.error("Memory subsystem is not enabled or could not be initialized.")
        return 1

    records, source_stats = collect_legacy_records(source_dir)
    result = import_records(backend, records)
    logger.info(
        (
            "Migration complete. Imported: %s, Failed: %s, Sources: "
            "json=%s markdown=%s project_schema=%s"
        ),
        result.imported,
        result.failed,
        source_stats.get("json", 0),
        source_stats.get("markdown", 0),
        source_stats.get("project_schema", 0),
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
