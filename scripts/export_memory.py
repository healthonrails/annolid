import json
import sys
import logging
import argparse
from pathlib import Path

try:
    from annolid.interfaces.memory.registry import (
        get_memory_backend,
        get_memory_service,
    )
except ImportError:
    print(
        "Error: Could not import annolid modules. Ensure you're running from the project root."
    )
    sys.exit(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def export_memories(output_file: Path):
    """Exports all LanceDB memories to a JSON Lines file."""
    backend = get_memory_backend()
    if not backend:
        logger.error("Memory subsystem is not enabled or backend unavailable.")
        return 1
    if not hasattr(backend, "export_jsonl"):
        logger.error("Backend does not support JSONL export.")
        return 1

    exported = backend.export_jsonl(output_file)
    logger.info(f"Exported {exported} memories to {output_file}")
    return int(exported)


def import_memories(input_file: Path):
    """Imports memories from a JSON Lines file, skipping existing IDs if possible, or generating new ones."""
    service = get_memory_service()
    if not service:
        logger.error("Memory subsystem is not enabled or backend unavailable.")
        return 1

    if not input_file.exists():
        logger.error(f"Input file not found: {input_file}")
        return 1

    success_count = 0

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            data = json.loads(line)
            try:
                # We will just use the standard store_memory to regenerate vectors seamlessly
                metadata = {}
                try:
                    if "metadata_json" in data and data["metadata_json"]:
                        metadata = json.loads(data["metadata_json"])
                except Exception:
                    pass

                service.store_memory(
                    text=data["text"],
                    scope=data.get("scope", "global"),
                    category=data.get("category", "other"),
                    source=data.get("source", "system"),
                    importance=float(data.get("importance", 0.5)),
                    tags=data.get("tags", []),
                    metadata=metadata,
                    dedupe=True,  # Automatically skip exact duplicates!
                )
                success_count += 1
            except Exception as e:
                logger.error(
                    f"Failed to import record (Text trunc: {data.get('text', '')[:20]}): {e}"
                )

    logger.info(f"Import process finished. Processed {success_count} records.")
    return success_count


def main(argv=None):
    parser = argparse.ArgumentParser(description="Export or import Annolid memories.")
    subparsers = parser.add_subparsers(dest="action", required=True)

    exp_parser = subparsers.add_parser("export", help="Export to JSONL")
    exp_parser.add_argument("output", type=Path, help="Output JSONL file path")

    imp_parser = subparsers.add_parser("import", help="Import from JSONL")
    imp_parser.add_argument("input", type=Path, help="Input JSONL file path")

    args = parser.parse_args(argv)

    if args.action == "export":
        export_memories(args.output)
        return 0
    if args.action == "import":
        import_memories(args.input)
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
