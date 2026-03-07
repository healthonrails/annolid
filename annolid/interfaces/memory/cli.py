import argparse
import json
from typing import Optional

from annolid.interfaces.memory.registry import (
    get_memory_backend,
    get_memory_service,
    get_retrieval_service,
)


def stats_command(args):
    service = get_memory_service()
    if not service:
        print("Memory subsystem is not enabled or could not be initialized.")
        return 1

    stats = service.stats(scope=args.scope)
    print(json.dumps(stats, indent=2))
    return 0


def search_command(args):
    service = get_retrieval_service()
    if not service:
        print("Memory subsystem is not enabled or could not be initialized.")
        return 1

    hits = service.search_memory(query=args.query, top_k=args.top_k, scope=args.scope)

    if not hits:
        print("No memories found.")
        return 0

    print(f"Found {len(hits)} memory/memories:")
    for i, hit in enumerate(hits):
        print(f"[{i + 1}] ({hit.score:.2f}) [Scope: {hit.scope}] [Cat: {hit.category}]")
        print(f"    {hit.text}")
        print("-" * 40)
    return 0


def delete_command(args):
    service = get_memory_service()
    if not service:
        print("Memory subsystem is not enabled or could not be initialized.")
        return 1

    deleted = service.delete_memory(args.id)
    if deleted:
        print(f"Successfully deleted memory ID: {args.id}")
        return 0
    else:
        print(f"Failed to delete memory ID: {args.id}. Not found or error.")
        return 1


def distill_command(args):
    try:
        from annolid.services.memory.summarization_service import (
            MemorySummarizationService,
        )

        summarizer = MemorySummarizationService()
        processed = summarizer.distill_scope(
            scope=args.scope,
            category=args.category,
            max_records_to_process=args.max_records,
        )
        print(
            f"Successfully distilled {processed} memory records in scope '{args.scope}'."
        )
        return 0
    except ImportError:
        print("Summarization service not available.")
        return 1


def cleanup_command(args):
    backend = get_memory_backend()
    if not backend:
        print("Memory subsystem is not enabled or backend unavailable.")
        return 1

    import time

    now_ms = int(time.time() * 1000)
    cutoff_ms = now_ms - (args.older_than_days * 24 * 60 * 60 * 1000)

    try:
        if not hasattr(backend, "list_memories"):
            print("Backend does not support memory cleanup.")
            return 1

        hits = backend.list_memories(top_k=1000, scope=args.scope)
        candidates = [
            hit
            for hit in hits
            if hit.timestamp_ms < cutoff_ms or hit.importance < args.min_importance
        ]
        if not candidates:
            print(
                f"No records found matching cleanup criteria in scope '{args.scope}'."
            )
            return 0

        deleted_count = 0
        for hit in candidates:
            if backend.delete(hit.id):
                deleted_count += 1

        print(
            f"Successfully cleaned up {deleted_count} records in scope '{args.scope}'."
        )
        return 0
    except Exception as e:
        print(f"Cleanup failed: {e}")
        return 1


def main(argv: Optional[list[str]] = None):
    parser = argparse.ArgumentParser(description="Annolid Memory CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Stats
    stats_parser = subparsers.add_parser("stats", help="Show memory statistics")
    stats_parser.add_argument("--scope", type=str, help="Filter stats by scope")

    # Search
    search_parser = subparsers.add_parser("search", help="Search memory")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--top_k", type=int, default=5, help="Number of results")
    search_parser.add_argument("--scope", type=str, help="Filter by scope")

    # Delete
    delete_parser = subparsers.add_parser("delete", help="Delete a memory")
    delete_parser.add_argument("id", type=str, help="Memory ID to delete")

    # Distill
    distill_parser = subparsers.add_parser(
        "distill", help="Summarize old memories in a scope"
    )
    distill_parser.add_argument("scope", type=str, help="Scope to distill")
    distill_parser.add_argument(
        "--category", type=str, help="Optional category to distill"
    )
    distill_parser.add_argument(
        "--max_records", type=int, default=50, help="Max records to distill"
    )

    # Cleanup
    cleanup_parser = subparsers.add_parser(
        "cleanup", help="Delete old or low importance memories"
    )
    cleanup_parser.add_argument("scope", type=str, help="Scope to clean")
    cleanup_parser.add_argument(
        "--older_than_days", type=int, default=30, help="Delete older than X days"
    )
    cleanup_parser.add_argument(
        "--min_importance", type=float, default=0.2, help="Delete if importance below X"
    )

    args = parser.parse_args(argv)

    if args.command == "stats":
        return stats_command(args)
    if args.command == "search":
        return search_command(args)
    if args.command == "delete":
        return delete_command(args)
    if args.command == "distill":
        return distill_command(args)
    if args.command == "cleanup":
        return cleanup_command(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
