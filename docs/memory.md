# Annolid Memory

The Annolid memory subsystem provides an optional, local-first storage layer for reusable context such as project notes, annotation conventions, workspace settings snapshots, workflow recipes, and agent memory.

Memory is retrieval-oriented. It does not replace canonical project configs, annotations, or datasets.

## Status

Current implementation status:

- Layered memory modules in `annolid/domain/memory`, `annolid/services/memory`, `annolid/infrastructure/memory/lancedb`, and `annolid/interfaces/memory`
- LanceDB-backed storage with optional semantic retrieval
- Scope/category/source taxonomy
- Explicit persistence helpers and interface adapters
- `annolid-run memory ...` CLI commands for stats, search, delete, distill, and cleanup
- Import/export/re-embedding maintenance scripts

Not yet complete:

- Full engine-level CRUD workflows in the GUI
- Rich structured settings model beyond snapshot storage
- End-to-end migration automation for legacy memory sources

## Install

Install the optional dependency group:

```bash
pip install -e ".[memory]"
```

For local validation in this repository, use `.venv`:

```bash
source .venv/bin/activate
```

## Architecture

The subsystem follows Annolid's layered architecture:

```text
annolid/
  domain/memory/
  services/memory/
  infrastructure/memory/lancedb/
  interfaces/memory/
```

Responsibilities:

- `domain`: backend-agnostic data models, protocols, scopes, taxonomy
- `services`: orchestration for storing, retrieval, context building, persistence
- `infrastructure`: LanceDB config, backend, retriever, embedder, reranker, migration helpers
- `interfaces`: registry, CLI, and adapters for bot, annotation, project, workflow, workspace

## Enablement

Memory is optional and controlled by environment variables:

```bash
export ANNOLID_MEMORY_ENABLED=true
export ANNOLID_MEMORY_BACKEND=lancedb
export ANNOLID_MEMORY_LANCEDB_PATH=~/.annolid/memory/lancedb
export ANNOLID_MEMORY_EMBEDDING_PROVIDER=jina
export ANNOLID_MEMORY_EMBEDDING_MODEL=jina-embeddings-v3
export ANNOLID_MEMORY_VECTOR_WEIGHT=0.65
export ANNOLID_MEMORY_BM25_WEIGHT=0.35
export ANNOLID_MEMORY_RERANK_PROVIDER=none
```

Supported backend selector values today:

- `lancedb`
- `none`

## Data Model

Primary models:

- `MemoryRecord`: write model for memory entries
- `MemoryHit`: read model returned by search/list operations

Important fields:

- `text`: human-readable summary used for keyword and semantic retrieval
- `scope`: such as `global`, `project:<id>`, `dataset:<id>`, `workspace:<id>`
- `category`: such as `project_note`, `annotation_rule`, `setting`, `workflow_recipe`
- `metadata`: structured references to canonical artifacts or workflow context

## Retrieval Behavior

The LanceDB retriever combines:

- vector similarity
- full-text search when available
- recency boost
- basic length normalization
- optional reranking

Blank-query context retrieval is supported. This matters for calls such as "show me project memory" where the caller wants scoped memory without providing a semantic query string.

## CLI

The memory CLI is available through the main runner:

```bash
annolid-run memory stats --scope global
annolid-run memory search "segmentation conventions" --top_k 5
annolid-run memory delete <memory_id>
annolid-run memory distill project:demo --max_records 50
annolid-run memory cleanup workspace:default --older_than_days 30 --min_importance 0.2
```

Notes:

- `search` performs semantic retrieval when embeddings are enabled
- `cleanup` removes low-importance or old records within a scope
- `distill` depends on the summarization service being available

## Developer API

Store memory directly:

```python
from annolid.domain.memory.scopes import MemoryCategory, MemoryScope, MemorySource
from annolid.interfaces.memory.registry import get_memory_service

service = get_memory_service()
service.store_memory(
    text="Use tail_base instead of tailroot for this dataset.",
    scope=MemoryScope.dataset("mouse-session-a"),
    category=MemoryCategory.ANNOTATION_RULE,
    source=MemorySource.ANNOTATION,
    importance=0.9,
    metadata={"species": "mouse"},
)
```

Build scoped context:

```python
from annolid.interfaces.memory.registry import get_context_service

context = get_context_service().build_project_context(project_id="demo-project")
```

Use higher-level persistence helpers:

```python
from annolid.interfaces.memory.registry import get_persistence_service
from annolid.domain.memory.scopes import MemoryScope

persistence = get_persistence_service()
persistence.save_settings_snapshot(
    scope=MemoryScope.workspace("lab-a"),
    description="Known-good export setup",
    settings={"fps": 10, "format": "csv"},
    context="Infrared videos",
)
```

## Adapters

Available adapters:

- `BotMemoryAdapter`
- `AnnotationMemoryAdapter`
- `ProjectMemoryAdapter`
- `WorkflowMemoryAdapter`
- `WorkspaceMemoryAdapter`

These adapters keep the rest of Annolid from depending directly on LanceDB details.

## Maintenance Scripts

Repository scripts:

```bash
source .venv/bin/activate
python scripts/export_memory.py export /tmp/annolid_memory.jsonl
python scripts/export_memory.py import /tmp/annolid_memory.jsonl
python scripts/migrate_memory.py --source-dir /path/to/legacy/json
python scripts/reembed_memory.py
```

What they do:

- `export_memory.py`: export/import JSONL records
- `migrate_memory.py`: import legacy JSON files into the current backend
- `reembed_memory.py`: recompute stored vectors for the current embedding config

## JSONL Contract

The export format is newline-delimited JSON. Each line is a single memory record without the vector payload.

Expected exported fields:

- `id`
- `text`
- `scope`
- `category`
- `source`
- `timestamp_ms`
- `importance`
- `token_count`
- `tags`
- `metadata_json`

Contract notes:

- vectors are intentionally omitted because they can be regenerated
- `metadata_json` is stored as a JSON string for backend compatibility
- import regenerates embeddings through `store_memory(...)`
- import treats identical records conservatively via the service dedupe path

## Testing

Run targeted memory tests:

```bash
source .venv/bin/activate
pytest tests/services/test_memory_service.py \
  tests/test_memory_cli.py \
  tests/test_memory_cli_commands.py \
  tests/test_memory_scripts.py \
  tests/services/test_persistence_service.py \
  tests/interfaces/test_adapters.py \
  tests/interfaces/test_workspace_memory_adapter.py \
  tests/memory/test_taxonomy.py \
  tests/memory/lancedb/test_migration.py \
  tests/memory/lancedb/test_backend.py \
  tests/memory/lancedb/test_backend_admin.py
```

## Design Constraints

The subsystem is intentionally conservative:

- writes are explicit by default
- memory stores compact summaries, not full raw artifacts
- canonical project files remain the source of truth
- optional dependencies must fail gracefully when unavailable
