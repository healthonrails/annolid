"""Shared memory taxonomy constants and helpers."""

from __future__ import annotations

from typing import Final

from annolid.domain.memory.scopes import MemoryCategory, MemorySource


MEMORY_CATEGORIES: Final[tuple[str, ...]] = (
    MemoryCategory.PREFERENCE,
    MemoryCategory.FACT,
    MemoryCategory.DECISION,
    MemoryCategory.ANNOTATION_RULE,
    MemoryCategory.ANNOTATION_NOTE,
    MemoryCategory.PROJECT_SCHEMA,
    MemoryCategory.PROJECT_NOTE,
    MemoryCategory.SETTING,
    MemoryCategory.WORKFLOW_RECIPE,
    MemoryCategory.TROUBLESHOOTING,
    MemoryCategory.PROMPT_TEMPLATE,
    MemoryCategory.ENTITY,
    MemoryCategory.OTHER,
)

MEMORY_SOURCES: Final[tuple[str, ...]] = (
    MemorySource.USER,
    MemorySource.BOT,
    MemorySource.SYSTEM,
    MemorySource.ANNOTATION,
    MemorySource.PROJECT,
    MemorySource.SETTINGS,
    MemorySource.WORKFLOW,
    MemorySource.IMPORT,
)


def is_valid_category(value: str) -> bool:
    return value in MEMORY_CATEGORIES


def is_valid_source(value: str) -> bool:
    return value in MEMORY_SOURCES
