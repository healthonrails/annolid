from annolid.domain.memory.scopes import MemoryCategory, MemorySource
from annolid.domain.memory.taxonomy import (
    MEMORY_CATEGORIES,
    MEMORY_SOURCES,
    is_valid_category,
    is_valid_source,
)


def test_taxonomy_contains_core_values() -> None:
    assert MemoryCategory.ANNOTATION_RULE in MEMORY_CATEGORIES
    assert MemorySource.WORKFLOW in MEMORY_SOURCES
    assert is_valid_category(MemoryCategory.PROJECT_NOTE)
    assert is_valid_source(MemorySource.SYSTEM)
    assert not is_valid_category("not_a_category")
