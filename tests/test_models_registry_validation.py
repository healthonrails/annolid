from __future__ import annotations

from annolid.gui.models_registry import (
    MODEL_REGISTRY,
    PATCH_SIMILARITY_DEFAULT_MODEL,
    PATCH_SIMILARITY_MODELS,
)


def test_model_registry_identifiers_are_unique_case_insensitive() -> None:
    identifiers = [cfg.identifier for cfg in MODEL_REGISTRY]
    lowered = [identifier.lower() for identifier in identifiers]

    assert len(lowered) == len(set(lowered))


def test_model_registry_entries_have_required_non_empty_fields() -> None:
    assert MODEL_REGISTRY
    for cfg in MODEL_REGISTRY:
        assert cfg.display_name.strip()
        assert cfg.identifier.strip()
        assert cfg.weight_file.strip()


def test_patch_similarity_default_model_is_registered() -> None:
    identifiers = {cfg.identifier for cfg in PATCH_SIMILARITY_MODELS}
    assert PATCH_SIMILARITY_DEFAULT_MODEL in identifiers
