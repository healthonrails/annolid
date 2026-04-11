from annolid.postprocessing.glitter import (
    _extend_unique,
    _label_in_collection,
    _normalize_label_names,
)


def test_normalize_label_names_accepts_csv_string():
    result = _normalize_label_names("mouse_1, mouse_2,  grooming")
    assert result == ["mouse_1", "mouse_2", "grooming"]


def test_normalize_label_names_accepts_iterables_and_trims_values():
    result = _normalize_label_names(["mouse_1", " mouse_2 ", None, ""])
    assert result == ["mouse_1", "mouse_2"]


def test_extend_unique_deduplicates_case_insensitive():
    base = ["mouse_1"]
    updated = _extend_unique(base, ["Mouse_1", "mouse_2"])
    assert updated == ["mouse_1", "mouse_2"]


def test_label_in_collection_matches_case_insensitive_exact_names():
    names = ["mouse_1", "LeftInteract"]
    assert _label_in_collection("mouse_1", names)
    assert _label_in_collection("leftinteract", names)
    assert not _label_in_collection("mouse", names)
