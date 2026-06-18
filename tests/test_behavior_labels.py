from annolid.behavior.labels import (
    NO_BEHAVIOR_LABEL,
    allowed_behavior_labels,
    canonicalize_behavior_label,
    is_no_behavior_label,
    normalize_behavior_label_list,
    text_indicates_no_behavior,
)


def test_normalize_behavior_label_list_splits_dedupes_and_skips_no_behavior() -> None:
    labels = normalize_behavior_label_list(
        [" grooming ", "Grooming", "unsupported rearing; walking", "no behavior"]
    )

    assert labels == ["grooming", "unsupported rearing", "walking"]


def test_allowed_behavior_labels_appends_no_behavior_once() -> None:
    labels = allowed_behavior_labels(["grooming", "no_behavior", "Grooming"])

    assert labels == ["grooming", NO_BEHAVIOR_LABEL]


def test_canonicalize_behavior_label_accepts_case_and_slug_variants() -> None:
    assert (
        canonicalize_behavior_label(
            "Unsupported-Rearing",
            ["grooming", "unsupported rearing"],
        )
        == "unsupported rearing"
    )
    assert (
        canonicalize_behavior_label("AGGRESSION BOUT", ["aggression_bout"])
        == "aggression_bout"
    )


def test_canonicalize_behavior_label_preserves_no_behavior_aliases() -> None:
    assert canonicalize_behavior_label("background", ["grooming"]) == "no_behavior"
    assert canonicalize_behavior_label("No behavior.", ["grooming"]) == "no_behavior"
    assert is_no_behavior_label("none of the above")
    assert text_indicates_no_behavior("No listed behavior is visible in this segment.")
