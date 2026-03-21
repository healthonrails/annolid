from __future__ import annotations

from annolid.services.novelty import novelty_preflight_check


def test_novelty_preflight_check_recommends_abort_for_high_overlap() -> None:
    payload = novelty_preflight_check(
        idea_title="Self-supervised rodent behavior segmentation",
        idea_summary=(
            "Self-supervised rodent behavior segmentation with weak supervision and "
            "active learning over interaction videos."
        ),
        related_work=[
            {
                "title": "Self-supervised rodent behavior segmentation",
                "abstract": (
                    "Self-supervised rodent behavior segmentation with weak supervision "
                    "and active learning over interaction videos."
                ),
            }
        ],
        abort_overlap_threshold=0.6,
    )
    assert payload["ok"] is True
    assert payload["recommendation"] == "abort"
    assert payload["scores"]["max_overlap"] >= 0.6


def test_novelty_preflight_check_low_coverage_promotes_differentiate() -> None:
    payload = novelty_preflight_check(
        idea_summary="A new framework for adaptive behavioral annotation in neuroscience.",
        related_work=[],
    )
    assert payload["ok"] is True
    assert payload["coverage_quality"] == "low"
    assert payload["recommendation"] == "differentiate"


def test_novelty_preflight_check_scores_and_ranking() -> None:
    payload = novelty_preflight_check(
        idea_title="Graph-aware behavior segmentation",
        idea_summary=(
            "Graph-aware segmentation combines temporal context and pose priors "
            "for rodent social interaction videos."
        ),
        related_work=[
            {
                "title": "Temporal pose priors for rodent videos",
                "abstract": "Temporal context and pose priors help behavior segmentation.",
            },
            {
                "title": "Unrelated microscopy workflow",
                "abstract": "Cell counting and tracking in microscopy images.",
            },
            {
                "title": "Graph neural segmentation for behavior",
                "abstract": "Graph models improve social interaction segmentation.",
            },
        ],
        top_k=2,
        differentiate_overlap_threshold=0.3,
        abort_overlap_threshold=0.9,
    )
    assert payload["ok"] is True
    assert payload["scores"]["related_work_count"] >= 2
    assert len(payload["top_overlaps"]) == 2
    assert (
        payload["top_overlaps"][0]["overlap_score"]
        >= payload["top_overlaps"][1]["overlap_score"]
    )
    assert payload["scores"]["max_overlap"] >= payload["scores"]["mean_top3_overlap"]
