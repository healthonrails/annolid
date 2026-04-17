from __future__ import annotations

from annolid.behavior.event_utils import (
    aggregate_aggression_bout_summary,
    aggression_sub_event_schema,
    infer_aggression_sub_event_counts_from_text,
    parse_aggression_sub_event_counts,
)


def test_aggression_sub_event_schema_contract() -> None:
    schema = aggression_sub_event_schema()
    assert schema["schema"] == "aggression_sub_events/v1"
    assert schema["bout_label"] == "aggression_bout"
    assert schema["sub_event_codes"] == [
        "slap_in_face",
        "run_away",
        "fight_initiation",
    ]


def test_parse_aggression_sub_event_counts_from_mixed_payload() -> None:
    parsed = parse_aggression_sub_event_counts(
        [
            {"event": "slap in the face", "count": 2},
            {"code": "run_away", "count": 1},
            "fight initiation",
        ]
    )
    assert parsed == {
        "slap_in_face": 2,
        "run_away": 1,
        "fight_initiation": 1,
    }


def test_infer_aggression_sub_event_counts_from_text() -> None:
    parsed = infer_aggression_sub_event_counts_from_text(
        "Aggression includes slap in the face, run away, and initiation of bigger fights."
    )
    assert parsed == {
        "slap_in_face": 1,
        "run_away": 1,
        "fight_initiation": 1,
    }


def test_aggregate_aggression_bout_summary_stable_counts() -> None:
    summary = aggregate_aggression_bout_summary(
        [
            {
                "label": "aggression_bout",
                "aggression_sub_events": {"slap_in_face": 1, "run_away": 1},
            },
            {
                "label": "aggression_bout",
                "description": "fight initiation followed by run away",
            },
            {
                "label": "grooming",
                "description": "slap in face text should be ignored outside bouts",
            },
        ]
    )
    assert summary["schema"] == "aggression_bout_summary/v1"
    assert summary["bout_count"] == 2
    assert summary["sub_event_counts"] == {
        "slap_in_face": 1,
        "run_away": 2,
        "fight_initiation": 1,
    }
    assert summary["sub_event_bout_counts"] == {
        "slap_in_face": 1,
        "run_away": 2,
        "fight_initiation": 1,
    }
    assert summary["bouts_with_initiation"] == 1
