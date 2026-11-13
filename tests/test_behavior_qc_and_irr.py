"""Focused regression and mathematical invariant tests for ethogram QC, IRR, and CLI verbs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Tuple

import pytest

from annolid.behavior.event_utils import classify_event_label, parse_recording_seconds
from annolid.behavior.irr import (
    IRRComputationError,
    compute_binary_cohens_kappa,
    compute_fleiss_kappa,
    compute_irr,
    compute_irr_from_csvs,
    compute_onset_matching_f1,
)
from annolid.behavior.project_schema import (
    BehaviorDefinition,
    CategoryDefinition,
    ModifierDefinition,
    ProjectSchema,
    SubjectDefinition,
    save_schema,
)
from annolid.behavior.qc import (
    validate_behavior_csv,
    validate_behavior_events,
)
from annolid.behavior.time_budget import (
    extract_behavior_intervals,
    summarize_intervals,
)
from annolid.engine.cli import main as annolid_run


def _write_events_csv(
    path: Path,
    rows: Iterable[Tuple[object, object, object, object, object]],
    *,
    include_modifiers: bool = False,
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if include_modifiers:
            writer.writerow(
                [
                    "Trial time",
                    "Recording time",
                    "Subject",
                    "Behavior",
                    "Event",
                    "Modifiers",
                ]
            )
        else:
            writer.writerow(
                ["Trial time", "Recording time", "Subject", "Behavior", "Event"]
            )
        for row in rows:
            writer.writerow(list(row))
    return path


def _build_ethogram_schema() -> ProjectSchema:
    return ProjectSchema(
        categories=[
            CategoryDefinition(id="locomotion", name="Locomotion"),
            CategoryDefinition(id="social", name="Social"),
        ],
        subjects=[
            SubjectDefinition(id="mouse_1", name="Mouse 1"),
            SubjectDefinition(id="mouse_2", name="Mouse 2"),
        ],
        modifiers=[
            ModifierDefinition(id="fast", name="Fast"),
            ModifierDefinition(id="gentle", name="Gentle"),
        ],
        behaviors=[
            BehaviorDefinition(
                code="walking",
                name="Walking",
                category_id="locomotion",
                modifier_ids=["fast"],
                is_state=True,
                exclusive_with=["grooming"],
            ),
            BehaviorDefinition(
                code="grooming",
                name="Grooming",
                category_id="locomotion",
                is_state=True,
                # Deliberately omit symmetric back-reference to test symmetric enforcement
                exclusive_with=[],
            ),
            BehaviorDefinition(
                code="bite",
                name="Bite",
                category_id="social",
                is_state=False,
            ),
        ],
    )


def test_event_utils_timestamp_and_point_parsing() -> None:
    assert classify_event_label("state start") == "start"
    assert classify_event_label("state stop") == "end"
    assert classify_event_label("point") == "point"
    assert classify_event_label("Point Event") == "point"
    assert classify_event_label("bogus") is None

    assert parse_recording_seconds("12.5") == pytest.approx(12.5)
    assert parse_recording_seconds("00:01:02.500") == pytest.approx(62.5)
    assert parse_recording_seconds("02:05.25") == pytest.approx(125.25)
    assert parse_recording_seconds("nan") is None
    assert parse_recording_seconds("inf") is None


def test_abutting_state_start_stop_intervals_remain_separate_and_clean() -> None:
    """1s segment labeling produces abutting [0,1] + [1,2] intervals with 'state start'/'state stop'."""
    rows = [
        {
            "recording time": "0.0",
            "subject": "Mouse 1",
            "behavior": "walking",
            "event": "state start",
        },
        {
            "recording time": "1.0",
            "subject": "Mouse 1",
            "behavior": "walking",
            "event": "state stop",
        },
        {
            "recording time": "1.0",
            "subject": "Mouse 1",
            "behavior": "walking",
            "event": "state start",
        },
        {
            "recording time": "2.0",
            "subject": "Mouse 1",
            "behavior": "walking",
            "event": "state stop",
        },
    ]
    qc = validate_behavior_events(rows)
    assert qc.passed
    assert qc.error_count == 0
    assert qc.warning_count == 0
    assert [(iv.start, iv.end) for iv in qc.intervals] == [(0.0, 1.0), (1.0, 2.0)]

    tb_intervals, tb_warnings = extract_behavior_intervals(rows)
    assert tb_warnings == []
    summary = summarize_intervals(tb_intervals)
    assert len(summary) == 1
    assert summary[0].occurrences == 2
    assert summary[0].min_duration == pytest.approx(1.0)
    assert summary[0].max_duration == pytest.approx(1.0)
    assert summary[0].total_duration == pytest.approx(2.0)


def test_qc_detects_unpaired_duplicate_exclusive_and_point_state_violations(
    tmp_path: Path,
) -> None:
    schema = _build_ethogram_schema()
    csv_path = _write_events_csv(
        tmp_path / "dirty_events.csv",
        [
            # Duplicate start on walking: 1.0, 1.5 -> closed at 3.0 (leaves 1 unclosed start too)
            (1.0, 1.0, "Mouse 1", "walking", "state start", "fast"),
            (1.5, 1.5, "Mouse 1", "walking", "state start", "gentle"),
            (3.0, 3.0, "Mouse 1", "walking", "state stop", ""),
            # Grooming overlaps walking [2.0, 4.0] even though only walking listed grooming
            (2.0, 2.0, "Mouse 1", "grooming", "state start", ""),
            (4.0, 4.0, "Mouse 1", "grooming", "state stop", ""),
            # Orphan end on grooming at 5.0
            (5.0, 5.0, "Mouse 1", "grooming", "state stop", ""),
            # Point behavior 'bite' misused with state start
            (6.0, 6.0, "Mouse 1", "bite", "state start", ""),
            # Valid point event on 'bite'
            (6.5, 6.5, "Mouse 1", "bite", "point", ""),
        ],
        include_modifiers=True,
    )

    report = validate_behavior_csv(csv_path, schema=schema)
    assert not report.passed
    codes = [issue.code for issue in report.issues]
    assert "duplicate_state_start" in codes
    assert "unclosed_state_start" in codes
    assert "exclusive_behavior_overlap" in codes
    assert "orphan_state_end" in codes
    assert "point_behavior_state_event" in codes
    assert "disallowed_modifier" in codes
    assert report.point_event_count == 1


def test_qc_zero_duration_interval_warns_without_error() -> None:
    rows = [
        {
            "recording time": "2.0",
            "subject": "Mouse 1",
            "behavior": "walking",
            "event": "start",
        },
        {
            "recording time": "2.0",
            "subject": "Mouse 1",
            "behavior": "walking",
            "event": "end",
        },
    ]
    report = validate_behavior_events(rows)
    assert report.passed
    assert report.error_count == 0
    assert report.warning_count == 1
    assert report.issues[0].code == "zero_duration_interval"
    assert report.valid_interval_count == 1


def test_irr_mathematical_invariants_and_edge_cases() -> None:
    # 1. Perfect non-trivial agreement -> kappa = 1.0, PABAK = 1.0, IoU = 1.0
    stats_perfect = compute_binary_cohens_kappa([1, 1, 0, 0], [1, 1, 0, 0])
    assert stats_perfect["cohens_kappa"] == pytest.approx(1.0)
    assert stats_perfect["pabak"] == pytest.approx(1.0)
    assert stats_perfect["iou"] == pytest.approx(1.0)

    # 2. Independent 50/50 chance agreement -> p_o = 0.5, p_e = 0.5, kappa = 0.0
    stats_chance = compute_binary_cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0])
    assert stats_chance["observed_agreement"] == pytest.approx(0.5)
    assert stats_chance["expected_agreement"] == pytest.approx(0.5)
    assert stats_chance["cohens_kappa"] == pytest.approx(0.0)
    assert stats_chance["pabak"] == pytest.approx(0.0)

    # 3. Complete disagreement -> p_o = 0.0, p_e = 0.5, kappa = -1.0
    stats_opposite = compute_binary_cohens_kappa([1, 1, 0, 0], [0, 0, 1, 1])
    assert stats_opposite["cohens_kappa"] == pytest.approx(-1.0)
    assert stats_opposite["pabak"] == pytest.approx(-1.0)

    # 4. Constant all-zero sequences -> no ZeroDivisionError, kappa = 1.0
    stats_const = compute_binary_cohens_kappa([0, 0, 0, 0], [0, 0, 0, 0])
    assert stats_const["cohens_kappa"] == pytest.approx(1.0)
    assert stats_const["iou"] == pytest.approx(1.0)

    # 5. Fleiss' kappa across 3 raters: unanimous vs balanced chance
    assert compute_fleiss_kappa(
        [[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]]
    ) == pytest.approx(1.0)
    # 4 bins, 3 raters: each bin has two 1s and one 0, or one 1 and two 0s with overall p1 = 0.5
    fleiss_partial = compute_fleiss_kappa(
        [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ]
    )
    # Every bin has 2 of one class and 1 of the other -> P_i = 2 / 6 = 1/3; p1 = 0.5 -> P_e = 0.5
    # Fleiss kappa = (1/3 - 1/2) / (1 - 1/2) = -1/3
    assert fleiss_partial == pytest.approx(-1.0 / 3.0)

    # 6. 1D onset tolerance matching achieves maximum cardinality
    assert compute_onset_matching_f1(
        [1.0, 5.0, 10.0],
        [1.3, 4.7, 12.0],
        tolerance_seconds=0.5,
    ) == pytest.approx(2.0 * 2 / 6.0)


def test_irr_single_behavior_binary_matches_multiclass_kappa() -> None:
    """For a single behavior + idle background, binary Cohen's kappa equals multiclass kappa."""
    rater_1 = [
        {
            "recording time": "1.0",
            "subject": "Mouse 1",
            "behavior": "grooming",
            "event": "start",
        },
        {
            "recording time": "4.0",
            "subject": "Mouse 1",
            "behavior": "grooming",
            "event": "end",
        },
    ]
    rater_2 = [
        {
            "recording time": "2.0",
            "subject": "Mouse 1",
            "behavior": "grooming",
            "event": "start",
        },
        {
            "recording time": "5.0",
            "subject": "Mouse 1",
            "behavior": "grooming",
            "event": "end",
        },
    ]
    report = compute_irr(
        [("r1", rater_1), ("r2", rater_2)],
        bin_seconds=1.0,
        session_start=0.0,
        session_end=10.0,
    )
    assert len(report.rows) == 1
    row = report.rows[0]
    # Bins [1,2,3] vs [2,3,4] out of 10 bins -> n11=2, n10=1, n01=1, n00=6
    # p_o = 0.8, p_1 = 0.3, p_2 = 0.3 -> p_e = 0.09 + 0.49 = 0.58
    # kappa = (0.8 - 0.58) / (1 - 0.58) = 0.22 / 0.42 = 11/21
    assert row.cohens_kappa == pytest.approx(11.0 / 21.0)
    assert report.multiclass_cohens_kappa == pytest.approx(row.cohens_kappa)
    assert row.iou == pytest.approx(0.5)


def test_irr_three_raters_csv_and_validation_errors(tmp_path: Path) -> None:
    c1 = _write_events_csv(
        tmp_path / "alice.csv",
        [
            (0.0, 1.0, "Mouse 1", "walking", "start"),
            (0.0, 3.0, "Mouse 1", "walking", "end"),
        ],
    )
    c2 = _write_events_csv(
        tmp_path / "bob.csv",
        [
            (0.0, 1.0, "Mouse 1", "walking", "start"),
            (0.0, 3.0, "Mouse 1", "walking", "end"),
        ],
    )
    c3 = _write_events_csv(
        tmp_path / "carol.csv",
        [
            (0.0, 1.0, "Mouse 1", "walking", "start"),
            (0.0, 2.0, "Mouse 1", "walking", "end"),
        ],
    )
    report = compute_irr_from_csvs(
        [c1, c2, c3],
        bin_seconds=1.0,
        session_start=0.0,
        session_end=5.0,
    )
    assert report.raters == ("alice", "bob", "carol")
    assert len(report.rows) == 1
    assert 0.5 < report.rows[0].fleiss_kappa < 1.0

    with pytest.raises(IRRComputationError):
        compute_irr_from_csvs([c1])


def test_annolid_run_behavior_cli_subcommands(tmp_path: Path, capsys) -> None:
    schema = _build_ethogram_schema()
    schema_path = tmp_path / "project.annolid.json"
    save_schema(schema, schema_path)

    clean_csv = _write_events_csv(
        tmp_path / "rater_a.csv",
        [
            (0.0, 1.0, "Mouse 1", "walking", "state start"),
            (0.0, 3.0, "Mouse 1", "walking", "state stop"),
            (0.0, 4.0, "Mouse 1", "grooming", "state start"),
            (0.0, 6.0, "Mouse 1", "grooming", "state stop"),
        ],
    )
    rater_b_csv = _write_events_csv(
        tmp_path / "rater_b.csv",
        [
            (0.0, 1.0, "Mouse 1", "walking", "state start"),
            (0.0, 3.0, "Mouse 1", "walking", "state stop"),
            (0.0, 4.1, "Mouse 1", "grooming", "state start"),
            (0.0, 6.0, "Mouse 1", "grooming", "state stop"),
        ],
    )

    # 1. Root --help lists Behavior group commands
    rc_help = annolid_run(["--help"])
    assert rc_help == 0
    help_text = capsys.readouterr().out
    assert "Behavior:" in help_text
    assert "behavior-qc" in help_text
    assert "behavior-irr" in help_text
    assert "time-budget" in help_text

    # 2. behavior-qc passes on clean CSV and writes JSON output
    qc_json = tmp_path / "qc.json"
    rc_qc = annolid_run(
        [
            "behavior-qc",
            str(clean_csv),
            "--schema",
            str(schema_path),
            "--strict",
            "--json",
            "--out",
            str(qc_json),
        ]
    )
    assert rc_qc == 0
    qc_payload = json.loads(capsys.readouterr().out)
    assert qc_payload["passed"] is True
    assert qc_payload["error_count"] == 0
    assert qc_json.exists()

    # 3. time-budget writes CSV + category sidecar + JSON report
    tb_csv = tmp_path / "budget.csv"
    tb_json = tmp_path / "budget.json"
    rc_tb = annolid_run(
        [
            "time-budget",
            str(clean_csv),
            "--schema",
            str(schema_path),
            "-o",
            str(tb_csv),
            "--json-out",
            str(tb_json),
            "--json",
        ]
    )
    assert rc_tb == 0
    tb_payload = json.loads(capsys.readouterr().out)
    assert len(tb_payload["rows"]) == 2
    assert tb_csv.exists()
    assert (tmp_path / "budget_categories.csv").exists()
    assert tb_json.exists()

    # 4. behavior-irr passes --min-kappa 0.85 on high-agreement pair and fails --min-kappa 0.9999
    irr_json = tmp_path / "irr.json"
    irr_csv = tmp_path / "irr.csv"
    rc_irr_ok = annolid_run(
        [
            "behavior-irr",
            str(clean_csv),
            str(rater_b_csv),
            "--schema",
            str(schema_path),
            "--bin-seconds",
            "0.1",
            "--session-start",
            "0.0",
            "--session-end",
            "10.0",
            "--min-kappa",
            "0.85",
            "--json",
            "--out",
            str(irr_json),
            "--csv-out",
            str(irr_csv),
        ]
    )
    assert rc_irr_ok == 0
    irr_payload = json.loads(capsys.readouterr().out)
    assert irr_payload["macro_cohens_kappa"] > 0.90
    assert irr_json.exists()
    assert irr_csv.exists()

    rc_irr_gate_fail = annolid_run(
        [
            "behavior-irr",
            str(clean_csv),
            str(rater_b_csv),
            "--bin-seconds",
            "0.1",
            "--session-start",
            "0.0",
            "--session-end",
            "10.0",
            "--min-kappa",
            "0.9999",
        ]
    )
    assert rc_irr_gate_fail == 2
    capsys.readouterr()
