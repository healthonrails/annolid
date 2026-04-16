from __future__ import annotations

from pathlib import Path

from annolid.behavior.time_budget import BoutDefinition, aggression_bout_definition
from annolid.services.time_budget import (
    compute_behavior_time_budget_report,
    write_behavior_time_budget_report_csv,
)


def test_compute_behavior_time_budget_report() -> None:
    rows = [
        ("0.0", 0.0, "Mouse-1", "grooming", "start"),
        ("2.0", 2.0, "Mouse-1", "grooming", "end"),
    ]
    report = compute_behavior_time_budget_report(rows)

    assert len(report.summary) == 1
    first = report.summary[0]
    assert first.subject == "Mouse-1"
    assert first.behavior == "grooming"
    assert abs(first.total_duration - 2.0) < 1e-6
    assert "grooming" in report.table_text


def test_write_behavior_time_budget_report_csv(tmp_path: Path) -> None:
    rows = [
        ("0.0", 0.0, "Mouse-1", "explore", "start"),
        ("1.0", 1.0, "Mouse-1", "explore", "end"),
    ]
    report = compute_behavior_time_budget_report(rows)
    output = tmp_path / "budget.csv"
    category_path = write_behavior_time_budget_report_csv(report, output)

    assert output.exists()
    text = output.read_text(encoding="utf-8")
    assert "explore" in text
    assert category_path is None


def test_compute_behavior_time_budget_report_with_aggression_bouts() -> None:
    rows = [
        ("0.0", 0.0, "Mouse-1", "slap_in_face", "start"),
        ("0.3", 0.3, "Mouse-1", "slap_in_face", "end"),
        ("0.6", 0.6, "Mouse-1", "run_away", "start"),
        ("1.1", 1.1, "Mouse-1", "run_away", "end"),
        ("1.2", 1.2, "Mouse-1", "fight_initiation", "start"),
        ("1.4", 1.4, "Mouse-1", "fight_initiation", "end"),
    ]
    report = compute_behavior_time_budget_report(
        rows,
        bout_definition=aggression_bout_definition(max_gap_seconds=0.5),
    )

    assert len(report.bout_summary) == 1
    first = report.bout_summary[0]
    assert first.bout_name == "aggression_bout"
    assert first.bout_count == 1
    assert int(first.behavior_counts["slap_in_face"]) == 1
    assert int(first.behavior_counts["run_away"]) == 1
    assert int(first.behavior_counts["fight_initiation"]) == 1
    assert first.initiation_count == 1
    assert "aggression_bout" in report.bout_table_text
    assert "fight_initiation Count" in report.bout_table_text


def test_write_behavior_time_budget_report_csv_writes_bout_sidecar(
    tmp_path: Path,
) -> None:
    rows = [
        ("0.0", 0.0, "Mouse-1", "slap_in_face", "start"),
        ("0.2", 0.2, "Mouse-1", "slap_in_face", "end"),
    ]
    definition = BoutDefinition(
        name="aggression_bout",
        behaviors=("slap_in_face",),
        max_gap_seconds=0.0,
        initiation_behaviors=(),
    )
    report = compute_behavior_time_budget_report(rows, bout_definition=definition)
    output = tmp_path / "budget.csv"

    category_path = write_behavior_time_budget_report_csv(
        report,
        output,
        bout_definition=definition,
    )

    assert category_path is None
    bout_path = tmp_path / "budget_bouts.csv"
    assert bout_path.exists()
    text = bout_path.read_text(encoding="utf-8")
    assert "slap_in_faceCount" in text
