from __future__ import annotations

from pathlib import Path

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
