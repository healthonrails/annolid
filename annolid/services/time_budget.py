from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from annolid.behavior.time_budget import (
    TimeBudgetComputationError,
    TimeBudgetRow,
    compute_time_budget,
    format_time_budget_table,
    summarize_by_category,
    write_time_budget_csv,
)
from annolid.core.behavior.spec import ProjectSchema


@dataclass(frozen=True)
class BehaviorTimeBudgetReport:
    summary: Sequence[TimeBudgetRow]
    warnings: Sequence[str]
    category_summary: Sequence[Tuple[str, float, int]]
    table_text: str


def compute_behavior_time_budget_report(
    rows: Iterable[Tuple[object, object, object, object, object]],
    *,
    schema: Optional[ProjectSchema] = None,
) -> BehaviorTimeBudgetReport:
    """Compute a reusable time-budget report from exported behavior rows."""
    data_rows = []
    local_warnings: List[str] = []
    for _trial_time, recording_time, subject, behavior, event_label in rows:
        if recording_time is None:
            local_warnings.append(
                "Skipping event because no timestamp could be determined: "
                f"behavior={behavior or '?'}, event={event_label or '?'}"
            )
            continue
        data_rows.append(
            {
                "trial time": _trial_time,
                "recording time": recording_time,
                "subject": str(subject or "").strip(),
                "behavior": str(behavior or "").strip(),
                "event": str(event_label or "").strip(),
            }
        )

    if not data_rows:
        return BehaviorTimeBudgetReport(
            summary=[],
            warnings=tuple(local_warnings),
            category_summary=(),
            table_text="No timestamped events remain after filtering.",
        )

    summary, compute_warnings = compute_time_budget(data_rows)
    warnings: List[str] = local_warnings + list(compute_warnings)
    category_summary: List[Tuple[str, float, int]] = []
    if schema is not None and summary:
        category_summary = summarize_by_category(summary, schema)

    return BehaviorTimeBudgetReport(
        summary=tuple(summary),
        warnings=tuple(warnings),
        category_summary=tuple(category_summary),
        table_text=format_time_budget_table(summary, schema=schema),
    )


def write_behavior_time_budget_report_csv(
    report: BehaviorTimeBudgetReport,
    output_path: Path,
    *,
    schema: Optional[ProjectSchema] = None,
) -> Optional[Path]:
    """Write the primary CSV and optional category summary CSV."""
    output_path = Path(output_path)
    write_time_budget_csv(report.summary, output_path, schema=schema)
    if schema is None or not report.category_summary:
        return None

    category_path = output_path.with_name(
        output_path.stem + "_categories" + output_path.suffix
    )
    with category_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Category", "TotalSeconds", "Occurrences"])
        for name, total, occurrences in report.category_summary:
            writer.writerow([name, f"{total:.6f}", int(occurrences)])
    return category_path


__all__ = [
    "BehaviorTimeBudgetReport",
    "TimeBudgetComputationError",
    "compute_behavior_time_budget_report",
    "write_behavior_time_budget_report_csv",
]
