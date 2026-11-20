from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

from annolid.behavior.time_budget import (
    BehaviorCooccurrenceRow,
    BoutDefinition,
    BoutSummaryRow,
    TimeBudgetComputationError,
    TimeBudgetRow,
    compute_behavior_cooccurrence,
    compute_bout_summary,
    compute_time_budget,
    extract_behavior_intervals,
    format_bout_summary_table,
    format_cooccurrence_table,
    format_time_budget_table,
    summarize_by_category,
    write_bout_summary_csv,
    write_time_budget_csv,
)
from annolid.core.behavior.spec import ProjectSchema


@dataclass(frozen=True)
class BehaviorTimeBudgetReport:
    summary: Sequence[TimeBudgetRow]
    warnings: Sequence[str]
    category_summary: Sequence[Tuple[str, float, int]]
    table_text: str
    bout_summary: Sequence[BoutSummaryRow] = ()
    bout_table_text: str = ""
    cooccurrence_summary: Sequence[BehaviorCooccurrenceRow] = ()
    cooccurrence_table_text: str = ""


def compute_behavior_time_budget_report(
    rows: Iterable[object],
    *,
    schema: Optional[ProjectSchema] = None,
    bout_definition: Optional[BoutDefinition] = None,
    session_duration: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    exclude_behaviors: Optional[Sequence[str]] = None,
    omit_excluded_rows: bool = False,
) -> BehaviorTimeBudgetReport:
    """Compute a reusable time-budget report from exported behavior rows."""
    data_rows = []
    local_warnings: List[str] = []
    for item in rows:
        if isinstance(item, Mapping):
            rec_t = item.get(
                "recording time",
                item.get("Recording time", item.get("timestamp", item.get("time"))),
            )
            beh = item.get("behavior", item.get("Behavior"))
            ev = item.get("event", item.get("Event", item.get("status")))
            subj = item.get("subject", item.get("Subject"))
            trial_t = item.get("trial time", item.get("Trial time", rec_t))
            mods = item.get("modifiers", item.get("Modifiers", ""))
            comment = item.get("comment", item.get("Comment", ""))
        elif isinstance(item, (tuple, list)) and len(item) >= 5:
            trial_t, rec_t, subj, beh, ev = item[:5]
            mods = item[5] if len(item) > 5 else ""
            comment = item[6] if len(item) > 6 else ""
        else:
            continue

        if rec_t is None:
            local_warnings.append(
                "Skipping event because no timestamp could be determined: "
                f"behavior={beh or '?'}, event={ev or '?'}"
            )
            continue
        data_rows.append(
            {
                "trial time": trial_t,
                "recording time": rec_t,
                "subject": str(subj or "").strip(),
                "behavior": str(beh or "").strip(),
                "event": str(ev or "").strip(),
                "modifiers": mods,
                "comment": comment,
            }
        )

    if not data_rows:
        return BehaviorTimeBudgetReport(
            summary=[],
            warnings=tuple(local_warnings),
            category_summary=(),
            table_text="No timestamped events remain after filtering.",
            bout_summary=(),
            bout_table_text="",
            cooccurrence_summary=(),
            cooccurrence_table_text="",
        )

    summary, compute_warnings = compute_time_budget(
        data_rows,
        session_duration=session_duration,
        start_time=start_time,
        end_time=end_time,
        exclude_behaviors=exclude_behaviors,
        omit_excluded_rows=omit_excluded_rows,
    )
    warnings: List[str] = local_warnings + list(compute_warnings)
    category_summary: List[Tuple[str, float, int]] = []
    if schema is not None and summary:
        category_summary = summarize_by_category(summary, schema)

    interval_rows, _ = extract_behavior_intervals(data_rows)
    cooccurrences = compute_behavior_cooccurrence(interval_rows)
    coocc_text = format_cooccurrence_table(cooccurrences) if cooccurrences else ""

    bout_summary: Sequence[BoutSummaryRow] = ()
    bout_table_text = ""
    if bout_definition is not None:
        computed_bouts = compute_bout_summary(
            interval_rows,
            definition=bout_definition,
        )
        bout_summary = tuple(computed_bouts)
        bout_table_text = format_bout_summary_table(
            computed_bouts,
            behavior_order=bout_definition.behaviors,
        )

    return BehaviorTimeBudgetReport(
        summary=tuple(summary),
        warnings=tuple(warnings),
        category_summary=tuple(category_summary),
        table_text=format_time_budget_table(summary, schema=schema),
        bout_summary=bout_summary,
        bout_table_text=bout_table_text,
        cooccurrence_summary=tuple(cooccurrences),
        cooccurrence_table_text=coocc_text,
    )


def write_behavior_time_budget_report_csv(
    report: BehaviorTimeBudgetReport,
    output_path: Path,
    *,
    schema: Optional[ProjectSchema] = None,
    bout_definition: Optional[BoutDefinition] = None,
) -> Optional[Path]:
    """Write the primary CSV and optional category summary CSV."""
    output_path = Path(output_path)
    write_time_budget_csv(report.summary, output_path, schema=schema)
    category_path: Optional[Path] = None
    if schema is not None and report.category_summary:
        category_path = output_path.with_name(
            output_path.stem + "_categories" + output_path.suffix
        )
        with category_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["Category", "TotalSeconds", "Occurrences"])
            for name, total, occurrences in report.category_summary:
                writer.writerow([name, f"{total:.6f}", int(occurrences)])
    if report.bout_summary:
        bout_path = output_path.with_name(
            output_path.stem + "_bouts" + output_path.suffix
        )
        write_bout_summary_csv(
            report.bout_summary,
            bout_path,
            behavior_order=bout_definition.behaviors if bout_definition else None,
        )
    return category_path


__all__ = [
    "BehaviorTimeBudgetReport",
    "TimeBudgetComputationError",
    "compute_behavior_time_budget_report",
    "write_behavior_time_budget_report_csv",
]
