"""Service-layer entry points for export workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from annolid.services.time_budget import (
    BehaviorTimeBudgetReport,
    write_behavior_time_budget_report_csv,
)


def build_yolo_dataset_from_index(*, index_file: Path, output_dir: Path, **kwargs: Any):
    from annolid.datasets.builders.label_index_yolo import build_yolo_from_label_index

    return build_yolo_from_label_index(
        index_file=Path(index_file),
        output_dir=Path(output_dir),
        **kwargs,
    )


def import_deeplabcut_dataset(*args: Any, **kwargs: Any):
    from annolid.datasets.importers.deeplabcut_training_data import (
        import_deeplabcut_training_data,
    )

    return import_deeplabcut_training_data(*args, **kwargs)


def export_labelme_json_to_csv(json_folder_path: str, **kwargs: Any):
    from annolid.annotation.labelme2csv import convert_json_to_csv

    return convert_json_to_csv(json_folder_path, **kwargs)


def export_behavior_time_budget(
    report: BehaviorTimeBudgetReport,
    output_path: Path,
    *,
    schema: Optional[object] = None,
):
    return write_behavior_time_budget_report_csv(
        report,
        Path(output_path),
        schema=schema,
    )


__all__ = [
    "build_yolo_dataset_from_index",
    "export_behavior_time_budget",
    "export_labelme_json_to_csv",
    "import_deeplabcut_dataset",
]
