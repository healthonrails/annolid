"""Service-layer entry points for dataset/pose import and export workflows."""

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


def import_deeplabcut_analysis(*args: Any, **kwargs: Any):
    from annolid.annotation.deeplabcut2labelme import deeplabcut_to_labelme_json

    return deeplabcut_to_labelme_json(*args, **kwargs)


def import_sleap_dataset(*args: Any, **kwargs: Any):
    from annolid.annotation.sleap2labelme import convert_sleap_h5_to_labelme_summary

    return convert_sleap_h5_to_labelme_summary(*args, **kwargs)


def import_pose_dataset(*args: Any, **kwargs: Any):
    from annolid.annotation.pose_interop.converters import import_pose_to_labelme

    return import_pose_to_labelme(*args, **kwargs)


def find_deeplabcut_analysis_file(video_path: str | Path) -> Optional[Path]:
    from annolid.annotation.pose_interop.deeplabcut import (
        find_dlc_analysis_file_for_video,
    )

    return find_dlc_analysis_file_for_video(video_path)


def export_pose_dataset(*args: Any, **kwargs: Any):
    from annolid.annotation.pose_interop.converters import (
        export_pose_dataset as _export_pose_dataset,
    )

    return _export_pose_dataset(*args, **kwargs)


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
    "export_pose_dataset",
    "find_deeplabcut_analysis_file",
    "import_deeplabcut_analysis",
    "import_deeplabcut_dataset",
    "import_pose_dataset",
    "import_sleap_dataset",
]
