from __future__ import annotations

import asyncio
import csv
import json
import os
from pathlib import Path

import pandas as pd
from PIL import Image

from annolid.annotation.labelme2csv import convert_json_to_csv
from annolid.core.agent.tools.dataset import AnnolidDatasetPrepareTool
from annolid.datasets.importers.deeplabcut_training_data import (
    DeepLabCutTrainingImportConfig,
    import_deeplabcut_training_data,
)
from annolid.postprocessing.zone_analysis_engine import GenericZoneEngine
from annolid.postprocessing.zone_schema import build_zone_shape, load_zone_shapes


def _write_labelme_pair(root: Path, stem: str = "frame_0001") -> None:
    root.mkdir(parents=True, exist_ok=True)
    img_path = root / f"{stem}.png"
    Image.new("RGB", (64, 64), color=(20, 30, 40)).save(img_path)
    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "mouse_nose",
                "points": [[20.0, 25.0]],
                "group_id": 1,
                "shape_type": "point",
                "flags": {"instance_label": "mouse", "display_label": "nose"},
            },
            {
                "label": "mouse",
                "points": [[10.0, 10.0], [40.0, 10.0], [40.0, 40.0], [10.0, 40.0]],
                "group_id": 1,
                "shape_type": "polygon",
                "flags": {},
            },
        ],
        "imagePath": img_path.name,
        "imageData": None,
        "imageHeight": 64,
        "imageWidth": 64,
    }
    (root / f"{stem}.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_coco_minimal(root: Path) -> None:
    ann_dir = root / "annotations"
    img_dir = root / "images"
    ann_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), color=(10, 10, 10)).save(img_dir / "img1.png")
    coco = {
        "images": [{"id": 1, "file_name": "img1.png", "width": 64, "height": 64}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 20, 20],
                "area": 400,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "mouse"}],
    }
    (ann_dir / "instances_train.json").write_text(json.dumps(coco), encoding="utf-8")
    (ann_dir / "instances_val.json").write_text(json.dumps(coco), encoding="utf-8")


def test_dataset_prepare_enforces_workspace_write_boundary_on_read_only_roots(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    readonly_labelme = tmp_path / "readonly_labelme"
    _write_labelme_pair(readonly_labelme / "train", "frame_0001")
    _write_labelme_pair(readonly_labelme / "val", "frame_0002")
    before_files = sorted(
        p.relative_to(readonly_labelme) for p in readonly_labelme.rglob("*")
    )

    tool = AnnolidDatasetPrepareTool(
        allowed_dir=workspace,
        allowed_read_roots=[readonly_labelme],
    )

    # Without output_dir inside workspace, mutating a read-only external root is blocked.
    denied_spec = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(readonly_labelme),
                mode="labelme_spec",
                allow_mutation=True,
            )
        )
    )
    assert denied_spec["ok"] is False
    assert (
        sorted(p.relative_to(readonly_labelme) for p in readonly_labelme.rglob("*"))
        == before_files
    )

    # With output_dir inside workspace, spec + YOLO export succeed without touching readonly_labelme.
    allowed_spec = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(readonly_labelme),
                mode="labelme_spec",
                output_dir=str(workspace / "spec_out"),
                allow_mutation=True,
            )
        )
    )
    assert allowed_spec["ok"] is True
    assert Path(allowed_spec["result"]["spec_path"]).is_relative_to(workspace.resolve())

    yolo_export = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(readonly_labelme),
                mode="yolo_from_labelme",
                output_dir=str(workspace / "yolo_out"),
                task="segmentation",
                allow_mutation=True,
            )
        )
    )
    assert yolo_export["ok"] is True
    assert Path(yolo_export["result"]["index_file"]).is_relative_to(workspace.resolve())
    assert Path(yolo_export["result"]["dataset_dir"]).is_relative_to(
        workspace.resolve()
    )
    assert (
        sorted(p.relative_to(readonly_labelme) for p in readonly_labelme.rglob("*"))
        == before_files
    )

    # COCO default spec path on read-only root is also blocked unless output_dir is inside workspace.
    readonly_coco = tmp_path / "readonly_coco"
    _write_coco_minimal(readonly_coco)
    coco_tool = AnnolidDatasetPrepareTool(
        allowed_dir=workspace,
        allowed_read_roots=[readonly_coco],
    )
    denied_coco = json.loads(
        asyncio.run(
            coco_tool.execute(
                dataset_folder=str(readonly_coco),
                mode="coco_spec",
                allow_mutation=True,
            )
        )
    )
    assert denied_coco["ok"] is False
    assert not (readonly_coco / "coco_spec.yaml").exists()


def test_deeplabcut_import_blocks_auxiliary_and_csv_path_escapes(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_img = outside / "secret_frame.png"
    Image.new("RGB", (64, 64), color=(1, 2, 3)).save(outside_img)

    dlc_root = workspace / "dlc_proj"
    seg_dir = dlc_root / "labeled-data" / "vid1"
    seg_dir.mkdir(parents=True)
    csv_text = "\n".join(
        [
            "scorer, test_scorer, test_scorer",
            "bodyparts, nose, nose",
            "coords, x, y",
            f"{outside_img}, 12.0, 15.0",
        ]
    )
    (seg_dir / "CollectedData_test.csv").write_text(csv_text, encoding="utf-8")

    # Direct importer rejects image paths resolving outside source_dir / labeled_data_root.
    summary = import_deeplabcut_training_data(
        DeepLabCutTrainingImportConfig(source_dir=dlc_root),
        write_pose_schema=False,
    )
    assert summary["json_written"] == 0
    assert not outside_img.with_suffix(".json").exists()

    # Agent tool blocks pose_schema_out and index_file outside workspace.
    tool = AnnolidDatasetPrepareTool(allowed_dir=workspace)
    escaped_schema = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(dlc_root),
                mode="deeplabcut_import",
                pose_schema_out=str(outside / "pose_schema.json"),
                allow_mutation=True,
            )
        )
    )
    assert escaped_schema["ok"] is False
    assert not (outside / "pose_schema.json").exists()

    escaped_index = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(dlc_root),
                mode="deeplabcut_import",
                index_file=str(outside / "label_index.jsonl"),
                allow_mutation=True,
            )
        )
    )
    assert escaped_index["ok"] is False
    assert not (outside / "label_index.jsonl").exists()


def test_zone_engine_caps_overlapping_barrier_and_aggregate_counts_and_supports_nested_points() -> (
    None
):
    zone_data = {
        "shapes": [
            build_zone_shape(
                "outer_chamber",
                [[0, 0], [100, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
                occupant_role="stim",
            ),
            build_zone_shape(
                "inner_chamber",
                [[20, 20], [60, 60]],
                shape_type="rectangle",
                zone_kind="chamber",
                occupant_role="stim",
            ),
            build_zone_shape(
                "mesh_north",
                [[0, 0], [100, 15]],
                shape_type="rectangle",
                zone_kind="barrier_edge",
            ),
            build_zone_shape(
                "mesh_west",
                [[0, 0], [15, 100]],
                shape_type="rectangle",
                zone_kind="barrier_edge",
            ),
        ]
    }
    specs = load_zone_shapes(zone_data)
    engine = GenericZoneEngine(specs, fps=10)
    df_cols = pd.DataFrame(
        [
            {
                "frame_number": 0,
                "instance_name": "mouse",
                "cx": 5.0,
                "cy": 5.0,
                "outer_chamber": 1,
                "inner_chamber": 1,
                "mesh_north": 1,
                "mesh_west": 1,
            },
            {
                "frame_number": 1,
                "instance_name": "mouse",
                "cx": 5.0,
                "cy": 5.0,
                "outer_chamber": 1,
                "inner_chamber": 1,
                "mesh_north": 1,
                "mesh_west": 1,
            },
        ]
    )
    res_cols = engine.analyze_instance(df_cols, "mouse")
    assert res_cols.frame_count == 2
    assert res_cols.barrier_adjacent_frames == 2
    assert res_cols.aggregate_occupancy_frames["all_chambers"] == 2
    assert res_cols.aggregate_occupancy_frames["stim_chambers"] == 2
    assert res_cols.aggregate_entry_counts["all_chambers"] == 1
    assert res_cols.aggregate_entry_counts["stim_chambers"] == 1

    # Opt-in coordinate multi-zone mode preserves enclosing chamber across nested sub-zone visit.
    nested_engine = GenericZoneEngine(
        specs,
        fps=10,
        allow_overlapping_point_zones=True,
    )
    df_xy = pd.DataFrame(
        [
            {"frame_number": 0, "instance_name": "mouse", "cx": 80.0, "cy": 80.0},
            {"frame_number": 1, "instance_name": "mouse", "cx": 40.0, "cy": 40.0},
            {"frame_number": 2, "instance_name": "mouse", "cx": 80.0, "cy": 80.0},
        ]
    )
    res_xy = nested_engine.analyze_instance(df_xy, "mouse")
    assert res_xy.occupancy_frames["outer_chamber"] == 3
    assert res_xy.entry_counts["outer_chamber"] == 1
    assert res_xy.occupancy_frames["inner_chamber"] == 1
    assert res_xy.entry_counts["inner_chamber"] == 1


def test_convert_json_to_csv_exports_circle_shapes_with_true_center(
    tmp_path: Path,
) -> None:
    folder = tmp_path / "session"
    folder.mkdir()
    zone_payload = {
        "shapes": [
            build_zone_shape(
                "arena_left",
                [[0, 0], [60, 100]],
                shape_type="rectangle",
                zone_kind="chamber",
            )
        ]
    }
    (tmp_path / "session_zones.json").write_text(
        json.dumps(zone_payload),
        encoding="utf-8",
    )

    frame_json = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "arena_dish",
                "points": [[50.0, 40.0], [60.0, 40.0]],
                "group_id": 2,
                "shape_type": "circle",
                "flags": {},
            }
        ],
        "imagePath": "session_000000000.jpg",
        "imageData": None,
        "imageHeight": 100,
        "imageWidth": 100,
    }
    (folder / "session_000000000.json").write_text(
        json.dumps(frame_json),
        encoding="utf-8",
    )

    tracking_csv = tmp_path / "session_tracking.csv"
    tracked_csv = tmp_path / "session_tracked.csv"
    out = convert_json_to_csv(
        str(folder),
        csv_file=str(tracking_csv),
        tracked_csv_file=str(tracked_csv),
    )
    assert out == str(tracking_csv)

    with tracking_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 1
    row = rows[0]
    assert row["instance_name"] == "arena_dish"
    assert int(row["tracking_id"]) == 2
    assert float(row["cx"]) == 50.0
    assert float(row["cy"]) == 40.0
    assert row["segmentation"] != ""
    assert float(row["x1"]) <= 41.0
    assert float(row["x2"]) >= 59.0

    with tracked_csv.open(newline="", encoding="utf-8") as fh:
        tracked_rows = list(csv.DictReader(fh))
    assert len(tracked_rows) == 1
    trow = tracked_rows[0]
    assert trow["instance_name"] == "arena_dish"
    assert float(trow["cx"]) == 50.0
    assert float(trow["cy"]) == 40.0
    assert int(trow["arena_left"]) == 1


def test_headless_qpa_defaults_to_offscreen_and_keeps_docks_compact() -> None:
    from qtpy import QtWidgets
    from annolid.gui.widgets.embedding_search_widget import EmbeddingSearchWidget
    from annolid.gui.widgets.video_manager import VideoManagerWidget

    assert os.environ.get("QT_QPA_PLATFORM") == "offscreen"
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    vm = VideoManagerWidget()
    es = EmbeddingSearchWidget()
    assert vm.minimumSizeHint().width() <= 295
    assert es.minimumSizeHint().width() <= 315
