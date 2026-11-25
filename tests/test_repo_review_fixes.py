"""Focused regression tests for repository audit fixes and new workflows."""

from __future__ import annotations

import base64
import csv
import json
import os
import subprocess
import sys
import textwrap
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# 1. Lean-install import isolation
# ---------------------------------------------------------------------------


def test_lean_install_entrypoints_import_without_optional_extras() -> None:
    """Core CLI, GUI, agent-tools, and service entrypoints must import cleanly
    when optional ML/scientific/cloud/audio/PDF extras are unavailable."""
    probe = textwrap.dedent(
        """
        import importlib
        import importlib.abc
        import sys

        BLOCKED = {
            "torch", "torchvision", "torchaudio", "ultralytics",
            "matplotlib", "scipy", "sklearn", "skimage", "h5py",
            "gdown", "tensorboard", "openai", "anthropic",
            "lancedb", "pyarrow", "tantivy", "fitz",
            "pyvips", "openslide", "ffpyplayer", "onnxruntime",
            "transformers", "diffusers", "accelerate", "safetensors",
            "sounddevice", "librosa", "numba", "pocket_tts",
        }

        class Blocker(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path, target=None):
                root = fullname.split(".", 1)[0]
                if root in BLOCKED:
                    raise ModuleNotFoundError(f"Blocked optional dep: {fullname}", name=fullname)
                return None

        sys.meta_path.insert(0, Blocker())

        targets = [
            "annolid.main",
            "annolid.engine.cli",
            "annolid.gui.launcher",
            "annolid.gui.app",
            "annolid.core.agent.tools",
            "annolid.gui.widgets.audio",
            "annolid.gui.widgets.ai_chat_widget",
            "annolid.annotation.polygons",
            "annolid.colab.service",
            "annolid.sim2real",
            "annolid.services.training_monitor_cli",
        ]
        for mod in targets:
            importlib.import_module(mod)
        print("LEAN_IMPORT_OK")
        """
    )
    env = dict(os.environ)
    env["QT_QPA_PLATFORM"] = "offscreen"
    env["ANNOLID_DISABLE_BOT_AUTOSTART"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", probe],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "LEAN_IMPORT_OK" in proc.stdout


# ---------------------------------------------------------------------------
# 2. GUI Undo / Redo, title, flags, shape safety
# ---------------------------------------------------------------------------


def test_canvas_undo_redo_roundtrip_and_branch_clear() -> None:
    from qtpy import QtCore, QtWidgets

    from annolid.gui.shape import Shape
    from annolid.gui.widgets.canvas import Canvas

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    canvas = Canvas()
    s1 = Shape(label="mouse_1", shape_type="polygon")
    s1.points = [QtCore.QPointF(0, 0), QtCore.QPointF(10, 0), QtCore.QPointF(10, 10)]
    s2 = Shape(label="mouse_2", shape_type="polygon")
    s2.points = [
        QtCore.QPointF(20, 20),
        QtCore.QPointF(30, 20),
        QtCore.QPointF(30, 30),
    ]

    assert not canvas.isShapeRestorable
    assert not canvas.isShapeRedoable

    canvas.loadShapes([s1], replace=True)
    assert canvas.isShapeRestorable
    assert not canvas.isShapeRedoable

    canvas.loadShapes([s1, s2], replace=True)
    assert len(canvas.shapes) == 2

    # Undo 1 -> back to [s1]
    canvas._restoring_history = True
    try:
        canvas.restoreShape()
        canvas.loadShapes(canvas.shapes, replace=True)
    finally:
        canvas._restoring_history = False
    assert [s.label for s in canvas.shapes] == ["mouse_1"]
    assert canvas.isShapeRestorable
    assert canvas.isShapeRedoable

    # Redo 1 -> back to [s1, s2]
    canvas._restoring_history = True
    try:
        canvas.redoShape()
        canvas.loadShapes(canvas.shapes, replace=True)
    finally:
        canvas._restoring_history = False
    assert [s.label for s in canvas.shapes] == ["mouse_1", "mouse_2"]
    assert not canvas.isShapeRedoable

    # Undo again, then make a fresh edit -> redo stack must clear.
    canvas._restoring_history = True
    try:
        canvas.restoreShape()
        canvas.loadShapes(canvas.shapes, replace=True)
    finally:
        canvas._restoring_history = False
    assert canvas.isShapeRedoable

    s3 = Shape(label="mouse_3", shape_type="point")
    s3.points = [QtCore.QPointF(5, 5)]
    canvas.loadShapes([s1, s3], replace=True)
    assert not canvas.isShapeRedoable

    canvas.resetState()
    assert not canvas.isShapeRestorable
    assert not canvas.isShapeRedoable
    canvas.close()


def test_shape_invalid_vertex_shape_raises_value_error() -> None:
    from qtpy import QtCore, QtGui

    from annolid.gui.shape import Shape

    shape = Shape(label="a", shape_type="polygon")
    shape.points = [QtCore.QPointF(1.0, 2.0)]
    path = QtGui.QPainterPath()
    orig = Shape.point_type
    try:
        Shape.point_type = 9999
        with pytest.raises(ValueError, match="unsupported vertex shape"):
            shape.drawVertex(path, 0)
    finally:
        Shape.point_type = orig


def test_flag_table_widget_dynamic_row_tracking_and_scoped_shortcuts() -> None:
    from qtpy import QtCore, QtWidgets

    from annolid.gui.widgets.flags import FlagTable, FlagTableWidget

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    flag_table = FlagTable()
    shortcuts = flag_table.findChildren(QtWidgets.QShortcut)
    if not shortcuts:
        from qtpy import QtGui

        shortcuts = flag_table.findChildren(QtGui.QShortcut)
    assert shortcuts
    assert all(sc.context() == QtCore.Qt.WidgetWithChildrenShortcut for sc in shortcuts)
    flag_table.close()

    widget = FlagTableWidget()
    toggled: list[tuple[str, bool]] = []
    widget.flagToggled.connect(lambda name, val: toggled.append((name, val)))
    widget.add_row("grooming", False)
    widget.add_row("rearing", False)
    assert widget._table.rowCount() == 2

    # Remove row 0 so "rearing" shifts from row 1 -> row 0, then toggle its checkbox.
    widget._table.removeRow(0)
    assert widget._table.rowCount() == 1
    cb = widget._table.cellWidget(0, widget.COLUMN_ACTIVE)
    cb.setChecked(True)
    assert toggled[-1] == ("rearing", True)
    widget.close()


def test_get_title_handles_regular_images_and_video_frame_zero() -> None:
    from annolid.gui.mixins.persistence_lifecycle_mixin import PersistenceLifecycleMixin

    dummy = SimpleNamespace(
        caption_widget=None,
        filename="/tmp/sample.png",
        video_file=None,
        video_loader=None,
        frame_number=None,
        fps=30.0,
    )
    assert (
        PersistenceLifecycleMixin.getTitle(dummy, clean=True) == "Annolid - sample.png"
    )
    assert (
        PersistenceLifecycleMixin.getTitle(dummy, clean=False)
        == "Annolid - sample.png*"
    )

    dummy_video = SimpleNamespace(
        caption_widget=None,
        filename="/tmp/frame_000000000.png",
        video_file="/tmp/clip.mp4",
        video_loader=SimpleNamespace(fps=30.0),
        behavior_controller=SimpleNamespace(events_count=0),
        _time_stamp="",
        frame_number=0,
        fps=30.0,
    )
    title_f0 = PersistenceLifecycleMixin.getTitle(dummy_video, clean=True)
    assert "Video Timestamp:00:00:00.000" in title_f0


# ---------------------------------------------------------------------------
# 3. Annotation IO, converters, SVG, and pose filtering
# ---------------------------------------------------------------------------


def test_labelme2csv_handles_mask_shape_without_unbound_local(tmp_path: Path) -> None:
    from PIL import Image

    from annolid.annotation.labelme2csv import convert_json_to_csv

    mask_img = Image.fromarray(np.ones((8, 8), dtype=np.uint8) * 255)
    buf = BytesIO()
    mask_img.save(buf, format="PNG")
    mask_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    payload = {
        "version": "5.0.0",
        "flags": {},
        "shapes": [
            {
                "label": "mouse_1",
                "points": [[10.0, 20.0], [18.0, 28.0]],
                "group_id": None,
                "shape_type": "mask",
                "mask": mask_b64,
                "flags": {},
            }
        ],
        "imagePath": "frame_000000001.png",
        "imageData": None,
        "imageHeight": 64,
        "imageWidth": 64,
    }
    (tmp_path / "frame_000000001.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    out_csv = tmp_path / "out.csv"
    convert_json_to_csv(str(tmp_path), str(out_csv), include_tracking_output=True)
    rows = list(csv.reader(out_csv.open(encoding="utf-8")))
    assert len(rows) == 2
    assert len(rows[0]) == len(rows[1]) == 11
    assert rows[1][7] == "mouse_1"


def test_labelme2yolo_circle_returns_two_tuple(tmp_path: Path) -> None:
    from annolid.annotation.labelme2yolo import Labelme2YOLO

    conv = Labelme2YOLO(json_dir=str(tmp_path), to_seg=False)
    conv.label_to_id_dict = {"ball": 0}
    label_id, bbox = conv.circle_shape_to_yolo(
        {"label": "ball", "points": [[50.0, 50.0], [60.0, 50.0]]},
        image_height=100,
        image_width=100,
    )
    assert label_id == 0
    assert len(bbox) == 4
    assert pytest.approx(bbox, abs=1e-4) == [0.5, 0.5, 0.2, 0.2]


def test_coco2yolo_segmentation_has_no_duplicate_class_or_bbox_vertices(
    tmp_path: Path,
) -> None:
    from PIL import Image

    from annolid.annotation.coco2yolo import create_dataset

    coco_dir = tmp_path / "coco"
    coco_dir.mkdir()
    Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(coco_dir / "img0.png")
    coco = {
        "images": [{"id": 1, "file_name": "img0.png", "width": 100, "height": 100}],
        "categories": [{"id": 7, "name": "mouse"}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 7,
                "bbox": [10, 10, 20, 20],
                "segmentation": [[10, 10, 30, 10, 30, 30, 10, 30]],
                "iscrowd": 0,
            }
        ],
    }
    coco_json = coco_dir / "annotations.json"
    coco_json.write_text(json.dumps(coco), encoding="utf-8")
    out_dir = tmp_path / "yolo_out"
    create_dataset(
        json_file=str(coco_json),
        results_dir=str(out_dir),
        dataset_type="train",
        is_segmentation=True,
    )
    txt_file = out_dir / "train" / "labels" / "img0.txt"
    tokens = txt_file.read_text(encoding="utf-8").strip().split()
    # 1 class id + 4 polygon vertices (8 floats) = 9 tokens (no extra bbox coords!)
    assert len(tokens) == 9
    assert tokens[0] == "0"


def test_keypoints_merge_shapes_preserves_ungrouped_instances() -> None:
    from annolid.annotation.keypoints import merge_shapes

    s1 = {
        "label": "mouse",
        "shape_type": "polygon",
        "points": [[0, 0], [10, 0], [10, 10]],
        "group_id": None,
    }
    s2 = {
        "label": "mouse",
        "shape_type": "polygon",
        "points": [[50, 50], [60, 50], [60, 60]],
        "group_id": None,
    }
    merged = merge_shapes([s1], [s2])
    assert len(merged) == 2


def test_masks_and_timestamps_edge_cases() -> None:
    from annolid.annotation.masks import (
        binary_mask_to_coco_rle,
        calculate_mask_centroid,
        mask_perimeter,
        mask_to_polygons,
    )
    from annolid.annotation.timestamps import (
        convert_frame_number_to_time,
        convert_time_to_frame_number,
    )

    empty = np.zeros((16, 16), dtype=np.uint8)
    empty_rle = binary_mask_to_coco_rle(empty)
    assert mask_perimeter(empty_rle) == 0.0
    assert calculate_mask_centroid(empty_rle) == (0, 0)

    two_blobs = np.zeros((40, 40), dtype=np.uint8)
    two_blobs[2:12, 2:12] = 1
    two_blobs[25:35, 25:35] = 1
    polys, _has_holes = mask_to_polygons(two_blobs, use_convex_hull=True)
    assert len(polys) == 2

    assert convert_time_to_frame_number("00:00:01.500", fps=30.0) == 45
    assert convert_frame_number_to_time(10, fps=0.0) == "00:00:00.000"


def test_svg_import_transform_composition_and_clippath(tmp_path: Path) -> None:
    from annolid.io.vector.svg_import import import_svg_paths

    svg = textwrap.dedent(
        """
        <svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
          <clipPath id="c1"><rect x="0" y="0" width="50" height="50"/></clipPath>
          <g transform="translate(10, 0) scale(2, 2)">
            <rect x="2" y="3" width="5" height="5" />
          </g>
        </svg>
        """
    )
    svg_path = tmp_path / "sample.svg"
    svg_path.write_text(svg, encoding="utf-8")
    paths = import_svg_paths(svg_path)
    # clipPath rect must be skipped; transformed rect origin (2, 3) -> translate(10,0) * scale(2,2) -> (14, 6)
    assert len(paths) == 1
    xs = [pt[0] for pt in paths[0].points]
    ys = [pt[1] for pt in paths[0].points]
    assert pytest.approx(min(xs), abs=1e-4) == 14.0
    assert pytest.approx(min(ys), abs=1e-4) == 6.0


def test_pose_filtering_splits_gaps_and_updates_annotation_store(
    tmp_path: Path,
) -> None:
    from annolid.annotation.pose_interop.model import (
        PoseDataset,
        PoseFrame,
        PoseInstance,
        PosePoint,
    )
    from annolid.postprocessing.pose_filtering import (
        PoseFilterConfig,
        _smooth_contiguous_runs,
        _write_filtered_labelme_preserving_non_points,
    )
    from annolid.utils.annotation_store import AnnotationStore

    frames = np.array([0, 1, 2, 10, 11, 12], dtype=int)
    xs = np.array([0.0, 0.0, 0.0, 100.0, 100.0, 100.0], dtype=float)
    ys = np.array([0.0, 0.0, 0.0, 50.0, 50.0, 50.0], dtype=float)
    valid = np.ones_like(xs, dtype=bool)
    cfg = PoseFilterConfig(window_length=3, method="median")
    s_xs, s_ys, _ = _smooth_contiguous_runs(xs, ys, valid, cfg, frame_indices=frames)
    # Because frames jump from 2 -> 10, runs must stay isolated (no blending across the gap).
    assert np.allclose(s_xs[:3], 0.0)
    assert np.allclose(s_xs[3:], 100.0)
    assert np.allclose(s_ys[:3], 0.0)
    assert np.allclose(s_ys[3:], 50.0)

    store = AnnotationStore(tmp_path / "video_annotations.ndjson")
    store.append_frame(
        {
            "frame": 0,
            "version": "5.0.0",
            "flags": {},
            "imagePath": "frame_000000000.png",
            "imageData": None,
            "imageHeight": 100,
            "imageWidth": 100,
            "shapes": [
                {
                    "label": "mouse_nose",
                    "shape_type": "point",
                    "points": [[1.0, 2.0]],
                    "group_id": None,
                    "flags": {},
                },
                {
                    "label": "mouse",
                    "shape_type": "polygon",
                    "points": [[0, 0], [10, 0], [10, 10]],
                    "group_id": None,
                    "flags": {},
                },
            ],
        },
    )
    filtered_ds = PoseDataset(
        keypoints=["nose"],
        instances=["mouse"],
        frames=[
            PoseFrame(
                frame_index=0,
                image_path="frame_000000000.png",
                image_height=100,
                image_width=100,
                instances=[
                    PoseInstance(
                        instance_name="mouse",
                        keypoints={"nose": PosePoint(x=42.0, y=43.0, visibility=2)},
                    )
                ],
            )
        ],
    )
    _write_filtered_labelme_preserving_non_points(
        source_dir=tmp_path,
        output_dir=tmp_path,
        filtered_ds=filtered_ds,
    )
    store.invalidate_cache()
    reloaded = store.get_frame(0)
    assert reloaded is not None
    point_shapes = [s for s in reloaded["shapes"] if s["shape_type"] == "point"]
    assert len(point_shapes) == 1
    assert point_shapes[0]["points"] == [[42.0, 43.0]]
    assert any(
        s["label"] == "mouse" and s["shape_type"] == "polygon"
        for s in reloaded["shapes"]
    )


# ---------------------------------------------------------------------------
# 4. Security & Sandbox hardening
# ---------------------------------------------------------------------------


def test_sandbox_execution_blocks_dunder_escape_and_allows_helpers() -> None:
    from annolid.infrastructure.sandbox.execution import (
        SandboxExecutionError,
        execute_generated_analysis,
    )

    with pytest.raises(SandboxExecutionError):
        execute_generated_analysis(
            "result = ().__class__.__bases__[0].__subclasses__()",
            inputs={},
        )

    with pytest.raises(SandboxExecutionError):
        execute_generated_analysis("import os\nresult = os.getcwd()", inputs={})

    helper_ok = execute_generated_analysis(
        "def double(x):\n    return x * 2\ndef run(inputs):\n    return {'answer': double(inputs['v'])}",
        inputs={"v": 21},
    )
    assert helper_ok == {"answer": 42}


def test_git_cli_tool_blocks_no_index_and_attached_c_flags(tmp_path: Path) -> None:
    import asyncio

    from annolid.core.agent.tools.git import GitCliTool

    tool = GitCliTool(allowed_dir=tmp_path)
    out1 = asyncio.run(tool.execute(command="diff --no-index /etc/hosts /etc/hosts"))
    assert "blocked" in out1.lower() or "error" in out1.lower()

    out2 = asyncio.run(tool.execute(command="-C/etc status"))
    assert "blocked" in out2.lower() or "error" in out2.lower()


def test_colab_packaging_handles_dotdir_and_rejects_symlink_escape(
    tmp_path: Path,
) -> None:
    import zipfile

    from annolid.colab.packaging import _write_dir_to_zip

    dot_parent = tmp_path / ".annolid" / "dataset"
    dot_parent.mkdir(parents=True)
    (dot_parent / "sample.txt").write_text("ok", encoding="utf-8")

    out_zip = tmp_path / "ds.zip"
    with zipfile.ZipFile(out_zip, "w") as zf:
        count = _write_dir_to_zip(
            zf, dot_parent, arc_prefix="dataset", allowed_read_roots=[tmp_path]
        )
    assert count == 1

    outside = tmp_path / "secret.txt"
    outside.write_text("topsecret", encoding="utf-8")
    inside = tmp_path / "safe_root"
    inside.mkdir()
    escape_link = inside / "leak.txt"
    escape_link.symlink_to(outside)

    with zipfile.ZipFile(tmp_path / "escape.zip", "w") as zf:
        with pytest.raises(PermissionError):
            _write_dir_to_zip(
                zf, inside, arc_prefix="safe", allowed_read_roots=[inside]
            )


def test_flybody_dotted_callable_rejects_arbitrary_modules() -> None:
    from annolid.simulation.adapters.flybody import _resolve_dotted_callable

    with pytest.raises(ValueError, match="outside allowed"):
        _resolve_dotted_callable("os:system")
    with pytest.raises(ValueError, match="outside allowed"):
        _resolve_dotted_callable("builtins:eval")


def test_exec_tool_blocks_parent_directory_traversal(tmp_path: Path) -> None:
    from annolid.core.agent.tools.shell import ExecTool

    tool = ExecTool(working_dir=str(tmp_path), restrict_to_workspace=True)
    err = tool._guard_command("cd .. && ls", str(tmp_path))
    assert err is not None


# ---------------------------------------------------------------------------
# 5. Behavior, BORIS, IRR, TCN, Sim2Real, Distribution guard, & Runs CLI
# ---------------------------------------------------------------------------


def test_normalize_event_label_word_boundaries() -> None:
    from annolid.behavior.event_utils import normalize_event_label

    assert normalize_event_label("defending") is None
    assert normalize_event_label("extended") is None
    assert normalize_event_label("nonstop") is None
    assert normalize_event_label("start_grooming") == "start"
    assert normalize_event_label("grooming_stop") == "end"
    assert normalize_event_label("end") == "end"


def test_boris_unsorted_and_zero_duration_roundtrip(tmp_path: Path) -> None:
    from annolid.behavior.boris_io import (
        ParsedEventRow,
        import_boris_project,
        write_boris_aggregated_csv,
    )

    project = {
        "project_name": "test_proj",
        "subjects_conf": {"0": {"name": "mouse_1"}},
        "behaviors_conf": {"0": {"code": "groom", "type": "State event"}},
        "observations": {
            "obs1": {
                "events": [
                    [5.0, "mouse_1", "groom", "", ""],
                    [1.0, "mouse_1", "groom", "", ""],
                ]
            }
        },
    }
    boris_path = tmp_path / "demo.boris"
    boris_path.write_text(json.dumps(project), encoding="utf-8")
    imported = import_boris_project(boris_path)
    obs = imported.observations[0]
    assert [(r.recording_time, r.event_kind) for r in obs.events] == [
        (1.0, "start"),
        (5.0, "end"),
    ]

    zero_rows = [
        ParsedEventRow(
            line_number=1,
            recording_time=2.0,
            raw_recording_time="2.0",
            trial_time=2.0,
            subject="mouse_1",
            has_explicit_subject=True,
            behavior="groom",
            raw_event="start",
            event_kind="start",
        ),
        ParsedEventRow(
            line_number=2,
            recording_time=2.0,
            raw_recording_time="2.0",
            trial_time=2.0,
            subject="mouse_1",
            has_explicit_subject=True,
            behavior="groom",
            raw_event="stop",
            event_kind="end",
        ),
    ]
    agg_csv = tmp_path / "agg.csv"
    write_boris_aggregated_csv(zero_rows, agg_csv)
    parsed = list(csv.DictReader(agg_csv.open(encoding="utf-8")))
    assert len(parsed) == 1
    assert float(parsed[0]["Duration (s)"]) == 0.0


def test_irr_empty_common_behaviors_fails_min_kappa(tmp_path: Path) -> None:
    from annolid.services.behavior_analytics import run_behavior_irr_cli

    r1 = tmp_path / "r1.csv"
    r2 = tmp_path / "r2.csv"
    r1.write_text(
        "Timestamp (s),Behavior,Event\n1.0,groom,start\n2.0,groom,stop\n",
        encoding="utf-8",
    )
    r2.write_text(
        "Timestamp (s),Behavior,Event\n1.0,rear,start\n2.0,rear,stop\n",
        encoding="utf-8",
    )
    rc = run_behavior_irr_cli(
        csv_paths=[str(r1), str(r2)],
        min_kappa=0.5,
        json_stdout=True,
    )
    assert rc == 2


def test_tcn_dataset_pads_with_ignore_index_and_nan_norm(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    from annolid.behavior.tcn import (
        TCNFeatureConfig,
        TCNSequenceDataset,
        TCNSession,
        fit_normalization,
    )

    feat_csv = tmp_path / "feat.csv"
    feat_csv.write_text(
        "frame,f1,f2\n0,1.0,NaN\n1,3.0,4.0\n2,5.0,6.0\n", encoding="utf-8"
    )
    lbl_csv = tmp_path / "lbl.csv"
    lbl_csv.write_text("frame,bg,groom\n0,1,0\n1,0,1\n2,1,0\n", encoding="utf-8")

    sess = TCNSession(session_id="s1", features=feat_csv, labels=lbl_csv)
    cfg = TCNFeatureConfig()
    norm = fit_normalization([sess], feature_config=cfg)
    assert np.all(np.isfinite(norm.mean))
    assert np.all(np.isfinite(norm.std))

    ds = TCNSequenceDataset(
        sessions=[sess],
        feature_config=cfg,
        label_names=["bg", "groom"],
        sequence_length=6,
    )
    sample = ds[0]
    assert int(sample["valid_length"]) == 3
    assert torch.equal(sample["labels"][3:], torch.full((3,), -100, dtype=torch.long))


def test_sim2real_export_keeps_source_groups_disjoint_across_splits(
    tmp_path: Path,
) -> None:
    from PIL import Image

    from annolid.sim2real.generators import _export_yolo_dataset

    img_dir = tmp_path / "src_imgs"
    img_dir.mkdir()
    records = []
    # 6 repeated copies of the same real source_group + 10 distinct sim frames
    real_img = img_dir / "real_0.png"
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(real_img)
    for i in range(6):
        records.append(
            {
                "stem": f"real_0_copy_{i}",
                "image_path": str(real_img),
                "width": 64,
                "height": 64,
                "instances": [
                    {
                        "class_name": "mouse",
                        "polygons": [[[10, 10], [30, 10], [30, 30], [10, 30]]],
                    }
                ],
                "source_group": f"real::{real_img.resolve()}",
            }
        )
    for i in range(10):
        p = img_dir / f"sim_{i}.png"
        Image.fromarray(np.full((64, 64, 3), i * 10, dtype=np.uint8)).save(p)
        records.append(
            {
                "stem": f"sim_{i}",
                "image_path": str(p),
                "width": 64,
                "height": 64,
                "instances": [
                    {
                        "class_name": "mouse",
                        "polygons": [[[10, 10], [30, 10], [30, 30], [10, 30]]],
                    }
                ],
                "source_group": f"sim::{i}",
            }
        )

    yolo_dir = tmp_path / "yolo"
    _yaml, _names, n_train, n_val = _export_yolo_dataset(
        records,
        yolo_dir=yolo_dir,
        task="yolo_seg",
        export_level="instances",
        species="mouse",
        val_fraction=0.25,
        seed=42,
    )
    train_stems = {p.stem for p in (yolo_dir / "images" / "train").glob("*")}
    val_stems = {p.stem for p in (yolo_dir / "images" / "val").glob("*")}
    # Real group copies must never appear in BOTH train and val.
    real_in_train = any("real_0" in s for s in train_stems)
    real_in_val = any("real_0" in s for s in val_stems)
    assert not (real_in_train and real_in_val)
    assert n_train > 0 and n_val > 0


def test_distribution_guard_rejects_any_mlpackage_and_nested_project_json() -> None:
    from scripts.check_distribution_artifacts import is_forbidden_member

    assert is_forbidden_member("annolid/custom/model.mlpackage/Data/model.mil")
    assert is_forbidden_member("annolid/nested/project.annolid.json")


def test_cli_help_unknown_flags_and_runs_subcommand(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from annolid.engine.cli import main as cli_main

    # time-budget --help must not crash with argparse percent formatting error
    with pytest.raises(SystemExit) as exc:
        cli_main(["time-budget", "--help"])
    assert exc.value.code == 0

    # unrecognized trailing flags on callable subcommands must error
    with pytest.raises(SystemExit) as exc_bad:
        cli_main(["list-models", "--definitely-not-a-flag"])
    assert exc_bad.value.code != 0

    # annolid-run runs list / inspect / diff / health
    run_a = tmp_path / "runs" / "exp_a"
    run_b = tmp_path / "runs" / "exp_b"
    run_a.mkdir(parents=True)
    run_b.mkdir(parents=True)
    (run_a / "results.csv").write_text(
        "epoch,train/box_loss,val/box_loss,metrics/mAP50(B)\n"
        "1,1.0,1.1,0.40\n2,0.8,0.9,0.55\n",
        encoding="utf-8",
    )
    (run_a / "args.yaml").write_text("lr0: 0.01\nepochs: 50\n", encoding="utf-8")
    (run_b / "results.csv").write_text(
        "epoch,train/box_loss,val/box_loss,metrics/mAP50(B)\n"
        "1,1.0,1.1,0.40\n2,NaN,0.9,0.55\n",
        encoding="utf-8",
    )
    (run_b / "args.yaml").write_text("lr0: 0.001\nepochs: 50\n", encoding="utf-8")

    capsys.readouterr()
    assert (
        cli_main(["runs", "list", "--runs-root", str(tmp_path / "runs"), "--json"]) == 0
    )
    listed = json.loads(capsys.readouterr().out)
    assert listed["count"] == 2

    assert cli_main(["runs", "inspect", str(run_a), "--json"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["last_step"] == 2

    assert (
        cli_main(["runs", "diff", str(run_a), str(run_b), "--only-diffs", "--json"])
        == 0
    )
    diffed = json.loads(capsys.readouterr().out)
    assert any(row["key"] == "lr0" and row["differs"] for row in diffed["rows"])

    rc_crit = cli_main(["runs", "health", str(run_b), "--fail-on-critical", "--json"])
    assert rc_crit == 2
    health_payload = json.loads(capsys.readouterr().out)
    assert health_payload["ok"] is False
    assert any(f["code"] == "nan_metric" for f in health_payload["findings"])


def test_colab_package_training_dataset_enforces_allowed_read_roots(
    tmp_path: Path,
) -> None:
    from annolid.colab.packaging import package_training_dataset

    allowed = tmp_path / "workspace"
    allowed.mkdir()
    outside = tmp_path / "outside_secret"
    outside.mkdir()
    (outside / "secret.jpg").write_bytes(b"secret")

    yaml_path = allowed / "data.yaml"
    yaml_path.write_text(
        f"path: {outside}\ntrain: .\nval: .\nnames: ['mouse']\n",
        encoding="utf-8",
    )
    with pytest.raises(PermissionError):
        package_training_dataset(
            yaml_path,
            model="yolo",
            staging_dir=tmp_path / "staging",
            allowed_read_roots=[allowed],
        )


def test_sim2real_evaluator_populates_val_pose_map50(tmp_path: Path) -> None:
    from annolid.sim2real.evaluator import evaluate_iteration_outputs

    train_out = tmp_path / "train_out"
    train_out.mkdir()
    (train_out / "results.csv").write_text(
        "epoch,train/box_loss,val/box_loss,metrics/mAP50(B),metrics/mAP50(P),metrics/mAP50-95(P)\n"
        "1,1.2,1.1,0.50,0.62,0.40\n"
        "2,0.9,0.8,0.65,0.81,0.58\n",
        encoding="utf-8",
    )
    ev = evaluate_iteration_outputs(
        example_videos=[],
        prediction_dirs=[],
        training_output_dir=train_out,
    )
    assert ev.best_train_epoch == 2
    assert ev.val_pose_map50 == pytest.approx(0.81, abs=1e-4)
    assert ev.sim2real_score == pytest.approx(0.81, abs=1e-4)


def test_coco2labelme_sanitizes_parent_traversal_and_zone_clean_nan() -> None:
    import pandas as pd

    from annolid.annotation.coco2labelme import _image_target_relative
    from annolid.services.zone_analytics import _clean_nan_records

    rel = _image_target_relative("../../etc/passwd.png", image_id=5)
    assert ".." not in rel.parts
    assert not rel.is_absolute()

    df = pd.DataFrame(
        {
            "metric": [1.0, float("inf"), float("nan")],
            "tags": [["a", "b"], [], ["c"]],
        }
    )
    cleaned = _clean_nan_records(df)
    assert cleaned[0]["metric"] == 1.0
    assert cleaned[1]["metric"] is None
    assert cleaned[2]["metric"] is None
    assert cleaned[0]["tags"] == ["a", "b"]


def test_tracking_gap_summary_reads_annotation_store(tmp_path: Path) -> None:
    from annolid.postprocessing.tracking_reports import summarize_tracking_gaps
    from annolid.utils.annotation_store import AnnotationStore

    vid_dir = tmp_path / "demo_clip"
    vid_dir.mkdir()
    store = AnnotationStore(vid_dir / "demo_clip_annotations.ndjson")
    store.append_frame(
        {
            "frame": 0,
            "shapes": [
                {
                    "label": "mouse_1",
                    "shape_type": "polygon",
                    "points": [[0, 0], [5, 0], [5, 5]],
                }
            ],
        }
    )
    store.append_frame(
        {
            "frame": 4,
            "shapes": [
                {
                    "label": "mouse_1",
                    "shape_type": "polygon",
                    "points": [[0, 0], [5, 0], [5, 5]],
                }
            ],
        }
    )
    summary = summarize_tracking_gaps(str(tmp_path / "demo_clip.mp4"), fps=10.0)
    assert summary["total_frames"] == 5
    assert any(inst["instance"] == "mouse_1" for inst in summary["instances"])
