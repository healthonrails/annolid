from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pytest
import yaml
from qtpy import QtCore, QtGui, QtWidgets

from annolid.annotation import coco2labelme, labelme2coco
from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    set_keypoint_visibility_on_shape_object,
)
from annolid.annotation.pose_schema import (
    PoseSchema,
    find_pose_schema_in_dir,
    get_pose_schema_preset,
    infer_pose_skeleton_edges,
    list_pose_schema_presets,
)
from annolid.behavior.project_schema import ProjectSchema
from annolid.gui.mixins.schema_behavior_loader_mixin import SchemaBehaviorLoaderMixin
from annolid.gui.mixins.tooling_dialogs_mixin import ToolingDialogsMixin
from annolid.gui.shape import Shape
from annolid.gui.widgets.canvas import Canvas
from annolid.gui.widgets.convert_coco_dialog import ConvertCOODialog
from annolid.gui.widgets.convert_coco2labelme_dialog import ConvertCOCO2LabelMeDialog
from annolid.gui.widgets.keypoint_sequencer import KeypointSequencerWidget
from annolid.gui.widgets.project_wizard import PoseSchemaPage
from annolid.postprocessing import skeletonization as skel_mod
from annolid.postprocessing.track_qc import evaluate_track_qc
from annolid.services.body_part_segmentation import _load_skeleton_edges
from annolid.utils.draw import draw_keypoint_connections, draw_pose_skeleton

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")


def _ensure_qapp() -> QtWidgets.QApplication:
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


def _make_point_shape(
    label: str,
    x: float,
    y: float,
    *,
    group_id=None,
    flags=None,
    other_data=None,
    visibility: int | None = None,
    selected: bool = False,
) -> Shape:
    shape = Shape(label=label, shape_type="point", group_id=group_id)
    shape.addPoint(QtCore.QPointF(float(x), float(y)))
    if flags:
        shape.flags = dict(flags)
    if other_data:
        shape.other_data = dict(other_data)
    if visibility is not None:
        set_keypoint_visibility_on_shape_object(shape, visibility)
    shape.selected = bool(selected)
    return shape


# ---------------------------------------------------------------------------
# 1. PoseSchema core, presets, COCO conversion, validation, and edge inference
# ---------------------------------------------------------------------------


def test_pose_schema_dedup_skeleton_alias_and_longest_prefix_stripping() -> None:
    schema = PoseSchema.from_dict(
        {
            "keypoints": ["nose", "left_ear", "right_ear", "tail_base"],
            # Uses legacy "skeleton" key with 1-indexed COCO int pairs + reverse duplicate
            "skeleton": [[1, 2], [2, 1], [1, 3], [1, 4]],
            "instances": ["mouse", "mouse_1", "mouse_1_"],
            "instance_separator": "_",
        }
    )
    assert schema.edges == [
        ("nose", "left_ear"),
        ("nose", "right_ear"),
        ("nose", "tail_base"),
    ]
    assert schema.instances == ["mouse", "mouse_1"]
    assert schema.strip_instance_prefix("mouse_1_nose") == ("mouse_1", "nose")
    assert schema.strip_instance_prefix("mouse_nose") == ("mouse", "nose")
    assert schema.to_coco_skeleton() == [[1, 2], [1, 3], [1, 4]]

    roundtrip = PoseSchema.from_coco_category(
        {"name": "mouse", "keypoints": schema.keypoints, "skeleton": [[1, 2], [1, 4]]}
    )
    assert roundtrip.keypoints == schema.keypoints
    assert roundtrip.edges == [("nose", "left_ear"), ("nose", "tail_base")]
    assert ("left_ear", "right_ear") in roundtrip.symmetry_pairs


def test_pose_schema_symmetry_inference_and_validation() -> None:
    pairs = PoseSchema.infer_symmetry_pairs(
        [
            "nose",
            "ear_l",
            "ear_r",
            "l_paw",
            "r_paw",
            "LEar",
            "REar",
            "tail_base",
            "tair_base",
        ]
    )
    assert ("ear_l", "ear_r") in pairs
    assert ("l_paw", "r_paw") in pairs
    assert ("LEar", "REar") in pairs
    # Must NOT match "l_" inside "tail_base"!
    assert ("tail_base", "tair_base") not in pairs

    bad = PoseSchema(
        keypoints=["nose", "nose", "left_ear"],
        edges=[
            ("nose", "nose"),
            ("nose", "missing_kp"),
            ("nose", "left_ear"),
            ("left_ear", "nose"),
        ],
        symmetry_pairs=[("left_ear", "right_ear")],
        flip_idx=[0, 2],
    )
    issues = bad.validate()
    assert any("Duplicate keypoint" in msg for msg in issues)
    assert any("Self-loop" in msg for msg in issues)
    assert any("missing_kp" in msg for msg in issues)
    assert any("Duplicate skeleton edge" in msg for msg in issues)
    assert any("flip_idx" in msg for msg in issues)


def test_pose_schema_presets_and_full_body_edge_inference(tmp_path: Path) -> None:
    names = list_pose_schema_presets()
    assert len(names) >= 5
    for name in names:
        preset = get_pose_schema_preset(name)
        assert len(preset.keypoints) >= 5
        assert len(preset.edges) >= 4
        assert preset.validate() == []

    rodent_edges = infer_pose_skeleton_edges(
        ["nose", "left_ear", "right_ear", "neck", "spine", "tail_base", "tail_tip"]
    )
    edge_set = {tuple(sorted(e)) for e in rodent_edges}
    assert tuple(sorted(("nose", "neck"))) in edge_set
    assert tuple(sorted(("neck", "spine"))) in edge_set
    assert tuple(sorted(("spine", "tail_base"))) in edge_set
    assert tuple(sorted(("tail_base", "tail_tip"))) in edge_set
    assert tuple(sorted(("nose", "left_ear"))) in edge_set

    schema_file = tmp_path / "pose_schema.yaml"
    get_pose_schema_preset("Fly (5 keypoints)").save(schema_file)
    assert find_pose_schema_in_dir(tmp_path) == schema_file


def test_pose_schema_normalize_prefixed_deduplicates_edges() -> None:
    schema = PoseSchema.from_dict(
        {
            "keypoints": [
                "resident_nose",
                "resident_tail_base",
                "intruder_nose",
                "intruder_tail_base",
            ],
            "edges": [
                ["resident_nose", "resident_tail_base"],
                ["intruder_nose", "intruder_tail_base"],
            ],
        }
    )
    schema.normalize_prefixed_keypoints()
    assert schema.instances == ["resident", "intruder"]
    assert schema.keypoints == ["nose", "tail_base"]
    assert schema.edges == [("nose", "tail_base")]


# ---------------------------------------------------------------------------
# 2. COCO export & import skeleton + visibility round-trip
# ---------------------------------------------------------------------------


def test_labelme2coco_and_coco2labelme_preserve_skeleton_and_visibility(
    tmp_path: Path,
) -> None:
    pytest.importorskip("pycocotools.mask")

    src = tmp_path / "labelme_pose"
    src.mkdir()
    schema = PoseSchema(
        keypoints=["nose", "left_ear", "right_ear", "tail_base"],
        edges=[("nose", "left_ear"), ("nose", "right_ear"), ("nose", "tail_base")],
        symmetry_pairs=[("left_ear", "right_ear")],
        instances=["resident", "intruder"],
    )
    schema.save(src / "pose_schema.json")

    for idx in range(2):
        img = src / f"frame_{idx:03d}.png"
        Image.new("RGB", (120, 100), color=(30, 30, 30)).save(img)
        payload = {
            "version": "5.5.0",
            "imagePath": img.name,
            "imageHeight": 100,
            "imageWidth": 120,
            "shapes": [
                {
                    "label": "resident_nose",
                    "shape_type": "point",
                    "points": [[20 + idx, 25]],
                    "group_id": None,
                    "kp_visibility": 2,
                    "flags": {"kp_visible": True},
                },
                {
                    "label": "resident_left_ear",
                    "shape_type": "point",
                    "points": [[15 + idx, 18]],
                    "group_id": None,
                    "kp_visibility": 1,
                    "flags": {"kp_visible": False},
                },
                {
                    "label": "resident_tail_base",
                    "shape_type": "point",
                    "points": [[50 + idx, 30]],
                    "group_id": None,
                    "kp_visibility": 2,
                    "flags": {"kp_visible": True},
                },
                {
                    "label": "intruder_nose",
                    "shape_type": "point",
                    "points": [[80 + idx, 70]],
                    "group_id": None,
                    "kp_visibility": 2,
                    "flags": {"kp_visible": True},
                },
                {
                    "label": "intruder_tail_base",
                    "shape_type": "point",
                    "points": [[105 + idx, 75]],
                    "group_id": None,
                    "kp_visibility": 2,
                    "flags": {"kp_visible": True},
                },
            ],
        }
        (src / f"frame_{idx:03d}.json").write_text(
            json.dumps(payload), encoding="utf-8"
        )

    coco_out = tmp_path / "coco_out"
    list(
        labelme2coco.convert(
            str(src),
            str(coco_out),
            labels_file=None,
            train_valid_split=0.5,
            output_mode="keypoints",
        )
    )

    train_coco = json.loads(
        (coco_out / "train" / "annotations.json").read_text(encoding="utf-8")
    )
    cat = train_coco["categories"][0]
    # Preserves PoseSchema order (not alphabetical) and 1-indexed COCO skeleton!
    assert cat["keypoints"] == ["nose", "left_ear", "right_ear", "tail_base"]
    assert cat["skeleton"] == [[1, 2], [1, 3], [1, 4]]
    # Two ungrouped instance-prefixed animals split into 2 annotations!
    assert len(train_coco["annotations"]) == 2
    occluded_seen = any(
        ann["keypoints"][3 * 1 + 2] == 1.0 for ann in train_coco["annotations"]
    )
    assert occluded_seen is True

    # Round-trip back to LabelMe with write_pose_schema=True
    lm_out = tmp_path / "lm_roundtrip"
    summary = coco2labelme.convert_coco_json_to_labelme_dataset(
        coco_out / "train" / "annotations.json",
        output_dir=lm_out,
        write_pose_schema=True,
    )
    assert summary.get("pose_schema_written") == 1
    extracted = PoseSchema.load(lm_out / "pose_schema.json")
    assert extracted.keypoints == ["nose", "left_ear", "right_ear", "tail_base"]
    assert extracted.edges == [
        ("nose", "left_ear"),
        ("nose", "right_ear"),
        ("nose", "tail_base"),
    ]


# ---------------------------------------------------------------------------
# 3. Canvas skeleton overlay resolution & rendering
# ---------------------------------------------------------------------------


def test_canvas_pose_edges_handles_underscore_kps_same_class_group_ids_and_visibility() -> (
    None
):
    _ensure_qapp()
    canvas = Canvas()
    try:
        canvas.setShowPoseEdges(True)
        schema = PoseSchema(
            keypoints=["nose", "left_ear", "right_ear", "tail_base"],
            edges=[
                ("nose", "left_ear"),
                ("nose", "right_ear"),
                ("nose", "tail_base"),
            ],
        )
        canvas.setPoseSchema(schema)

        # Case A: Single-animal underscore keypoints without group_id or schema.instances
        canvas.shapes = [
            _make_point_shape("nose", 40, 20),
            _make_point_shape(
                "left_ear", 30, 15, visibility=KeypointVisibility.OCCLUDED
            ),
            _make_point_shape(
                "right_ear", 50, 15, visibility=KeypointVisibility.MISSING
            ),
            _make_point_shape("tail_base", 40, 80, selected=True),
        ]
        segs = canvas._collect_pose_edge_segments()
        # nose-left_ear (occluded) and nose-tail_base (highlighted); right_ear is MISSING so skipped!
        assert len(segs) == 2
        occluded_flags = [s[4] for s in segs]
        highlighted_flags = [s[5] for s in segs]
        assert occluded_flags.count(True) == 1
        assert highlighted_flags.count(True) == 1

        # Case B: Two mice with identical instance_label="mouse" differentiated by group_id
        canvas.shapes = [
            _make_point_shape(
                "nose", 10, 10, group_id=1, other_data={"instance_label": "mouse"}
            ),
            _make_point_shape(
                "tail_base", 10, 50, group_id=1, other_data={"instance_label": "mouse"}
            ),
            _make_point_shape(
                "nose", 90, 10, group_id=2, other_data={"instance_label": "mouse"}
            ),
            _make_point_shape(
                "tail_base", 90, 50, group_id=2, other_data={"instance_label": "mouse"}
            ),
        ]
        segs_multi = canvas._collect_pose_edge_segments()
        assert len(segs_multi) == 2
        # Verify bones stay within each animal (dx == 0, never crossing x=10 to x=90!)
        for p1, p2, _bkey, _idx, _occ, _hi in segs_multi:
            assert p1.x() == pytest.approx(p2.x())
        # Distinct instance indices
        assert len({s[3] for s in segs_multi}) == 2

        # Case C: Hierarchical ":" notation
        canvas.shapes = [
            _make_point_shape("mouse_1:nose", 15, 20),
            _make_point_shape("mouse_1:tail_base", 15, 60),
        ]
        segs_hier = canvas._collect_pose_edge_segments()
        assert len(segs_hier) == 1

        # Render onto QImage headlessly to confirm QPainter path has no Qt errors
        img = QtGui.QImage(120, 120, QtGui.QImage.Format_ARGB32)
        img.fill(QtCore.Qt.black)
        painter = QtGui.QPainter(img)
        try:
            canvas._draw_pose_edges(painter)
        finally:
            painter.end()
    finally:
        canvas.close()


def test_canvas_pose_edges_pixel_level_multi_instance_and_dashed_occlusion() -> None:
    _ensure_qapp()
    canvas = Canvas()
    try:
        canvas.setShowPoseEdges(True)
        canvas.scale = 1.0
        schema = PoseSchema(
            keypoints=["nose", "tail_base", "left_ear", "right_ear"],
            edges=[
                ("nose", "tail_base"),
                ("nose", "left_ear"),
                ("nose", "right_ear"),
            ],
        )
        canvas.setPoseSchema(schema)
        canvas.shapes = [
            # Animal 1 (x=40): visible nose-tail_base, occluded nose-left_ear, missing right_ear
            _make_point_shape(
                "nose", 40, 20, group_id=1, other_data={"instance_label": "mouse"}
            ),
            _make_point_shape(
                "tail_base", 40, 100, group_id=1, other_data={"instance_label": "mouse"}
            ),
            _make_point_shape(
                "left_ear",
                10,
                100,
                group_id=1,
                other_data={"instance_label": "mouse"},
                visibility=KeypointVisibility.OCCLUDED,
            ),
            _make_point_shape(
                "right_ear",
                70,
                100,
                group_id=1,
                other_data={"instance_label": "mouse"},
                visibility=KeypointVisibility.MISSING,
            ),
            # Animal 2 (x=160): same class label "mouse", distinct group_id=2
            _make_point_shape(
                "nose", 160, 20, group_id=2, other_data={"instance_label": "mouse"}
            ),
            _make_point_shape(
                "tail_base",
                160,
                100,
                group_id=2,
                other_data={"instance_label": "mouse"},
            ),
        ]

        img = QtGui.QImage(200, 120, QtGui.QImage.Format_ARGB32)
        img.fill(QtCore.Qt.black)
        painter = QtGui.QPainter(img)
        try:
            canvas._draw_pose_edges(painter)
        finally:
            painter.end()

        # Distinct colors for animal 1 (cyan: B > R) vs animal 2 (amber: R > B)
        c1 = img.pixelColor(40, 60)
        c2 = img.pixelColor(160, 60)
        assert c1.blue() > c1.red() + 50
        assert c2.red() > c2.blue() + 50

        # Zero cross-animal pixels in the gap column x=100 between the two mice
        for y in range(10, 110):
            gap_px = img.pixelColor(100, y)
            assert (gap_px.red(), gap_px.green(), gap_px.blue()) == (0, 0, 0)

        # Visible bone (40,20)-(40,100) is solid (100% lit along interior)
        solid_lit = sum(1 for y in range(30, 91) if img.pixelColor(40, y).blue() > 80)
        assert solid_lit == 61

        # Occluded bone (40,20)-(10,100) is dashed (has both lit dash pixels and dark gap pixels)
        occ_samples = []
        for t_idx in range(10, 90):
            t = t_idx / 100.0
            sx = int(round(40 + (10 - 40) * t))
            sy = int(round(20 + (100 - 20) * t))
            occ_samples.append(img.pixelColor(sx, sy).blue() > 30)
        assert any(occ_samples) and not all(occ_samples)

        # Missing right_ear edge (40,20)-(70,100) has zero lit pixels at midpoint (55,60)
        miss_px = img.pixelColor(55, 60)
        assert (miss_px.red(), miss_px.green(), miss_px.blue()) == (0, 0, 0)
    finally:
        canvas.close()


def test_skeleton_preview_theme_and_coco_arm_layout() -> None:
    _ensure_qapp()
    from annolid.gui.widgets.keypoint_sequencer import _SkeletonPreviewWidget

    preview = _SkeletonPreviewWidget()
    try:
        coco = get_pose_schema_preset("Human COCO (17 keypoints)")
        preview.set_schema(coco)
        pts = preview._layout_points(coco.keypoints)
        # Arms flare outward relative to torso so shoulder-hip bones do not pass behind elbows
        assert pts["left_elbow"].x() < pts["left_shoulder"].x()
        assert pts["right_elbow"].x() > pts["right_shoulder"].x()
        assert pts["left_shoulder"].y() == pytest.approx(pts["right_shoulder"].y())

        # Verify theme-aware light vs dark background rendering
        preview.resize(280, 260)
        for bg_hex, expect_bright in (("#f8fafc", True), ("#111827", False)):
            pal = preview.palette()
            pal.setColor(QtGui.QPalette.Window, QtGui.QColor(bg_hex))
            preview.setPalette(pal)
            img = QtGui.QImage(280, 260, QtGui.QImage.Format_ARGB32)
            preview.render(img)
            corner = img.pixelColor(20, 20)
            if expect_bright:
                assert corner.lightness() > 200
            else:
                assert corner.lightness() < 60
    finally:
        preview.close()


# ---------------------------------------------------------------------------
# 4. OpenCV skeleton rendering
# ---------------------------------------------------------------------------


def test_opencv_draw_pose_skeleton_and_float_connections() -> None:
    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    schema = PoseSchema(
        keypoints=["nose", "left_ear", "tail_base"],
        edges=[("nose", "left_ear"), ("nose", "tail_base")],
    )
    draw_pose_skeleton(
        frame,
        {
            "mouse_1": {
                "nose": (30.4, 20.6, 2),
                "left_ear": (20.2, 15.8, 1),  # occluded -> dashed
                "tail_base": (30.0, 90.0, 2),
            },
            "mouse_2": {
                "nose": (90.0, 25.0, 2),
                "tail_base": (90.0, 95.0, 0),  # missing -> skipped
            },
        },
        pose_schema=schema,
    )
    assert int(frame.sum()) > 0

    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)
    draw_keypoint_connections(
        frame2,
        {"nose": (10.5, 20.5), "tail_base": (60.2, 70.8)},
        [("nose", "tail_base", (0, 255, 255))],
    )
    assert int(frame2.sum()) > 0


# ---------------------------------------------------------------------------
# 5. Skeleton preview, KeypointSequencerWidget, and ProjectWizard PoseSchemaPage
# ---------------------------------------------------------------------------


def test_sequencer_presets_edge_combos_preview_layout_and_clean_load(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    widget = KeypointSequencerWidget()
    try:
        assert widget.apply_preset("Mouse (11 keypoints)") is True
        schema = widget.current_schema()
        assert schema is not None
        assert len(schema.keypoints) == 11
        assert len(schema.edges) >= 8

        # Preview aligns left/right symmetry pairs horizontally at same y!
        pts = widget.preview._layout_points(schema.keypoints)
        assert pts["left_ear"].y() == pytest.approx(pts["right_ear"].y())
        assert pts["left_ear"].x() < pts["right_ear"].x()

        # Pending edge start appears in status even with zero locked keypoints!
        widget._on_preview_keypoint_clicked("nose")
        assert "Edge start: nose" in widget.status_label.text()
        assert widget.preview._pending_edge_start == "nose"
        widget._on_preview_keypoint_clicked("nose")  # cancel
        assert widget.preview._pending_edge_start is None

        # Clear edges and auto-infer back
        widget._clear_edges()
        assert widget.current_schema().edges == []
        widget._auto_fill_edges()
        assert len(widget.current_schema().edges) >= 5

        # Combo-based add edge
        widget._clear_edges()
        widget.edge_a_combo.setCurrentText("nose")
        widget.edge_b_combo.setCurrentText("neck")
        widget._add_edge_from_combos()
        assert widget.current_schema().edges == [("nose", "neck")]

        # Loading a distinct schema file from disk replaces stale mouse keypoints
        fly_path = tmp_path / "fly_schema.json"
        get_pose_schema_preset("Fly (5 keypoints)").save(fly_path)
        assert widget.load_schema_from_path(str(fly_path), quiet=True) is True
        assert widget.keypoint_order() == [
            "head",
            "thorax",
            "abdomen",
            "left_wing",
            "right_wing",
        ]
    finally:
        widget.close()


def test_project_wizard_pose_schema_page_preserves_edges_and_presets(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    page = PoseSchemaPage()
    try:
        assert page.pose_container.isHidden() is True
        page.enable_pose_check.setChecked(True)
        # Regression: must show pose_container on PySide6/Qt6 where Qt.Checked is an enum!
        assert page.pose_container.isHidden() is False
        page.preset_combo.setCurrentText("Mouse (11 keypoints)")
        schema = page.get_pose_schema()
        assert len(schema.keypoints) == 11
        assert len(schema.edges) >= 8
        assert ("left_ear", "right_ear") in schema.symmetry_pairs

        page.enable_pose_check.setChecked(False)
        assert page.pose_container.isHidden() is True
        page.enable_pose_check.setChecked(True)
        assert page.pose_container.isHidden() is False

        # Round-trip save/load via _load_schema_into_ui preserves edges & instances
        custom = PoseSchema(
            keypoints=["nose", "neck", "tail_base"],
            edges=[("nose", "neck"), ("neck", "tail_base")],
            instances=["resident", "intruder"],
            instance_separator=":",
        )
        page._load_schema_into_ui(custom)
        got = page.get_pose_schema()
        assert got.edges == [("nose", "neck"), ("neck", "tail_base")]
        assert got.instances == ["resident", "intruder"]
        assert got.instance_separator == ":"

        # Preview click toggles edge cleanly
        page._on_preview_keypoint_clicked("nose")
        page._on_preview_keypoint_clicked("tail_base")
        assert ("nose", "tail_base") in page.get_pose_schema().edges
    finally:
        page.close()


# ---------------------------------------------------------------------------
# 6. GUI auto-discovery and dialog wiring
# ---------------------------------------------------------------------------


def test_gui_pose_schema_loads_from_project_path_and_dlc_config(
    tmp_path: Path,
) -> None:
    _ensure_qapp()

    class _DummyWindow(SchemaBehaviorLoaderMixin, ToolingDialogsMixin):
        def __init__(self) -> None:
            self.project_schema = None
            self.project_schema_path = None
            self._pose_schema = None
            self._pose_schema_path = None
            self.video_results_folder = None
            self.annotation_dir = None
            self.video_file = None
            self.outputDir = None
            self.lastOpenDir = None
            self.canvas = Canvas()
            self.keypoint_sequence_widget = KeypointSequencerWidget()
            self.keypoint_sequence_dock = None
            self.uniqLabelList = QtWidgets.QListWidget()

    win = _DummyWindow()
    try:
        # 1. ProjectSchema with pose_schema_path ONLY (as produced by ProjectWizard!)
        proj_dir = tmp_path / "proj"
        proj_dir.mkdir()
        get_pose_schema_preset("Fly (5 keypoints)").save(proj_dir / "pose_schema.json")
        win.project_schema_path = proj_dir / "project.annolid.json"
        win.project_schema = ProjectSchema(pose_schema_path="pose_schema.json")
        win._configure_pose_schema_from_project()
        assert win._pose_schema is not None
        assert ("head", "thorax") in win._pose_schema.edges
        assert win.canvas._pose_schema is win._pose_schema

        # 2. open_pose_schema_dialog loads disk pose_schema BEFORE label fallback (no polygon pollution!)
        win._pose_schema = None
        win._pose_schema_path = None
        win.project_schema = None
        win.video_results_folder = proj_dir
        win.uniqLabelList.addItem("arena_polygon")
        win.open_pose_schema_dialog()
        assert "arena_polygon" not in win.keypoint_sequence_widget.keypoint_order()
        assert win.keypoint_sequence_widget.keypoint_order() == [
            "head",
            "thorax",
            "abdomen",
            "left_wing",
            "right_wing",
        ]

        # 3. DLC CSV overlay auto-loads sibling config.yaml skeleton
        dlc_dir = tmp_path / "dlc_run"
        dlc_dir.mkdir()
        (dlc_dir / "config.yaml").write_text(
            yaml.safe_dump(
                {
                    "scorer": "dlc",
                    "bodyparts": ["snout", "ear_l", "ear_r"],
                    "skeleton": [["snout", "ear_l"], ["snout", "ear_r"]],
                }
            ),
            encoding="utf-8",
        )
        csv_path = dlc_dir / "vidDLC_net.csv"
        csv_path.write_text(
            "\n".join(
                [
                    "scorer,dlc,dlc,dlc,dlc,dlc,dlc",
                    "bodyparts,snout,snout,snout,ear_l,ear_l,ear_l",
                    "coords,x,y,likelihood,x,y,likelihood",
                    "0,10,20,0.9,15,25,0.9",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        win._pose_schema = None
        win._pose_schema_path = None
        assert win._load_deeplabcut_table(str(csv_path)) is True
        assert win._pose_schema is not None
        assert ("snout", "ear_l") in win._pose_schema.edges

        # 4. Dialog smoke tests for new pose_schema options
        coco_dlg = ConvertCOODialog(annotation_dir=str(proj_dir))
        assert coco_dlg.pose_schema_edit.text().endswith("pose_schema.json")
        coco_dlg.close()

        c2l_dlg = ConvertCOCO2LabelMeDialog()
        assert c2l_dlg.write_pose_schema_check.isChecked() is True
        c2l_dlg.close()
    finally:
        win.canvas.close()
        win.keypoint_sequence_widget.close()
        win.uniqLabelList.close()


# ---------------------------------------------------------------------------
# 7. Track QC, Body-Part Service, and Morphological Skeletonization
# ---------------------------------------------------------------------------


def test_track_qc_handles_prefixed_edges_without_duplicates(tmp_path: Path) -> None:
    ann_dir = tmp_path / "qc_prefixed"
    ann_dir.mkdir()

    # PoseSchema with prefixed edges AND reversed duplicate!
    pose = PoseSchema(
        keypoints=["nose", "tail_base"],
        edges=[
            ("mouse_1_nose", "mouse_1_tail_base"),
            ("mouse_1_tail_base", "mouse_1_nose"),
        ],
        instances=["mouse_1"],
    )
    for f in range(6):
        tail_x = 180.0 if f == 3 else 40.0  # huge stretch on frame 3
        payload = {
            "version": "5.0.1",
            "imagePath": f"f_{f:06d}.png",
            "imageData": None,
            "shapes": [
                {
                    "label": "mouse_1_nose",
                    "shape_type": "point",
                    "points": [[20.0, 20.0]],
                    "flags": {},
                },
                {
                    "label": "mouse_1_tail_base",
                    "shape_type": "point",
                    "points": [[tail_x, 20.0]],
                    "flags": {},
                },
            ],
        }
        (ann_dir / f"f_{f:06d}.json").write_text(json.dumps(payload), encoding="utf-8")

    report = evaluate_track_qc(ann_dir, pose_schema=pose)
    edge_issues = [i for i in report.issues if i.code == "anatomical_edge_outlier"]
    assert len(edge_issues) == 1
    assert edge_issues[0].frame_start == 3


def test_body_part_service_auto_discovers_and_strips_prefixed_edges(
    tmp_path: Path,
) -> None:
    schema = PoseSchema(
        keypoints=["nose", "neck"],
        edges=[("mouse_1_nose", "mouse_1_neck")],
        instances=["mouse_1"],
    )
    schema.save(tmp_path / "pose_schema.json")
    edges = _load_skeleton_edges(None, annotation_dir=tmp_path)
    assert edges == (("nose", "neck"),)


def test_skeletonization_pure_cv2_fallback_and_dedup_labelme_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Block BOTH skimage AND cv2.ximgproc!
    monkeypatch.setattr(skel_mod, "_skimage_skeletonize", None)
    monkeypatch.setattr(skel_mod, "_skimage_medial_axis", None)
    monkeypatch.setattr(cv2, "ximgproc", None, raising=False)

    # Empty mask safety
    empty_sp = skel_mod.ShapeProcessor(np.zeros((20, 20), dtype=bool))
    assert empty_sp.labeled_points == {}
    assert empty_sp.local_thickness((999, 999)) == 0.0
    center, axis = empty_sp.compute_symmetry_axis()
    assert center.shape == (2,) and axis.shape == (2,)

    # Rectangular mask produces non-empty skeleton + fast boundary distance
    mask = np.zeros((32, 48), dtype=bool)
    mask[8:24, 6:42] = True
    sp = skel_mod.ShapeProcessor(mask)
    assert int(sp.skeleton.sum()) > 0
    assert sp.distance_to_polygon_edge((24.0, 16.0)) > 0.0
    assert "head" in sp.labeled_points and "tail" in sp.labeled_points

    # LabelMe JSON output is LabelMe-safe, idempotent on re-run, and skips pose_schema.json
    (tmp_path / "pose_schema.json").write_text(
        '{"keypoints":["nose"]}', encoding="utf-8"
    )
    frame_json = tmp_path / "frame_00000001.json"
    frame_json.write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "flags": {},
                "shapes": [
                    {
                        "label": "mouse_1",
                        "points": [[6.0, 8.0], [41.0, 8.0], [41.0, 23.0], [6.0, 23.0]],
                        "group_id": None,
                        "shape_type": "polygon",
                        "flags": {},
                    }
                ],
                "imagePath": "frame_00000001.png",
                "imageData": None,
                "imageHeight": 32,
                "imageWidth": 48,
            }
        ),
        encoding="utf-8",
    )

    summary1 = skel_mod.main(str(tmp_path), ["mouse_1"])
    assert summary1["processed_files"] == 1
    assert summary1["points_written"] == 5

    summary2 = skel_mod.main(str(tmp_path), ["mouse_1"])
    assert summary2["processed_files"] == 1
    data = json.loads(frame_json.read_text(encoding="utf-8"))
    point_shapes = [s for s in data["shapes"] if s["shape_type"] == "point"]
    # No duplicate points after re-running!
    assert len(point_shapes) == 5
    for pt in point_shapes:
        assert isinstance(pt["flags"].get("kp_visible"), bool)
        assert pt["instance_label"] == "mouse_1"
