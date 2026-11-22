from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import yaml

from annolid.annotation.coco2deeplabcut import convert_coco_to_dlc
from annolid.annotation.deeplabcut2labelme import deeplabcut_to_labelme_json
from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    visibility_from_labelme_shape,
)
from annolid.annotation.pose_interop import (
    PoseDataset,
    PoseFrame,
    PoseInstance,
    PosePoint,
    detect_pose_source_format,
    export_pose_dataset,
    find_dlc_analysis_file_for_video,
    parse_dlc_table,
    read_pose_dataset,
    write_sleap_analysis_h5,
    write_sleap_slp,
)
from annolid.annotation.pose_schema import PoseSchema
from annolid.annotation.sleap2labelme import (
    SleapH5Reader,
    convert_sleap_h5_to_labelme,
    convert_sleap_h5_to_labelme_summary,
)
from annolid.core.agent.tools.dataset import (
    AnnolidDatasetInspectTool,
    AnnolidDatasetPrepareTool,
)
from annolid.datasets.importers.deeplabcut_training_data import (
    DeepLabCutTrainingImportConfig,
    import_deeplabcut_training_data,
)
from annolid.engine.cli import main as cli_main
from annolid.utils.annotation_store import AnnotationStore


def _write_pandas_fixed_dlc_h5(
    h5_path: Path,
    *,
    scorer: str,
    individuals: list[str] | None,
    bodyparts: list[str],
    coords: list[str],
    index_values: list[object],
    values: np.ndarray,
    key: str = "df_with_missing",
) -> None:
    """Create a synthetic Pandas fixed-format HDF5 file matching DeepLabCut's layout."""
    h5py = pytest.importorskip("h5py")
    str_dt = h5py.string_dtype(encoding="utf-8")

    with h5py.File(h5_path, "w") as h5:
        grp = h5.create_group(key)
        col_tuples: list[tuple[str, ...]] = []
        if individuals is not None:
            for ind in individuals:
                for bp in bodyparts:
                    for c in coords:
                        col_tuples.append((scorer, ind, bp, c))
            level_names = ["scorer", "individuals", "bodyparts", "coords"]
            nlevels = 4
        else:
            for bp in bodyparts:
                for c in coords:
                    col_tuples.append((scorer, bp, c))
            level_names = ["scorer", "bodyparts", "coords"]
            nlevels = 3

        grp.attrs["axis0_nlevels"] = nlevels
        for lvl in range(nlevels):
            uniq: list[str] = []
            codes: list[int] = []
            for tup in col_tuples:
                val = tup[lvl]
                if val not in uniq:
                    uniq.append(val)
                codes.append(uniq.index(val))
            ds_lvl = grp.create_dataset(
                f"axis0_level{lvl}", data=np.array(uniq, dtype=object), dtype=str_dt
            )
            ds_lvl.attrs["name"] = level_names[lvl]
            grp.create_dataset(
                f"axis0_label{lvl}", data=np.array(codes, dtype=np.int64)
            )

        if all(isinstance(v, int) for v in index_values):
            grp.create_dataset("axis1", data=np.array(index_values, dtype=np.int64))
        else:
            grp.create_dataset(
                "axis1",
                data=np.array([str(v) for v in index_values], dtype=object),
                dtype=str_dt,
            )

        grp.attrs["nblocks"] = 1
        grp.create_dataset(
            "block0_items", data=np.arange(len(col_tuples), dtype=np.int64)
        )
        grp.create_dataset("block0_values", data=np.asarray(values, dtype=np.float64))


def _assert_boolean_only_flags(shapes: list[dict]) -> None:
    for shape in shapes:
        flags = shape.get("flags")
        assert isinstance(flags, dict)
        for k, v in flags.items():
            assert isinstance(v, bool), f"Non-boolean flag {k}={v!r} breaks LabelMe"


def test_dlc_multi_animal_csv_import_preserves_identities_scores_and_boolean_flags(
    tmp_path: Path,
) -> None:
    csv_path = tmp_path / "trialDLC_dlcrnet_ms5_miceShuffle1_50000_el_filtered.csv"
    csv_path.write_text(
        "\n".join(
            [
                "scorer,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet,DLC_dlcrnet",
                "individuals,mouse1,mouse1,mouse1,mouse1,mouse1,mouse1,mouse2,mouse2,mouse2,mouse2,mouse2,mouse2",
                "bodyparts,nose,nose,nose,tail_base,tail_base,tail_base,nose,nose,nose,tail_base,tail_base,tail_base",
                "coords,x,y,likelihood,x,y,likelihood,x,y,likelihood,x,y,likelihood",
                "0,10.5,20.5,0.95,30.0,40.0,0.85,100.0,110.0,0.90,120.0,130.0,0.20",
                "1,12.0,22.0,0.99,,,102.0,112.0,0.88,125.0,135.0,0.75",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    # Write a matching DLC config.yaml with custom skeleton edges
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "scorer": "DLC_dlcrnet",
                "multianimalproject": True,
                "individuals": ["mouse1", "mouse2"],
                "multianimalbodyparts": ["nose", "tail_base"],
                "skeleton": [["nose", "tail_base"]],
            }
        ),
        encoding="utf-8",
    )

    out_dir = tmp_path / "out_labelme"
    summary = deeplabcut_to_labelme_json(
        csv_path=csv_path,
        output_dir=out_dir,
        min_likelihood=0.5,
    )

    assert summary["multi_animal"] is True
    assert summary["json_written"] == 2
    assert summary["instances"] == ["mouse1", "mouse2"]
    assert summary["keypoints"] == ["nose", "tail_base"]

    f0 = json.loads((out_dir / "000000000.json").read_text(encoding="utf-8"))
    _assert_boolean_only_flags(f0["shapes"])
    # mouse2 tail_base has likelihood 0.20 < 0.5 -> filtered out -> 3 points remain
    assert len(f0["shapes"]) == 3
    by_pair = {(s["instance_label"], s["display_label"]): s for s in f0["shapes"]}
    assert ("mouse1", "nose") in by_pair
    assert ("mouse1", "tail_base") in by_pair
    assert ("mouse2", "nose") in by_pair
    assert ("mouse2", "tail_base") not in by_pair
    assert by_pair[("mouse1", "nose")]["group_id"] == 0
    assert by_pair[("mouse2", "nose")]["group_id"] == 1
    assert by_pair[("mouse1", "nose")]["score"] == pytest.approx(0.95)

    schema = PoseSchema.load(out_dir / "pose_schema.json")
    assert schema.keypoints == ["nose", "tail_base"]
    assert schema.edges == [("nose", "tail_base")]
    assert schema.instances == ["mouse1", "mouse2"]


def test_dlc_auto_discovery_and_missing_file_raises(tmp_path: Path) -> None:
    video = tmp_path / "session01.mp4"
    video.write_bytes(b"")

    with pytest.raises(FileNotFoundError):
        deeplabcut_to_labelme_json(video_path=video)

    raw_csv = tmp_path / "session01DLC_resnet50_projShuffle1_10000.csv"
    filtered_csv = tmp_path / "session01DLC_resnet50_projShuffle1_10000_filtered.csv"
    header = (
        "scorer,s,s,s\nbodyparts,nose,nose,nose\ncoords,x,y,likelihood\n0,1.0,2.0,0.9\n"
    )
    raw_csv.write_text(header, encoding="utf-8")
    filtered_csv.write_text(header, encoding="utf-8")

    found = find_dlc_analysis_file_for_video(video)
    assert found == filtered_csv


def test_dlc_h5_pure_h5py_reader_single_and_multi_animal(tmp_path: Path) -> None:
    h5_path = tmp_path / "trialDLC_dlcrnet.h5"
    vals = np.array(
        [
            [10.0, 20.0, 0.9, 50.0, 60.0, 0.8],
            [11.0, 21.0, 0.95, np.nan, np.nan, 0.1],
        ],
        dtype=np.float64,
    )
    _write_pandas_fixed_dlc_h5(
        h5_path,
        scorer="DLC_dlcrnet",
        individuals=["m1", "m2"],
        bodyparts=["nose"],
        coords=["x", "y", "likelihood"],
        index_values=[0, 1],
        values=vals,
    )

    assert detect_pose_source_format(h5_path) == "deeplabcut_h5"
    table = parse_dlc_table(h5_path)
    assert table.multi_animal is True
    assert table.individuals == ("m1", "m2")
    assert table.bodyparts == ("nose",)

    ds = read_pose_dataset(h5_path)
    assert len(ds.frames) == 2
    assert ds.frames[0].instances[0].keypoints["nose"].x == pytest.approx(10.0)
    assert ds.frames[0].instances[1].keypoints["nose"].y == pytest.approx(60.0)
    assert len(ds.frames[1].instances) == 1


def test_dlc_training_import_windows_paths_config_skeleton_multi_animal_and_h5(
    tmp_path: Path,
) -> None:
    proj = tmp_path / "dlc_proj"
    seg = proj / "labeled-data" / "vid1"
    seg.mkdir(parents=True)
    img1 = seg / "img001.png"
    img2 = seg / "img002.png"
    Image.new("RGB", (64, 48), color=(10, 20, 30)).save(img1)
    Image.new("RGB", (64, 48), color=(30, 20, 10)).save(img2)

    (proj / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "scorer": "alice",
                "multianimalproject": True,
                "individuals": ["mouseA", "mouseB"],
                "multianimalbodyparts": ["snout", "ear_l", "ear_r"],
                "skeleton": [["snout", "ear_l"], ["snout", "ear_r"]],
            }
        ),
        encoding="utf-8",
    )

    # Multi-animal CSV with Windows backslash paths
    (seg / "CollectedData_alice.csv").write_text(
        "\n".join(
            [
                "scorer,alice,alice,alice,alice,alice,alice,alice,alice",
                "individuals,mouseA,mouseA,mouseA,mouseA,mouseB,mouseB,mouseB,mouseB",
                "bodyparts,snout,snout,ear_l,ear_l,snout,snout,ear_l,ear_l",
                "coords,x,y,x,y,x,y,x,y",
                r"labeled-data\vid1\img001.png,10.0,12.0,14.0,16.0,30.0,32.0,34.0,36.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    # Also write an H5-only CollectedData file for img002
    _write_pandas_fixed_dlc_h5(
        seg / "CollectedData_bob.h5",
        scorer="bob",
        individuals=["mouseA", "mouseB"],
        bodyparts=["snout"],
        coords=["x", "y"],
        index_values=["labeled-data/vid1/img002.png"],
        values=np.array([[20.0, 22.0, 40.0, 42.0]], dtype=np.float64),
    )

    summary = import_deeplabcut_training_data(
        DeepLabCutTrainingImportConfig(source_dir=proj, overwrite=True),
        write_pose_schema=True,
    )
    assert summary["json_written"] == 2
    assert summary["multi_instance"] is True
    assert summary["errors"] == 0
    assert summary["config_yaml"] == str((proj / "config.yaml").resolve())

    j1 = json.loads(img1.with_suffix(".json").read_text(encoding="utf-8"))
    _assert_boolean_only_flags(j1["shapes"])
    gids = {s["instance_label"]: s["group_id"] for s in j1["shapes"]}
    assert gids == {"mouseA": 0, "mouseB": 1}

    schema = PoseSchema.load(proj / "labeled-data" / "pose_schema.json")
    assert ("snout", "ear_l") in schema.edges
    assert ("snout", "ear_r") in schema.edges
    assert img2.with_suffix(".json").exists()


def test_dlc_analysis_and_training_project_round_trip(tmp_path: Path) -> None:
    lm_dir = tmp_path / "annolid_labels"
    lm_dir.mkdir()
    img0 = lm_dir / "clip_000000000.png"
    img1 = lm_dir / "clip_000000001.png"
    Image.new("RGB", (80, 60), color=(0, 0, 0)).save(img0)
    Image.new("RGB", (80, 60), color=(255, 255, 255)).save(img1)

    PoseSchema(
        keypoints=["nose", "left_ear", "right_ear", "tail_base"],
        edges=[("nose", "left_ear"), ("nose", "right_ear"), ("nose", "tail_base")],
        symmetry_pairs=[("left_ear", "right_ear")],
        instances=["mouse_1", "mouse_2"],
    ).save(lm_dir / "pose_schema.json")

    for fidx, img in ((0, img0), (1, img1)):
        payload = {
            "version": "5.5.0",
            "flags": {},
            "imagePath": img.name,
            "imageData": None,
            "imageWidth": 80,
            "imageHeight": 60,
            "shapes": [
                {
                    "label": "mouse_1_nose",
                    "instance_label": "mouse_1",
                    "display_label": "nose",
                    "group_id": 0,
                    "points": [[10.0 + fidx, 15.0 + fidx]],
                    "shape_type": "point",
                    "flags": {"kp_visible": True},
                    "kp_visibility": 2,
                    "score": 0.91,
                },
                {
                    "label": "mouse_2_nose",
                    "instance_label": "mouse_2",
                    "display_label": "nose",
                    "group_id": 1,
                    "points": [[50.0 + fidx, 25.0 + fidx]],
                    "shape_type": "point",
                    "flags": {"kp_visible": True},
                    "kp_visibility": 2,
                    "score": 0.87,
                },
                {
                    "label": "mouse_2_tail_base",
                    "instance_label": "mouse_2",
                    "display_label": "tail_base",
                    "group_id": 1,
                    "points": [[60.0 + fidx, 35.0 + fidx]],
                    "shape_type": "point",
                    "flags": {"kp_visible": True},
                    "kp_visibility": 2,
                    "score": 0.83,
                },
            ],
        }
        img.with_suffix(".json").write_text(json.dumps(payload), encoding="utf-8")

    # 1. Export -> DLC Analysis CSV -> Re-read
    dlc_csv = tmp_path / "exported_dlc.csv"
    exp_summary = export_pose_dataset(
        lm_dir, dlc_csv, target_format="deeplabcut-csv", scorer="lab_scorer"
    )
    assert exp_summary["multi_animal"] is True
    assert exp_summary["frames_exported"] == 2

    re_read = read_pose_dataset(dlc_csv)
    assert re_read.multi_animal is True
    assert len(re_read.frames) == 2
    f1_by_inst = {inst.instance_name: inst for inst in re_read.frames[1].instances}
    assert f1_by_inst["mouse_1"].keypoints["nose"].x == pytest.approx(11.0)
    assert f1_by_inst["mouse_2"].keypoints["tail_base"].y == pytest.approx(36.0)
    assert f1_by_inst["mouse_2"].keypoints["nose"].score == pytest.approx(0.87)

    # 2. Export -> DLC Training Project -> Re-import with import_deeplabcut_training_data
    dlc_proj = tmp_path / "dlc_project"
    proj_summary = export_pose_dataset(
        lm_dir,
        dlc_proj,
        target_format="deeplabcut-project",
        scorer="lab_scorer",
        video_name="clip",
    )
    assert Path(proj_summary["collected_data_csv"]).is_file()
    assert Path(proj_summary["config_yaml"]).is_file()
    assert proj_summary["images_copied"] == 2

    imp_summary = import_deeplabcut_training_data(
        DeepLabCutTrainingImportConfig(source_dir=dlc_proj, overwrite=True),
        write_pose_schema=True,
    )
    assert imp_summary["json_written"] == 2
    assert imp_summary["multi_instance"] is True
    re_schema = PoseSchema.load(dlc_proj / "labeled-data" / "pose_schema.json")
    assert ("nose", "tail_base") in re_schema.edges


def test_coco_to_dlc_writes_real_dlc_csv_and_preserves_legacy_json(
    tmp_path: Path,
) -> None:
    coco_path = tmp_path / "coco_kp.json"
    coco_payload = {
        "images": [
            {"id": 0, "file_name": "frame_000000.png", "width": 100, "height": 80},
            {"id": 1, "file_name": "frame_000001.png", "width": 100, "height": 80},
        ],
        "categories": [
            {
                "id": 1,
                "name": "mouse",
                "keypoints": ["nose", "tail"],
                "skeleton": [[1, 2]],
            }
        ],
        "annotations": [
            {
                "id": 10,
                "image_id": 0,
                "category_id": 1,
                "bbox": [10, 10, 40, 20],
                "keypoints": [15.0, 20.0, 2, 45.0, 22.0, 2],
                "score": 0.92,
            },
            {
                "id": 11,
                "image_id": 1,
                "category_id": 1,
                "bbox": [12, 12, 40, 20],
                "keypoints": [17.0, 21.0, 2, 0.0, 0.0, 0],
                "score": 0.88,
            },
        ],
    }
    coco_path.write_text(json.dumps(coco_payload), encoding="utf-8")

    out_csv = tmp_path / "coco_out.csv"
    convert_coco_to_dlc(coco_path, out_csv, scorer="coco_export")
    assert detect_pose_source_format(out_csv) == "deeplabcut_csv"

    ds = read_pose_dataset(out_csv)
    assert ds.keypoints == ["nose", "tail"]
    assert ds.frames[0].instances[0].keypoints["nose"].x == pytest.approx(15.0)
    assert "tail" not in ds.frames[1].instances[0].keypoints

    # Legacy JSON extension remains supported
    out_json = tmp_path / "legacy_out.json"
    convert_coco_to_dlc(coco_path, out_json)
    legacy = json.loads(out_json.read_text(encoding="utf-8"))
    assert isinstance(legacy, list) and len(legacy) == 2
    assert legacy[0]["x"] == pytest.approx(30.0)


def test_sleap_analysis_h5_and_slp_round_trips_and_feature_fixes(
    tmp_path: Path,
) -> None:
    pytest.importorskip("h5py")

    # Build a rich PoseDataset with occluded keypoint, scores, symmetry, edges, and embedded images
    img_dir = tmp_path / "imgs"
    img_dir.mkdir()
    img0 = img_dir / "000000000.png"
    img1 = img_dir / "000000002.png"
    Image.new("RGB", (64, 48), color=(10, 30, 50)).save(img0)
    Image.new("RGB", (64, 48), color=(50, 30, 10)).save(img1)

    dataset = PoseDataset(
        frames=[
            PoseFrame(
                frame_index=0,
                image_path=img0.name,
                source_image_path=str(img0),
                image_width=64,
                image_height=48,
                instances=[
                    PoseInstance(
                        instance_name="mouse_black",
                        track_id=0,
                        score=0.95,
                        is_predicted=False,
                        keypoints={
                            "nose": PosePoint(10.0, 20.0, score=0.99, visibility=2),
                            "left_ear": PosePoint(14.0, 18.0, score=0.92, visibility=1),
                            "right_ear": PosePoint(
                                14.0, 22.0, score=0.94, visibility=2
                            ),
                        },
                    ),
                    PoseInstance(
                        instance_name="mouse_white",
                        track_id=1,
                        score=0.88,
                        is_predicted=False,
                        keypoints={
                            "nose": PosePoint(40.0, 25.0, score=0.91, visibility=2),
                            "right_ear": PosePoint(
                                44.0, 27.0, score=0.85, visibility=2
                            ),
                        },
                    ),
                ],
            ),
            PoseFrame(
                frame_index=2,
                image_path=img1.name,
                source_image_path=str(img1),
                image_width=64,
                image_height=48,
                instances=[
                    PoseInstance(
                        instance_name="mouse_black",
                        track_id=0,
                        score=0.90,
                        is_predicted=True,
                        keypoints={
                            "nose": PosePoint(12.0, 21.0, score=0.90, visibility=2),
                        },
                    )
                ],
            ),
        ],
        keypoints=["nose", "left_ear", "right_ear"],
        edges=[("nose", "left_ear"), ("nose", "right_ear")],
        symmetry_pairs=[("left_ear", "right_ear")],
        instances=["mouse_black", "mouse_white"],
        multi_animal=True,
    )

    # 1. SLEAP Analysis HDF5 export + import round-trip
    analysis_h5 = tmp_path / "tracking.analysis.h5"
    write_sleap_analysis_h5(dataset, analysis_h5, video_path="trial.mp4")
    assert detect_pose_source_format(analysis_h5) == "sleap_analysis_h5"

    out_h5_lm = tmp_path / "from_analysis_h5"
    h5_summary = convert_sleap_h5_to_labelme_summary(
        analysis_h5, out_h5_lm, save_frames=False
    )
    assert h5_summary["sleap_kind"] == "analysis_h5"
    assert h5_summary["json_written"] == 2
    assert h5_summary["instances"] == ["mouse_black", "mouse_white"]
    h5_f0 = json.loads((out_h5_lm / "000000000.json").read_text(encoding="utf-8"))
    _assert_boolean_only_flags(h5_f0["shapes"])
    assert len(h5_f0["shapes"]) == 5
    h5_schema = PoseSchema.load(out_h5_lm / "pose_schema.json")
    assert ("nose", "left_ear") in h5_schema.edges
    assert ("nose", "right_ear") in h5_schema.edges

    # 2. SLEAP .pkg.slp export (with embedded images, occluded point, track names, symmetry) + re-import
    pkg_slp = tmp_path / "session.pkg.slp"
    write_sleap_slp(dataset, pkg_slp, video_path="trial.mp4", embed_images=True)
    assert detect_pose_source_format(pkg_slp) == "sleap_slp"

    out_slp_lm = tmp_path / "from_pkg_slp"
    returned_dir = convert_sleap_h5_to_labelme(pkg_slp, out_slp_lm, save_frames=True)
    assert returned_dir == out_slp_lm.resolve()
    assert (out_slp_lm / "000000000.png").is_file()
    assert (out_slp_lm / "000000002.png").is_file()

    slp_f0 = json.loads((out_slp_lm / "000000000.json").read_text(encoding="utf-8"))
    _assert_boolean_only_flags(slp_f0["shapes"])
    by_key = {(s["instance_label"], s["display_label"]): s for s in slp_f0["shapes"]}
    # Occluded point (left_ear, visibility=1) is preserved instead of silently dropped!
    assert ("mouse_black", "left_ear") in by_key
    assert visibility_from_labelme_shape(by_key[("mouse_black", "left_ear")]) == int(
        KeypointVisibility.OCCLUDED
    )
    assert by_key[("mouse_black", "left_ear")]["flags"]["kp_visible"] is False
    assert by_key[("mouse_white", "nose")]["group_id"] == 1

    slp_f2 = json.loads((out_slp_lm / "000000002.json").read_text(encoding="utf-8"))
    assert slp_f2["shapes"][0]["flags"]["predicted"] is True
    assert slp_f2["shapes"][0]["score"] == pytest.approx(0.90)

    slp_schema = PoseSchema.load(out_slp_lm / "pose_schema.json")
    assert slp_schema.symmetry_pairs == [("left_ear", "right_ear")]
    assert slp_schema.edges == [("nose", "left_ear"), ("nose", "right_ear")]

    # 3. Non-packaged .slp without embedded frames degrades gracefully when save_frames=True
    plain_slp = tmp_path / "external_only.slp"
    write_sleap_slp(
        dataset,
        plain_slp,
        video_path="/nonexistent/my_experiment.mp4",
        embed_images=False,
    )
    with SleapH5Reader(plain_slp) as reader:
        assert reader.video_name(0) == "my_experiment"
    out_plain = tmp_path / "from_plain_slp"
    plain_summary = convert_sleap_h5_to_labelme_summary(
        plain_slp, out_plain, save_frames=True
    )
    assert plain_summary["json_written"] == 2
    assert plain_summary["warnings"]


def test_sleap_prefer_user_instances_avoids_duplicating_predictions(
    tmp_path: Path,
) -> None:
    h5py = pytest.importorskip("h5py")
    slp_path = tmp_path / "user_and_pred.slp"
    with h5py.File(slp_path, "w") as f:
        meta = {
            "nodes": [{"name": "nose"}],
            "skeletons": [{"nodes": [{"id": 0}], "links": [], "graph": {"name": "s"}}],
        }
        f.create_dataset("metadata", data=json.dumps(meta).encode("utf-8"))
        f.create_dataset(
            "frames",
            data=np.array(
                [(0, 0, 0, 0, 2)],
                dtype=np.dtype(
                    [
                        ("frame_id", "<u8"),
                        ("video", "<u4"),
                        ("frame_idx", "<u8"),
                        ("instance_id_start", "<u8"),
                        ("instance_id_end", "<u8"),
                    ]
                ),
            ),
        )
        f.create_dataset(
            "instances",
            data=np.array(
                [(0, 0, 0, 0, 0, 0, 1), (1, 1, 0, 0, 0, 0, 1)],
                dtype=np.dtype(
                    [
                        ("instance_id", "<i8"),
                        ("instance_type", "u1"),
                        ("frame_id", "<u8"),
                        ("skeleton", "<u4"),
                        ("track", "<i4"),
                        ("point_id_start", "<u8"),
                        ("point_id_end", "<u8"),
                    ]
                ),
            ),
        )
        f.create_dataset(
            "points",
            data=np.array(
                [(10.0, 20.0, True)],
                dtype=np.dtype([("x", "<f8"), ("y", "<f8"), ("visible", "?")]),
            ),
        )
        f.create_dataset(
            "pred_points",
            data=np.array(
                [(99.0, 88.0, True, 0.7)],
                dtype=np.dtype(
                    [("x", "<f8"), ("y", "<f8"), ("visible", "?"), ("score", "<f8")]
                ),
            ),
        )

    ds_pref = read_pose_dataset(slp_path, prefer_user_instances=True)
    assert len(ds_pref.frames[0].instances) == 1
    assert ds_pref.frames[0].instances[0].keypoints["nose"].x == pytest.approx(10.0)

    ds_both = read_pose_dataset(slp_path, prefer_user_instances=False)
    assert len(ds_both.frames[0].instances) == 2


def test_sleap_legacy_jsonpickle_py_id_skeleton_metadata() -> None:
    from annolid.annotation.pose_interop.sleap import _parse_sleap_metadata_skeletons

    # In legacy jsonpickle SLEAP metadata, py/id is a 1-based object ID across all
    # py/object instances in the JSON tree (NOT a 0-based index into meta["nodes"]).
    meta = {
        "provenance": {"py/object": "sleap.io.Provenance", "info": "legacy"},
        "nodes": [
            {"py/object": "sleap.skeleton.Node", "name": "head"},
            {"py/object": "sleap.skeleton.Node", "name": "thorax"},
            {"py/object": "sleap.skeleton.Node", "name": "abdomen"},
        ],
        "skeletons": [
            {
                "graph": {"name": "fly"},
                "nodes": [{"py/id": 2}, {"py/id": 3}, {"py/id": 4}],
                "links": [
                    {"source": {"py/id": 2}, "target": {"py/id": 3}, "type": 1},
                    {"source": {"py/id": 3}, "target": {"py/id": 4}, "type": 1},
                ],
            }
        ],
    }
    skels = _parse_sleap_metadata_skeletons(meta)
    assert skels is not None and len(skels) == 1
    assert skels[0].name == "fly"
    assert skels[0].names == ["head", "thorax", "abdomen"]
    assert skels[0].edges == [("head", "thorax"), ("thorax", "abdomen")]


def test_export_from_annotation_store_and_tracking_csv(tmp_path: Path) -> None:
    pytest.importorskip("h5py")
    # 1. AnnotationStore export
    vid_dir = tmp_path / "video_run"
    vid_dir.mkdir()
    store = AnnotationStore(vid_dir / "video_run_annotations.ndjson")
    store.append_frame(
        {
            "frame": 0,
            "imageWidth": 100,
            "imageHeight": 80,
            "shapes": [
                {
                    "label": "nose",
                    "instance_label": "mouse_1",
                    "display_label": "nose",
                    "group_id": 0,
                    "shape_type": "point",
                    "points": [[12.0, 24.0]],
                    "flags": {"kp_visible": True},
                }
            ],
        },
    )
    out_dlc = tmp_path / "store_dlc.csv"
    res = export_pose_dataset(vid_dir, out_dlc, target_format="deeplabcut-csv")
    assert res["frames_exported"] == 1
    re_ds = read_pose_dataset(out_dlc)
    assert re_ds.frames[0].instances[0].keypoints["nose"].x == pytest.approx(12.0)

    # 2. Annolid tracking CSV export -> SLEAP Analysis H5
    tracking_csv = tmp_path / "session_tracking.csv"
    tracking_csv.write_text(
        "\n".join(
            [
                "frame_number,x1,y1,x2,y2,cx,cy,instance_name,class_score,segmentation,tracking_id",
                "0,10,20,10,20,10.0,20.0,mouse_1:nose,0.95,,0",
                "0,30,40,30,40,30.0,40.0,mouse_2:nose,0.91,,1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    out_h5 = tmp_path / "from_tracking.analysis.h5"
    h5_res = export_pose_dataset(tracking_csv, out_h5, target_format="sleap-h5")
    assert h5_res["multi_animal"] is True
    assert sorted(h5_res["instances"]) == ["mouse_1", "mouse_2"]
    ds_h5 = read_pose_dataset(out_h5)
    assert len(ds_h5.frames[0].instances) == 2


def test_gui_dlc_overlay_loading_supports_multi_animal_csv_and_h5(
    tmp_path: Path,
) -> None:
    from annolid.gui.mixins.frame_playback_mixin import FramePlaybackMixin
    from annolid.gui.mixins.schema_behavior_loader_mixin import (
        SchemaBehaviorLoaderMixin,
    )

    class DummyWindow(SchemaBehaviorLoaderMixin, FramePlaybackMixin):
        def __init__(self) -> None:
            self._df_deeplabcut = None
            self._df_deeplabcut_columns = None
            self._df_deeplabcut_scorer = None
            self._df_deeplabcut_bodyparts = None
            self._df_deeplabcut_animal_ids = None
            self._df_deeplabcut_multi_animal = False
            self.loaded_shapes: list = []

        def loadShapes(self, shapes) -> None:
            self.loaded_shapes = list(shapes)

    csv_path = tmp_path / "multi_dlc.csv"
    csv_path.write_text(
        "\n".join(
            [
                "scorer,s,s,s,s,s,s",
                "individuals,m1,m1,m1,m2,m2,m2",
                "bodyparts,nose,nose,nose,nose,nose,nose",
                "coords,x,y,likelihood,x,y,likelihood",
                "0,15.0,25.0,0.97,55.0,65.0,0.89",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    win = DummyWindow()
    assert win._load_deeplabcut_table(str(csv_path)) is True
    assert win._df_deeplabcut_multi_animal is True
    win._load_deeplabcut_results(0)
    assert len(win.loaded_shapes) == 2
    labels = {s.label: s for s in win.loaded_shapes}
    assert "m1_nose" in labels
    assert "m2_nose" in labels
    assert labels["m1_nose"].group_id == 0
    assert labels["m2_nose"].group_id == 1
    assert labels["m1_nose"].points[0].x() == pytest.approx(15.0)
    assert labels["m1_nose"].other_data["score"] == pytest.approx(0.97)


def test_cli_import_pose_and_export_pose_round_trip(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    dlc_in = tmp_path / "in_dlc.csv"
    dlc_in.write_text(
        "\n".join(
            [
                "scorer,lp,lp,lp,lp,lp,lp",
                "bodyparts,nose,nose,nose,tail_base,tail_base,tail_base",
                "coords,x,y,likelihood,x,y,likelihood",
                "0,10.0,20.0,0.95,30.0,40.0,0.90",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    lm_out = tmp_path / "lm_cli"
    rc = cli_main(["import-pose", str(dlc_in), "-o", str(lm_out)])
    assert rc == 0
    imp_payload = json.loads(capsys.readouterr().out)
    assert imp_payload["json_written"] == 1
    assert (lm_out / "pose_schema.json").is_file()

    slp_out = tmp_path / "exported.slp"
    rc2 = cli_main(
        ["export-pose", str(lm_out), "-o", str(slp_out), "--format", "sleap-slp"]
    )
    assert rc2 == 0
    exp_payload = json.loads(capsys.readouterr().out)
    assert exp_payload["target_format"] == "sleap_slp"
    assert slp_out.is_file()


def test_agent_dataset_tools_detect_and_import_sleap_with_boundary(
    tmp_path: Path,
) -> None:
    pytest.importorskip("h5py")
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    outside.mkdir()
    ds_root = workspace / "sleap_dataset"
    ds_root.mkdir(parents=True)

    img = ds_root / "frame0.png"
    Image.new("RGB", (32, 32), color=(1, 2, 3)).save(img)
    slp_file = ds_root / "labels.pkg.slp"
    write_sleap_slp(
        PoseDataset(
            frames=[
                PoseFrame(
                    frame_index=0,
                    image_path=img.name,
                    source_image_path=str(img),
                    image_width=32,
                    image_height=32,
                    instances=[
                        PoseInstance(
                            instance_name="mouse_0",
                            track_id=0,
                            keypoints={"nose": PosePoint(5.0, 6.0)},
                        )
                    ],
                )
            ],
            keypoints=["nose"],
            instances=["mouse_0"],
        ),
        slp_file,
        embed_images=True,
    )

    inspect_tool = AnnolidDatasetInspectTool(allowed_dir=workspace)
    inspected = json.loads(
        asyncio.run(inspect_tool.execute(dataset_folder=str(ds_root)))
    )
    assert inspected["ok"] is True
    assert "sleap" in inspected["dataset"]["external_formats"]

    prep_tool = AnnolidDatasetPrepareTool(allowed_dir=workspace)
    # Boundary check: reject output_dir outside workspace
    denied = json.loads(
        asyncio.run(
            prep_tool.execute(
                dataset_folder=str(ds_root),
                mode="sleap_import",
                output_dir=str(outside / "escape"),
                allow_mutation=True,
            )
        )
    )
    assert denied["ok"] is False

    ok = json.loads(
        asyncio.run(
            prep_tool.execute(
                dataset_folder=str(ds_root),
                mode="sleap_import",
                output_dir=str(ds_root / "converted"),
                allow_mutation=True,
                write_index=True,
            )
        )
    )
    assert ok["ok"] is True
    assert ok["result"]["json_written"] == 1
    assert Path(ok["result"]["index_file"]).is_file()


def test_gui_dlc_and_sleap_dialogs_smoke(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    from qtpy import QtWidgets

    from annolid.gui.widgets.convert_deeplabcut_dialog import ConvertDLCDialog
    from annolid.gui.widgets.convert_sleap_dialog import ConvertSleapDialog

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    assert app is not None

    dlc_dlg = ConvertDLCDialog()
    assert dlc_dlg.mode_combo.count() == 3
    dlc_dlg.mode_combo.setCurrentIndex(2)
    assert dlc_dlg.stack.currentIndex() == 2
    dlc_dlg.close()

    sleap_dlg = ConvertSleapDialog()
    assert sleap_dlg.mode_combo.count() == 2
    sleap_dlg.mode_combo.setCurrentIndex(1)
    assert sleap_dlg.stack.currentIndex() == 1
    sleap_dlg.close()


def test_dlc_pytables_table_format_h5_pure_h5py_and_safe_unpickle(
    tmp_path: Path,
) -> None:
    import pickle

    h5py = pytest.importorskip("h5py")
    from annolid.annotation.pose_interop.deeplabcut import _safe_unpickle_literals

    # 1. Security: literal-only pickle decoder MUST reject executable opcodes (GLOBAL/REDUCE)
    malicious = b"cos\nsystem\n(S'echo pwned'\ntR."
    with pytest.raises(ValueError, match="Disallowed pickle opcode"):
        _safe_unpickle_literals(malicious)

    # Valid protocol 0 & 4 literal pickles decode identically
    sample_cols = [
        ("DLC_scorer", "mouse1", "nose", "x"),
        ("DLC_scorer", "mouse1", "nose", "y"),
        ("DLC_scorer", "mouse1", "nose", "likelihood"),
        ("DLC_scorer", "mouse2", "nose", "x"),
        ("DLC_scorer", "mouse2", "nose", "y"),
        ("DLC_scorer", "mouse2", "nose", "likelihood"),
    ]
    for proto in (0, 2, 4):
        assert (
            _safe_unpickle_literals(pickle.dumps(sample_cols, protocol=proto))
            == sample_cols
        )

    # 2. PyTables 'table' format multi-animal analysis HDF5 (no PyTables installed)
    h5_analysis = tmp_path / "table_maDLC.h5"
    row_dt = np.dtype([("index", "<i8"), ("values_block_0", "<f8", (6,))])
    rows = np.array(
        [
            (0, [11.5, 21.5, 0.96, 51.5, 61.5, 0.88]),
            (1, [12.5, 22.5, 0.91, np.nan, np.nan, 0.12]),
        ],
        dtype=row_dt,
    )
    with h5py.File(h5_analysis, "w") as h5:
        grp = h5.create_group("df_with_missing")
        grp.attrs["table_type"] = "appendable_frame"
        grp.attrs["non_index_axes"] = np.bytes_(
            pickle.dumps([(0, sample_cols)], protocol=0)
        )
        grp.attrs["values_cols"] = np.bytes_(
            pickle.dumps(["values_block_0"], protocol=0)
        )
        grp.attrs["index_cols"] = np.bytes_(pickle.dumps([(0, "index")], protocol=0))
        ds = grp.create_dataset("table", data=rows)
        ds.attrs["values_block_0_kind"] = np.bytes_(
            pickle.dumps(sample_cols, protocol=0)
        )

    assert detect_pose_source_format(h5_analysis) == "deeplabcut_h5"
    ds_out = read_pose_dataset(h5_analysis, min_likelihood=0.5)
    assert ds_out.multi_animal is True
    assert ds_out.instances == ["mouse1", "mouse2"]
    assert ds_out.frames[0].instances[0].keypoints["nose"].x == pytest.approx(11.5)
    assert ds_out.frames[0].instances[1].keypoints["nose"].y == pytest.approx(61.5)
    assert len(ds_out.frames[1].instances) == 1

    # 3. PyTables 'table' format MultiIndex training CollectedData_*.h5 (levels_0/1/2)
    proj = tmp_path / "dlc_table_proj"
    seg = proj / "labeled-data" / "clip1"
    seg.mkdir(parents=True)
    img = seg / "img0001.png"
    Image.new("RGB", (40, 30), color=(5, 10, 15)).save(img)

    train_cols = [("alice", "nose", "x"), ("alice", "nose", "y")]
    train_dt = np.dtype(
        [
            ("index", "<i8"),
            ("level_0", "S16"),
            ("level_1", "S16"),
            ("level_2", "S16"),
            ("values_block_0", "<f8", (2,)),
        ]
    )
    train_rows = np.array(
        [(0, b"labeled-data", b"clip1", b"img0001.png", [14.0, 18.0])],
        dtype=train_dt,
    )
    h5_train = seg / "CollectedData_alice.h5"
    with h5py.File(h5_train, "w") as h5:
        grp = h5.create_group("df_with_missing")
        grp.attrs["levels"] = np.bytes_(
            pickle.dumps(["level_0", "level_1", "level_2"], protocol=0)
        )
        grp.attrs["non_index_axes"] = np.bytes_(
            pickle.dumps([(0, train_cols)], protocol=0)
        )
        grp.attrs["values_cols"] = np.bytes_(
            pickle.dumps(["values_block_0"], protocol=0)
        )
        ds = grp.create_dataset("table", data=train_rows)
        ds.attrs["values_block_0_kind"] = np.bytes_(
            pickle.dumps(train_cols, protocol=0)
        )

    summary = import_deeplabcut_training_data(
        DeepLabCutTrainingImportConfig(source_dir=proj, overwrite=True),
        write_pose_schema=True,
    )
    assert summary["json_written"] == 1
    payload = json.loads(img.with_suffix(".json").read_text(encoding="utf-8"))
    assert payload["shapes"][0]["points"] == [[14.0, 18.0]]


def test_sleap_zip_containers_json_and_wrapped_hdf5_and_zipslip_safety(
    tmp_path: Path,
) -> None:
    import io
    import zipfile

    pytest.importorskip("h5py")

    # Helper PNG bytes
    buf = io.BytesIO()
    Image.new("RGB", (48, 36), color=(12, 34, 56)).save(buf, format="PNG")
    png_bytes = buf.getvalue()

    # 1. Legacy SLEAP JSON-in-ZIP .slp archive with embedded image, tracks, edges, occluded kp, and user+pred
    json_slp_zip = tmp_path / "legacy_json.slp"
    sleap_json = {
        "nodes": [{"name": "nose"}, {"name": "tail"}],
        "skeletons": [
            {
                "graph": {"name": "mouse"},
                "nodes": [{"id": 0}, {"id": 1}],
                "links": [{"source": {"id": 0}, "target": {"id": 1}, "type": 1}],
            }
        ],
        "tracks": [{"name": "mouse_A"}, {"name": "mouse_B"}],
        "videos": [{"backend": {"filename": "trial_cam1.mp4"}}],
        "labeled_frames": [
            {
                "frame_idx": 3,
                "video": 0,
                "instances": [
                    {
                        "py/object": "sleap.instance.Instance",
                        "skeleton": 0,
                        "track": 0,
                        "points": {
                            "0": {"x": 10.0, "y": 15.0, "visible": True},
                            "1": {"x": 20.0, "y": 25.0, "visible": False},
                        },
                    },
                    {
                        "py/object": "sleap.instance.PredictedInstance",
                        "skeleton": 0,
                        "track": 0,
                        "score": 0.77,
                        "points": {
                            "0": {"x": 99.0, "y": 99.0, "visible": True, "score": 0.8}
                        },
                    },
                ],
            }
        ],
    }
    with zipfile.ZipFile(json_slp_zip, "w") as zf:
        zf.writestr("labels.json", json.dumps(sleap_json))
        zf.writestr("video0/000000003.png", png_bytes)
        # Path-traversal entry must be ignored safely
        zf.writestr("../escape.png", png_bytes)

    assert detect_pose_source_format(json_slp_zip) == "sleap_slp"
    out_json_zip = tmp_path / "from_zip_json"
    summary = convert_sleap_h5_to_labelme_summary(
        json_slp_zip, out_json_zip, save_frames=True
    )
    assert summary["sleap_kind"] == "zip_slp"
    assert summary["json_written"] == 1
    assert summary["frames_extracted"] == 1
    assert (out_json_zip / "000000003.png").is_file()
    assert not (tmp_path / "escape.png").exists()

    f3 = json.loads((out_json_zip / "000000003.json").read_text(encoding="utf-8"))
    # prefer_user_instances=True keeps only user instance (2 points: visible nose + occluded tail)
    assert len(f3["shapes"]) == 2
    by_kp = {s["display_label"]: s for s in f3["shapes"]}
    assert by_kp["nose"]["instance_label"] == "mouse_A"
    assert visibility_from_labelme_shape(by_kp["tail"]) == int(
        KeypointVisibility.OCCLUDED
    )
    schema = PoseSchema.load(out_json_zip / "pose_schema.json")
    assert schema.edges == [("nose", "tail")]

    # 2. ZIP container wrapping an HDF5 .pkg.slp
    img_p = tmp_path / "src0.png"
    img_p.write_bytes(png_bytes)
    inner_pkg = tmp_path / "inner.pkg.slp"
    write_sleap_slp(
        PoseDataset(
            frames=[
                PoseFrame(
                    frame_index=0,
                    image_path=img_p.name,
                    source_image_path=str(img_p),
                    image_width=48,
                    image_height=36,
                    instances=[
                        PoseInstance(
                            instance_name="fly_1",
                            track_id=0,
                            keypoints={"head": PosePoint(5.0, 6.0)},
                        )
                    ],
                )
            ],
            keypoints=["head"],
            instances=["fly_1"],
        ),
        inner_pkg,
        embed_images=True,
    )
    wrapped_zip = tmp_path / "wrapped.slp"
    with zipfile.ZipFile(wrapped_zip, "w") as zf:
        zf.write(inner_pkg, arcname="inner.pkg.slp")

    assert detect_pose_source_format(wrapped_zip) == "sleap_slp"
    out_wrapped = tmp_path / "from_wrapped_zip"
    w_summary = convert_sleap_h5_to_labelme_summary(
        wrapped_zip, out_wrapped, save_frames=True
    )
    assert w_summary["json_written"] == 1
    assert (out_wrapped / "000000000.png").is_file()
    w0 = json.loads((out_wrapped / "000000000.json").read_text(encoding="utf-8"))
    assert w0["shapes"][0]["instance_label"] == "fly_1"
    assert w0["shapes"][0]["points"] == [[5.0, 6.0]]
