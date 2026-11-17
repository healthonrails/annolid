"""Comprehensive regression tests for body-part hierarchy contracts, optimizations, and ethogram analytics."""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from annolid.annotation import labelme2coco
from annolid.annotation.labelme2yolo import Labelme2YOLO
from annolid.features.dinov3_extractor import Dinov3Config, Dinov3FeatureExtractor
from annolid.postprocessing.body_part_ethogram import (
    analyze_body_part_ethogram,
    export_body_part_ethogram_csvs,
)
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_analysis_engine import GenericZoneEngine
from annolid.postprocessing.zone_schema import build_zone_shape, load_zone_shapes
from annolid.segmentation.body_parts import (
    BodyPartSegmentationConfig,
    InstanceBodyPartSegmenter,
    _trace_medial_bone_on_canvas,
    estimate_body_frame_pose,
)
from annolid.services.body_part_segmentation import run_body_part_segmentation
from annolid.utils.body_part_shapes import (
    build_body_part_shape_dict,
    infer_intra_part_skeleton_edges,
    split_body_part_csv_instance_name,
)


def _write_synthetic_labelme_hierarchy_dataset(dataset_dir: Path) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(4):
        img = np.full((120, 160, 3), 210, dtype=np.uint8)
        shift = idx * 4
        cv2.rectangle(img, (20 + shift, 30), (80 + shift, 66), (40, 40, 45), -1)
        cv2.rectangle(img, (92 - shift, 62), (144 - shift, 96), (235, 232, 225), -1)
        img_name = f"sample_{idx:03d}.png"
        cv2.imwrite(str(dataset_dir / img_name), img)

        m1_poly = [
            [float(20 + shift), 30.0],
            [float(80 + shift), 30.0],
            [float(80 + shift), 66.0],
            [float(20 + shift), 66.0],
        ]
        m2_poly = [
            [float(92 - shift), 62.0],
            [float(144 - shift), 62.0],
            [float(144 - shift), 96.0],
            [float(92 - shift), 96.0],
        ]
        head_poly = [
            [float(62 + shift), 32.0],
            [float(79 + shift), 32.0],
            [float(79 + shift), 64.0],
            [float(62 + shift), 64.0],
        ]
        torso_poly = [
            [float(34 + shift), 32.0],
            [float(61 + shift), 32.0],
            [float(61 + shift), 64.0],
            [float(34 + shift), 64.0],
        ]
        tail_poly = [
            [float(21 + shift), 44.0],
            [float(33 + shift), 44.0],
            [float(33 + shift), 52.0],
            [float(21 + shift), 52.0],
        ]

        shapes: List[dict[str, Any]] = [
            {
                "label": "feeder_zone",
                "points": [[5.0, 5.0], [50.0, 5.0], [50.0, 25.0], [5.0, 25.0]],
                "group_id": None,
                "shape_type": "polygon",
                "flags": {"zone": True},
                "shape_category": "zone",
            },
            {
                "label": "mouse_1",
                "points": m1_poly,
                "group_id": 1,
                "shape_type": "polygon",
                "flags": {},
            },
            {
                "label": "mouse_2",
                "points": m2_poly,
                "group_id": 2,
                "shape_type": "polygon",
                "flags": {},
            },
            {
                "label": "nose",
                "points": [[float(76 + shift), 48.0]],
                "group_id": 1,
                "shape_type": "point",
                "flags": {"kp_visible": True},
            },
            {
                "label": "neck",
                "points": [[float(58 + shift), 48.0]],
                "group_id": 1,
                "shape_type": "point",
                "flags": {"kp_visible": True},
            },
            {
                "label": "tail_base",
                "points": [[float(32 + shift), 48.0]],
                "group_id": 1,
                "shape_type": "point",
                "flags": {"kp_visible": True},
            },
            {
                "label": "nose",
                "points": [[float(96 - shift), 79.0]],
                "group_id": 2,
                "shape_type": "point",
                "flags": {"kp_visible": True},
            },
            {
                "label": "neck",
                "points": [[float(114 - shift), 79.0]],
                "group_id": 2,
                "shape_type": "point",
                "flags": {"kp_visible": True},
            },
            {
                "label": "tail_base",
                "points": [[float(140 - shift), 79.0]],
                "group_id": 2,
                "shape_type": "point",
                "flags": {"kp_visible": True},
            },
            # Child body-part shapes placed AFTER parent shapes with the same group_id=1!
            build_body_part_shape_dict(
                points=head_poly,
                parent_instance="mouse_1",
                part_label="head",
                group_id=1,
                confidence=0.92,
            ),
            build_body_part_shape_dict(
                points=torso_poly,
                parent_instance="mouse_1",
                part_label="torso",
                group_id=1,
                confidence=0.95,
            ),
            build_body_part_shape_dict(
                points=tail_poly,
                parent_instance="mouse_1",
                part_label="tail",
                group_id=1,
                confidence=0.88,
            ),
        ]
        payload = {
            "version": "5.4.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": img_name,
            "imageData": None,
            "imageHeight": 120,
            "imageWidth": 160,
        }
        (dataset_dir / f"sample_{idx:03d}.json").write_text(
            json.dumps(payload, indent=2) + "\n", encoding="utf-8"
        )


def test_labelme2coco_hierarchy_export_levels_and_keypoints_protection(
    tmp_path: Path,
) -> None:
    src_dir = tmp_path / "labelme_src"
    _write_synthetic_labelme_hierarchy_dataset(src_dir)

    # 1. Default export_level="instances": must skip zones and child body parts!
    inst_out = tmp_path / "coco_inst"
    list(
        labelme2coco.convert(
            str(src_dir),
            str(inst_out),
            labels_file=None,
            train_valid_split=0.75,
            output_mode="segmentation",
            export_level="instances",
        )
    )
    inst_train = json.loads((inst_out / "annotations_train.json").read_text())
    inst_cats = [c["name"] for c in inst_train["categories"]]
    assert set(inst_cats) == {"mouse_1", "mouse_2"}
    assert all(not ann.get("is_body_part") for ann in inst_train["annotations"])

    # 2. export_level="parts": exports canonical anatomical part classes with parent metadata!
    parts_out = tmp_path / "coco_parts"
    list(
        labelme2coco.convert(
            str(src_dir),
            str(parts_out),
            labels_file=None,
            train_valid_split=0.75,
            output_mode="segmentation",
            export_level="parts",
        )
    )
    parts_train = json.loads((parts_out / "annotations_train.json").read_text())
    part_cats = [c["name"] for c in parts_train["categories"]]
    assert set(part_cats) == {"head", "torso", "tail"}
    assert len(parts_train["annotations"]) > 0
    for ann in parts_train["annotations"]:
        assert ann["is_body_part"] is True
        assert ann["parent_instance"] == "mouse_1"

    # 3. export_level="both": includes both parent instances and child body-part categories.
    both_out = tmp_path / "coco_both"
    list(
        labelme2coco.convert(
            str(src_dir),
            str(both_out),
            labels_file=None,
            train_valid_split=0.75,
            output_mode="segmentation",
            export_level="both",
        )
    )
    both_train = json.loads((both_out / "annotations_train.json").read_text())
    both_cats = {c["name"] for c in both_train["categories"]}
    assert {"mouse_1", "mouse_2", "head", "torso", "tail"} <= both_cats

    # 4. output_mode="keypoints": child body-part polygons must not pollute categories or keypoints!
    kp_out = tmp_path / "coco_kp"
    list(
        labelme2coco.convert(
            str(src_dir),
            str(kp_out),
            labels_file=None,
            train_valid_split=0.75,
            output_mode="keypoints",
        )
    )
    kp_train = json.loads((kp_out / "annotations_train.json").read_text())
    kp_cats = {c["name"] for c in kp_train["categories"]}
    assert kp_cats == {"mouse_1", "mouse_2"}
    for ann in kp_train["annotations"]:
        assert ann["num_keypoints"] == 3


def test_labelme2yolo_hierarchy_export_levels_and_pose_class_label_safety(
    tmp_path: Path,
) -> None:
    # 1. Pose export on dataset containing keypoints + child body parts sharing group_id=1:
    #    must preserve mouse_1 / mouse_2 class labels instead of overwriting with 'tail'!
    pose_dir = tmp_path / "yolo_pose_src"
    _write_synthetic_labelme_hierarchy_dataset(pose_dir)
    conv_pose = Labelme2YOLO(str(pose_dir), export_level="instances")
    assert set(conv_pose.label_to_id_dict.keys()) == {"mouse_1", "mouse_2"}
    conv_pose.convert(val_size=0.25, test_size=0.0)
    assert conv_pose.annotation_type == "pose"
    yaml_text = (pose_dir / "YOLO_dataset" / "data.yaml").read_text(encoding="utf-8")
    assert "mouse_1" in yaml_text
    assert "mouse_2" in yaml_text
    assert "tail:" not in yaml_text
    assert "feeder_zone" not in yaml_text

    # 2. Parts-only segmentation export: ignores keypoints & whole animals, exports canonical parts!
    parts_dir = tmp_path / "yolo_parts_src"
    _write_synthetic_labelme_hierarchy_dataset(parts_dir)
    conv_parts = Labelme2YOLO(str(parts_dir), export_level="parts")
    assert set(conv_parts.label_to_id_dict.keys()) == {"head", "torso", "tail"}
    conv_parts.convert(val_size=0.25, test_size=0.0)
    assert conv_parts.annotation_type == "segmentation"
    lbl_files = list((parts_dir / "YOLO_dataset" / "labels").rglob("*.txt"))
    assert len(lbl_files) == 4
    first_lines = [
        ln.strip()
        for ln in lbl_files[0].read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    assert len(first_lines) == 3  # head, torso, tail


def test_zone_engine_and_tracking_analyzer_filter_body_parts_and_support_anchor_part(
    tmp_path: Path,
) -> None:
    assert split_body_part_csv_instance_name("mouse_1:head") == ("mouse_1", "head")
    assert split_body_part_csv_instance_name("mouse_1") == ("mouse_1", "")

    feeder_points = [[60.0, 10.0], [100.0, 10.0], [100.0, 60.0], [60.0, 60.0]]
    zone_specs = load_zone_shapes(
        {
            "shapes": [
                build_zone_shape(
                    "feeder",
                    feeder_points,
                    shape_type="polygon",
                    zone_kind="chamber",
                    phase="phase_1",
                    occupant_role="stim",
                    access_state="open",
                )
            ]
        }
    )
    rows = [
        # Frame 0: mouse_1 outside feeder (both body and head)
        {"frame_number": 0, "instance_name": "mouse_1", "cx": 30.0, "cy": 35.0},
        {"frame_number": 0, "instance_name": "mouse_1:head", "cx": 45.0, "cy": 35.0},
        {"frame_number": 0, "instance_name": "mouse_1:tail", "cx": 15.0, "cy": 35.0},
        {"frame_number": 0, "instance_name": "mouse_2", "cx": 130.0, "cy": 80.0},
        # Frame 1: mouse_1 whole-body centroid still outside (x=52), but snout/head pokes INSIDE feeder (x=68)!
        {"frame_number": 1, "instance_name": "mouse_1", "cx": 52.0, "cy": 35.0},
        {"frame_number": 1, "instance_name": "mouse_1:head", "cx": 68.0, "cy": 35.0},
        {"frame_number": 1, "instance_name": "mouse_1:tail", "cx": 28.0, "cy": 35.0},
        {"frame_number": 1, "instance_name": "mouse_2", "cx": 128.0, "cy": 80.0},
        # Frame 2: mouse_1:head missing on this frame -> anchor_part="head" must fall back cleanly to mouse_1!
        {"frame_number": 2, "instance_name": "mouse_1", "cx": 72.0, "cy": 35.0},
        {"frame_number": 2, "instance_name": "mouse_2", "cx": 125.0, "cy": 80.0},
    ]
    df = pd.DataFrame(rows)

    default_engine = GenericZoneEngine(zone_specs, fps=10.0)
    default_res = default_engine.analyze_dataframe(df)
    assert set(default_res.keys()) == {"mouse_1", "mouse_2"}
    assert default_res["mouse_1"].occupancy_frames["feeder"] == 1  # Only frame 2

    head_engine = GenericZoneEngine(zone_specs, fps=10.0, anchor_part="head")
    head_res = head_engine.analyze_dataframe(df)
    assert set(head_res.keys()) == {"mouse_1", "mouse_2"}
    assert (
        head_res["mouse_1"].occupancy_frames["feeder"] == 2
    )  # Frame 1 (via mouse_1:head) + Frame 2 (fallback to mouse_1)!

    # Verify TrackingResultsAnalyzer pairwise distances_df excludes child '<parent>:<part>' rows.
    vid_path = tmp_path / "trial.mp4"
    writer = cv2.VideoWriter(
        str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (160, 120)
    )
    for _ in range(3):
        writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    writer.release()

    zone_json = tmp_path / "zones.json"
    zone_json.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "feeder",
                        "shape_type": "polygon",
                        "points": feeder_points,
                        "flags": {"zone": True},
                        "shape_category": "zone",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    df.to_csv(tmp_path / "trial_tracking.csv", index=False)
    df_tracked = df.copy()
    df_tracked["feeder"] = 0
    df_tracked.to_csv(tmp_path / "trial_tracked.csv", index=False)

    analyzer = TrackingResultsAnalyzer(
        vid_path, zone_file=zone_json, fps=10.0, anchor_part="head"
    )
    analyzer.merge_and_calculate_distance()
    pairs = {
        tuple(sorted((str(r["instance_name_1"]), str(r["instance_name_2"]))))
        for _, r in analyzer.distances_df.iterrows()
    }
    assert pairs == {("mouse_1", "mouse_2")}


class _StubViTBackbone(nn.Module):
    """Lightweight deterministic conv-patch ViT stub for offline unit testing."""

    def __init__(self, patch_size: int = 16, embed_dim: int = 24, num_layers: int = 4):
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=patch_size,
            num_hidden_layers=num_layers,
            num_register_tokens=2,
        )
        torch.manual_seed(42)
        self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.layers = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(num_layers)]
        )

    def forward(
        self, pixel_values: torch.Tensor, output_hidden_states: bool = False
    ) -> Any:
        B = pixel_values.shape[0]
        patches = self.proj(pixel_values)  # [B, D, gh, gw]
        D = patches.shape[1]
        tokens = patches.flatten(2).transpose(1, 2)  # [B, N, D]
        cls_and_reg = torch.zeros(
            (B, 1 + self.config.num_register_tokens, D),
            dtype=tokens.dtype,
            device=tokens.device,
        )
        cur = torch.cat([cls_and_reg, tokens], dim=1)
        hidden = [cur]
        for layer in self.layers:
            cur = torch.tanh(layer(cur)) + 0.5 * cur
            hidden.append(cur)
        return SimpleNamespace(
            last_hidden_state=cur,
            hidden_states=tuple(hidden) if output_hidden_states else None,
        )


def _make_stub_dinov3_extractor() -> Dinov3FeatureExtractor:
    ext = object.__new__(Dinov3FeatureExtractor)
    ext.cfg = Dinov3Config(
        short_side=64,
        device="cpu",
        use_amp=False,
        return_layer="all",
        layers=(-2, -1),
    )
    ext.model_id = "stub-dinov3"
    ext.cache_dir = None
    ext.device = torch.device("cpu")
    ext.processor = SimpleNamespace(
        image_mean=[0.485, 0.456, 0.406], image_std=[0.229, 0.224, 0.225]
    )
    ext.model = _StubViTBackbone().eval()
    ext._mean = [0.485, 0.456, 0.406]
    ext._std = [0.229, 0.224, 0.225]
    ext.patch_size = 16
    ext._num_special_tokens = 3
    ext.num_hidden_layers = 4
    ext._layers = (-2, -1)
    return ext


def test_dinov3_extract_batch_and_segmenter_batch_crop_parity() -> None:
    extractor = _make_stub_dinov3_extractor()
    rng = np.random.default_rng(123)
    imgs = [rng.integers(0, 255, size=(64, 64, 3), dtype=np.uint8) for _ in range(3)]

    singles = [
        extractor.extract(im, return_layer="all", normalize=True, return_type="numpy")
        for im in imgs
    ]
    batched = extractor.extract_batch(
        imgs, return_layer="all", normalize=True, return_type="numpy"
    )
    assert len(batched) == 3
    for s_arr, b_arr in zip(singles, batched):
        assert s_arr.shape == (2, 24, 4, 4)
        assert b_arr.shape == s_arr.shape
        np.testing.assert_allclose(s_arr, b_arr, atol=1e-5)

    # Verify InstanceBodyPartSegmenter calls extract_batch once for multi-instance frames!
    calls: List[int] = []
    orig_batch = extractor.extract_batch

    def _spy_batch(images_seq: Any, **kwargs: Any) -> Any:
        seq = list(images_seq)
        calls.append(len(seq))
        return orig_batch(seq, **kwargs)

    extractor.extract_batch = _spy_batch  # type: ignore[method-assign]
    seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="keypoint_guided",
            feature_backend="hybrid",
            crop_size=64,
        ),
        extractor=extractor,
    )
    frame = rng.integers(30, 220, size=(120, 160, 3), dtype=np.uint8)
    m1 = np.zeros((120, 160), dtype=bool)
    m2 = np.zeros((120, 160), dtype=bool)
    m1[20:60, 20:75] = True
    m2[65:105, 85:140] = True
    kps = {
        "mouse_1": {
            "nose": (70.0, 40.0),
            "neck": (52.0, 40.0),
            "tail_base": (24.0, 40.0),
        },
        "mouse_2": {
            "nose": (90.0, 85.0),
            "neck": (110.0, 85.0),
            "tail_base": (136.0, 85.0),
        },
    }
    preds = seg.segment_frame(
        frame, {"mouse_1": m1, "mouse_2": m2}, keypoints_by_instance=kps
    )
    assert calls == [2]
    assert set(preds.keys()) == {"mouse_1", "mouse_2"}


def test_auto_skeleton_inference_and_curved_medial_bone_tracing() -> None:
    edges = infer_intra_part_skeleton_edges(
        [
            "nose",
            "left_ear",
            "right_ear",
            "neck",
            "spine_1",
            "spine_2",
            "tail_base",
            "tail_mid",
            "tail_tip",
            "left_elbow",
            "left_forepaw",
            "right_knee",
            "right_hindpaw",
        ]
    )
    edge_set = {tuple(sorted(e)) for e in edges}
    assert ("nose", "left_ear") in edges or ("left_ear", "nose") in edges
    assert ("neck", "spine_1") in edges
    assert ("spine_1", "spine_2") in edges
    assert ("tail_base", "tail_mid") in edges
    assert ("tail_mid", "tail_tip") in edges
    assert ("left_elbow", "left_forepaw") in edges
    assert ("right_knee", "right_hindpaw") in edges
    # Never bridge left <-> right limbs directly!
    assert tuple(sorted(("left_forepaw", "right_hindpaw"))) not in edge_set
    assert tuple(sorted(("left_ear", "right_ear"))) not in edge_set

    # Curved C-shaped silhouette where straight chord between (60, 16) and (60, 84) crosses background air!
    mask_u8 = np.zeros((100, 100), dtype=np.uint8)
    cv2.ellipse(mask_u8, (55, 50), (34, 36), 0, 95, 265, 1, thickness=14)
    work_mask = mask_u8 > 0
    pose = estimate_body_frame_pose(work_mask)
    assert pose.tortuosity > 1.20

    # Verify straight midpoint (60, 50) is outside work_mask, yet traced bone stays 100% inside work_mask!
    assert not work_mask[50, 55]
    ys_top, xs_top = np.nonzero(work_mask[:25, :])
    ys_bot, xs_bot = np.nonzero(work_mask[75:, :])
    p0 = (float(xs_top.mean()), float(ys_top.mean()))
    p1 = (float(xs_bot.mean()), float(ys_bot.mean() + 75.0))
    canvas = np.zeros((100, 100), dtype=np.uint8)
    _trace_medial_bone_on_canvas(
        canvas, work_mask, pose.thickness_map, p0, p1, radius=2, prefer_thin=False
    )
    drawn = canvas > 0
    assert int(np.count_nonzero(drawn)) > 25
    assert int(np.count_nonzero(drawn & (~work_mask))) == 0


def test_posture_archetype_bank_and_bidirectional_midvideo_seed_propagation(
    tmp_path: Path,
) -> None:
    # 1. Posture-indexed prototype bank retains distinct elongated vs curled spatial priors.
    seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(mode="seeded", feature_backend="fast")
    )
    rgb_elong = np.full((100, 140, 3), 180, dtype=np.uint8)
    m_elong = np.zeros((100, 140), dtype=bool)
    m_elong[40:60, 20:110] = True
    rgb_elong[40:60, 88:110] = (90, 70, 70)
    rgb_elong[40:60, 42:88] = (45, 45, 50)
    rgb_elong[45:55, 20:42] = (145, 115, 115)
    seeds_elong = {
        "mouse_1": {
            "head": m_elong & (np.arange(140)[None, :] >= 88),
            "torso": m_elong
            & (np.arange(140)[None, :] >= 42)
            & (np.arange(140)[None, :] < 88),
            "tail": m_elong & (np.arange(140)[None, :] < 42),
        }
    }
    seg.fit_seed_frame(rgb_elong, {"mouse_1": m_elong}, seeds_elong)

    rgb_curl = np.full((100, 140, 3), 180, dtype=np.uint8)
    c_u8 = np.zeros((100, 140), dtype=np.uint8)
    cv2.circle(c_u8, (70, 50), 22, 1, -1)
    m_curl = c_u8 > 0
    yy, xx = np.indices((100, 140))
    rgb_curl[m_curl & (xx >= 76)] = (90, 70, 70)
    rgb_curl[m_curl & (xx < 76)] = (45, 45, 50)
    seeds_curl = {
        "mouse_1": {
            "head": m_curl & (xx >= 76),
            "torso": m_curl & (xx < 76) & (yy <= 62),
            "tail": m_curl & (xx < 76) & (yy > 62),
        }
    }
    seg.fit_seed_frame(rgb_curl, {"mouse_1": m_curl}, seeds_curl)
    head_proto = next(
        p for p in seg._instance_prototypes["mouse_1"] if p.part_label == "head"
    )
    assert len(head_proto.posture_archetypes) == 2

    # 2. Bidirectional video propagation from a SINGLE manual seed at mid-video Frame 3!
    vid_path = tmp_path / "midseed.mp4"
    ann_dir = tmp_path / "midseed"
    ann_dir.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), 8.0, (160, 120)
    )
    for fid in range(6):
        dx = fid * 4
        bgr = np.full((120, 160, 3), 215, dtype=np.uint8)
        # Tail (left), Torso (middle), Head (right)
        cv2.rectangle(bgr, (20 + dx, 52), (38 + dx, 60), (150, 130, 180), -1)
        cv2.rectangle(bgr, (38 + dx, 42), (82 + dx, 70), (45, 45, 50), -1)
        cv2.rectangle(bgr, (82 + dx, 45), (104 + dx, 67), (80, 70, 120), -1)
        writer.write(bgr)

        parent_poly = [
            [float(20 + dx), 42.0],
            [float(104 + dx), 42.0],
            [float(104 + dx), 70.0],
            [float(20 + dx), 70.0],
        ]
        shapes_f: List[dict[str, Any]] = [
            {
                "label": "mouse_1",
                "points": parent_poly,
                "group_id": 1,
                "shape_type": "polygon",
                "flags": {},
            }
        ]
        if fid == 3:
            # Manual seed ONLY on Frame 3!
            cv2.imwrite(str(ann_dir / f"midseed_{fid:09d}.png"), bgr)
            shapes_f.extend(
                [
                    build_body_part_shape_dict(
                        points=[
                            [float(82 + dx), 45.0],
                            [float(104 + dx), 45.0],
                            [float(104 + dx), 67.0],
                            [float(82 + dx), 67.0],
                        ],
                        parent_instance="mouse_1",
                        part_label="head",
                        group_id=1,
                        mode="manual",
                    ),
                    build_body_part_shape_dict(
                        points=[
                            [float(38 + dx), 42.0],
                            [float(82 + dx), 42.0],
                            [float(82 + dx), 70.0],
                            [float(38 + dx), 70.0],
                        ],
                        parent_instance="mouse_1",
                        part_label="torso",
                        group_id=1,
                        mode="manual",
                    ),
                    build_body_part_shape_dict(
                        points=[
                            [float(20 + dx), 52.0],
                            [float(38 + dx), 52.0],
                            [float(38 + dx), 60.0],
                            [float(20 + dx), 60.0],
                        ],
                        parent_instance="mouse_1",
                        part_label="tail",
                        group_id=1,
                        mode="manual",
                    ),
                ]
            )
        (ann_dir / f"midseed_{fid:09d}.json").write_text(
            json.dumps(
                {
                    "version": "5.4.1",
                    "flags": {},
                    "shapes": shapes_f,
                    "imagePath": f"midseed_{fid:09d}.png",
                    "imageData": None,
                    "imageHeight": 120,
                    "imageWidth": 160,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    writer.release()

    summary = run_body_part_segmentation(
        vid_path,
        annotation_dir=ann_dir,
        config=BodyPartSegmentationConfig(
            mode="seeded",
            feature_backend="fast",
            persist_json=True,
        ),
        export_csv=True,
        export_ethogram=True,
        fps=8.0,
    )
    assert summary["seed_frames_fitted"] == 1
    assert summary["frames_processed"] == 6
    assert summary["instances_segmented"] == 6
    assert summary["parts_emitted"] >= 10
    assert summary["ethogram"] is not None
    assert Path(summary["ethogram"]["kinematics_csv"]).exists()

    # Verify Frame 0 (propagated backward from Frame 3!) has non-empty head & torso body parts!
    f0_payload = json.loads((ann_dir / "midseed_000000000.json").read_text())
    f0_part_labels = {
        sh.get("part_label")
        for sh in f0_payload["shapes"]
        if sh.get("flags", {}).get("is_body_part")
    }
    assert {"head", "torso"} <= f0_part_labels


def test_body_part_ethogram_kinematics_and_directed_social_bouts(
    tmp_path: Path,
) -> None:
    ann_dir = tmp_path / "etho_trial"
    ann_dir.mkdir(parents=True, exist_ok=True)

    for fid in range(6):
        # Frames 0..2: nose-to-nose contact between mouse_1 and mouse_2
        # Frames 3..5: mouse_1 head sniffs mouse_2 tail base (anogenital_sniff) while mouse_1 bends head 60 deg!
        if fid < 3:
            m1_head = [[74.0, 44.0], [90.0, 44.0], [90.0, 56.0], [74.0, 56.0]]
            m1_torso = [[42.0, 40.0], [74.0, 40.0], [74.0, 60.0], [42.0, 60.0]]
            m1_tail = [[22.0, 48.0], [42.0, 48.0], [42.0, 52.0], [22.0, 52.0]]
            m1_parent = [[22.0, 40.0], [90.0, 40.0], [90.0, 60.0], [22.0, 60.0]]

            m2_head = [[92.0, 44.0], [108.0, 44.0], [108.0, 56.0], [92.0, 56.0]]
            m2_torso = [[108.0, 40.0], [140.0, 40.0], [140.0, 60.0], [108.0, 60.0]]
            m2_tail = [[140.0, 48.0], [156.0, 48.0], [156.0, 52.0], [140.0, 52.0]]
            m2_parent = [[92.0, 40.0], [156.0, 40.0], [156.0, 60.0], [92.0, 60.0]]
        else:
            # mouse_1 turns head upward toward mouse_2's tail at y=28..38 while torso stays horizontal -> strong deflection!
            m1_head = [[66.0, 22.0], [82.0, 22.0], [82.0, 38.0], [66.0, 38.0]]
            m1_torso = [[42.0, 38.0], [74.0, 38.0], [74.0, 58.0], [42.0, 58.0]]
            m1_tail = [[20.0, 46.0], [42.0, 46.0], [42.0, 50.0], [20.0, 50.0]]
            m1_parent = [[20.0, 22.0], [82.0, 22.0], [82.0, 58.0], [20.0, 58.0]]

            m2_tail = [[83.0, 24.0], [98.0, 24.0], [98.0, 32.0], [83.0, 32.0]]
            m2_torso = [[98.0, 18.0], [130.0, 18.0], [130.0, 38.0], [98.0, 38.0]]
            m2_head = [[130.0, 20.0], [146.0, 20.0], [146.0, 36.0], [130.0, 36.0]]
            m2_parent = [[83.0, 18.0], [146.0, 18.0], [146.0, 38.0], [83.0, 38.0]]

        shapes = [
            {
                "label": "mouse_1",
                "points": m1_parent,
                "group_id": 1,
                "shape_type": "polygon",
                "flags": {},
            },
            {
                "label": "mouse_2",
                "points": m2_parent,
                "group_id": 2,
                "shape_type": "polygon",
                "flags": {},
            },
            build_body_part_shape_dict(
                points=m1_head,
                parent_instance="mouse_1",
                part_label="head",
                group_id=1,
                confidence=0.92,
            ),
            build_body_part_shape_dict(
                points=m1_torso,
                parent_instance="mouse_1",
                part_label="torso",
                group_id=1,
                confidence=0.94,
            ),
            build_body_part_shape_dict(
                points=m1_tail,
                parent_instance="mouse_1",
                part_label="tail",
                group_id=1,
                confidence=0.86,
            ),
            build_body_part_shape_dict(
                points=m2_head,
                parent_instance="mouse_2",
                part_label="head",
                group_id=2,
                confidence=0.91,
            ),
            build_body_part_shape_dict(
                points=m2_torso,
                parent_instance="mouse_2",
                part_label="torso",
                group_id=2,
                confidence=0.93,
            ),
            build_body_part_shape_dict(
                points=m2_tail,
                parent_instance="mouse_2",
                part_label="tail",
                group_id=2,
                confidence=0.88,
            ),
        ]
        (ann_dir / f"etho_trial_{fid:09d}.json").write_text(
            json.dumps({"shapes": shapes, "imageHeight": 120, "imageWidth": 180}),
            encoding="utf-8",
        )

    report = analyze_body_part_ethogram(
        ann_dir, fps=10.0, contact_margin_px=6.0, min_bout_frames=2
    )
    assert len(report.kinematics_df) == 12
    m1_f4 = report.kinematics_df[
        (report.kinematics_df["frame_number"] == 4)
        & (report.kinematics_df["instance_name"] == "mouse_1")
    ].iloc[0]
    assert math.isfinite(float(m1_f4["head_body_deflection_deg"]))
    assert abs(float(m1_f4["head_body_deflection_deg"])) > 35.0

    bout_types = {(b.actor, b.recipient, b.behavior) for b in report.bouts}
    assert ("mouse_1", "mouse_2", "nose_to_nose") in bout_types
    assert ("mouse_1", "mouse_2", "anogenital_sniff") in bout_types
    # Directed check: mouse_2 did NOT sniff mouse_1's tail!
    assert ("mouse_2", "mouse_1", "anogenital_sniff") not in bout_types

    exported = export_body_part_ethogram_csvs(report, tmp_path, stem="etho_trial")
    assert Path(exported["kinematics_csv"]).exists()
    assert Path(exported["contacts_csv"]).exists()
    assert Path(exported["bouts_csv"]).exists()


def test_two_mice_scene_domain_randomization_contracts() -> None:
    from scripts.generate_two_mice_body_part_videos import (
        DEFAULT_SCENARIOS,
        DOMAIN_RANDOMIZATION_SCENARIOS,
    )

    scene_js = (
        Path(__file__).resolve().parents[1]
        / "annolid"
        / "gui"
        / "assets"
        / "threejs"
        / "two_mice_scene.js"
    ).read_text(encoding="utf-8")
    for token in (
        "DATASET_COAT_PAIR",
        "DATASET_LIGHTING",
        "DATASET_OCCLUDER",
        "DATASET_ARENA",
        "DATASET_HEADSTAGE",
        "c57_pair",
        "ir_night",
        "calms_gray",
        "home_cage",
        "bottom-up",
        "wire_bars",
        "ventral_gait",
        "calms_resident_intruder",
    ):
        assert token in scene_js

    assert len(DEFAULT_SCENARIOS) == 4
    assert len(DOMAIN_RANDOMIZATION_SCENARIOS) == 5
    variant_names = {s.name for s in DOMAIN_RANDOMIZATION_SCENARIOS}
    assert variant_names == {
        "same_strain_c57_huddle",
        "ir_night_explore",
        "ventral_glass_gait",
        "wire_bar_occluded",
        "calms_resident_intruder",
    }


def test_wire_bar_occluded_multi_component_parent_and_multi_polygon_parts() -> None:
    rgb = np.full((120, 180, 3), 205, dtype=np.uint8)
    parent_mask = np.zeros((120, 180), dtype=bool)
    # Tail (20..48), Torso (48..112), Head (112..142)
    parent_mask[54:62, 20:48] = True
    parent_mask[42:74, 48:112] = True
    parent_mask[46:70, 112:142] = True

    rgb[54:62, 20:48] = (165, 130, 140)
    rgb[42:74, 48:112] = (38, 38, 42)
    rgb[46:70, 112:142] = (65, 55, 75)

    # Slice two 8-px stainless wire bars right across torso (x=76..84) and tail/rump (x=36..44)!
    bar_mask = np.zeros((120, 180), dtype=bool)
    bar_mask[:, 36:44] = True
    bar_mask[:, 76:84] = True
    rgb[bar_mask] = (175, 180, 186)
    occluded_parent = parent_mask & (~bar_mask)

    seg = InstanceBodyPartSegmenter(
        BodyPartSegmentationConfig(
            mode="keypoint_guided",
            feature_backend="fast",
            keypoint_granularity="coarse",
        )
    )
    # Notice: NO keypoint on the distal tail tip island (x=20..36), only tail_mid at x=46!
    kps = {
        "mouse_1": {
            "nose": (134.0, 58.0),
            "neck": (110.0, 58.0),
            "spine_1": (92.0, 58.0),
            "spine_2": (62.0, 58.0),
            "tail_mid": (46.0, 58.0),
        }
    }
    preds = seg.segment_frame(
        rgb, {"mouse_1": occluded_parent}, keypoints_by_instance=kps
    )["mouse_1"]
    assert {"head", "torso", "tail"} <= set(preds.keys())

    # 1. Zero containment violations onto wire bars!
    for p in preds.values():
        full_m = p.to_full_mask(120, 180)
        assert int(np.count_nonzero(full_m & bar_mask)) == 0
        assert int(np.count_nonzero(full_m & (~occluded_parent))) == 0

    # 2. Bar-split torso emits >= 2 disjoint sub-polygons in pred.polygons!
    assert len(preds["torso"].polygons) >= 2

    # 3. Distal tail island across the 8-px bar gap (x=20..36) is reached by tail!
    tail_full = preds["tail"].to_full_mask(120, 180)
    assert int(np.count_nonzero(tail_full[:, 20:36])) >= 80

    # 4. Multi-polygon LabelMe JSON parent fragments for the same (label, group_id) merge into one instance!
    from annolid.segmentation.body_parts import parse_frame_instances_and_seed_parts

    multi_parent_payload = {
        "imageHeight": 120,
        "imageWidth": 180,
        "shapes": [
            {
                "label": "mouse_1",
                "group_id": 1,
                "shape_type": "polygon",
                "points": [[20.0, 54.0], [35.0, 54.0], [35.0, 61.0], [20.0, 61.0]],
            },
            {
                "label": "mouse_1",
                "group_id": 1,
                "shape_type": "polygon",
                "points": [[48.0, 42.0], [75.0, 42.0], [75.0, 73.0], [48.0, 73.0]],
            },
            {
                "label": "mouse_1",
                "group_id": 1,
                "shape_type": "polygon",
                "points": [[84.0, 42.0], [141.0, 42.0], [141.0, 73.0], [84.0, 73.0]],
            },
        ],
    }
    p_masks, p_polys, p_gids, _, _ = parse_frame_instances_and_seed_parts(
        multi_parent_payload, default_height=120, default_width=180
    )
    assert list(p_masks.keys()) == ["mouse_1"]
    assert p_polys["mouse_1"] == []
    assert p_gids["mouse_1"] == 1
    preds_from_json = seg.segment_frame(
        rgb, p_masks, keypoints_by_instance=kps, parent_polygons=p_polys
    )["mouse_1"]
    assert len(preds_from_json["torso"].polygons) >= 2


def test_ethogram_disentangles_rearing_from_curled_grooming_and_northbound_walking() -> (
    None
):
    def _rect(x0: float, y0: float, x1: float, y1: float) -> List[List[float]]:
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

    frames = []
    # Phase A (Frames 0-1): Northbound (-y) horizontal walking -> high elongation, full area (~1800 px)
    for fid in (0, 1):
        shapes = [
            {
                "label": "mouse_1",
                "points": _rect(75, 20, 105, 85),
                "group_id": 1,
                "shape_type": "polygon",
            },
            build_body_part_shape_dict(
                points=_rect(78, 20, 102, 36),
                parent_instance="mouse_1",
                part_label="head",
                group_id=1,
            ),
            build_body_part_shape_dict(
                points=_rect(75, 36, 105, 72),
                parent_instance="mouse_1",
                part_label="torso",
                group_id=1,
            ),
            build_body_part_shape_dict(
                points=_rect(86, 72, 94, 85),
                parent_instance="mouse_1",
                part_label="tail",
                group_id=1,
            ),
        ]
        frames.append((fid, {"shapes": shapes}))

    # Phase B (Frames 2-3): Flat top-down curled grooming ball -> area still ~1760 px, strong lateral head deflection & short head-tail span!
    for fid in (2, 3):
        shapes = [
            {
                "label": "mouse_1",
                "points": _rect(68, 38, 112, 78),
                "group_id": 1,
                "shape_type": "polygon",
            },
            build_body_part_shape_dict(
                points=_rect(92, 58, 110, 74),
                parent_instance="mouse_1",
                part_label="head",
                group_id=1,
            ),
            build_body_part_shape_dict(
                points=_rect(68, 38, 104, 66),
                parent_instance="mouse_1",
                part_label="torso",
                group_id=1,
            ),
            build_body_part_shape_dict(
                points=_rect(86, 68, 102, 78),
                parent_instance="mouse_1",
                part_label="tail",
                group_id=1,
            ),
        ]
        frames.append((fid, {"shapes": shapes}))

    # Phase C (Frames 4-5): Top-down upright rear -> cos(theta) foreshortened area (~900 px < 1800 px), compact & axially collinear!
    for fid in (4, 5):
        shapes = [
            {
                "label": "mouse_1",
                "points": _rect(76, 42, 104, 74),
                "group_id": 1,
                "shape_type": "polygon",
            },
            build_body_part_shape_dict(
                points=_rect(80, 42, 100, 52),
                parent_instance="mouse_1",
                part_label="head",
                group_id=1,
            ),
            build_body_part_shape_dict(
                points=_rect(76, 52, 104, 68),
                parent_instance="mouse_1",
                part_label="torso",
                group_id=1,
            ),
            build_body_part_shape_dict(
                points=_rect(86, 68, 94, 74),
                parent_instance="mouse_1",
                part_label="tail",
                group_id=1,
            ),
        ]
        frames.append((fid, {"shapes": shapes}))

    # Phase D (Frames 6-7): Oblique bipedal rear -> upward torso->head pitch + visible bipedal limbs + hindpaw below nose!
    for fid in (6, 7):
        shapes = [
            {
                "label": "mouse_1",
                "points": _rect(76, 30, 106, 76),
                "group_id": 1,
                "shape_type": "polygon",
            },
            build_body_part_shape_dict(
                points=_rect(80, 30, 100, 44),
                parent_instance="mouse_1",
                part_label="head",
                group_id=1,
            ),
            build_body_part_shape_dict(
                points=_rect(78, 44, 102, 68),
                parent_instance="mouse_1",
                part_label="torso",
                group_id=1,
            ),
            build_body_part_shape_dict(
                points=_rect(76, 58, 86, 76),
                parent_instance="mouse_1",
                part_label="limbs",
                group_id=1,
            ),
            build_body_part_shape_dict(
                points=_rect(96, 68, 106, 76),
                parent_instance="mouse_1",
                part_label="tail",
                group_id=1,
            ),
            {
                "label": "nose",
                "points": [[90.0, 32.0]],
                "group_id": 1,
                "shape_type": "point",
                "flags": {"kp_visible": True},
            },
            {
                "label": "left_hindpaw",
                "points": [[82.0, 74.0]],
                "group_id": 1,
                "shape_type": "point",
                "flags": {"kp_visible": True},
            },
        ]
        frames.append((fid, {"shapes": shapes}))

    rep = analyze_body_part_ethogram(frames, fps=10.0, camera_view="auto")
    kdf = rep.kinematics_df.set_index("frame_number")

    # Northbound walking is NOT misclassified as rearing
    assert float(kdf.loc[0, "rearing_index"]) < 0.25
    assert str(kdf.loc[0, "posture_state"]) == "locomotion"

    # Curled grooming ball has high curl_groom_index and suppressed rearing_index!
    assert float(kdf.loc[2, "curl_groom_index"]) > 0.60
    assert float(kdf.loc[2, "rearing_index"]) < 0.25
    assert str(kdf.loc[2, "posture_state"]) == "curled_grooming"

    # Top-down foreshortened rear has high rearing_index and low curl_groom_index!
    assert float(kdf.loc[4, "rearing_index"]) >= 0.48
    assert float(kdf.loc[4, "curl_groom_index"]) < 0.25
    assert str(kdf.loc[4, "posture_state"]) == "rearing"

    # Oblique bipedal rear has high rearing_index and rearing posture_state!
    assert float(kdf.loc[6, "rearing_index"]) >= 0.55
    assert str(kdf.loc[6, "posture_state"]) == "rearing"
