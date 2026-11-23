"""Unit tests for body-part segmentation dataset builders, mask->pose solver, and Cutie-BodyPartSeg head."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from annolid.data.body_part_dataset import (
    HIERARCHICAL_CLASS_TO_ID,
    SEMANTIC_PART_NAME_TO_ID,
    expand_snout_muzzle_mask,
    extract_hierarchical_body_part_polygons,
    fine_id_map_to_semantic_parts,
    format_yolo_seg_lines,
    summarize_body_part_dataset,
    write_coco_seg_split,
    write_yolo_seg_data_yaml,
)
from annolid.postprocessing.body_part_to_pose import (
    INSECT_3TAGMA_KEYPOINTS,
    BodyPartPoseConfig,
    extract_insect_pose_from_part_masks,
    extract_pose_from_part_masks,
    group_part_detections_to_pose_frame,
)
from annolid.segmentation.cutie_vos.model.cutie_body_part_seg import (
    CutieBodyPartSegConfig,
    CutieBodyPartSegHead,
    compute_body_part_seg_loss,
    load_cutie_body_part_seg_checkpoint,
    save_cutie_body_part_seg_checkpoint,
)


def _draw_synthetic_mouse_id_maps(
    h: int = 240,
    w: int = 320,
    cx: int = 160,
    cy: int = 120,
    tid: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build synthetic top-down mouse facing +x (right) with exact 3D fine/coarse/track ID maps."""
    track = np.zeros((h, w), dtype=np.uint8)
    fine = np.zeros((h, w), dtype=np.uint8)
    coarse = np.zeros((h, w), dtype=np.uint8)

    # Torso centered at (cx-15, cy), semi-axes (35, 22)
    cv2.ellipse(track, (cx - 15, cy), (35, 22), 0, 0, 360, tid, -1)
    cv2.ellipse(fine, (cx - 15, cy), (35, 22), 0, 0, 360, 6, -1)
    cv2.ellipse(coarse, (cx - 15, cy), (35, 22), 0, 0, 360, 2, -1)

    # Head at (cx+28, cy), semi-axes (18, 13)
    cv2.ellipse(track, (cx + 28, cy), (18, 13), 0, 0, 360, tid, -1)
    cv2.ellipse(fine, (cx + 28, cy), (18, 13), 0, 0, 360, 1, -1)
    cv2.ellipse(coarse, (cx + 28, cy), (18, 13), 0, 0, 360, 1, -1)

    # Snout tip at (cx+48, cy), radius 6
    cv2.circle(track, (cx + 48, cy), 6, tid, -1)
    cv2.circle(fine, (cx + 48, cy), 6, 2, -1)
    cv2.circle(coarse, (cx + 48, cy), 6, 1, -1)

    # Ears: top (y=cy-14, fine=3), bottom (y=cy+14, fine=4)
    cv2.circle(track, (cx + 24, cy - 14), 6, tid, -1)
    cv2.circle(fine, (cx + 24, cy - 14), 6, 3, -1)
    cv2.circle(coarse, (cx + 24, cy - 14), 6, 1, -1)

    cv2.circle(track, (cx + 24, cy + 14), 6, tid, -1)
    cv2.circle(fine, (cx + 24, cy + 14), 6, 4, -1)
    cv2.circle(coarse, (cx + 24, cy + 14), 6, 1, -1)

    # Tail extending left from (cx-50, cy) to (cx-100, cy+5)
    cv2.line(track, (cx - 50, cy), (cx - 100, cy + 5), tid, 5)
    cv2.line(fine, (cx - 50, cy), (cx - 100, cy + 5), 11, 5)
    cv2.line(coarse, (cx - 50, cy), (cx - 100, cy + 5), 4, 5)

    return track, fine, coarse


def test_body_part_dataset_builder_yolo_coco_and_semantic_roundtrip(
    tmp_path: Path,
) -> None:
    track, fine, coarse = _draw_synthetic_mouse_id_maps()
    sem = fine_id_map_to_semantic_parts(track, fine, target_track_id=1)
    assert set(np.unique(sem).tolist()) == set(SEMANTIC_PART_NAME_TO_ID.values())

    records = extract_hierarchical_body_part_polygons(track, fine, coarse)
    names = [r.class_name for r in records]
    assert "resident" in names
    assert "head" in names
    assert "snout" in names
    assert names.count("ear") == 2
    assert "torso" in names
    assert "tail" in names

    lines = format_yolo_seg_lines(records, image_width=320, image_height=240)
    assert len(lines) == len(records)
    for line in lines:
        toks = line.split()
        assert len(toks) >= 7 and (len(toks) - 1) % 2 == 0
        vals = [float(x) for x in toks[1:]]
        assert all(0.0 <= v <= 1.0 for v in vals)

    # Write minimal YOLO-Seg + COCO-Seg + semantic dataset and audit via summarize_body_part_dataset
    for split in ("train", "val"):
        (tmp_path / "images" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "labels" / split).mkdir(parents=True, exist_ok=True)
        (tmp_path / "semantic_masks" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(
            str(tmp_path / "images" / split / "f0.jpg"),
            np.zeros((240, 320, 3), dtype=np.uint8),
        )
        (tmp_path / "labels" / split / "f0.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        cv2.imwrite(str(tmp_path / "semantic_masks" / split / "f0.png"), sem)

    yaml_path = write_yolo_seg_data_yaml(tmp_path)
    assert yaml_path.is_file()
    coco_path = write_coco_seg_split(
        tmp_path / "annotations" / "train.json",
        images=[
            {"id": 1, "file_name": "images/train/f0.jpg", "width": 320, "height": 240}
        ],
        annotations=[
            {
                "id": idx + 1,
                "image_id": 1,
                "category_id": r.class_id + 1,
                "segmentation": [[c for pt in r.points_xy for c in pt]],
                "bbox": list(r.bbox_xywh),
                "area": r.area,
                "iscrowd": 0,
            }
            for idx, r in enumerate(records)
        ],
    )
    assert coco_path.is_file()

    stats = summarize_body_part_dataset(tmp_path)
    assert stats.images_by_split == {"train": 1, "val": 1}
    assert stats.semantic_masks_by_split == {"train": 1, "val": 1}
    assert stats.polygons_by_class["resident"] == 2
    assert stats.polygons_by_class["ear"] == 4
    assert stats.empty_label_files == 0


def test_body_part_to_pose_geometry_chirality_and_multi_animal_grouping() -> None:
    track, fine, coarse = _draw_synthetic_mouse_id_maps(cx=160, cy=120, tid=1)
    sem = fine_id_map_to_semantic_parts(track, fine, target_track_id=1)
    part_masks = {
        "head": (sem == 1) | (sem == 2) | (sem == 3),
        "snout": sem == 2,
        "ear": sem == 3,
        "torso": sem == 4,
        "tail": sem == 5,
    }

    # 1. CalMS21 chirality: facing +x -> left_ear & left_hip sit at +y (bottom), right at -y (top)
    kps_calms, heading = extract_pose_from_part_masks(
        part_masks,
        instance_mask=(track == 1) & (coarse != 4),
        config=BodyPartPoseConfig(chirality="calms21"),
    )
    assert heading is not None and heading[0] > 0.9
    assert kps_calms["nose"].x > 205.0
    assert abs(kps_calms["nose"].y - 120.0) < 4.0
    assert kps_calms["tail_base"].x < 120.0
    assert kps_calms["left_ear"].y > kps_calms["right_ear"].y
    assert kps_calms["left_hip"].y > kps_calms["right_hip"].y

    # 2. Anatomical chirality: facing +x -> left_ear & left_hip sit at -y (top)
    kps_anat, _ = extract_pose_from_part_masks(
        part_masks,
        instance_mask=(track == 1) & (coarse != 4),
        config=BodyPartPoseConfig(chirality="anatomical"),
    )
    assert kps_anat["left_ear"].y < kps_anat["right_ear"].y
    assert kps_anat["left_hip"].y < kps_anat["right_hip"].y

    # 3. Multi-animal part grouping
    t2, f2, c2 = _draw_synthetic_mouse_id_maps(h=240, w=320, cx=160, cy=195, tid=2)
    dets = []
    for name, tid_m, fm, cm in (
        ("resident", track, fine, coarse),
        ("intruder", t2, f2, c2),
    ):
        dets.append(
            {"class_name": name, "score": 0.95, "mask": (tid_m > 0) & (cm != 4)}
        )
        dets.append(
            {
                "class_name": "head",
                "score": 0.9,
                "mask": (fm == 1) | (fm == 2) | (fm == 3) | (fm == 4),
            }
        )
        dets.append({"class_name": "snout", "score": 0.85, "mask": fm == 2})
        dets.append({"class_name": "ear", "score": 0.85, "mask": fm == 3})
        dets.append({"class_name": "ear", "score": 0.85, "mask": fm == 4})
        dets.append({"class_name": "torso", "score": 0.92, "mask": fm == 6})
        dets.append({"class_name": "tail", "score": 0.80, "mask": fm == 11})

    pframe, headings = group_part_detections_to_pose_frame(
        dets,
        frame_index=7,
        image_hw=(240, 320),
        config=BodyPartPoseConfig(chirality="calms21"),
    )
    assert pframe.frame_index == 7
    assert {i.instance_name for i in pframe.instances} == {"resident", "intruder"}
    assert set(headings.keys()) == {"resident", "intruder"}
    for inst in pframe.instances:
        assert len(inst.keypoints) == 7
    assert HIERARCHICAL_CLASS_TO_ID["tail"] == 6


def test_cutie_body_part_seg_head_forward_loss_and_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    cfg = CutieBodyPartSegConfig(
        hidden_dim=64, tucker_rank=32, roi_size=32, feat_size=16
    )
    head = CutieBodyPartSegHead(cfg)
    assert sum(p.numel() for p in head.parameters()) < 300_000

    p4 = torch.randn(2, cfg.c_p4, cfg.feat_size, cfg.feat_size)
    p8 = torch.randn(2, cfg.c_p8, cfg.feat_size // 2, cfg.feat_size // 2)
    msk = torch.rand(2, 1, cfg.roi_size, cfg.roi_size)
    img = torch.rand(2, 3, cfg.roi_size, cfg.roi_size)
    prev = torch.softmax(
        torch.randn(2, cfg.num_classes, cfg.roi_size, cfg.roi_size), dim=1
    )

    logits = head(p4, p8, msk, img, prev)
    assert logits.shape == (2, cfg.num_classes, cfg.roi_size, cfg.roi_size)

    tgt = torch.randint(0, cfg.num_classes, (2, cfg.roi_size, cfg.roi_size))
    loss, metrics = compute_body_part_seg_loss(logits, tgt)
    assert torch.isfinite(loss)
    assert 0.0 <= metrics["fg_dice"] <= 1.0
    loss.backward()

    ckpt = save_cutie_body_part_seg_checkpoint(
        head, tmp_path / "bps.pt", metadata={"v": 1}, fp16=True
    )
    loaded, meta = load_cutie_body_part_seg_checkpoint(ckpt, device="cpu")
    assert meta["v"] == 1
    with torch.no_grad():
        out2 = loaded(p4, p8, msk, img, None)
    assert out2.shape == (2, cfg.num_classes, cfg.roi_size, cfg.roi_size)


def test_expand_snout_muzzle_mask_tiny_and_occluded_fallback() -> None:
    track, fine, coarse = _draw_synthetic_mouse_id_maps(cx=160, cy=120, tid=1)
    head_m = fine == 1
    torso_m = fine == 6

    # Case 1: Tiny 2-pixel rhinarium tip expands into segmentable anterior muzzle cap
    tiny_snout = np.zeros_like(head_m, dtype=bool)
    tiny_snout[120, 205:207] = True
    muzzle = expand_snout_muzzle_mask(tiny_snout, head_m, torso_m)
    assert int(np.count_nonzero(muzzle)) > 40
    assert bool(muzzle[120, 205])

    # Case 2: Completely occluded rhinarium falls back to anterior 22% of head along torso->head
    empty_snout = np.zeros_like(head_m, dtype=bool)
    muzzle_fb = expand_snout_muzzle_mask(empty_snout, head_m, torso_m)
    assert int(np.count_nonzero(muzzle_fb)) > 40
    ys, xs = np.nonzero(muzzle_fb)
    _, hxs = np.nonzero(head_m)
    assert float(xs.mean()) > float(hxs.mean())
    assert int(coarse.max()) > 0


def test_group_part_detections_merges_headstage_split_parent_and_detached_head() -> (
    None
):
    """When a cranial headstage splits a mouse silhouette, detached head parts still attach cleanly."""
    h, w = 240, 320
    torso_mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(torso_mask, (110, 120), (42, 24), 0, 0, 360, 1.0, -1)

    # Anterior head fragment separated by ~28px bright headstage gap (>65px from torso centroid!)
    head_frag = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(head_frag, (195, 120), (18, 12), 0, 0, 360, 1.0, -1)

    # Secondary head detection near cranium/headstage + primary anterior rostrum
    head_cranium = np.zeros((h, w), dtype=np.float32)
    cv2.circle(head_cranium, (168, 120), 11, 1.0, -1)

    dets = [
        {"class_name": "resident", "score": 0.90, "mask": torso_mask},
        {"class_name": "resident", "score": 0.45, "mask": head_frag},
        {"class_name": "torso", "score": 0.88, "mask": torso_mask},
        {"class_name": "head", "score": 0.72, "mask": head_cranium},
        {"class_name": "head", "score": 0.65, "mask": head_frag},
    ]
    pframe, _ = group_part_detections_to_pose_frame(
        dets,
        frame_index=0,
        image_hw=(h, w),
        config=BodyPartPoseConfig(chirality="calms21"),
    )
    assert len(pframe.instances) == 1
    nose = pframe.instances[0].keypoints["nose"]
    # Nose must reach the anterior rostrum (~210), NOT stop at the cranium/torso!
    assert nose.x > 202.0


def test_group_part_detections_multi_animal_n4_same_class_and_fragment_merge() -> None:
    """N=4 same-class mice receive unique names/IDs while still merging headstage fragments."""
    h, w = 480, 640
    centers = [(150, 120), (470, 120), (150, 350), (470, 350)]
    dets = []
    for idx, (cx, cy) in enumerate(centers):
        t_m, f_m, c_m = _draw_synthetic_mouse_id_maps(
            h=h, w=w, cx=cx, cy=cy, tid=idx + 1
        )
        body_m = ((t_m > 0) & (c_m != 4)).astype(np.float32)
        if idx == 0:
            # Split mouse 0's silhouette by a headstage gap + add a high-IoU duplicate detection
            torso_only = (f_m == 6).astype(np.float32)
            head_frag = ((f_m == 1) | (f_m == 2)).astype(np.float32)
            dets.append({"class_name": "mouse", "score": 0.96, "mask": torso_only})
            dets.append({"class_name": "mouse", "score": 0.90, "mask": torso_only})
            dets.append({"class_name": "mouse", "score": 0.55, "mask": head_frag})
        else:
            dets.append(
                {"class_name": "mouse", "score": 0.92 - 0.02 * idx, "mask": body_m}
            )
        dets.append(
            {
                "class_name": "head",
                "score": 0.88,
                "mask": ((f_m == 1) | (f_m == 2) | (f_m == 3) | (f_m == 4)).astype(
                    np.float32
                ),
            }
        )
        dets.append(
            {
                "class_name": "snout",
                "score": 0.85,
                "mask": (f_m == 2).astype(np.float32),
            }
        )
        dets.append(
            {"class_name": "ear", "score": 0.85, "mask": (f_m == 3).astype(np.float32)}
        )
        dets.append(
            {"class_name": "ear", "score": 0.85, "mask": (f_m == 4).astype(np.float32)}
        )
        dets.append(
            {
                "class_name": "torso",
                "score": 0.90,
                "mask": (f_m == 6).astype(np.float32),
            }
        )
        dets.append(
            {
                "class_name": "tail",
                "score": 0.80,
                "mask": (f_m == 11).astype(np.float32),
            }
        )

    pframe, headings = group_part_detections_to_pose_frame(
        dets,
        frame_index=12,
        image_hw=(h, w),
        parent_classes=("mouse",),
        dedup_parents=False,
        config=BodyPartPoseConfig(chirality="calms21"),
    )
    assert len(pframe.instances) == 4
    names = [inst.instance_name for inst in pframe.instances]
    tids = [inst.track_id for inst in pframe.instances]
    assert names == ["mouse_1", "mouse_2", "mouse_3", "mouse_4"]
    assert len(set(tids)) == 4
    assert set(headings.keys()) == set(names)
    for inst in pframe.instances:
        assert len(inst.keypoints) == 7
        assert inst.keypoints["nose"].x > inst.keypoints["tail_base"].x
        assert inst.keypoints["left_ear"].y > inst.keypoints["right_ear"].y


def test_group_part_detections_scale_invariant_low_res_192() -> None:
    """Scale-invariant thresholds group a small 192x192 mouse cleanly without retuning."""
    h, w = 192, 192
    torso = np.zeros((h, w), dtype=np.float32)
    head = np.zeros((h, w), dtype=np.float32)
    snout = np.zeros((h, w), dtype=np.float32)
    tail = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(torso, (90, 96), (14, 8), 0, 0, 360, 1.0, -1)
    cv2.circle(head, (109, 96), 6, 1.0, -1)
    cv2.circle(snout, (114, 96), 3, 1.0, -1)
    cv2.line(tail, (75, 96), (52, 99), 1.0, 2)
    parent = np.maximum(torso, head)
    dets = [
        {"class_name": "mouse", "score": 0.90, "mask": parent},
        {"class_name": "torso", "score": 0.88, "mask": torso},
        {"class_name": "head", "score": 0.85, "mask": head},
        {"class_name": "snout", "score": 0.80, "mask": snout},
        {"class_name": "tail", "score": 0.78, "mask": tail},
    ]
    pframe, _ = group_part_detections_to_pose_frame(
        dets,
        frame_index=0,
        image_hw=(h, w),
        parent_classes=("mouse",),
        config=BodyPartPoseConfig(chirality="calms21", min_part_area_px=3),
    )
    assert len(pframe.instances) == 1
    kps = pframe.instances[0].keypoints
    assert kps["nose"].x > 112.0
    assert kps["tail_base"].x < 79.0


def test_extract_insect_pose_from_part_masks_drosophila() -> None:
    """Chirality-free insect tagma masks (head, thorax, abdomen) yield 5 ordered landmarks."""
    h, w = 230, 280
    head = np.zeros((h, w), dtype=np.float32)
    thorax = np.zeros((h, w), dtype=np.float32)
    abdomen = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(head, (194, 67), (22, 26), 15, 0, 360, 1.0, -1)
    cv2.ellipse(thorax, (132, 60), (34, 30), -10, 0, 360, 1.0, -1)
    cv2.ellipse(abdomen, (88, 118), (36, 52), 25, 0, 360, 1.0, -1)

    kps, heading = extract_insect_pose_from_part_masks(
        {"head": head, "thorax": thorax, "abdomen": abdomen}
    )
    assert heading is not None and heading[0] > 0.5
    assert tuple(kps.keys()) == INSECT_3TAGMA_KEYPOINTS
    assert kps["head_tip"].x > kps["neck_joint"].x > kps["thorax_center"].x
    assert kps["thorax_center"].x > kps["thorax_abdomen_joint"].x > kps["abdomen_tip"].x
    assert kps["abdomen_tip"].y > kps["thorax_center"].y
