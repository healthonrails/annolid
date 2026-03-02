from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Type

import numpy as np

from annolid.gui.shape import Shape
from annolid.utils.annotation_store import AnnotationStore, load_labelme_json


def clean_instance_label(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    return text.rstrip("_-:|")


def normalize_group_id(value: object) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):  # bool is also int
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and np.isfinite(value):
        return int(value)
    if isinstance(value, str):
        raw = value.strip()
        if raw.isdigit():
            return int(raw)
    return None


def next_group_id(used: set[int]) -> int:
    candidate = max(used) + 1 if used else 0
    while candidate in used:
        candidate += 1
    return candidate


def sanitize_inference_config(cfg: Optional[dict]) -> Dict[str, object]:
    payload = dict(cfg or {})
    tta_merge = str(payload.get("tta_merge", "mean") or "mean").strip().lower()
    if tta_merge not in {"mean", "max"}:
        tta_merge = "mean"

    try:
        min_score = float(payload.get("min_keypoint_score", 0.0))
    except Exception:
        min_score = 0.0
    if not np.isfinite(min_score) or min_score < 0:
        min_score = 0.0

    return {
        "tta_hflip": bool(payload.get("tta_hflip", False)),
        "tta_merge": str(tta_merge),
        "min_keypoint_score": float(min_score),
        "stabilize_lr": bool(payload.get("stabilize_lr", True)),
    }


def load_prompt_shapes(
    output_directory: Path,
    *,
    frame_index: int,
    labelme_json_path,
    legacy_labelme_json_path,
) -> List[Dict[str, object]]:
    frame_index = int(frame_index)
    candidates = (
        Path(labelme_json_path(output_directory, frame_index=frame_index)),
        Path(legacy_labelme_json_path(output_directory, frame_index=frame_index)),
    )
    for path in candidates:
        try:
            if path.exists() and path.stat().st_size > 0:
                payload = load_labelme_json(path)
                shapes = payload.get("shapes") or []
                if isinstance(shapes, list):
                    return [s for s in shapes if isinstance(s, dict)]
        except Exception:
            continue

    store = AnnotationStore.for_frame_path(
        Path(labelme_json_path(output_directory, frame_index=frame_index))
    )
    record = store.get_frame(frame_index)
    shapes = record.get("shapes") if isinstance(record, dict) else None
    if isinstance(shapes, list):
        return [s for s in shapes if isinstance(s, dict)]
    return []


def instance_masks_from_shapes(
    shapes: Sequence[Dict[str, object]],
    *,
    frame_hw: Tuple[int, int],
    pose_schema: Optional[object],
    instance_label_to_gid: Dict[str, int],
) -> List[Tuple[int, np.ndarray]]:
    try:
        import cv2  # type: ignore
    except Exception:
        return []

    height, width = int(frame_hw[0]), int(frame_hw[1])
    if height <= 0 or width <= 0:
        return []

    schema_instances: Dict[str, int] = {}
    if pose_schema is not None and getattr(pose_schema, "instances", None):
        for idx, name in enumerate(getattr(pose_schema, "instances", ())):
            clean = clean_instance_label(name)
            if clean and clean.lower() not in schema_instances:
                schema_instances[clean.lower()] = int(idx)

    polygon_like: List[Dict[str, object]] = []
    for shape in shapes:
        shape_type = str(shape.get("shape_type") or "").strip().lower()
        if shape_type in ("polygon", "rectangle", "circle"):
            polygon_like.append(shape)

    label_counts: Dict[str, int] = {}
    for shape in polygon_like:
        clean = clean_instance_label(shape.get("label"))
        if not clean:
            continue
        key = clean.lower()
        label_counts[key] = int(label_counts.get(key, 0)) + 1

    instance_masks: List[Tuple[int, np.ndarray]] = []
    used_gids: set[int] = set()
    for shape in polygon_like:
        shape_type = str(shape.get("shape_type") or "").strip().lower()
        points = shape.get("points") or []
        if not isinstance(points, list) or not points:
            continue

        flags = shape.get("flags") if isinstance(shape.get("flags"), dict) else {}
        gid = normalize_group_id(shape.get("group_id"))
        if gid is None:
            gid = normalize_group_id(shape.get("instance_id"))
        if gid is None and flags:
            gid = normalize_group_id(flags.get("instance_id"))

        label = clean_instance_label(shape.get("label"))
        if gid is None and label:
            by_schema = schema_instances.get(label.lower())
            if by_schema is not None:
                gid = int(by_schema)
        if gid is None and label and int(label_counts.get(label.lower(), 0)) == 1:
            existing = instance_label_to_gid.get(label.lower())
            if existing is not None:
                gid = int(existing)
        if gid is None:
            gid = next_group_id(used_gids)
        elif int(gid) in used_gids:
            gid = next_group_id(used_gids)
        used_gids.add(int(gid))
        if label and int(label_counts.get(label.lower(), 0)) == 1:
            instance_label_to_gid[label.lower()] = int(gid)

        mask = np.zeros((height, width), dtype=np.uint8)
        if shape_type == "polygon":
            poly = []
            for pt in points:
                if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                    continue
                poly.append((float(pt[0]), float(pt[1])))
            if len(poly) < 3:
                continue
            poly_arr = np.rint(np.array(poly, dtype=np.float32)).astype(np.int32)
            cv2.fillPoly(mask, [poly_arr], 1)
        elif shape_type == "rectangle":
            if len(points) < 2:
                continue
            a, b = points[0], points[1]
            if (
                not isinstance(a, (list, tuple))
                or not isinstance(b, (list, tuple))
                or len(a) < 2
                or len(b) < 2
            ):
                continue
            x1 = int(round(min(float(a[0]), float(b[0]))))
            y1 = int(round(min(float(a[1]), float(b[1]))))
            x2 = int(round(max(float(a[0]), float(b[0]))))
            y2 = int(round(max(float(a[1]), float(b[1]))))
            x1 = max(0, min(width - 1, x1))
            x2 = max(0, min(width, x2))
            y1 = max(0, min(height - 1, y1))
            y2 = max(0, min(height, y2))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
        elif shape_type == "circle":
            if len(points) < 2:
                continue
            c, e = points[0], points[1]
            if (
                not isinstance(c, (list, tuple))
                or not isinstance(e, (list, tuple))
                or len(c) < 2
                or len(e) < 2
            ):
                continue
            cx, cy = float(c[0]), float(c[1])
            ex, ey = float(e[0]), float(e[1])
            r = int(round(((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5))
            if r <= 0:
                continue
            cv2.circle(mask, (int(round(cx)), int(round(cy))), r, 1, thickness=-1)

        if not np.any(mask):
            continue
        instance_masks.append((int(gid), mask.astype(bool)))

    instance_masks.sort(key=lambda item: int(item[0]))
    return instance_masks


def extract_results(
    *,
    frame_bgr: np.ndarray,
    model: object,
    model_type: str,
    keypoint_names: Optional[Sequence[str]],
    inference_config: Optional[dict],
    bboxes: Optional[np.ndarray] = None,
    instance_masks: Optional[Sequence[Tuple[int, np.ndarray]]] = None,
    instance_ids: Optional[Sequence[object]] = None,
    shape_factory: Type[Shape] = Shape,
    log,
) -> list:
    annotations = []
    dino_cfg = sanitize_inference_config(inference_config)
    tta_hflip = bool(dino_cfg.get("tta_hflip", False))
    tta_merge = str(dino_cfg.get("tta_merge", "mean") or "mean")
    min_keypoint_score = float(dino_cfg.get("min_keypoint_score", 0.0))
    stabilize_lr = bool(dino_cfg.get("stabilize_lr", True))

    peaks = []
    try:
        if instance_masks:
            from annolid.segmentation.dino_kpseg.inference_utils import (
                build_instance_crops,
                predict_on_instance_crops,
            )

            crops = build_instance_crops(
                frame_bgr,
                list(instance_masks),
                pad_px=8,
                use_mask_gate=True,
            )
            predictions = predict_on_instance_crops(
                model,
                crops,
                return_patch_masks=False,
                stabilize_lr=bool(stabilize_lr),
                tta_hflip=bool(tta_hflip),
                tta_merge=str(tta_merge),
            )
        elif bboxes is not None and len(bboxes) > 0:
            predictions = model.predict_instances(
                frame_bgr,
                bboxes_xyxy=bboxes,
                instance_ids=instance_ids,
                return_patch_masks=False,
                stabilize_lr=bool(stabilize_lr),
                tta_hflip=bool(tta_hflip),
                tta_merge=str(tta_merge),
            )
        else:
            peaks = model.predict_multi_peaks(
                frame_bgr,
                threshold=None,
                topk=5,
                nms_radius_px=12.0,
                tta_hflip=bool(tta_hflip),
                tta_merge=str(tta_merge),
            )
            predictions = [
                (
                    None,
                    model.predict(
                        frame_bgr,
                        return_patch_masks=False,
                        stabilize_lr=bool(stabilize_lr),
                        tta_hflip=bool(tta_hflip),
                        tta_merge=str(tta_merge),
                    ),
                )
            ]
    except Exception as exc:
        log.error("DinoKPSEG inference failed: %s", exc, exc_info=True)
        return annotations

    kp_names = keypoint_names or getattr(model, "keypoint_names", None)
    if not kp_names and predictions:
        first_pred = predictions[0][1]
        kp_names = [str(i) for i in range(len(first_pred.keypoints_xy))]

    if (bboxes is None or len(bboxes) == 0) and not instance_masks:
        emitted_peaks = 0
        if kp_names:
            for kpt_id, channel_peaks in enumerate(peaks):
                label = kp_names[kpt_id] if kpt_id < len(kp_names) else str(kpt_id)
                for rank, (x, y, score) in enumerate(channel_peaks):
                    if float(score) < float(min_keypoint_score):
                        continue
                    point_shape = shape_factory(
                        label,
                        shape_type="point",
                        description=str(model_type),
                        flags={},
                        group_id=None,
                    )
                    point_shape.points = [[float(x), float(y)]]
                    point_shape.other_data["score"] = float(score)
                    point_shape.other_data["peak_rank"] = int(rank)
                    point_shape.other_data["multi_peak"] = True
                    annotations.append(point_shape)
                    emitted_peaks += 1
            if emitted_peaks > 0:
                return annotations

    for instance_id, prediction in predictions:
        group_id = int(instance_id) if instance_id is not None else None
        for kpt_id, (xy, score) in enumerate(
            zip(prediction.keypoints_xy, prediction.keypoint_scores)
        ):
            if float(score) < float(min_keypoint_score):
                continue
            label = kp_names[kpt_id] if kpt_id < len(kp_names) else str(kpt_id)
            x, y = float(xy[0]), float(xy[1])

            point_shape = shape_factory(
                label,
                shape_type="point",
                description=str(model_type),
                flags={},
                group_id=group_id,
            )
            point_shape.points = [[x, y]]
            point_shape.other_data["score"] = float(score)
            if group_id is not None:
                point_shape.other_data["instance_id"] = int(group_id)
            annotations.append(point_shape)

    return annotations
