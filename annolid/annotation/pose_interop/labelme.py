"""Bidirectional bridge between Annolid LabelMe / Tracking CSV / COCO and ``PoseDataset``."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from annolid.annotation.keypoint_visibility import (
    KeypointVisibility,
    visibility_from_labelme_shape,
)
from annolid.annotation.pose_interop.model import (
    PoseDataset,
    PoseFrame,
    PoseInstance,
    PosePoint,
    dedup_pairs,
    dedup_preserve_order,
    is_finite_number,
)
from annolid.annotation.pose_schema import PoseSchema
from annolid.utils.annotation_compat import __version__ as LABELME_VERSION
from annolid.utils.annotation_store import AnnotationStore, load_labelme_json
from annolid.utils.atomic_io import atomic_write_json
from annolid.utils.body_part_shapes import (
    extract_body_part_label,
    extract_body_part_parent_instance,
    split_body_part_csv_instance_name,
)
from annolid.utils.zone_shapes import is_zone_shape_payload

_NON_LABELME_JSON_NAMES = frozenset(
    {
        "pose_schema.json",
        "pose_schema_user.json",
        "project.annolid.json",
        "manifest.json",
        "cohort_manifest.json",
    }
)
_FRAME_NUM_RE = re.compile(r"(\d+)$")
_IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def _extract_frame_index_from_stem(stem: str) -> Optional[int]:
    match = _FRAME_NUM_RE.search(stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _is_non_frame_metadata_json(path: Path) -> bool:
    lower = path.name.lower()
    if lower in _NON_LABELME_JSON_NAMES:
        return True
    if lower.startswith("pose_schema") and lower.endswith(".json"):
        return True
    if lower.endswith(("_zones.json", "_tracking_stats.json", "_gap_report.json")):
        return True
    return False


def _find_sidecar_image(json_path: Path, payload: Mapping[str, Any]) -> Optional[Path]:
    image_path_raw = payload.get("imagePath")
    if isinstance(image_path_raw, str) and image_path_raw.strip():
        cand = Path(image_path_raw.strip())
        if not cand.is_absolute():
            cand = json_path.parent / cand
        if cand.is_file():
            return cand
    base = json_path.with_suffix("")
    for ext in _IMAGE_EXTS:
        cand = base.with_suffix(ext)
        if cand.is_file():
            return cand
    return None


def _extract_shape_score(shape: Mapping[str, Any]) -> Optional[float]:
    for key in ("score", "class_score", "confidence", "likelihood"):
        val = shape.get(key)
        if is_finite_number(val):
            return float(val)
    return None


def _resolve_point_instance_and_keypoint(
    shape: Mapping[str, Any],
    *,
    payload_instance: str,
    pose_schema: Optional[PoseSchema],
) -> Tuple[str, str, Optional[int]]:
    """Resolve ``(instance_name, keypoint_name, group_id)`` from a LabelMe point shape."""
    raw_gid = shape.get("group_id")
    gid: Optional[int] = None
    if raw_gid is not None and not isinstance(raw_gid, bool):
        try:
            gid = int(raw_gid)
        except (TypeError, ValueError):
            gid = None

    flags = shape.get("flags") if isinstance(shape.get("flags"), Mapping) else {}
    raw_label = str(shape.get("label") or "").strip()

    explicit_instance = ""
    for key in ("instance_label", "parent_instance", "instance_name"):
        val = shape.get(key)
        if isinstance(val, str) and val.strip():
            explicit_instance = val.strip()
            break
        fval = flags.get(key)
        if isinstance(fval, str) and fval.strip():
            explicit_instance = fval.strip()
            break

    explicit_kp = ""
    for key in ("display_label", "part_label", "body_part"):
        val = shape.get(key)
        if isinstance(val, str) and val.strip():
            explicit_kp = val.strip()
            break
        fval = flags.get(key)
        if isinstance(fval, str) and fval.strip():
            explicit_kp = fval.strip()
            break

    if explicit_instance and explicit_kp:
        return explicit_instance, explicit_kp, gid

    # Check PoseSchema instance prefix stripping (e.g., "mouse1_nose" -> ("mouse1", "nose"))
    if pose_schema is not None and raw_label:
        stripped_inst, stripped_kp = pose_schema.strip_instance_prefix(raw_label)
        if stripped_inst and stripped_kp:
            return explicit_instance or stripped_inst, explicit_kp or stripped_kp, gid

    # Check hierarchical ":" separator (e.g., "mouse_1:nose")
    if ":" in raw_label:
        prefix, rest = raw_label.split(":", 1)
        if prefix.strip() and rest.strip():
            return explicit_instance or prefix.strip(), explicit_kp or rest.strip(), gid

    # Check explicit_instance prefix inside raw_label (e.g., "mouse_nose" with instance="mouse")
    inst_hint = explicit_instance or payload_instance
    if inst_hint and raw_label and not explicit_kp:
        if raw_label.lower().startswith(inst_hint.lower()) and len(raw_label) > len(
            inst_hint
        ):
            suffix = raw_label[len(inst_hint) :]
            if suffix and suffix[0] in ("_", "-", ":", "|", " "):
                trimmed = suffix.lstrip("_-:| ").strip()
                if trimmed:
                    return inst_hint, trimmed, gid

    # Fallback: use group_id or payload_instance or body_part helper
    resolved_inst = explicit_instance
    if not resolved_inst:
        bp_parent = extract_body_part_parent_instance(shape)
        if bp_parent:
            resolved_inst = bp_parent
    if not resolved_inst and gid is not None:
        resolved_inst = f"instance_{gid}"
    if not resolved_inst and payload_instance:
        resolved_inst = payload_instance
    if not resolved_inst:
        resolved_inst = "instance_0"

    resolved_kp = explicit_kp or extract_body_part_label(shape) or raw_label
    return resolved_inst, resolved_kp, gid


def _parse_labelme_payload_to_frame(
    payload: Mapping[str, Any],
    *,
    frame_index: int,
    pose_schema: Optional[PoseSchema],
    source_json_path: Optional[Path] = None,
    video_name: str = "",
) -> PoseFrame:
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        shapes = []

    payload_instance = str(payload.get("instance_label") or "").strip()
    if not payload_instance and isinstance(payload.get("annolid_flags_meta"), Mapping):
        payload_instance = str(
            payload["annolid_flags_meta"].get("instance_label") or ""
        ).strip()

    # Group point shapes by (instance_name, group_id) while preserving order
    grouped: Dict[Tuple[str, Optional[int]], PoseInstance] = {}
    instance_order: List[Tuple[str, Optional[int]]] = []

    for shape in shapes:
        if not isinstance(shape, Mapping):
            continue
        if is_zone_shape_payload(shape):
            continue
        if str(shape.get("shape_type") or "").strip().lower() != "point":
            continue
        points = shape.get("points")
        if not isinstance(points, list) or not points:
            continue
        pt0 = points[0]
        if not isinstance(pt0, (list, tuple)) or len(pt0) < 2:
            continue
        if not (is_finite_number(pt0[0]) and is_finite_number(pt0[1])):
            continue

        inst_name, kp_name, gid = _resolve_point_instance_and_keypoint(
            shape,
            payload_instance=payload_instance,
            pose_schema=pose_schema,
        )
        if not kp_name:
            continue

        vis = visibility_from_labelme_shape(shape)
        vis_code = int(vis) if vis is not None else int(KeypointVisibility.VISIBLE)
        if vis_code <= int(KeypointVisibility.MISSING):
            continue

        score = _extract_shape_score(shape)
        flags = shape.get("flags") if isinstance(shape.get("flags"), Mapping) else {}
        is_pred = bool(flags.get("predicted", False))

        key = (inst_name, gid)
        if key not in grouped:
            grouped[key] = PoseInstance(
                instance_name=inst_name,
                keypoints={},
                track_id=gid,
                is_predicted=is_pred,
            )
            instance_order.append(key)
        inst_obj = grouped[key]
        inst_obj.keypoints[kp_name] = PosePoint(
            x=float(pt0[0]),
            y=float(pt0[1]),
            score=score,
            visibility=vis_code,
        )
        if is_pred:
            inst_obj.is_predicted = True

    instances = [grouped[k] for k in instance_order if grouped[k].keypoints]
    for inst_obj in instances:
        scores = [
            pt.score
            for pt in inst_obj.keypoints.values()
            if pt.score is not None and math.isfinite(pt.score)
        ]
        if scores and inst_obj.score is None:
            inst_obj.score = sum(scores) / len(scores)

    img_w = payload.get("imageWidth")
    img_h = payload.get("imageHeight")
    src_img = (
        _find_sidecar_image(source_json_path, payload)
        if source_json_path is not None
        else None
    )

    return PoseFrame(
        frame_index=int(frame_index),
        instances=instances,
        image_path=str(payload.get("imagePath") or (src_img.name if src_img else "")),
        image_width=int(img_w) if is_finite_number(img_w) and int(img_w) > 0 else None,
        image_height=int(img_h) if is_finite_number(img_h) and int(img_h) > 0 else None,
        video_index=0,
        video_name=video_name,
        source_json_path=str(source_json_path) if source_json_path else None,
        source_image_path=str(src_img) if src_img else None,
    )


def read_pose_from_labelme_dir(
    source_dir: str | Path,
    *,
    pose_schema: Optional[PoseSchema | str | Path] = None,
    recursive: bool = False,
    include_annotation_store: bool = True,
) -> PoseDataset:
    """Read keypoint pose annotations from a LabelMe JSON directory (and AnnotationStore)."""
    root = Path(source_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"LabelMe directory not found: {root}")
    if root.is_file() and root.suffix.lower() == ".json":
        payload = load_labelme_json(root)
        schema_obj = (
            PoseSchema.load(pose_schema)
            if isinstance(pose_schema, (str, Path))
            else pose_schema
        )
        if schema_obj is None:
            cand_schema = root.parent / "pose_schema.json"
            if cand_schema.is_file():
                try:
                    schema_obj = PoseSchema.load(cand_schema)
                except Exception:
                    schema_obj = None
        fidx = _extract_frame_index_from_stem(root.stem) or 0
        frame = _parse_labelme_payload_to_frame(
            payload,
            frame_index=fidx,
            pose_schema=schema_obj,
            source_json_path=root,
            video_name=root.parent.name,
        )
        return PoseDataset(
            frames=[frame],
            keypoints=list(schema_obj.keypoints) if schema_obj else [],
            edges=list(schema_obj.edges) if schema_obj else [],
            symmetry_pairs=list(schema_obj.symmetry_pairs) if schema_obj else [],
            instances=list(schema_obj.instances) if schema_obj else [],
            source_format="labelme_json",
            metadata={"source_path": str(root)},
        )

    schema_obj: Optional[PoseSchema]
    if isinstance(pose_schema, PoseSchema):
        schema_obj = pose_schema
    elif isinstance(pose_schema, (str, Path)):
        schema_obj = PoseSchema.load(pose_schema)
    else:
        schema_obj = None
        for candidate in (
            root / "pose_schema.json",
            root / "pose_schema.yaml",
            root.parent / "pose_schema.json",
        ):
            if candidate.is_file():
                try:
                    schema_obj = PoseSchema.load(candidate)
                    break
                except Exception:
                    continue

    # Collect per-frame payloads from AnnotationStore(s) first, then override with manual JSONs.
    frame_payloads: Dict[int, Tuple[Mapping[str, Any], Optional[Path]]] = {}
    ordered_non_numeric: List[Tuple[Path, Mapping[str, Any]]] = []

    if include_annotation_store:
        store_files = sorted(
            root.rglob("*_annotations.ndjson")
            if recursive
            else root.glob("*_annotations.ndjson")
        )
        for store_path in store_files:
            store = AnnotationStore(store_path)
            for fidx in sorted(store.iter_frames()):
                rec = store.get_frame(fidx)
                if isinstance(rec, Mapping):
                    frame_payloads[int(fidx)] = (rec, None)

    json_iter = root.rglob("*.json") if recursive else root.glob("*.json")
    for json_path in sorted(json_iter):
        if not json_path.is_file() or _is_non_frame_metadata_json(json_path):
            continue
        try:
            payload = load_labelme_json(json_path)
        except Exception:
            continue
        if not isinstance(payload, Mapping) or "shapes" not in payload:
            continue
        fidx = _extract_frame_index_from_stem(json_path.stem)
        if fidx is not None and not recursive:
            frame_payloads[fidx] = (payload, json_path)
        else:
            ordered_non_numeric.append((json_path, payload))

    frames: List[PoseFrame] = []
    for fidx in sorted(frame_payloads.keys()):
        payload, jpath = frame_payloads[fidx]
        frame = _parse_labelme_payload_to_frame(
            payload,
            frame_index=fidx,
            pose_schema=schema_obj,
            source_json_path=jpath,
            video_name=root.name,
        )
        if frame.instances:
            frames.append(frame)

    next_idx = (max(frame_payloads.keys()) + 1) if frame_payloads else 0
    for jpath, payload in ordered_non_numeric:
        frame = _parse_labelme_payload_to_frame(
            payload,
            frame_index=next_idx,
            pose_schema=schema_obj,
            source_json_path=jpath,
            video_name=jpath.parent.name,
        )
        next_idx += 1
        if frame.instances:
            frames.append(frame)

    dataset = PoseDataset(
        frames=frames,
        keypoints=list(schema_obj.keypoints) if schema_obj else [],
        edges=list(schema_obj.edges) if schema_obj else [],
        symmetry_pairs=list(schema_obj.symmetry_pairs) if schema_obj else [],
        instances=list(schema_obj.instances) if schema_obj else [],
        scorer="annolid",
        source_format="labelme_dir",
        metadata={"source_path": str(root)},
    )
    dataset.keypoints = dataset.resolved_keypoints()
    dataset.instances = dataset.resolved_instances()
    dataset.multi_animal = dataset.is_multi_animal()
    return dataset


def read_pose_from_tracking_csv(
    csv_path: str | Path,
    *,
    pose_schema: Optional[PoseSchema | str | Path] = None,
    default_instance: str = "instance_0",
) -> PoseDataset:
    """Read keypoint trajectories from an Annolid ``*_tracking.csv`` or ``*_tracked.csv``."""
    path = Path(csv_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Tracking CSV not found: {path}")

    schema_obj: Optional[PoseSchema]
    if isinstance(pose_schema, PoseSchema):
        schema_obj = pose_schema
    elif isinstance(pose_schema, (str, Path)):
        schema_obj = PoseSchema.load(pose_schema)
    else:
        schema_obj = None

    by_frame: Dict[int, Dict[str, PoseInstance]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"Empty tracking CSV: {path}")
        for row in reader:
            raw_frame = row.get("frame_number")
            raw_name = str(row.get("instance_name") or "").strip()
            raw_cx = row.get("cx")
            raw_cy = row.get("cy")
            if not raw_name or not (
                is_finite_number(raw_frame)
                and is_finite_number(raw_cx)
                and is_finite_number(raw_cy)
            ):
                continue

            fidx = int(float(raw_frame))
            x_val = float(raw_cx)
            y_val = float(raw_cy)
            raw_score = row.get("class_score")
            score = float(raw_score) if is_finite_number(raw_score) else None

            raw_tid = row.get("tracking_id")
            tid: Optional[int] = None
            if is_finite_number(raw_tid):
                tid = int(float(raw_tid))

            inst_name = ""
            kp_name = ""
            if ":" in raw_name:
                parent, part = split_body_part_csv_instance_name(raw_name)
                if parent and part:
                    inst_name, kp_name = parent, part
            if not kp_name and schema_obj is not None:
                stripped_inst, stripped_kp = schema_obj.strip_instance_prefix(raw_name)
                if stripped_inst and stripped_kp:
                    inst_name, kp_name = stripped_inst, stripped_kp
            if not kp_name:
                inst_name = f"track_{tid}" if tid is not None else default_instance
                kp_name = raw_name

            frame_map = by_frame.setdefault(fidx, {})
            inst_obj = frame_map.get(inst_name)
            if inst_obj is None:
                inst_obj = PoseInstance(
                    instance_name=inst_name,
                    keypoints={},
                    track_id=tid,
                    is_predicted=True,
                )
                frame_map[inst_name] = inst_obj
            inst_obj.keypoints[kp_name] = PosePoint(
                x=x_val,
                y=y_val,
                score=score,
                visibility=int(KeypointVisibility.VISIBLE),
            )

    frames = [
        PoseFrame(
            frame_index=fidx,
            instances=list(by_frame[fidx].values()),
            video_name=path.stem,
        )
        for fidx in sorted(by_frame.keys())
    ]

    dataset = PoseDataset(
        frames=frames,
        keypoints=list(schema_obj.keypoints) if schema_obj else [],
        edges=list(schema_obj.edges) if schema_obj else [],
        symmetry_pairs=list(schema_obj.symmetry_pairs) if schema_obj else [],
        instances=list(schema_obj.instances) if schema_obj else [],
        scorer="annolid",
        source_format="annolid_tracking_csv",
        metadata={"source_path": str(path)},
    )
    dataset.keypoints = dataset.resolved_keypoints()
    dataset.instances = dataset.resolved_instances()
    dataset.multi_animal = dataset.is_multi_animal()
    return dataset


def read_pose_from_coco_json(coco_path: str | Path) -> PoseDataset:
    """Read COCO keypoint (or bbox-center fallback) annotations into a ``PoseDataset``."""
    path = Path(coco_path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Invalid COCO JSON payload: {path}")

    images = {
        int(img["id"]): img
        for img in (payload.get("images") or [])
        if isinstance(img, dict) and "id" in img
    }
    categories = {
        int(cat["id"]): cat
        for cat in (payload.get("categories") or [])
        if isinstance(cat, dict) and "id" in cat
    }

    all_kp_names: List[str] = []
    all_edges: List[Tuple[str, str]] = []
    for cat in categories.values():
        kps = cat.get("keypoints")
        if isinstance(kps, list) and kps:
            cat_kps = [str(k).strip() for k in kps if str(k).strip()]
            all_kp_names.extend(cat_kps)
            skel = cat.get("skeleton")
            if isinstance(skel, list):
                for pair in skel:
                    if isinstance(pair, (list, tuple)) and len(pair) == 2:
                        try:
                            ia, ib = int(pair[0]) - 1, int(pair[1]) - 1
                        except (TypeError, ValueError):
                            continue
                        if 0 <= ia < len(cat_kps) and 0 <= ib < len(cat_kps):
                            all_edges.append((cat_kps[ia], cat_kps[ib]))

    anns_by_img: Dict[int, List[Dict[str, Any]]] = {}
    for ann in payload.get("annotations") or []:
        if not isinstance(ann, dict) or "image_id" not in ann:
            continue
        anns_by_img.setdefault(int(ann["image_id"]), []).append(ann)

    frames: List[PoseFrame] = []
    img_ids = sorted(set(images.keys()) | set(anns_by_img.keys()))
    for order_idx, img_id in enumerate(img_ids):
        img_meta = images.get(img_id, {})
        file_name = str(img_meta.get("file_name") or "")
        frame_idx = (
            _extract_frame_index_from_stem(Path(file_name).stem) if file_name else None
        )
        if frame_idx is None:
            frame_idx = int(img_id)

        anns = anns_by_img.get(img_id, [])
        frame_instances: List[PoseInstance] = []
        per_cat_count: Dict[str, int] = {}

        for ann in anns:
            cat_id = int(ann.get("category_id", 0))
            cat = categories.get(cat_id, {})
            cat_name = str(cat.get("name") or f"class_{cat_id}").strip()
            cat_kps = (
                [str(k).strip() for k in cat.get("keypoints", [])]
                if isinstance(cat.get("keypoints"), list)
                else []
            )
            track_id_raw = ann.get("track_id", ann.get("instance_id"))
            if track_id_raw is not None and is_finite_number(track_id_raw):
                tid = int(track_id_raw)
                inst_name = f"{cat_name}_{tid}"
            else:
                idx_for_cat = per_cat_count.get(cat_name, 0)
                per_cat_count[cat_name] = idx_for_cat + 1
                tid = idx_for_cat
                inst_name = f"{cat_name}_{idx_for_cat}" if len(anns) > 1 else cat_name

            raw_kps = ann.get("keypoints")
            kps: Dict[str, PosePoint] = {}
            if isinstance(raw_kps, list) and len(raw_kps) >= 3:
                n_kps = len(raw_kps) // 3
                for k_i in range(n_kps):
                    x = raw_kps[3 * k_i]
                    y = raw_kps[3 * k_i + 1]
                    v = raw_kps[3 * k_i + 2]
                    if not (
                        is_finite_number(x)
                        and is_finite_number(y)
                        and is_finite_number(v)
                    ):
                        continue
                    vf = float(v)
                    if vf <= 0:
                        continue
                    vis_code = (
                        int(KeypointVisibility.OCCLUDED)
                        if int(round(vf)) == 1
                        else int(KeypointVisibility.VISIBLE)
                    )
                    kp_label = cat_kps[k_i] if k_i < len(cat_kps) else f"kp_{k_i}"
                    kps[kp_label] = PosePoint(
                        x=float(x),
                        y=float(y),
                        score=float(ann["score"])
                        if is_finite_number(ann.get("score"))
                        else 1.0,
                        visibility=vis_code,
                    )
            elif isinstance(ann.get("bbox"), list) and len(ann["bbox"]) >= 4:
                bx, by, bw, bh = ann["bbox"][:4]
                if all(is_finite_number(v) for v in (bx, by, bw, bh)):
                    cx = float(bx) + float(bw) / 2.0
                    cy = float(by) + float(bh) / 2.0
                    kps[cat_name] = PosePoint(
                        x=cx,
                        y=cy,
                        score=float(ann["score"])
                        if is_finite_number(ann.get("score"))
                        else 1.0,
                        visibility=int(KeypointVisibility.VISIBLE),
                    )
                    inst_name = "centroid"

            if kps:
                frame_instances.append(
                    PoseInstance(
                        instance_name=inst_name,
                        keypoints=kps,
                        track_id=tid,
                        score=float(ann["score"])
                        if is_finite_number(ann.get("score"))
                        else None,
                    )
                )

        if frame_instances:
            frames.append(
                PoseFrame(
                    frame_index=int(frame_idx),
                    instances=frame_instances,
                    image_path=file_name,
                    image_width=int(img_meta["width"])
                    if is_finite_number(img_meta.get("width"))
                    else None,
                    image_height=int(img_meta["height"])
                    if is_finite_number(img_meta.get("height"))
                    else None,
                    video_name=path.stem,
                )
            )

    dataset = PoseDataset(
        frames=frames,
        keypoints=dedup_preserve_order(all_kp_names),
        edges=dedup_pairs(all_edges),
        scorer="coco",
        source_format="coco_json",
        metadata={"source_path": str(path)},
    )
    dataset.keypoints = dataset.resolved_keypoints()
    dataset.instances = dataset.resolved_instances()
    dataset.multi_animal = dataset.is_multi_animal()
    return dataset


def build_label_mapper(
    source_names: Sequence[str],
    pose_schema: Optional[PoseSchema],
) -> Dict[str, str]:
    """Map source keypoint names to canonical ``PoseSchema`` names (case-insensitive)."""
    mapper: Dict[str, str] = {}
    if pose_schema is None or not getattr(pose_schema, "keypoints", None):
        return {name: name for name in source_names}
    try:
        pose_schema.normalize_prefixed_keypoints()
    except Exception:
        pass
    canonical = {
        str(kp).strip().lower(): str(kp).strip() for kp in pose_schema.keypoints
    }
    for name in source_names:
        mapper[name] = canonical.get(str(name).strip().lower(), name)
    return mapper


def build_labelme_point_shape(
    *,
    keypoint_label: str,
    x: float,
    y: float,
    instance_label: Optional[str] = None,
    group_id: Optional[int] = None,
    visibility: int = int(KeypointVisibility.VISIBLE),
    score: Optional[float] = None,
    is_predicted: bool = False,
    prefix_with_instance: bool = False,
    instance_separator: str = "_",
) -> Dict[str, Any]:
    """Construct a LabelMe-safe point shape dict (boolean-only ``flags``)."""
    kp_clean = str(keypoint_label).strip()
    inst_clean = str(instance_label or "").strip()
    sep = str(instance_separator or "_")
    if prefix_with_instance and inst_clean:
        label = f"{inst_clean}{sep}{kp_clean}"
    else:
        label = kp_clean

    vis_int = (
        int(KeypointVisibility.OCCLUDED)
        if int(visibility) == int(KeypointVisibility.OCCLUDED)
        else int(KeypointVisibility.VISIBLE)
    )
    flags: Dict[str, bool] = {
        "kp_visible": bool(vis_int == int(KeypointVisibility.VISIBLE)),
    }
    if is_predicted:
        flags["predicted"] = True

    shape: Dict[str, Any] = {
        "label": label,
        "points": [[float(x), float(y)]],
        "group_id": int(group_id)
        if group_id is not None and int(group_id) >= 0
        else None,
        "shape_type": "point",
        "flags": flags,
        "visible": True,
        "kp_visibility": vis_int,
    }
    if inst_clean:
        shape["instance_label"] = inst_clean
        shape["display_label"] = kp_clean
    if score is not None and math.isfinite(float(score)):
        shape["score"] = round(float(score), 6)
    return shape


def write_pose_to_labelme_dir(
    dataset: PoseDataset,
    out_dir: str | Path,
    *,
    overwrite: bool = True,
    write_pose_schema: bool = True,
    pose_schema_out: Optional[str | Path] = None,
    pose_schema_override: Optional[PoseSchema | str | Path] = None,
    pose_schema_preset: Optional[str] = "auto",
    filename_prefix: Optional[str] = None,
    prefix_labels_with_instance: bool = False,
    instance_separator: str = "_",
    include_occluded: bool = True,
    frame_to_image: Optional[Dict[Tuple[int, int], str]] = None,
    frame_to_wh: Optional[Dict[Tuple[int, int], Tuple[int, int]]] = None,
    prefix_multi_video: bool = True,
) -> Dict[str, Any]:
    """Write a ``PoseDataset`` to LabelMe JSON files (+ optional ``pose_schema.json``)."""
    out_root = Path(out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    user_schema: Optional[PoseSchema]
    if isinstance(pose_schema_override, PoseSchema):
        user_schema = pose_schema_override
    elif isinstance(pose_schema_override, (str, Path)):
        user_schema = PoseSchema.load(pose_schema_override)
    else:
        user_schema = None

    label_map = build_label_mapper(dataset.resolved_keypoints(), user_schema)
    instances_order = dataset.resolved_instances()
    inst_to_gid = {name: idx for idx, name in enumerate(instances_order)}
    is_multi = dataset.is_multi_animal()

    used_video_indices = sorted({int(f.video_index) for f in dataset.frames})
    multi_video = prefix_multi_video and len(used_video_indices) > 1

    written = 0
    skipped = 0
    total_points = 0
    written_files: List[str] = []

    for frame in sorted(dataset.frames, key=lambda f: (f.video_index, f.frame_index)):
        key = (int(frame.video_index), int(frame.frame_index))
        shapes: List[Dict[str, Any]] = []

        for local_idx, inst in enumerate(frame.instances):
            inst_name = str(inst.instance_name or "").strip()
            if inst.track_id is not None and int(inst.track_id) >= 0:
                gid: Optional[int] = int(inst.track_id)
            elif is_multi:
                gid = inst_to_gid.get(inst_name, local_idx)
            else:
                gid = None

            for kp_name, pt in inst.keypoints.items():
                if not pt.is_valid:
                    continue
                if not include_occluded and int(pt.visibility) == int(
                    KeypointVisibility.OCCLUDED
                ):
                    continue
                mapped_kp = label_map.get(kp_name, kp_name)
                shapes.append(
                    build_labelme_point_shape(
                        keypoint_label=mapped_kp,
                        x=pt.x,
                        y=pt.y,
                        instance_label=inst_name or None,
                        group_id=gid,
                        visibility=pt.visibility,
                        score=pt.score,
                        is_predicted=inst.is_predicted,
                        prefix_with_instance=prefix_labels_with_instance,
                        instance_separator=instance_separator,
                    )
                )
                total_points += 1

        if filename_prefix:
            json_name = f"{filename_prefix}_{int(frame.frame_index):09d}.json"
        elif multi_video:
            vprefix = frame.video_name or f"video{int(frame.video_index)}"
            json_name = f"{vprefix}_{int(frame.frame_index):09d}.json"
        else:
            json_name = f"{int(frame.frame_index):09d}.json"

        out_json = out_root / json_name
        if out_json.exists() and not overwrite:
            skipped += 1
            continue

        img_rel = (
            frame_to_image.get(key, "")
            if frame_to_image is not None
            else (frame.image_path or "")
        )
        if frame_to_wh is not None and key in frame_to_wh:
            w, h = frame_to_wh[key]
        else:
            w = int(frame.image_width or 0)
            h = int(frame.image_height or 0)

        payload: Dict[str, Any] = {
            "version": LABELME_VERSION,
            "flags": {},
            "shapes": shapes,
            "imagePath": img_rel,
            "imageData": None,
            "imageHeight": int(h),
            "imageWidth": int(w),
        }
        if not is_multi and len(instances_order) == 1:
            payload["instance_label"] = instances_order[0]

        atomic_write_json(out_json, payload, indent=2)
        written += 1
        written_files.append(str(out_json))

    schema_written: Optional[str] = None
    if write_pose_schema and dataset.resolved_keypoints():
        schema = dataset.to_pose_schema(
            preset=pose_schema_preset,
            instance_separator=instance_separator,
        )
        schema_target = (
            Path(pose_schema_out).expanduser()
            if pose_schema_out is not None
            else (out_root / "pose_schema.json")
        )
        if not schema_target.is_absolute():
            schema_target = out_root / schema_target
        schema.save(schema_target)
        schema_written = str(schema_target.resolve())

    if user_schema is not None:
        try:
            user_schema.normalize_prefixed_keypoints()
        except Exception:
            pass
        user_schema.save(out_root / "pose_schema_user.json")

    return {
        "output_dir": str(out_root),
        "source_format": dataset.source_format,
        "frames_total": len(dataset.frames),
        "json_written": written,
        "json_skipped_existing": skipped,
        "points_written": total_points,
        "multi_animal": is_multi,
        "keypoints": dataset.resolved_keypoints(),
        "instances": instances_order,
        "pose_schema": schema_written,
        "warnings": list(dataset.warnings),
    }
