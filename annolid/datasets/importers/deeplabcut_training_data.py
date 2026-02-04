from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
from annolid.utils.annotation_compat import __version__ as LABELME_VERSION
from PIL import Image, ImageOps

from annolid.annotation.pose_schema import PoseSchema


@dataclass(frozen=True)
class DeepLabCutTrainingImportConfig:
    """Import DeepLabCut `labeled-data/**/CollectedData_*.csv` into LabelMe JSON."""

    source_dir: Path
    labeled_data_root: Path = Path("labeled-data")
    instance_label: str = "mouse"
    overwrite: bool = False
    recursive: bool = True


def _detect_dlc_header_levels(csv_path: Path) -> int:
    lines = csv_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(lines) < 3:
        return 3
    first = lines[0].split(",", 1)[0].strip().lower()
    second = lines[1].split(",", 1)[0].strip().lower() if len(lines) > 1 else ""
    third = lines[2].split(",", 1)[0].strip().lower() if len(lines) > 2 else ""
    fourth = lines[3].split(",", 1)[0].strip().lower() if len(lines) > 3 else ""

    if (
        first == "scorer"
        and second in ("individuals", "animals")
        and third == "bodyparts"
        and fourth == "coords"
    ):
        return 4
    if first == "scorer" and second == "bodyparts" and third == "coords":
        return 3
    return 3


def _iter_collected_data_csvs(root: Path, *, recursive: bool) -> Iterable[Path]:
    pattern = "CollectedData_*.csv"
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def _is_number(value: Any) -> bool:
    try:
        return (
            value is not None
            and not (isinstance(value, float) and math.isnan(value))
            and float(value) == float(value)
        )
    except Exception:
        return False


def _infer_mouse_edges(keypoints: Sequence[str]) -> List[Tuple[str, str]]:
    kp = {str(name).strip() for name in keypoints if str(name).strip()}

    def add(edges: List[Tuple[str, str]], a: str, b: str) -> None:
        if a in kp and b in kp and a != b:
            edges.append((a, b))

    edges: List[Tuple[str, str]] = []
    add(edges, "nose", "neck")
    add(edges, "neck", "back")
    add(edges, "back", "root_tail")
    add(edges, "root_tail", "mid_tail")
    add(edges, "mid_tail", "tip_tail")

    add(edges, "left_ear", "nose")
    add(edges, "right_ear", "nose")

    add(edges, "left_front_limb", "neck")
    add(edges, "right_front_limb", "neck")
    add(edges, "left_hind_limb", "back")
    add(edges, "right_hind_limb", "back")

    add(edges, "left_front_claw", "left_front_limb")
    add(edges, "right_front_claw", "right_front_limb")
    add(edges, "left_hind_claw", "left_hind_limb")
    add(edges, "right_hind_claw", "right_hind_limb")

    return list(dict.fromkeys(edges))


def build_pose_schema_from_keypoints(
    keypoints: Sequence[str],
    *,
    instances: Optional[Sequence[str]] = None,
    instance_separator: str = "_",
    preset: Optional[str] = "mouse",
) -> PoseSchema:
    base_keypoints = [str(k).strip() for k in keypoints if str(k).strip()]
    symmetry_pairs = PoseSchema.infer_symmetry_pairs(base_keypoints)
    edges: List[Tuple[str, str]] = []
    if preset and str(preset).lower() == "mouse":
        edges = _infer_mouse_edges(base_keypoints)
    return PoseSchema(
        keypoints=base_keypoints,
        edges=edges,
        symmetry_pairs=symmetry_pairs,
        instances=[str(i).strip() for i in (instances or []) if str(i).strip()],
        instance_separator=str(instance_separator or "_"),
    )


def _resolve_image_path(
    *,
    source_dir: Path,
    labeled_root: Path,
    csv_dir: Path,
    path_parts: Sequence[str],
) -> Optional[Path]:
    parts = [str(p).strip() for p in path_parts if str(p).strip()]
    candidates: List[Path] = []

    if parts:
        joined = Path(*parts)
        candidates.append(source_dir / joined)
        candidates.append(labeled_root / joined)
        if len(parts) >= 2:
            candidates.append(labeled_root / Path(*parts[1:]))
        candidates.append(csv_dir / parts[-1])
        if len(parts) >= 2:
            candidates.append(csv_dir / parts[-2] / parts[-1])
    for candidate in candidates:
        try:
            candidate = candidate.expanduser()
            if candidate.exists():
                return candidate.resolve()
        except Exception:
            continue
    return None


def _dlc_keypoint_columns(columns: pd.MultiIndex) -> Tuple[List[Tuple[Any, ...]], bool]:
    """Return keypoint column tuples and whether this is multi-animal (4-level)."""
    nlevels = columns.nlevels
    if nlevels not in (3, 4):
        return [], False
    coord_level = nlevels - 1
    kp_cols: List[Tuple[Any, ...]] = []
    for col in columns:
        coord = str(col[coord_level]).strip().lower()
        if coord not in ("x", "y", "likelihood"):
            continue
        bodypart = str(col[coord_level - 1]).strip()
        if not bodypart or bodypart.lower().startswith("unnamed:"):
            continue
        scorer = str(col[0]).strip()
        if (
            not scorer
            or scorer.lower().startswith("unnamed:")
            or scorer.lower() == "scorer"
        ):
            continue
        kp_cols.append(tuple(col))
    return kp_cols, (nlevels == 4)


def _extract_keypoints_from_row(
    row: pd.Series,
    *,
    kp_cols: Sequence[Tuple[Any, ...]],
    multi_animal: bool,
    default_instance: str,
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Return (shapes, instances, base_keypoints)."""
    shapes: List[Dict[str, Any]] = []
    instances: List[str] = []
    base_keypoints: List[str] = []

    # Index columns for multi-index: (..., 'x'|'y'|'likelihood')
    # single: (scorer, bodypart, coord)
    # multi: (scorer, individual, bodypart, coord)
    by_key: Dict[Tuple[str, str], Dict[str, float]] = {}

    for col in kp_cols:
        coord = str(col[-1]).strip().lower()
        if multi_animal:
            individual = str(col[1]).strip()
            bodypart = str(col[2]).strip()
        else:
            individual = default_instance
            bodypart = str(col[1]).strip()

        if not bodypart or bodypart.lower().startswith("unnamed:"):
            continue

        value = row.get(col)
        if not _is_number(value):
            continue
        entry = by_key.setdefault((individual, bodypart), {})
        entry[coord] = float(value)

    for (individual, bodypart), values in by_key.items():
        if "x" not in values or "y" not in values:
            continue
        instance_label = individual or default_instance
        if instance_label and instance_label not in instances:
            instances.append(instance_label)
        if bodypart and bodypart not in base_keypoints:
            base_keypoints.append(bodypart)
        shapes.append(
            {
                "label": f"{instance_label}_{bodypart}" if instance_label else bodypart,
                "points": [[values["x"], values["y"]]],
                "shape_type": "point",
                "flags": {
                    "instance_label": str(instance_label),
                    "display_label": str(bodypart),
                },
                "visible": True,
            }
        )

    return shapes, instances, base_keypoints


def import_deeplabcut_training_data(
    cfg: DeepLabCutTrainingImportConfig,
    *,
    write_pose_schema: bool = False,
    pose_schema_out: Optional[Path] = None,
    pose_schema_preset: str = "mouse",
    instance_separator: str = "_",
) -> Dict[str, Any]:
    source_dir = Path(cfg.source_dir).expanduser().resolve()
    labeled_root = Path(cfg.labeled_data_root)
    labeled_root = (
        labeled_root if labeled_root.is_absolute() else (source_dir / labeled_root)
    )
    if not labeled_root.exists():
        raise FileNotFoundError(f"labeled-data root not found: {labeled_root}")

    collected_csvs = [
        p
        for p in _iter_collected_data_csvs(labeled_root, recursive=cfg.recursive)
        if p.is_file()
    ]
    written = 0
    skipped_existing = 0
    missing_images = 0
    errors = 0
    keypoints_seen: List[str] = []
    instances_seen: List[str] = []

    for csv_path in sorted(collected_csvs):
        try:
            header_levels = _detect_dlc_header_levels(csv_path)
            df = pd.read_csv(csv_path, header=list(range(header_levels)))
        except Exception:
            errors += 1
            continue

        if not isinstance(df.columns, pd.MultiIndex):
            errors += 1
            continue

        kp_cols, multi_animal = _dlc_keypoint_columns(df.columns)
        if not kp_cols:
            continue

        # Identify path columns: those whose last level isn't a coordinate.
        coord_set = {"x", "y", "likelihood"}
        path_cols = [
            col for col in df.columns if str(col[-1]).strip().lower() not in coord_set
        ]

        for _, row in df.iterrows():
            try:
                path_parts = []
                if path_cols:
                    for col in path_cols[:3]:
                        value = row.get(col)
                        if value is None:
                            continue
                        text = str(value).strip()
                        if text and text.lower() != "nan":
                            path_parts.append(text)
                else:
                    idx = row.name
                    if isinstance(idx, str) and idx.strip():
                        path_parts.append(idx.strip())

                image_path = _resolve_image_path(
                    source_dir=source_dir,
                    labeled_root=labeled_root,
                    csv_dir=csv_path.parent,
                    path_parts=path_parts,
                )
                if image_path is None or not image_path.exists():
                    missing_images += 1
                    continue
                out_json = image_path.with_suffix(".json")

                shapes, instances, keypoints = _extract_keypoints_from_row(
                    row,
                    kp_cols=kp_cols,
                    multi_animal=multi_animal,
                    default_instance=str(cfg.instance_label),
                )
                for kp in keypoints:
                    if kp not in keypoints_seen:
                        keypoints_seen.append(kp)
                for inst in instances:
                    if inst not in instances_seen:
                        instances_seen.append(inst)

                if out_json.exists() and not bool(cfg.overwrite):
                    skipped_existing += 1
                    continue

                try:
                    pil = Image.open(image_path)
                    pil = ImageOps.exif_transpose(pil)
                    width, height = pil.size
                except Exception:
                    errors += 1
                    continue

                payload: Dict[str, Any] = {
                    "version": LABELME_VERSION,
                    "flags": {},
                    "shapes": shapes,
                    "imagePath": image_path.name,
                    "imageData": None,
                    "imageHeight": int(height),
                    "imageWidth": int(width),
                }
                if not multi_animal:
                    payload["instance_label"] = str(cfg.instance_label)

                out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
                written += 1
            except Exception:
                errors += 1
                continue

    schema_path = None
    if write_pose_schema and keypoints_seen:
        insts: List[str] = []
        if len(instances_seen) > 1:
            insts = list(instances_seen)
        schema = build_pose_schema_from_keypoints(
            keypoints_seen,
            instances=insts,
            instance_separator=instance_separator,
            preset=pose_schema_preset,
        )
        schema_out = pose_schema_out or (labeled_root / "pose_schema.json")
        schema_out = (
            schema_out if schema_out.is_absolute() else (source_dir / schema_out)
        )
        schema.save(schema_out)
        schema_path = str(schema_out)

    return {
        "source_dir": str(source_dir),
        "labeled_data_root": str(labeled_root),
        "collected_csv_files": len(collected_csvs),
        "instance_label": str(cfg.instance_label),
        "multi_instance": len(instances_seen) > 1,
        "keypoints": list(keypoints_seen),
        "instances": list(instances_seen),
        "pose_schema": schema_path,
        "json_written": written,
        "json_skipped_existing": skipped_existing,
        "missing_images": missing_images,
        "errors": errors,
    }
