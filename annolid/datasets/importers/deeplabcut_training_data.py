from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image, ImageOps
import pandas as pd

from annolid.annotation.keypoint_visibility import KeypointVisibility
from annolid.annotation.pose_interop.deeplabcut import (
    detect_dlc_csv_header_levels,
    discover_dlc_config_near,
    parse_dlc_table,
)
from annolid.annotation.pose_interop.labelme import build_labelme_point_shape
from annolid.annotation.pose_interop.model import (
    dedup_pairs,
    dedup_preserve_order,
    infer_edges_for_keypoints,
    is_finite_number,
)
from annolid.annotation.pose_schema import PoseSchema
from annolid.utils.annotation_compat import __version__ as LABELME_VERSION
from annolid.utils.atomic_io import atomic_write_json

_MAX_RECORDED_ERRORS = 25


@dataclass(frozen=True)
class DeepLabCutTrainingImportConfig:
    """Import DeepLabCut / Lightning Pose ``labeled-data/**/CollectedData_*.{csv,h5}`` into LabelMe JSON."""

    source_dir: Path
    labeled_data_root: Path = Path("labeled-data")
    instance_label: str = "mouse"
    overwrite: bool = False
    recursive: bool = True


def _detect_dlc_header_levels(csv_path: Path) -> int:
    return detect_dlc_csv_header_levels(csv_path)


def _iter_collected_data_files(root: Path, *, recursive: bool) -> Iterable[Path]:
    """Yield ``CollectedData_*.csv`` and ``CollectedData_*.h5`` (deduping sibling pairs in favor of CSV)."""
    patterns = ("CollectedData_*.csv", "CollectedData_*.h5")
    found: List[Path] = []
    for pattern in patterns:
        it = root.rglob(pattern) if recursive else root.glob(pattern)
        found.extend(p for p in it if p.is_file())

    # If both CollectedData_scorer.csv and CollectedData_scorer.h5 sit side by side,
    # prefer the CSV to avoid importing the same rows twice.
    csv_stems = {
        (p.parent.resolve(), p.stem.lower())
        for p in found
        if p.suffix.lower() == ".csv"
    }
    for path in sorted(found):
        if (
            path.suffix.lower() == ".h5"
            and (
                path.parent.resolve(),
                path.stem.lower(),
            )
            in csv_stems
        ):
            continue
        yield path


def _iter_collected_data_csvs(root: Path, *, recursive: bool) -> Iterable[Path]:
    pattern = "CollectedData_*.csv"
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def _is_number(value: Any) -> bool:
    return is_finite_number(value)


def _infer_mouse_edges(keypoints: Sequence[str]) -> List[Tuple[str, str]]:
    return infer_edges_for_keypoints(keypoints, preset="mouse")


def build_pose_schema_from_keypoints(
    keypoints: Sequence[str],
    *,
    instances: Optional[Sequence[str]] = None,
    instance_separator: str = "_",
    preset: Optional[str] = "mouse",
    edges: Optional[Sequence[Tuple[str, str]]] = None,
) -> PoseSchema:
    base_keypoints = dedup_preserve_order(keypoints)
    symmetry_pairs = PoseSchema.infer_symmetry_pairs(base_keypoints)
    resolved_edges = (
        dedup_pairs(edges)
        if edges
        else infer_edges_for_keypoints(base_keypoints, preset=preset)
    )
    return PoseSchema(
        keypoints=base_keypoints,
        edges=resolved_edges,
        symmetry_pairs=symmetry_pairs,
        instances=dedup_preserve_order(instances or []),
        instance_separator=str(instance_separator or "_"),
    )


def _normalize_path_parts(path_parts: Sequence[str]) -> List[str]:
    """Normalize Windows and POSIX path segments while preserving absolute markers."""
    normalized: List[str] = []
    for raw in path_parts:
        text = str(raw or "").strip()
        if not text or text.lower() == "nan":
            continue
        # Convert Windows backslashes to POSIX separators for cross-platform CSV imports.
        posix_text = text.replace("\\", "/")
        if posix_text.startswith("/") and not normalized:
            normalized.append(posix_text)
        else:
            for seg in posix_text.split("/"):
                seg = seg.strip()
                if seg:
                    normalized.append(seg)
    return normalized


def _resolve_image_path(
    *,
    source_dir: Path,
    labeled_root: Path,
    csv_dir: Path,
    path_parts: Sequence[str],
) -> Optional[Path]:
    parts = _normalize_path_parts(path_parts)
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

    resolved_source = source_dir.expanduser().resolve()
    resolved_labeled = labeled_root.expanduser().resolve()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
            if not (
                resolved.is_relative_to(resolved_source)
                or resolved.is_relative_to(resolved_labeled)
            ):
                continue
            if resolved.is_file():
                return resolved
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
    instance_to_gid: Optional[Dict[str, int]] = None,
    instance_separator: str = "_",
) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
    """Return ``(shapes, instances, base_keypoints)`` with LabelMe-safe boolean flags."""
    shapes: List[Dict[str, Any]] = []
    instances: List[str] = []
    base_keypoints: List[str] = []
    gid_map = instance_to_gid if instance_to_gid is not None else {}

    # single: (scorer, bodypart, coord)
    # multi:  (scorer, individual, bodypart, coord)
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

        gid: Optional[int] = None
        if multi_animal and instance_label:
            if instance_label not in gid_map:
                gid_map[instance_label] = len(gid_map)
            gid = gid_map[instance_label]

        shapes.append(
            build_labelme_point_shape(
                keypoint_label=bodypart,
                x=values["x"],
                y=values["y"],
                instance_label=instance_label or None,
                group_id=gid,
                visibility=int(KeypointVisibility.VISIBLE),
                score=values.get("likelihood"),
                is_predicted=False,
                prefix_with_instance=bool(instance_label),
                instance_separator=instance_separator,
            )
        )

    return shapes, instances, base_keypoints


def _extract_row_path_parts(
    row: pd.Series, path_cols: Sequence[Tuple[Any, ...]]
) -> List[str]:
    parts: List[str] = []
    if path_cols:
        for col in path_cols[:3]:
            value = row.get(col)
            if value is None:
                continue
            text = str(value).strip()
            if text and text.lower() != "nan":
                parts.append(text)
    if not parts:
        idx = row.name
        if isinstance(idx, tuple):
            for elem in idx:
                text = str(elem).strip()
                if text and text.lower() != "nan":
                    parts.append(text)
        elif isinstance(idx, str) and idx.strip():
            parts.append(idx.strip())
    return parts


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

    dlc_config = discover_dlc_config_near(labeled_root) or discover_dlc_config_near(
        source_dir
    )

    collected_files = list(
        _iter_collected_data_files(labeled_root, recursive=cfg.recursive)
    )
    written = 0
    skipped_existing = 0
    missing_images = 0
    errors = 0
    error_details: List[str] = []
    keypoints_seen: List[str] = []
    instances_seen: List[str] = []
    instance_to_gid: Dict[str, int] = {}

    if dlc_config:
        for idx, ind in enumerate(dlc_config.individuals):
            instance_to_gid.setdefault(ind, idx)

    def _record_error(message: str) -> None:
        nonlocal errors
        errors += 1
        if len(error_details) < _MAX_RECORDED_ERRORS:
            error_details.append(message)

    for data_path in collected_files:
        try:
            table = parse_dlc_table(data_path, for_training=True)
        except Exception as exc:
            _record_error(f"Failed to parse {data_path.name}: {exc}")
            continue

        df = table.df
        kp_cols = table.keypoint_columns
        multi_animal = table.multi_animal
        path_cols = table.path_columns

        for row_num, (_, row) in enumerate(df.iterrows()):
            try:
                path_parts = _extract_row_path_parts(row, path_cols)
                image_path = _resolve_image_path(
                    source_dir=source_dir,
                    labeled_root=labeled_root,
                    csv_dir=data_path.parent,
                    path_parts=path_parts,
                )
                if image_path is None or not image_path.is_file():
                    missing_images += 1
                    continue
                out_json = image_path.with_suffix(".json")

                shapes, instances, keypoints = _extract_keypoints_from_row(
                    row,
                    kp_cols=kp_cols,
                    multi_animal=multi_animal,
                    default_instance=str(cfg.instance_label),
                    instance_to_gid=instance_to_gid,
                    instance_separator=instance_separator,
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
                    with Image.open(image_path) as pil_raw:
                        pil = ImageOps.exif_transpose(pil_raw)
                        width, height = pil.size
                except Exception as exc:
                    _record_error(f"Could not read image '{image_path.name}': {exc}")
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

                atomic_write_json(out_json, payload, indent=2)
                written += 1
            except Exception as exc:
                _record_error(f"Failed row {row_num} in {data_path.name}: {exc}")
                continue

    if dlc_config and dlc_config.bodyparts:
        keypoints_seen = dedup_preserve_order(
            [kp for kp in dlc_config.bodyparts if kp in keypoints_seen] + keypoints_seen
        )

    schema_path = None
    if write_pose_schema and keypoints_seen:
        insts: List[str] = list(instances_seen) if len(instances_seen) > 1 else []
        config_edges = (
            list(dlc_config.skeleton) if (dlc_config and dlc_config.skeleton) else None
        )
        schema = build_pose_schema_from_keypoints(
            keypoints_seen,
            instances=insts,
            instance_separator=instance_separator,
            preset=pose_schema_preset,
            edges=config_edges,
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
        "collected_csv_files": len(collected_files),
        "instance_label": str(cfg.instance_label),
        "multi_instance": len(instances_seen) > 1,
        "keypoints": list(keypoints_seen),
        "instances": list(instances_seen),
        "pose_schema": schema_path,
        "config_yaml": (
            str(dlc_config.config_path)
            if dlc_config and dlc_config.config_path
            else None
        ),
        "json_written": written,
        "json_skipped_existing": skipped_existing,
        "missing_images": missing_images,
        "errors": errors,
        "error_details": error_details,
    }
