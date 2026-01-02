from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, Optional, Sequence, Tuple

from annolid.annotation.labelme2yolo import Labelme2YOLO
from annolid.datasets.labelme_collection import (
    iter_label_index_records,
    is_labeled_labelme_json,
    resolve_image_path,
)
from annolid.utils.annotation_store import load_labelme_json
from annolid.utils.logger import logger


@dataclass(frozen=True)
class IndexedPair:
    json_path: Path
    image_path: Path


def _short_hash(value: str, length: int = 10) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def _install_file(src: Path, dst: Path, *, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src)
        return
    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
        except OSError:
            shutil.copy2(src, dst)
        return
    raise ValueError(f"Unsupported mode: {mode!r}")


def iter_indexed_pairs(
    index_file: Path,
    *,
    include_empty: bool = False,
) -> Iterator[Tuple[Optional[IndexedPair], str]]:
    """Yield (pair, reason) from a label index file, skipping missing entries robustly."""
    index_file = Path(index_file).expanduser().resolve()
    for rec in iter_label_index_records(index_file):
        image_path_raw = rec.get("image_path")
        json_path_raw = rec.get("json_path")
        if not isinstance(image_path_raw, str) or not image_path_raw:
            yield None, "missing_image_path"
            continue
        image_path = Path(image_path_raw).expanduser()

        json_path: Optional[Path] = None
        if isinstance(json_path_raw, str) and json_path_raw:
            json_path = Path(json_path_raw).expanduser()

        if json_path is None:
            inferred = image_path.with_suffix(".json")
            if inferred.exists():
                json_path = inferred

        if json_path is None or not json_path.exists():
            yield None, "missing_json"
            continue
        if not include_empty and not is_labeled_labelme_json(json_path):
            yield None, "empty_shapes"
            continue

        if not image_path.exists():
            resolved = resolve_image_path(json_path)
            if resolved is None or not resolved.exists():
                yield None, "missing_image"
                continue
            image_path = resolved

        yield IndexedPair(json_path=json_path.resolve(), image_path=image_path.resolve()), "ok"


def build_yolo_from_label_index(
    *,
    index_file: Path,
    output_dir: Path,
    dataset_name: str = "YOLO_dataset",
    val_size: float = 0.1,
    test_size: float = 0.1,
    link_mode: str = "hardlink",
    task: str = "auto",
    include_empty: bool = False,
    overwrite: bool = True,
    keep_staging: bool = False,
    pred_worker=None,
    stop_event=None,
) -> Dict[str, object]:
    """Convert an Annolid label index JSONL into a YOLO dataset directory.

    This stages per-frame JSON+PNG pairs into a temporary directory so existing
    Labelme2YOLO conversion can run. Missing/deleted JSON or images are skipped.
    """
    index_file = Path(index_file).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = []
    stats = {
        "records_total": 0,
        "pairs_ok": 0,
        "skipped_missing_json": 0,
        "skipped_missing_image": 0,
        "skipped_missing_image_path": 0,
        "skipped_empty_shapes": 0,
        "skipped_other": 0,
    }

    for pair, reason in iter_indexed_pairs(index_file, include_empty=include_empty):
        stats["records_total"] += 1
        if pair is None:
            if reason == "missing_json":
                stats["skipped_missing_json"] += 1
            elif reason == "missing_image":
                stats["skipped_missing_image"] += 1
            elif reason == "missing_image_path":
                stats["skipped_missing_image_path"] += 1
            elif reason == "empty_shapes":
                stats["skipped_empty_shapes"] += 1
            else:
                stats["skipped_other"] += 1
            continue
        pairs.append(pair)

    stats["pairs_ok"] = len(pairs)
    if not pairs:
        return {
            "status": "no_pairs",
            "index_file": str(index_file),
            "output_dir": str(output_dir),
            **stats,
        }

    requested_task = (task or "auto").strip().lower()
    if requested_task not in {"auto", "segmentation", "pose"}:
        raise ValueError(f"Unsupported task: {task!r}")

    effective_task = requested_task
    if requested_task == "auto":
        dataset_has_points = False
        for pair in pairs:
            try:
                payload = load_labelme_json(pair.json_path)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            shapes = payload.get("shapes") or []
            if not isinstance(shapes, list):
                continue
            if any(
                isinstance(s, dict) and (s.get("shape_type")
                                         or "polygon").lower() == "point"
                for s in shapes
            ):
                dataset_has_points = True
                break
        # Prefer pose when any keypoints exist to avoid producing mixed-format datasets.
        effective_task = "pose" if dataset_has_points else "segmentation"

    staging_root = output_dir / "annolid_logs"
    staging_root.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(prefix="index_staging_", dir=str(staging_root)))

    try:
        total = len(pairs)
        for i, pair in enumerate(pairs, start=1):
            if stop_event is not None and stop_event.is_set():
                break

            unique = _short_hash(str(pair.json_path))
            stem = f"{pair.json_path.stem}__{unique}"
            staged_json = staging_dir / f"{stem}.json"
            staged_png = staging_dir / f"{stem}.png"

            # Ensure the staged image uses .png to satisfy existing Labelme2YOLO filtering.
            if pair.image_path.suffix.lower() == ".png":
                _install_file(pair.image_path, staged_png, mode=link_mode)
            else:
                try:
                    from PIL import Image

                    with Image.open(pair.image_path) as img:
                        img.save(staged_png)
                except Exception:
                    # Fallback: copy bytes even if extension mismatches.
                    _install_file(pair.image_path, staged_png, mode="copy")

            # Write a staged JSON that points at the staged image name so downstream
            # converters do not rely on the original on-disk layout.
            try:
                payload = load_labelme_json(pair.json_path)
            except Exception as exc:
                logger.warning("Skipping unreadable JSON %s: %s",
                               pair.json_path, exc)
                continue
            if not isinstance(payload, dict):
                logger.warning(
                    "Skipping non-dict JSON payload: %s", pair.json_path)
                continue
            payload = dict(payload)

            if effective_task == "segmentation":
                shapes = payload.get("shapes") or []
                if not isinstance(shapes, list):
                    shapes = []
                payload["shapes"] = [
                    s
                    for s in shapes
                    if isinstance(s, dict) and (s.get("shape_type") or "polygon").lower() != "point"
                ]
            # For pose, keep polygons + points so polygons can provide better bounds.

            payload["imagePath"] = staged_png.name
            payload.pop("imageData", None)
            staged_json.write_text(
                json.dumps(payload, separators=(",", ":")),
                encoding="utf-8",
            )

            if pred_worker is not None and total > 0:
                pct = int(round((i / total) * 100))
                pred_worker.progress_signal.emit(max(0, min(100, pct)))

        dataset_dir = output_dir / dataset_name
        if dataset_dir.exists() and overwrite:
            shutil.rmtree(dataset_dir)

        converter = Labelme2YOLO(
            str(staging_dir),
            yolo_dataset_name=dataset_name,
        )
        converter.convert(val_size=val_size, test_size=test_size)

        staged_dataset_dir = staging_dir / dataset_name
        if not staged_dataset_dir.exists():
            raise RuntimeError(
                f"YOLO dataset folder not created: {staged_dataset_dir}")

        shutil.move(str(staged_dataset_dir), str(dataset_dir))

        return {
            "status": "ok",
            "index_file": str(index_file),
            "output_dir": str(output_dir),
            "dataset_dir": str(dataset_dir),
            "staging_dir": str(staging_dir) if keep_staging else None,
            "task": effective_task,
            **stats,
        }
    finally:
        if not keep_staging:
            try:
                shutil.rmtree(staging_dir, ignore_errors=True)
            except Exception:
                logger.debug(
                    "Failed to cleanup staging directory.", exc_info=True)
