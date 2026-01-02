"""Utilities for collecting LabelMe-style PNG/JSON annotation pairs.

This is useful for aggregating manually corrected annotations saved across
multiple Annolid sessions into a single dataset directory.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional, Sequence, Set

from annolid.utils.annotation_store import load_labelme_json
from annolid.utils.logger import logger


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_LABEL_INDEX_DIRNAME = "annolid_logs"
DEFAULT_LABEL_INDEX_NAME = "annolid_dataset.jsonl"


def default_label_index_path(dataset_root: Path) -> Path:
    dataset_root = Path(dataset_root).expanduser().resolve()
    return dataset_root / DEFAULT_LABEL_INDEX_DIRNAME / DEFAULT_LABEL_INDEX_NAME


@dataclass(frozen=True)
class CollectedPair:
    source_json: Path
    source_image: Path
    dest_json: Path
    dest_image: Path


def _find_sidecar_image(json_path: Path) -> Optional[Path]:
    base = json_path.with_suffix("")
    for ext in IMAGE_EXTENSIONS:
        candidate = base.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def resolve_image_path(json_path: Path) -> Optional[Path]:
    """Resolve the image corresponding to a LabelMe JSON file."""
    sidecar = _find_sidecar_image(json_path)
    if sidecar is not None:
        return sidecar

    try:
        payload = load_labelme_json(json_path)
    except Exception:
        return None

    image_path = payload.get("imagePath")
    if not image_path:
        return None

    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = json_path.parent / candidate
    if candidate.exists():
        return candidate

    # As a final fallback, look for common extensions with the same stem.
    base = (json_path.parent / Path(image_path).stem)
    for ext in IMAGE_EXTENSIONS:
        alt = base.with_suffix(ext)
        if alt.exists():
            return alt
    return None


def is_labeled_labelme_json(json_path: Path) -> bool:
    """Return True when the JSON contains at least one shape."""
    try:
        payload = load_labelme_json(json_path)
    except Exception:
        return False
    shapes = payload.get("shapes", [])
    return bool(isinstance(shapes, list) and len(shapes) > 0)


def build_label_index_record(
    *,
    image_path: Path,
    json_path: Optional[Path],
    source: str,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "record_version": 1,
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "image_path": str(Path(image_path).expanduser().resolve()),
        "json_path": str(Path(json_path).expanduser().resolve()) if json_path else None,
    }

    if json_path and Path(json_path).exists():
        try:
            payload = load_labelme_json(json_path)
        except Exception:
            payload = None
        if isinstance(payload, dict):
            shapes = payload.get("shapes", [])
            if isinstance(shapes, list):
                record["shapes_count"] = len(shapes)
                record["labels"] = sorted(
                    {str(s.get("label")) for s in shapes if isinstance(s, dict) and s.get("label")}
                )
            for key in ("imageHeight", "imageWidth", "imagePath"):
                if key in payload:
                    record[key] = payload.get(key)

        try:
            stat = Path(json_path).stat()
        except OSError:
            pass
        else:
            record["json_mtime_ns"] = stat.st_mtime_ns
            record["json_size"] = stat.st_size

    try:
        stat = Path(image_path).stat()
    except OSError:
        pass
    else:
        record["image_mtime_ns"] = stat.st_mtime_ns
        record["image_size"] = stat.st_size

    return record


def _iter_jsonl_records(path: Path) -> Iterator[Dict[str, object]]:
    if not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(payload, dict):
                    yield payload
    except OSError:
        return


def iter_label_index_records(index_file: Path) -> Iterator[Dict[str, object]]:
    """Iterate label index JSONL records."""
    yield from _iter_jsonl_records(Path(index_file))


def load_indexed_image_paths(index_file: Path) -> Set[str]:
    """Return all image paths already present in an index file."""
    indexed: Set[str] = set()
    for rec in iter_label_index_records(Path(index_file)):
        image_path = rec.get("image_path")
        if isinstance(image_path, str) and image_path:
            indexed.add(image_path)
    return indexed


def append_label_index_record(
    index_file: Path,
    record: Dict[str, object],
) -> None:
    index_file = Path(index_file).expanduser().resolve()
    _ensure_dir(index_file.parent)
    with index_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, separators=(",", ":")))
        fh.write("\n")


def index_labelme_pair(
    *,
    json_path: Path,
    index_file: Path,
    image_path: Optional[Path] = None,
    include_empty: bool = False,
    source: str = "annolid",
) -> Optional[Dict[str, object]]:
    """Append one labeled pair to an index file (no file copying)."""
    json_path = Path(json_path)
    if not json_path.exists():
        return None

    if not include_empty and not is_labeled_labelme_json(json_path):
        return None

    resolved_image = Path(image_path) if image_path else resolve_image_path(json_path)
    if resolved_image is None or not resolved_image.exists():
        return None

    record = build_label_index_record(
        image_path=resolved_image,
        json_path=json_path,
        source=source,
    )
    append_label_index_record(index_file, record)
    return record


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _install_file(
    src: Path,
    dst: Path,
    *,
    mode: str,
    overwrite: bool,
) -> None:
    if dst.exists():
        if not overwrite:
            return
        try:
            dst.unlink()
        except OSError:
            pass

    _ensure_dir(dst.parent)

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "symlink":
        dst.symlink_to(src)
        return

    if mode == "hardlink":
        try:
            os.link(src, dst)
        except OSError:
            shutil.copy2(src, dst)
        return

    raise ValueError(f"Unsupported collection mode: {mode!r}")


def collect_labelme_pair(
    json_path: Path,
    dataset_root: Path,
    *,
    image_path: Optional[Path] = None,
    group: Optional[str] = None,
    mode: str = "hardlink",
    overwrite: bool = True,
    include_empty: bool = False,
    manifest_name: str = "manifest.jsonl",
) -> Optional[CollectedPair]:
    """Collect one JSON+image pair into the dataset root.

    Args:
        json_path: Source LabelMe JSON path.
        dataset_root: Root dataset directory that will receive collected files.
        image_path: Optional explicit image path; otherwise inferred.
        group: Optional subfolder name under dataset_root (defaults to json parent).
        mode: One of "hardlink", "copy", "symlink".
        overwrite: Replace existing collected files when True.
        include_empty: Collect JSON files with empty "shapes" when True.
        manifest_name: Append collection events to this jsonl file in dataset_root.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        return None

    if not include_empty and not is_labeled_labelme_json(json_path):
        return None

    resolved_image = Path(image_path) if image_path else resolve_image_path(json_path)
    if resolved_image is None or not resolved_image.exists():
        return None

    dataset_root = Path(dataset_root).expanduser().resolve()
    if group is None:
        group_name = json_path.parent.name or "labels"
    else:
        group_name = str(group)
    dest_dir = dataset_root if group_name.strip() == "" else (dataset_root / group_name)
    dest_json = dest_dir / json_path.name
    dest_image = dest_dir / resolved_image.name

    _install_file(json_path, dest_json, mode=mode, overwrite=overwrite)
    _install_file(resolved_image, dest_image, mode=mode, overwrite=overwrite)

    manifest_path = dataset_root / manifest_name
    try:
        record: Dict[str, object] = {
            "collected_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "mode": mode,
            "group": group_name,
            "source_json": str(json_path.resolve()),
            "source_image": str(resolved_image.resolve()),
            "dest_json": str(dest_json),
            "dest_image": str(dest_image),
            "source_json_mtime_ns": json_path.stat().st_mtime_ns,
            "source_image_mtime_ns": resolved_image.stat().st_mtime_ns,
            "source_json_size": json_path.stat().st_size,
            "source_image_size": resolved_image.stat().st_size,
        }
        _ensure_dir(manifest_path.parent)
        with manifest_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, separators=(",", ":")))
            fh.write("\n")
    except Exception as exc:
        logger.warning("Failed to update label collection manifest: %s", exc)

    return CollectedPair(
        source_json=json_path,
        source_image=resolved_image,
        dest_json=dest_json,
        dest_image=dest_image,
    )


def iter_labelme_json_files(source: Path, *, recursive: bool = True) -> Iterator[Path]:
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".json":
        yield source
        return
    if not source.exists():
        return
    globber = source.rglob if recursive else source.glob
    for path in globber("*.json"):
        name = path.name.lower()
        if name.endswith(".subjects.json"):
            continue
        yield path


def collect_labelme_dataset(
    sources: Sequence[Path],
    dataset_root: Path,
    *,
    recursive: bool = True,
    mode: str = "hardlink",
    overwrite: bool = False,
    include_empty: bool = False,
    flat: bool = False,
) -> Dict[str, int]:
    """Collect all label pairs found in the sources into dataset_root."""
    collected = 0
    skipped = 0
    missing_image = 0

    for src in sources:
        for json_path in iter_labelme_json_files(Path(src), recursive=recursive):
            image_path = resolve_image_path(json_path)
            if image_path is None:
                missing_image += 1
                continue
            group = "" if flat else json_path.parent.name
            pair = collect_labelme_pair(
                json_path,
                dataset_root,
                image_path=image_path,
                group=group,
                mode=mode,
                overwrite=overwrite,
                include_empty=include_empty,
            )
            if pair is None:
                skipped += 1
            else:
                collected += 1

    return {
        "collected": collected,
        "skipped": skipped,
        "missing_image": missing_image,
    }


def index_labelme_dataset(
    sources: Sequence[Path],
    *,
    index_file: Path,
    recursive: bool = True,
    include_empty: bool = False,
    dedupe: bool = True,
    source: str = "annolid_cli",
) -> Dict[str, int]:
    """Scan sources and append labeled image/json absolute paths to an index file."""
    appended = 0
    skipped = 0
    missing_image = 0

    indexed_images: Set[str] = set()
    if dedupe:
        indexed_images = load_indexed_image_paths(index_file)

    for src in sources:
        for json_path in iter_labelme_json_files(Path(src), recursive=recursive):
            image_path = resolve_image_path(json_path)
            if image_path is None:
                missing_image += 1
                continue
            image_abs = str(Path(image_path).expanduser().resolve())
            if dedupe and image_abs in indexed_images:
                skipped += 1
                continue

            record = index_labelme_pair(
                json_path=Path(json_path),
                index_file=Path(index_file),
                image_path=Path(image_path),
                include_empty=include_empty,
                source=source,
            )
            if record is None:
                skipped += 1
                continue

            appended += 1
            if dedupe:
                indexed_images.add(image_abs)

    return {"appended": appended, "skipped": skipped, "missing_image": missing_image}


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annolid label-index",
        description="Index LabelMe PNG/JSON pairs by absolute path (no copying).",
    )
    p.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source folder (repeatable) containing per-frame JSON/PNG pairs.",
    )
    p.add_argument(
        "--dataset-root",
        required=True,
        help="Directory that will store the index file.",
    )
    p.add_argument("--recursive", action="store_true", default=True)
    p.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top-level of each source directory.",
    )
    p.add_argument(
        "--include-empty",
        action="store_true",
        help="Collect JSON files even when they contain no shapes.",
    )
    p.add_argument(
        "--index-file",
        default=str(Path(DEFAULT_LABEL_INDEX_DIRNAME) / DEFAULT_LABEL_INDEX_NAME),
        help=(
            "Index file path relative to --dataset-root "
            f"(default: {DEFAULT_LABEL_INDEX_DIRNAME}/{DEFAULT_LABEL_INDEX_NAME})."
        ),
    )
    p.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Append even if the image path already exists in the index.",
    )
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(list(argv) if argv is not None else None)
    dataset_root = Path(args.dataset_root)
    index_file = Path(args.index_file)
    if not index_file.is_absolute():
        index_file = dataset_root / index_file

    summary = index_labelme_dataset(
        [Path(p) for p in args.source],
        index_file=index_file,
        recursive=bool(args.recursive),
        include_empty=bool(args.include_empty),
        dedupe=not bool(args.allow_duplicates),
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
