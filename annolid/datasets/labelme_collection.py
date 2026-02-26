"""Utilities for collecting LabelMe-style PNG/JSON annotation pairs.

This is useful for aggregating manually corrected annotations saved across
multiple Annolid sessions into a single dataset directory.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Set, Tuple

import yaml
from annolid.utils.annotation_store import load_labelme_json
from annolid.utils.logger import logger
from annolid.utils.log_paths import (
    APP_LABEL_INDEX_SUBDIR,
    APP_LOGS_DIRNAME,
    ANNOLID_LOGS_DIRNAME,
    resolve_annolid_logs_root,
)


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_LABEL_INDEX_DIRNAME = f"{APP_LOGS_DIRNAME}/{APP_LABEL_INDEX_SUBDIR}"
DEFAULT_LABEL_INDEX_NAME = "annolid_dataset.jsonl"
_SPLIT_NAME_PATTERN = re.compile(
    r"^(train(?:ing)?|val|valid(?:ation)?|test)(?:[_-].+)?$",
    re.IGNORECASE,
)


def default_annolid_logs_root(dataset_root: Optional[Path] = None) -> Path:
    """Resolve the annolid_logs root with bot-workspace fallback."""
    return resolve_annolid_logs_root(dataset_root)


def default_label_index_path(dataset_root: Optional[Path] = None) -> Path:
    logs_root = default_annolid_logs_root(dataset_root)
    return logs_root / APP_LABEL_INDEX_SUBDIR / DEFAULT_LABEL_INDEX_NAME


def resolve_label_index_path(
    index_file: Path, dataset_root: Optional[Path] = None
) -> Path:
    """Resolve index file path, anchoring relative paths to dataset_root/logs.

    When dataset_root is missing, relative paths are anchored to the bot workspace
    logs directory.
    """
    candidate = Path(index_file).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    logs_root = default_annolid_logs_root(dataset_root)
    parts = [p for p in candidate.parts if p not in (".", "")]
    if not parts:
        return (logs_root / DEFAULT_LABEL_INDEX_NAME).resolve()
    if parts[0] in {ANNOLID_LOGS_DIRNAME, APP_LOGS_DIRNAME}:
        parts = parts[1:]
    if not parts:
        return (logs_root / APP_LABEL_INDEX_SUBDIR / DEFAULT_LABEL_INDEX_NAME).resolve()
    resolved = logs_root.joinpath(*parts)
    if resolved.suffix:
        # Plain filename should default under logs/label_index
        if len(parts) == 1:
            return (logs_root / APP_LABEL_INDEX_SUBDIR / parts[0]).resolve()
        return resolved.resolve()
    return (resolved / DEFAULT_LABEL_INDEX_NAME).resolve()


def normalize_labelme_sources(
    sources: Sequence[Path],
) -> Tuple[List[Path], List[Path]]:
    existing: List[Path] = []
    missing: List[Path] = []
    seen: Set[Path] = set()
    for src in sources:
        candidate = Path(src).expanduser()
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            existing.append(resolved)
        else:
            missing.append(candidate)
    return existing, missing


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


def _load_labelme_payload(json_path: Path) -> Optional[Dict[str, object]]:
    try:
        payload = load_labelme_json(json_path)
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def resolve_image_path(
    json_path: Path,
    *,
    payload: Optional[Dict[str, object]] = None,
) -> Optional[Path]:
    """Resolve the image corresponding to a LabelMe JSON file."""
    sidecar = _find_sidecar_image(json_path)
    if sidecar is not None:
        return sidecar

    resolved_payload = (
        payload if isinstance(payload, dict) else _load_labelme_payload(json_path)
    )
    if not isinstance(resolved_payload, dict):
        return None

    image_path = resolved_payload.get("imagePath")
    if not image_path:
        return None

    candidate = Path(image_path)
    if not candidate.is_absolute():
        candidate = json_path.parent / candidate
    if candidate.exists():
        return candidate

    # As a final fallback, look for common extensions with the same stem.
    base = json_path.parent / Path(image_path).stem
    for ext in IMAGE_EXTENSIONS:
        alt = base.with_suffix(ext)
        if alt.exists():
            return alt
    return None


def is_labeled_labelme_json(
    json_path: Path,
    *,
    payload: Optional[Dict[str, object]] = None,
) -> bool:
    """Return True when the JSON contains at least one shape."""
    resolved_payload = (
        payload if isinstance(payload, dict) else _load_labelme_payload(json_path)
    )
    if not isinstance(resolved_payload, dict):
        return False
    shapes = resolved_payload.get("shapes", [])
    return bool(isinstance(shapes, list) and len(shapes) > 0)


def build_label_index_record(
    *,
    image_path: Path,
    json_path: Optional[Path],
    source: str,
    payload: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "record_version": 1,
        "indexed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source": source,
        "image_path": str(Path(image_path).expanduser().resolve()),
        "json_path": str(Path(json_path).expanduser().resolve()) if json_path else None,
    }

    if json_path and Path(json_path).exists():
        payload_data = (
            payload
            if isinstance(payload, dict)
            else _load_labelme_payload(Path(json_path))
        )
        if isinstance(payload_data, dict):
            shapes = payload_data.get("shapes", [])
            if isinstance(shapes, list):
                record["shapes_count"] = len(shapes)
                record["labels"] = sorted(
                    {
                        str(s.get("label"))
                        for s in shapes
                        if isinstance(s, dict) and s.get("label")
                    }
                )
            for key in ("imageHeight", "imageWidth", "imagePath"):
                if key in payload_data:
                    record[key] = payload_data.get(key)

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

    payload = _load_labelme_payload(json_path)

    if not include_empty and not is_labeled_labelme_json(json_path, payload=payload):
        return None

    resolved_image = (
        Path(image_path)
        if image_path
        else resolve_image_path(json_path, payload=payload)
    )
    if resolved_image is None or not resolved_image.exists():
        return None

    record = build_label_index_record(
        image_path=resolved_image,
        json_path=json_path,
        source=source,
        payload=payload,
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


def iter_labelme_pairs(
    sources: Sequence[Path],
    *,
    recursive: bool = True,
    include_empty: bool = False,
) -> List[Tuple[Path, Path]]:
    pairs: List[Tuple[Path, Path]] = []
    for src in sources:
        for json_path in iter_labelme_json_files(Path(src), recursive=recursive):
            if not include_empty and not is_labeled_labelme_json(json_path):
                continue
            image_path = resolve_image_path(json_path)
            if image_path is None or not Path(image_path).exists():
                continue
            pairs.append((Path(json_path), Path(image_path)))
    return pairs


def _group_key_from_path(path: Path, *, group_by: str, regex: Optional[str]) -> str:
    group_by = str(group_by or "parent").strip().lower()
    if group_by == "none":
        return ""
    if group_by == "parent":
        return path.parent.name or path.name
    if group_by == "grandparent":
        return path.parent.parent.name or path.parent.name or path.name
    if group_by == "stem_prefix":
        stem = path.stem
        for sep in ("_", "-", "."):
            if sep in stem:
                return stem.split(sep)[0]
        return stem
    if group_by == "regex" and regex:
        import re

        match = re.search(regex, path.as_posix())
        if match:
            if match.groups():
                return match.group(1)
            return match.group(0)
    return path.parent.name or path.name


def _normalize_split_name(name: str) -> Optional[str]:
    token = str(name or "").strip().lower()
    match = _SPLIT_NAME_PATTERN.match(token)
    if not match:
        return None
    base = match.group(1).lower()
    if base.startswith("train"):
        return "train"
    if base.startswith("val") or base.startswith("valid"):
        return "val"
    if base == "test":
        return "test"
    return None


def _infer_split_from_relative_path(rel_path: Path) -> Optional[str]:
    # Infer from directory names only (not from file stem).
    for part in reversed(rel_path.parts[:-1]):
        split = _normalize_split_name(part)
        if split:
            return split
    return None


def infer_split_pairs_from_sources(
    pairs: Sequence[Tuple[Path, Path]],
    *,
    sources: Sequence[Path],
) -> Optional[Dict[str, List[Tuple[Path, Path]]]]:
    roots = []
    for src in sources:
        candidate = Path(src).expanduser()
        if not candidate.exists():
            continue
        try:
            roots.append(candidate.resolve())
        except OSError:
            roots.append(candidate)
    if not roots:
        return None

    # Prefer the most specific source root first.
    roots.sort(key=lambda p: len(p.parts), reverse=True)
    splits: Dict[str, List[Tuple[Path, Path]]] = {"train": [], "val": [], "test": []}
    unmatched: List[Tuple[Path, Path]] = []

    for pair in pairs:
        json_path = Path(pair[0])
        try:
            resolved = json_path.resolve()
        except OSError:
            resolved = json_path

        split_name: Optional[str] = None
        for root in roots:
            try:
                rel = resolved.relative_to(root)
            except ValueError:
                continue
            split_name = _infer_split_from_relative_path(rel)
            if split_name:
                break

        if split_name:
            splits[split_name].append(pair)
        else:
            unmatched.append(pair)

    if not any(splits.values()):
        return None

    # Keep non-matching files in train by default.
    if unmatched:
        splits["train"].extend(unmatched)

    return splits


def split_labelme_pairs(
    pairs: Sequence[Tuple[Path, Path]],
    *,
    val_size: float = 0.1,
    test_size: float = 0.0,
    seed: int = 0,
    group_by: str = "parent",
    group_regex: Optional[str] = None,
) -> Dict[str, List[Tuple[Path, Path]]]:
    import random

    total = len(pairs)
    if total == 0:
        return {"train": [], "val": [], "test": []}

    val_size = max(0.0, float(val_size))
    test_size = max(0.0, float(test_size))
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.0")

    rng = random.Random(int(seed))
    group_by_norm = str(group_by or "parent").strip().lower()

    if group_by_norm == "none":
        indices = list(range(total))
        rng.shuffle(indices)
        val_n = int(round(val_size * total))
        test_n = int(round(test_size * total))
        val_idx = set(indices[:val_n])
        test_idx = set(indices[val_n : val_n + test_n])
        train = []
        val = []
        test = []
        for i, pair in enumerate(pairs):
            if i in val_idx:
                val.append(pair)
            elif i in test_idx:
                test.append(pair)
            else:
                train.append(pair)
        return {"train": train, "val": val, "test": test}

    groups: Dict[str, List[Tuple[Path, Path]]] = {}
    for json_path, image_path in pairs:
        key = _group_key_from_path(
            Path(image_path), group_by=group_by_norm, regex=group_regex
        )
        groups.setdefault(key, []).append((json_path, image_path))

    keys = list(groups.keys())
    rng.shuffle(keys)
    target_val = max(0, int(round(val_size * total)))
    target_test = max(0, int(round(test_size * total)))

    train: List[Tuple[Path, Path]] = []
    val: List[Tuple[Path, Path]] = []
    test: List[Tuple[Path, Path]] = []
    for key in keys:
        bucket = groups[key]
        if len(val) < target_val:
            val.extend(bucket)
        elif len(test) < target_test:
            test.extend(bucket)
        else:
            train.extend(bucket)

    return {"train": train, "val": val, "test": test}


def infer_labelme_keypoint_names(
    pairs: Sequence[Tuple[Path, Path]],
    *,
    max_files: int = 500,
    min_count: int = 1,
) -> List[str]:
    counts: Dict[str, int] = {}
    inspected = 0
    for json_path, _ in pairs:
        if inspected >= int(max_files):
            break
        inspected += 1
        try:
            payload = load_labelme_json(json_path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        shapes = payload.get("shapes")
        if not isinstance(shapes, list):
            continue
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            if shape.get("shape_type") != "point":
                continue
            label = str(shape.get("label") or "").strip()
            if not label:
                continue
            counts[label] = counts.get(label, 0) + 1

    items = [(name, cnt) for name, cnt in counts.items() if cnt >= int(min_count)]
    items.sort(key=lambda x: (-x[1], x[0]))
    return [name for name, _ in items]


def write_labelme_index(
    pairs: Sequence[Tuple[Path, Path]],
    *,
    index_file: Path,
    source: str = "annolid_cli",
) -> None:
    index_file = Path(index_file).expanduser().resolve()
    _ensure_dir(index_file.parent)
    for json_path, image_path in pairs:
        record = build_label_index_record(
            image_path=Path(image_path),
            json_path=Path(json_path),
            source=source,
        )
        append_label_index_record(index_file, record)


def generate_labelme_spec_and_splits(
    *,
    sources: Sequence[Path],
    dataset_root: Path,
    recursive: bool = True,
    include_empty: bool = False,
    split_dir: Optional[str] = None,
    val_size: float = 0.1,
    test_size: float = 0.0,
    seed: int = 0,
    group_by: str = "parent",
    group_regex: Optional[str] = None,
    keypoint_names: Optional[Sequence[str]] = None,
    kpt_dims: int = 3,
    infer_flip_idx: bool = True,
    max_keypoint_files: int = 500,
    min_keypoint_count: int = 1,
    spec_path: Optional[Path] = None,
    split_sources: Optional[Dict[str, Sequence[Path]]] = None,
    source: str = "annolid_cli",
) -> Dict[str, object]:
    dataset_root = Path(dataset_root).expanduser().resolve()
    split_dir_name = str(split_dir or DEFAULT_LABEL_INDEX_DIRNAME)
    split_dir_path = dataset_root / split_dir_name
    split_dir_path.mkdir(parents=True, exist_ok=True)

    if split_sources:
        splits = {
            "train": iter_labelme_pairs(
                split_sources.get("train", []),
                recursive=recursive,
                include_empty=include_empty,
            ),
            "val": iter_labelme_pairs(
                split_sources.get("val", []),
                recursive=recursive,
                include_empty=include_empty,
            ),
            "test": iter_labelme_pairs(
                split_sources.get("test", []),
                recursive=recursive,
                include_empty=include_empty,
            ),
        }
        pairs = splits["train"] + splits["val"] + splits["test"]
    else:
        pairs = iter_labelme_pairs(
            sources,
            recursive=recursive,
            include_empty=include_empty,
        )
        inferred_splits = infer_split_pairs_from_sources(pairs, sources=sources)
        if inferred_splits:
            splits = inferred_splits
        else:
            splits = split_labelme_pairs(
                pairs,
                val_size=float(val_size),
                test_size=float(test_size),
                seed=int(seed),
                group_by=str(group_by),
                group_regex=(str(group_regex) if group_regex else None),
            )

    train_index = split_dir_path / "labelme_train.jsonl"
    val_index = split_dir_path / "labelme_val.jsonl"
    test_index = split_dir_path / "labelme_test.jsonl"

    write_labelme_index(splits.get("train", []), index_file=train_index, source=source)
    if splits.get("val"):
        write_labelme_index(splits.get("val", []), index_file=val_index, source=source)
    else:
        val_index = None
    if splits.get("test"):
        write_labelme_index(
            splits.get("test", []), index_file=test_index, source=source
        )
    else:
        test_index = None

    names_list = [str(n).strip() for n in (keypoint_names or []) if str(n).strip()]
    if not names_list:
        names_list = infer_labelme_keypoint_names(
            pairs,
            max_files=int(max_keypoint_files),
            min_count=int(min_keypoint_count),
        )
    if not names_list:
        raise ValueError(
            "Could not infer keypoint names from LabelMe JSONs. "
            "Provide keypoint names explicitly."
        )

    flip_idx = None
    if bool(infer_flip_idx):
        from annolid.segmentation.dino_kpseg.keypoints import infer_flip_idx_from_names

        flip_idx = infer_flip_idx_from_names(names_list, kpt_count=len(names_list))

    spec_out = spec_path or (dataset_root / "labelme_spec.yaml")
    spec_file = build_labelme_spec(
        dataset_root=dataset_root,
        train_index=train_index if splits.get("train") else None,
        val_index=val_index,
        test_index=test_index,
        keypoint_names=names_list,
        kpt_dims=int(kpt_dims),
        flip_idx=flip_idx,
        output_yaml=Path(spec_out),
    )
    return {
        "spec_path": str(spec_file),
        "split_counts": {
            "train": len(splits.get("train", [])),
            "val": len(splits.get("val", [])),
            "test": len(splits.get("test", [])),
        },
        "train_index": str(train_index),
        "val_index": str(val_index) if val_index else None,
        "test_index": str(test_index) if test_index else None,
        "keypoint_names": names_list,
        "flip_idx": flip_idx,
    }


def build_labelme_spec(
    *,
    dataset_root: Path,
    train_index: Optional[Path],
    val_index: Optional[Path],
    test_index: Optional[Path],
    keypoint_names: List[str],
    kpt_dims: int = 3,
    flip_idx: Optional[List[int]] = None,
    output_yaml: Path,
) -> Path:
    payload: Dict[str, object] = {
        "format": "labelme",
        "path": str(Path(dataset_root).expanduser().resolve()),
        "kpt_shape": [int(len(keypoint_names)), int(kpt_dims)],
        "keypoint_names": list(keypoint_names),
    }
    if train_index:
        payload["train"] = str(Path(train_index).name)
    if val_index:
        payload["val"] = str(Path(val_index).name)
    if test_index:
        payload["test"] = str(Path(test_index).name)
    if flip_idx:
        payload["flip_idx"] = list(flip_idx)

    output_yaml = Path(output_yaml).expanduser().resolve()
    _ensure_dir(output_yaml.parent)
    output_yaml.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output_yaml


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
    progress_callback: Optional[Callable[[Dict[str, object]], None]] = None,
    stop_event=None,
    event_sample_limit: int = 25,
) -> Dict[str, object]:
    """Scan sources and append labeled image/json absolute paths to an index file.

    Returns:
        Backward-compatible counters plus bot-friendly structured fields:
        - total_json, processed, stopped
        - skip_reasons: per-reason counts
        - events_sample: bounded list of processed/skipped examples
    """
    appended = 0
    skipped = 0
    missing_image = 0
    processed = 0
    stopped = False
    limit = max(0, int(event_sample_limit))
    skip_reasons: Dict[str, int] = {
        "duplicate_image": 0,
        "empty_labels": 0,
        "missing_image": 0,
        "invalid_json": 0,
    }
    events_sample: List[Dict[str, object]] = []

    indexed_images: Set[str] = set()
    if dedupe:
        indexed_images = load_indexed_image_paths(index_file)

    def _safe_abs(path: Path) -> str:
        candidate = Path(path).expanduser()
        try:
            return str(candidate.resolve())
        except OSError:
            return str(candidate)

    source_paths = [Path(src) for src in sources]
    all_json_files: List[Path] = []
    for src in source_paths:
        all_json_files.extend(
            list(iter_labelme_json_files(Path(src), recursive=recursive))
        )
    all_json_files = sorted(
        all_json_files, key=lambda p: str(Path(p).expanduser()).lower()
    )
    total_json = len(all_json_files)

    for json_path in all_json_files:
        if stop_event is not None and stop_event.is_set():
            stopped = True
            break
        processed += 1
        payload: Optional[Dict[str, object]] = None
        sidecar_image = _find_sidecar_image(json_path)
        if (not include_empty) or sidecar_image is None:
            payload = _load_labelme_payload(json_path)
        image_path = (
            sidecar_image
            if sidecar_image is not None
            else resolve_image_path(json_path, payload=payload)
        )
        event: Dict[str, object] = {
            "json_path": _safe_abs(Path(json_path)),
            "status": "skipped",
        }
        if image_path is not None:
            event["image_path"] = _safe_abs(Path(image_path))
        else:
            event["image_path"] = None

        reason = ""
        if image_path is None:
            reason = "missing_image"
            missing_image += 1
            skipped += 1
            skip_reasons[reason] += 1
        else:
            image_abs = str(Path(image_path).expanduser().resolve())
            if dedupe and image_abs in indexed_images:
                reason = "duplicate_image"
                skipped += 1
                skip_reasons[reason] += 1
            elif (not include_empty) and payload is None:
                reason = "invalid_json"
                skipped += 1
                skip_reasons[reason] += 1
            elif (not include_empty) and (
                not is_labeled_labelme_json(json_path, payload=payload)
            ):
                reason = "empty_labels"
                skipped += 1
                skip_reasons[reason] += 1
            else:
                record = build_label_index_record(
                    image_path=Path(image_path),
                    json_path=Path(json_path),
                    source=source,
                    payload=payload,
                )
                append_label_index_record(Path(index_file), record)
                appended += 1
                event["status"] = "appended"
                if dedupe:
                    indexed_images.add(image_abs)

        if reason:
            event["reason"] = reason

        if len(events_sample) < limit:
            events_sample.append(event)

        if progress_callback is not None and total_json > 0:
            progress_payload = {
                "total_json": total_json,
                "processed": processed,
                "appended": appended,
                "skipped": skipped,
                "missing_image": missing_image,
                "skip_reasons": dict(skip_reasons),
                "percent": int(round((processed / total_json) * 100)),
            }
            progress_callback(progress_payload)

    summary: Dict[str, object] = {
        "appended": appended,
        "skipped": skipped,
        "missing_image": missing_image,
        "total_json": total_json,
        "processed": processed,
        "stopped": stopped,
        "skip_reasons": skip_reasons,
        "events_sample": events_sample,
    }
    return summary


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
    dataset_root = Path(args.dataset_root).expanduser().resolve()
    index_file = resolve_label_index_path(
        Path(args.index_file), dataset_root=dataset_root
    )

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
