from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from annolid.utils.annotation_store import AnnotationStore


@dataclass(frozen=True)
class BatchRelabelResult:
    root: Path
    old_label: str
    new_label: str
    dry_run: bool
    json_files_scanned: int
    json_files_updated: int
    store_files_scanned: int
    store_files_updated: int
    shapes_renamed: int
    records_updated: int


def _rename_shapes_in_payload(payload: Dict[str, Any], old: str, new: str) -> int:
    shapes = payload.get("shapes")
    if not isinstance(shapes, list):
        return 0
    changed = 0
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        label = str(shape.get("label") or "")
        if label == old:
            shape["label"] = new
            changed += 1
    return changed


def _write_json_atomic(
    path: Path, payload: Dict[str, Any], *, pretty: bool = True
) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = (
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        if pretty
        else json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _process_labelme_json(
    path: Path, old: str, new: str, *, dry_run: bool
) -> Tuple[int, bool]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return 0, False
    if not isinstance(payload, dict):
        return 0, False
    # Do not rewrite annotation-store stubs that only reference NDJSON.
    if "annotation_store" in payload and not isinstance(payload.get("shapes"), list):
        return 0, False

    changed_shapes = _rename_shapes_in_payload(payload, old, new)
    if changed_shapes <= 0:
        return 0, False
    if not dry_run:
        _write_json_atomic(path, payload)
    return changed_shapes, True


def _process_annotation_store(
    path: Path, old: str, new: str, *, dry_run: bool
) -> Tuple[int, int, bool]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return 0, 0, False

    changed_shapes = 0
    changed_records = 0
    out_lines: List[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
        try:
            payload = json.loads(stripped)
        except Exception:
            out_lines.append(line)
            continue
        if not isinstance(payload, dict):
            out_lines.append(line)
            continue

        changed = _rename_shapes_in_payload(payload, old, new)
        if changed > 0:
            changed_records += 1
            changed_shapes += changed
            out_lines.append(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            )
        else:
            out_lines.append(line)

    if changed_records <= 0:
        return 0, 0, False

    if not dry_run:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
        tmp.replace(path)
        AnnotationStore._CACHE.pop(path, None)
    return changed_shapes, changed_records, True


def collect_label_counts(
    *,
    root: Path | str,
    include_json_files: bool = True,
    include_annotation_stores: bool = True,
) -> Dict[str, int]:
    root_path = Path(root).expanduser().resolve()
    counts: Dict[str, int] = {}

    def _bump(label: str) -> None:
        token = str(label or "").strip()
        if not token:
            token = "unknown"
        counts[token] = int(counts.get(token, 0)) + 1

    if include_json_files:
        for path in root_path.rglob("*.json"):
            if not path.is_file():
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            if "annotation_store" in payload and not isinstance(
                payload.get("shapes"), list
            ):
                continue
            shapes = payload.get("shapes")
            if not isinstance(shapes, list):
                continue
            for shape in shapes:
                if not isinstance(shape, dict):
                    continue
                _bump(str(shape.get("label") or ""))

    if include_annotation_stores:
        pattern = f"*{AnnotationStore.STORE_SUFFIX}"
        for path in root_path.rglob(pattern):
            if not path.is_file():
                continue
            try:
                lines = path.read_text(encoding="utf-8").splitlines()
            except Exception:
                continue
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    payload = json.loads(stripped)
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
                    _bump(str(shape.get("label") or ""))
    return counts


def run_batch_relabel(
    *,
    root: Path | str,
    old_label: str,
    new_label: str,
    include_json_files: bool = True,
    include_annotation_stores: bool = True,
    dry_run: bool = False,
) -> BatchRelabelResult:
    root_path = Path(root).expanduser().resolve()
    old = str(old_label or "").strip()
    new = str(new_label or "").strip()
    if not old:
        raise ValueError("old_label must be non-empty")
    if not new:
        raise ValueError("new_label must be non-empty")
    if old == new:
        raise ValueError("old_label and new_label are identical")

    json_scanned = 0
    json_updated = 0
    store_scanned = 0
    store_updated = 0
    shapes_renamed = 0
    records_updated = 0

    if include_json_files:
        for path in root_path.rglob("*.json"):
            if not path.is_file():
                continue
            json_scanned += 1
            changed_shapes, changed_file = _process_labelme_json(
                path, old, new, dry_run=dry_run
            )
            shapes_renamed += int(changed_shapes)
            if changed_file:
                json_updated += 1

    if include_annotation_stores:
        pattern = f"*{AnnotationStore.STORE_SUFFIX}"
        for path in root_path.rglob(pattern):
            if not path.is_file():
                continue
            store_scanned += 1
            changed_shapes, changed_rows, changed_file = _process_annotation_store(
                path, old, new, dry_run=dry_run
            )
            shapes_renamed += int(changed_shapes)
            records_updated += int(changed_rows)
            if changed_file:
                store_updated += 1

    return BatchRelabelResult(
        root=root_path,
        old_label=old,
        new_label=new,
        dry_run=bool(dry_run),
        json_files_scanned=int(json_scanned),
        json_files_updated=int(json_updated),
        store_files_scanned=int(store_scanned),
        store_files_updated=int(store_updated),
        shapes_renamed=int(shapes_renamed),
        records_updated=int(records_updated),
    )
