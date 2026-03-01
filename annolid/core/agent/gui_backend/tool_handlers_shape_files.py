from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from annolid.utils.annotation_store import AnnotationStore, load_labelme_json

_ALLOWED_SHAPE_TYPES = {
    "polygon",
    "rectangle",
    "circle",
    "line",
    "point",
    "linestrip",
    "mask",
}


def _normalize_shape_type(shape_type: str) -> str:
    return str(shape_type or "").strip().lower()


def _resolve_user_path(path_text: str, *, workspace: Path) -> Path:
    candidate = Path(str(path_text or "").strip()).expanduser()
    if not candidate.is_absolute():
        candidate = (workspace / candidate).expanduser()
    return candidate.resolve()


def _normalize_allowed_roots(
    *,
    workspace: Path,
    allowed_roots: Optional[Iterable[str | Path]] = None,
) -> List[Path]:
    roots: List[Path] = []
    seen: set[str] = set()
    for candidate in [workspace, *(allowed_roots or [])]:
        try:
            resolved = Path(candidate).expanduser().resolve()
        except Exception:
            continue
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        roots.append(resolved)
    return roots


def _is_path_within(path: Path, root: Path) -> bool:
    try:
        return path == root or root in path.parents
    except Exception:
        return False


def _validate_path_allowed(path: Path, *, allowed_roots: List[Path]) -> None:
    if any(_is_path_within(path, root) for root in allowed_roots):
        return
    roots_text = ", ".join(str(root) for root in allowed_roots) or "<none>"
    raise PermissionError(
        f"Path is outside allowed directories: {path}. Allowed: {roots_text}"
    )


def _coerce_optional_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_json_dict(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _resolve_annotation_target(
    *,
    path_text: str,
    frame: Optional[int],
    workspace: Path,
    allowed_roots: Optional[Iterable[str | Path]] = None,
) -> Dict[str, Any]:
    normalized_roots = _normalize_allowed_roots(
        workspace=workspace, allowed_roots=allowed_roots
    )
    path = _resolve_user_path(path_text, workspace=workspace)
    _validate_path_allowed(path, allowed_roots=normalized_roots)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    if path.suffix.lower() == ".json":
        payload = _read_json_dict(path)
        if "annotation_store" in payload and not isinstance(
            payload.get("shapes"), list
        ):
            store_name = str(payload.get("annotation_store") or "").strip()
            store = AnnotationStore.for_frame_path(path, store_name or None)
            _validate_path_allowed(
                store.store_path.resolve(), allowed_roots=normalized_roots
            )
            resolved_frame = _coerce_optional_int(frame)
            if resolved_frame is None:
                resolved_frame = _coerce_optional_int(payload.get("frame"))
            if resolved_frame is None:
                resolved_frame = AnnotationStore.frame_number_from_path(path)
            if resolved_frame is None:
                raise ValueError(f"Frame is required for annotation-store stub: {path}")
            return {
                "kind": "store",
                "input_path": path,
                "store_path": store.store_path,
                "frame": int(resolved_frame),
                "from_stub": True,
            }
        return {
            "kind": "json",
            "input_path": path,
        }

    if str(path.name).endswith(AnnotationStore.STORE_SUFFIX):
        _validate_path_allowed(path, allowed_roots=normalized_roots)
        return {
            "kind": "store",
            "input_path": path,
            "store_path": path,
            "frame": _coerce_optional_int(frame),
            "from_stub": False,
        }

    raise ValueError(
        "Unsupported annotation format. Provide a LabelMe .json file "
        f"or a *{AnnotationStore.STORE_SUFFIX} file."
    )


def _shape_matches(
    shape: Dict[str, Any],
    *,
    label_contains: str,
    exact_label: str,
    shape_type: str,
) -> bool:
    label = str(shape.get("label") or "")
    shape_type_name = _normalize_shape_type(str(shape.get("shape_type") or ""))
    if exact_label and label != exact_label:
        return False
    if label_contains and label_contains.lower() not in label.lower():
        return False
    if shape_type and shape_type_name != shape_type:
        return False
    return True


def _shape_entries(
    shapes: Iterable[Any],
    *,
    label_contains: str,
    exact_label: str,
    shape_type: str,
    max_results: int,
    include_points: bool,
) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for idx, shape in enumerate(shapes):
        if not isinstance(shape, dict):
            continue
        if not _shape_matches(
            shape,
            label_contains=label_contains,
            exact_label=exact_label,
            shape_type=shape_type,
        ):
            continue
        points = list(shape.get("points") or [])
        entry: Dict[str, Any] = {
            "index": int(idx),
            "label": str(shape.get("label") or ""),
            "shape_type": _normalize_shape_type(str(shape.get("shape_type") or "")),
            "num_points": int(len(points)),
            "group_id": shape.get("group_id"),
            "description": shape.get("description"),
        }
        if include_points:
            entry["points"] = points
        entries.append(entry)
        if len(entries) >= max_results:
            break
    return entries


def _write_json_atomic(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)


def _load_store_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _parse_store_records(lines: List[str]) -> List[Tuple[int, Dict[str, Any]]]:
    records: List[Tuple[int, Dict[str, Any]]] = []
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            payload = json.loads(stripped)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        records.append((idx, payload))
    return records


def _pick_default_frame(records: List[Tuple[int, Dict[str, Any]]]) -> Optional[int]:
    if not records:
        return None
    numeric_frames: List[int] = []
    for _, payload in records:
        frame_value = _coerce_optional_int(payload.get("frame"))
        if frame_value is not None:
            numeric_frames.append(frame_value)
    if numeric_frames:
        return max(numeric_frames)
    latest_payload = records[-1][1]
    return _coerce_optional_int(latest_payload.get("frame"))


def _write_store_lines(path: Path, lines: List[str]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    AnnotationStore._CACHE.pop(path, None)


def list_shapes_in_annotation_tool(
    *,
    path: str,
    frame: Optional[int] = None,
    label_contains: str = "",
    exact_label: str = "",
    shape_type: str = "",
    max_results: int = 200,
    include_points: bool = False,
    workspace: Path,
    allowed_roots: Optional[Iterable[str | Path]] = None,
) -> Dict[str, Any]:
    type_text = _normalize_shape_type(shape_type)
    if type_text and type_text not in _ALLOWED_SHAPE_TYPES:
        return {
            "ok": False,
            "error": f"Unsupported shape_type: {shape_type}",
            "allowed_shape_types": sorted(_ALLOWED_SHAPE_TYPES),
        }
    limit = max(1, min(int(max_results), 500))
    exact = str(exact_label or "").strip()
    contains = str(label_contains or "").strip()

    try:
        target = _resolve_annotation_target(
            path_text=path,
            frame=frame,
            workspace=workspace,
            allowed_roots=allowed_roots,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    try:
        if target["kind"] == "json":
            payload = load_labelme_json(target["input_path"])
            shapes = list(payload.get("shapes") or [])
            entries = _shape_entries(
                shapes,
                label_contains=contains,
                exact_label=exact,
                shape_type=type_text,
                max_results=limit,
                include_points=bool(include_points),
            )
            return {
                "ok": True,
                "path": str(target["input_path"]),
                "source": "labelme_json",
                "total_shapes": int(len(shapes)),
                "returned_count": int(len(entries)),
                "shapes": entries,
            }

        store_path = Path(target["store_path"])
        lines = _load_store_lines(store_path)
        records = _parse_store_records(lines)
        resolved_frame = _coerce_optional_int(target.get("frame"))
        if resolved_frame is None:
            resolved_frame = _pick_default_frame(records)
        if resolved_frame is None:
            return {
                "ok": False,
                "error": f"No readable frame records in annotation store: {store_path}",
            }

        latest_record: Optional[Dict[str, Any]] = None
        for _, payload in records:
            if _coerce_optional_int(payload.get("frame")) == resolved_frame:
                latest_record = payload
        if latest_record is None:
            return {
                "ok": False,
                "error": f"Frame {resolved_frame} not found in annotation store.",
                "store_path": str(store_path),
            }

        shapes = list(latest_record.get("shapes") or [])
        entries = _shape_entries(
            shapes,
            label_contains=contains,
            exact_label=exact,
            shape_type=type_text,
            max_results=limit,
            include_points=bool(include_points),
        )
        return {
            "ok": True,
            "path": str(target["input_path"]),
            "store_path": str(store_path),
            "source": "annotation_store_stub"
            if bool(target.get("from_stub"))
            else "annotation_store",
            "frame": int(resolved_frame),
            "total_shapes": int(len(shapes)),
            "returned_count": int(len(entries)),
            "shapes": entries,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def relabel_shapes_in_annotation_tool(
    *,
    path: str,
    old_label: str,
    new_label: str,
    frame: Optional[int] = None,
    shape_type: str = "",
    apply_all_frames: bool = False,
    dry_run: bool = False,
    workspace: Path,
    allowed_roots: Optional[Iterable[str | Path]] = None,
) -> Dict[str, Any]:
    old = str(old_label or "").strip()
    new = str(new_label or "").strip()
    if not old:
        return {"ok": False, "error": "old_label is required"}
    if not new:
        return {"ok": False, "error": "new_label is required"}
    if old == new:
        return {"ok": False, "error": "old_label and new_label are identical"}

    type_text = _normalize_shape_type(shape_type)
    if type_text and type_text not in _ALLOWED_SHAPE_TYPES:
        return {
            "ok": False,
            "error": f"Unsupported shape_type: {shape_type}",
            "allowed_shape_types": sorted(_ALLOWED_SHAPE_TYPES),
        }

    try:
        target = _resolve_annotation_target(
            path_text=path,
            frame=frame,
            workspace=workspace,
            allowed_roots=allowed_roots,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    def _relabel_list(shapes: List[Any]) -> int:
        changed = 0
        for shape in shapes:
            if not isinstance(shape, dict):
                continue
            if str(shape.get("label") or "") != old:
                continue
            if (
                type_text
                and _normalize_shape_type(str(shape.get("shape_type") or ""))
                != type_text
            ):
                continue
            shape["label"] = new
            changed += 1
        return changed

    try:
        if target["kind"] == "json":
            json_path = Path(target["input_path"])
            payload = _read_json_dict(json_path)
            shapes = list(payload.get("shapes") or [])
            changed_shapes = _relabel_list(shapes)
            if changed_shapes > 0:
                payload["shapes"] = shapes
                if not bool(dry_run):
                    _write_json_atomic(json_path, payload)
            return {
                "ok": True,
                "path": str(json_path),
                "source": "labelme_json",
                "changed_shapes": int(changed_shapes),
                "changed_records": int(1 if changed_shapes > 0 else 0),
                "dry_run": bool(dry_run),
            }

        store_path = Path(target["store_path"])
        lines = _load_store_lines(store_path)
        records = _parse_store_records(lines)
        if not records:
            return {
                "ok": False,
                "error": f"No readable frame records in annotation store: {store_path}",
            }

        target_frame = _coerce_optional_int(target.get("frame"))
        if target_frame is None and not bool(apply_all_frames):
            target_frame = _pick_default_frame(records)
        if target_frame is None and not bool(apply_all_frames):
            return {"ok": False, "error": "frame is required for this annotation store"}

        changed_shapes = 0
        changed_records = 0
        for idx, payload in records:
            row_frame = _coerce_optional_int(payload.get("frame"))
            if not bool(apply_all_frames) and row_frame != target_frame:
                continue
            row_shapes = list(payload.get("shapes") or [])
            row_changed = _relabel_list(row_shapes)
            if row_changed <= 0:
                continue
            changed_shapes += row_changed
            changed_records += 1
            payload["shapes"] = row_shapes
            lines[idx] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

        if changed_records > 0 and not bool(dry_run):
            _write_store_lines(store_path, lines)

        return {
            "ok": True,
            "path": str(target["input_path"]),
            "store_path": str(store_path),
            "source": "annotation_store_stub"
            if bool(target.get("from_stub"))
            else "annotation_store",
            "frame": target_frame,
            "apply_all_frames": bool(apply_all_frames),
            "changed_shapes": int(changed_shapes),
            "changed_records": int(changed_records),
            "dry_run": bool(dry_run),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def delete_shapes_in_annotation_tool(
    *,
    path: str,
    frame: Optional[int] = None,
    label_contains: str = "",
    exact_label: str = "",
    shape_type: str = "",
    max_delete: int = 100000,
    apply_all_frames: bool = False,
    delete_all: bool = False,
    dry_run: bool = False,
    workspace: Path,
    allowed_roots: Optional[Iterable[str | Path]] = None,
) -> Dict[str, Any]:
    contains = str(label_contains or "").strip()
    exact = str(exact_label or "").strip()
    type_text = _normalize_shape_type(shape_type)
    limit = max(1, min(int(max_delete), 1000000))

    if type_text and type_text not in _ALLOWED_SHAPE_TYPES:
        return {
            "ok": False,
            "error": f"Unsupported shape_type: {shape_type}",
            "allowed_shape_types": sorted(_ALLOWED_SHAPE_TYPES),
        }

    has_filter = bool(contains or exact or type_text)
    if not has_filter and not bool(delete_all):
        return {
            "ok": False,
            "error": "Refusing to delete all shapes without filters; set delete_all=true.",
        }

    try:
        target = _resolve_annotation_target(
            path_text=path,
            frame=frame,
            workspace=workspace,
            allowed_roots=allowed_roots,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    def _delete_from_shapes(shapes: List[Any]) -> Tuple[List[Any], int]:
        kept: List[Any] = []
        removed = 0
        for shape in shapes:
            if not isinstance(shape, dict):
                kept.append(shape)
                continue
            should_remove = bool(delete_all) or _shape_matches(
                shape,
                label_contains=contains,
                exact_label=exact,
                shape_type=type_text,
            )
            if should_remove and removed < limit:
                removed += 1
                continue
            kept.append(shape)
        return kept, removed

    try:
        if target["kind"] == "json":
            json_path = Path(target["input_path"])
            payload = _read_json_dict(json_path)
            shapes = list(payload.get("shapes") or [])
            kept, removed = _delete_from_shapes(shapes)
            if removed > 0:
                payload["shapes"] = kept
                if not bool(dry_run):
                    _write_json_atomic(json_path, payload)
            return {
                "ok": True,
                "path": str(json_path),
                "source": "labelme_json",
                "deleted_shapes": int(removed),
                "changed_records": int(1 if removed > 0 else 0),
                "dry_run": bool(dry_run),
            }

        store_path = Path(target["store_path"])
        lines = _load_store_lines(store_path)
        records = _parse_store_records(lines)
        if not records:
            return {
                "ok": False,
                "error": f"No readable frame records in annotation store: {store_path}",
            }

        target_frame = _coerce_optional_int(target.get("frame"))
        if target_frame is None and not bool(apply_all_frames):
            target_frame = _pick_default_frame(records)
        if target_frame is None and not bool(apply_all_frames):
            return {"ok": False, "error": "frame is required for this annotation store"}

        deleted_shapes = 0
        changed_records = 0
        for idx, payload in records:
            row_frame = _coerce_optional_int(payload.get("frame"))
            if not bool(apply_all_frames) and row_frame != target_frame:
                continue
            row_shapes = list(payload.get("shapes") or [])
            kept, removed = _delete_from_shapes(row_shapes)
            if removed <= 0:
                continue
            deleted_shapes += int(removed)
            changed_records += 1
            payload["shapes"] = kept
            lines[idx] = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            if deleted_shapes >= limit:
                break

        if changed_records > 0 and not bool(dry_run):
            _write_store_lines(store_path, lines)

        return {
            "ok": True,
            "path": str(target["input_path"]),
            "store_path": str(store_path),
            "source": "annotation_store_stub"
            if bool(target.get("from_stub"))
            else "annotation_store",
            "frame": target_frame,
            "apply_all_frames": bool(apply_all_frames),
            "deleted_shapes": int(deleted_shapes),
            "changed_records": int(changed_records),
            "dry_run": bool(dry_run),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
