from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List

from annolid.datasets.labelme_collection import iter_label_index_records
from annolid.utils.annotation_store import load_labelme_json


def label_stats_snapshot_path(index_file: Path) -> Path:
    index_path = Path(index_file).expanduser().resolve()
    return index_path.with_suffix(".stats.json")


def _parse_date(value: object) -> date | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).date()
    except ValueError:
        return None


def _extract_annotator(payload: dict, shape: dict | None = None) -> str:
    candidates: List[object] = []
    if isinstance(shape, dict):
        sflags = shape.get("flags")
        if isinstance(sflags, dict):
            candidates.extend(
                [
                    sflags.get("annotator"),
                    sflags.get("user"),
                    sflags.get("editor"),
                ]
            )

    flags = payload.get("flags")
    if isinstance(flags, dict):
        candidates.extend(
            [flags.get("annotator"), flags.get("user"), flags.get("editor")]
        )

    candidates.extend(
        [payload.get("annotator"), payload.get("user"), payload.get("editor")]
    )
    for raw in candidates:
        token = str(raw or "").strip()
        if token:
            return token
    return "unknown"


def _recent_activity_rows(
    day_activity: Dict[date, int], *, today: date
) -> List[tuple[str, int]]:
    rows: List[tuple[str, int]] = []
    for i in range(6, -1, -1):
        d = today - timedelta(days=i)
        rows.append((d.isoformat(), int(day_activity.get(d, 0))))
    return rows


def build_stats_from_label_index(
    *,
    index_file: Path,
    project_root: Path | None = None,
    now: datetime | None = None,
) -> Dict[str, object]:
    index_file = Path(index_file).expanduser().resolve()
    current_dt = now or datetime.now()
    today = current_dt.date()

    record_counts_by_json: Dict[str, int] = defaultdict(int)
    latest_record_by_json: Dict[str, Dict[str, object]] = {}
    image_paths: set[str] = set()
    day_activity: Dict[date, int] = defaultdict(int)
    records_total = 0

    for rec in iter_label_index_records(index_file):
        records_total += 1
        json_path = rec.get("json_path")
        if not isinstance(json_path, str) or not json_path:
            continue

        record_counts_by_json[json_path] += 1
        latest_record_by_json[json_path] = rec

        image_path = rec.get("image_path")
        if isinstance(image_path, str) and image_path:
            image_paths.add(image_path)

        indexed_day = _parse_date(rec.get("indexed_at"))
        if indexed_day is not None:
            day_activity[indexed_day] += 1

    label_counter: Counter[str] = Counter()
    shape_type_counter: Counter[str] = Counter()
    annotator_file_counter: Counter[str] = Counter()
    annotator_shape_counter: Counter[str] = Counter()

    labeled_files = 0
    total_shapes = 0

    for json_path, rec in latest_record_by_json.items():
        payload = None
        path_obj = Path(json_path).expanduser()
        if not path_obj.is_absolute():
            if project_root is not None:
                path_obj = (
                    Path(project_root).expanduser().resolve() / path_obj
                ).resolve()
            else:
                path_obj = path_obj.resolve()

        if path_obj.exists():
            try:
                payload = load_labelme_json(path_obj)
            except Exception:
                payload = None

        shapes = None
        if isinstance(payload, dict):
            shapes_data = payload.get("shapes")
            if isinstance(shapes_data, list):
                shapes = [s for s in shapes_data if isinstance(s, dict)]

        if shapes is not None:
            shape_count = len(shapes)
            file_annotator = _extract_annotator(payload)
            annotator_file_counter[file_annotator] += 1
            for shape in shapes:
                label = str(shape.get("label") or "unknown").strip() or "unknown"
                shape_type = (
                    str(shape.get("shape_type") or "unknown").strip() or "unknown"
                )
                shape_annotator = _extract_annotator(payload, shape=shape)
                label_counter[label] += 1
                shape_type_counter[shape_type] += 1
                annotator_shape_counter[shape_annotator] += 1
        else:
            shape_count = int(rec.get("shapes_count") or 0)
            for label in rec.get("labels") or []:
                token = str(label or "").strip()
                if token:
                    label_counter[token] += 1
            annotator_file_counter["unknown"] += 1

        if shape_count > 0:
            labeled_files += 1
        total_shapes += shape_count

    total_annotation_files = len(latest_record_by_json)
    created_files = total_annotation_files
    edited_files = sum(1 for n in record_counts_by_json.values() if n > 1)

    total_images = len(image_paths)
    coverage = (labeled_files / total_images * 100.0) if total_images > 0 else 0.0

    payload: Dict[str, object] = {
        "schema_version": 1,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "index_file": str(index_file),
        "project_root": str(Path(project_root).expanduser().resolve())
        if project_root
        else None,
        "source": "label_index",
        "records_total": int(records_total),
        "total_images": int(total_images),
        "total_annotation_files": int(total_annotation_files),
        "labeled_files": int(labeled_files),
        "total_shapes": int(total_shapes),
        "created_files": int(created_files),
        "edited_files": int(edited_files),
        "coverage_percent": float(coverage),
        "top_labels": [[k, int(v)] for k, v in label_counter.most_common(8)],
        "shape_type_counts": [
            [k, int(v)] for k, v in shape_type_counter.most_common(8)
        ],
        "annotator_file_counts": [
            [k, int(v)] for k, v in annotator_file_counter.most_common(8)
        ],
        "annotator_shape_counts": [
            [k, int(v)] for k, v in annotator_shape_counter.most_common(8)
        ],
        "activity_last_7_days": _recent_activity_rows(day_activity, today=today),
    }
    return payload


def save_label_stats_snapshot(
    *,
    index_file: Path,
    stats: Dict[str, object],
) -> Path:
    out_path = label_stats_snapshot_path(index_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, ensure_ascii=True, indent=2, sort_keys=True)
        fh.write("\n")
    return out_path


def load_label_stats_snapshot(index_file: Path) -> Dict[str, object] | None:
    out_path = label_stats_snapshot_path(index_file)
    if not out_path.exists():
        return None
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def update_label_stats_snapshot(
    *,
    index_file: Path,
    project_root: Path | None = None,
    now: datetime | None = None,
) -> Dict[str, object]:
    stats = build_stats_from_label_index(
        index_file=index_file, project_root=project_root, now=now
    )
    save_label_stats_snapshot(index_file=index_file, stats=stats)
    return stats
