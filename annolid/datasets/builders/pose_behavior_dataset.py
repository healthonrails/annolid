"""Automated build pipeline for pose + behavior datasets.

This orchestrates LabelMe annotation export, schema-aware class mapping,
validation/test splits, and summary reporting so users can go from
annotations to a YOLO-compatible dataset with minimal manual steps.
"""

from __future__ import annotations

import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

from annolid.annotation.labelme2yolo import Labelme2YOLO
from annolid.core.behavior.spec import ProjectSchema, load_behavior_spec
from annolid.utils.logger import logger


SUMMARY_FILENAME = "dataset_summary.json"


def build_pose_behavior_dataset(
    source_dir: Path,
    *,
    output_dir: Optional[Path] = None,
    schema_path: Optional[Path] = None,
    class_map_path: Optional[Path] = None,
    dataset_name: str = "pose_behavior_dataset",
    val_size: float = 0.1,
    test_size: float = 0.1,
    include_visibility: bool = False,
) -> Dict[str, object]:
    """Create a YOLO dataset from LabelMe annotations with schema integration."""
    source_dir = Path(source_dir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Annotation directory not found: {source_dir}")

    resolved_schema, schema_location = _resolve_schema(schema_path, source_dir)
    converter = Labelme2YOLO(
        str(source_dir),
        yolo_dataset_name=dataset_name,
        include_visibility=include_visibility,
        class_map=_load_class_map(class_map_path) if class_map_path else None,
        schema_path=str(schema_location) if schema_location else None,
    )
    if resolved_schema and converter.project_schema is None:
        converter.project_schema = resolved_schema
        if schema_location:
            # type: ignore[attr-defined]
            converter._schema_path = schema_location

    logger.info(
        "Starting dataset build for %s (val_size=%.2f, test_size=%.2f)",
        source_dir,
        val_size,
        test_size,
    )
    converter.convert(val_size=val_size, test_size=test_size)

    dataset_root = source_dir / dataset_name
    if output_dir:
        output_dir = Path(output_dir).resolve()
        if output_dir != dataset_root:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            shutil.copytree(dataset_root, output_dir, dirs_exist_ok=False)
            dataset_root = output_dir

    summary = _summarize_dataset(dataset_root, schema=resolved_schema)
    summary_path = dataset_root / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Dataset summary written to %s", summary_path)
    return summary


def _resolve_schema(
    schema_path: Optional[Path],
    source_dir: Path,
) -> Tuple[Optional[ProjectSchema], Optional[Path]]:
    if schema_path:
        schema_path = Path(schema_path)
        if not schema_path.exists():
            raise FileNotFoundError(f"Schema file not found: {schema_path}")
        schema, _ = load_behavior_spec(path=schema_path)
        return schema, schema_path

    candidate = Labelme2YOLO._discover_schema_path(source_dir)  # type: ignore[attr-defined]
    if candidate:
        try:
            schema, _ = load_behavior_spec(path=candidate)
        except Exception as exc:
            logger.warning("Failed to load schema at %s: %s", candidate, exc)
            return None, None
        return schema, candidate
    return None, None


def _load_class_map(class_map_path: Path):
    return Labelme2YOLO.load_class_map_file(str(class_map_path))


def _summarize_dataset(
    dataset_root: Path, schema: Optional[ProjectSchema]
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "dataset_root": str(dataset_root),
        "splits": {},
        "class_counts": {},
        "subject_counts": {},
    }
    labels_dir = dataset_root / "labels"
    if not labels_dir.exists():
        return summary

    names = _load_names(dataset_root / "data.yaml")
    class_counts: Counter = Counter()
    subject_counts: Counter = Counter()
    split_counts: Dict[str, int] = {}

    for split in ("train", "val", "test"):
        split_dir = labels_dir / split
        if not split_dir.exists():
            continue
        files = sorted(split_dir.glob("*.txt"))
        split_counts[split] = len(files)
        for label_file in files:
            class_ids = _parse_label_file(label_file)
            class_counts.update(class_ids)
            subj_file = label_file.with_suffix(".subjects.json")
            if subj_file.exists():
                try:
                    data = json.loads(subj_file.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    logger.warning("Failed to parse subject metadata: %s", subj_file)
                else:
                    subject_counts.update(
                        entry["subject_id"]
                        for entry in data.get("subjects", [])
                        if entry.get("subject_id")
                    )

    summary["splits"] = split_counts
    summary["class_counts"] = {
        names[class_id] if class_id in names else str(class_id): count
        for class_id, count in sorted(class_counts.items())
    }
    if schema:
        subject_name_map = {subj.id: subj.name or subj.id for subj in schema.subjects}
    else:
        subject_name_map = defaultdict(lambda: None)  # type: ignore
    summary["subject_counts"] = {
        subject_name_map.get(subject_id, subject_id): count
        for subject_id, count in sorted(subject_counts.items())
    }
    return summary


def _parse_label_file(label_file: Path) -> Iterable[int]:
    try:
        with label_file.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    class_id = int(line.split()[0])
                except (ValueError, IndexError):
                    continue
                yield class_id
    except OSError as exc:
        logger.warning("Failed to read label file %s: %s", label_file, exc)


def _load_names(data_yaml_path: Path) -> Dict[int, str]:
    if not data_yaml_path.exists():
        return {}
    try:
        import yaml  # type: ignore

        payload = yaml.safe_load(data_yaml_path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - optional dependency
        logger.warning("Failed to load data.yaml names: %s", exc)
        return {}
    names = payload.get("names", {})
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}
