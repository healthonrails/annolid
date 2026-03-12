from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import yaml

from annolid.datasets.coco import (
    build_coco_spec_from_dataset_path,
    infer_coco_task,
    materialize_coco_spec_as_yolo,
)
from annolid.datasets.importers.deeplabcut_training_data import (
    DeepLabCutTrainingImportConfig,
)
from annolid.datasets.labelme_collection import (
    DEFAULT_LABEL_INDEX_DIRNAME,
    DEFAULT_LABEL_INDEX_NAME,
    generate_labelme_spec_and_splits,
    index_labelme_dataset,
    infer_split_pairs_from_sources,
    iter_labelme_pairs,
)
from annolid.services.export import (
    build_yolo_dataset_from_index,
    import_deeplabcut_dataset,
)
from annolid.utils.annotation_store import load_labelme_json

from .common import (
    _normalize_allowed_read_roots,
    _resolve_read_path,
    _resolve_write_path,
)
from .function_base import FunctionTool


def _read_yaml_dict(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return payload


def _resolve_yaml_output_path(
    raw_output: str,
    *,
    default_path: Path,
    allowed_dir: Path | None,
) -> Path:
    if not str(raw_output or "").strip():
        return default_path
    resolved = _resolve_write_path(raw_output, allowed_dir=allowed_dir)
    if resolved.exists() and resolved.is_dir():
        return resolved / default_path.name
    if resolved.suffix.lower() in {".yml", ".yaml"}:
        return resolved
    return resolved / default_path.name


def _shape_counts_from_labelme(
    json_paths: Sequence[Path], *, max_files: int = 200
) -> dict[str, int]:
    counts: dict[str, int] = {}
    inspected = 0
    for json_path in json_paths:
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
            shape_type = (
                str(shape.get("shape_type") or "polygon").strip().lower() or "polygon"
            )
            counts[shape_type] = counts.get(shape_type, 0) + 1
    return counts


def _infer_coco_payload(dataset_root: Path) -> dict[str, object] | None:
    payload = build_coco_spec_from_dataset_path(dataset_root)
    if payload is None or not isinstance(payload, dict):
        return None
    return payload


def _infer_coco_task(payload: dict[str, object], *, config_path: Path) -> str | None:
    try:
        return infer_coco_task(config_path=config_path, payload=payload)
    except Exception:
        return None


def _find_trainable_specs(dataset_root: Path) -> dict[str, str]:
    specs: dict[str, str] = {}
    labelme_spec = dataset_root / "labelme_spec.yaml"
    if labelme_spec.exists():
        specs["labelme_spec"] = str(labelme_spec.resolve())
    coco_spec = dataset_root / "coco_spec.yaml"
    if coco_spec.exists():
        specs["coco_spec"] = str(coco_spec.resolve())
    data_yaml = dataset_root / "data.yaml"
    if data_yaml.exists():
        specs["data_yaml"] = str(data_yaml.resolve())
    for candidate in sorted(dataset_root.rglob("data.yaml")):
        if candidate.resolve() == data_yaml.resolve() if data_yaml.exists() else False:
            continue
        specs.setdefault("nested_data_yaml", str(candidate.resolve()))
        break
    return specs


def _detect_external_dataset_kinds(dataset_root: Path) -> list[str]:
    root = Path(dataset_root).expanduser().resolve()
    kinds: list[str] = []
    labeled_data = root / "labeled-data"
    if labeled_data.exists() and any(labeled_data.rglob("CollectedData_*.csv")):
        kinds.append("deeplabcut")
    coco_payload = _infer_coco_payload(root)
    annotations_dir = root / "annotations"
    if annotations_dir.exists():
        if any(annotations_dir.glob("instances_*.json")):
            kinds.append("coco_instances")
        if any(annotations_dir.glob("person_keypoints_*.json")):
            kinds.append("coco_keypoints")
    if coco_payload is not None and not any(kind.startswith("coco_") for kind in kinds):
        kinds.append("coco")
    return kinds


def _recommend_models_from_summary(
    *,
    shape_counts: dict[str, int],
    external_kinds: Sequence[str],
    coco_task: str | None,
    yolo_cfg: dict[str, object] | None,
    labelme_cfg: dict[str, object] | None,
) -> list[dict[str, str]]:
    recommendations: list[dict[str, str]] = []
    has_points = bool(shape_counts.get("point"))
    if labelme_cfg:
        recommendations.append(
            {
                "model": "dino_kpseg",
                "task": "keypoint_segmentation",
                "reason": "LabelMe pose spec is available for DinoKPSEG training.",
            }
        )
    if "coco_keypoints" in external_kinds or coco_task == "pose":
        recommendations.append(
            {
                "model": "dino_kpseg",
                "task": "keypoint_segmentation",
                "reason": "COCO pose datasets can train DinoKPSEG through a COCO spec.",
            }
        )
        recommendations.append(
            {
                "model": "yolo",
                "task": "pose",
                "reason": "COCO pose datasets can be materialized into a YOLO pose dataset.",
            }
        )
        recommendations.append(
            {
                "model": "keypoint_rcnn",
                "task": "pose",
                "reason": "COCO keypoint annotations are compatible with keypoint detection workflows.",
            }
        )
    elif "coco_instances" in external_kinds or coco_task == "detect":
        recommendations.append(
            {
                "model": "yolo",
                "task": "detection",
                "reason": "COCO detection datasets can be materialized into a YOLO detection dataset.",
            }
        )
        recommendations.append(
            {
                "model": "maskrcnn_detectron2",
                "task": "segmentation",
                "reason": "COCO instance annotations are compatible with Detectron2/Mask R-CNN workflows.",
            }
        )
    if yolo_cfg is not None:
        kpt_shape = yolo_cfg.get("kpt_shape")
        if isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 2:
            recommendations.append(
                {
                    "model": "yolo",
                    "task": "pose",
                    "reason": "YOLO pose data.yaml is available with kpt_shape.",
                }
            )
            recommendations.append(
                {
                    "model": "keypoint_rcnn",
                    "task": "pose",
                    "reason": "Pose dataset metadata is present and can support keypoint models.",
                }
            )
        else:
            shape_task = "segmentation" if shape_counts.get("polygon") else "detection"
            recommendations.append(
                {
                    "model": "yolo",
                    "task": shape_task,
                    "reason": "YOLO data.yaml is available for direct training.",
                }
            )
    elif "deeplabcut" in external_kinds:
        recommendations.append(
            {
                "model": "dino_kpseg",
                "task": "keypoint_segmentation",
                "reason": "DeepLabCut training data can be imported into LabelMe pose annotations for DinoKPSEG training.",
            }
        )
        recommendations.append(
            {
                "model": "yolo",
                "task": "pose",
                "reason": "DeepLabCut training data can be imported, indexed, and exported into a YOLO pose dataset.",
            }
        )
    elif has_points:
        recommendations.append(
            {
                "model": "dino_kpseg",
                "task": "keypoint_segmentation",
                "reason": "Raw LabelMe folder contains point annotations but still needs a spec file.",
            }
        )
        recommendations.append(
            {
                "model": "yolo",
                "task": "pose",
                "reason": "Raw LabelMe folder contains point annotations and can be exported to YOLO pose format.",
            }
        )
    elif shape_counts.get("polygon"):
        recommendations.append(
            {
                "model": "yolo",
                "task": "segmentation",
                "reason": "Raw LabelMe folder contains polygon annotations and can be exported to YOLO segmentation format.",
            }
        )
    return recommendations


def inspect_dataset_folder(dataset_root: Path) -> dict[str, object]:
    root = Path(dataset_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Expected a dataset folder, got: {root}")

    trainable_specs = _find_trainable_specs(root)
    labelme_cfg = None
    if "labelme_spec" in trainable_specs:
        try:
            labelme_cfg = _read_yaml_dict(Path(trainable_specs["labelme_spec"]))
        except Exception:
            labelme_cfg = None
    yolo_cfg = None
    yolo_cfg_path = trainable_specs.get("data_yaml") or trainable_specs.get(
        "nested_data_yaml"
    )
    if yolo_cfg_path:
        try:
            yolo_cfg = _read_yaml_dict(Path(yolo_cfg_path))
        except Exception:
            yolo_cfg = None
    coco_cfg = None
    coco_cfg_path = trainable_specs.get("coco_spec")
    if coco_cfg_path:
        try:
            coco_cfg = _read_yaml_dict(Path(coco_cfg_path))
        except Exception:
            coco_cfg = None
    else:
        coco_cfg = _infer_coco_payload(root)
    inferred_coco_task = (
        _infer_coco_task(
            coco_cfg,
            config_path=Path(coco_cfg_path)
            if coco_cfg_path
            else root / "coco_spec.yaml",
        )
        if coco_cfg is not None
        else None
    )

    external_kinds = _detect_external_dataset_kinds(root)
    labelme_pairs = iter_labelme_pairs([root], recursive=True, include_empty=True)
    labelme_jsons = [pair[0] for pair in labelme_pairs]
    labeled_pairs = iter_labelme_pairs([root], recursive=True, include_empty=False)
    inferred_splits = (
        infer_split_pairs_from_sources(labelme_pairs, sources=[root])
        if labelme_pairs
        else {}
    )
    shape_counts = _shape_counts_from_labelme(labelme_jsons)

    dataset_kinds: list[str] = []
    if labelme_jsons:
        dataset_kinds.append("labelme")
    dataset_kinds.extend(kind for kind in external_kinds if kind not in dataset_kinds)
    if yolo_cfg is not None:
        dataset_kinds.append("yolo")
    if coco_cfg is not None:
        dataset_kinds.append("coco_spec" if coco_cfg_path else "coco")
    if labelme_cfg is not None:
        dataset_kinds.append("labelme_spec")

    has_saved_trainable_spec = bool(labelme_cfg or yolo_cfg or coco_cfg_path)
    ready_for_training = has_saved_trainable_spec
    next_actions: list[str] = []
    if not has_saved_trainable_spec and "deeplabcut" in external_kinds:
        next_actions.append(
            "Run annolid_dataset_prepare with mode=deeplabcut_import to convert DeepLabCut training data into LabelMe plus an Annolid index."
        )
    if coco_cfg is not None and not coco_cfg_path:
        next_actions.append(
            "Run annolid_dataset_prepare with mode=coco_spec to write a reusable COCO spec YAML for training."
        )
        next_actions.append(
            "DinoKPSEG training can also auto-stage an inferred COCO spec when you call annolid_train_start with dataset_folder."
        )
        next_actions.append(
            "For YOLO training from COCO, run annolid_dataset_prepare with mode=coco_to_yolo first."
        )
    if not has_saved_trainable_spec and labelme_jsons:
        if shape_counts.get("point"):
            next_actions.append(
                "Run annolid_dataset_prepare with mode=labelme_spec to generate LabelMe train/val/test spec files."
            )
            next_actions.append(
                "Run annolid_dataset_prepare with mode=yolo_from_labelme to export a YOLO pose dataset."
            )
        else:
            next_actions.append(
                "Run annolid_dataset_prepare with mode=yolo_from_labelme to export a YOLO dataset from LabelMe annotations."
            )
    if ready_for_training:
        next_actions.append(
            "Use annolid_train_start with dataset_folder set to this path to auto-resolve the trainable dataset config."
        )

    return {
        "dataset_root": str(root),
        "dataset_kinds": dataset_kinds,
        "trainable_specs": trainable_specs,
        "ready_for_training": ready_for_training,
        "inferred_formats": ["coco"]
        if coco_cfg is not None and not coco_cfg_path
        else [],
        "external_formats": external_kinds,
        "labelme": {
            "json_files": len(labelme_jsons),
            "labeled_pairs": len(labeled_pairs),
            "shape_type_counts": shape_counts,
            "inferred_split_counts": {
                "train": len(inferred_splits.get("train", [])),
                "val": len(inferred_splits.get("val", [])),
                "test": len(inferred_splits.get("test", [])),
            }
            if inferred_splits
            else {},
        },
        "labelme_spec": {
            "path": trainable_specs.get("labelme_spec"),
            "kpt_shape": labelme_cfg.get("kpt_shape") if labelme_cfg else None,
            "keypoint_names": list(labelme_cfg.get("keypoint_names", []))
            if isinstance(labelme_cfg.get("keypoint_names"), list)
            else [],
        }
        if labelme_cfg
        else None,
        "coco_spec": {
            "path": coco_cfg_path,
            "format": coco_cfg.get("format") if coco_cfg else None,
            "task": (inferred_coco_task),
            "train": coco_cfg.get("train") if coco_cfg else None,
            "val": coco_cfg.get("val") if coco_cfg else None,
            "test": coco_cfg.get("test") if coco_cfg else None,
        }
        if coco_cfg
        else None,
        "yolo": {
            "path": yolo_cfg_path,
            "kpt_shape": yolo_cfg.get("kpt_shape") if yolo_cfg else None,
            "train": yolo_cfg.get("train") if yolo_cfg else None,
            "val": yolo_cfg.get("val") if yolo_cfg else None,
            "test": yolo_cfg.get("test") if yolo_cfg else None,
            "names": yolo_cfg.get("names") if yolo_cfg else None,
        }
        if yolo_cfg is not None
        else None,
        "recommended_models": _recommend_models_from_summary(
            shape_counts=shape_counts,
            external_kinds=external_kinds,
            coco_task=inferred_coco_task,
            yolo_cfg=yolo_cfg,
            labelme_cfg=labelme_cfg,
        ),
        "next_actions": next_actions,
    }


class AnnolidDatasetInspectTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir,
            allowed_read_roots,
        )

    @property
    def name(self) -> str:
        return "annolid_dataset_inspect"

    @property
    def description(self) -> str:
        return (
            "Inspect a dataset folder, detect whether it already contains a trainable "
            "spec such as data.yaml, labelme_spec.yaml, or coco_spec.yaml, summarize "
            "raw datasets, and recommend the next training-preparation step."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_folder": {"type": "string", "minLength": 1},
            },
            "required": ["dataset_folder"],
        }

    async def execute(self, dataset_folder: str, **kwargs: Any) -> str:
        del kwargs
        try:
            root = _resolve_read_path(
                dataset_folder,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            report = inspect_dataset_folder(root)
            return json.dumps({"ok": True, "dataset": report}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False)


class AnnolidDatasetPrepareTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir,
            allowed_read_roots,
        )

    @property
    def name(self) -> str:
        return "annolid_dataset_prepare"

    @property
    def description(self) -> str:
        return (
            "Prepare a dataset folder for training. Supports generating a LabelMe "
            "spec with train/val/test splits, inferring a COCO spec, importing "
            "DeepLabCut data, or exporting datasets into YOLO format."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "dataset_folder": {"type": "string", "minLength": 1},
                "mode": {
                    "type": "string",
                    "enum": [
                        "labelme_spec",
                        "yolo_from_labelme",
                        "deeplabcut_import",
                        "coco_spec",
                        "coco_to_yolo",
                    ],
                },
                "allow_mutation": {"type": "boolean"},
                "val_size": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "test_size": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "seed": {"type": "integer"},
                "keypoint_names": {"type": "array", "items": {"type": "string"}},
                "kpt_dims": {"type": "integer", "enum": [2, 3]},
                "task": {"type": "string", "enum": ["auto", "pose", "segmentation"]},
                "dataset_name": {"type": "string"},
                "output_dir": {"type": "string"},
                "include_empty": {"type": "boolean"},
                "labeled_data": {"type": "string"},
                "instance_label": {"type": "string"},
                "overwrite": {"type": "boolean"},
                "write_pose_schema": {"type": "boolean"},
                "pose_schema_out": {"type": "string"},
                "pose_schema_preset": {"type": "string"},
                "instance_separator": {"type": "string"},
                "write_index": {"type": "boolean"},
                "index_file": {"type": "string"},
            },
            "required": ["dataset_folder", "mode"],
        }

    async def execute(
        self,
        dataset_folder: str,
        mode: str,
        allow_mutation: bool = False,
        val_size: float = 0.1,
        test_size: float = 0.0,
        seed: int = 0,
        keypoint_names: list[str] | None = None,
        kpt_dims: int = 3,
        task: str = "auto",
        dataset_name: str = "YOLO_dataset",
        output_dir: str = "",
        include_empty: bool = False,
        labeled_data: str = "labeled-data",
        instance_label: str = "mouse",
        overwrite: bool = False,
        write_pose_schema: bool = True,
        pose_schema_out: str = "",
        pose_schema_preset: str = "mouse",
        instance_separator: str = "_",
        write_index: bool = True,
        index_file: str = "",
        **kwargs: Any,
    ) -> str:
        del kwargs
        if not allow_mutation:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        "Preparing a dataset modifies files. Retry with "
                        "allow_mutation=true only when that is intended."
                    ),
                    "dataset_folder": dataset_folder,
                    "mode": mode,
                },
                ensure_ascii=False,
            )
        try:
            root = _resolve_read_path(
                dataset_folder,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            mode_name = str(mode or "").strip().lower()
            if mode_name == "labelme_spec":
                result = generate_labelme_spec_and_splits(
                    sources=[root],
                    dataset_root=root,
                    recursive=True,
                    include_empty=bool(include_empty),
                    val_size=float(val_size),
                    test_size=float(test_size),
                    seed=int(seed),
                    keypoint_names=[
                        str(name).strip()
                        for name in (keypoint_names or [])
                        if str(name).strip()
                    ]
                    or None,
                    kpt_dims=int(kpt_dims),
                    infer_flip_idx=True,
                    source="annolid_bot",
                )
            elif mode_name == "yolo_from_labelme":
                index_dir = root / DEFAULT_LABEL_INDEX_DIRNAME
                index_file = index_dir / DEFAULT_LABEL_INDEX_NAME
                index_summary = index_labelme_dataset(
                    [root],
                    index_file=index_file,
                    recursive=True,
                    include_empty=bool(include_empty),
                    dedupe=True,
                    source="annolid_bot",
                )
                destination = (
                    _resolve_write_path(output_dir, allowed_dir=self._allowed_dir)
                    if str(output_dir or "").strip()
                    else root
                )
                export_summary = build_yolo_dataset_from_index(
                    index_file=index_file,
                    output_dir=destination,
                    dataset_name=str(dataset_name or "YOLO_dataset").strip()
                    or "YOLO_dataset",
                    val_size=float(val_size),
                    test_size=float(test_size),
                    task=str(task or "auto").strip().lower() or "auto",
                    include_empty=bool(include_empty),
                )
                result = {
                    "index_file": str(index_file.resolve()),
                    "index_summary": index_summary,
                    **export_summary,
                }
            elif mode_name == "deeplabcut_import":
                schema_out_path = (
                    Path(pose_schema_out)
                    if str(pose_schema_out or "").strip()
                    else Path(str(labeled_data or "labeled-data")) / "pose_schema.json"
                )
                import_summary = import_deeplabcut_dataset(
                    DeepLabCutTrainingImportConfig(
                        source_dir=root,
                        labeled_data_root=Path(str(labeled_data or "labeled-data")),
                        instance_label=str(instance_label or "mouse"),
                        overwrite=bool(overwrite),
                        recursive=True,
                    ),
                    write_pose_schema=bool(write_pose_schema),
                    pose_schema_out=schema_out_path,
                    pose_schema_preset=str(pose_schema_preset or "mouse"),
                    instance_separator=str(instance_separator or "_"),
                )
                resolved_index_file = (
                    Path(index_file)
                    if str(index_file or "").strip()
                    else Path(DEFAULT_LABEL_INDEX_DIRNAME) / DEFAULT_LABEL_INDEX_NAME
                )
                if not resolved_index_file.is_absolute():
                    resolved_index_file = root / resolved_index_file
                if write_index:
                    index_summary = index_labelme_dataset(
                        sources=[root / str(labeled_data or "labeled-data")],
                        index_file=resolved_index_file,
                        recursive=True,
                        include_empty=False,
                        dedupe=True,
                        source="annolid_bot",
                    )
                    result = {
                        **import_summary,
                        "index_file": str(resolved_index_file.resolve()),
                        "index_summary": index_summary,
                    }
                else:
                    result = import_summary
            elif mode_name == "coco_spec":
                payload = build_coco_spec_from_dataset_path(root)
                if payload is None:
                    raise ValueError(
                        "Could not infer a COCO dataset layout from dataset_folder."
                    )
                spec_out = _resolve_yaml_output_path(
                    output_dir,
                    default_path=root / "coco_spec.yaml",
                    allowed_dir=self._allowed_dir,
                )
                spec_out.parent.mkdir(parents=True, exist_ok=True)
                spec_out.write_text(
                    yaml.safe_dump(payload, sort_keys=False),
                    encoding="utf-8",
                )
                result = {
                    "spec_path": str(spec_out.resolve()),
                    "task": infer_coco_task(
                        config_path=spec_out,
                        payload=payload,
                    ),
                }
            elif mode_name == "coco_to_yolo":
                payload = build_coco_spec_from_dataset_path(root)
                if payload is None:
                    raise ValueError(
                        "Could not infer a COCO dataset layout from dataset_folder."
                    )
                spec_path = root / "coco_spec.yaml"
                spec_path.write_text(
                    yaml.safe_dump(payload, sort_keys=False),
                    encoding="utf-8",
                )
                destination = (
                    _resolve_write_path(output_dir, allowed_dir=self._allowed_dir)
                    if str(output_dir or "").strip()
                    else root / "YOLO_dataset"
                )
                data_yaml = materialize_coco_spec_as_yolo(
                    config_path=spec_path,
                    output_dir=destination,
                )
                result = {
                    "spec_path": str(spec_path.resolve()),
                    "data_yaml": str(Path(data_yaml).resolve()),
                    "output_dir": str(Path(destination).resolve()),
                }
            else:
                raise ValueError(f"Unsupported mode: {mode!r}")
            return json.dumps(
                {
                    "ok": True,
                    "dataset_folder": str(root),
                    "mode": mode_name,
                    "result": result,
                },
                ensure_ascii=False,
            )
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "dataset_folder": dataset_folder,
                    "mode": mode,
                },
                ensure_ascii=False,
            )


__all__ = [
    "AnnolidDatasetInspectTool",
    "AnnolidDatasetPrepareTool",
    "inspect_dataset_folder",
]
