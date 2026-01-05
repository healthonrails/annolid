from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

from annolid.engine.registry import get_load_failures, get_model, list_models, load_builtin_models


def _cmd_list_models(_: argparse.Namespace) -> int:
    failures = load_builtin_models()
    if failures:
        details = get_load_failures()
        for name in failures:
            print(
                f"[annolid-run] Failed to import {name}: {details.get(name,'')}", file=sys.stderr)
    rows = [
        {
            "name": m.name,
            "train": m.supports_train,
            "predict": m.supports_predict,
            "description": m.description,
        }
        for m in list_models(load_builtins=False)
    ]
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_collect_labels(args: argparse.Namespace) -> int:
    from annolid.datasets.labelme_collection import (
        DEFAULT_LABEL_INDEX_NAME,
        DEFAULT_LABEL_INDEX_DIRNAME,
        index_labelme_dataset,
    )

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    index_file_default = str(
        Path(DEFAULT_LABEL_INDEX_DIRNAME) / DEFAULT_LABEL_INDEX_NAME)
    index_file = Path(getattr(args, "index_file", index_file_default))
    if not index_file.is_absolute():
        index_file = dataset_root / index_file

    summary = index_labelme_dataset(
        sources=[Path(p) for p in args.source],
        index_file=index_file,
        recursive=bool(args.recursive),
        include_empty=bool(args.include_empty),
        dedupe=not bool(args.allow_duplicates),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_index_to_yolo(args: argparse.Namespace) -> int:
    from annolid.datasets.builders.label_index_yolo import build_yolo_from_label_index

    summary = build_yolo_from_label_index(
        index_file=Path(args.index_file),
        output_dir=Path(args.output_dir),
        dataset_name=str(args.dataset_name),
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        link_mode=str(args.link_mode),
        task=str(args.task),
        include_empty=bool(args.include_empty),
        keep_staging=bool(args.keep_staging),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_import_deeplabcut_training_data(args: argparse.Namespace) -> int:
    from annolid.datasets.importers.deeplabcut_training_data import (
        DeepLabCutTrainingImportConfig,
        import_deeplabcut_training_data,
    )
    from annolid.datasets.labelme_collection import index_labelme_dataset

    source_dir = Path(args.source_dir).expanduser().resolve()
    labeled_data = Path(args.labeled_data)
    labeled_data = labeled_data if labeled_data.is_absolute() else (source_dir / labeled_data)

    summary = import_deeplabcut_training_data(
        DeepLabCutTrainingImportConfig(
            source_dir=source_dir,
            labeled_data_root=Path(args.labeled_data),
            instance_label=str(args.instance_label),
            overwrite=bool(args.overwrite),
            recursive=not bool(args.no_recursive),
        ),
        write_pose_schema=bool(args.write_pose_schema),
        pose_schema_out=Path(args.pose_schema_out) if args.pose_schema_out else None,
        pose_schema_preset=str(args.pose_schema_preset),
        instance_separator=str(args.instance_separator or "_"),
    )

    if bool(args.write_index):
        index_file = Path(args.index_file)
        if not index_file.is_absolute():
            index_file = source_dir / index_file
        index_summary = index_labelme_dataset(
            sources=[labeled_data],
            index_file=index_file,
            recursive=True,
            include_empty=False,
            dedupe=True,
        )
        summary["index_file"] = str(index_file)
        summary["index_summary"] = index_summary

    print(json.dumps(summary, indent=2))
    return 0


def _build_root_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annolid-run",
        description="Unified training/inference CLI (plugin-based).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser(
        "list-models", help="List available model plugins.")
    list_p.set_defaults(_handler=_cmd_list_models)

    collect_p = sub.add_parser(
        "collect-labels",
        help="Index LabelMe PNG/JSON pairs by absolute path (no copying).",
    )
    collect_p.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source folder (repeatable) containing per-frame JSON/PNG pairs.",
    )
    collect_p.add_argument(
        "--dataset-root",
        required=True,
        help="Directory that will store the index file.",
    )
    collect_p.add_argument("--recursive", action="store_true", default=True)
    collect_p.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top-level of each source directory.",
    )
    collect_p.add_argument(
        "--include-empty",
        action="store_true",
        help="Collect JSON files even when they contain no shapes.",
    )
    collect_p.add_argument(
        "--index-file",
        default="annolid_logs/annolid_dataset.jsonl",
        help="Index file path relative to --dataset-root (default: annolid_logs/annolid_dataset.jsonl).",
    )
    collect_p.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Append even if the image path already exists in the index.",
    )
    collect_p.set_defaults(_handler=_cmd_collect_labels)

    yolo_p = sub.add_parser(
        "index-to-yolo",
        help="Convert an Annolid dataset index JSONL into a YOLO dataset.",
    )
    yolo_p.add_argument("--index-file", required=True,
                        help="Path to annolid_dataset.jsonl (JSONL index).")
    yolo_p.add_argument("--output-dir", required=True,
                        help="Directory to write the YOLO dataset.")
    yolo_p.add_argument("--dataset-name", default="YOLO_dataset",
                        help="Output dataset folder name.")
    yolo_p.add_argument("--val-size", type=float, default=0.1)
    yolo_p.add_argument("--test-size", type=float, default=0.1)
    yolo_p.add_argument("--link-mode", choices=("hardlink",
                        "copy", "symlink"), default="hardlink")
    yolo_p.add_argument(
        "--task",
        choices=("auto", "segmentation", "pose"),
        default="auto",
        help="How to handle mixed polygon+point annotations (default: auto).",
    )
    yolo_p.add_argument("--include-empty", action="store_true",
                        help="Include JSON records with no shapes.")
    yolo_p.add_argument("--keep-staging", action="store_true",
                        help="Keep temporary staging files.")
    yolo_p.set_defaults(_handler=_cmd_index_to_yolo)

    imp_p = sub.add_parser(
        "import-deeplabcut-training-data",
        help="Convert DeepLabCut labeled-data (CollectedData_*.csv) into LabelMe JSON next to images, then (optionally) index it.",
    )
    imp_p.add_argument("--source-dir", required=True,
                       help="Dataset root, e.g. /Users/.../mouse_training_data")
    imp_p.add_argument("--labeled-data", default="labeled-data",
                       help="labeled-data root relative to --source-dir (default: labeled-data)")
    imp_p.add_argument("--instance-label", default="mouse",
                       help="Instance label prefix to use for point shapes (default: mouse)")
    imp_p.add_argument("--overwrite", action="store_true",
                       help="Overwrite existing per-image LabelMe JSON files.")
    imp_p.add_argument("--no-recursive", action="store_true",
                       help="Do not search for CollectedData_*.csv recursively under --labeled-data.")
    imp_p.add_argument(
        "--write-pose-schema",
        action="store_true",
        help="Write pose_schema.json derived from DeepLabCut bodyparts.",
    )
    imp_p.add_argument(
        "--pose-schema-out",
        default="labeled-data/pose_schema.json",
        help="Path (relative to --source-dir) for the pose schema (default: labeled-data/pose_schema.json).",
    )
    imp_p.add_argument(
        "--pose-schema-preset",
        default="mouse",
        help="Edge preset to use when building the schema (default: mouse).",
    )
    imp_p.add_argument(
        "--instance-separator",
        default="_",
        help="Separator used for instance prefixes (default: _).",
    )
    imp_p.add_argument(
        "--index-file",
        default="annolid_logs/annolid_dataset.jsonl",
        help="Index file path relative to --source-dir (default: annolid_logs/annolid_dataset.jsonl).",
    )
    imp_p.add_argument("--no-index", dest="write_index",
                       action="store_false", help="Skip writing annolid_dataset.jsonl")
    imp_p.set_defaults(
        write_index=True, _handler=_cmd_import_deeplabcut_training_data)

    train_p = sub.add_parser("train", help="Train a model.")
    train_p.add_argument("model", help="Model plugin name (see list-models).")
    train_p.add_argument("--help-model", action="store_true",
                         help="Show model-specific help.")
    train_p.set_defaults(_handler="train")

    pred_p = sub.add_parser("predict", help="Run inference.")
    pred_p.add_argument("model", help="Model plugin name (see list-models).")
    pred_p.add_argument("--help-model", action="store_true",
                        help="Show model-specific help.")
    pred_p.set_defaults(_handler="predict")

    return p


def _dispatch_model_subcommand(
    *,
    base_args: argparse.Namespace,
    argv: list[str],
) -> int:
    model_name = str(base_args.model)
    plugin = get_model(model_name)

    mode = base_args._handler
    if mode == "train":
        if not plugin.__class__.supports_train():
            raise SystemExit(
                f"Model {model_name!r} does not support training.")
        p = argparse.ArgumentParser(prog=f"annolid-run train {model_name}")
        plugin.add_train_args(p)
        if base_args.help_model:
            p.print_help()
            return 0
        args = p.parse_args(argv)
        return int(plugin.train(args))

    if mode == "predict":
        if not plugin.__class__.supports_predict():
            raise SystemExit(
                f"Model {model_name!r} does not support inference.")
        p = argparse.ArgumentParser(prog=f"annolid-run predict {model_name}")
        plugin.add_predict_args(p)
        if base_args.help_model:
            p.print_help()
            return 0
        args = p.parse_args(argv)
        return int(plugin.predict(args))

    raise SystemExit(f"Unknown mode: {mode}")


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    p = _build_root_parser()
    args, rest = p.parse_known_args(argv)

    handler = getattr(args, "_handler", None)
    if callable(handler):
        return int(handler(args))

    return _dispatch_model_subcommand(base_args=args, argv=rest)


if __name__ == "__main__":
    raise SystemExit(main())
