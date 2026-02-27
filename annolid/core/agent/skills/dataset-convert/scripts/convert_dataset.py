#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from annolid.annotation.coco2labelme import (
    convert_coco_annotations_dir_to_labelme_dataset,
    convert_coco_json_to_labelme_dataset,
)
from annolid.annotation.labelme2coco import convert as labelme_to_coco_convert
from annolid.segmentation.dino_kpseg.data import (
    load_coco_pose_spec,
    materialize_coco_pose_as_yolo,
)


def _resolve_labelme_input_dir(input_dir: Path) -> Path:
    # labelme2coco currently scans only top-level *.json files.
    # Prefer the user path; fall back to common nested layout: <root>/images/*.json.
    if any(input_dir.glob("*.json")):
        return input_dir
    images_dir = input_dir / "images"
    if images_dir.exists() and any(images_dir.glob("*.json")):
        return images_dir
    return input_dir


def _run_labelme_to_coco(args: argparse.Namespace) -> int:
    input_dir = _resolve_labelme_input_dir(Path(args.input_dir).expanduser().resolve())
    output_paths: List[str] = []
    for progress, path in labelme_to_coco_convert(
        input_annotated_dir=str(input_dir),
        output_annotated_dir=str(Path(args.output_dir).expanduser().resolve()),
        labels_file=args.labels_file,
        train_valid_split=float(args.train_valid_split),
        output_mode=str(args.mode),
    ):
        if path:
            output_paths.append(str(path))

    unique_paths = sorted(set(output_paths))
    output_dir = Path(args.output_dir).expanduser().resolve()
    generated_annotations = [
        str(p)
        for p in (
            output_dir / "train" / "annotations.json",
            output_dir / "valid" / "annotations.json",
            output_dir / "annotations_train.json",
            output_dir / "annotations_valid.json",
        )
        if p.exists()
    ]
    payload: Dict[str, object] = {
        "ok": True,
        "mode": str(args.mode),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "generated_annotations": generated_annotations,
        "outputs": unique_paths,
        "outputs_count": len(unique_paths),
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run_coco_to_labelme(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).expanduser().resolve()
    images_dir = (
        Path(args.images_dir).expanduser().resolve() if args.images_dir else None
    )
    link_mode = str(args.link_mode)

    if args.coco_json:
        summary = convert_coco_json_to_labelme_dataset(
            Path(args.coco_json).expanduser().resolve(),
            output_dir=output_dir,
            images_dir=images_dir,
            link_mode=link_mode,
        )
    else:
        summary = convert_coco_annotations_dir_to_labelme_dataset(
            Path(args.annotations_dir).expanduser().resolve(),
            output_dir=output_dir,
            images_dir=images_dir,
            recursive=bool(args.recursive),
            link_mode=link_mode,
        )

    payload: Dict[str, object] = {
        "ok": True,
        "output_dir": str(output_dir),
        "summary": summary,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _run_coco_spec_to_yolo(args: argparse.Namespace) -> int:
    spec_yaml = Path(args.spec_yaml).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    spec = load_coco_pose_spec(spec_yaml)
    yolo_yaml = materialize_coco_pose_as_yolo(
        spec=spec,
        output_dir=output_dir,
        link_mode=str(args.link_mode),
    )
    payload: Dict[str, object] = {
        "ok": True,
        "spec_yaml": str(spec_yaml),
        "output_dir": str(output_dir),
        "data_yaml": str(yolo_yaml),
    }
    print(json.dumps(payload, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Annolid dataset format conversion helper."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_labelme_to_coco = sub.add_parser(
        "labelme-to-coco",
        help="Convert a LabelMe dataset directory into COCO train/valid JSON.",
    )
    p_labelme_to_coco.add_argument("--input-dir", required=True)
    p_labelme_to_coco.add_argument("--output-dir", required=True)
    p_labelme_to_coco.add_argument(
        "--mode",
        choices=("segmentation", "keypoints"),
        default="segmentation",
        help="COCO export mode (default: segmentation).",
    )
    p_labelme_to_coco.add_argument(
        "--labels-file",
        default="labels.txt",
        help="Optional labels file for segmentation mode.",
    )
    p_labelme_to_coco.add_argument(
        "--train-valid-split",
        type=float,
        default=0.7,
        help="Train split ratio/count accepted by labelme2coco.",
    )
    p_labelme_to_coco.set_defaults(_handler=_run_labelme_to_coco)

    p_coco_to_labelme = sub.add_parser(
        "coco-to-labelme",
        help="Convert COCO JSON(s) to a LabelMe dataset.",
    )
    source_group = p_coco_to_labelme.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--coco-json", default=None)
    source_group.add_argument("--annotations-dir", default=None)
    p_coco_to_labelme.add_argument("--output-dir", required=True)
    p_coco_to_labelme.add_argument(
        "--images-dir",
        default=None,
        help="Optional explicit image root.",
    )
    p_coco_to_labelme.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively find *.json under --annotations-dir.",
    )
    p_coco_to_labelme.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
    )
    p_coco_to_labelme.set_defaults(_handler=_run_coco_to_labelme)

    p_coco_spec_to_yolo = sub.add_parser(
        "coco-spec-to-yolo",
        help="Materialize a COCO pose spec YAML into a YOLO pose dataset.",
    )
    p_coco_spec_to_yolo.add_argument("--spec-yaml", required=True)
    p_coco_spec_to_yolo.add_argument("--output-dir", required=True)
    p_coco_spec_to_yolo.add_argument(
        "--link-mode",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
    )
    p_coco_spec_to_yolo.set_defaults(_handler=_run_coco_spec_to_yolo)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    handler = getattr(args, "_handler", None)
    if handler is None:
        parser.print_help()
        return 2
    return int(handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
