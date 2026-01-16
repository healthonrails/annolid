from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from annolid.engine.registry import ModelPluginBase, register_model


def _split_classes(value: Optional[str]) -> List[str]:
    if not value:
        return []
    out: List[str] = []
    for part in str(value).replace("\n", ",").split(","):
        name = part.strip()
        if name:
            out.append(name)
    return out


def _looks_like_prompt_free_yoloe(weight: str) -> bool:
    w = str(weight or "").lower()
    return "yoloe" in w and ("-pf." in w or w.endswith("-pf") or "-pf_" in w)


def _load_visual_prompts_json(path: Path) -> Tuple[Dict[str, Any], Optional[List[str]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(
            f"Visual prompts file must contain a JSON object: {path}")

    prompts = raw.get("visual_prompts", raw)
    if not isinstance(prompts, dict):
        raise ValueError(
            f"Visual prompts must be a JSON object with keys 'bboxes' and 'cls': {path}"
        )

    names = raw.get("names", None)
    if names is None and isinstance(prompts.get("names", None), list):
        names = prompts.get("names", None)
    if names is not None:
        if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
            raise ValueError(
                f"'names' must be a JSON list of strings if provided: {path}"
            )

    bboxes = prompts.get("bboxes", None)
    cls = prompts.get("cls", None)
    if bboxes is None or cls is None:
        raise ValueError(
            f"Visual prompts JSON must include 'bboxes' and 'cls': {path}"
        )
    if not isinstance(bboxes, list) or not isinstance(cls, list):
        raise ValueError(
            f"Visual prompts 'bboxes' and 'cls' must be lists (will be converted to numpy arrays internally): {path}"
        )
    if len(bboxes) != len(cls):
        raise ValueError(
            f"Visual prompts length mismatch: {len(bboxes)} bboxes vs {len(cls)} cls: {path}"
        )

    return {"bboxes": bboxes, "cls": cls}, list(names) if names else None


def _load_visual_prompts_from_labelme(path: Path) -> Tuple[Dict[str, Any], List[str]]:
    from annolid.utils.annotation_store import load_labelme_json

    data = load_labelme_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"LabelMe JSON must be an object: {path}")
    shapes = data.get("shapes", [])
    if not isinstance(shapes, list):
        raise ValueError(f"LabelMe JSON 'shapes' must be a list: {path}")

    rectangles: List[Tuple[str, List[float]]] = []
    for shape in shapes:
        if not isinstance(shape, dict):
            continue
        if str(shape.get("shape_type", "")).lower() != "rectangle":
            continue
        label = str(shape.get("label", "") or "").strip()
        points = shape.get("points", None)
        if not label or not isinstance(points, list) or len(points) < 2:
            continue
        p0, p1 = points[0], points[1]
        if (
            not isinstance(p0, (list, tuple))
            or not isinstance(p1, (list, tuple))
            or len(p0) < 2
            or len(p1) < 2
        ):
            continue
        x1, y1 = float(min(p0[0], p1[0])), float(min(p0[1], p1[1]))
        x2, y2 = float(max(p0[0], p1[0])), float(max(p0[1], p1[1]))
        rectangles.append((label, [x1, y1, x2, y2]))

    if not rectangles:
        raise ValueError(f"No labeled rectangle shapes found in: {path}")

    class_names = sorted({label for label, _ in rectangles})
    mapping = {name: idx for idx, name in enumerate(class_names)}
    bboxes: List[List[float]] = []
    cls: List[int] = []
    for label, bbox in rectangles:
        bboxes.append(bbox)
        cls.append(int(mapping[label]))

    return {"bboxes": bboxes, "cls": cls}, class_names


@register_model
class YOLOLabelMePlugin(ModelPluginBase):
    name = "yolo_labelme"
    description = "Run Ultralytics YOLO/YOLOE inference and export LabelMe JSON (supports YOLOE text + visual prompts)."

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--weights",
            default="yoloe-26s-seg.pt",
            help="YOLO/YOLOE weights (asset name or path). For prompt-free YOLOE use '*-pf.pt'.",
        )
        parser.add_argument(
            "--source",
            required=True,
            help="Image/video path. Writes LabelMe JSON into an output folder.",
        )
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Output directory. Default: <source-without-suffix>/",
        )
        parser.add_argument(
            "--classes",
            default="",
            help="Comma-separated class names for YOLOE text prompting (e.g. 'person,bus'). Ignored for prompt-free YOLOE.",
        )
        parser.add_argument(
            "--visual-prompts",
            default=None,
            help="JSON file containing visual prompts with keys 'bboxes' and 'cls' (optionally 'names').",
        )
        parser.add_argument(
            "--visual-prompts-labelme",
            default=None,
            help="LabelMe JSON file; uses labeled rectangle shapes as YOLOE visual prompts (labels become class names).",
        )
        parser.add_argument("--start-frame", type=int, default=0)
        parser.add_argument("--end-frame", type=int, default=None)
        parser.add_argument("--step", type=int, default=1)
        parser.add_argument(
            "--no-tracking",
            action="store_true",
            help="Disable tracking (uses predict() instead of track()).",
        )
        parser.add_argument(
            "--tracker",
            default=None,
            help="Optional Ultralytics tracker config (e.g. 'botsort.yaml').",
        )
        parser.add_argument(
            "--save-pose-bbox",
            action="store_true",
            help="For pose models, also save bounding boxes alongside keypoints.",
        )

    def predict(self, args: argparse.Namespace) -> int:
        from annolid.segmentation.yolos import InferenceProcessor

        weights = str(args.weights)
        source = str(args.source)
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else Path(source).expanduser().resolve().with_suffix("")
        )

        visual_prompts = None
        class_names: List[str] = []

        if args.visual_prompts and args.visual_prompts_labelme:
            raise ValueError(
                "Provide only one of --visual-prompts or --visual-prompts-labelme."
            )

        if args.visual_prompts:
            prompts_path = Path(args.visual_prompts).expanduser().resolve()
            visual_prompts, names_from_file = _load_visual_prompts_json(
                prompts_path)
            if names_from_file:
                class_names = list(names_from_file)

        if args.visual_prompts_labelme:
            labelme_path = Path(
                args.visual_prompts_labelme).expanduser().resolve()
            visual_prompts, class_names = _load_visual_prompts_from_labelme(
                labelme_path)

        if not class_names:
            class_names = _split_classes(str(args.classes))

        if _looks_like_prompt_free_yoloe(weights):
            class_names = []
            visual_prompts = None

        processor = InferenceProcessor(
            model_name=weights,
            model_type="yolo",
            class_names=class_names or None,
            persist_json=True,
        )

        message = processor.run_inference(
            source=source,
            visual_prompts=visual_prompts,
            start_frame=int(args.start_frame),
            end_frame=(int(args.end_frame)
                       if args.end_frame is not None else None),
            step=int(args.step),
            skip_existing=False,
            output_directory=output_dir,
            enable_tracking=not bool(args.no_tracking),
            tracker=(str(args.tracker).strip() if args.tracker else None),
            save_pose_bbox=bool(args.save_pose_bbox) if bool(
                args.save_pose_bbox) else None,
        )
        print(str(message))
        return 0
