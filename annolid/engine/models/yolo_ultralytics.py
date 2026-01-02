from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

from annolid.engine.registry import ModelPluginBase, register_model


@register_model
class YOLOUltralyticsPlugin(ModelPluginBase):
    name = "yolo"
    description = "Ultralytics YOLO train/predict wrapper (delegates to ultralytics.YOLO)."

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--weights", default="yolo11n-seg.pt",
                            help="Initial YOLO weights (asset name or path)")
        parser.add_argument("--data", required=True,
                            help="Path to YOLO data.yaml")
        parser.add_argument("--epochs", type=int, default=100)
        parser.add_argument("--imgsz", type=int, default=640)
        parser.add_argument("--batch", type=int, default=8)
        parser.add_argument("--device", default=None,
                            help="'' (auto), 'cpu', 'mps', or '0' for CUDA GPU 0")
        parser.add_argument("--project", default=None,
                            help="Output directory (Ultralytics project)")
        parser.add_argument("--plots", action="store_true",
                            help="Save training plots")
        parser.add_argument(
            "--override",
            action="append",
            default=[],
            help="Extra ultralytics train kwargs as key=value (repeatable)",
        )

    def train(self, args: argparse.Namespace) -> int:
        from annolid.yolo import configure_ultralytics_cache, resolve_weight_path

        configure_ultralytics_cache()
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YOLO training requires the optional dependency 'ultralytics'.") from exc

        os.environ.setdefault("MPLBACKEND", "Agg")
        try:
            import matplotlib  # type: ignore

            matplotlib.use("Agg", force=True)
        except Exception:
            pass

        weight_path = resolve_weight_path(str(args.weights))
        model = YOLO(str(weight_path))

        overrides: Dict[str, Any] = {}
        for item in list(args.override or []):
            if "=" not in str(item):
                raise ValueError(
                    f"Invalid --override {item!r}; expected key=value")
            k, v = str(item).split("=", 1)
            overrides[k.strip()] = v.strip()

        kwargs: Dict[str, Any] = {
            "data": str(Path(args.data).expanduser().resolve()),
            "epochs": int(args.epochs),
            "imgsz": int(args.imgsz),
            "batch": int(args.batch),
            "device": (str(args.device).strip() if args.device is not None else None),
            "project": (str(Path(args.project).expanduser().resolve()) if args.project else None),
            "plots": bool(args.plots),
        }
        kwargs.update(overrides)

        res = model.train(**kwargs)
        # Ultralytics returns a Results object; print a stable summary.
        try:
            save_dir = getattr(res, "save_dir", None)
            if save_dir:
                print(str(save_dir))
        except Exception:
            pass
        return 0

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--weights", default="yolo11n-seg.pt",
                            help="YOLO weights (asset name or path)")
        parser.add_argument("--source", required=True,
                            help="Image/video path or glob")
        parser.add_argument("--device", default=None,
                            help="'' (auto), 'cpu', 'mps', or '0' for CUDA GPU 0")
        parser.add_argument("--conf", type=float,
                            default=0.25, help="Confidence threshold")
        parser.add_argument("--iou", type=float, default=0.7,
                            help="NMS IoU threshold")
        parser.add_argument("--project", default=None, help="Output directory")
        parser.add_argument("--name", default="predict",
                            help="Run name under project dir")
        parser.add_argument("--save", action="store_true",
                            help="Save visualizations")
        parser.add_argument("--save-txt", action="store_true",
                            help="Save predictions as txt")

    def predict(self, args: argparse.Namespace) -> int:
        from annolid.yolo import configure_ultralytics_cache, resolve_weight_path

        configure_ultralytics_cache()
        try:
            from ultralytics import YOLO  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "YOLO inference requires the optional dependency 'ultralytics'.") from exc

        weight_path = resolve_weight_path(str(args.weights))
        model = YOLO(str(weight_path))
        model.predict(
            source=str(args.source),
            device=(str(args.device).strip()
                    if args.device is not None else None),
            conf=float(args.conf),
            iou=float(args.iou),
            project=(str(Path(args.project).expanduser().resolve())
                     if args.project else None),
            name=str(args.name),
            save=bool(args.save),
            save_txt=bool(args.save_txt),
        )
        return 0
