from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from annolid.engine.registry import ModelPluginBase, register_model


def _parse_class_indices(value: Optional[str]) -> Optional[List[int]]:
    if value is None or not str(value).strip():
        return None
    indices: List[int] = []
    for token in str(value).replace(" ", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            indices.append(int(token))
        except ValueError as exc:
            raise ValueError(
                f"Invalid --classes value {value!r}; expected comma-separated integer class indices."
            ) from exc
    return indices or None


def _locate_trained_checkpoint(
    project_arg: Optional[str], name_arg: Optional[str]
) -> Optional[Path]:
    if not project_arg:
        return None
    root = Path(project_arg)
    run_dir = root / name_arg if name_arg else root
    for filename in ("best.pt", "last.pt"):
        candidate = run_dir / "weights" / filename
        if candidate.is_file():
            return candidate
    return None


@register_model
class YOLOUltralyticsPlugin(ModelPluginBase):
    name = "yolo"
    description = (
        "Ultralytics YOLO fine-tuning and inference wrapper "
        "(accepts YOLO data.yaml, dataset directories, or COCO datasets)."
    )
    required_capabilities = ("yolo",)
    train_help_sections = (
        (
            "Required inputs",
            (
                "--data",
                "--weights",
                "--run-config",
            ),
        ),
        (
            "Outputs and run location",
            (
                "--project",
                "--name",
            ),
        ),
        (
            "Model and runtime",
            (
                "--device",
                "--imgsz",
                "--workers",
                "--cache",
            ),
        ),
        (
            "Training controls",
            (
                "--preset",
                "--epochs",
                "--batch",
                "--patience",
                "--lr0",
                "--freeze",
                "--optimizer",
                "--cos-lr",
                "--resume",
                "--override",
            ),
        ),
        (
            "Saving and reporting",
            ("--plots",),
        ),
    )
    train_examples = (
        "annolid-run train yolo --data /path/to/dataset --weights yolo11s-seg.pt --preset small_dataset",
        "annolid-run train yolo --data /path/to/data.yaml --weights yolo26s-pose.pt --epochs 150 --imgsz 640 --batch 16",
        "annolid-run train yolo --run-config annolid/configs/runs/yolo_train.yaml --data /path/to/data.yaml",
        "annolid-run train yolo --data /path/to/data.yaml --weights /path/to/last.pt --resume",
    )
    predict_help_sections = (
        (
            "Required inputs",
            (
                "--source",
                "--weights",
            ),
        ),
        (
            "Outputs and run location",
            (
                "--project",
                "--name",
            ),
        ),
        (
            "Model and runtime",
            (
                "--device",
                "--imgsz",
                "--half",
            ),
        ),
        (
            "Inference controls",
            (
                "--conf",
                "--iou",
                "--max-det",
                "--vid-stride",
                "--classes",
                "--track",
                "--tracker",
            ),
        ),
        (
            "Saving and reporting",
            (
                "--save",
                "--save-txt",
            ),
        ),
    )
    predict_examples = (
        "annolid-run predict yolo --weights /path/to/best.pt --source /path/to/video.mp4 --save",
        "annolid-run predict yolo --weights yolo11n-seg.pt --source /path/to/video.mp4 --track --conf 0.3 --save",
    )

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--run-config",
            default=None,
            help="Path to run config YAML. CLI flags override YAML values.",
        )
        parser.add_argument(
            "--weights",
            default="yolo11n-seg.pt",
            help="Initial YOLO weights (asset name or local .pt checkpoint)",
        )
        parser.add_argument(
            "--data",
            required=True,
            help="Path to YOLO data.yaml, dataset directory, or COCO dataset",
        )
        parser.add_argument(
            "--preset",
            choices=("balanced", "small_dataset", "fast", "high_accuracy"),
            default=None,
            help="Curated fine-tuning preset (explicit CLI flags override preset defaults)",
        )
        parser.add_argument(
            "--epochs",
            type=int,
            default=None,
            help="Number of training epochs (default: preset value or 100)",
        )
        parser.add_argument(
            "--imgsz",
            type=int,
            default=None,
            help="Training image size in pixels (default: preset value or 640)",
        )
        parser.add_argument(
            "--batch",
            type=int,
            default=None,
            help="Batch size (default: preset value or 8)",
        )
        parser.add_argument(
            "--patience",
            type=int,
            default=None,
            help="Early stopping patience in epochs (0 disables early stopping)",
        )
        parser.add_argument(
            "--lr0",
            type=float,
            default=None,
            help="Initial learning rate",
        )
        parser.add_argument(
            "--freeze",
            type=int,
            default=None,
            help="Number of leading backbone layers to freeze during fine-tuning",
        )
        parser.add_argument(
            "--optimizer",
            default=None,
            help="Optimizer choice (e.g. auto, SGD, Adam, AdamW)",
        )
        parser.add_argument(
            "--cos-lr",
            action="store_true",
            help="Enable cosine learning-rate decay schedule",
        )
        parser.add_argument(
            "--cache",
            default=None,
            help="Dataset image caching mode: false, ram, or disk",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=None,
            help="Number of dataloader worker processes",
        )
        parser.add_argument(
            "--resume",
            action="store_true",
            help="Resume an interrupted training run from --weights (e.g. last.pt)",
        )
        parser.add_argument(
            "--device",
            default=None,
            help="'' (auto), 'cpu', 'mps', or '0' for CUDA GPU 0",
        )
        parser.add_argument(
            "--project", default=None, help="Output directory (Ultralytics project)"
        )
        parser.add_argument(
            "--name",
            default=None,
            help="Run subdirectory name under --project",
        )
        parser.add_argument("--plots", action="store_true", help="Save training plots")
        parser.add_argument(
            "--override",
            action="append",
            default=[],
            help="Extra ultralytics train kwargs as key=value (repeatable)",
        )

    def train(self, args: argparse.Namespace) -> int:
        from annolid.utils.runs import allocate_run_dir, shared_runs_root
        from annolid.yolo import (
            get_yolo_train_preset,
            infer_yolo_task,
            resolve_weight_path,
            resolve_yolo_device,
        )
        from annolid.yolo.dataset_prep import (
            check_training_preflight,
            count_dataset_images,
            prepare_yolo_dataset,
        )
        from annolid.yolo.ultralytics_cli import (
            build_yolo_train_command,
            ensure_parent_dir,
            format_yolo_cli_error,
            run_yolo_cli,
        )

        preset = get_yolo_train_preset(args.preset) if args.preset else None
        epochs = int(
            args.epochs
            if args.epochs is not None
            else (preset.epochs if preset is not None else 100)
        )
        imgsz = int(
            args.imgsz
            if args.imgsz is not None
            else (preset.imgsz if preset is not None else 640)
        )
        batch = int(
            args.batch
            if args.batch is not None
            else (preset.batch if preset is not None else 8)
        )
        patience = (
            int(args.patience)
            if args.patience is not None
            else (int(preset.patience) if preset is not None else None)
        )
        freeze = (
            int(args.freeze)
            if args.freeze is not None
            else (
                int(preset.freeze)
                if preset is not None and int(preset.freeze) > 0
                else None
            )
        )

        overrides: Dict[str, Any] = {}
        if preset is not None:
            overrides["lr0"] = float(preset.lr0)
            overrides["lrf"] = float(preset.lrf)
            overrides["weight_decay"] = float(preset.weight_decay)
            overrides["close_mosaic"] = int(preset.close_mosaic)
            overrides["optimizer"] = str(preset.optimizer)
            if preset.cos_lr:
                overrides["cos_lr"] = True

        if args.lr0 is not None:
            overrides["lr0"] = float(args.lr0)
        if args.optimizer is not None and str(args.optimizer).strip():
            overrides["optimizer"] = str(args.optimizer).strip()
        if bool(args.cos_lr):
            overrides["cos_lr"] = True
        if args.workers is not None:
            overrides["workers"] = int(args.workers)

        for item in list(args.override or []):
            if "=" not in str(item):
                raise ValueError(f"Invalid --override {item!r}; expected key=value")
            k, v = str(item).split("=", 1)
            overrides[k.strip()] = v.strip()

        weight_path = resolve_weight_path(str(args.weights))
        model_arg = ensure_parent_dir(str(weight_path))
        expected_task = infer_yolo_task(str(args.weights))

        project_arg: Optional[str] = None
        name_arg: Optional[str] = (
            str(args.name).strip()
            if args.name is not None and str(args.name).strip()
            else None
        )
        if args.project:
            project_arg = str(Path(args.project).expanduser().resolve())
        else:
            try:
                if name_arg:
                    project_arg = str((shared_runs_root() / "yolo").resolve())
                else:
                    run_dir = allocate_run_dir(
                        task="yolo",
                        model=Path(str(args.weights)).stem or "yolo",
                    )
                    project_arg = str(run_dir.parent)
                    name_arg = run_dir.name
            except OSError:
                project_arg = None

        resolved_device = resolve_yolo_device(args.device)

        with prepare_yolo_dataset(args.data, expected_task=expected_task) as prepared:
            data_cfg = prepared.data_cfg
            train_n = count_dataset_images(data_cfg.get("train"))
            val_n = count_dataset_images(data_cfg.get("val"))
            nc = int(data_cfg.get("nc") or len(data_cfg.get("names") or []))
            print(
                f"[YOLO] Dataset ready: train={train_n} val={val_n} "
                f"classes={nc} task={expected_task or 'detect'} device={resolved_device}"
            )
            if preset is not None:
                print(
                    f"[YOLO] Using preset '{preset.key}' ({preset.display_name}): "
                    f"epochs={epochs}, batch={batch}, imgsz={imgsz}, "
                    f"patience={patience}, freeze={freeze or 0}"
                )
            for warning in check_training_preflight(
                prepared.data_yaml,
                batch_size=batch,
                device=resolved_device,
                plots=bool(args.plots),
                expected_task=expected_task,
            ):
                print(f"[YOLO][WARN] {warning}")

            cmd = build_yolo_train_command(
                model=str(model_arg),
                data=str(prepared.data_yaml),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=resolved_device,
                project=project_arg,
                name=name_arg,
                plots=bool(args.plots) if bool(args.plots) else None,
                resume=bool(args.resume),
                freeze=freeze,
                patience=patience,
                cache=(
                    str(args.cache).strip()
                    if args.cache is not None and str(args.cache).strip()
                    else None
                ),
                overrides=overrides,
            )

            def sink(line: str) -> None:
                try:
                    sys.stdout.write(line)
                    sys.stdout.flush()
                except Exception:
                    pass

            completed = run_yolo_cli(cmd, output_sink=sink)
            if completed.returncode != 0:
                raise RuntimeError(
                    format_yolo_cli_error(
                        completed,
                        model=str(args.weights),
                        device=resolved_device,
                    )
                )

        checkpoint = _locate_trained_checkpoint(project_arg, name_arg)
        if checkpoint is not None:
            print(f"\n[YOLO] Training finished. Best weights: {checkpoint}")
            print(
                "[YOLO] Run inference with:\n"
                f"  annolid-run predict yolo_labelme --weights {checkpoint} --source /path/to/video.mp4"
            )
        return 0

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--weights",
            default="yolo11n-seg.pt",
            help="YOLO weights (asset name or path)",
        )
        parser.add_argument("--source", required=True, help="Image/video path or glob")
        parser.add_argument(
            "--device",
            default=None,
            help="'' (auto), 'cpu', 'mps', or '0' for CUDA GPU 0",
        )
        parser.add_argument(
            "--imgsz",
            type=int,
            default=None,
            help="Inference image size in pixels (e.g. 640)",
        )
        parser.add_argument(
            "--half",
            action="store_true",
            help="Use FP16 half precision on supported GPUs",
        )
        parser.add_argument(
            "--conf", type=float, default=0.25, help="Confidence threshold"
        )
        parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
        parser.add_argument(
            "--max-det",
            type=int,
            default=None,
            help="Maximum detections per frame",
        )
        parser.add_argument(
            "--vid-stride",
            type=int,
            default=1,
            help="Video frame-rate stride (1 processes every frame)",
        )
        parser.add_argument(
            "--classes",
            default=None,
            help="Optional comma-separated class indices to keep (e.g. '0,1')",
        )
        parser.add_argument(
            "--track",
            action="store_true",
            help="Run multi-object tracking (model.track) instead of standalone predict()",
        )
        parser.add_argument(
            "--tracker",
            default=None,
            help="Optional Ultralytics tracker config (e.g. 'botsort.yaml', 'bytetrack.yaml')",
        )
        parser.add_argument("--project", default=None, help="Output directory")
        parser.add_argument(
            "--name", default="predict", help="Run name under project dir"
        )
        parser.add_argument("--save", action="store_true", help="Save visualizations")
        parser.add_argument(
            "--save-txt", action="store_true", help="Save predictions as txt"
        )

    def predict(self, args: argparse.Namespace) -> int:
        from annolid.yolo import (
            configure_ultralytics_cache,
            import_ultralytics_symbol,
            resolve_weight_path,
            resolve_yolo_device,
        )

        configure_ultralytics_cache()
        YOLO = import_ultralytics_symbol("YOLO", purpose="YOLO inference")

        weight_path = resolve_weight_path(str(args.weights))
        model = YOLO(str(weight_path))

        resolved_device = resolve_yolo_device(args.device)
        class_indices = _parse_class_indices(getattr(args, "classes", None))

        infer_kwargs: Dict[str, Any] = {
            "source": str(args.source),
            "device": resolved_device,
            "conf": float(args.conf),
            "iou": float(args.iou),
            "project": (
                str(Path(args.project).expanduser().resolve()) if args.project else None
            ),
            "name": str(args.name),
            "save": bool(args.save),
            "save_txt": bool(args.save_txt),
            "stream": True,
        }
        if args.imgsz is not None:
            infer_kwargs["imgsz"] = int(args.imgsz)
        if bool(getattr(args, "half", False)):
            infer_kwargs["half"] = True
        if getattr(args, "max_det", None) is not None:
            infer_kwargs["max_det"] = int(args.max_det)
        if getattr(args, "vid_stride", None) is not None and int(args.vid_stride) > 1:
            infer_kwargs["vid_stride"] = int(args.vid_stride)
        if class_indices is not None:
            infer_kwargs["classes"] = class_indices

        if bool(getattr(args, "track", False)):
            infer_kwargs["persist"] = True
            if getattr(args, "tracker", None):
                infer_kwargs["tracker"] = str(args.tracker).strip()
            results_iter = model.track(**infer_kwargs)
        else:
            results_iter = model.predict(**infer_kwargs)

        frames_processed = 0
        total_detections = 0
        save_dir: Optional[str] = None
        for result in results_iter:
            frames_processed += 1
            boxes = getattr(result, "boxes", None)
            if boxes is not None:
                try:
                    total_detections += int(len(boxes))
                except Exception:
                    pass
            if save_dir is None:
                res_dir = getattr(result, "save_dir", None)
                if res_dir:
                    save_dir = str(res_dir)

        summary_parts = [
            f"frames={frames_processed}",
            f"detections={total_detections}",
            f"device={resolved_device}",
        ]
        if save_dir:
            summary_parts.append(f"output={save_dir}")
        print(f"[YOLO] Inference complete ({', '.join(summary_parts)})")
        return 0
