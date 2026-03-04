from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from annolid.engine.registry import ModelPluginBase, register_model
from annolid.engine.run_config import get_cfg_value, load_run_config


@dataclass(frozen=True)
class Detectron2TrainSettings:
    dataset_dir: str
    output_dir: Optional[str]
    max_iterations: int
    batch_size: int
    weights: Optional[str]
    model_config: str
    score_threshold: float
    overlap_threshold: float
    base_lr: float
    num_workers: int
    checkpoint_period: int
    roi_batch_size_per_image: int
    sampler_train: str
    repeat_threshold: float


def _resolve_train_settings(args: argparse.Namespace) -> Detectron2TrainSettings:
    payload: Dict[str, Any] = {}
    if getattr(args, "run_config", None):
        payload = load_run_config(str(args.run_config))

    defaults = {
        "model_config": "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        "max_iterations": 3000,
        "batch_size": 8,
        "score_threshold": 0.15,
        "overlap_threshold": 0.7,
        "base_lr": 0.0025,
        "num_workers": 2,
        "checkpoint_period": 1000,
        "roi_batch_size_per_image": 128,
        "sampler_train": "RepeatFactorTrainingSampler",
        "repeat_threshold": 0.3,
    }

    dataset_dir_cfg = get_cfg_value(
        payload, "dataset_dir", "train.dataset_dir", "dataset.path"
    )
    output_dir_cfg = get_cfg_value(
        payload, "output_dir", "train.output_dir", "output.path"
    )
    weights_cfg = get_cfg_value(payload, "weights", "train.weights", "model.weights")

    dataset_dir = args.dataset_dir if args.dataset_dir is not None else dataset_dir_cfg
    if not dataset_dir:
        raise ValueError(
            "dataset_dir is required. Provide --dataset-dir or set it in run config."
        )

    def pick(name: str, *cfg_keys: str) -> Any:
        arg_value = getattr(args, name, None)
        if arg_value is not None:
            return arg_value
        cfg_value = get_cfg_value(payload, *cfg_keys)
        if cfg_value is not None:
            return cfg_value
        return defaults[name]

    output_dir = args.output_dir if args.output_dir is not None else output_dir_cfg
    model_config = str(
        pick("model_config", "model_config", "train.model_config", "model.config")
    )
    settings = Detectron2TrainSettings(
        dataset_dir=str(Path(dataset_dir).expanduser().resolve()),
        output_dir=(
            str(Path(output_dir).expanduser().resolve()) if output_dir else None
        ),
        max_iterations=int(
            pick(
                "max_iterations",
                "max_iterations",
                "train.max_iterations",
                "solver.max_iterations",
            )
        ),
        batch_size=int(
            pick("batch_size", "batch_size", "train.batch_size", "solver.batch_size")
        ),
        weights=(
            str(Path(args.weights).expanduser().resolve())
            if args.weights
            else (
                str(Path(weights_cfg).expanduser().resolve()) if weights_cfg else None
            )
        ),
        model_config=model_config,
        score_threshold=float(
            pick(
                "score_threshold",
                "score_threshold",
                "train.score_threshold",
                "detectron2.score_threshold",
            )
        ),
        overlap_threshold=float(
            pick(
                "overlap_threshold",
                "overlap_threshold",
                "train.overlap_threshold",
                "detectron2.overlap_threshold",
            )
        ),
        base_lr=float(pick("base_lr", "base_lr", "solver.base_lr")),
        num_workers=int(pick("num_workers", "num_workers", "dataloader.num_workers")),
        checkpoint_period=int(
            pick("checkpoint_period", "checkpoint_period", "solver.checkpoint_period")
        ),
        roi_batch_size_per_image=int(
            pick(
                "roi_batch_size_per_image",
                "roi_batch_size_per_image",
                "model.roi_batch_size_per_image",
                "detectron2.roi_batch_size_per_image",
            )
        ),
        sampler_train=str(
            pick("sampler_train", "sampler_train", "dataloader.sampler_train")
        ),
        repeat_threshold=float(
            pick("repeat_threshold", "repeat_threshold", "dataloader.repeat_threshold")
        ),
    )
    return settings


@register_model
class MaskRCNNDetectron2Plugin(ModelPluginBase):
    name = "maskrcnn_detectron2"
    description = "Detectron2 Mask R-CNN train/predict wrapper."

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-dir",
            default=None,
            help="Dataset directory containing train/valid COCO folders",
        )
        parser.add_argument(
            "--output-dir", default=None, help="Training outputs directory"
        )
        parser.add_argument(
            "--run-config",
            default=None,
            help="Path to run config YAML. CLI flags override YAML values.",
        )
        parser.add_argument("--max-iterations", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument(
            "--weights", default=None, help="Optional initial weights checkpoint path"
        )
        parser.add_argument(
            "--model-config",
            default=None,
            help="Detectron2 model zoo config",
        )
        parser.add_argument("--score-threshold", type=float, default=None)
        parser.add_argument("--overlap-threshold", type=float, default=None)
        parser.add_argument("--base-lr", type=float, default=None)
        parser.add_argument("--num-workers", type=int, default=None)
        parser.add_argument("--checkpoint-period", type=int, default=None)
        parser.add_argument("--roi-batch-size-per-image", type=int, default=None)
        parser.add_argument("--sampler-train", default=None)
        parser.add_argument("--repeat-threshold", type=float, default=None)

    def train(self, args: argparse.Namespace) -> int:
        try:
            from annolid.segmentation.maskrcnn.detectron2_train import Segmentor
        except Exception as exc:
            raise RuntimeError(
                "maskrcnn_detectron2 training requires the optional dependency 'detectron2'."
            ) from exc

        settings = _resolve_train_settings(args)
        seg = Segmentor(
            dataset_dir=settings.dataset_dir,
            out_put_dir=settings.output_dir,
            score_threshold=settings.score_threshold,
            overlap_threshold=settings.overlap_threshold,
            max_iterations=settings.max_iterations,
            batch_size=settings.batch_size,
            model_pth_path=settings.weights,
            model_config=settings.model_config,
            base_lr=settings.base_lr,
            num_workers=settings.num_workers,
            checkpoint_period=settings.checkpoint_period,
            roi_batch_size_per_image=settings.roi_batch_size_per_image,
            sampler_train=settings.sampler_train,
            repeat_threshold=settings.repeat_threshold,
        )
        seg.train()
        print(str(seg.out_put_dir))
        return 0

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-dir",
            required=True,
            help="Dataset directory containing train/valid COCO folders",
        )
        parser.add_argument(
            "--weights", required=True, help="Trained detectron2 model weights (.pth)"
        )
        parser.add_argument("--image", required=True, help="Image path")
        parser.add_argument("--score-threshold", type=float, default=0.15)
        parser.add_argument("--overlap-threshold", type=float, default=0.95)
        parser.add_argument(
            "--model-config", default=None, help="Override detectron2 model zoo config"
        )
        parser.add_argument(
            "--no-display", action="store_true", help="Do not open an OpenCV window"
        )

    def predict(self, args: argparse.Namespace) -> int:
        try:
            from annolid.inference.predict import Segmentor
        except Exception as exc:
            raise RuntimeError(
                "maskrcnn_detectron2 inference requires the optional dependency 'detectron2'."
            ) from exc

        seg = Segmentor(
            dataset_dir=str(Path(args.dataset_dir).expanduser().resolve()),
            model_pth_path=str(Path(args.weights).expanduser().resolve()),
            score_threshold=float(args.score_threshold),
            overlap_threshold=float(args.overlap_threshold),
            model_config=(str(args.model_config) if args.model_config else None),
        )
        seg.on_image(
            str(Path(args.image).expanduser().resolve()),
            display=not bool(args.no_display),
        )
        return 0
