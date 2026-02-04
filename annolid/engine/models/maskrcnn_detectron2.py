from __future__ import annotations

import argparse
from pathlib import Path

from annolid.engine.registry import ModelPluginBase, register_model


@register_model
class MaskRCNNDetectron2Plugin(ModelPluginBase):
    name = "maskrcnn_detectron2"
    description = "Detectron2 Mask R-CNN train/predict wrapper."

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-dir",
            required=True,
            help="Dataset directory containing train/valid COCO folders",
        )
        parser.add_argument(
            "--output-dir", default=None, help="Training outputs directory"
        )
        parser.add_argument("--max-iterations", type=int, default=3000)
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument(
            "--weights", default=None, help="Optional initial weights checkpoint path"
        )
        parser.add_argument(
            "--model-config",
            default="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
            help="Detectron2 model zoo config",
        )
        parser.add_argument("--score-threshold", type=float, default=0.15)
        parser.add_argument("--overlap-threshold", type=float, default=0.7)

    def train(self, args: argparse.Namespace) -> int:
        try:
            from annolid.segmentation.maskrcnn.detectron2_train import Segmentor
        except Exception as exc:
            raise RuntimeError(
                "maskrcnn_detectron2 training requires the optional dependency 'detectron2'."
            ) from exc

        seg = Segmentor(
            dataset_dir=str(Path(args.dataset_dir).expanduser().resolve()),
            out_put_dir=(
                str(Path(args.output_dir).expanduser().resolve())
                if args.output_dir
                else None
            ),
            score_threshold=float(args.score_threshold),
            overlap_threshold=float(args.overlap_threshold),
            max_iterations=int(args.max_iterations),
            batch_size=int(args.batch_size),
            model_pth_path=(
                str(Path(args.weights).expanduser().resolve()) if args.weights else None
            ),
            model_config=str(args.model_config),
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
