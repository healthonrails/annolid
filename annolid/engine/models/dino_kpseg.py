from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from annolid.engine.registry import ModelPluginBase, register_model
from annolid.utils.runs import new_run_dir, shared_runs_root


from annolid.segmentation.dino_kpseg.cli_utils import parse_layers


@register_model
class DinoKPSEGPlugin(ModelPluginBase):
    name = "dino_kpseg"
    description = "DINOv3 feature + small conv head for keypoint mask segmentation."

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data", required=True,
                            help="Path to YOLO pose data.yaml")
        parser.add_argument("--output", default=None,
                            help="Run output directory (optional)")
        parser.add_argument("--runs-root", default=None,
                            help="Runs root (optional)")
        parser.add_argument("--run-name", default=None,
                            help="Optional run name (default: timestamp)")
        parser.add_argument(
            "--model-name",
            default="facebook/dinov3-vits16-pretrain-lvd1689m",
            help="Hugging Face model id or dinov3 alias",
        )
        parser.add_argument("--short-side", type=int, default=768)
        parser.add_argument("--layers", type=str, default="-1",
                            help="Comma-separated transformer block indices")
        parser.add_argument("--radius-px", type=float, default=6.0)
        parser.add_argument("--hidden-dim", type=int, default=128)
        parser.add_argument("--lr", type=float, default=2e-3)
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--threshold", type=float, default=0.4)
        parser.add_argument("--device", default=None)
        parser.add_argument("--no-cache", action="store_true",
                            help="Disable feature caching")
        parser.add_argument("--early-stop-patience", type=int,
                            default=0, help="Early stop patience (0=off)")
        parser.add_argument("--early-stop-min-delta", type=float,
                            default=0.0, help="Min metric improvement to reset patience")
        parser.add_argument("--early-stop-min-epochs", type=int,
                            default=0, help="Do not early-stop before this epoch")
        parser.add_argument("--tb-add-graph", action="store_true",
                            help="Export model graph to TensorBoard (can be slow)")
        parser.add_argument("--augment", action="store_true",
                            help="Enable YOLO-like pose augmentations")
        parser.add_argument("--hflip", type=float, default=0.5,
                            help="Horizontal flip probability")
        parser.add_argument("--degrees", type=float, default=0.0,
                            help="Random rotation degrees (+/-)")
        parser.add_argument("--translate", type=float,
                            default=0.0, help="Random translate fraction (+/-)")
        parser.add_argument("--scale", type=float, default=0.0,
                            help="Random scale fraction (+/-)")
        parser.add_argument("--brightness", type=float,
                            default=0.0, help="Brightness jitter (+/-)")
        parser.add_argument("--contrast", type=float,
                            default=0.0, help="Contrast jitter (+/-)")
        parser.add_argument("--saturation", type=float,
                            default=0.0, help="Saturation jitter (+/-)")
        parser.add_argument("--seed", type=int, default=None,
                            help="Optional augmentation RNG seed")

    def train(self, args: argparse.Namespace) -> int:
        from annolid.segmentation.dino_kpseg.train import train as train_kpseg
        from annolid.segmentation.dino_kpseg.data import DinoKPSEGAugmentConfig

        layers = parse_layers(args.layers)
        if args.output:
            out_dir = Path(args.output).expanduser().resolve()
        else:
            runs_root = (
                Path(args.runs_root).expanduser().resolve()
                if args.runs_root
                else shared_runs_root()
            )
            out_dir = new_run_dir(
                task="dino_kpseg", model="train", runs_root=runs_root, run_name=args.run_name)
        best = train_kpseg(
            data_yaml=Path(args.data).expanduser().resolve(),
            output_dir=out_dir,
            model_name=str(args.model_name),
            short_side=int(args.short_side),
            layers=layers,
            radius_px=float(args.radius_px),
            hidden_dim=int(args.hidden_dim),
            lr=float(args.lr),
            epochs=int(args.epochs),
            threshold=float(args.threshold),
            device=(str(args.device).strip() if args.device else None),
            cache_features=not bool(args.no_cache),
            early_stop_patience=int(args.early_stop_patience),
            early_stop_min_delta=float(args.early_stop_min_delta),
            early_stop_min_epochs=int(args.early_stop_min_epochs),
            tb_add_graph=bool(args.tb_add_graph),
            augment=DinoKPSEGAugmentConfig(
                enabled=bool(args.augment),
                hflip_prob=float(args.hflip),
                degrees=float(args.degrees),
                translate=float(args.translate),
                scale=float(args.scale),
                brightness=float(args.brightness),
                contrast=float(args.contrast),
                saturation=float(args.saturation),
                seed=(int(args.seed) if args.seed is not None else None),
            ),
        )
        print(str(best))
        return 0

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--weights", required=True,
                            help="Path to DinoKPSEG checkpoint (.pt)")
        parser.add_argument("--image", required=True, help="Input image path")
        parser.add_argument("--device", default=None)
        parser.add_argument("--threshold", type=float, default=None)
        parser.add_argument("--out", default=None,
                            help="Optional JSON output path (default: stdout)")
        parser.add_argument(
            "--return-patch-masks",
            action="store_true",
            help="Include patch-grid masks in the output JSON (can be large).",
        )

    def predict(self, args: argparse.Namespace) -> int:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError(
                "DinoKPSEG predict requires opencv-python.") from exc

        from annolid.segmentation.dino_kpseg.predictor import DinoKPSEGPredictor

        img_path = Path(args.image).expanduser().resolve()
        frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        predictor = DinoKPSEGPredictor(
            Path(args.weights).expanduser().resolve(), device=args.device)
        pred = predictor.predict(
            frame_bgr,
            threshold=args.threshold,
            return_patch_masks=bool(args.return_patch_masks),
        )

        payload = {
            "model": "dino_kpseg",
            "weights": str(Path(args.weights).expanduser().resolve()),
            "image": str(img_path),
            "keypoints_xy": [[float(x), float(y)] for x, y in pred.keypoints_xy],
            "keypoint_scores": [float(s) for s in pred.keypoint_scores],
            "keypoint_names": predictor.keypoint_names,
            "resized_hw": [int(pred.resized_hw[0]), int(pred.resized_hw[1])],
            "patch_size": int(pred.patch_size),
            "masks_patch": pred.masks_patch.tolist() if pred.masks_patch is not None else None,
        }

        out_path: Optional[str] = args.out
        text = json.dumps(payload, indent=2)
        if out_path:
            Path(out_path).expanduser().resolve(
            ).write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0
