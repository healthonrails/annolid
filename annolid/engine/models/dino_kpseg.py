from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from annolid.engine.registry import ModelPluginBase, register_model
from annolid.utils.runs import allocate_run_dir, shared_runs_root


from annolid.segmentation.dino_kpseg.cli_utils import parse_layers
from annolid.segmentation.dino_kpseg import defaults as dino_defaults


@register_model
class DinoKPSEGPlugin(ModelPluginBase):
    name = "dino_kpseg"
    description = "DINOv3 feature + small conv head for keypoint mask segmentation."

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--data", required=True, help="Path to YOLO pose data.yaml")
        parser.add_argument(
            "--output", default=None, help="Run output directory (optional)"
        )
        parser.add_argument("--runs-root", default=None, help="Runs root (optional)")
        parser.add_argument(
            "--run-name", default=None, help="Optional run name (default: timestamp)"
        )
        parser.add_argument(
            "--model-name",
            default=dino_defaults.MODEL_NAME,
            help="Hugging Face model id or dinov3 alias",
        )
        parser.add_argument("--short-side", type=int, default=dino_defaults.SHORT_SIDE)
        parser.add_argument(
            "--layers",
            type=str,
            default=dino_defaults.LAYERS,
            help="Comma-separated transformer block indices",
        )
        parser.add_argument("--radius-px", type=float, default=dino_defaults.RADIUS_PX)
        parser.add_argument(
            "--mask-type",
            choices=("disk", "gaussian"),
            default=dino_defaults.MASK_TYPE,
            help="Keypoint supervision mask type",
        )
        parser.add_argument(
            "--heatmap-sigma",
            type=float,
            default=None,
            help="Gaussian sigma in pixels (original image space). Defaults to radius_px/2.",
        )
        parser.add_argument(
            "--instance-mode",
            choices=("auto", "union", "per_instance"),
            default=dino_defaults.INSTANCE_MODE,
            help="How to handle multiple pose instances per image.",
        )
        parser.add_argument(
            "--bbox-scale",
            type=float,
            default=dino_defaults.BBOX_SCALE,
            help="Scale factor for per-instance bounding box crops.",
        )
        parser.add_argument("--hidden-dim", type=int, default=dino_defaults.HIDDEN_DIM)
        parser.add_argument("--lr", type=float, default=dino_defaults.LR)
        parser.add_argument("--epochs", type=int, default=dino_defaults.EPOCHS)
        parser.add_argument("--threshold", type=float, default=dino_defaults.THRESHOLD)
        parser.add_argument("--device", default=None)
        parser.add_argument(
            "--no-cache", action="store_true", help="Disable feature caching"
        )
        parser.add_argument(
            "--head-type",
            choices=("conv", "attn", "hybrid"),
            default=dino_defaults.HEAD_TYPE,
            help="Head architecture",
        )
        parser.add_argument(
            "--attn-heads",
            type=int,
            default=dino_defaults.ATTN_HEADS,
            help="Attention heads (attn head only)",
        )
        parser.add_argument(
            "--attn-layers",
            type=int,
            default=dino_defaults.ATTN_LAYERS,
            help="Attention layers (attn head only)",
        )
        parser.add_argument(
            "--lr-pair-loss-weight",
            type=float,
            default=dino_defaults.LR_PAIR_LOSS_WEIGHT,
            help="Optional symmetric-pair regularizer weight (0=off).",
        )
        parser.add_argument(
            "--lr-pair-margin-px",
            type=float,
            default=dino_defaults.LR_PAIR_MARGIN_PX,
            help="Optional minimum separation margin in pixels for symmetric pairs (0=off).",
        )
        parser.add_argument(
            "--lr-side-loss-weight",
            type=float,
            default=dino_defaults.LR_SIDE_LOSS_WEIGHT,
            help="Optional left/right side-consistency loss weight (0=off). Uses orientation anchors when available.",
        )
        parser.add_argument(
            "--lr-side-loss-margin",
            type=float,
            default=dino_defaults.LR_SIDE_LOSS_MARGIN,
            help="Margin for side-consistency in [0,1] (0=enforce opposite sign).",
        )
        parser.add_argument(
            "--dice-loss-weight",
            type=float,
            default=dino_defaults.DICE_LOSS_WEIGHT,
            help="Dice loss weight (0=off).",
        )
        parser.add_argument(
            "--coord-loss-weight",
            type=float,
            default=dino_defaults.COORD_LOSS_WEIGHT,
            help="Coordinate regression loss weight (0=off).",
        )
        parser.add_argument(
            "--coord-loss-type",
            choices=("smooth_l1", "l1", "l2"),
            default=dino_defaults.COORD_LOSS_TYPE,
            help="Coordinate regression loss type.",
        )
        parser.add_argument(
            "--bce-type",
            choices=("bce", "focal"),
            default=dino_defaults.BCE_TYPE,
            help="Loss type for mask supervision (default: bce).",
        )
        parser.add_argument(
            "--focal-alpha",
            type=float,
            default=dino_defaults.FOCAL_ALPHA,
            help="Alpha for focal BCE.",
        )
        parser.add_argument(
            "--focal-gamma",
            type=float,
            default=dino_defaults.FOCAL_GAMMA,
            help="Gamma for focal BCE.",
        )
        parser.add_argument(
            "--coord-warmup-epochs",
            type=int,
            default=dino_defaults.COORD_WARMUP_EPOCHS,
            help="Warm up coordinate loss over N epochs (0=off).",
        )
        parser.add_argument(
            "--radius-schedule",
            choices=("none", "linear"),
            default="none",
            help="Schedule radius_px across epochs (default: none).",
        )
        parser.add_argument("--radius-start-px", type=float, default=None)
        parser.add_argument("--radius-end-px", type=float, default=None)
        parser.add_argument(
            "--overfit-n",
            type=int,
            default=0,
            help="Overfit mode: train/val on N images (0=off).",
        )
        parser.add_argument(
            "--early-stop-patience",
            type=int,
            default=dino_defaults.EARLY_STOP_PATIENCE,
            help="Early stop patience (0=off)",
        )
        parser.add_argument(
            "--early-stop-min-delta",
            type=float,
            default=dino_defaults.EARLY_STOP_MIN_DELTA,
            help="Min metric improvement to reset patience",
        )
        parser.add_argument(
            "--early-stop-min-epochs",
            type=int,
            default=dino_defaults.EARLY_STOP_MIN_EPOCHS,
            help="Do not early-stop before this epoch",
        )
        parser.add_argument(
            "--tb-add-graph",
            action="store_true",
            help="Export model graph to TensorBoard (can be slow)",
        )
        parser.add_argument(
            "--tb-projector",
            action="store_true",
            help="Write a TensorBoard Projector embedding view for DinoKPSEG patch features.",
        )
        parser.add_argument(
            "--tb-projector-split",
            choices=("train", "val", "both"),
            default=dino_defaults.TB_PROJECTOR_SPLIT,
            help="Which dataset split(s) to sample for the projector (default: val).",
        )
        parser.add_argument(
            "--tb-projector-max-images",
            type=int,
            default=dino_defaults.TB_PROJECTOR_MAX_IMAGES,
        )
        parser.add_argument(
            "--tb-projector-max-patches",
            type=int,
            default=dino_defaults.TB_PROJECTOR_MAX_PATCHES,
        )
        parser.add_argument(
            "--tb-projector-per-image-per-keypoint",
            type=int,
            default=dino_defaults.TB_PROJECTOR_PER_IMAGE_PER_KEYPOINT,
        )
        parser.add_argument(
            "--tb-projector-pos-threshold",
            type=float,
            default=dino_defaults.TB_PROJECTOR_POS_THRESHOLD,
        )
        parser.add_argument(
            "--tb-projector-crop-px",
            type=int,
            default=dino_defaults.TB_PROJECTOR_CROP_PX,
        )
        parser.add_argument(
            "--tb-projector-sprite-border-px",
            type=int,
            default=dino_defaults.TB_PROJECTOR_SPRITE_BORDER_PX,
        )
        parser.add_argument("--tb-projector-add-negatives", action="store_true")
        parser.add_argument(
            "--tb-projector-neg-threshold",
            type=float,
            default=dino_defaults.TB_PROJECTOR_NEG_THRESHOLD,
        )
        parser.add_argument(
            "--tb-projector-negatives-per-image",
            type=int,
            default=dino_defaults.TB_PROJECTOR_NEGATIVES_PER_IMAGE,
        )
        parser.set_defaults(
            tb_projector_add_negatives=bool(dino_defaults.TB_PROJECTOR_ADD_NEGATIVES)
        )
        aug_group = parser.add_mutually_exclusive_group()
        aug_group.add_argument(
            "--augment",
            dest="augment",
            action="store_true",
            help="Enable augmentations",
        )
        aug_group.add_argument(
            "--no-augment",
            dest="augment",
            action="store_false",
            help="Disable augmentations",
        )
        parser.set_defaults(
            augment=bool(dino_defaults.AUGMENT_ENABLED),
            tb_projector=bool(dino_defaults.TB_PROJECTOR),
        )
        parser.add_argument(
            "--hflip",
            type=float,
            default=dino_defaults.HFLIP,
            help="Horizontal flip probability",
        )
        parser.add_argument(
            "--degrees",
            type=float,
            default=dino_defaults.DEGREES,
            help="Random rotation degrees (+/-)",
        )
        parser.add_argument(
            "--translate",
            type=float,
            default=dino_defaults.TRANSLATE,
            help="Random translate fraction (+/-)",
        )
        parser.add_argument(
            "--scale",
            type=float,
            default=dino_defaults.SCALE,
            help="Random scale fraction (+/-)",
        )
        parser.add_argument(
            "--brightness",
            type=float,
            default=dino_defaults.BRIGHTNESS,
            help="Brightness jitter (+/-)",
        )
        parser.add_argument(
            "--contrast",
            type=float,
            default=dino_defaults.CONTRAST,
            help="Contrast jitter (+/-)",
        )
        parser.add_argument(
            "--saturation",
            type=float,
            default=dino_defaults.SATURATION,
            help="Saturation jitter (+/-)",
        )
        parser.add_argument(
            "--seed",
            type=int,
            default=None,
            help="Optional RNG seed (also used for augmentations)",
        )

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
            out_dir = allocate_run_dir(
                task="dino_kpseg",
                model="train",
                runs_root=runs_root,
                run_name=args.run_name,
            )
        best = train_kpseg(
            data_yaml=Path(args.data).expanduser().resolve(),
            output_dir=out_dir,
            model_name=str(args.model_name),
            short_side=int(args.short_side),
            layers=layers,
            radius_px=float(args.radius_px),
            mask_type=str(args.mask_type),
            heatmap_sigma_px=(
                float(args.heatmap_sigma) if args.heatmap_sigma is not None else None
            ),
            instance_mode=str(args.instance_mode),
            bbox_scale=float(args.bbox_scale),
            hidden_dim=int(args.hidden_dim),
            lr=float(args.lr),
            epochs=int(args.epochs),
            threshold=float(args.threshold),
            device=(str(args.device).strip() if args.device else None),
            cache_features=not bool(args.no_cache),
            head_type=str(args.head_type),
            attn_heads=int(args.attn_heads),
            attn_layers=int(args.attn_layers),
            lr_pair_loss_weight=float(args.lr_pair_loss_weight),
            lr_pair_margin_px=float(args.lr_pair_margin_px),
            lr_side_loss_weight=float(args.lr_side_loss_weight),
            lr_side_loss_margin=float(args.lr_side_loss_margin),
            dice_loss_weight=float(args.dice_loss_weight),
            coord_loss_weight=float(args.coord_loss_weight),
            coord_loss_type=str(args.coord_loss_type),
            bce_type=str(getattr(args, "bce_type", dino_defaults.BCE_TYPE)),
            focal_alpha=float(getattr(args, "focal_alpha", dino_defaults.FOCAL_ALPHA)),
            focal_gamma=float(getattr(args, "focal_gamma", dino_defaults.FOCAL_GAMMA)),
            coord_warmup_epochs=int(
                getattr(args, "coord_warmup_epochs", dino_defaults.COORD_WARMUP_EPOCHS)
            ),
            radius_schedule=str(getattr(args, "radius_schedule", "none")),
            radius_start_px=(
                float(getattr(args, "radius_start_px", 0.0))
                if getattr(args, "radius_start_px", None) is not None
                else None
            ),
            radius_end_px=(
                float(getattr(args, "radius_end_px", 0.0))
                if getattr(args, "radius_end_px", None) is not None
                else None
            ),
            overfit_n=int(getattr(args, "overfit_n", 0)),
            seed=(int(args.seed) if args.seed is not None else None),
            early_stop_patience=int(args.early_stop_patience),
            early_stop_min_delta=float(args.early_stop_min_delta),
            early_stop_min_epochs=int(args.early_stop_min_epochs),
            tb_add_graph=bool(args.tb_add_graph),
            tb_projector=bool(
                getattr(args, "tb_projector", dino_defaults.TB_PROJECTOR)
            ),
            tb_projector_split=str(
                getattr(args, "tb_projector_split", dino_defaults.TB_PROJECTOR_SPLIT)
            ),
            tb_projector_max_images=int(
                getattr(
                    args,
                    "tb_projector_max_images",
                    dino_defaults.TB_PROJECTOR_MAX_IMAGES,
                )
            ),
            tb_projector_max_patches=int(
                getattr(
                    args,
                    "tb_projector_max_patches",
                    dino_defaults.TB_PROJECTOR_MAX_PATCHES,
                )
            ),
            tb_projector_per_image_per_keypoint=int(
                getattr(
                    args,
                    "tb_projector_per_image_per_keypoint",
                    dino_defaults.TB_PROJECTOR_PER_IMAGE_PER_KEYPOINT,
                )
            ),
            tb_projector_pos_threshold=float(
                getattr(
                    args,
                    "tb_projector_pos_threshold",
                    dino_defaults.TB_PROJECTOR_POS_THRESHOLD,
                )
            ),
            tb_projector_crop_px=int(
                getattr(
                    args, "tb_projector_crop_px", dino_defaults.TB_PROJECTOR_CROP_PX
                )
            ),
            tb_projector_sprite_border_px=int(
                getattr(
                    args,
                    "tb_projector_sprite_border_px",
                    dino_defaults.TB_PROJECTOR_SPRITE_BORDER_PX,
                )
            ),
            tb_projector_add_negatives=bool(
                getattr(
                    args,
                    "tb_projector_add_negatives",
                    dino_defaults.TB_PROJECTOR_ADD_NEGATIVES,
                )
            ),
            tb_projector_neg_threshold=float(
                getattr(
                    args,
                    "tb_projector_neg_threshold",
                    dino_defaults.TB_PROJECTOR_NEG_THRESHOLD,
                )
            ),
            tb_projector_negatives_per_image=int(
                getattr(
                    args,
                    "tb_projector_negatives_per_image",
                    dino_defaults.TB_PROJECTOR_NEGATIVES_PER_IMAGE,
                )
            ),
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
        parser.add_argument(
            "--weights", required=True, help="Path to DinoKPSEG checkpoint (.pt)"
        )
        parser.add_argument("--image", required=True, help="Input image path")
        parser.add_argument("--device", default=None)
        parser.add_argument("--threshold", type=float, default=None)
        parser.add_argument(
            "--out", default=None, help="Optional JSON output path (default: stdout)"
        )
        parser.add_argument(
            "--return-patch-masks",
            action="store_true",
            help="Include patch-grid masks in the output JSON (can be large).",
        )

    def predict(self, args: argparse.Namespace) -> int:
        try:
            import cv2  # type: ignore
        except Exception as exc:
            raise RuntimeError("DinoKPSEG predict requires opencv-python.") from exc

        from annolid.segmentation.dino_kpseg.predictor import DinoKPSEGPredictor

        img_path = Path(args.image).expanduser().resolve()
        frame_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")

        predictor = DinoKPSEGPredictor(
            Path(args.weights).expanduser().resolve(), device=args.device
        )
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
            "masks_patch": pred.masks_patch.tolist()
            if pred.masks_patch is not None
            else None,
        }

        out_path: Optional[str] = args.out
        text = json.dumps(payload, indent=2)
        if out_path:
            Path(out_path).expanduser().resolve().write_text(text, encoding="utf-8")
        else:
            print(text)
        return 0
