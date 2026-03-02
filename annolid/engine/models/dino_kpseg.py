from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from annolid.engine.registry import ModelPluginBase, register_model
from annolid.utils.runs import allocate_run_dir, shared_runs_root


from annolid.segmentation.dino_kpseg.cli_utils import parse_layers
from annolid.segmentation.dino_kpseg import defaults as dino_defaults


def _parse_weight_list(text: Optional[str], *, n: int) -> tuple[float, ...]:
    raw = str(text or "").strip()
    if not raw:
        return tuple(1.0 for _ in range(int(n)))
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != int(n):
        raise ValueError(f"Expected {int(n)} comma-separated floats, got {len(parts)}")
    out = []
    for token in parts:
        out.append(float(token))
    return tuple(out)


@register_model
class DinoKPSEGPlugin(ModelPluginBase):
    name = "dino_kpseg"
    description = "DINOv3 feature + small conv head for keypoint mask segmentation."

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--data",
            required=True,
            help="Path to dataset YAML (YOLO pose, LabelMe spec, or COCO spec)",
        )
        parser.add_argument(
            "--data-format",
            choices=("auto", "yolo", "labelme", "coco"),
            default="auto",
            help="Dataset annotation format (default: auto-detect from YAML).",
        )
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
        parser.add_argument(
            "--feature-merge",
            choices=("concat", "mean", "max"),
            default=dino_defaults.FEATURE_MERGE,
            help="How to merge multi-layer DINO features.",
        )
        parser.add_argument(
            "--feature-align-dim",
            default=str(dino_defaults.FEATURE_ALIGN_DIM),
            help="Optional trainable 1x1 feature alignment dim (0=off or auto).",
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
        parser.add_argument("--batch", type=int, default=dino_defaults.BATCH)
        parser.add_argument(
            "--accumulate",
            type=int,
            default=1,
            help="Gradient accumulation steps.",
        )
        parser.add_argument(
            "--grad-clip",
            type=float,
            default=1.0,
            help="Gradient clipping max norm (0=off).",
        )
        parser.add_argument(
            "--log-every-steps",
            type=int,
            default=100,
            help="Emit step-level progress logs every N train/val batches (0=off).",
        )
        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--balanced-bce",
            dest="balanced_bce",
            action="store_true",
            help="Enable per-batch positive class reweighting for BCE (recommended).",
        )
        group.add_argument(
            "--no-balanced-bce",
            dest="balanced_bce",
            action="store_false",
            help="Disable positive class reweighting for BCE.",
        )
        parser.set_defaults(balanced_bce=True)
        parser.add_argument(
            "--max-pos-weight",
            type=float,
            default=50.0,
            help="Clamp for balanced BCE pos_weight (prevents instability).",
        )
        parser.add_argument("--threshold", type=float, default=dino_defaults.THRESHOLD)
        parser.add_argument("--device", default=None)
        parser.add_argument(
            "--no-cache", action="store_true", help="Disable feature caching"
        )
        sched_group = parser.add_mutually_exclusive_group()
        sched_group.add_argument(
            "--cos-lr",
            dest="cos_lr",
            action="store_true",
            help="Use cosine LR decay (recommended).",
        )
        sched_group.add_argument(
            "--no-cos-lr",
            dest="cos_lr",
            action="store_false",
            help="Disable cosine LR decay.",
        )
        parser.set_defaults(cos_lr=True)
        parser.add_argument(
            "--warmup-epochs",
            type=int,
            default=3,
            help="Linear LR warmup epochs (0=off).",
        )
        parser.add_argument(
            "--lrf",
            "--lr-final-frac",
            dest="lr_final_frac",
            type=float,
            default=0.01,
            help="Final LR fraction for cosine schedule.",
        )
        parser.add_argument(
            "--flat-epoch",
            type=int,
            default=dino_defaults.FLAT_EPOCH,
            help="Hold base LR through this epoch before cosine decay starts (0=off).",
        )
        parser.add_argument(
            "--schedule-profile",
            choices=("baseline", "aggressive_s"),
            default=dino_defaults.SCHEDULE_PROFILE,
            help="Optional schedule preset. aggressive_s sets epoch windows to [4,64,120] with 12 no-aug tail.",
        )
        parser.add_argument(
            "--head-type",
            choices=("conv", "relational", "multitask"),
            default=dino_defaults.HEAD_TYPE,
            help="Head architecture.",
        )
        parser.add_argument(
            "--relational-heads",
            "--attn-heads",
            dest="attn_heads",
            type=int,
            default=dino_defaults.ATTN_HEADS,
            help="Relational attention heads (legacy alias: --attn-heads).",
        )
        parser.add_argument(
            "--relational-layers",
            "--attn-layers",
            dest="attn_layers",
            type=int,
            default=dino_defaults.ATTN_LAYERS,
            help="Relational attention layers (legacy alias: --attn-layers).",
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
            "--obj-loss-weight",
            type=float,
            default=dino_defaults.OBJ_LOSS_WEIGHT,
            help="Auxiliary objectness loss weight (multitask head).",
        )
        parser.add_argument(
            "--box-loss-weight",
            type=float,
            default=dino_defaults.BOX_LOSS_WEIGHT,
            help="Auxiliary box regression loss weight (multitask head).",
        )
        parser.add_argument(
            "--inst-loss-weight",
            type=float,
            default=dino_defaults.INST_LOSS_WEIGHT,
            help="Auxiliary instance-mask loss weight (multitask head).",
        )
        parser.add_argument(
            "--multitask-aux-warmup-epochs",
            type=int,
            default=dino_defaults.MULTITASK_AUX_WARMUP_EPOCHS,
            help="Warm up multitask auxiliary losses over N epochs (0=off).",
        )
        ema_group = parser.add_mutually_exclusive_group()
        ema_group.add_argument(
            "--ema",
            dest="use_ema",
            action="store_true",
            help="Enable EMA model for validation/checkpoints.",
        )
        ema_group.add_argument(
            "--no-ema",
            dest="use_ema",
            action="store_false",
            help="Disable EMA model.",
        )
        parser.set_defaults(use_ema=bool(dino_defaults.USE_EMA))
        parser.add_argument(
            "--ema-decay",
            type=float,
            default=dino_defaults.EMA_DECAY,
            help="EMA decay factor (0..1).",
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
            default=dino_defaults.RADIUS_SCHEDULE,
            help="Schedule radius_px across epochs.",
        )
        parser.add_argument(
            "--radius-start-px",
            type=float,
            default=dino_defaults.RADIUS_START_PX,
        )
        parser.add_argument(
            "--radius-end-px",
            type=float,
            default=dino_defaults.RADIUS_END_PX,
        )
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
            "--best-metric",
            choices=("pck@8px", "pck_weighted", "val_loss", "train_loss"),
            default=dino_defaults.BEST_METRIC,
            help="Metric for best checkpoint selection.",
        )
        parser.add_argument(
            "--early-stop-metric",
            choices=("auto", "pck@8px", "pck_weighted", "val_loss", "train_loss"),
            default=dino_defaults.EARLY_STOP_METRIC,
            help="Metric for early stopping (default: auto -> same as best-metric).",
        )
        parser.add_argument(
            "--pck-weighted-weights",
            type=str,
            default=dino_defaults.PCK_WEIGHTED_WEIGHTS,
            help="Comma-separated weights for pck_weighted thresholds [2,4,8,16].",
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
            "--aug-start-epoch",
            type=int,
            default=dino_defaults.AUG_START_EPOCH,
            help="First epoch where train augmentations are active.",
        )
        parser.add_argument(
            "--aug-stop-epoch",
            type=int,
            default=dino_defaults.AUG_STOP_EPOCH,
            help="Last epoch where train augmentations are active (0=until no-aug tail).",
        )
        parser.add_argument(
            "--no-aug-epoch",
            type=int,
            default=dino_defaults.NO_AUG_EPOCH,
            help="Disable train augmentations for the final N epochs (0=off).",
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
            data_format=str(args.data_format),
            output_dir=out_dir,
            model_name=str(args.model_name),
            short_side=int(args.short_side),
            layers=layers,
            feature_merge=str(
                getattr(args, "feature_merge", dino_defaults.FEATURE_MERGE)
            ),
            feature_align_dim=getattr(
                args, "feature_align_dim", dino_defaults.FEATURE_ALIGN_DIM
            ),
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
            batch_size=int(getattr(args, "batch", dino_defaults.BATCH)),
            accumulate=int(getattr(args, "accumulate", 1)),
            grad_clip=float(getattr(args, "grad_clip", 1.0)),
            balanced_bce=bool(getattr(args, "balanced_bce", True)),
            max_pos_weight=float(getattr(args, "max_pos_weight", 50.0)),
            cos_lr=bool(getattr(args, "cos_lr", True)),
            warmup_epochs=int(getattr(args, "warmup_epochs", 3)),
            lr_final_frac=float(getattr(args, "lr_final_frac", 0.01)),
            flat_epoch=int(getattr(args, "flat_epoch", dino_defaults.FLAT_EPOCH)),
            schedule_profile=str(
                getattr(args, "schedule_profile", dino_defaults.SCHEDULE_PROFILE)
            ),
            device=(str(args.device).strip() if args.device else None),
            cache_features=not bool(args.no_cache),
            head_type=str(args.head_type),
            attn_heads=int(args.attn_heads),
            attn_layers=int(args.attn_layers),
            lr_pair_loss_weight=float(args.lr_pair_loss_weight),
            lr_pair_margin_px=float(args.lr_pair_margin_px),
            lr_side_loss_weight=float(args.lr_side_loss_weight),
            lr_side_loss_margin=float(args.lr_side_loss_margin),
            log_every_steps=int(getattr(args, "log_every_steps", 100)),
            dice_loss_weight=float(args.dice_loss_weight),
            coord_loss_weight=float(args.coord_loss_weight),
            coord_loss_type=str(args.coord_loss_type),
            obj_loss_weight=float(
                getattr(args, "obj_loss_weight", dino_defaults.OBJ_LOSS_WEIGHT)
            ),
            box_loss_weight=float(
                getattr(args, "box_loss_weight", dino_defaults.BOX_LOSS_WEIGHT)
            ),
            inst_loss_weight=float(
                getattr(args, "inst_loss_weight", dino_defaults.INST_LOSS_WEIGHT)
            ),
            multitask_aux_warmup_epochs=int(
                getattr(
                    args,
                    "multitask_aux_warmup_epochs",
                    dino_defaults.MULTITASK_AUX_WARMUP_EPOCHS,
                )
            ),
            use_ema=bool(getattr(args, "use_ema", dino_defaults.USE_EMA)),
            ema_decay=float(getattr(args, "ema_decay", dino_defaults.EMA_DECAY)),
            bce_type=str(getattr(args, "bce_type", dino_defaults.BCE_TYPE)),
            focal_alpha=float(getattr(args, "focal_alpha", dino_defaults.FOCAL_ALPHA)),
            focal_gamma=float(getattr(args, "focal_gamma", dino_defaults.FOCAL_GAMMA)),
            coord_warmup_epochs=int(
                getattr(args, "coord_warmup_epochs", dino_defaults.COORD_WARMUP_EPOCHS)
            ),
            radius_schedule=str(
                getattr(args, "radius_schedule", dino_defaults.RADIUS_SCHEDULE)
            ),
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
            best_metric=str(getattr(args, "best_metric", dino_defaults.BEST_METRIC)),
            early_stop_metric=str(
                getattr(args, "early_stop_metric", dino_defaults.EARLY_STOP_METRIC)
            ),
            pck_weighted_weights=_parse_weight_list(
                getattr(
                    args, "pck_weighted_weights", dino_defaults.PCK_WEIGHTED_WEIGHTS
                ),
                n=4,
            ),
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
            aug_start_epoch=int(
                getattr(args, "aug_start_epoch", dino_defaults.AUG_START_EPOCH)
            ),
            aug_stop_epoch=int(
                getattr(args, "aug_stop_epoch", dino_defaults.AUG_STOP_EPOCH)
            ),
            no_aug_epoch=int(getattr(args, "no_aug_epoch", dino_defaults.NO_AUG_EPOCH)),
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
            "--tta-hflip",
            action="store_true",
            help="Enable horizontal-flip test-time augmentation.",
        )
        parser.add_argument(
            "--tta-merge",
            choices=("mean", "max"),
            default="mean",
            help="How to merge original and flipped predictions when --tta-hflip is enabled.",
        )
        parser.add_argument(
            "--min-keypoint-score",
            type=float,
            default=0.0,
            help="Drop keypoints below this confidence score in output payload.",
        )
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
        from annolid.segmentation.dino_kpseg.inference_utils import (
            filter_keypoints_by_score,
        )

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
            tta_hflip=bool(getattr(args, "tta_hflip", False)),
            tta_merge=str(getattr(args, "tta_merge", "mean")),
        )
        pred, kept_indices = filter_keypoints_by_score(
            pred,
            min_score=float(getattr(args, "min_keypoint_score", 0.0)),
            return_indices=True,
        )
        keypoint_names = predictor.keypoint_names
        if keypoint_names is not None:
            keypoint_names = [
                str(keypoint_names[int(i)])
                for i in kept_indices
                if 0 <= int(i) < len(keypoint_names)
            ]

        payload = {
            "model": "dino_kpseg",
            "weights": str(Path(args.weights).expanduser().resolve()),
            "image": str(img_path),
            "keypoints_xy": [[float(x), float(y)] for x, y in pred.keypoints_xy],
            "keypoint_scores": [float(s) for s in pred.keypoint_scores],
            "keypoint_names": keypoint_names,
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
