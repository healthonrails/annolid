from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from annolid.behavior.data_loading.datasets import BehaviorDataset
from annolid.behavior.pipeline import (
    BACKBONE_CHOICES,
    DEFAULT_DINOV3_MODEL,
    build_classifier,
    build_transform,
)
from annolid.engine.registry import ModelPluginBase, register_model
from annolid.utils.runs import allocate_run_dir, shared_runs_root

logger = logging.getLogger(__name__)


@register_model
class BehaviorClassifierPlugin(ModelPluginBase):
    name = "behavior_classifier"
    description = "Video-folder behavior classifier (train/infer)."

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--video-folder", required=True, help="Folder of labeled videos"
        )
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument("--epochs", type=int, default=10)
        parser.add_argument("--learning-rate", type=float, default=1e-3)
        parser.add_argument("--checkpoint-dir", default="checkpoints")
        parser.add_argument(
            "--tensorboard-log-dir",
            default="",
            help="TensorBoard log directory (default: shared Annolid runs root).",
        )
        parser.add_argument("--validation-split", type=float, default=0.2)
        parser.add_argument(
            "--feature-backbone", choices=BACKBONE_CHOICES, default="dinov3"
        )
        parser.add_argument(
            "--dinov3-model-name", type=str, default=DEFAULT_DINOV3_MODEL
        )
        parser.add_argument("--unfreeze-dinov3", action="store_true")
        parser.add_argument("--feature-dim", type=int, default=None)
        parser.add_argument("--transformer-dim", type=int, default=768)
        parser.add_argument(
            "--device", default=None, help="cuda|mps|cpu (default: auto)"
        )

    def train(self, args: argparse.Namespace) -> int:
        from annolid.behavior.training.train import train_model

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        if args.device:
            device = torch.device(str(args.device))
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = build_transform(args.feature_backbone)
        dataset = BehaviorDataset(str(args.video_folder), transform=transform)
        num_classes = int(dataset.get_num_classes())

        val_size = int(float(args.validation_split) * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset, batch_size=int(args.batch_size), shuffle=True, num_workers=4
        )
        val_loader = DataLoader(
            val_dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=4
        )

        model = build_classifier(
            num_classes=num_classes,
            backbone=str(args.feature_backbone),
            device=device,
            dinov3_model=str(args.dinov3_model_name),
            feature_dim=args.feature_dim,
            transformer_dim=int(args.transformer_dim),
            unfreeze_dinov3=bool(args.unfreeze_dinov3),
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=float(args.learning_rate))
        criterion = nn.CrossEntropyLoss()

        checkpoint_dir = str(Path(args.checkpoint_dir).expanduser().resolve())
        os.makedirs(checkpoint_dir, exist_ok=True)
        tb_dir_raw = str(getattr(args, "tensorboard_log_dir", "") or "").strip()
        if tb_dir_raw:
            tb_dir = Path(tb_dir_raw).expanduser().resolve()
        else:
            tb_dir = (
                allocate_run_dir(
                    task="behavior_classifier",
                    model=str(args.feature_backbone),
                    runs_root=shared_runs_root(),
                    run_name=Path(str(args.video_folder)).name,
                )
                / "tensorboard"
            )
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(str(tb_dir))
        train_model(
            model,
            train_loader,
            val_loader,
            int(args.epochs),
            device,
            optimizer,
            criterion,
            checkpoint_dir,
            writer,
        )

        print(str(Path(checkpoint_dir) / "best_model.pth"))
        return 0

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--video-folder", required=True, help="Folder of videos for inference"
        )
        parser.add_argument(
            "--checkpoint-path", required=True, help="Path to trained weights (.pth)"
        )
        parser.add_argument("--batch-size", type=int, default=1)
        parser.add_argument(
            "--feature-backbone", choices=BACKBONE_CHOICES, default="clip"
        )
        parser.add_argument(
            "--dinov3-model-name", type=str, default=DEFAULT_DINOV3_MODEL
        )
        parser.add_argument("--feature-dim", type=int, default=None)
        parser.add_argument("--transformer-dim", type=int, default=768)
        parser.add_argument(
            "--device", default=None, help="cuda|mps|cpu (default: auto)"
        )

    def predict(self, args: argparse.Namespace) -> int:
        from annolid.behavior.inference import predict as run_predict
        from annolid.behavior.pipeline import load_classifier

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        if args.device:
            device = torch.device(str(args.device))
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        transform = build_transform(args.feature_backbone)
        dataset = BehaviorDataset(str(args.video_folder), transform=transform)
        num_classes = int(dataset.get_num_classes())
        loader = DataLoader(
            dataset, batch_size=int(args.batch_size), shuffle=False, num_workers=4
        )

        model = load_classifier(
            checkpoint_path=str(Path(args.checkpoint_path).expanduser().resolve()),
            num_classes=num_classes,
            backbone=str(args.feature_backbone),
            device=device,
            dinov3_model=str(args.dinov3_model_name),
            feature_dim=args.feature_dim,
            transformer_dim=int(args.transformer_dim),
        )
        preds = run_predict(model, loader, device)
        for video_name, pred in preds:
            print(f"{video_name}\t{int(pred)}")
        return 0
