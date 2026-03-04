"""
Torchvision-based Mask R-CNN training for Annolid.

Uses ``torchvision.models.detection.maskrcnn_resnet50_fpn_v2`` instead of
detectron2 — no external build-from-source dependency required.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from annolid.segmentation.maskrcnn.coco_dataset import (
    build_train_loader,
    build_val_loader,
    load_class_names,
)
from annolid.utils.tb_utils import (
    _draw_loss_curve_image,
    _sanitize_tb_image_tensor,
    _try_create_summary_writer,
    sanitize_tensorboard_tag,
)
from annolid.utils.runs import allocate_run_dir, shared_runs_root


def _get_device() -> str:
    """Return the best available torch device for Mask R-CNN training.

    MPS (Apple Silicon GPU) is intentionally skipped because torchvision's
    Mask R-CNN triggers Metal command-buffer failures on MPS for certain
    batch configurations.  CUDA is used when available; otherwise CPU.
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _pin_memory_for_device(device: str) -> bool:
    """Only enable pin_memory on CUDA; MPS and CPU don't benefit and may warn."""
    return device == "cuda"


def _build_model(num_classes: int, weights: Optional[str] = None) -> torch.nn.Module:
    """Construct a Mask R-CNN model with the correct number of output classes.

    Args:
        num_classes: Number of foreground classes **+ 1** for background.
        weights: Path to a ``.pth`` checkpoint to load, or ``None`` to use
            COCO-pretrained weights from torchvision.
    """
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    if weights is None:
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

        model = maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        )
    else:
        model = maskrcnn_resnet50_fpn_v2(weights=None)

    # Replace the box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        # Handle both raw state_dict and {"model": state_dict} wrappers
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    return model


# ---------------------------------------------------------------------------
# TensorBoard logger
# ---------------------------------------------------------------------------


class MaskRCNNTBLogger:
    """TensorBoard logger for Mask R-CNN training.

    Mirrors the patterns from :class:`~annolid.yolo.tensorboard_logging.YOLORunTensorBoardLogger`
    and :class:`~annolid.segmentation.dino_kpseg.train` so all three model
    families share the same GUI entry point (``VisualizationWindow``).
    """

    def __init__(self, run_dir: str | Path, writer=None) -> None:
        self.run_dir = Path(run_dir).expanduser().resolve()
        self._writer = writer
        self._csv_path = self.run_dir / "training_log.csv"
        self._csv_file = None
        self._csv_writer = None
        self._last_step_logged: Optional[int] = None

    @property
    def writer(self):
        if self._writer is None:
            tb_dir = self.run_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self._writer = _try_create_summary_writer(tb_dir)
        return self._writer

    # --- lifecycle ----------------------------------------------------------

    def _ensure_csv(self) -> None:
        if self._csv_writer is not None:
            return
        self._csv_file = open(self._csv_path, "w", newline="", encoding="utf-8")
        self._csv_writer = csv.DictWriter(
            self._csv_file,
            fieldnames=[
                "iteration",
                "train_loss",
                "loss_classifier",
                "loss_box_reg",
                "loss_mask",
                "loss_objectness",
                "loss_rpn_box_reg",
                "lr",
            ],
        )
        self._csv_writer.writeheader()

    def close(self) -> None:
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception:
                pass
        w = self._writer
        if w is not None:
            try:
                w.flush()
                w.close()
            except Exception:
                pass

    # --- logging methods ----------------------------------------------------

    def log_hparams(self, hparams: Dict[str, Any]) -> None:
        """Write hyperparameters as TensorBoard text."""
        w = self.writer
        if w is None:
            return
        lines = ["**Mask R-CNN Hyperparameters**", ""]
        for k in sorted(hparams):
            lines.append(f"- `{k}`: {hparams[k]}")
        w.add_text("maskrcnn/hparams", "\n".join(lines), 0)
        w.flush()

    def log_iter(self, iteration: int, loss_dict: Dict[str, Any], lr: float) -> None:
        """Write per-iteration loss scalars and update training_log.csv."""
        w = self.writer
        total = sum(
            v.item() if hasattr(v, "item") else float(v) for v in loss_dict.values()
        )
        if w is not None:
            for k, v in loss_dict.items():
                tag = sanitize_tensorboard_tag(f"loss/{k}")
                w.add_scalar(
                    tag, float(v.item() if hasattr(v, "item") else v), iteration
                )
            w.add_scalar("loss/total", float(total), iteration)
            w.add_scalar("train/lr", float(lr), iteration)
        # CSV
        try:
            self._ensure_csv()
            row: Dict[str, Any] = {
                "iteration": iteration,
                "train_loss": total,
                "lr": lr,
            }
            for k, v in loss_dict.items():
                row[k] = float(v.item() if hasattr(v, "item") else v)
            self._csv_writer.writerow(row)  # type: ignore[union-attr]
        except Exception:
            pass
        self._last_step_logged = iteration

    def log_eval(self, eval_results: Dict[str, float], iteration: int) -> None:
        """Write COCO AP metrics."""
        w = self.writer
        if w is None:
            return
        for k, v in eval_results.items():
            tag = sanitize_tensorboard_tag(f"metrics/{k}")
            w.add_scalar(tag, float(v), iteration)
        w.flush()

    def log_loss_curve(self, iteration: int) -> None:
        """Render and emit the training loss curve as a TensorBoard image."""
        w = self.writer
        if w is None or not self._csv_path.exists():
            return
        try:
            img = _draw_loss_curve_image(
                self._csv_path,
                title="Mask R-CNN loss curve",
                step_key="iteration",
                train_key="train_loss",
                val_key="val_loss",
            )
            if img is not None:
                safe = _sanitize_tb_image_tensor(img)
                w.add_image("maskrcnn/loss_curve", safe, iteration)
                w.flush()
        except Exception:
            pass

    def log_sample_images(
        self, images: list, targets: list, iteration: int, max_samples: int = 4
    ) -> None:
        """Log a grid of training images with ground-truth masks and bounding boxes."""
        w = self.writer
        if w is None or not images:
            return
        try:
            import torchvision.utils as tv_utils  # type: ignore

            imgs_with_targets = []
            for i in range(min(len(images), max_samples)):
                img = images[i].cpu().clamp(0, 1)  # CHW float [0, 1]
                target = targets[i]

                # Convert to uint8 for torchvision drawing routines
                img_u8 = (img * 255).byte()

                # Draw masks
                if "masks" in target and len(target["masks"]) > 0:
                    masks = target["masks"].cpu().bool()
                    img_u8 = tv_utils.draw_segmentation_masks(img_u8, masks, alpha=0.5)

                # Draw boxes
                if "boxes" in target and len(target["boxes"]) > 0:
                    boxes = target["boxes"].cpu()
                    img_u8 = tv_utils.draw_bounding_boxes(
                        img_u8, boxes, width=2, colors="red"
                    )

                # Back to float [0, 1] for grid / tensorboard
                imgs_with_targets.append(img_u8.float() / 255.0)

            grid = tv_utils.make_grid(
                imgs_with_targets, nrow=min(len(imgs_with_targets), 4), padding=2
            )
            safe = _sanitize_tb_image_tensor(grid)
            w.add_image("maskrcnn/train_samples", safe, iteration)
            w.flush()
        except Exception:
            pass

    def log_prediction_images(
        self, images: list, predictions: list, iteration: int, max_samples: int = 4
    ) -> None:
        """Log a grid of validation images with predicted masks and bounding boxes.

        Args:
            images: List of CHW float tensors [0, 1]
            predictions: List of dicts with 'boxes', 'masks', 'scores'
        """
        w = self.writer
        if w is None or not images:
            return
        try:
            import torchvision.utils as tv_utils  # type: ignore

            imgs_with_preds = []
            for i in range(min(len(images), max_samples)):
                img = images[i].cpu().clamp(0, 1)  # CHW float [0, 1]
                pred = predictions[i]

                # Convert to uint8 for torchvision drawing routines
                img_u8 = (img * 255).byte()

                # Draw masks (only confident ones, e.g. > 0.3 or 0.5)
                # maskrcnn predicts soft masks [0, 1] for each detection
                if "masks" in pred and len(pred["masks"]) > 0:
                    # B, 1, H, W -> B, H, W -> boolean mask based on 0.5 threshold
                    masks = pred["masks"].squeeze(1).cpu() > 0.5
                    # Filter by score if available (to reduce visual noise)
                    if "scores" in pred:
                        keep = pred["scores"].cpu() > 0.3
                        masks = masks[keep]
                    if len(masks) > 0:
                        img_u8 = tv_utils.draw_segmentation_masks(
                            img_u8, masks, alpha=0.5
                        )

                # Draw boxes
                if "boxes" in pred and len(pred["boxes"]) > 0:
                    boxes = pred["boxes"].cpu()
                    if "scores" in pred:
                        keep = pred["scores"].cpu() > 0.3
                        boxes = boxes[keep]
                    if len(boxes) > 0:
                        img_u8 = tv_utils.draw_bounding_boxes(
                            img_u8, boxes, width=2, colors="blue"
                        )

                # Back to float [0, 1] for grid / tensorboard
                imgs_with_preds.append(img_u8.float() / 255.0)

            grid = tv_utils.make_grid(
                imgs_with_preds, nrow=min(len(imgs_with_preds), 4), padding=2
            )
            safe = _sanitize_tb_image_tensor(grid)
            w.add_image("maskrcnn/val_predictions", safe, iteration)
            w.flush()
        except Exception:
            pass

    def finalize(self, *, ok: bool, error: Optional[str] = None) -> None:
        """Write final status text and flush."""
        w = self.writer
        if w is None:
            return
        step = self._last_step_logged or 0
        self.log_loss_curve(step)
        w.add_text("annolid/status", "ok" if ok else "failed", step)
        if error:
            w.add_text("annolid/error", f"```\n{error}\n```", step)
        try:
            w.flush()
        except Exception:
            pass


class Segmentor:
    """Train and evaluate a torchvision Mask R-CNN model.

    Drop-in replacement for the previous detectron2-based ``Segmentor``.
    No detectron2 dependency required.
    """

    def __init__(
        self,
        dataset_dir=None,
        out_put_dir=None,
        score_threshold=0.15,
        overlap_threshold=0.7,
        max_iterations=3000,
        batch_size=8,
        model_pth_path=None,
        model_config=None,
        base_lr=0.0025,
        num_workers=2,
        checkpoint_period=1000,
        roi_batch_size_per_image=128,
        sampler_train="RepeatFactorTrainingSampler",
        repeat_threshold=0.3,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.checkpoint_period = checkpoint_period
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold
        self.base_lr = base_lr
        self.num_workers = num_workers

        self.logger = logging.getLogger(__name__)

        if out_put_dir is None:
            self.out_put_dir = str(
                allocate_run_dir(
                    task="maskrcnn",
                    model="train",
                    runs_root=shared_runs_root(),
                )
            )
        else:
            self.out_put_dir = out_put_dir

        os.makedirs(self.out_put_dir, exist_ok=True)

        # Load class names from the COCO annotations
        train_ann = Path(self.dataset_dir) / "train" / "annotations.json"
        self.class_names = load_class_names(str(train_ann))
        # +1 for background class
        self.num_classes = len(self.class_names) + 1

        self.logger.info(f"Classes ({len(self.class_names)}): {self.class_names}")
        self.logger.info(f"Max iterations: {max_iterations}")
        self.logger.info(f"Batch size: {batch_size}")
        self.logger.info(f"Dataset dir: {dataset_dir}")
        self.logger.info(f"Output dir: {self.out_put_dir}")

        self.device = _get_device()
        self.model = _build_model(self.num_classes, weights=model_pth_path)
        self.model.to(self.device)

        # TensorBoard logger — writes to {out_put_dir}/tensorboard/
        self.tb_logger = MaskRCNNTBLogger(run_dir=self.out_put_dir)

    def train(self):
        """Run the training loop."""
        train_loader = build_train_loader(
            self.dataset_dir,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
        )

        self.model.train()

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(
            params, lr=self.base_lr, momentum=0.9, weight_decay=0.0005
        )
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, self.max_iterations // 3), gamma=0.1
        )

        # Log hyperparameters
        self.tb_logger.log_hparams(
            {
                "dataset_dir": self.dataset_dir,
                "num_classes": self.num_classes,
                "class_names": self.class_names,
                "max_iterations": self.max_iterations,
                "batch_size": self.batch_size,
                "base_lr": self.base_lr,
                "num_workers": self.num_workers,
                "checkpoint_period": self.checkpoint_period,
                "device": self.device,
            }
        )

        iteration = 0
        epoch = 0
        start_time = time.time()

        while iteration < self.max_iterations:
            epoch += 1
            for images, targets in train_loader:
                if iteration >= self.max_iterations:
                    break

                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

                # Log a representative batch of images at iteration 0
                if iteration == 0:
                    self.tb_logger.log_sample_images(images, targets, iteration)

                loss_dict = self.model(images, targets)
                losses = sum(loss_dict.values())

                optimizer.zero_grad()
                losses.backward()
                optimizer.step()

                current_lr = optimizer.param_groups[0]["lr"]
                lr_scheduler.step()

                iteration += 1

                if iteration % 50 == 0:
                    elapsed = time.time() - start_time
                    loss_str = ", ".join(
                        f"{k}: {v.item():.4f}" for k, v in loss_dict.items()
                    )
                    self.logger.info(
                        f"iter {iteration}/{self.max_iterations} "
                        f"({elapsed:.0f}s) — {loss_str} — "
                        f"total: {losses.item():.4f}"
                    )
                    self.tb_logger.log_iter(iteration, loss_dict, current_lr)

                # Also log prediction images
                if self.tb_logger is not None:
                    # Pull a batch from early in the val loader to visualize
                    val_loader = build_val_loader(
                        self.dataset_dir,
                        batch_size=min(4, self.batch_size),
                        num_workers=self.num_workers,
                    )
                    self.model.eval()
                    with torch.no_grad():
                        for val_images, _ in val_loader:
                            val_images = list(img.to(self.device) for img in val_images)
                            preds = self.model(val_images)
                            self.tb_logger.log_prediction_images(
                                val_images, preds, iteration
                            )
                            break  # Only log one batch
                    self.model.train()

                if iteration % self.checkpoint_period == 0:
                    self._save_checkpoint(iteration)
                    self.tb_logger.log_loss_curve(iteration)
                    try:
                        eval_results = self.evaluate_model()
                        if eval_results:
                            self.tb_logger.log_eval(eval_results, iteration)
                    except Exception as exc:
                        self.logger.info(f"Mid-training evaluation skipped: {exc}")

        # Save final checkpoint
        self._save_checkpoint(iteration, final=True)
        self.logger.info(f"Training complete after {iteration} iterations.")

        try:
            eval_results = self.evaluate_model()
            if eval_results:
                self.tb_logger.log_eval(eval_results, iteration)
        except Exception as exc:
            self.logger.info(f"Evaluation skipped: {exc}")

        self.tb_logger.finalize(ok=True)

    def _save_checkpoint(self, iteration: int, final: bool = False) -> str:
        suffix = "final" if final else f"{iteration:07d}"
        path = Path(self.out_put_dir) / f"model_{suffix}.pth"
        torch.save(
            {"model": self.model.state_dict(), "iteration": iteration}, str(path)
        )
        self.logger.info(f"Checkpoint saved: {path}")
        return str(path)

    def evaluate_model(self):
        """Evaluate the model on the validation split using pycocotools."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        val_ann_path = Path(self.dataset_dir) / "valid" / "annotations.json"
        if not val_ann_path.exists():
            self.logger.info("No valid/annotations.json — skipping evaluation.")
            return None

        val_loader = build_val_loader(
            self.dataset_dir, batch_size=1, num_workers=self.num_workers
        )

        self.model.eval()
        coco_gt = COCO(str(val_ann_path))
        results = []

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)

                for target, output in zip(targets, outputs):
                    image_id = target["image_id"]
                    if isinstance(image_id, torch.Tensor):
                        image_id = image_id.item()

                    boxes = output["boxes"].cpu()
                    scores = output["scores"].cpu()
                    labels = output["labels"].cpu()

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        results.append(
                            {
                                "image_id": image_id,
                                "category_id": int(labels[i]),
                                "bbox": [x1, y1, x2 - x1, y2 - y1],
                                "score": float(scores[i]),
                            }
                        )

        if not results:
            self.logger.info("No detections on validation set.")
            return None

        results_file = Path(self.out_put_dir) / "coco_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        coco_dt = coco_gt.loadRes(str(results_file))
        coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        eval_results = {
            "AP": coco_eval.stats[0],
            "AP50": coco_eval.stats[1],
            "AP75": coco_eval.stats[2],
        }
        self.logger.info(f"Evaluation results: {eval_results}")

        out_eval_file = Path(self.out_put_dir) / "evaluation_results.txt"
        with open(out_eval_file, "w") as f:
            f.write(str(eval_results))

        self.model.train()
        return eval_results
