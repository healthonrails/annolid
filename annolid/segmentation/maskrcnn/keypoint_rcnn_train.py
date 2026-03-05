"""
Torchvision-based Keypoint R-CNN training for Annolid.

Uses ``torchvision.models.detection.keypointrcnn_resnet50_fpn`` — the same
backbone family as MaskRCNN but with a dedicated keypoint head.  No external
Detectron2 dependency required.

Design mirrors ``annolid.segmentation.maskrcnn.detectron2_train.Segmentor``
so both trainers share the same GUI entry point (``VisualizationWindow``) and
CLI launcher pattern.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from annolid.segmentation.maskrcnn.coco_dataset import (
    build_keypoint_train_loader,
    build_keypoint_val_loader,
    load_keypoint_meta,
)
from annolid.utils.tb_utils import (
    _draw_loss_curve_image,
    _sanitize_tb_image_tensor,
    _try_create_summary_writer,
    sanitize_tensorboard_tag,
)
from annolid.utils.runs import allocate_run_dir, shared_runs_root


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------


def _get_device() -> str:
    """Return the best available torch device (CUDA > CPU; MPS skipped)."""
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------


def _build_keypoint_model(
    num_classes: int,
    num_keypoints: int,
    weights: Optional[str] = None,
) -> torch.nn.Module:
    """Construct a Keypoint R-CNN model.

    Args:
        num_classes: Number of foreground classes **+ 1** for background.
        num_keypoints: Number of body keypoints per instance.
        weights: Path to a ``.pth`` checkpoint to load, or ``None`` to use
            COCO-pretrained weights from torchvision.
    """
    from torchvision.models.detection import (
        keypointrcnn_resnet50_fpn,
        KeypointRCNN_ResNet50_FPN_Weights,
    )
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

    if weights is None:
        model = keypointrcnn_resnet50_fpn(
            weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1
        )
    else:
        model = keypointrcnn_resnet50_fpn(weights=None)

    # Replace box head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace keypoint head
    in_features_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
        in_features_kp, num_keypoints
    )

    if weights is not None:
        state = torch.load(weights, map_location="cpu", weights_only=True)
        if "model" in state:
            state = state["model"]
        model.load_state_dict(state, strict=False)

    return model


# ---------------------------------------------------------------------------
# Image overlay renderer
# ---------------------------------------------------------------------------


def compute_oks(
    gt_kps: List[float],
    pred_kps: List[float],
    box_area: float,
    sigmas: Optional[List[float]] = None,
) -> float:
    """Compute Object Keypoint Similarity (OKS) between one GT and one Pred.

    Args:
        gt_kps: Flat list [x, y, v, ...] of ground truth keypoints.
        pred_kps: Flat list [x, y, v, ...] of predicted keypoints.
        box_area: Area of the ground truth bounding box.
        sigmas: Optional list of sigmas per keypoint. If None, uses COCO default 0.05.

    Returns:
        float: OKS score [0.0, 1.0]. Returns 0.0 if no visible GT keypoints.
    """
    import math
    import numpy as np

    if not gt_kps or not pred_kps:
        return 0.0

    k = len(gt_kps) // 3
    if sigmas is None:
        sigmas = [0.05] * k

    xg = np.array(gt_kps[0::3])
    yg = np.array(gt_kps[1::3])
    vg = np.array(gt_kps[2::3])

    xp = np.array(pred_kps[0::3])
    yp = np.array(pred_kps[1::3])

    # Only compute for visible (v > 0) GT keypoints
    visible_idx = np.where(vg > 0)[0]
    if len(visible_idx) == 0:
        return 0.0

    oks_sum = 0.0
    for i in visible_idx:
        dx = xg[i] - xp[i]
        dy = yg[i] - yp[i]
        sigma = sigmas[i]
        # OKS exponent: -d^2 / (2 * area * (2 * sigma)^2)
        exponent = -(dx**2 + dy**2) / (2 * box_area * (2 * sigma) ** 2 + 1e-9)
        oks_sum += math.exp(exponent)

    return float(oks_sum / len(visible_idx))


_KP_COLORS = [
    (255, 0, 0),  # red
    (0, 255, 0),  # green
    (0, 0, 255),  # blue
    (255, 165, 0),  # orange
    (148, 0, 211),  # purple
    (0, 206, 209),  # teal
    (255, 20, 147),  # pink
    (0, 128, 0),  # dark green
    (255, 215, 0),  # gold
    (30, 144, 255),  # dodger blue
    (220, 20, 60),  # crimson
    (64, 224, 208),  # turquoise
    (255, 99, 71),  # tomato
    (154, 205, 50),  # yellow green
    (100, 149, 237),  # cornflower
    (255, 69, 0),  # red-orange
    (0, 255, 127),  # spring green
]


def draw_keypoint_overlay(
    img_tensor: "torch.Tensor",
    keypoints: Optional[List[List[float]]],
    boxes: Optional[List[List[float]]],
    keypoint_names: Optional[List[str]] = None,
    skeleton: Optional[List[List[int]]] = None,
    score: float = 1.0,
    alpha_gt: bool = True,
    oks_score: Optional[float] = None,
) -> Any:
    """Draw keypoints, skeleton connections, and bounding boxes on an image.

    Args:
        img_tensor: Float32 CHW tensor in [0, 1] range.
        keypoints: List of [x, y, v] per keypoint (v > 0.5 = visible).
        boxes: List of [x1, y1, x2, y2] bounding boxes.
        keypoint_names: Optional label for each keypoint.
        skeleton: Optional 1-indexed bone pairs [[i, j], …].
        score: Detection confidence score (shown in corner).
        alpha_gt: If True, render GT style (filled circles);
                  else render prediction style (hollow circles).
        oks_score: Optional OKS metric to overlay.

    Returns:
        HWC uint8 numpy array (RGB), suitable for TensorBoard add_image.
    """
    import cv2
    import numpy as np

    # Convert to HWC uint8
    arr = (
        (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    )
    vis = arr.copy()

    # Draw bounding boxes
    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), (200, 200, 0), 2)

            # Draw scores and metrics above box
            label_text = ""
            if score < 1.0:
                label_text += f"{score:.2f}"
            if oks_score is not None:
                label_text += f" | OKS: {oks_score:.2f}"

            if label_text:
                # Add background for text legibility
                (w, h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
                )
                cv2.rectangle(
                    vis,
                    (x1, max(0, y1 - 20)),
                    (x1 + w, max(5, y1 - 5) + 5),
                    (200, 200, 0),
                    -1,
                )
                cv2.putText(
                    vis,
                    label_text.strip(" |"),
                    (x1, max(15, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

    # Draw skeleton connections first (beneath keypoints)
    if keypoints and skeleton:
        for pair in skeleton:
            if len(pair) < 2:
                continue
            i_idx, j_idx = int(pair[0]) - 1, int(pair[1]) - 1  # 1-indexed → 0-indexed
            if (
                i_idx < 0
                or j_idx < 0
                or i_idx >= len(keypoints)
                or j_idx >= len(keypoints)
            ):
                continue
            xi, yi, vi = keypoints[i_idx]
            xj, yj, vj = keypoints[j_idx]
            if vi > 0.5 and vj > 0.5:
                color_i = _KP_COLORS[i_idx % len(_KP_COLORS)]
                cv2.line(vis, (int(xi), int(yi)), (int(xj), int(yj)), color_i, 2)

    # Draw keypoints
    if keypoints:
        for ki, (kx, ky, kv) in enumerate(keypoints):
            if kv <= 0.5:
                continue
            color = _KP_COLORS[ki % len(_KP_COLORS)]
            cx, cy = int(kx), int(ky)
            if alpha_gt:
                cv2.circle(vis, (cx, cy), 5, color, -1)  # filled
            else:
                cv2.circle(vis, (cx, cy), 5, color, 2)  # hollow (prediction)
                cv2.circle(vis, (cx, cy), 2, color, -1)  # inner dot
            name = (
                keypoint_names[ki]
                if keypoint_names and ki < len(keypoint_names)
                else f"kp{ki}"
            )
            cv2.putText(
                vis,
                name,
                (cx + 5, cy - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    return vis  # HWC uint8 RGB


# ---------------------------------------------------------------------------
# TensorBoard logger (mirrors MaskRCNNTBLogger)
# ---------------------------------------------------------------------------


class KeypointRCNNTBLogger:
    """TensorBoard logger for Keypoint R-CNN training."""

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
                "loss_keypoint",
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

    def log_hparams(self, hparams: Dict[str, Any]) -> None:
        w = self.writer
        if w is None:
            return
        lines = ["**Keypoint R-CNN Hyperparameters**", ""]
        for k in sorted(hparams):
            lines.append(f"- `{k}`: {hparams[k]}")
        w.add_text("kprcnn/hparams", "\n".join(lines), 0)
        w.flush()

    def log_iter(self, iteration: int, loss_dict: Dict[str, Any], lr: float) -> None:
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
        w = self.writer
        if w is None:
            return
        for k, v in eval_results.items():
            tag = sanitize_tensorboard_tag(f"metrics/{k}")
            w.add_scalar(tag, float(v), iteration)
        w.flush()

    def log_loss_curve(self, iteration: int) -> None:
        w = self.writer
        if w is None or not self._csv_path.exists():
            return
        try:
            img = _draw_loss_curve_image(
                self._csv_path,
                title="Keypoint R-CNN loss curve",
                step_key="iteration",
                train_key="train_loss",
                val_key="val_loss",
            )
            if img is not None:
                safe = _sanitize_tb_image_tensor(img)
                w.add_image("kprcnn/loss_curve", safe, iteration)
                w.flush()
        except Exception:
            pass

    def finalize(self, *, ok: bool, error: Optional[str] = None) -> None:
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

    def log_gt_examples(
        self,
        images: List[Any],
        targets: List[Dict[str, Any]],
        keypoint_names: Optional[List[str]] = None,
        skeleton: Optional[List[List[int]]] = None,
        max_images: int = 4,
        iteration: int = 0,
    ) -> None:
        """Log a grid of ground-truth keypoint overlays to TensorBoard.

        Called once at the start of training so you can verify that data
        loading + augmentation is working correctly.

        Args:
            images: List of float32 CHW tensors (batch from DataLoader, CPU).
            targets: Corresponding list of target dicts.
            keypoint_names: Ordered keypoint label strings.
            skeleton: 1-indexed bone pairs.
            max_images: Maximum number of images to log.
            iteration: Global step to tag the event with.
        """
        import torch

        w = self.writer
        if w is None:
            return
        try:
            panels: List[Any] = []
            for img, tgt in list(zip(images, targets))[:max_images]:
                img_cpu = img.cpu() if hasattr(img, "cpu") else img
                kps_tensor = tgt.get("keypoints")  # [N, K, 3] or None
                boxes_tensor = tgt.get("boxes")  # [N, 4] or None
                kps = (
                    kps_tensor[0].tolist()
                    if (kps_tensor is not None and len(kps_tensor) > 0)
                    else None
                )
                boxes = (
                    [boxes_tensor[0].tolist()]
                    if (boxes_tensor is not None and len(boxes_tensor) > 0)
                    else None
                )
                vis = draw_keypoint_overlay(
                    img_cpu,
                    kps,
                    boxes,
                    keypoint_names=keypoint_names,
                    skeleton=skeleton,
                    alpha_gt=True,
                )
                panels.append(torch.from_numpy(vis).permute(2, 0, 1).float() / 255.0)

            if panels:
                # Make all images the same size (pad to max H, W)
                max_h = max(p.shape[1] for p in panels)
                max_w = max(p.shape[2] for p in panels)
                padded = []
                for p in panels:
                    ph, pw = max_h - p.shape[1], max_w - p.shape[2]
                    import torch.nn.functional as F

                    padded.append(F.pad(p, (0, pw, 0, ph)))
                grid = torch.stack(padded)  # [B, C, H, W]
                w.add_images("examples/gt_keypoints", grid, iteration)
                w.flush()
        except Exception as exc:
            logger.debug("log_gt_examples failed: %s", exc)

    def log_pred_examples(
        self,
        images: List[Any],
        gt_targets: List[Dict[str, Any]],
        pred_outputs: List[Dict[str, Any]],
        keypoint_names: Optional[List[str]] = None,
        skeleton: Optional[List[List[int]]] = None,
        score_threshold: float = 0.3,
        max_images: int = 4,
        iteration: int = 0,
    ) -> None:
        """Log side-by-side GT vs prediction overlays to TensorBoard.

        Renders a ``[GT | Pred]`` panel for each image so you can visually
        inspect how well the model is localising keypoints compared to ground
        truth at each checkpoint.

        Args:
            images: List of float32 CHW tensors (CPU).
            gt_targets: Ground-truth targets from the DataLoader.
            pred_outputs: Raw model outputs (model.eval() forward pass).
            keypoint_names: Ordered keypoint label strings.
            skeleton: 1-indexed bone pairs.
            score_threshold: Minimum detection score to display.
            max_images: Maximum number of panels to log.
            iteration: Global step.
        """
        import numpy as np
        import torch
        import torch.nn.functional as F

        w = self.writer
        if w is None:
            return
        try:
            panels: List[Any] = []
            for img, tgt, out in list(zip(images, gt_targets, pred_outputs))[
                :max_images
            ]:
                img_cpu = img.cpu() if hasattr(img, "cpu") else img

                # --- GT panel ---
                kps_tensor = tgt.get("keypoints")
                boxes_tensor = tgt.get("boxes")
                gt_kps = (
                    kps_tensor[0].cpu().tolist()
                    if (kps_tensor is not None and len(kps_tensor) > 0)
                    else None
                )
                gt_boxes = (
                    [boxes_tensor[0].cpu().tolist()]
                    if (boxes_tensor is not None and len(boxes_tensor) > 0)
                    else None
                )
                gt_vis = draw_keypoint_overlay(
                    img_cpu,
                    gt_kps,
                    gt_boxes,
                    keypoint_names=keypoint_names,
                    skeleton=skeleton,
                    alpha_gt=True,
                )

                # --- Prediction panel ---
                pred_boxes = out["boxes"].cpu()
                pred_scores = out["scores"].cpu()
                pred_kps_t = out.get("keypoints")
                pred_kps_list: List[List[float]] = []
                pred_box_list: List[List[float]] = []
                pred_score_float = 0.0
                oks_val = None

                for i, score in enumerate(pred_scores):
                    if float(score) < score_threshold:
                        continue
                    pred_score_float = float(score)
                    if pred_kps_t is not None:
                        pred_kps_list = pred_kps_t[i].cpu().tolist()
                    pred_box_list = [pred_boxes[i].tolist()]

                    # Compute OKS for this top detection
                    if gt_boxes and gt_kps and pred_kps_list:
                        # Convert model [K, 3] to flat list [x, y, v, ...] for compute_oks
                        flat_pred = [val for pt in pred_kps_list for val in pt]
                        # Area = w * h
                        gx1, gy1, gx2, gy2 = gt_boxes[0]
                        area = (gx2 - gx1) * (gy2 - gy1)
                        if area > 0:
                            oks_val = compute_oks(gt_kps, flat_pred, area)
                    break  # show top-1 detection

                pred_vis = draw_keypoint_overlay(
                    img_cpu,
                    pred_kps_list or None,
                    pred_box_list or None,
                    keypoint_names=keypoint_names,
                    skeleton=skeleton,
                    score=pred_score_float,
                    alpha_gt=False,
                    oks_score=oks_val,
                )

                # --- Side-by-side: put divider line between GT and Pred ---
                divider = np.full((gt_vis.shape[0], 4, 3), 128, dtype=np.uint8)
                panel_np = np.concatenate(
                    [gt_vis, divider, pred_vis], axis=1
                )  # HW(2W+4)3
                panel_t = torch.from_numpy(panel_np).permute(2, 0, 1).float() / 255.0
                panels.append(panel_t)

            if panels:
                max_h = max(p.shape[1] for p in panels)
                max_w = max(p.shape[2] for p in panels)
                padded = []
                for p in panels:
                    ph, pw = max_h - p.shape[1], max_w - p.shape[2]
                    padded.append(F.pad(p, (0, pw, 0, ph)))
                grid = torch.stack(padded)  # [B, C, H, W]
                w.add_images("examples/pred_vs_gt", grid, iteration)
                w.flush()
        except Exception as exc:
            logger.debug("log_pred_examples failed: %s", exc)


# ---------------------------------------------------------------------------
# Keypoint Segmentor
# ---------------------------------------------------------------------------


class KeypointSegmentor:
    """Train and evaluate a torchvision Keypoint R-CNN model.

    Expects a COCO keypoints dataset at ``dataset_dir/train/annotations.json``
    with ``categories[*].keypoints`` defined.
    """

    def __init__(
        self,
        dataset_dir: Optional[str] = None,
        out_put_dir: Optional[str] = None,
        score_threshold: float = 0.15,
        max_iterations: int = 3000,
        batch_size: int = 2,
        model_pth_path: Optional[str] = None,
        base_lr: float = 0.0025,
        num_workers: int = 0,
        checkpoint_period: int = 1000,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.max_iterations = max_iterations
        self.checkpoint_period = checkpoint_period
        self.score_threshold = score_threshold
        self.base_lr = base_lr
        self.num_workers = num_workers

        self.logger = logging.getLogger(__name__)

        if out_put_dir is None:
            self.out_put_dir = str(
                allocate_run_dir(
                    task="keypoint_rcnn",
                    model="train",
                    runs_root=shared_runs_root(),
                )
            )
        else:
            self.out_put_dir = out_put_dir

        os.makedirs(self.out_put_dir, exist_ok=True)

        # Load keypoint metadata from the training annotations.
        train_ann = Path(self.dataset_dir) / "train" / "annotations.json"
        self.kp_meta = load_keypoint_meta(str(train_ann))
        self.num_keypoints: int = self.kp_meta["num_keypoints"]
        self.keypoint_names: List[str] = self.kp_meta["keypoint_names"]

        if self.num_keypoints == 0:
            raise ValueError(
                "No keypoints found in the training annotations. "
                "Make sure categories[*].keypoints is populated in annotations.json."
            )

        # +1 for background class (single foreground class for keypoint detection)
        self.num_classes = 2

        self.logger.info(f"Keypoints ({self.num_keypoints}): {self.keypoint_names}")
        self.logger.info(f"Max iterations: {max_iterations}")
        self.logger.info(f"Batch size: {batch_size}")

        self.device = _get_device()
        self.model = _build_keypoint_model(
            self.num_classes, self.num_keypoints, weights=model_pth_path
        )
        self.model.to(self.device)

        self.tb_logger = KeypointRCNNTBLogger(run_dir=self.out_put_dir)

    def train(self) -> None:
        """Run the Keypoint R-CNN training loop."""
        train_loader = build_keypoint_train_loader(
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
            optimizer,
            step_size=max(1, self.max_iterations // 3),
            gamma=0.1,
        )

        self.tb_logger.log_hparams(
            {
                "dataset_dir": self.dataset_dir,
                "num_keypoints": self.num_keypoints,
                "keypoint_names": self.keypoint_names,
                "max_iterations": self.max_iterations,
                "batch_size": self.batch_size,
                "base_lr": self.base_lr,
                "num_workers": self.num_workers,
                "device": self.device,
            }
        )

        # Log a few ground-truth examples from the first training batch so the
        # user can verify that data loading and augmentation are correct.
        try:
            first_images, first_targets = next(iter(train_loader))
            self.tb_logger.log_gt_examples(
                list(first_images),
                list(first_targets),
                keypoint_names=self.keypoint_names,
                skeleton=self.kp_meta.get("skeleton"),
                iteration=0,
            )
        except Exception as exc:
            self.logger.debug("Could not log GT examples: %s", exc)

        iteration = 0
        start_time = time.time()

        while iteration < self.max_iterations:
            for images, targets in train_loader:
                if iteration >= self.max_iterations:
                    break

                images = [img.to(self.device) for img in images]
                targets = [
                    {k: v.to(self.device) for k, v in t.items()} for t in targets
                ]

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

                if iteration % self.checkpoint_period == 0:
                    self._save_checkpoint(iteration)
                    self.tb_logger.log_loss_curve(iteration)
                    try:
                        self._eval_iteration = iteration
                        eval_results = self.evaluate_model()
                        if eval_results:
                            self.tb_logger.log_eval(eval_results, iteration)
                    except Exception as exc:
                        self.logger.info(f"Mid-training evaluation skipped: {exc}")

        self._save_checkpoint(iteration, final=True)
        self.logger.info(
            f"Keypoint R-CNN training complete after {iteration} iterations."
        )

        try:
            self._eval_iteration = iteration
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
            {
                "model": self.model.state_dict(),
                "iteration": iteration,
                "num_keypoints": self.num_keypoints,
                "keypoint_names": self.keypoint_names,
                "num_classes": self.num_classes,
            },
            str(path),
        )
        self.logger.info(f"Checkpoint saved: {path}")
        return str(path)

    def evaluate_model(self) -> Optional[Dict[str, float]]:
        """Evaluate the model on the validation split using OKS (COCO keypoint AP)."""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        val_ann_path = Path(self.dataset_dir) / "valid" / "annotations.json"
        if not val_ann_path.exists():
            self.logger.info("No valid/annotations.json — skipping evaluation.")
            return None

        val_loader = build_keypoint_val_loader(
            self.dataset_dir,
            batch_size=1,
            num_workers=self.num_workers,
        )

        self.model.eval()
        coco_gt = COCO(str(val_ann_path))
        results: List[Dict[str, Any]] = []

        # Collect images+output for overlay logging (keep first few).
        overlay_images: List[Any] = []
        overlay_targets: List[Dict[str, Any]] = []
        overlay_outputs: List[Dict[str, Any]] = []
        max_overlay = 4

        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                outputs = self.model(images)

                # Collect images for overlays (stay on CPU).
                if len(overlay_images) < max_overlay:
                    for img_gpu, tgt, out in zip(images, targets, outputs):
                        if len(overlay_images) >= max_overlay:
                            break
                        overlay_images.append(img_gpu.cpu())
                        overlay_targets.append(
                            {
                                k: v.cpu() if hasattr(v, "cpu") else v
                                for k, v in tgt.items()
                            }
                        )
                        overlay_outputs.append(
                            {
                                k: v.cpu() if hasattr(v, "cpu") else v
                                for k, v in out.items()
                            }
                        )

                for target, output in zip(targets, outputs):
                    image_id = target["image_id"]
                    if isinstance(image_id, torch.Tensor):
                        image_id = image_id.item()

                    boxes = output["boxes"].cpu()
                    scores = output["scores"].cpu()
                    labels = output["labels"].cpu()
                    keypoints = output.get("keypoints")  # [N, K, 3]

                    for i in range(len(boxes)):
                        x1, y1, x2, y2 = boxes[i].tolist()
                        kp_entry: Optional[List[float]] = None
                        if keypoints is not None:
                            kp_i = keypoints[i].cpu().tolist()  # [K, 3]
                            kp_entry = [
                                coord for kp in kp_i for coord in kp
                            ]  # flat [x,y,v, x,y,v, …]

                        record: Dict[str, Any] = {
                            "image_id": image_id,
                            "category_id": int(labels[i]),
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(scores[i]),
                        }
                        if kp_entry is not None:
                            record["keypoints"] = kp_entry
                        results.append(record)

        if not results:
            self.logger.info("No detections on validation set.")
            return None

        # Log prediction overlays before COCO eval.
        if overlay_images and hasattr(self, "_eval_iteration"):
            try:
                self.tb_logger.log_pred_examples(
                    overlay_images,
                    overlay_targets,
                    overlay_outputs,
                    keypoint_names=self.keypoint_names,
                    skeleton=self.kp_meta.get("skeleton"),
                    score_threshold=self.score_threshold,
                    iteration=self._eval_iteration,
                )
            except Exception as exc:
                self.logger.debug("log_pred_examples failed: %s", exc)

        results_file = Path(self.out_put_dir) / "coco_kp_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f)

        try:
            coco_dt = coco_gt.loadRes(str(results_file))
            coco_eval = COCOeval(coco_gt, coco_dt, "keypoints")
            coco_eval.evaluate()
            coco_eval.accumulate()
            coco_eval.summarize()

            eval_results = {
                "AP_kp": coco_eval.stats[0],
                "AP_kp_50": coco_eval.stats[1],
                "AP_kp_75": coco_eval.stats[2],
            }
            self.logger.info(f"Keypoint evaluation results: {eval_results}")
            out_eval_file = Path(self.out_put_dir) / "evaluation_results.txt"
            with open(out_eval_file, "w") as f:
                f.write(str(eval_results))
            return eval_results
        except Exception as exc:
            self.logger.warning(f"OKS evaluation failed: {exc}")
            return None
        finally:
            self.model.train()
