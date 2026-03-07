from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from annolid.engine.registry import ModelPluginBase, register_model


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KeypointRCNNTrainSettings:
    dataset_dir: str
    output_dir: Optional[str]
    max_iterations: int
    batch_size: int
    weights: Optional[str]
    score_threshold: float
    base_lr: float
    num_workers: int
    checkpoint_period: int


def _resolve_train_settings(args: argparse.Namespace) -> KeypointRCNNTrainSettings:
    dataset_dir = getattr(args, "dataset_dir", None)
    if not dataset_dir:
        raise ValueError("dataset_dir is required. Provide --dataset-dir.")
    return KeypointRCNNTrainSettings(
        dataset_dir=str(Path(dataset_dir).expanduser().resolve()),
        output_dir=(
            str(Path(args.output_dir).expanduser().resolve())
            if getattr(args, "output_dir", None)
            else None
        ),
        max_iterations=int(getattr(args, "max_iterations", None) or 3000),
        batch_size=int(getattr(args, "batch_size", None) or 2),
        weights=getattr(args, "weights", None),
        score_threshold=float(getattr(args, "score_threshold", None) or 0.15),
        base_lr=float(getattr(args, "base_lr", None) or 0.0025),
        num_workers=int(getattr(args, "num_workers", None) or 0),
        checkpoint_period=int(getattr(args, "checkpoint_period", None) or 1000),
    )


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


@register_model
class KeypointRCNNPlugin(ModelPluginBase):
    """Torchvision Keypoint R-CNN train/predict plugin.

    Uses ``torchvision.models.detection.keypointrcnn_resnet50_fpn`` — the
    same torchvision-only approach as ``maskrcnn_detectron2``, following
    Detectron2 COCO-Keypoints conventions without any Detectron2 dependency.

    Train input: COCO keypoints dataset with::

        categories[*].keypoints  → list of body-part names
        categories[*].skeleton   → list of [i, j] bone pairs (1-indexed)
        annotations[*].keypoints → flat [x, y, v, …] per instance

    Checkpoints embed ``keypoint_names`` so the predictor can reconstruct
    body-landmark names without the original dataset.
    """

    name = "keypoint_rcnn"
    description = (
        "Torchvision Keypoint R-CNN (ResNet-50 FPN) — animal body keypoint detection. "
        "No Detectron2 required."
    )
    train_help_sections = (
        (
            "Required inputs",
            ("--dataset-dir",),
        ),
        (
            "Outputs and run location",
            (
                "--output-dir",
                "--weights",
            ),
        ),
        (
            "Training controls",
            (
                "--max-iterations",
                "--batch-size",
                "--base-lr",
                "--checkpoint-period",
                "--num-workers",
                "--score-threshold",
            ),
        ),
    )
    predict_help_sections = (
        (
            "Required inputs",
            (
                "--weights",
                "--image",
            ),
        ),
        (
            "Outputs and display",
            (
                "--output-json",
                "--no-display",
            ),
        ),
        (
            "Inference controls",
            ("--score-threshold",),
        ),
    )

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #

    def add_train_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--dataset-dir",
            default=None,
            help=(
                "Dataset directory containing train/ and valid/ sub-folders, "
                "each with annotations.json (COCO keypoints format)."
            ),
        )
        parser.add_argument(
            "--output-dir", default=None, help="Training outputs directory."
        )
        parser.add_argument("--max-iterations", type=int, default=None)
        parser.add_argument("--batch-size", type=int, default=None)
        parser.add_argument(
            "--weights",
            default=None,
            help="Optional initial weights checkpoint (.pth).",
        )
        parser.add_argument("--score-threshold", type=float, default=None)
        parser.add_argument("--base-lr", type=float, default=None)
        parser.add_argument("--num-workers", type=int, default=None)
        parser.add_argument("--checkpoint-period", type=int, default=None)

    def train(self, args: argparse.Namespace) -> int:
        from annolid.segmentation.maskrcnn.keypoint_rcnn_train import KeypointSegmentor

        settings = _resolve_train_settings(args)
        seg = KeypointSegmentor(
            dataset_dir=settings.dataset_dir,
            out_put_dir=settings.output_dir,
            score_threshold=settings.score_threshold,
            max_iterations=settings.max_iterations,
            batch_size=settings.batch_size,
            model_pth_path=settings.weights,
            base_lr=settings.base_lr,
            num_workers=settings.num_workers,
            checkpoint_period=settings.checkpoint_period,
        )
        seg.train()
        print(str(seg.out_put_dir))
        return 0

    # ------------------------------------------------------------------ #
    # Predict
    # ------------------------------------------------------------------ #

    def add_predict_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--weights", required=True, help="Trained model checkpoint (.pth)."
        )
        parser.add_argument(
            "--image", required=True, help="Image path to run inference on."
        )
        parser.add_argument("--score-threshold", type=float, default=0.5)
        parser.add_argument(
            "--output-json",
            default=None,
            help=(
                "Path to write Labelme-format JSON with predicted keypoints. "
                "Defaults to <image_basename>_keypoints.json alongside the image."
            ),
        )
        parser.add_argument(
            "--no-display", action="store_true", help="Do not open an OpenCV window."
        )

    def predict(self, args: argparse.Namespace) -> int:
        import cv2
        import torch
        import json as _json

        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

        weights_path = str(Path(args.weights).expanduser().resolve())
        image_path = str(Path(args.image).expanduser().resolve())
        score_thr = float(args.score_threshold)

        # Load checkpoint
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        num_keypoints: int = state.get("num_keypoints", 17)
        keypoint_names = state.get(
            "keypoint_names", [f"kp{i}" for i in range(num_keypoints)]
        )
        num_classes: int = state.get("num_classes", 2)

        # Rebuild model
        model = keypointrcnn_resnet50_fpn(weights=None)
        in_feat = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
        in_feat_kp = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
        model.roi_heads.keypoint_predictor = KeypointRCNNPredictor(
            in_feat_kp, num_keypoints
        )
        model_state = state.get("model", state)
        model.load_state_dict(model_state, strict=False)
        model.eval()

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)

        # Load image
        bgr = cv2.imread(image_path)
        if bgr is None:
            print(f"[keypoint_rcnn] Cannot read image: {image_path}")
            return 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.as_tensor(
            rgb.astype("float32").transpose(2, 0, 1) / 255.0
        ).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])

        output = outputs[0]
        boxes = output["boxes"].cpu().numpy()
        scores = output["scores"].cpu().numpy()
        keypoints = output.get("keypoints")  # [N, K, 3]

        # Build Labelme-format shapes for visualization / export
        h, w = bgr.shape[:2]
        shapes = []
        for i, (box, score) in enumerate(zip(boxes, scores)):
            if score < score_thr:
                continue
            if keypoints is not None:
                kps_i = keypoints[i].cpu().numpy()  # [K, 3]
                for ki, (kx, ky, kv) in enumerate(kps_i):
                    if kv < 0.5:
                        continue
                    name = keypoint_names[ki] if ki < len(keypoint_names) else f"kp{ki}"
                    shapes.append(
                        {
                            "label": name,
                            "points": [[float(kx), float(ky)]],
                            "group_id": int(i),
                            "shape_type": "point",
                            "flags": {"score": float(score)},
                        }
                    )
                # Bounding box
                x1, y1, x2, y2 = box.tolist()
                shapes.append(
                    {
                        "label": "animal",
                        "points": [[x1, y1], [x2, y2]],
                        "group_id": int(i),
                        "shape_type": "rectangle",
                        "flags": {"score": float(score)},
                    }
                )

        labelme_data = {
            "version": "5.0",
            "flags": {},
            "shapes": shapes,
            "imagePath": Path(image_path).name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }

        out_json = getattr(args, "output_json", None)
        if not out_json:
            out_json = str(
                Path(image_path).with_suffix("").parent
                / (Path(image_path).stem + "_keypoints.json")
            )
        with open(out_json, "w", encoding="utf-8") as fh:
            _json.dump(labelme_data, fh, indent=2)
        print(f"[keypoint_rcnn] Keypoints saved to: {out_json}")

        # Optional OpenCV visualization
        if not getattr(args, "no_display", False):
            vis = bgr.copy()
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for i, (box, score) in enumerate(zip(boxes, scores)):
                if score < score_thr:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 200), 2)
                if keypoints is not None:
                    kps_i = keypoints[i].cpu().numpy()
                    for ki, (kx, ky, kv) in enumerate(kps_i):
                        if kv < 0.5:
                            continue
                        color = colors[ki % len(colors)]
                        cv2.circle(vis, (int(kx), int(ky)), 4, color, -1)
                        name = (
                            keypoint_names[ki]
                            if ki < len(keypoint_names)
                            else f"kp{ki}"
                        )
                        cv2.putText(
                            vis,
                            name,
                            (int(kx) + 4, int(ky) - 4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            color,
                            1,
                        )
            cv2.imshow("Keypoint R-CNN", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        return 0
