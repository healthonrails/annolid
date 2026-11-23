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
    required_capabilities = ("ml",)
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
                "--source",
            ),
        ),
        (
            "Outputs and display",
            (
                "--output-json",
                "--output-dir",
                "--save-video",
                "--no-display",
            ),
        ),
        (
            "Inference controls",
            (
                "--score-threshold",
                "--batch-size",
                "--stride",
                "--max-frames",
                "--device",
            ),
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
            "--image",
            default=None,
            help="Single image path to run inference on (or use --source for video/folder).",
        )
        parser.add_argument(
            "--source",
            default=None,
            help="Video file, image file, or directory of images to run batched Keypoint R-CNN inference on.",
        )
        parser.add_argument("--score-threshold", type=float, default=0.5)
        parser.add_argument(
            "--batch-size",
            type=int,
            default=4,
            help="Batch size for video or folder inference (default: 4).",
        )
        parser.add_argument(
            "--stride",
            type=int,
            default=1,
            help="Process every N-th frame for video inference (default: 1).",
        )
        parser.add_argument(
            "--max-frames",
            type=int,
            default=None,
            help="Optional maximum number of frames to process for video/folder inference.",
        )
        parser.add_argument(
            "--device",
            default=None,
            help="Optional torch device override (cuda, mps, cpu).",
        )
        parser.add_argument(
            "--output-json",
            default=None,
            help=(
                "Path to write LabelMe-format JSON with predicted keypoints (single image mode). "
                "Defaults to <image_basename>_keypoints.json alongside the image."
            ),
        )
        parser.add_argument(
            "--output-dir",
            default=None,
            help="Output directory for per-frame LabelMe JSONs + pose_schema.json in video/folder mode.",
        )
        parser.add_argument(
            "--save-video",
            default=None,
            help="Optional labeled skeleton video output path (.mp4) for video inference.",
        )
        parser.add_argument(
            "--no-display", action="store_true", help="Do not open an OpenCV window."
        )

    def predict(self, args: argparse.Namespace) -> int:
        import json as _json
        import cv2
        import torch
        from torchvision.models.detection import keypointrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.keypoint_rcnn import KeypointRCNNPredictor

        source_raw = getattr(args, "source", None) or getattr(args, "image", None)
        if not source_raw:
            raise ValueError(
                "Provide either --image <path> or --source <video|folder|image>."
            )

        weights_path = str(Path(args.weights).expanduser().resolve())
        source_path = Path(source_raw).expanduser().resolve()
        score_thr = float(args.score_threshold)

        # Load checkpoint
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
        num_keypoints: int = int(state.get("num_keypoints", 17))
        keypoint_names = list(
            state.get("keypoint_names", [f"kp{i}" for i in range(num_keypoints)])
        )
        num_classes: int = int(state.get("num_classes", 2))

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

        req_dev = getattr(args, "device", None)
        device = (
            str(req_dev)
            if req_dev
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        model.to(device)

        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg"}
        is_video_or_dir = (
            source_path.is_dir() or source_path.suffix.lower() in video_exts
        )

        if is_video_or_dir:
            summary = predict_keypoint_rcnn_source(
                model=model,
                source=source_path,
                keypoint_names=keypoint_names,
                output_dir=getattr(args, "output_dir", None),
                score_threshold=score_thr,
                batch_size=max(1, int(getattr(args, "batch_size", 4) or 4)),
                stride=max(1, int(getattr(args, "stride", 1) or 1)),
                max_frames=getattr(args, "max_frames", None),
                save_video=getattr(args, "save_video", None),
                device=device,
            )
            print(_json.dumps(summary, indent=2))
            return 0

        # Single-image mode (backward compatible)
        bgr = cv2.imread(str(source_path))
        if bgr is None:
            print(f"[keypoint_rcnn] Cannot read image: {source_path}")
            return 1
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_tensor = torch.as_tensor(
            rgb.astype("float32").transpose(2, 0, 1) / 255.0
        ).to(device)

        with torch.no_grad():
            outputs = model([img_tensor])

        output = outputs[0]
        h, w = bgr.shape[:2]
        shapes = _keypoint_rcnn_output_to_labelme_shapes(
            output,
            keypoint_names=keypoint_names,
            score_threshold=score_thr,
        )

        labelme_data = {
            "version": "5.1.1",
            "flags": {},
            "shapes": shapes,
            "imagePath": source_path.name,
            "imageData": None,
            "imageHeight": h,
            "imageWidth": w,
        }

        out_json = getattr(args, "output_json", None)
        if not out_json:
            out_json = str(source_path.parent / (source_path.stem + "_keypoints.json"))
        with open(out_json, "w", encoding="utf-8") as fh:
            _json.dump(labelme_data, fh, indent=2)
        print(f"[keypoint_rcnn] Keypoints saved to: {out_json}")

        if not getattr(args, "no_display", False):
            boxes = output["boxes"].detach().cpu().numpy()
            scores = output["scores"].detach().cpu().numpy()
            keypoints = output.get("keypoints")
            vis = bgr.copy()
            colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
            for i, (box, score) in enumerate(zip(boxes, scores)):
                if score < score_thr:
                    continue
                x1, y1, x2, y2 = box.astype(int)
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 200, 200), 2)
                if keypoints is not None:
                    kps_i = keypoints[i].detach().cpu().numpy()
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


def _keypoint_rcnn_output_to_labelme_shapes(
    output: dict,
    *,
    keypoint_names: list[str],
    score_threshold: float,
) -> list[dict]:
    """Convert one torchvision Keypoint R-CNN output dict to LabelMe-safe shapes."""
    from annolid.annotation.keypoint_visibility import KeypointVisibility

    boxes = output["boxes"].detach().cpu().numpy()
    scores = output["scores"].detach().cpu().numpy()
    keypoints = output.get("keypoints")
    kp_scores = output.get("keypoints_scores")
    shapes: list[dict] = []

    for i, (box, score) in enumerate(zip(boxes, scores)):
        inst_score = float(score)
        if inst_score < float(score_threshold):
            continue
        inst_label = f"animal_{i}"
        if keypoints is not None:
            kps_i = keypoints[i].detach().cpu().numpy()
            kps_sc_i = (
                kp_scores[i].detach().cpu().numpy() if kp_scores is not None else None
            )
            for ki, (kx, ky, kv) in enumerate(kps_i):
                if float(kv) < 0.5:
                    continue
                name = keypoint_names[ki] if ki < len(keypoint_names) else f"kp{ki}"
                pt_score = inst_score
                if kps_sc_i is not None and ki < len(kps_sc_i):
                    # Convert raw logit to probability if outside [0, 1].
                    raw_s = float(kps_sc_i[ki])
                    pt_score = (
                        raw_s
                        if 0.0 <= raw_s <= 1.0
                        else float(
                            1.0
                            / (1.0 + pow(2.718281828, -max(-20.0, min(20.0, raw_s))))
                        )
                    )
                shapes.append(
                    {
                        "label": name,
                        "points": [[float(kx), float(ky)]],
                        "group_id": int(i),
                        "shape_type": "point",
                        "flags": {"kp_visible": True, "predicted": True},
                        "kp_visibility": int(KeypointVisibility.VISIBLE),
                        "instance_label": inst_label,
                        "display_label": name,
                        "score": round(pt_score, 6),
                    }
                )
        x1, y1, x2, y2 = [float(v) for v in box.tolist()]
        shapes.append(
            {
                "label": inst_label,
                "points": [[x1, y1], [x2, y2]],
                "group_id": int(i),
                "shape_type": "rectangle",
                "flags": {"predicted": True},
                "instance_id": int(i),
                "score": round(inst_score, 6),
            }
        )
    return shapes


def predict_keypoint_rcnn_source(
    *,
    model,
    source: str | Path,
    keypoint_names: list[str],
    output_dir: Optional[str | Path] = None,
    score_threshold: float = 0.5,
    batch_size: int = 4,
    stride: int = 1,
    max_frames: Optional[int] = None,
    save_video: Optional[str | Path] = None,
    device: str = "cpu",
) -> dict:
    """Run batched Keypoint R-CNN inference on a video or image directory, writing LabelMe JSONs."""
    import cv2
    import torch

    from annolid.annotation.pose_schema import PoseSchema
    from annolid.postprocessing.pose_video_renderer import (
        PoseVideoRenderConfig,
        render_pose_video,
    )
    from annolid.utils.atomic_io import atomic_write_json

    src = Path(source).expanduser().resolve()
    out_root = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else (
            src.parent / f"{src.stem}_keypoint_rcnn"
            if src.is_file()
            else src / "keypoint_rcnn_preds"
        )
    )
    out_root.mkdir(parents=True, exist_ok=True)

    schema = PoseSchema(
        keypoints=list(keypoint_names),
        edges=PoseSchema.infer_edges(keypoint_names, preset="auto"),
        symmetry_pairs=PoseSchema.infer_symmetry_pairs(keypoint_names),
    )
    schema_path = schema.save(out_root / "pose_schema.json")

    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    frames_processed = 0
    total_instances = 0
    eff_batch = max(1, int(batch_size))
    eff_stride = max(1, int(stride))

    def _run_batch(batch_items: list[tuple[int, str, object]]) -> None:
        nonlocal frames_processed, total_instances
        if not batch_items:
            return
        tensors = []
        for _fidx, _img_name, bgr in batch_items:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            tensors.append(
                torch.as_tensor(rgb.astype("float32").transpose(2, 0, 1) / 255.0).to(
                    device
                )
            )
        with torch.no_grad():
            outputs = model(tensors)
        for (fidx, img_name, bgr), out in zip(batch_items, outputs):
            h, w = bgr.shape[:2]
            shapes = _keypoint_rcnn_output_to_labelme_shapes(
                out,
                keypoint_names=keypoint_names,
                score_threshold=score_threshold,
            )
            inst_ids = {
                s.get("group_id") for s in shapes if s.get("shape_type") == "point"
            }
            total_instances += len(inst_ids)
            payload = {
                "version": "5.1.1",
                "flags": {},
                "shapes": shapes,
                "imagePath": img_name,
                "imageData": None,
                "imageHeight": int(h),
                "imageWidth": int(w),
            }
            json_name = (
                f"{Path(img_name).stem}.json"
                if src.is_dir()
                else f"{src.stem}_{int(fidx):09d}.json"
            )
            atomic_write_json(out_root / json_name, payload, indent=2)
            frames_processed += 1

    if src.is_dir():
        img_files = sorted(
            p for p in src.iterdir() if p.is_file() and p.suffix.lower() in image_exts
        )
        batch: list[tuple[int, str, object]] = []
        for idx, img_p in enumerate(img_files):
            if idx % eff_stride != 0:
                continue
            if max_frames is not None and (frames_processed + len(batch)) >= int(
                max_frames
            ):
                break
            bgr = cv2.imread(str(img_p))
            if bgr is None:
                continue
            batch.append((idx, img_p.name, bgr))
            if len(batch) >= eff_batch:
                _run_batch(batch)
                batch = []
        _run_batch(batch)
    else:
        cap = cv2.VideoCapture(str(src))
        if not cap.isOpened():
            raise RuntimeError(f"Unable to open video: {src}")
        try:
            batch = []
            fidx = 0
            while True:
                if max_frames is not None and (frames_processed + len(batch)) >= int(
                    max_frames
                ):
                    break
                ret, bgr = cap.read()
                if not ret or bgr is None:
                    break
                if fidx % eff_stride == 0:
                    batch.append((fidx, f"{src.stem}_{fidx:09d}.png", bgr))
                    if len(batch) >= eff_batch:
                        _run_batch(batch)
                        batch = []
                fidx += 1
            _run_batch(batch)
        finally:
            cap.release()

    rendered_video: Optional[str] = None
    if save_video and src.is_file():
        res = render_pose_video(
            src,
            out_root,
            save_video,
            pose_schema=schema_path,
            config=PoseVideoRenderConfig(color_mode="instance"),
        )
        rendered_video = res.output_video_path

    return {
        "source": str(src),
        "output_dir": str(out_root),
        "pose_schema": str(schema_path),
        "frames_processed": frames_processed,
        "instances_detected": total_instances,
        "labeled_video": rendered_video,
    }
