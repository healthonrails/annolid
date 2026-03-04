"""
Torchvision-based Mask R-CNN inference / tracking for Annolid.

No detectron2 dependency.  Uses the same torchvision model as
``detectron2_train.py`` and the standalone :class:`BBoxIOUTracker`.
"""

from __future__ import annotations

import glob
import queue
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch

import pycocotools.mask as mask_util
from torchvision.ops import nms

from annolid.annotation.keypoints import save_labels
from annolid.annotation.masks import mask_iou
from annolid.data import videos
from annolid.data.videos import key_frames
from annolid.postprocessing.quality_control import TracksResults, pred_dict_to_labelme
from annolid.segmentation.maskrcnn.coco_dataset import load_class_names
from annolid.tracker.simple_instances import SimpleInstances


def _get_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_model(num_classes: int, weights_path: str, device: str):
    """Load a trained torchvision Mask R-CNN model for inference."""
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

    model = maskrcnn_resnet50_fpn_v2(weights=None)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes
    )

    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if "model" in state:
        state = state["model"]
    model.load_state_dict(state, strict=False)
    model.to(device)
    model.eval()
    return model


def _to_instances(output: dict, image_size: tuple) -> SimpleInstances:
    """Convert a torchvision model output dict to SimpleInstances."""
    inst = SimpleInstances(image_size=image_size)
    inst.pred_boxes = output["boxes"]
    inst.pred_classes = output["labels"]
    inst.scores = output["scores"]
    if "masks" in output:
        # torchvision masks are [N, 1, H, W] probabilities → [N, H, W] binary
        inst.pred_masks = (output["masks"].squeeze(1) > 0.5).byte()
    return inst


class Segmentor:
    """Instance segmentation / tracking using torchvision Mask R-CNN.

    Drop-in replacement for the former detectron2-based ``Segmentor``.
    """

    def __init__(
        self,
        dataset_dir=None,
        model_pth_path=None,
        score_threshold=0.15,
        overlap_threshold=0.95,
        model_config=None,
        num_instances_per_class=1,
    ) -> None:
        self.dataset_dir = dataset_dir
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold

        # Load class names from the COCO annotations
        train_ann = Path(self.dataset_dir) / "train" / "annotations.json"
        self.class_names = load_class_names(str(train_ann))
        self.num_classes = len(self.class_names) + 1  # +1 for background

        self.subject_queue: queue.PriorityQueue = queue.PriorityQueue(3)
        self.left_object_queue: queue.PriorityQueue = queue.PriorityQueue(3)
        self.right_object_queue: queue.PriorityQueue = queue.PriorityQueue(3)
        self.right_interact_queue: queue.PriorityQueue = queue.PriorityQueue(3)
        self.left_interact_queue: queue.PriorityQueue = queue.PriorityQueue(3)
        self.subject_instance_name = "Mouse"
        self.left_object_name = "LeftTeaball"
        self.right_object_name = "RightTeaball"
        self.left_interact_name = "LeftInteract"
        self.right_interact_name = "RightInteract"
        self.num_instances_per_class = num_instances_per_class
        self.custom_activation: dict = {}

        self.device = _get_device()
        self.model = _build_model(self.num_classes, model_pth_path, self.device)

    def _predict(self, image_bgr: np.ndarray) -> SimpleInstances:
        """Run inference on a single BGR image and return SimpleInstances."""
        img_rgb = image_bgr[:, :, ::-1].copy()
        img_tensor = torch.as_tensor(
            img_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model([img_tensor])

        output = outputs[0]
        h, w = image_bgr.shape[:2]
        instances = _to_instances(output, (h, w))
        return instances.to("cpu")

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------

    def to_labelme(self, instances, image_path: str, height: int, width: int) -> str:
        """Convert ``instances`` to a LabelMe JSON file and return its path."""
        results = self._process_instances(instances, width=width)
        df_res = pd.DataFrame(results)
        df_res = df_res.groupby(["instance_name"], sort=False).head(
            self.num_instances_per_class
        )
        results = df_res.to_dict(orient="records")
        frame_label_list = []
        for res in results:
            label_list = pred_dict_to_labelme(res)
            frame_label_list += label_list
        img_ext = Path(image_path).suffix
        json_path = image_path.replace(img_ext, ".json")
        save_labels(
            json_path,
            str(Path(image_path).name),
            frame_label_list,
            height,
            width,
            imageData=None,
            save_image_to_json=False,
        )
        return json_path

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def on_image(self, image_path: str, display: bool = True) -> None:
        """Run instance segmentation on a single image."""
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        instances = self._predict(image)

        if len(instances) >= 1:
            self.to_labelme(instances, image_path, height, width)

        if display:
            vis_img = self._visualize(image, instances)
            cv2.imshow("Frame", vis_img)
            cv2.waitKey(0)

    def _visualize(self, image: np.ndarray, instances: SimpleInstances) -> np.ndarray:
        """Draw bounding boxes, labels, and semi-transparent masks on *image*."""
        vis = image.copy()
        boxes = instances.pred_boxes
        scores = instances.scores
        classes = instances.pred_classes
        has_mask = instances.has("pred_masks")

        # Generate distinct colours for each class
        np.random.seed(42)
        colors = np.random.randint(0, 255, size=(self.num_classes + 1, 3)).tolist()

        for i in range(len(instances)):
            if scores[i] < self.score_threshold:
                continue

            cls_id = int(classes[i])
            # Map 1-based label back to 0-based class_names index
            cls_idx = cls_id - 1 if cls_id > 0 else 0
            color = tuple(colors[cls_id])
            label = (
                self.class_names[cls_idx]
                if 0 <= cls_idx < len(self.class_names)
                else f"cls_{cls_id}"
            )
            score = float(scores[i])

            x1, y1, x2, y2 = [int(v) for v in boxes[i].tolist()]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            text = f"{label} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis, (x1, y1 - th - 6), (x1 + tw, y1), color, -1)
            cv2.putText(
                vis,
                text,
                (x1, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            if has_mask:
                mask = instances.pred_masks[i].numpy()
                colored_mask = np.zeros_like(vis)
                colored_mask[mask > 0] = color
                vis = cv2.addWeighted(vis, 1.0, colored_mask, 0.4, 0)

        return vis

    def on_image_folder(self, image_folder: str) -> None:
        """Run inference on all JPG/PNG images in a folder."""
        imgs = glob.glob(str(Path(image_folder) / "*.jpg"))
        if not imgs:
            imgs = glob.glob(str(Path(image_folder) / "*.png"))
        for img_path in imgs:
            self.on_image(img_path, display=False)

    def on_video(
        self,
        video_path: str,
        skip_frames: int = 1,
        on_keyframes: bool = False,
        tracking: bool = False,
        output_dir: Optional[str] = None,
    ):
        """Run inference (and optionally tracking) on a video file."""
        if not Path(video_path).exists():
            return

        self.cap = cv2.VideoCapture(video_path)
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tracker = None
        if tracking:
            from annolid.tracker.bbox_iou_tracker import BBoxIOUTracker

            tracker = BBoxIOUTracker(
                video_height=height,
                video_width=width,
                max_num_instances=200,
                max_lost_frame_count=30,
                min_box_rel_dim=0.02,
                min_instance_period=1,
                track_iou_threshold=0.3,
            )

        if on_keyframes:
            out_img_dir = key_frames(video_path)
            self.on_image_folder(out_img_dir)

        tracking_results = []
        frame_number = 0
        for frame in videos.frame_from_video(self.cap, num_frames):
            if frame_number % skip_frames == 0:
                instances = self._predict(frame)
                out_dict: dict = {}
                if tracker:
                    instances = tracker.update(instances)
                num_instance = len(instances)
                if num_instance == 0:
                    out_dict["frame_number"] = frame_number
                    out_dict["x1"] = None
                    out_dict["y1"] = None
                    out_dict["x2"] = None
                    out_dict["y2"] = None
                    out_dict["instance_name"] = None
                    out_dict["class_score"] = None
                    out_dict["segmentation"] = None
                    out_dict["tracking_id"] = None
                    tracking_results.append(out_dict)
                else:
                    _res = self._process_instances(instances, frame_number, width)
                    tracking_results += _res
            frame_number += 1
            if frame_number % 100 == 0:
                print("Processing frame number: ", frame_number)

        df = pd.DataFrame(tracking_results)
        df_top = df.groupby(["frame_number", "instance_name"], sort=False).head(
            self.num_instances_per_class
        )
        if output_dir:
            output_dir_path = Path(output_dir)
            output_dir_path.mkdir(parents=True, exist_ok=True)
            tracking_results_csv = (
                output_dir_path / f"{Path(video_path).stem}"
                "_mask_rcnn_tracking_results_with_segmentation.csv"
            )
        else:
            tracking_results_csv = (
                str(Path(video_path).with_suffix(""))
                + "_mask_rcnn_tracking_results_with_segmentation.csv"
            )
        df_top.to_csv(tracking_results_csv)

        if on_keyframes:
            print(f"Done. Please check your results in folder: {out_img_dir}")
            return out_img_dir
        print("Done!")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _save_pred_history(
        self, out_dict: dict, instance_name: str, instance_queue: queue.PriorityQueue
    ) -> None:
        if out_dict["instance_name"] == instance_name:
            if instance_queue.full():
                try:
                    instance_queue.get()
                except TypeError:
                    print(
                        "Comparison between instances of 'dict' and 'dict' is not supported."
                    )
            else:
                instance_queue.put((1 - out_dict["class_score"], out_dict))

    def _overlap_with_subject_instance(self, out_dict: dict) -> bool:
        if self.subject_queue.qsize() == 0:
            return True
        subject_instance_best_score = self.subject_queue.get()
        _iou = mask_iou(
            subject_instance_best_score[1]["segmentation"], out_dict["segmentation"]
        )
        self.subject_queue.put(subject_instance_best_score)
        return _iou > 0

    def _overlap_with_left_object(self, out_dict: dict) -> bool:
        if self.left_object_queue.qsize() == 0:
            return True
        left_object_best_score = self.left_object_queue.get()
        _iou = mask_iou(
            left_object_best_score[1]["segmentation"], out_dict["segmentation"]
        )
        self.left_object_queue.put(left_object_best_score)
        return _iou > 0

    def _overlap_with_right_object(self, out_dict: dict) -> bool:
        if self.right_object_queue.qsize() == 0:
            return True
        right_object_best_score = self.right_object_queue.get()
        _iou = mask_iou(
            right_object_best_score[1]["segmentation"], out_dict["segmentation"]
        )
        self.right_object_queue.put(right_object_best_score)
        return _iou > 0

    def subject_overlap_with_right_object(self) -> bool:
        if self.right_object_queue.qsize() == 0:
            return True
        right_object_best_score = self.right_object_queue.get()
        subject_best_score = self.subject_queue.get()
        _iou = mask_iou(
            right_object_best_score[1]["segmentation"],
            subject_best_score[1]["segmentation"],
        )
        self.right_object_queue.put(right_object_best_score)
        self.subject_queue.put(subject_best_score)
        return _iou > 0

    def subject_overlap_with_left_object(self) -> bool:
        if self.left_object_queue.qsize() == 0:
            return True
        left_object_best_score = self.left_object_queue.get()
        subject_best_score = self.subject_queue.get()
        _iou = mask_iou(
            left_object_best_score[1]["segmentation"],
            subject_best_score[1]["segmentation"],
        )
        self.left_object_queue.put(left_object_best_score)
        self.subject_queue.put(subject_best_score)
        return _iou > 0

    def _process_instances(self, instances, frame_number: int = 0, width=None) -> list:
        """Convert ``SimpleInstances`` to a list of result dicts."""
        results = []
        out_dict: dict = {}
        num_instance = len(instances)
        boxes = instances.pred_boxes
        scores = instances.scores
        classes = instances.pred_classes

        if instances.has("ID"):
            tracking_ids = instances.ID
        else:
            tracking_ids = None

        # Apply global NMS across all classes
        _keep = nms(boxes, scores, self.overlap_threshold)
        boxes = boxes[_keep]
        scores = scores[_keep]
        classes = classes[_keep]

        boxes = boxes.numpy().tolist()
        scores = scores.tolist()
        classes = classes.tolist()

        has_mask = instances.has("pred_masks")
        if has_mask:
            pred_masks = instances.pred_masks[_keep]
            rles = [
                mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[
                    0
                ]
                for mask in pred_masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")
        else:
            rles = []

        assert len(rles) == len(boxes)

        if num_instance != len(rles):
            num_instance = len(rles)

        for k in range(num_instance):
            box = boxes[k]
            out_dict["frame_number"] = frame_number
            out_dict["x1"] = box[0]
            out_dict["y1"] = box[1]
            out_dict["x2"] = box[2]
            out_dict["y2"] = box[3]
            out_dict["cx"] = (out_dict["x1"] + out_dict["x2"]) / 2
            out_dict["cy"] = (out_dict["y1"] + out_dict["y2"]) / 2
            # Map 1-based torchvision labels to 0-based class_names
            cls_idx = int(classes[k]) - 1
            if 0 <= cls_idx < len(self.class_names):
                out_dict["instance_name"] = self.class_names[cls_idx]
            else:
                out_dict["instance_name"] = f"cls_{classes[k]}"
            out_dict["class_score"] = scores[k]
            out_dict["segmentation"] = rles[k] if k < len(rles) else None
            if len(self.class_names) <= 1:
                out_dict["tracking_id"] = 0
            else:
                out_dict["tracking_id"] = (
                    tracking_ids[k] if tracking_ids is not None else None
                )

            if scores[k] >= self.score_threshold:
                out_dict["instance_name"] = TracksResults.switch_left_right(
                    out_dict, width=width
                )

                if out_dict["instance_name"] == self.subject_instance_name:
                    self._save_pred_history(
                        out_dict, self.subject_instance_name, self.subject_queue
                    )
                elif out_dict["instance_name"] == self.left_object_name:
                    self._save_pred_history(
                        out_dict, self.left_object_name, self.left_object_queue
                    )
                elif out_dict["instance_name"] == self.right_object_name:
                    self._save_pred_history(
                        out_dict, self.right_object_name, self.right_object_queue
                    )
                elif out_dict["instance_name"] == self.left_interact_name:
                    self._save_pred_history(
                        out_dict, self.left_interact_name, self.left_interact_queue
                    )
                    if not self._overlap_with_subject_instance(out_dict):
                        out_dict = {}
                        continue
                    if not self._overlap_with_left_object(out_dict):
                        out_dict = {}
                        continue
                elif out_dict["instance_name"] == self.right_interact_name:
                    self._save_pred_history(
                        out_dict, self.right_interact_name, self.left_interact_queue
                    )
                    if not self._overlap_with_subject_instance(out_dict):
                        out_dict = {}
                        continue
                    if not self._overlap_with_right_object(out_dict):
                        out_dict = {}
                        continue

                results.append(out_dict)
            out_dict = {}
        return results

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------

    def extract_mask_roi_features(self, frame):
        """Extract Mask R-CNN ROI features from a video frame."""
        im = frame[:, :, ::-1]
        height, width = im.shape[:2]
        image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1) / 255.0).to(
            self.device
        )

        self.model.eval()
        with torch.no_grad():
            images = self.model.transform([image])
            features = self.model.backbone(images.tensors)
            proposals, _ = self.model.rpn(images, features)
            # Return backbone features as a proxy
            return features

    def extract_backbone_features(self, frame):
        """Extract backbone feature maps from a video frame."""
        im = frame[:, :, ::-1]
        image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1) / 255.0).to(
            self.device
        )

        self.model.eval()
        with torch.no_grad():
            images = self.model.transform([image])
            return self.model.backbone(images.tensors)

    def get_activation_frome_layer(self, layer_name: str):
        """Return a forward-hook that captures output from a named layer."""

        def hook(model, input, output):
            self.custom_activation[layer_name] = output

        return hook
