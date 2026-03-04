"""
Detectron2-based Mask R-CNN inference / tracking for Annolid.

All ``detectron2`` imports are deferred to runtime inside ``Segmentor.__init__``
so this module can be imported on machines where detectron2 is not installed.
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


def _get_device() -> str:
    """Return the best available torch device string."""
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class Segmentor:
    """Instance segmentation / tracking using Detectron2 Mask R-CNN.

    All detectron2 imports happen inside ``__init__`` so this class can be
    referenced without detectron2 installed; the error is only raised when the
    class is actually instantiated.
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
        try:
            from detectron2 import model_zoo
            from detectron2.config import get_cfg
            from detectron2.data import MetadataCatalog
            from detectron2.data.datasets import builtin_meta, register_coco_instances
            from detectron2.engine import DefaultPredictor
        except ImportError as exc:
            raise ImportError(
                "detectron2 is required for mask-rcnn inference. "
                "See https://detectron2.readthedocs.io/tutorials/install.html"
            ) from exc

        self.dataset_dir = dataset_dir
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold

        if model_config is None:
            model_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

        dataset_name = Path(self.dataset_dir).stem
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

        try:
            register_coco_instances(
                f"{dataset_name}_train",
                {},
                f"{self.dataset_dir}/train/annotations.json",
                f"{self.dataset_dir}/train/",
            )
            register_coco_instances(
                f"{dataset_name}_valid",
                {},
                f"{self.dataset_dir}/valid/annotations.json",
                f"{self.dataset_dir}/valid/",
            )
        except AssertionError as e:
            print(e)

        _dataset_metadata = MetadataCatalog.get(f"{dataset_name}_train")
        _dataset_metadata.thing_colors = [
            cc["color"] for cc in builtin_meta.COCO_CATEGORIES
        ]
        num_classes = len(_dataset_metadata.thing_classes)
        self.class_names = _dataset_metadata.thing_classes

        self.cfg = get_cfg()
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))
        self.cfg.MODEL.WEIGHTS = model_pth_path
        self.cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        self.cfg.MODEL.DEVICE = _get_device()
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.overlap_threshold

        # NMS threshold used on RPN proposals
        self.cfg.MODEL.RPN.NMS_THRESH = self.overlap_threshold

        self.predictor = DefaultPredictor(self.cfg)

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
        """Run instance segmentation on a single image.

        Args:
            image_path: Path to the image file.
            display: Whether to show an OpenCV window with results.
        """
        from detectron2.data import MetadataCatalog
        from detectron2.utils.visualizer import ColorMode, Visualizer

        image = cv2.imread(image_path)
        height, width, _ = image.shape
        preds = self.predictor(image)
        instances = preds["instances"].to("cpu")

        if len(instances) >= 1:
            self.to_labelme(instances, image_path, height, width)

        if display:
            viz = Visualizer(
                image[:, :, ::-1],
                metadata=MetadataCatalog.get(self.cfg.DATASETS.TRAIN[0]),
                instance_mode=ColorMode.SEGMENTATION,
            )
            output = viz.draw_instance_predictions(instances)
            cv2.imshow("Frame", output.get_image()[:, :, ::-1])
            cv2.waitKey(0)

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
        """Run inference (and optionally tracking) on a video file.

        Args:
            video_path: Path to the input video.
            skip_frames: Process every N-th frame (1 = every frame).
            on_keyframes: Pre-extract key frames and run on those.
            tracking: Enable D2's IOU-based Hungarian tracker.
            output_dir: Directory to write the output CSV.  If ``None``, the
                CSV is written next to the video file.
        """
        if not Path(video_path).exists():
            return

        self.cap = cv2.VideoCapture(video_path)
        num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        tracker = None
        if tracking:
            from detectron2.config import instantiate

            tracker_cfg = {
                "_target_": (
                    "detectron2.tracking.iou_weighted_hungarian_bbox_iou_tracker"
                    ".IOUWeightedHungarianBBoxIOUTracker"
                ),
                "video_height": height,
                "video_width": width,
                "max_num_instances": 200,
                "max_lost_frame_count": 30,
                "min_box_rel_dim": 0.02,
                "min_instance_period": 1,
                "track_iou_threshold": 0.3,
            }
            tracker = instantiate(tracker_cfg)

        if on_keyframes:
            out_img_dir = key_frames(video_path)
            self.on_image_folder(out_img_dir)

        tracking_results = []
        frame_number = 0
        for frame in videos.frame_from_video(self.cap, num_frames):
            if frame_number % skip_frames == 0:
                outputs = self.predictor(frame)
                out_dict: dict = {}
                instances = outputs["instances"].to("cpu")
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
        """Convert Detectron2 ``Instances`` to a list of result dicts.

        Applies per-class NMS and maps class indices to human-readable names.

        Args:
            instances: Detectron2 ``Instances`` on CPU.
            frame_number: Frame index used for the output records.
            width: Frame width in pixels (used for left/right label switching).

        Returns:
            List of dicts, each representing one detected instance.
        """
        results = []
        out_dict: dict = {}
        num_instance = len(instances)
        boxes = instances.pred_boxes.tensor
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
            out_dict["instance_name"] = self.class_names[classes[k]]
            out_dict["class_score"] = scores[k]
            out_dict["segmentation"] = rles[k]
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
        """Extract Mask R-CNN ROI features from a video frame.

        Args:
            frame: BGR video frame (numpy array as returned by OpenCV).

        Returns:
            ROI mask pooler features as a tensor.
        """
        im = frame[:, :, ::-1]
        height, width = im.shape[:2]
        image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        images = self.predictor.model.preprocess_image(inputs)
        features = self.predictor.model.backbone(images.tensor)
        proposals, _ = self.predictor.model.proposal_generator(images, features)
        instances, _ = self.predictor.model.roi_heads(images, features, proposals)
        mask_features = [
            features[f] for f in self.predictor.model.roi_heads.in_features
        ]
        mask_features = self.predictor.model.roi_heads.mask_pooler(
            mask_features, [x.pred_boxes for x in instances]
        )
        return mask_features

    def extract_backbone_features(self, frame):
        """Extract backbone feature maps from a video frame.

        Args:
            frame: BGR video frame (numpy array as returned by OpenCV).

        Returns:
            Dict of feature-map tensors keyed by FPN level name.
        """
        im = frame[:, :, ::-1]
        height, width = im.shape[:2]
        image = torch.as_tensor(im.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": height, "width": width}]
        images = self.predictor.model.preprocess_image(inputs)
        return self.predictor.model.backbone(images.tensor)

    def get_activation_frome_layer(self, layer_name: str):
        """Return a forward-hook that captures output from a named layer.

        Usage example::

            hook = seg.get_activation_frome_layer("cls_score")
            seg.predictor.model.roi_heads.box_predictor.register_forward_hook(hook)

        Args:
            layer_name: Arbitrary key under which the activation is stored in
                ``self.custom_activation``.
        """

        def hook(model, input, output):
            self.custom_activation[layer_name] = output

        return hook
