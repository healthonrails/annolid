import glob
import cv2
import numpy as np
import torch
import pandas as pd
import queue
from pathlib import Path
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.utils.visualizer import ColorMode, Visualizer
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import builtin_meta
from detectron2.data import get_detection_dataset_dicts
from annolid.postprocessing.quality_control import pred_dict_to_labelme
import pycocotools.mask as mask_util
from annolid.annotation.keypoints import save_labels
from annolid.postprocessing.quality_control import TracksResults
from annolid.annotation.masks import mask_iou
from annolid.data.videos import key_frames


class Segmentor():
    def __init__(self,
                 dataset_dir=None,
                 model_pth_path=None,
                 score_threshold=0.15,
                 overlap_threshold=0.5
                 ) -> None:
        self.dataset_dir = dataset_dir
        self.score_threshold = score_threshold

        dataset_name = Path(self.dataset_dir).stem
        self.subject_queue = queue.PriorityQueue(3)
        self.left_object_queue = queue.PriorityQueue(3)
        self.right_object_queue = queue.PriorityQueue(3)
        self.right_interact_queue = queue.PriorityQueue(3)
        self.left_interact_queue = queue.PriorityQueue(3)
        self.subject_instance_name = 'Mouse'
        self.left_object_name = 'LeftTeaball'
        self.right_object_name = 'RightTeaball'
        self.left_interact_name = 'LeftInteract'
        self.right_interact_name = 'RightInteract'

        register_coco_instances(f"{dataset_name}_train", {
        }, f"{self.dataset_dir}/train/annotations.json", f"{self.dataset_dir}/train/")
        register_coco_instances(f"{dataset_name}_valid", {
        }, f"{self.dataset_dir}/valid/annotations.json", f"{self.dataset_dir}/valid/")

        dataset_dicts = get_detection_dataset_dicts([f"{dataset_name}_train"])

        _dataset_metadata = MetadataCatalog.get(f"{dataset_name}_train")
        _dataset_metadata.thing_colors = [cc['color']
                                          for cc in builtin_meta.COCO_CATEGORIES]
        num_classes = len(_dataset_metadata.thing_classes)
        self.class_names = _dataset_metadata.thing_classes

        self.cfg = get_cfg()
        # load model config and pretrained model
        self.cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        self.cfg.MODEL.WEIGHTS = model_pth_path
        self.cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        self.cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = overlap_threshold

        # NMS threshold used on RPN proposals
        self.cfg.MODEL.RPN.NMS_THRESH = overlap_threshold

        self.predictor = DefaultPredictor(self.cfg)

    def to_labelme(self,
                   instances,
                   image_path,
                   height,
                   width):
        results = self._process_instances(instances, width=width)
        df_res = pd.DataFrame(results)
        df_res = df_res.groupby(['instance_name'], sort=False).head(1)
        results = df_res.to_dict(orient='records')
        frame_label_list = []
        for res in results:
            label_list = pred_dict_to_labelme(res)
            frame_label_list += label_list
        img_ext = Path(image_path).suffix
        json_path = image_path.replace(img_ext, ".json")
        save_labels(json_path,
                    str(Path(image_path).name),
                    frame_label_list,
                    height,
                    width,
                    imageData=None,
                    save_image_to_json=False
                    )
        return json_path

    def on_image(self, image_path, display=True):
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        preds = self.predictor(image)
        instances = preds["instances"].to('cpu')

        self.to_labelme(instances, image_path, height, width)

        if display:
            viz = Visualizer(image[:, :, ::-1],
                             metadata=MetadataCatalog.get(
                self.cfg.DATASETS.TRAIN[0]),
                instance_mode=ColorMode.SEGMENTATION
            )
            output = viz.draw_instance_predictions(
                instances
            )
            cv2.imshow("Frame", output.get_image()[:, :, ::-1])
            cv2.waitKey(0)

    def _save_pred_history(self,
                           out_dict,
                           instance_name,
                           instance_queue
                           ):

        if out_dict['instance_name'] == instance_name:
            if instance_queue.full():
                instance_high_score = instance_queue.get()
                instance_queue.get()
                instance_queue.put(instance_high_score)
            else:
                instance_queue.put(
                    (1-out_dict['class_score'], out_dict))

    def _overlap_with_subject_instance(self, out_dict):
        if self.subject_queue.qsize() == 0:
            return True
        subject_instance_best_score = self.subject_queue.get()
        _iou = mask_iou(
            subject_instance_best_score[1]['segmentation'],
            out_dict['segmentation']
        )
        self.subject_queue.put(subject_instance_best_score)
        if _iou <= 0:
            return False
        return True

    def _overlap_with_left_object(self,
                                  out_dict):
        if self.left_object_queue.qsize() == 0:
            return True
        left_object_best_score = self.left_object_queue.get()
        _iou = mask_iou(
            left_object_best_score[1]['segmentation'],
            out_dict['segmentation']
        )
        self.left_object_queue.put(left_object_best_score)
        return _iou > 0

    def _overlap_with_right_object(self,
                                   out_dict):
        if self.right_object_queue.qsize() == 0:
            return True
        right_object_best_score = self.right_object_queue.get()
        _iou = mask_iou(
            right_object_best_score[1]['segmentation'],
            out_dict['segmentation']
        )
        self.right_object_queue.put(right_object_best_score)
        return _iou > 0

    def subject_overlap_with_right_object(self):
        if self.right_object_queue.qsize() == 0:
            return True
        right_object_best_score = self.right_object_queue.get()
        subject_best_score = self.subject_queue.get()
        _iou = mask_iou(
            right_object_best_score[1]['segmentation'],
            subject_best_score[1]['segmentation']
        )
        self.right_object_queue.put(right_object_best_score)
        self.subject_queue.put(subject_best_score)
        return _iou > 0

    def subject_overlap_with_left_object(self):
        if self.left_object_queue.qsize() == 0:
            return True
        left_object_best_score = self.left_object_queue.get()
        subject_best_score = self.subject_queue.get()
        _iou = mask_iou(
            left_object_best_score[1]['segmentation'],
            subject_best_score[1]['segmentation']
        )
        self.left_object_queue.put(left_object_best_score)
        self.subject_queue.put(subject_best_score)
        return _iou > 0

    def _process_instances(self,
                           instances,
                           frame_number=0,
                           width=None
                           ):
        results = []
        out_dict = {}
        num_instance = len(instances)
        boxes = instances.pred_boxes.tensor.numpy()
        boxes = boxes.tolist()
        scores = instances.scores.tolist()
        classes = instances.pred_classes.tolist()

        has_mask = instances.has("pred_masks")

        if has_mask:
            rles = [
                mask_util.encode(
                    np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
                for mask in instances.pred_masks
            ]
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

        assert len(rles) == len(boxes)
        for k in range(num_instance):
            box = boxes[k]
            out_dict['frame_number'] = frame_number
            out_dict['x1'] = box[0]
            out_dict['y1'] = box[1]
            out_dict['x2'] = box[2]
            out_dict['y2'] = box[3]
            out_dict['cx'] = (out_dict['x1'] + out_dict['x2']) / 2
            out_dict['cy'] = (out_dict['y1'] + out_dict['y2']) / 2
            out_dict['instance_name'] = self.class_names[classes[k]]
            out_dict['class_score'] = scores[k]
            out_dict['segmentation'] = rles[k]

            if scores[k] >= self.score_threshold:
                out_dict['instance_name'] = TracksResults.switch_left_right(
                    out_dict, width=width)

                if out_dict['instance_name'] == self.subject_instance_name:
                    self._save_pred_history(out_dict,
                                            self.subject_instance_name,
                                            self.subject_queue)

                elif out_dict['instance_name'] == self.left_object_name:
                    self._save_pred_history(out_dict,
                                            self.left_object_name,
                                            self.left_object_queue)
                elif out_dict['instance_name'] == self.right_object_name:
                    self._save_pred_history(out_dict,
                                            self.right_object_name,
                                            self.right_object_queue)
                elif out_dict['instance_name'] == self.left_interact_name:
                    self._save_pred_history(out_dict,
                                            self.left_interact_name,
                                            self.left_interact_queue)
                    # check overlap with subject animal
                    if not self._overlap_with_subject_instance(out_dict):
                        out_dict = {}
                        continue
                    # check overlap with left object

                    if not self._overlap_with_left_object(out_dict):
                        out_dict = {}
                        continue

                    # if not self.subject_overlap_with_left_object():
                    #     out_dict = {}
                    #     continue

                elif out_dict['instance_name'] == self.right_interact_name:
                    self._save_pred_history(out_dict,
                                            self.right_interact_name,
                                            self.left_interact_queue)
                    if not self._overlap_with_subject_instance(out_dict):
                        out_dict = {}
                        continue

                    if not self._overlap_with_right_object(out_dict):
                        out_dict = {}
                        continue

                    # if not self.subject_overlap_with_right_object():
                    #     out_dict = {}
                    #     continue

                results.append(out_dict)
            out_dict = {}
        return results

    def on_image_folder(self,
                        image_folder
                        ):
        imgs = glob.glob(str(Path(image_folder) / '*.jpg'))
        if len(imgs) <= 0:
            imgs = glob.glob(str(Path(image_folder) / '*.png'))
        for img_path in imgs:
            self.on_image(img_path, display=False)

    def on_video(self, video_path):
        if not Path(video_path).exists():
            return
        out_img_dir = key_frames(video_path)
        self.on_image_folder(out_img_dir)
        print(f"Done. Please check you results in folder: {out_img_dir}")
        return out_img_dir
