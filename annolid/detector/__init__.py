import os
import torch
from annolid.utils.config import get_config
__all__ = ['build_detector']


def build_detector(cfg_file="./configs/yolov3.yaml",
                   use_cuda=None
                   ):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    cfg = get_config(cfg_file)
    yolo_version = 3 if '3' in os.path.basename(cfg_file) else 5
    if yolo_version == 3:
        from .YOLOv3 import YOLOv3
        return YOLOv3(cfg.YOLOV3.CFG,
                      cfg.YOLOV3.WEIGHT,
                      cfg.YOLOV3.CLASS_NAMES,
                      score_thresh=cfg.YOLOV3.SCORE_THRESH,
                      nms_thresh=cfg.YOLOV3.NMS_THRESH,
                      is_xywh=True,
                      use_cuda=use_cuda)
