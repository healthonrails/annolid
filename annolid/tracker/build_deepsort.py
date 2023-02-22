from annolid.tracker.deep_sort import DeepSort
from annolid.utils.config import get_config
import torch


def build_tracker(cfg_file="./configs/deep_sort.yaml",
                  use_cuda=None):
    if use_cuda is None:
        use_cuda = torch.cuda.is_available()
    cfg = get_config(cfg_file)
    return DeepSort(cfg.DEEPSORT.REID_CKPT,
                    max_dist=cfg.DEEPSORT.MAX_DIST,
                    min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
                    nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
                    max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                    max_age=cfg.DEEPSORT.MAX_AGE,
                    n_init=cfg.DEEPSORT.N_INIT,
                    nn_budget=cfg.DEEPSORT.NN_BUDGET,
                    use_cuda=use_cuda)
