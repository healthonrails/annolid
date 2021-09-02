import os
import torch
from pathlib import Path
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import builtin_meta
from detectron2.data import get_detection_dataset_dicts


class Segmentor():
    def __init__(self,
                 dataset_dir=None,
                 out_put_dir=None,
                 score_threshold=0.15,
                 overlap_threshold=0.7,
                 max_iterations=3000,
                 ) -> None:
        self.dataset_dir = dataset_dir

        if out_put_dir is None:
            self.out_put_dir = str(
                Path(self.dataset_dir).parent / 'Annolid_training_outputs')
        else:
            self.out_put_dir = out_put_dir

        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold

        dataset_name = Path(self.dataset_dir).stem

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
        self.cfg.DATASETS.TRAIN = (f"{dataset_name}_train",)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        self.cfg.MODEL.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.overlap_threshold

        # NMS threshold used on RPN proposals
        self.cfg.MODEL.RPN.NMS_THRESH = self.overlap_threshold

        self.cfg.DATASETS.TEST = ()

        self.cfg.DATALOADER.NUM_WORKERS = 2  # @param
        self.cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
        self.cfg.DATALOADER.REPEAT_THRESHOLD = 0.3
        self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo
        self.cfg.SOLVER.IMS_PER_BATCH = 8  # @param
        self.cfg.SOLVER.BASE_LR = 0.0025  # @param # pick a good LR
        print(f"{max_iterations} seems good enough for 100 label frames")
        # @param    # 3000 iterations seems good enough for 100 frames dataset; you will need to train longer for a practical dataset
        self.cfg.SOLVER.MAX_ITER = max_iterations
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # @param
        # @param   # faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        self.cfg.OUTPUT_DIR = self.out_put_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.trainer = DefaultTrainer(self.cfg)
        self.trainer.resume_or_load(resume=True)

    def train(self):
        self.trainer.train()


if __name__ == '__main__':
    segmentor = Segmentor(
        "/Users/chenyang/Downloads/teaball_dataset_coco_dataset/")
    segmentor.train()
