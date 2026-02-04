import os
import torch
import logging
from pathlib import Path
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.data.datasets import builtin_meta
from detectron2.data import get_detection_dataset_dicts
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


class Segmentor:
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
    ) -> None:
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size

        self.logger = logging.getLogger(__name__)
        if model_config is None:
            model_config = "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"

        if out_put_dir is None:
            self.out_put_dir = str(
                Path(self.dataset_dir).parent / "Annolid_training_outputs"
            )
        else:
            self.out_put_dir = out_put_dir

        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold

        self.dataset_name = Path(self.dataset_dir).stem

        try:
            register_coco_instances(
                f"{self.dataset_name}_train",
                {},
                f"{self.dataset_dir}/train/annotations.json",
                f"{self.dataset_dir}/train/",
            )
            register_coco_instances(
                f"{self.dataset_name}_valid",
                {},
                f"{self.dataset_dir}/valid/annotations.json",
                f"{self.dataset_dir}/valid/",
            )
        except AssertionError as e:
            self.logger.info(e)

        get_detection_dataset_dicts([f"{self.dataset_name}_train"])

        _dataset_metadata = MetadataCatalog.get(f"{self.dataset_name}_train")
        _dataset_metadata.thing_colors = [
            cc["color"] for cc in builtin_meta.COCO_CATEGORIES
        ]
        num_classes = len(_dataset_metadata.thing_classes)
        self.class_names = _dataset_metadata.thing_classes

        self.cfg = get_cfg()
        # load model config and pretrained model
        self.cfg.merge_from_file(model_zoo.get_config_file(model_config))
        self.cfg.DATASETS.TRAIN = (f"{self.dataset_name}_train",)
        self.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.score_threshold
        self.cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
        self.cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = self.overlap_threshold

        # NMS threshold used on RPN proposals
        self.cfg.MODEL.RPN.NMS_THRESH = self.overlap_threshold

        self.cfg.DATASETS.TEST = ()

        self.cfg.DATALOADER.NUM_WORKERS = 2  # @param
        self.cfg.DATALOADER.SAMPLER_TRAIN = "RepeatFactorTrainingSampler"
        self.cfg.DATALOADER.REPEAT_THRESHOLD = 0.3
        if model_pth_path is not None:
            self.cfg.MODEL.WEIGHTS = model_pth_path
        else:
            self.cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                model_config
            )  # Let training initialize from model zoo
        self.cfg.SOLVER.IMS_PER_BATCH = self.batch_size  # @param
        self.cfg.SOLVER.BASE_LR = 0.0025  # @param # pick a good LR
        self.logger.info(f"Max iterations {max_iterations}")
        self.logger.info(f"Batch size is: {batch_size}")
        self.logger.info(f"Dataset dir is : {dataset_dir}")
        self.logger.info(f"Model config file is : {model_config}")
        # @param    # 3000 iterations seems good enough for 100 frames dataset;
        #  you will need to train longer for a practical dataset
        self.cfg.SOLVER.MAX_ITER = max_iterations
        self.cfg.SOLVER.CHECKPOINT_PERIOD = 1000  # @param
        # @param   # faster, and good enough for this toy dataset (default: 512)
        self.cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
        # (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
        self.cfg.OUTPUT_DIR = self.out_put_dir
        os.makedirs(self.cfg.OUTPUT_DIR, exist_ok=True)
        self.trainer = DefaultTrainer(self.cfg)
        if model_pth_path is not None:
            self.trainer.resume_or_load(resume=False)
        else:
            self.trainer.resume_or_load(resume=True)

    def train(self):
        self.trainer.train()
        try:
            self.evaluate_model()
        except AssertionError as ae:
            # skip evaluation in case the valid dataset is empty
            self.logger.info(ae)

    def evaluate_model(self):
        evaluator = COCOEvaluator(
            f"{self.dataset_name}_valid",
            self.cfg,
            False,
            output_dir=self.cfg.OUTPUT_DIR,
        )

        val_loader = build_detection_test_loader(self.cfg, f"{self.dataset_name}_valid")
        val_res = inference_on_dataset(self.trainer.model, val_loader, evaluator)
        self.logger.info(val_res)
        out_val_res_file = str(Path(self.cfg.OUTPUT_DIR) / "evalulation_results.txt")
        with open(out_val_res_file, "w") as text_file:
            text_file.write(str(val_res))
        return val_res
