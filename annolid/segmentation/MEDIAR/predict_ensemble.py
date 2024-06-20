import os
import torch
import time
from datetime import datetime
import numpy as np
from annolid.segmentation.MEDIAR.train_tools import *
from annolid.segmentation.MEDIAR.train_tools.models import MEDIARFormer
from annolid.utils.weights import WeightDownloader
from annolid.segmentation.MEDIAR.core.MEDIAR import EnsemblePredictor
from annolid.gui.shape import MaskShape
from annolid.annotation.keypoints import save_labels
from annolid.segmentation.MEDIAR.train_tools.data_utils.transforms import get_pred_transforms
from annolid.utils.logger import logger


class MEDIARPredictor(EnsemblePredictor):
    """
    Class for conducting cell detection using MEDIAR model.
    https://github.com/Lee-Gihun/MEDIAR
    @article{lee2022mediar,
    title={Mediar: Harmony of data-centric and model-centric for multi-modality microscopy},
    author={Lee, Gihun and Kim, SangMook and Kim, Joonkee and Yun, Se-Young},
    journal={arXiv preprint arXiv:2212.03465},
    year={2022}
    }

    Args:
        weights_dir (str): Directory to store downloaded weights.
        input_path (str): Directory containing input images.
        output_path (str): Directory to store output predictions.
    """

    def __init__(self,
                 input_path="./data/images",
                 output_path="./output",
                 weights_dir="None",
                 ):
        self.weights_dir = weights_dir
        self.input_path = input_path
        self.output_path = output_path
        self.model1 = None
        self.model2 = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.download_weights()
        self.load_models()
        super(MEDIARPredictor, self).__init__(self.model1, self.model2,
                                              self.device, self.input_path,
                                              self.output_path, algo_params={"use_tta": True})

    def _setups(self):
        self.pred_transforms = get_pred_transforms()
        os.makedirs(self.output_path, exist_ok=True)

        now = datetime.now()
        dt_string = now.strftime("%m%d_%H%M")
        self.exp_name = (
            self.exp_name + dt_string if self.exp_name is not None else dt_string
        )

        self.img_names = [img_file for img_file in sorted(
            os.listdir(self.input_path)) if '.json' not in img_file]
        logger.info(f"Working on the images: {self.img_names}")

    def download_weights(self, weights_dir=None):
        """
        Download pretrained weights for MEDIAR models.
        """
        if weights_dir is None:
            self.weights_dir = os.path.join(
                os.path.dirname(__file__), "weights")
            downloader = WeightDownloader(self.weights_dir)

        # Define weight URLs, expected checksums, and file names
        weight_urls = [
            "https://drive.google.com/uc?id=168MtudjTMLoq9YGTyoD2Rjl_d3Gy6c_L",
            "https://drive.google.com/uc?id=1JJ2-QKTCk-G7sp5ddkqcifMxgnyOrXjx"
        ]
        expected_checksums = [
            "e0ccb052828a9f05e21b2143939583c5",
            "a595336926767afdf1ffb1baffd5ab7f"
        ]
        weight_file_names = ["from_phase1.pth", "from_phase2.pth"]

        # Download weights for each URL
        for url, checksum, file_name in zip(weight_urls, expected_checksums, weight_file_names):
            downloader.download_weights(url, checksum, file_name)

    def load_models(self):
        """
        Load pretrained MEDIAR models.
        """
        model_args = {
            "classes": 3,
            "decoder_channels": [1024, 512, 256, 128, 64],
            "decoder_pab_channels": 256,
            "encoder_name": 'mit_b5',
            "in_channels": 3
        }
        self.model1 = MEDIARFormer(**model_args)
        self.model1.load_state_dict(torch.load(f"{self.weights_dir}/from_phase1.pth",
                                               map_location="cpu"), strict=False)

        self.model2 = MEDIARFormer(**model_args)
        self.model2.load_state_dict(torch.load(f"{self.weights_dir}/from_phase2.pth",
                                               map_location="cpu"), strict=False)

    @torch.no_grad()
    def conduct_prediction(self):
        self.model.to(self.device)
        self.model.eval()
        total_time = 0
        total_times = []

        for img_name in self.img_names:
            img_data = self._get_img_data(img_name)
            img_data = img_data.to(self.device)

            start = time.time()

            pred_mask = self._inference(img_data)
            pred_mask = self._post_process(pred_mask.squeeze(0).cpu().numpy())

            self.write_pred_mask(
                pred_mask, self.output_path, img_name, self.make_submission
            )
            shape_list = self.save_prediction(
                pred_mask, image_name=img_name)
            end = time.time()

            time_cost = end - start
            total_times.append(time_cost)
            total_time += time_cost
            logger.info(
                f"Prediction finished: {img_name}; img size = {img_data.shape}; costing: {time_cost:.2f}s"
            )

        logger.info(f"\n Total Time Cost: {total_time:.2f}s")

        return shape_list

    def _save_annotation(self, filename, mask_dict,
                         frame_shape,
                         img_ext='.png'):
        if len(frame_shape) == 3:
            height, width, _ = frame_shape
        else:
            height, width = frame_shape
        label_list = []
        for label_id, mask in mask_dict.items():
            label = str(label_id)
            current_shape = MaskShape(label=label,
                                      flags={},
                                      description='cell segmentation')
            current_shape.mask = mask
            _shapes = current_shape.toPolygons(
                epsilon=2.0)
            if len(_shapes) < 0:
                continue
            current_shape = _shapes[0]
            points = [[point.x(), point.y()] for point in current_shape.points]
            current_shape.points = points
            label_list.append(current_shape)
        img_abs_path = filename.replace('.json', img_ext)
        save_labels(filename=filename,
                    imagePath=img_abs_path,
                    label_list=label_list,
                    height=height,
                    width=width,
                    save_image_to_json=False)
        return label_list

    def save_prediction(self, pred_mask, image_name="img1"):
        """
        Save prediction for a specific image.

        Args:
            image_name (str): Name of the input image file.
        """
        img_filename = os.path.join(self.input_path, image_name)
        _, ext = os.path.splitext(img_filename)
        # Replace the extension with ".json"
        json_filename = img_filename.replace(ext, '.json')
        json_filename = os.path.abspath(json_filename)
        mask_dict = {f'cell_{label_id}': (pred_mask == label_id)
                     for label_id in np.unique(pred_mask)[1:]}
        shape_list = self._save_annotation(json_filename, mask_dict,
                                           pred_mask.shape, img_ext=ext)

        cell_count = len(np.unique(pred_mask)) - 1  # exclude the background
        logger.info(f"\n{cell_count} Cells detected!")
        return shape_list


if __name__ == "__main__":
    mediar_predictor = MEDIARPredictor()
    shape_list = mediar_predictor.conduct_prediction()
