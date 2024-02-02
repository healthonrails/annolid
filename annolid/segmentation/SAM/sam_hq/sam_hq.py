import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from annolid.utils.devices import has_gpu


class SamHQSegmenter:
    """
    SamHQSegmenter class for segmenting objects using HQ-SAM model.
    References: 
    @inproceedings{sam_hq,
    title={Segment Anything in High Quality},
    author={Ke, Lei and Ye, Mingqiao and Danelljan, Martin and Liu, 
    Yifan and Tai, Yu-Wing and Tang, Chi-Keung and Yu, Fisher},
    booktitle={NeurIPS},
    year={2023}
}  
    https://github.com/SysCV/sam-hq
    """

    _REMOTE_MODEL_URL = "https://huggingface.co/lkeab/hq-sam/resolve/main/"
    _MD5_DICT = {'vit_l': "08947267966e4264fb39523eccc33f86"}

    def __init__(self, checkpoint_path=None, model_type="vit_l", device="cpu"):
        """
        Initialize SamHQSegmenter instance.

        Parameters:
        - checkpoint_path (str): Path to the model checkpoint file.
        - model_type (str): Type of the SAM model to be used.
        - device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        """
        if has_gpu() and torch.cuda.is_available():
            device = 'cuda'
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            checkpoint_path = self._download_model(model_type)

        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)

    def _download_model(self, model_type):
        import gdown
        """
        Download the model checkpoint file if it does not exist locally.

        Parameters:
        - model_type (str): Type of the SAM model to be used.

        Returns:
        - str: Path to the downloaded model checkpoint file.
        """
        current_directory = os.path.dirname(os.path.abspath(__file__))
        model_name = f"sam_hq_{model_type}.pth"
        model_abs_path = os.path.join(current_directory, model_name)

        if not os.path.exists(model_abs_path):
            url = self._REMOTE_MODEL_URL + model_name
            expected_md5 = self._MD5_DICT[model_type]
            gdown.cached_download(url,
                                  model_abs_path,
                                  md5=expected_md5
                                  )

        return model_abs_path

    def segment_objects(self, image, bboxes):
        """
        Segment objects in the image using HQ-SAM model.

        Parameters:
        - image (numpy.ndarray): Input image array.
        - bboxes (list): List of bounding boxes for objects in the image.
        Returns:
        - tuple: A tuple containing masks, scores, and input boxes.
        """
        hq_token_only = False
        self.predictor.set_image(image)
        input_box = torch.tensor(bboxes, device=self.predictor.device)
        transformed_box = self.predictor.transform.apply_boxes_torch(
            input_box, image.shape[:2])
        masks, scores, _ = self.predictor.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_box,
            multimask_output=False,
            hq_token_only=hq_token_only,
        )
        masks = masks.squeeze(1).cpu().numpy()
        scores = scores.squeeze(1).cpu().numpy()
        input_box = input_box.cpu().numpy()
        return masks, scores, input_box

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        """
        Display the mask on the specified axis.

        Parameters:
        - mask (numpy.ndarray): Input mask array.
        - ax: Axis object for plotting.
        - random_color (bool): Flag to use random color for mask display.
        """
        color = np.random.random(3) if random_color else np.array(
            [30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    @staticmethod
    def show_points(coords, labels, ax, marker_size=375):
        """
        Display points on the specified axis.

        Parameters:
        - coords (numpy.ndarray): Coordinates of points.
        - labels (numpy.ndarray): Labels of points.
        - ax: Axis object for plotting.
        - marker_size (int): Size of the marker.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', 
                   marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', 
                   marker='*', s=marker_size, edgecolor='white',
                   linewidth=1.25)

    @staticmethod
    def show_box(box, ax):
        """
        Display the bounding box on the specified axis.

        Parameters:
        - box (list): List of coordinates representing the bounding box.
        - ax: Axis object for plotting.
        """
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h,
                                   edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

    @staticmethod
    def show_res(masks, scores, input_point, input_label, input_box, filename, image):
        """
        Display segmentation results.

        Parameters:
        - masks (numpy.ndarray): Segmentation masks.
        - scores (numpy.ndarray): Confidence scores.
        - input_point: Input points.
        - input_label: Input labels.
        - input_box: Input bounding boxes.
        - filename (str): Filename for saving the results.
        - image (numpy.ndarray): Input image array.
        """
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(image)
            SamHQSegmenter.show_mask(mask, plt.gca())
            if input_box is not None:
                box = input_box[i]
                SamHQSegmenter.show_box(box, plt.gca())
            if (input_point is not None) and (input_label is not None):
                SamHQSegmenter.show_points(input_point, input_label, plt.gca())

            print(f"Score: {score:.3f}")
            plt.axis('off')
            plt.savefig(filename + '_' + str(i) + '.png',
                        bbox_inches='tight', pad_inches=-0.1)
            plt.close()

    @staticmethod
    def show_res_multi(masks, scores, input_point, input_label, input_box, filename, image):
        """
        Display multiple segmentation results.

        Parameters:
        - masks (numpy.ndarray): Segmentation masks.
        - scores (numpy.ndarray): Confidence scores.
        - input_point: Input points.
        - input_label: Input labels.
        - input_box: Input bounding boxes.
        - filename (str): Filename for saving the results.
        - image (numpy.ndarray): Input image array.
        """
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            SamHQSegmenter.show_mask(mask, plt.gca(), random_color=True)
        for box in input_box:
            SamHQSegmenter.show_box(box, plt.gca())
        for score in scores:
            print(f"Score: {score:.3f}")
        plt.axis('off')
        plt.savefig(filename + '.png', bbox_inches='tight', pad_inches=-0.1)
        plt.close()


if __name__ == "__main__":
    from annolid.detector.grounding_dino import GroundingDINO

    gd = GroundingDINO()
    segmenter = SamHQSegmenter()

    image = cv2.imread('../sam-hq/demo/input_imgs/example5.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = "Eagle"
    gd_boxes = gd.predict_bboxes(image, text)

    bboxes = [list(box) for box, _ in gd_boxes]
    # bboxes = [[45,260,515,470], [310,228,424,296]]
    print(bboxes)
    output_path = '../sam-hq/demo/hq_sam_result/'
    os.makedirs(output_path, exist_ok=True)

    masks, scores, bboxes = segmenter.segment_objects(
        image, bboxes)
    print(masks, scores, bboxes)
    SamHQSegmenter.show_res_multi(
        masks, scores, None, None, bboxes, output_path + text, image)
