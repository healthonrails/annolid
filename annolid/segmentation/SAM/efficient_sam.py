import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image


class EfficientSAM:
    """
    Class for EfficientSAM segmentation.
    Reference: 
    https://github.com/yformer/EfficientSAM
    @article{xiong2023efficientsam,
    title={EfficientSAM: Leveraged Masked Image Pretraining for Efficient Segment Anything},
    author={Yunyang Xiong, Bala Varadarajan, Lemeng Wu, Xiaoyu Xiang, Fanyi Xiao, Chenchen Zhu,
        Xiaoliang Dai, Dilin Wang, Fei Sun, Forrest Iandola, Raghuraman Krishnamoorthi, Vikas Chandra},
    journal={arXiv:2312.00863},
    year={2023}
    }
    """

    def __init__(self, model):
        """
        Initialize the EfficientSAM with a given model.

        Parameters:
        - model: The model for prediction.
        """
        self.model = model

    def run_box_or_points(self, img_path, pts_sampled, pts_labels):
        """
        Run the box or points algorithm and return the predicted result.

        Parameters:
        - img_path (str): Path to the input image.
        - pts_sampled (list): List of sampled points.
        - pts_labels (list): List of labels corresponding to sampled points.

        Returns:
        - numpy.ndarray: Predicted result as a numpy array.
        """
        image_np = np.array(Image.open(img_path))
        img_tensor = ToTensor()(image_np)
        pts_sampled = torch.reshape(torch.tensor(pts_sampled), [1, 1, -1, 2])
        pts_labels = torch.reshape(torch.tensor(pts_labels), [1, 1, -1])
        predicted_logits, predicted_iou = self.model(
            img_tensor[None, ...],
            pts_sampled,
            pts_labels,
        )

        sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
        predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
        predicted_logits = torch.take_along_dim(
            predicted_logits, sorted_ids[..., None, None], dim=2
        )

        return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()

    def show_mask(self, mask, ax, random_color=False):
        """
        Show the mask on the given axis.

        Parameters:
        - mask (numpy.ndarray): Mask array.
        - ax: Matplotlib axis to display the mask.
        - random_color (bool): Whether to use a random color for the mask.
        """
        color = np.concatenate([np.random.random(3), np.array(
            [0.6])], axis=0) if random_color else np.array([30 / 255, 144 / 255, 255 / 255, 0.8])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)

    def show_points(self, coords, labels, ax, marker_size=375):
        """
        Show the points on the given axis.

        Parameters:
        - coords (numpy.ndarray): Coordinates of points.
        - labels (numpy.ndarray): Labels corresponding to points.
        - ax: Matplotlib axis to display the points.
        - marker_size (int): Size of the markers.
        """
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        self._scatter_points(ax, pos_points, color="green",
                             marker_size=marker_size)
        self._scatter_points(ax, neg_points, color="red",
                             marker_size=marker_size)

    def _scatter_points(self, ax, points, color, marker_size):
        ax.scatter(
            points[:, 0],
            points[:, 1],
            color=color,
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

    def show_box(self, box, ax):
        """
        Show the bounding box on the given axis.

        Parameters:
        - box (list): List containing [x0, y0, x1, y1] coordinates of the box.
        - ax: Matplotlib axis to display the box.
        """
        x0, y0, x1, y1 = box
        w, h = x1 - x0, y1 - y0
        ax.add_patch(
            plt.Rectangle((x0, y0), w, h, edgecolor="yellow",
                          facecolor=(0, 0, 0, 0), lw=5)
        )

    def show_annotations(self, mask, ax):
        """
        Show annotations using the given mask on the given axis.

        Parameters:
        - mask (numpy.ndarray): Mask array.
        - ax: Matplotlib axis to display the annotations.
        """
        ax.set_autoscale_on(False)
        img = np.ones((mask.shape[0], mask.shape[1], 4))
        img[:, :, 3] = 0
        color_mask = [0, 1, 0, 0.7]
        img[np.logical_not(mask)] = color_mask
        ax.imshow(img)
