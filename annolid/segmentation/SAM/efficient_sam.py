import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor


class EfficientSAM:
    """
    Class for EfficientSAM segmentation using TorchScript models.
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

    def __init__(self, model_path):
        """
        Initialize the EfficientSAM class.

        Parameters:
        - model_path (str): Path to the TorchScript model file.
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path).to(self.device)

    def _load_model(self, model_path):
        """
        Load the TorchScript model from the given path.

        Parameters:
        - model_path (str): Path to the TorchScript model file.

        Returns:
        - torch.jit.ScriptModule: Loaded TorchScript model.
        """
        try:
            # Download and load the model
            return torch.jit.load(model_path)
        except Exception as e:
            raise ValueError(f"Error loading the model: {e}")

    def _preprocess_image(self, image_path):
        """
        Preprocess the input image.

        Parameters:
        - image_path (str): Path to the input image.

        Returns:
        - torch.Tensor: Preprocessed image tensor.
        """
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_tensor = ToTensor()(image)
        return img_tensor[None, ...].to(self.device)

    def _get_mask_and_iou(self, img_tensor, points_sampled, labels):
        """
        Get segmentation mask and IoU predictions.

        Parameters:
        - img_tensor (torch.Tensor): Preprocessed image tensor.
        - points_sampled (torch.Tensor): Sampled points tensor.
        - labels (torch.Tensor): Tensor containing labels.

        Returns:
        - Tuple[np.ndarray, np.ndarray]: Segmentation masks and IoU predictions.
        """
        predicted_logits, predicted_iou = self.model(
            img_tensor, points_sampled.to(self.device), labels.to(self.device))
        predicted_logits = predicted_logits.cpu()
        all_masks = torch.ge(torch.sigmoid(
            predicted_logits[0, 0, :, :, :]), 0.5).numpy()
        predicted_iou = predicted_iou[0, 0, ...].cpu().detach().numpy()
        return all_masks, predicted_iou

    def run_ours_point(self, image_path, points_sampled):
        """
        Run point segmentation.

        Parameters:
        - image_path (str): Path to the input image.
        - points_sampled (np.ndarray): Sampled points.

        Returns:
        - np.ndarray: Selected mask using predicted IoU.
        """
        img_tensor = self._preprocess_image(image_path)
        points_sampled = torch.reshape(torch.tensor(points_sampled), [
                                       1, 1, -1, 2]).to(self.device)
        max_num_pts = points_sampled.shape[2]
        labels = torch.ones(1, 1, max_num_pts).to(self.device)

        all_masks, predicted_iou = self._get_mask_and_iou(
            img_tensor, points_sampled, labels)

        max_predicted_iou = -1
        selected_mask_using_predicted_iou = None
        for m in range(all_masks.shape[0]):
            curr_predicted_iou = predicted_iou[m]
            if curr_predicted_iou > max_predicted_iou or selected_mask_using_predicted_iou is None:
                max_predicted_iou = curr_predicted_iou
                selected_mask_using_predicted_iou = all_masks[m]

        return selected_mask_using_predicted_iou

    def run_ours_box(self, image_path, points_sampled):
        """
        Run box segmentation.

        Parameters:
        - image_path (str): Path to the input image.
        - points_sampled (np.ndarray): Sampled points.

        Returns:
        - np.ndarray: Selected mask using predicted IoU.
        """
        img_tensor = self._preprocess_image(image_path)
        bbox = torch.reshape(torch.tensor(points_sampled),
                             [1, 1, 2, 2]).to(self.device)
        bbox_labels = torch.reshape(torch.tensor(
            [2, 3]), [1, 1, 2]).to(self.device)

        all_masks, predicted_iou = self._get_mask_and_iou(
            img_tensor, bbox, bbox_labels)

        max_predicted_iou = -1
        selected_mask_using_predicted_iou = None
        for m in range(all_masks.shape[0]):
            curr_predicted_iou = predicted_iou[m]
            if curr_predicted_iou > max_predicted_iou or selected_mask_using_predicted_iou is None:
                max_predicted_iou = curr_predicted_iou
                selected_mask_using_predicted_iou = all_masks[m]

        return selected_mask_using_predicted_iou

    @staticmethod
    def show_anns_ours(mask, ax):
        """
        Show segmentation annotations.

        Parameters:
        - mask (np.ndarray): Segmentation mask.
        - ax (matplotlib.axes.Axes): Matplotlib axes to display the mask.
        """
        ax.set_autoscale_on(False)
        img = np.ones((mask[0].shape[0], mask[0].shape[1], 4))
        img[:, :, 3] = 0
        for ann in mask:
            m = ann
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask
        ax.imshow(img)
