import hashlib
import os
import cv2
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from annolid.utils.cancellation import (
    InferenceCancelledError,
    cancellable_download_file,
    default_model_cache_dir,
    is_stop_requested,
    torch_cancellable_forward,
)
from annolid.utils.devices import clear_device_cache, has_gpu
from annolid.utils.logger import logger


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

    _HUB_REPO_ID = "lkeab/hq-sam"
    _MODEL_TYPES = {"vit_l", "vit_h", "vit_b"}
    _DEFAULT_BOX_BATCH_SIZE = 64

    def __init__(
        self,
        checkpoint_path=None,
        model_type="vit_l",
        device="cpu",
        *,
        stop_event=None,
    ):
        """
        Initialize SamHQSegmenter instance.

        Parameters:
        - checkpoint_path (str): Path to the model checkpoint file.
        - model_type (str): Type of the SAM model to be used.
        - device (str): Device to run the model on (e.g., 'cpu' or 'cuda').
        - stop_event (threading.Event, optional): Cooperative cancellation event.
        """
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM-HQ initialization cancelled.")

        try:
            from segment_anything import (
                sam_model_registry,
                SamPredictor,
                SamAutomaticMaskGenerator,
            )
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "SamHQSegmenter requires the optional dependency 'segment_anything'. "
                'Install it with: pip install "segment-anything @ git+https://github.com/SysCV/sam-hq.git"'
            ) from exc

        self._sam_model_registry = sam_model_registry
        self._SamPredictor = SamPredictor
        self._SamAutomaticMaskGenerator = SamAutomaticMaskGenerator
        self.sam = None
        self.predictor = None
        self._cached_image_signature = None

        if torch.cuda.is_available() and has_gpu():
            device = "cuda"
        if checkpoint_path is None or not os.path.exists(checkpoint_path):
            checkpoint_path = self._download_model(model_type, stop_event=stop_event)

        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM-HQ initialization cancelled.")

        self.sam = self._sam_model_registry[model_type](checkpoint=checkpoint_path)
        try:
            if is_stop_requested(stop_event):
                raise InferenceCancelledError("SAM-HQ initialization cancelled.")
            self.sam.to(device=device)
            self.sam.eval()
            if is_stop_requested(stop_event):
                raise InferenceCancelledError("SAM-HQ initialization cancelled.")
            self.predictor = self._SamPredictor(self.sam)
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        """Release cached image embeddings and model memory."""
        self._cached_image_signature = None
        if getattr(self, "predictor", None) is not None and hasattr(
            self.predictor, "reset_image"
        ):
            try:
                self.predictor.reset_image()
            except Exception:
                pass
        self.predictor = None
        self.sam = None
        clear_device_cache()

    @staticmethod
    def _compute_image_signature(image: np.ndarray) -> tuple:
        arr = np.ascontiguousarray(image)
        digest = hashlib.blake2b(arr.view(np.uint8), digest_size=16).hexdigest()
        return (arr.shape, str(arr.dtype), digest)

    def _ensure_image_encoded(self, image: np.ndarray, *, stop_event=None) -> None:
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM-HQ image encoding cancelled.")
        signature = self._compute_image_signature(image)
        if self._cached_image_signature == signature and bool(
            getattr(self.predictor, "is_image_set", True)
        ):
            return
        try:
            with (
                torch.inference_mode(),
                torch_cancellable_forward(getattr(self, "sam", None), stop_event),
            ):
                self.predictor.set_image(image)
        except InferenceCancelledError:
            self._cached_image_signature = None
            reset_fn = getattr(self.predictor, "reset_image", None)
            if callable(reset_fn):
                try:
                    reset_fn()
                except Exception:
                    pass
            raise
        self._cached_image_signature = signature

    def _download_model(self, model_type, *, stop_event=None):
        """
        Download the model checkpoint file if it does not exist locally.

        Parameters:
        - model_type (str): Type of the SAM model to be used.
        - stop_event (threading.Event, optional): Cooperative cancellation event.

        Returns:
        - str: Path to the downloaded model checkpoint file.
        """
        if model_type not in self._MODEL_TYPES:
            raise ValueError(f"Unknown SAM-HQ model type '{model_type}'")
        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM-HQ checkpoint download cancelled.")

        model_name = f"sam_hq_{model_type}.pth"
        local_cache_file = default_model_cache_dir("sam_hq") / model_name
        try:
            if local_cache_file.is_file() and local_cache_file.stat().st_size > 0:
                return str(local_cache_file)
        except OSError:
            pass

        try:
            return hf_hub_download(
                repo_id=self._HUB_REPO_ID,
                filename=model_name,
                local_files_only=True,
            )
        except Exception:
            pass

        if is_stop_requested(stop_event):
            raise InferenceCancelledError("SAM-HQ checkpoint download cancelled.")

        try:
            from huggingface_hub import hf_hub_url
        except ImportError:
            hf_hub_url = None

        if callable(hf_hub_url):
            headers = None
            try:
                from huggingface_hub.utils import build_hf_headers

                headers = dict(build_hf_headers())
            except Exception:
                headers = None
            url = hf_hub_url(repo_id=self._HUB_REPO_ID, filename=model_name)
            return cancellable_download_file(
                url,
                local_cache_file,
                stop_event=stop_event,
                cache_subdir="sam_hq",
                headers=headers,
            )

        return hf_hub_download(
            repo_id=self._HUB_REPO_ID,
            filename=model_name,
        )

    def segment_objects(
        self,
        image,
        bboxes,
        *,
        batch_size: int | None = None,
        stop_event=None,
    ):
        """
        Segment objects in the image using HQ-SAM model.

        Parameters:
        - image (numpy.ndarray): Input image array.
        - bboxes (list): List of bounding boxes for objects in the image.
        Returns:
        - tuple: A tuple containing masks, scores, and input boxes.
        """
        if image is None or bboxes is None or len(bboxes) == 0:
            return [], [], []
        if is_stop_requested(stop_event):
            return [], [], np.empty((0, 4), dtype=np.float32)

        hq_token_only = False
        try:
            self._ensure_image_encoded(image, stop_event=stop_event)
        except InferenceCancelledError:
            return [], [], np.empty((0, 4), dtype=np.float32)

        if is_stop_requested(stop_event):
            return [], [], np.empty((0, 4), dtype=np.float32)

        chunk_size = max(1, int(batch_size or self._DEFAULT_BOX_BATCH_SIZE))
        bboxes_np = np.asarray(bboxes, dtype=np.float32).reshape(-1, 4)

        all_masks = []
        all_scores = []
        try:
            with (
                torch.inference_mode(),
                torch_cancellable_forward(getattr(self, "sam", None), stop_event),
            ):
                for start in range(0, len(bboxes_np), chunk_size):
                    if is_stop_requested(stop_event):
                        return [], [], np.empty((0, 4), dtype=np.float32)
                    batch_boxes = bboxes_np[start : start + chunk_size]
                    input_box = torch.as_tensor(
                        batch_boxes,
                        dtype=torch.float32,
                        device=self.predictor.device,
                    )
                    transformed_box = self.predictor.transform.apply_boxes_torch(
                        input_box, image.shape[:2]
                    )
                    try:
                        masks, scores, _ = self.predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_box,
                            multimask_output=False,
                            hq_token_only=hq_token_only,
                        )
                    except TypeError as exc:
                        if "hq_token_only" not in str(exc):
                            raise
                        masks, scores, _ = self.predictor.predict_torch(
                            point_coords=None,
                            point_labels=None,
                            boxes=transformed_box,
                            multimask_output=False,
                        )
                    all_masks.append(masks.squeeze(1).cpu().numpy())
                    all_scores.append(scores.squeeze(1).cpu().numpy())
        except InferenceCancelledError:
            return [], [], np.empty((0, 4), dtype=np.float32)
        except RuntimeError as exc:
            logger.warning("SAM-HQ object segmentation failed: %s", exc)
            return [], [], []

        if not all_masks:
            return [], [], np.empty((0, 4), dtype=np.float32)

        merged_masks = (
            np.concatenate(all_masks, axis=0) if len(all_masks) > 1 else all_masks[0]
        )
        merged_scores = (
            np.concatenate(all_scores, axis=0) if len(all_scores) > 1 else all_scores[0]
        )
        return merged_masks, merged_scores, bboxes_np

    def segment_everything(self, image, points_per_side=32, *, stop_event=None):
        """
        Segment objects in the given image using SamAutomaticMaskGenerator.
        """
        if is_stop_requested(stop_event):
            return []
        mask_generator = self._SamAutomaticMaskGenerator(
            self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=0.80,
            stability_score_thresh=0.90,
        )

        try:
            with (
                torch.inference_mode(),
                torch_cancellable_forward(getattr(self, "sam", None), stop_event),
            ):
                anns = mask_generator.generate(image)
        except InferenceCancelledError:
            return []
        except RuntimeError as exc:
            logger.warning("SAM-HQ automatic mask generation failed: %s", exc)
            return []

        return anns

    @staticmethod
    def show_mask(mask, ax, random_color=False):
        """
        Display the mask on the specified axis.

        Parameters:
        - mask (numpy.ndarray): Input mask array.
        - ax: Axis object for plotting.
        - random_color (bool): Flag to use random color for mask display.
        """
        color = (
            np.random.random(3)
            if random_color
            else np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        )
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
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )

    @staticmethod
    def show_box(box, ax):
        """
        Display the bounding box on the specified axis.
        """
        import matplotlib.pyplot as plt

        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )

    @staticmethod
    def show_res(masks, scores, input_point, input_label, input_box, filename, image):
        """
        Display segmentation results.
        """
        import matplotlib.pyplot as plt

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
            plt.axis("off")
            plt.savefig(
                filename + "_" + str(i) + ".png", bbox_inches="tight", pad_inches=-0.1
            )
            plt.close()

    @staticmethod
    def show_res_multi(
        masks, scores, input_point, input_label, input_box, filename, image
    ):
        """
        Display multiple segmentation results.
        """
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        for mask in masks:
            SamHQSegmenter.show_mask(mask, plt.gca(), random_color=True)
        for box in input_box:
            SamHQSegmenter.show_box(box, plt.gca())
        for score in scores:
            print(f"Score: {score:.3f}")
        plt.axis("off")
        plt.savefig(filename + ".png", bbox_inches="tight", pad_inches=-0.1)
        plt.close()


if __name__ == "__main__":
    from annolid.detector.grounding_dino import GroundingDINO

    gd = GroundingDINO()
    segmenter = SamHQSegmenter()

    image = cv2.imread("../sam-hq/demo/input_imgs/example5.png")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    text = "Eagle"
    gd_boxes = gd.predict_bboxes(image, text)

    bboxes = [list(box) for box, _ in gd_boxes]
    # bboxes = [[45,260,515,470], [310,228,424,296]]
    print(bboxes)
    output_path = "../sam-hq/demo/hq_sam_result/"
    os.makedirs(output_path, exist_ok=True)

    masks, scores, bboxes = segmenter.segment_objects(image, bboxes)
    print(masks, scores, bboxes)
    SamHQSegmenter.show_res_multi(
        masks, scores, None, None, bboxes, output_path + text, image
    )
