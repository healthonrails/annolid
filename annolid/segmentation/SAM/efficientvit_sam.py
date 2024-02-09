import argparse
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
import gdown
import yaml
from copy import deepcopy
from typing import Any, Tuple, Union

"""
References: 

EfficientViT-SAM: Accelerated Segment Anything Model Without Performance Loss
@article{cai2022efficientvit,
  title={Efficientvit: Enhanced linear attention for high-resolution 
  low-computation visual recognition},
  author={Cai, Han and Gan, Chuang and Han, Song},
  journal={arXiv preprint arXiv:2205.14756},
  year={2022}
}
https://github.com/mit-han-lab/efficientvit

ONNX Export
# Export Encoder
```python deployment/sam/onnx/export_encoder.py --model xl1 \
--weight_url assets/checkpoints/sam/xl1.pt \
 --output assets/export_models/sam/onnx/xl1_encoder.onnx ```
# Export Decoder
```python deployment/sam/onnx/export_decoder.py --model xl1 \
 --weight_url assets/checkpoints/sam/xl1.pt \
 --output assets/export_models/sam/onnx/xl1_decoder.onnx \
 --return-single-mask
```

Example usage for bboxes input
```python efficientvit_sam.py  --model xl1 --encoder_model xl1_encoder.onnx \
--decoder_model xl1_decoder.onnx --mode boxes \
 --boxes "[[16,8,220,180],[230,190,440,400]]"
 ```

"""


class SamResize:
    """
    Resize image to the specified size.
    """

    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image.permute(2, 0, 1)

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Resize the image tensor.
        """
        target_size = self.get_preprocess_shape(
            image.shape[0], image.shape[1], self.size)
        return resize(image.permute(2, 0, 1), target_size)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int,
                             long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


def show_mask(mask, ax, random_color=False):
    """
    Display the mask.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    """
    Display points.
    """
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0], pos_points[:, 1], color="green",
        marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )
    ax.scatter(
        neg_points[:, 0], neg_points[:, 1], color="red",
        marker="*", s=marker_size, edgecolor="white", linewidth=1.25
    )


def show_box(box, ax):
    """
    Display bounding box.
    """
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor="green",
                               facecolor=(0, 0, 0, 0), lw=2))


def preprocess(x, img_size):
    """
    Preprocess the input image.
    """
    pixel_mean = [123.675 / 255, 116.28 / 255, 103.53 / 255]
    pixel_std = [58.395 / 255, 57.12 / 255, 57.375 / 255]

    x = torch.tensor(x)
    resize_transform = SamResize(img_size)
    x = resize_transform(x).float() / 255
    x = transforms.Normalize(mean=pixel_mean, std=pixel_std)(x)

    h, w = x.shape[-2:]
    th, tw = img_size, img_size
    assert th >= h and tw >= w
    x = F.pad(x, (0, tw - w, 0, th - h), value=0).unsqueeze(0).numpy()

    return x


def resize_longest_image_size(input_image_size: torch.Tensor,
                              longest_side: int) -> torch.Tensor:
    input_image_size = input_image_size.to(torch.float32)
    scale = longest_side / torch.max(input_image_size)
    transformed_size = scale * input_image_size
    transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
    return transformed_size


def mask_postprocessing(masks: torch.Tensor,
                        orig_im_size: torch.Tensor) -> torch.Tensor:
    img_size = 1024
    masks = torch.tensor(masks)
    orig_im_size = torch.tensor(orig_im_size)
    masks = F.interpolate(
        masks,
        size=(img_size, img_size),
        mode="bilinear",
        align_corners=False,
    )

    prepadded_size = resize_longest_image_size(orig_im_size, img_size)
    masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]
    orig_im_size = orig_im_size.to(torch.int64)
    h, w = orig_im_size[0], orig_im_size[1]
    masks = F.interpolate(masks, size=(
        h, w), mode="bilinear", align_corners=False)
    return masks


class SamEncoder:
    """
    Encoder for EfficientViTSAM model.
    """

    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError(
                "Invalid device, please use 'cuda' or 'cpu' device.")

        self.session = ort.InferenceSession(
            model_path, opt, providers=provider, **kwargs)
        self.input_name = self.session.get_inputs()[0].name

    def _extract_feature(self, tensor: np.ndarray) -> np.ndarray:
        feature = self.session.run(None, {self.input_name: tensor})[0]
        return feature

    def __call__(self, img: np.array, *args: Any, **kwds: Any) -> Any:
        return self._extract_feature(img)


class SamDecoder:
    """
    Decoder for EfficientViTSAM model.
    """

    def __init__(
        self, model_path: str, device: str = "cpu",
        target_size: int = 1024,
        mask_threshold: float = 0.0, **kwargs
    ):
        opt = ort.SessionOptions()

        if device == "cuda":
            provider = ["CUDAExecutionProvider"]
        elif device == "cpu":
            provider = ["CPUExecutionProvider"]
        else:
            raise ValueError(
                "Invalid device, please use 'cuda' or 'cpu' device.")

        self.target_size = target_size
        self.mask_threshold = mask_threshold
        self.session = ort.InferenceSession(
            model_path, opt, providers=provider, **kwargs)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int,
                             long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def run(
        self,
        img_embeddings: np.ndarray,
        origin_image_size: Union[list, tuple],
        point_coords: Union[list, np.ndarray] = None,
        point_labels: Union[list, np.ndarray] = None,
        boxes: Union[list, np.ndarray] = None,
        return_logits: bool = False,
    ):
        input_size = self.get_preprocess_shape(
            *origin_image_size, long_side_length=self.target_size)

        if point_coords is None and point_labels is None and boxes is None:
            raise ValueError(
                "Unable to segment, please input at least one box or point.")

        if img_embeddings.shape != (1, 256, 64, 64):
            raise ValueError("Got wrong embedding shape!")

        if point_coords is not None:
            point_coords = self.apply_coords(
                point_coords, origin_image_size, input_size).astype(np.float32)

        if boxes is not None:
            boxes = self.apply_boxes(
                boxes, origin_image_size, input_size).astype(np.float32)
            box_label = np.array([[2, 3] for _ in range(
                boxes.shape[0])], dtype=np.float32).reshape((-1, 2))
            point_coords = boxes
            point_labels = box_label

        input_dict = {"image_embeddings": img_embeddings,
                      "point_coords": point_coords, "point_labels": point_labels}
        low_res_masks, iou_predictions = self.session.run(None, input_dict)

        masks = mask_postprocessing(low_res_masks, origin_image_size)

        if not return_logits:
            masks = masks > self.mask_threshold
        return masks, iou_predictions, low_res_masks

    def apply_coords(self, coords, original_size, new_size):
        old_h, old_w = original_size
        new_h, new_w = new_size
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes, original_size, new_size):
        boxes = self.apply_coords(
            boxes.reshape(-1, 2, 2), original_size, new_size)
        return boxes


class EfficientViTSAM:
    """
    EfficientViTSAM model for image segmentation.
    """
    _REMOTE_MODEL_URL = "https://github.com/healthonrails/annolid/releases/download/v1.1.3/"
    _MD5_DICT = {"xl1_decoder.onnx": "a5ad8f15eee579a043d133762a994a5c",
                 "xl1_encoder.onnx": "69dafb5fb92edd6324ffa34fc2b2602b"
                 }

    def __init__(self,
                 model_type='xl1',
                 encoder_model="xl1_encoder.onnx",
                 decoder_model="xl1_decoder.onnx",
                 mode="boxes"):
        self.model_type = model_type
        current_file_path = os.path.abspath(__file__)
        self.current_folder = os.path.dirname(current_file_path)
        encoder_model = self._get_or_download(encoder_model)
        decoder_model = self._get_or_download(decoder_model)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder_model = SamEncoder(
            model_path=encoder_model, device=device)
        self.decoder_model = SamDecoder(
            model_path=decoder_model, device=device)
        self.mode = mode

    def _get_or_download(self,
                         model_path,
                         ):
        abs_model_path = os.path.join(self.current_folder, model_path)
        if not os.path.exists(abs_model_path):
            url = self._REMOTE_MODEL_URL + model_path
            expected_md5 = self._MD5_DICT[model_path]
            gdown.cached_download(url,
                                  abs_model_path,
                                  md5=expected_md5
                                  )
        return abs_model_path

    def preprocess_image(self, raw_img):
        """
        Preprocess the raw image.
        """
        if self.model_type in ["l0", "l1", "l2"]:
            img = preprocess(raw_img, img_size=512)
        elif self.model_type in ["xl0", "xl1"]:
            img = preprocess(raw_img, img_size=1024)
        else:
            raise NotImplementedError
        return img

    def run_inference(self, cv_image, bboxes, point=None):
        """
        Run inference on the input image.
        """
        raw_img = cv_image
        origin_image_size = raw_img.shape[:2]
        img = self.preprocess_image(raw_img)
        img_embeddings = self.encoder_model(img)
        if self.mode == "point":
            H, W, _ = raw_img.shape
            point = np.array(point, dtype=np.float32)
            point_coords = point[..., :2]
            point_labels = point[..., 2]
            masks, _, _ = self.decoder_model.run(
                img_embeddings=img_embeddings,
                origin_image_size=origin_image_size,
                point_coords=point_coords,
                point_labels=point_labels,
            )
            return masks.cpu().numpy()

        elif self.mode == "boxes":
            boxes = np.array(bboxes, dtype=np.float32)
            masks, _, _ = self.decoder_model.run(
                img_embeddings=img_embeddings,
                origin_image_size=origin_image_size,
                boxes=boxes,
            )
            return masks.cpu().numpy()
        else:
            return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str,
                        required=True, help="model type.")
    parser.add_argument(
        "--encoder_model", type=str,
        required=True, help="Path to the efficientvit_sam onnx encoder model."
    )
    parser.add_argument(
        "--decoder_model", type=str,
        required=True, help="Path to the efficientvit_sam onnx decoder model."
    )
    parser.add_argument("--img_path", type=str, default="cat.jpg")
    parser.add_argument("--out_path", type=str,
                        default="efficientvit_sam_demo_onnx.png")
    parser.add_argument("--mode", type=str, default="point",
                        choices=["point", "boxes"])
    parser.add_argument("--point", type=str, default=None)
    parser.add_argument("--boxes", type=str, default=None)
    args = parser.parse_args()

    encoder = SamEncoder(model_path=args.encoder_model)
    decoder = SamDecoder(model_path=args.decoder_model)
    eff_sam = EfficientViTSAM()

    raw_img = cv2.cvtColor(cv2.imread(args.img_path), cv2.COLOR_BGR2RGB)
    origin_image_size = raw_img.shape[:2]
    if args.model in ["l0", "l1", "l2"]:
        img = preprocess(raw_img, img_size=512)
    elif args.model in ["xl0", "xl1"]:
        img = preprocess(raw_img, img_size=1024)
    else:
        raise NotImplementedError

    img_embeddings = encoder(img)

    if args.mode == "point":
        H, W, _ = raw_img.shape
        point = np.array(yaml.safe_load(
            args.point or f"[[[{W // 2}, {H // 2}, {1}]]]"), dtype=np.float32)
        point_coords = point[..., :2]
        point_labels = point[..., 2]
        masks, _, _ = decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            point_coords=point_coords,
            point_labels=point_labels,
        )

        plt.figure(figsize=(10, 10))
        plt.imshow(raw_img)
        for mask in masks:
            show_mask(mask, plt.gca())
        show_points(point_coords, point_labels, plt.gca())
        plt.axis("off")
        plt.savefig(args.out_path, bbox_inches="tight",
                    dpi=300, pad_inches=0.0)
        print(f"Result saved in {args.out_path}")

    elif args.mode == "boxes":
        boxes = np.array(yaml.safe_load(args.boxes), dtype=np.float32)
        masks, _, _ = decoder.run(
            img_embeddings=img_embeddings,
            origin_image_size=origin_image_size,
            boxes=boxes,
        )
        plt.figure(figsize=(10, 10))
        plt.imshow(raw_img)
        for mask in masks:
            show_mask(mask, plt.gca())
        for box in boxes:
            show_box(box, plt.gca())
        plt.axis("off")
        plt.savefig(args.out_path, bbox_inches="tight",
                    dpi=300, pad_inches=0.0)
        print(f"Result saved in {args.out_path}")
        _masks = eff_sam.run_inference(raw_img, boxes)
        print(_masks.shape, _masks)
    else:
        raise NotImplementedError
