import random
import torch
import os
import gdown
from PIL import Image
import numpy as np
import argparse
from .util.slconfig import SLConfig
from .util.misc import nested_tensor_from_tensor_list
from .datasets import transforms as T
import warnings
from annolid.utils.devices import get_device
from typing import Union

warnings.filterwarnings("ignore")

DEFAULT_CONF_THRESH = 0.23


class ObjectCounter:
    """
    A class for counting objects in images based on text prompts and optional visual exemplars.
    """
    device = get_device()

    def __init__(self, model_path: str = "checkpoint_best_regular.pth",
                 config_path: str = "cfg_app.py"
                 ):
        """
        Initializes the ObjectCounter with the model and configuration.

        Args:
            model_path: Path to the pretrained model checkpoint.
            config_path: Path to the configuration file.
            device: Device to load the model onto.
        """
        self.device = get_device()
        self.here = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(self.here, config_path)
        model_path = os.path.join(self.here, model_path)
        self.annolid_git_repo = "https://github.com/healthonrails/annolid/releases/download/v1.2.0"
        self._REMOTE_MODEL_URL = f"{self.annolid_git_repo}/checkpoint_best_regular.pth"
        self._MD5 = "1492bfdd161ac1de471d0aafb32b174d"
        if not os.path.exists(model_path):
            gdown.cached_download(self._REMOTE_MODEL_URL,
                                  model_path,
                                  md5=self._MD5
                                  )

        self._REMOTE_BERT_MODEL_URL = f"{self.annolid_git_repo}/model.safetensors"
        self._BERT_MD5 = "cd18ceb6b110c04a8033ce01de41b0b7"
        self._BERT_MODEL_PATH = os.path.join(
            self.here, "checkpoints/bert-base-uncased/model.safetensors")
        if not os.path.exists(self._BERT_MODEL_PATH):
            gdown.cached_download(self._REMOTE_BERT_MODEL_URL,
                                  self._BERT_MODEL_PATH,
                                  md5=self._BERT_MD5
                                  )

        self._REMOTE_GROUNDINGDINO_MODEL_URL = f"{self.annolid_git_repo}/groundingdino_swinb_cogcoor.pth"
        self._GROUNDINGDINO_MD5 = "611367df01ee834e3baa408f54d31f02"
        self._GROUNDINGDINO_MODEL_PATH = os.path.join(
            self.here, "checkpoints/groundingdino_swinb_cogcoor.pth")
        if not os.path.exists(self._GROUNDINGDINO_MODEL_PATH):
            gdown.cached_download(self._REMOTE_GROUNDINGDINO_MODEL_URL,
                                  self._GROUNDINGDINO_MODEL_PATH,
                                  md5=self._GROUNDINGDINO_MD5
                                  )

        self.model = self._load_model(model_path, config_path, self.device)
        self.transform = self._build_transforms()

    def _build_transforms(self, size: int = 800, max_size: int = 1333) -> T.Compose:
        """Builds data transformations for image preprocessing."""
        normalize = T.Compose(
            [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        )
        return T.Compose([T.RandomResize([size], max_size=max_size), normalize])

    def _load_model(self, model_path: str, config_path: str, device: str) -> torch.nn.Module:
        """Loads the counting model from a checkpoint."""
        cfg = SLConfig.fromfile(config_path)
        cfg.merge_from_dict(
            {"text_encoder_type": os.path.join(self.here, "checkpoints/bert-base-uncased")})

        parser = argparse.ArgumentParser("Model Config")
        args = parser.parse_args()

        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k, v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)
            elif args_vars[k] != v:
                print(
                    f"Warning: Overriding config parameter '{k}' with command-line value.")

        seed = 42
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        from .models.registry import MODULE_BUILD_FUNCS

        if args.modelname not in MODULE_BUILD_FUNCS._module_dict:
            raise ValueError(
                f"Model name '{args.modelname}' not found in registry.")
        build_func = MODULE_BUILD_FUNCS.get(args.modelname)
        model, _, _ = build_func(args)

        checkpoint = torch.load(model_path, map_location="cpu")["model"]
        model.load_state_dict(checkpoint, strict=False)
        model.to(device).eval()
        return model

    def _get_box_inputs(self, prompts: list) -> list:
        """Extracts bounding box coordinates from prompt data."""
        return [
            [prompt[0], prompt[1], prompt[3], prompt[4]]
            for prompt in prompts
            if prompt[2] == 2.0 and prompt[5] == 3.0
        ]

    def _get_ind_to_filter(self, text: str, word_ids: list, keywords: str) -> list:
        """Determines the indices of word IDs to filter based on keywords."""
        if not keywords:
            return list(range(len(word_ids)))

        input_words = text.split()
        keywords_list = [keyword.strip() for keyword in keywords.split(",")]

        word_inds = []
        for keyword in keywords_list:
            try:
                start_index = 0 if not word_inds else word_inds[-1] + 1
                ind = input_words.index(keyword, start_index)
                word_inds.append(ind)
            except ValueError:
                raise ValueError(
                    f"Keyword '{keyword}' not found in input text: '{text}'")

        inds_to_filter = [ind for ind, word_id in enumerate(
            word_ids) if word_id in word_inds]
        return inds_to_filter

    def _convert_boxes_to_xyxy(self, image: Image.Image, boxes: np.ndarray) -> list:
        """Converts normalized bounding boxes to x1,y1,x2,y2 format."""
        h, w = image.height, image.width
        xyxy_boxes = []
        for box in boxes:
            center_x, center_y, box_w, box_h = box
            x1 = int((center_x - box_w / 2) * w)
            y1 = int((center_y - box_h / 2) * h)
            x2 = int((center_x + box_w / 2) * w)
            y2 = int((center_y + box_h / 2) * h)
            xyxy_boxes.append([x1, y1, x2, y2])
        return xyxy_boxes

    def count_objects(
        self,
        image: Union[str, Image.Image],
        text_prompt: str,
        exemplar_image: Union[str, Image.Image] = None,
        exemplar_boxes: list = None,
        confidence_threshold: float = DEFAULT_CONF_THRESH,
        keywords: str = "",
    ) -> list:
        """Counts objects and returns bounding boxes in x1,y1,x2,y2 format.

        Args:
            image: Path to the input image or a PIL Image object.
            text_prompt: Textual description of the object to count.
            exemplar_image: Path to the exemplar image or a PIL Image object (optional).
            exemplar_boxes: List of exemplar bounding boxes in normalized [xmin, ymin, xmax, ymax] format (optional).
            confidence_threshold: Confidence threshold for object detection (optional).
            keywords: Comma-separated keywords to filter detected objects (optional).

        Returns:
            A list of detected bounding boxes in x1,y1,x2,y2 format.
        """
        if isinstance(image, str):
            image_pil = Image.open(image).convert("RGB")
        elif isinstance(image, Image.Image):
            image_pil = image.convert("RGB")
        else:
            raise ValueError(
                "image must be a file path (str) or a PIL Image object."
            )

        exemplar_prompts = {"image": None, "points": []}
        if exemplar_image:
            if isinstance(exemplar_image, str):
                exemplar_prompts["image"] = Image.open(
                    exemplar_image).convert("RGB")
            elif isinstance(exemplar_image, Image.Image):
                exemplar_prompts["image"] = exemplar_image.convert("RGB")
            else:
                raise ValueError(
                    "exemplar_image must be a file path (str) or a PIL Image object."
                )

            if exemplar_boxes:
                exemplar_prompts["points"] = [
                    [box[0], box[1], 2.0, box[2], box[3], 3.0] for box in exemplar_boxes
                ]

        input_image, _ = self.transform(
            image_pil, {"exemplars": torch.tensor([])})
        input_image = input_image.unsqueeze(0).to(self.device)
        exemplars_boxes_tensor = self._get_box_inputs(
            exemplar_prompts.get("points", []))

        input_image_exemplars = None
        exemplars_tensor = []
        if exemplar_prompts.get("image") is not None:
            exemplar_image_pil = exemplar_prompts["image"]
            input_image_exemplars, exemplars_transformed = self.transform(
                exemplar_image_pil, {
                    "exemplars": torch.tensor(exemplars_boxes_tensor)}
            )
            input_image_exemplars = input_image_exemplars.unsqueeze(0).to(
                self.device)
            exemplars_tensor = [exemplars_transformed["exemplars"].to(
                self.device)]

        with torch.no_grad():
            model_output = self.model(
                nested_tensor_from_tensor_list(input_image),
                nested_tensor_from_tensor_list(
                    input_image_exemplars) if input_image_exemplars is not None else None,
                exemplars_tensor,
                [torch.tensor([0]).to(self.device)] * len(input_image),
                captions=[text_prompt + " ."] * len(input_image),
            )

        ind_to_filter = self._get_ind_to_filter(
            text_prompt, model_output["token"][0].word_ids, keywords
        )
        logits = model_output["pred_logits"].sigmoid()[0][:, ind_to_filter]
        boxes = model_output["pred_boxes"][0]

        if keywords.strip():
            box_mask = (logits > confidence_threshold).sum(
                dim=-1) == len(ind_to_filter)
        else:
            box_mask = logits.max(dim=-1).values > confidence_threshold

        filtered_boxes = boxes[box_mask, :].cpu().numpy()
        return self._convert_boxes_to_xyxy(image_pil, filtered_boxes)


if __name__ == "__main__":
    input_image_path = "strawberry.jpg"
    text_prompt = "blueberries"
    exemplar_image_path = "strawberry.jpg"
    exemplar_boxes = [[0.1, 0.1, 0.2, 0.2]]

    pretrain_model_path = "checkpoint_best_regular.pth"
    config_path = "cfg_app.py"

    # Initialize the object counter
    object_counter = ObjectCounter(
        pretrain_model_path, config_path=config_path)

    # Get bounding boxes using image path
    detected_boxes_path = object_counter.count_objects(
        input_image_path,
        text_prompt,
        exemplar_image=exemplar_image_path,
        exemplar_boxes=exemplar_boxes,
        confidence_threshold=0.3,
        keywords="blueberries",
    )
    print("Detected Boxes (path input):", detected_boxes_path)

    # Get bounding boxes using PIL Image object
    image_pil = Image.open(input_image_path)
    exemplar_image_pil = Image.open(exemplar_image_path)
    detected_boxes_pil = object_counter.count_objects(
        image_pil,
        text_prompt,
        exemplar_image=exemplar_image_pil,
        exemplar_boxes=exemplar_boxes,
        confidence_threshold=0.3,
        keywords="blueberries",
    )
    print("Detected Boxes (PIL input):", detected_boxes_pil)
