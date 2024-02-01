import os
import cv2
import numpy as np
from typing import Dict
from annolid.inference.onnx_model import ONNXBaseModel
from annolid.gui.shape import Shape
from annolid.annotation.keypoints import save_labels


class GroundingDINO:
    """
    Open-Set object detection model using Grounding_DINO
    Reference: @article{liu2023grounding,
  title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
  author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang,
    Jie and Li, Chunyuan and Yang, Jianwei and Su, Hang and Zhu, Jun and others},
  journal={arXiv preprint arXiv:2303.05499},
  year={2023}
}
    https://github.com/IDEA-Research/GroundingDINO
    Modified from:
    https://github.com/CVHub520/X-AnyLabeling/blob/main/anylabeling/services/auto_labeling/grounding_dino.py
    Onnx Model: 
    https://github.com/CVHub520/X-AnyLabeling/releases/download/v1.0.0/groundingdino_swinb_cogcoor_quant.onnx
    """

    def __init__(self,
                 model_abs_path=None,
                 model_type="groundingdino_swinb_cogcoor") -> None:
        """
        Initialize GroundingDINO model
        Args:
            model_abs_path (str): Absolute path to the model
            model_type (str): Type of the model
        """
        if model_abs_path is None:
            current_directory = os.path.dirname(os.path.abspath(__file__))
            model_abs_path = os.path.join(
                current_directory, model_type + '_quant.onnx')

        self.net = ONNXBaseModel(model_abs_path)
        self.model_configs = self._get_configs(model_type)
        self.net.max_text_len = self.model_configs['max_text_len']
        self.net.tokenizer = self._get_tokenizer(
            self.model_configs['text_encoder_type'])
        self.box_threshold = 0.3
        self.text_threshold = 0.2
        self.target_size = (1200, 800)

    def preprocess(self, image, text_prompt):
        """
        Preprocess image and text for inference
        Args:
            image: Input image
            text_prompt: Text prompt
        Returns:
            image: Preprocessed image
            inputs: Preprocessed inputs
            captions: Processed captions
        """
        image = cv2.resize(image, self.target_size,
                           interpolation=cv2.INTER_LINEAR)
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0).astype(np.float32)

        captions = self._get_caption(str(text_prompt))
        tokenized_raw_results = self.net.tokenizer.encode(captions)
        tokenized = {
            "input_ids": np.array([tokenized_raw_results.ids], dtype=np.int64),
            "token_type_ids": np.array([tokenized_raw_results.type_ids], dtype=np.int64),
            "attention_mask": np.array([tokenized_raw_results.attention_mask]),
        }
        special_tokens = [101, 102, 1012, 1029]
        text_self_attention_masks, position_ids, _ = self._generate_masks_with_special_tokens_and_transfer_map(
            tokenized, special_tokens
        )
        if text_self_attention_masks.shape[1] > self.net.max_text_len:
            text_self_attention_masks = text_self_attention_masks[:,
                                                                  :self.net.max_text_len, :self.net.max_text_len]
            position_ids = position_ids[:, :self.net.max_text_len]
            tokenized["input_ids"] = tokenized["input_ids"][:,
                                                            :self.net.max_text_len]
            tokenized["attention_mask"] = tokenized["attention_mask"][:,
                                                                      :self.net.max_text_len]
            tokenized["token_type_ids"] = tokenized["token_type_ids"][:,
                                                                      :self.net.max_text_len]

        inputs = {
            "img": image,
            "input_ids": np.array(tokenized["input_ids"], dtype=np.int64),
            "attention_mask": np.array(tokenized["attention_mask"], dtype=bool),
            "token_type_ids": np.array(tokenized["token_type_ids"], dtype=np.int64),
            "position_ids": np.array(position_ids, dtype=np.int64),
            "text_token_mask": np.array(text_self_attention_masks, dtype=bool),
        }
        return image, inputs, captions

    def postprocess(self, outputs, caption, with_logits=True, token_spans=None):
        """
        Postprocess model outputs.

        Args:
            outputs: Tuple containing logits and boxes
            caption (str): Caption associated with the image
            with_logits (bool): Flag indicating whether to include logits in the output
            token_spans: Token spans to consider (currently not implemented)

        Returns:
            boxes_filt: Filtered bounding boxes
            pred_phrases: Predicted phrases along with associated confidence scores if with_logits is True
        """
        logits, boxes = outputs
        prediction_logits_ = np.squeeze(logits, 0)
        logits_filt = self.sigmoid(prediction_logits_)
        boxes_filt = np.squeeze(boxes, 0)

        if token_spans is None:
            filt_mask = logits_filt.max(axis=1) > self.box_threshold
            logits_filt = logits_filt[filt_mask]
            boxes_filt = boxes_filt[filt_mask]

            tokenlizer = self.net.tokenizer
            tokenized_raw_results = tokenlizer.encode(caption)
            tokenized = {
                "input_ids": np.array(tokenized_raw_results.ids, dtype=np.int64),
                "token_type_ids": np.array(tokenized_raw_results.type_ids, dtype=np.int64),
                "attention_mask": np.array(tokenized_raw_results.attention_mask),
            }

            pred_phrases = []
            for logit in logits_filt:
                posmap = logit > self.text_threshold
                pred_phrase = self._get_phrases_from_posmap(
                    posmap, tokenized, tokenlizer)
                if with_logits:
                    pred_phrases.append([pred_phrase, logit.max()])
                else:
                    pred_phrases.append([pred_phrase, 1.0])
        else:
            raise NotImplementedError(
                "Using token_spans is not implemented yet")

        return boxes_filt, pred_phrases

    def predict_shapes(self, image, text_prompt=None):
        """
        Predict shapes from image.

        Args:
            image: Input image
            text_prompt: Text prompt associated with the image

        Returns:
            shapes: List of predicted shapes
        """
        if image is None:
            return []

        blob, inputs, caption = self.preprocess(image, text_prompt)
        outputs = self.net.get_onnx_inference(
            blob, input_data=inputs, extract=False)
        boxes_filt, pred_phrases = self.postprocess(outputs, caption)

        shapes = []
        img_h, img_w, _ = image.shape
        boxes = self._rescale_boxes(boxes_filt, img_h, img_w)
        for box, label_info in zip(boxes, pred_phrases):
            x1, y1, x2, y2 = box
            label, _ = label_info
            shape = Shape(label=str(label), shape_type="rectangle", flags={})
            shape.addPoint((x1, y1))
            shape.addPoint((x2, y2))
            shapes.append(shape)

        return shapes

    def predict_bboxes(self, image, text_prompt=None):
        """
        Predict shapes from image.

        Args:
            image: Input image
            text_prompt: Text prompt associated with the image

        Returns:
            bboxes: List of predicted bboxes
        """
        if image is None:
            return []

        blob, inputs, caption = self.preprocess(image, text_prompt)
        outputs = self.net.get_onnx_inference(
            blob, input_data=inputs, extract=False)
        boxes_filt, pred_phrases = self.postprocess(outputs, caption)

        bboxes = []
        img_h, img_w, _ = image.shape
        boxes = self._rescale_boxes(boxes_filt, img_h, img_w)
        for box, label_info in zip(boxes, pred_phrases):
            label, _ = label_info
            bboxes.append((box, label))

        return bboxes

    @staticmethod
    def sigmoid(x):
        """
        Sigmoid function

        Args:
            x: Input array

        Returns:
            Array after applying the sigmoid function
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def _rescale_boxes(boxes, img_h, img_w):
        """
        Rescale bounding boxes.

        Args:
            boxes: Bounding boxes
            img_h: Image height
            img_w: Image width

        Returns:
            converted_boxes: Rescaled bounding boxes
        """
        converted_boxes = []
        for box in boxes:
            # from 0..1 to 0..W, 0..H
            converted_box = box * np.array([img_w, img_h, img_w, img_h])
            # from xywh to xyxy
            converted_box[:2] -= converted_box[2:] / 2
            converted_box[2:] += converted_box[:2]
            converted_boxes.append(converted_box)
        return np.array(converted_boxes, dtype=float)

    @staticmethod
    def _get_configs(model_type):
        """
        Get model configurations based on the model type.

        Args:
            model_type (str): Type of the model

        Returns:
            configs: Dictionary containing model configurations
        """
        if model_type == "groundingdino_swinb_cogcoor":
            configs = {
                "batch_size": 1,
                "modelname": "groundingdino",
                "backbone": "swin_B_384_22k",
                "position_embedding": "sine",
                "pe_temperatureH": 20,
                "pe_temperatureW": 20,
                "return_interm_indices": [1, 2, 3],
                "backbone_freeze_keywords": None,
                "enc_layers": 6,
                "dec_layers": 6,
                "pre_norm": False,
                "dim_feedforward": 2048,
                "hidden_dim": 256,
                "dropout": 0.0,
                "nheads": 8,
                "num_queries": 900,
                "query_dim": 4,
                "num_patterns": 0,
                "num_feature_levels": 4,
                "enc_n_points": 4,
                "dec_n_points": 4,
                "two_stage_type": "standard",
                "two_stage_bbox_embed_share": False,
                "two_stage_class_embed_share": False,
                "transformer_activation": "relu",
                "dec_pred_bbox_embed_share": True,
                "dn_box_noise_scale": 1.0,
                "dn_label_noise_ratio": 0.5,
                "dn_label_coef": 1.0,
                "dn_bbox_coef": 1.0,
                "embed_init_tgt": True,
                "dn_labelbook_size": 2000,
                "max_text_len": 256,
                "text_encoder_type": "bert-base-uncased",
                "use_text_enhancer": True,
                "use_fusion_layer": True,
                "use_checkpoint": True,
                "use_transformer_ckpt": True,
                "use_text_cross_attention": True,
                "text_dropout": 0.0,
                "fusion_dropout": 0.0,
                "fusion_droppath": 0.1,
                "sub_sentence_present": True,
            }
        elif model_type == "groundingdino_swint_ogc":
            configs = {
                "batch_size": 1,
                "modelname": "groundingdino",
                "backbone": "swin_T_224_1k",
                "position_embedding": "sine",
                "pe_temperatureH": 20,
                "pe_temperatureW": 20,
                "return_interm_indices": [1, 2, 3],
                "backbone_freeze_keywords": None,
                "enc_layers": 6,
                "dec_layers": 6,
                "pre_norm": False,
                "dim_feedforward": 2048,
                "hidden_dim": 256,
                "dropout": 0.0,
                "nheads": 8,
                "num_queries": 900,
                "query_dim": 4,
                "num_patterns": 0,
                "num_feature_levels": 4,
                "enc_n_points": 4,
                "dec_n_points": 4,
                "two_stage_type": "standard",
                "two_stage_bbox_embed_share": False,
                "two_stage_class_embed_share": False,
                "transformer_activation": "relu",
                "dec_pred_bbox_embed_share": True,
                "dn_box_noise_scale": 1.0,
                "dn_label_noise_ratio": 0.5,
                "dn_label_coef": 1.0,
                "dn_bbox_coef": 1.0,
                "embed_init_tgt": True,
                "dn_labelbook_size": 2000,
                "max_text_len": 256,
                "text_encoder_type": "bert-base-uncased",
                "use_text_enhancer": True,
                "use_fusion_layer": True,
                "use_checkpoint": True,
                "use_transformer_ckpt": True,
                "use_text_cross_attention": True,
                "text_dropout": 0.0,
                "fusion_dropout": 0.0,
                "fusion_droppath": 0.1,
                "sub_sentence_present": True,
            }
        return configs

    @staticmethod
    def _get_caption(text_prompt):
        """
        Get processed caption from the text prompt.

        Args:
            text_prompt (str): Text prompt

        Returns:
            caption (str): Processed caption
        """
        caption = text_prompt.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        return caption

    @staticmethod
    def _get_tokenizer(text_encoder_type):
        """
        Get tokenizer based on the text encoder type.

        Args:
            text_encoder_type (str): Type of text encoder

        Returns:
            tokenizer: Instance of the tokenizer
        """
        from tokenizers import Tokenizer
        current_dir = os.path.dirname(__file__)
        cfg_name = text_encoder_type.replace("-", "_") + "_tokenizer.json"
        cfg_file = os.path.join(current_dir, "configs", cfg_name)
        tokenizer = Tokenizer.from_file(cfg_file)
        return tokenizer

    @staticmethod
    def _get_phrases_from_posmap(posmap: np.ndarray,
                                 tokenized: Dict,
                                 tokenizer,
                                 left_idx: int = 0,
                                 right_idx: int = 255):
        """
        Get phrases from position map.

        Args:
            posmap: Position map
            tokenized: Tokenized input
            tokenizer: Tokenizer
            left_idx: Left index
            right_idx: Right index

        Returns:
            phrases: Phrases extracted from the position map
        """
        assert isinstance(posmap, np.ndarray), "posmap must be numpy.ndarray"
        if posmap.ndim == 1:
            posmap[0: left_idx + 1] = False
            posmap[right_idx:] = False
            non_zero_idx = np.where(posmap)[0]
            token_ids = [tokenized["input_ids"][i] for i in non_zero_idx]
            return tokenizer.decode(token_ids)
        else:
            raise NotImplementedError("posmap must be 1-dim")

    @staticmethod
    def _generate_masks_with_special_tokens_and_transfer_map(tokenized, special_tokens_list):
        """
        Generate masks with special tokens and transfer map.

        Args:
            tokenized: Tokenized input
            special_tokens_list: List of special tokens

        Returns:
            attention_mask: Attention mask
            position_ids: Position IDs
            cate_to_token_mask_list: List of category to token masks
        """
        input_ids = tokenized["input_ids"]
        bs, num_token = input_ids.shape
        # special_tokens_mask: bs, num_token.
        # 1 for special tokens. 0 for normal tokens
        special_tokens_mask = np.zeros((bs, num_token), dtype=bool)
        for special_token in special_tokens_list:
            special_tokens_mask |= input_ids == special_token

        # idxs: each row is a list of indices of special tokens
        idxs = np.argwhere(special_tokens_mask)

        # generate attention mask and positional ids
        attention_mask = np.eye(num_token, dtype=bool).reshape(
            1, num_token, num_token
        )
        attention_mask = np.tile(attention_mask, (bs, 1, 1))
        position_ids = np.zeros((bs, num_token), dtype=int)
        cate_to_token_mask_list = [[] for _ in range(bs)]
        previous_col = 0
        for i in range(idxs.shape[0]):
            row, col = idxs[i]
            if (col == 0) or (col == num_token - 1):
                attention_mask[row, col, col] = True
                position_ids[row, col] = 0
            else:
                attention_mask[
                    row, previous_col + 1: col + 1, previous_col + 1: col + 1
                ] = True
                position_ids[row, previous_col + 1: col + 1] = np.arange(
                    0, col - previous_col
                )
                c2t_maski = np.zeros((num_token), dtype=bool)
                c2t_maski[previous_col + 1: col] = True
                cate_to_token_mask_list[row].append(c2t_maski)
            previous_col = col

        cate_to_token_mask_list = [
            np.stack(cate_to_token_mask_listi, axis=0)
            for cate_to_token_mask_listi in cate_to_token_mask_list
        ]

        return attention_mask, position_ids, cate_to_token_mask_list


if __name__ == '__main__':
    model_abs_path = "groundingdino_swinb_cogcoor_quant.onnx"
    gd = GroundingDINO(model_abs_path)
    image_path = "bird1_000000002.png"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    shapes = gd.predict_shapes(image, text_prompt="bird")
    filename = image_path.replace('.png', '.json')

    save_labels(filename=filename, imagePath=image_path, label_list=shapes,
                height=height, width=width)
