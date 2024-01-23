import cv2
import numpy as np
import onnxruntime
import matplotlib.pyplot as plt


class ONNXInference:
    def __init__(self, encoder_model_path, decoder_model_path):
        self.target_size = 1024
        self.input_size = (684, 1024)
        self.initialize_sessions(encoder_model_path, decoder_model_path)

    def initialize_sessions(self, encoder_model_path, decoder_model_path):
        providers = [p for p in onnxruntime.get_available_providers(
        ) if p != "TensorrtExecutionProvider"]
        self.encoder_session = onnxruntime.InferenceSession(
            encoder_model_path, providers=providers)
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.decoder_session = onnxruntime.InferenceSession(
            decoder_model_path, providers=providers)

    def get_input_points(self, prompt):
        points, labels = [], []
        for mark in prompt:
            if mark["type"] == "point":
                points.append(mark["data"])
                labels.append(mark["label"])
            elif mark["type"] == "rectangle":
                points.extend([mark["data"][:2], mark["data"][2:]])
                labels.extend([2, 3])
        points, labels = np.array(points), np.array(labels)
        return points, labels

    def run_encoder(self, encoder_inputs):
        features = self.encoder_session.run(None, encoder_inputs)
        image_embeddings, interm_embeddings = features[0], np.stack(
            features[1:])
        return image_embeddings, interm_embeddings

    def get_preprocess_shape(self, oldh, oldw, long_side_length):
        scale = long_side_length / max(oldh, oldw)
        newh, neww = int(oldh * scale + 0.5), int(oldw * scale + 0.5)
        return newh, neww

    def apply_coords(self, coords, original_size, target_length):
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(old_h, old_w, target_length)
        coords = coords.astype(float)
        coords[..., 0] *= new_w / old_w
        coords[..., 1] *= new_h / old_h
        return coords

    def run_decoder(self, image_embeddings, interm_embeddings,
                    original_size, transform_matrix, prompt):
        input_points, input_labels = self.get_input_points(prompt)
        onnx_coord = np.concatenate(
            [input_points, np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate(
            [input_labels, np.array([-1])], axis=0)[None, :].astype(np.float32)
        onnx_coord = self.apply_coords(
            onnx_coord, self.input_size, self.target_size).astype(np.float32)
        onnx_coord = np.concatenate([onnx_coord, np.ones(
            (1, onnx_coord.shape[1], 1), dtype=np.float32)], axis=2)
        onnx_coord = np.matmul(onnx_coord, transform_matrix.T)
        onnx_coord = onnx_coord[:, :, :2].astype(np.float32)
        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)
        decoder_inputs = {
            "image_embeddings": image_embeddings,
            "interm_embeddings": interm_embeddings,
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input,
            "orig_im_size": np.array(self.input_size, dtype=np.float32),
        }
        masks, _, _ = self.decoder_session.run(None, decoder_inputs)
        inv_transform_matrix = np.linalg.inv(transform_matrix)
        transformed_masks = self.transform_masks(
            masks, original_size, inv_transform_matrix)
        return transformed_masks

    def transform_masks(self, masks, original_size, transform_matrix):
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = [cv2.warpAffine(mask, transform_matrix[:2],
                                          (original_size[1], original_size[0]),
                                          flags=cv2.INTER_LINEAR) for mask in masks[batch]]
            output_masks.append(batch_masks)
        return np.array(output_masks)

    def encode(self, cv_image):
        original_size = cv_image.shape[:2]
        scale_x = self.input_size[1] / cv_image.shape[1]
        scale_y = self.input_size[0] / cv_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
        cv_image = cv2.warpAffine(cv_image, transform_matrix[:2],
                                  (self.input_size[1], self.input_size[0]),
                                  flags=cv2.INTER_LINEAR)
        encoder_inputs = {self.encoder_input_name: cv_image.astype(np.float32)}
        image_embeddings, interm_embeddings = self.run_encoder(encoder_inputs)
        return {
            "image_embeddings": image_embeddings,
            "interm_embeddings": interm_embeddings,
            "original_size": original_size,
            "transform_matrix": transform_matrix,
        }

    def predict_masks(self, embedding, prompt):
        masks = self.run_decoder(embedding["image_embeddings"],
                                 embedding["interm_embeddings"],
                                 embedding["original_size"],
                                 embedding["transform_matrix"], prompt)
        return masks

    def post_process(self, masks):
        masks[masks > 0.0] = 255
        masks[masks <= 0.0] = 0
        masks = masks.astype(np.uint8)
        contours, _ = cv2.findContours(
            masks, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        approx_contours = [cv2.approxPolyDP(
            contour, 0.001 * cv2.arcLength(contour, True), True)
            for contour in contours]

        polygon_points_list = [
            approx.reshape(-1, 2).tolist()
            for approx in approx_contours if len(approx) >= 3]

        return polygon_points_list


class ONNXInferenceWrapper:
    def __init__(self, encoder_model_path, decoder_model_path):
        self.model = ONNXInference(encoder_model_path, decoder_model_path)

    def infer(self, image_path, prompt):
        try:
            cv_image = cv2.imread(image_path)
            embedding = self.model.encode(cv_image)
            masks = self.model.predict_masks(embedding, prompt)
            masks = masks.astype(np.uint8)
            return masks
        except Exception as e:
            print(f"Error during inference: {e}")
            return None


def show_mask(mask, ax, random_color=False):
    color = np.random.random(3) if random_color else np.array(
        [30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red',
               marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


if __name__ == "__main__":
    encoder_model_path = "sam_hq_vit_l_encoder_quant.onnx"
    decoder_model_path = "sam_hq_vit_l_decoder.onnx"
    onnx_wrapper = ONNXInferenceWrapper(encoder_model_path, decoder_model_path)
    prompt = [{"type": "point", "data": [965, 351], "label": 1}]
    image_path = "bird1_000000002.png"
    result_masks = onnx_wrapper.infer(image_path, prompt)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    input_point = np.array([[965, 351]])
    input_label = np.array([1])

    show_points(input_point, input_label, plt.gca())
    show_mask(result_masks, plt.gca())
    plt.axis('off')
    plt.show()
