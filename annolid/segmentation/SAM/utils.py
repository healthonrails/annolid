import numpy as np
import cv2
import torch
import pycocotools.mask as mask_util
from PIL import Image


def get_mask_features(image, mask, model, preprocess=None):
    """
    Computes the features of the mask portion of an image.

    Args:
        image (ndarray): The input image.
        mask (ndarray): The mask for cropping the image.
        model: The CLIP model used for encoding image features.

    Returns:
        mask_features (Tensor): The features of the mask portion of the image.
    """
    # Apply the mask to the image
    masked_image = image.copy()
    masked_image[~mask] = 0

    # Preprocess the masked image
    masked_image = preprocess(Image.fromarray(masked_image))

    # Convert the image to tensor and move to GPU
    # image_input = torch.tensor(masked_image).unsqueeze(0).cuda()
    image_input = masked_image.unsqueeze(0).cuda().float()

    with torch.no_grad():
        # Encode image features
        image_features = model.encode_image(image_input).float()

    return image_features.detach().cpu().numpy()


def generate_mask_id(mask_features, existing_masks, threshold=6.0, distance_metric="euclidean"):
    """
    Generates an ID for the mask based on its features and compares it with existing masks.

    Args:
        mask_features (ndarray): The features of the mask.
        existing_masks (list): List of existing masks and their features.
        threshold (float): Similarity threshold for considering a match (default: 0.9).
        distance_metric (str): Distance metric to be used (default: "euclidean").
                               Options: "euclidean", "cosine".

    Returns:
        mask_id (int): The generated ID for the mask.
    """
    from scipy.spatial.distance import euclidean, cosine

    mask_id = -1  # Initialize the mask ID

    if distance_metric == "euclidean":
        distance_function = euclidean
    elif distance_metric == "cosine":
        distance_function = cosine
    else:
        raise ValueError(
            "Invalid distance metric. Choose either 'euclidean' or 'cosine'.")

    for idx, (existing_id, existing_features) in enumerate(existing_masks):
        similarity = distance_function(
            mask_features.flatten(), existing_features.flatten())

        if similarity < threshold:
            mask_id = existing_id
            break

    if mask_id == -1:
        # Assign a new ID if no match is found
        mask_id = len(existing_masks) + 1
        existing_masks.append((mask_id, mask_features.flatten()))

    return mask_id


def convert_to_annolid_format(frame_number,
                              masks,
                              frame=None,
                              model=None,
                              min_mask_area=float('-inf'),
                              max_mask_area=float('inf'),
                              existing_masks=None
                              ):
    """Converts predicted SAM masks information to annolid format.

    Args:
        frame_number (int): The frame number associated with the masks.
        masks (list): List of dictionaries representing the predicted masks.
            Each dictionary should contain the following keys:
                -segmentation : the mask
                -area : the area of the mask in pixels
                -bbox : the boundary box of the mask in XYWH format
                -predicted_iou : the model's own prediction for the quality of the mask
                -point_coords : the sampled input point that generated this mask
                -stability_score : an additional measure of mask quality
                -crop_box : the crop of the image used to generate this mask in XYWH format

    Returns:
        list: List of dictionaries representing the masks in annolid format.
            Each dictionary contains the following keys:
                - "frame_number": The frame number associated with the masks.
                - "x1", "y1", "x2", "y2": The coordinates of the bounding box in XYXY format.
                - "instance_name": The name of the instance/object.
                - "class_score": The predicted IoU (Intersection over Union) for the mask.
                - "segmentation": The segmentation mask.
                - "tracking_id": The tracking ID associated with the mask.

    """
    pred_rows = []
    for mask in masks:
        mask_area = mask.get("area", 0)
        if min_mask_area <= mask_area <= max_mask_area:
            x1 = mask.get("bbox")[0]
            y1 = mask.get("bbox")[1]
            x2 = mask.get("bbox")[0] + mask.get("bbox")[2]
            y2 = mask.get("bbox")[1] + mask.get("bbox")[3]
            score = mask.get("predicted_iou", '')
            segmentation = mask.get("segmentation", '')
            mask_features = get_mask_features(frame, segmentation, model)
            mask_id = generate_mask_id(mask_features, existing_masks)
            instance_name = mask.get("instance_name", f'instance_{mask_id}')
            segmentation = mask_util.encode(segmentation)
            tracking_id = mask.get("tracking_id", "")

            pred_rows.append({
                "frame_number": frame_number,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "instance_name": instance_name,
                "class_score": score,
                "segmentation": segmentation,
                "tracking_id": tracking_id
            })

    return pred_rows


def crop_image_with_masks(image,
                          masks,
                          max_area=8000,
                          min_area=500,
                          width_height_ratio=0.9):
    """
    Crop the image based on provided masks and apply the masks to each cropped region.

    Args:
        image (numpy.ndarray): The input image.
        masks (list): A list of dictionaries containing mask data.
        max_area (int): Max area of the mask
        min_area (int): Min area of the mask
        width_height_ratio(float): Min width / height

    Returns:
        list: A list of cropped images with applied masks.
    """
    cropped_images = []

    for mask_data in masks:
        # Extract mask and bounding box data
        bbox = mask_data['bbox']
        seg = mask_data['segmentation']
        x, y, w, h = bbox

        # Crop the image based on the bounding box
        cropped_image = image[y:y+h, x:x+w]

        # Create an 8-bit mask from the segmentation data
        mask = np.asarray(seg[y:y+h, x:x+w], dtype=np.uint8) * 255
        # Apply the mask to the cropped image
        cropped_image = cv2.bitwise_and(
            cropped_image, cropped_image, mask=mask)
        cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)
        if (mask_data['area'] >= min_area and
            mask_data['area'] <= max_area and
                w/h >= width_height_ratio):
            cropped_images.append(cropped_image)

    return cropped_images


def process_video_and_save_tracking_results(video_file, mask_generator, model=None):
    """
    Process a video file, generate tracking results with segmentation masks,
    and save the results to a CSV file.

    Args:
        video_file (str): Path to the video file.
        mask_generator: An instance of the mask generator class.

    Returns:
        None
    """
    import decord as de
    import pandas as pd
    video_reader = de.VideoReader(video_file)
    tracking_results = []
    existing_masks = []

    for key_index in video_reader.get_key_indices():
        frame = video_reader[key_index].asnumpy()
        masks = mask_generator.generate(frame)
        tracking_results += convert_to_annolid_format(
            key_index, masks, frame, model, existing_masks=existing_masks)
        print(key_index)

    dataframe = pd.DataFrame(tracking_results)
    output_file = f"{video_file.split('.')[0]}_mask_tracking_results_with_segmentation.csv"
    dataframe.to_csv(output_file)
