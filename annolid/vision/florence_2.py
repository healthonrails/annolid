import os
import cv2
import torch
from PIL import Image
from annolid.annotation.keypoints import save_labels
from annolid.gui.shape import Shape
from transformers import AutoProcessor, AutoModelForCausalLM

# Setup device and dtype


def get_device_and_dtype():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float32
    return device, torch_dtype


def save_annotations(filename,
                     mask_dict,
                     frame_shape,
                     caption=None,
                     ):
    """
    Saves annotations to a JSON file in LabelMe format.

    Args:
        filename (str): The name of the output JSON file.
        mask_dict (dict): A dictionary containing labels and polygon coordinates.
        frame_shape (tuple): The shape of the frame (height, width, channels).

    Returns:
        list: List of MaskShape objects.
    """
    height, width, _ = frame_shape
    image_path = os.path.splitext(filename)[0] + '.png'
    label_list = []

    # Iterate through the dictionary which contains polygons and labels
    for label, polygons in zip(mask_dict['labels'], mask_dict['polygons']):
        for polygon in polygons:
            # Ensure the polygon has at least three points (a valid polygon)
            if len(polygon) < 3:
                print(
                    f"Skipping invalid polygon with {len(polygon)} points: {polygon}")
                continue

            # Convert Florence's polygon coordinates into points format
            points = [[x, y] for x, y in polygon]

            # Create a MaskShape object for each set of points
            current_shape = Shape(
                label=label,  # Use the detected object or default label
                description='florence',
                flags={},
            )
            current_shape.points = points

            # Append the shape to the label list
            label_list.append(current_shape)

    # Save all the labels into a LabelMe format JSON
    save_labels(filename=filename, imagePath=image_path, label_list=label_list,
                height=height, width=width, save_image_to_json=False, caption=caption)

    return label_list


def convert_to_mask_dict(results, text_prompt="<REFERRING_EXPRESSION_SEGMENTATION>", text_input=None):
    """
    Converts the model results into a valid mask dictionary that can be used for saving annotations.

    Args:
        results (dict): The output from the model containing polygons and labels.
        text_prompt (str): The specific task prompt used for extracting results.
        text_input (str): The fallback label to use if no label is detected by the model.

    Returns:
        dict: A dictionary with 'polygons' and 'labels' keys.
    """
    polygons = results[text_prompt]['polygons']
    labels = results[text_prompt]['labels']

    mask_dict = {
        'polygons': [],
        'labels': []
    }

    for i, polygon_set in enumerate(polygons):
        # Use the detected label or fallback to text_input if label is missing or empty
        label = labels[i] if labels[i] else text_input

        # Convert each polygon's coordinates to (x, y) pairs
        formatted_polygons = []
        for polygon in polygon_set:
            if len(polygon) % 2 == 0:
                formatted_polygon = [[polygon[j], polygon[j+1]]
                                     for j in range(0, len(polygon), 2)]
                formatted_polygons.append(formatted_polygon)

        if formatted_polygons:
            mask_dict['polygons'].append(formatted_polygons)
            mask_dict['labels'].append(label)

    return mask_dict

# Load the model and processor


def load_model_and_processor(model_name: str, device, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True)
    return model, processor


def florence2(processor, model, task_prompt, image, text_input=None):
    """
    Calling the Microsoft Florence2 model
    """
    if text_input is None:
        prompt = task_prompt
    else:
        prompt = task_prompt + text_input

    inputs = processor(text=prompt, images=image, return_tensors="pt")

    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=1024,
        early_stopping=False,
        do_sample=False,
        num_beams=3,
    )
    generated_text = processor.batch_decode(generated_ids,
                                            skip_special_tokens=False)[0]
    parsed_answer = processor.post_process_generation(
        generated_text,
        task=task_prompt,
        image_size=(image.width, image.height))

    return parsed_answer

# Stream video and process every nth frame


def process_nth_frame_from_video(video_path: str, n: int, model,
                                 processor, prompt: str,
                                 task: str, text_input: str):
    video_dir = os.path.splitext(video_path)[0]
    # Create the directory if it doesn't exist
    os.makedirs(video_dir, exist_ok=True)
    video_capture = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video_capture.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_count = 0

    while True:
        success, frame = video_capture.read()

        if not success:
            break  # Exit the loop if there are no more frames

        frame_shape = frame.shape
        filename = os.path.join(video_dir, f'{frame_count:09}.json')

        # Process every nth frame
        if frame_count % n == 0:
            # Convert the frame (OpenCV format) to PIL Image format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)
            more_detailed_caption_task = '<MORE_DETAILED_CAPTION>'

            parsed_answer = florence2(
                processor, model, more_detailed_caption_task, image)

            # Extract the more detailed caption (from the '<MORE_DETAILED_CAPTION>' task)
            detailed_caption = parsed_answer.get(
                more_detailed_caption_task, '')
            parsed_answer = florence2(
                processor, model, prompt, image, text_input)

            # Convert polygons and labels
            mask_dict = convert_to_mask_dict(
                parsed_answer, text_input=text_input)

            # Save annotations with the detailed caption in the flags
            save_annotations(filename, mask_dict, frame_shape,
                             caption=detailed_caption)

        frame_count += 1

    video_capture.release()


# Main function to run the prediction
def run_prediction(model_name: str, video_path: str,
                   n: int,
                   prompt: str, text_input: str, task=None):
    # Get device and dtype
    device, torch_dtype = get_device_and_dtype()

    # Load model and processor
    model, processor = load_model_and_processor(
        model_name, device, torch_dtype)

    # Stream video and process every nth frame
    process_nth_frame_from_video(
        video_path, n, model, processor, prompt, task, text_input)


if __name__ == "__main__":
    model_name = "microsoft/Florence-2-large"
    video_path = os.path.expanduser(
        "~/Downloads/mouse.mp4")  # Specify the video path
    n = 10  # Example: Process every 10th frame
    text_input = 'a black mouse'
    task = "<REFERRING_EXPRESSION_SEGMENTATION>"
    prompt = task

    # Run the prediction, passing text_input as fallback for labels
    run_prediction(model_name, video_path, n, prompt, text_input, task)
