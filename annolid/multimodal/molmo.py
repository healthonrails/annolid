import os
import cv2
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig
from PIL import Image
import torch
from annolid.annotation.keypoints import save_labels
from annolid.utils.files import construct_filename

# Define constants
MODEL_NAME = "allenai/Molmo-7B-D-0924"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Cache the model and processor outside the function
processor = None
model = None


def save_caption(
    filename,
    frame_shape,
    caption="",
):
    """
    Saves a caption to a JSON file in LabelMe format with basic metadata.

    Args:
        filename (str): The name of the output JSON file.
        frame_shape (tuple): The shape of the frame (height, width, channels).
        caption (str): The caption to include in the JSON file.
    """
    height, width, _ = frame_shape
    height, width, _ = frame_shape
    image_path = os.path.splitext(filename)[0] + ".png"
    label_list = []

    # Save all the labels into a LabelMe format JSON
    save_labels(
        filename=filename,
        imagePath=image_path,
        label_list=label_list,
        height=height,
        width=width,
        save_image_to_json=False,
        caption=caption,
    )

    return label_list


def load_model_and_processor():
    global processor, model  # Use global keyword to modify global variables
    if processor is None:
        processor = AutoProcessor.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto" if DEVICE == "cpu" else {"": DEVICE},
        )
    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            device_map="auto" if DEVICE == "cpu" else {"": DEVICE},
        ).to(DEVICE)
        model.eval()


# Function to describe an image


def describe_image(image, max_new_tokens=200):
    load_model_and_processor()  # Ensure model and processor are loaded
    # Use the correct .process method (not direct call) and ensure correct device placement:
    inputs = processor.process(
        images=[image],
        text="Analyze and describe the mouse’s actions, posture, and interactions in this image, detailing any observable behaviors, body language, and interactions with its surroundings.",
    )
    # Move to device *after* processing
    inputs = {k: v.to(DEVICE).unsqueeze(0) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate_from_batch(
            inputs,
            generation_config=GenerationConfig(
                max_new_tokens=max_new_tokens, stop_strings="<|endoftext|>"
            ),
            tokenizer=processor.tokenizer,
        )  # Use generate_from_batch

    # Get generated tokens
    generated_tokens = output[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens, skip_special_tokens=True
    )  # Decode using tokenizer
    return generated_text


# Main function to process video frames and describe each


def process_video(video_path, sample_rate=1):  # Reduced default sample rate for testing
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video: unable to read {video_path}.")
        return []

    descriptions = []
    video_dir = os.path.splitext(video_path)[0]

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if sample_rate > 1 and (frame_index % sample_rate) != 0:
            frame_index += 1
            continue

        try:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_shape = frame_rgb.shape
            filename = construct_filename(
                video_dir, frame_index, extension=".json", padding=9
            )
            pil_image = Image.fromarray(frame_rgb)
            description = describe_image(pil_image)
            save_caption(filename, frame_shape, description)
            descriptions.append((frame_index, description))
            print(f"Frame {frame_index}: {description}")

        except Exception as e:  # Catch errors during processing, continue to next frame
            print(f"Error processing frame {frame_index}: {e}")

        frame_index += 1

    cap.release()
    return descriptions


if __name__ == "__main__":
    video_path = "mouse.mp4"  # Replace with your video path
    descriptions = process_video(video_path)
    print(descriptions)
