import os
import cv2
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

# Setup device and dtype


def get_device_and_dtype():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    return device, torch_dtype

# Load the model and processor


def load_model_and_processor(model_name: str, device, torch_dtype):
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=True)
    return model, processor

# Stream video and process every nth frame


def process_nth_frame_from_video(video_path: str, n: int, model,
                                 processor, prompt: str,
                                 task: str = "<OD>"):
    video_capture = cv2.VideoCapture(video_path)

    # Check if video opened successfully
    if not video_capture.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    frame_count = 0

    while True:
        success, frame = video_capture.read()
        if not success:
            break  # Exit the loop if there are no more frames

        # Process every nth frame
        if frame_count % n == 0:
            # Convert the frame (OpenCV format) to PIL Image format
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame_rgb)

            # Prepare inputs
            inputs = processor(text=prompt, images=image,
                               return_tensors="pt").to(model.device)

            # Generate text
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                num_beams=3,
                do_sample=False
            )

            # Decode and post-process
            parsed_answer = processor.batch_decode(
                generated_ids, skip_special_tokens=False)[0]
            parsed_answer = processor.post_process_generation(
                parsed_answer, task=task, image_size=(image.width, image.height))

            print(parsed_answer)  # Print or log the generated text

        frame_count += 1

    video_capture.release()

# Main function to run the prediction


def run_prediction(model_name: str, video_path: str, n: int, prompt: str):
    # Get device and dtype
    device, torch_dtype = get_device_and_dtype()

    # Load model and processor
    model, processor = load_model_and_processor(
        model_name, device, torch_dtype)

    # Stream video and process every nth frame
    process_nth_frame_from_video(video_path, n, model, processor, prompt)


if __name__ == "__main__":
    model_name = "microsoft/Florence-2-large"
    video_path = os.path.expanduser(
        "~/Downloads/mouse.mp4")  # Specify the video path
    n = 10  # Example: Process every 10th frame
    prompt = "<OD>"

    # Run the prediction
    run_prediction(model_name, video_path, n, prompt)
