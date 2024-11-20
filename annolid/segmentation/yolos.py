import os
from ultralytics import YOLO, SAM
from annolid.gui.shape import Shape
from annolid.annotation.keypoints import save_labels


class InferenceProcessor:
    def __init__(self, model_name, model_type, class_names=None):
        """
        Initializes the InferenceProcessor with a specified model.

        Args:
            model_name (str): Path or identifier for the model file.
            model_type (str): Type of model ('yolo' or 'sam').
            class_names (list, optional): List of class names for YOLO.
        """
        self.model_type = model_type
        self.model = self._load_model(model_name, class_names)
        self.frame_count = 0  # Initialize the frame counter

    def _load_model(self, model_name, class_names):
        """
        Loads the specified model based on the model type.

        Args:
            model_name (str): Path or identifier for the model file.
            class_names (list, optional): List of class names for YOLO.

        Returns:
            A YOLO or SAM model instance.
        """
        if self.model_type == 'yolo':
            model = YOLO(model_name)
            if 'world' in model_name and class_names:
                model.set_classes(class_names)
            return model
        elif self.model_type == 'sam':
            model = SAM(model_name)
            model.info()  # Optional: Display model information
            return model
        else:
            raise ValueError("Unsupported model type. Use 'yolo' or 'sam'.")

    def run_inference(self, source):
        """
        Runs inference on the specified source and saves results to LabelMe JSON.

        Args:
            source (str): Path to the video or image source.
        """
        # Ensure the output directory exists
        output_directory = os.path.splitext(source)[0]
        os.makedirs(output_directory, exist_ok=True)

        results = self.model(source, stream=True)

        # Process each frame result
        for result in results:
            frame_shape = (result.orig_shape[0], result.orig_shape[1], 3)
            id_to_labels = {0: "mouse", 1: "teaball"}  # Example label map
            yolo_results = self.extract_yolo_results(result)

            self.save_yolo_to_labelme(
                yolo_results, id_to_labels, frame_shape, output_directory
            )

    def extract_yolo_results(self, result):
        """
        Extracts YOLO results from the inference result object.

        Args:
            result: YOLO result object.

        Returns:
            A list of dictionaries containing bounding boxes and class IDs.
        """
        yolo_results = []
        for box in result.boxes:
            yolo_results.append({
                "cls": box.cls,  # Class ID
                "xyxy": box.xyxy  # Bounding box coordinates
            })
        return yolo_results

    def save_yolo_to_labelme(self, yolo_results, id_to_labels, frame_shape,
                             output_dir):
        """
        Converts YOLO results to LabelMe JSON format and saves them.

        Args:
            yolo_results (list): YOLO results containing bounding boxes and labels.
            id_to_labels (dict): Mapping of object IDs to readable labels.
            frame_shape (tuple): Shape of the frame as (height, width, channels).
            output_dir (str): Directory to save the LabelMe JSON files.
        """
        height, width, _ = frame_shape

        # Construct the JSON filename using the frame count
        json_filename = f"{self.frame_count:09d}.json"
        output_path = os.path.join(output_dir, json_filename)
        label_list = []

        for result in yolo_results:
            label_id = int(result["cls"].item())
            bbox = result["xyxy"].squeeze().tolist()
            if bbox:
                if id_to_labels is not None:
                    label = id_to_labels.get(label_id, f"class_{label_id}")
                else:
                    label = f"{label_id}"

                # Convert bounding box to a polygon
                x_min, y_min, x_max, y_max = bbox
                points = [
                    [x_min, y_min],  # Top-left
                    [x_max, y_min],  # Top-right
                    [x_max, y_max],  # Bottom-right
                    [x_min, y_max],  # Bottom-left
                ]

                # Create a MaskShape object
                shape = Shape(label=label, flags={},
                                    description="yolo_prediction")
  
                shape.points = points
                label_list.append(shape)

        save_labels(
            filename=output_path,
            imagePath="",
            label_list=label_list,
            height=height,
            width=width,
            save_image_to_json=False,
        )

        # Increment the frame counter after saving
        self.frame_count += 1


# Example usage
if __name__ == "__main__":
    video_path = "~/Downloads/mouse.mp4"

    yolo_processor = InferenceProcessor(
        "yolo11n.pt", model_type="yolo", class_names=["mouse", "teaball"]
    )
    yolo_processor.run_inference(video_path)
