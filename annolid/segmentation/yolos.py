import os
import numpy as np
from ultralytics import YOLO, SAM
from annolid.gui.shape import Shape
from pathlib import Path
from annolid.annotation.keypoints import save_labels
from collections import defaultdict
from annolid.annotation.polygons import simplify_polygons


class InferenceProcessor:
    def __init__(self, model_name, model_type, class_names=None):
        """
        Initializes the InferenceProcessor with a specified model.

        Args:
            model_name (str): Path or identifier for the model file.
            model_type (str): Type of model ('yolo' or 'sam').
            class_names (list, optional): List of class names for YOLO. 
                                          Defaults to None. Only provide 
                                          if the model doesn't have classes
                                          built-in and you need to set them.
        """
        self.model_type = model_type
        model_name = self._find_best_model(model_name)
        self.model = self._load_model(model_name, class_names)
        self.frame_count = 0
        self.track_history = defaultdict(lambda: [])
        self.keypoint_names = None
        if 'pose' in model_name:
            from annolid.utils.config import get_config
            _here = os.path.dirname(__file__)
            cfg_folder = Path(_here).parent
            keypoint_config_file = cfg_folder / 'configs' / 'keypoints.yaml'
            keypoint_config = get_config(keypoint_config_file)
            self.keypoint_names = keypoint_config['KEYPOINTS'].split(" ")
            print(f"Keypoint names: {self.keypoint_names}")

    def _find_best_model(self, model_name):
        """
        Searches for 'best.pt' in potential directories and returns its path.
        If not found, uses a default model.
        """
        search_paths = [
            os.path.expanduser("~/Downloads/best.pt"),
            os.path.expanduser(
                "~/Downloads/runs/segment/train/weights/best.pt"),
            os.path.expanduser("~/Downloads/segment/train/weights/best.pt"),
            "runs/segment/train/weights/best.pt",
            "segment/train/weights/best.pt"
        ]
        for path in search_paths:
            if os.path.isfile(path):
                print(f"Found model: {path}")
                return path
        print("best.pt not found, using default model")
        return model_name

    def _load_model(self, model_name, class_names):
        """Loads the specified model."""
        if self.model_type == 'yolo':
            model = YOLO(model_name)
            if class_names:  # Only set classes if provided
                model.set_classes(class_names)
            return model
        elif self.model_type == 'sam':
            model = SAM(model_name)
            model.info()
            return model
        else:
            raise ValueError("Unsupported model type. Use 'yolo' or 'sam'.")

    def run_inference(self, source):
        """Runs inference and saves results to LabelMe JSON."""
        output_directory = os.path.splitext(source)[0]
        os.makedirs(output_directory, exist_ok=True)

        results = self.model.track(source, persist=True, stream=True)

        for result in results:  # Corrected: Iterate through results generator
            # Check if boxes exist
            if result.boxes is not None and len(result.boxes):
                frame_shape = (result.orig_shape[0], result.orig_shape[1], 3)
                yolo_results = self.extract_yolo_results(result)
                self.save_yolo_to_labelme(
                    yolo_results, frame_shape, output_directory)
            else:
                self.frame_count += 1

        return f"Done#{self.frame_count}"

    def extract_yolo_results(self, result,
                             save_bbox=False,
                             save_track=False):
        """Extracts YOLO results, emulating boxes if none are found."""
        yolo_results = []

        # Emulate boxes if none found, otherwise use actual boxes
        if not result.boxes:
            return yolo_results
        else:
            boxes = result.boxes.xywh.cpu()
            track_ids = result.boxes.id.int().cpu().tolist() if result.boxes.id is not None else [
                "" for _ in range(len(boxes))]  # Check for track_ids
            masks = result.masks
            names = result.names
            keypoints = result.keypoints
            cls_ids = [int(_box.cls) for _box in result.boxes]
            confidences = result.boxes.conf.cpu().tolist() if result.boxes.conf is not None else [
                0.0 for _ in range(len(boxes))]  # Check for confidences

        if keypoints is not None:
            for _keypoints in enumerate(keypoints.xy):
                _keypoint_points = _keypoints[1].cpu().tolist()
                for kpt_id, kpt in enumerate(_keypoint_points):
                    kpt_label = f"{kpt_id}" if self.keypoint_names is None else self.keypoint_names[
                        kpt_id]
                    keypoint_shape = Shape(kpt_label,
                                           shape_type='point',
                                           description=self.model_type,
                                           flags={},
                                           )
                    keypoint_shape.points = [kpt]
                    yolo_results.append(keypoint_shape)

        for _id, box in enumerate(boxes):
            x, y, w, h = box.tolist()
            class_name = names[cls_ids[_id]]
            mask = masks[_id] if masks is not None else None

            if save_track:
                track_id = track_ids[_id]
                # Get the track history (will be empty if track_id is "")
                track = self.track_history[track_id]
                # Store only if track_id is not empty
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

            if save_bbox:
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2

                box_label = f"{class_name}"
                box_points = [[x1, y1], [x2, y2]]
                bbox_shape = Shape(box_label, shape_type='rectangle',
                                   description=self.model_type,
                                   flags={},
                                   )
                bbox_shape.points = box_points
                yolo_results.append(bbox_shape)

            # Only create track polygon if history exists and track_id is valid.
            if save_track and len(track) > 1 and track_id != "":
                track_points = np.array(track).tolist()
                shape_track = Shape(f"track_{track_id}",
                                    shape_type="polygon",
                                    description=self.model_type,
                                    flags={},
                                    visible=True,
                                    )
                shape_track.points = track_points
                yolo_results.append(shape_track)

            if mask is not None:
                try:
                    polygons = simplify_polygons(mask.xy)
                    for polygon in polygons:
                        contour_points = polygon.tolist()
                        if len(contour_points) > 2:
                            seg_label = f"{class_name}"
                            segmentation_shape = Shape(
                                seg_label,
                                shape_type='polygon',
                                description=self.model_type,
                                flags={},
                                visible=True,
                            )
                            segmentation_shape.points = contour_points
                            yolo_results.append(segmentation_shape)
                except Exception as e:
                    print(f"Error processing mask: {e}")

        return yolo_results

    def save_yolo_to_labelme(self, yolo_results, frame_shape, output_dir):
        """Saves YOLO results to LabelMe JSON."""
        height, width, _ = frame_shape
        json_filename = f"{self.frame_count:09d}.json"
        output_path = os.path.join(output_dir, json_filename)
        label_list = yolo_results  # Directly use yolo_results

        save_labels(
            filename=output_path,
            imagePath="",
            label_list=label_list,
            height=height,
            width=width,
            save_image_to_json=False,
        )
        self.frame_count += 1


if __name__ == "__main__":
    # Replace with your video path
    video_path = os.path.expanduser("~/Downloads/people-detection.mp4")

    # Automatically find best.pt or use default
    yolo_processor = InferenceProcessor("yolo11n-seg.pt", model_type="yolo")
    yolo_processor.run_inference(video_path)
