from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from ultralytics import SAM, YOLO, YOLOE

from annolid.annotation.keypoints import save_labels
from annolid.annotation.polygons import simplify_polygons
from annolid.gui.shape import Shape
from annolid.utils.logger import logger
from annolid.yolo import configure_ultralytics_cache, resolve_weight_path


class InferenceProcessor:
    def __init__(self, model_name: str, model_type: str, class_names: Optional[list] = None) -> None:
        """
        Initializes the InferenceProcessor with a specified model.

        Args:
            model_name (str): Path or identifier for the model file.
            model_type (str): Type of model ('yolo' or 'sam').
            class_names (list, optional): List of class names for YOLO. 
                                          Only provided if the model doesn't have
                                          built-in classes.
        """
        self.model_type = (model_type or "").strip().lower()
        configure_ultralytics_cache()
        self.model_path = resolve_weight_path(model_name)
        self.model_name = str(self.model_path)
        self._is_coreml = self._detect_coreml_export(self.model_name)
        self.model = self._load_model(class_names)
        self.frame_count: int = 0
        self.track_history = defaultdict(list)
        self.keypoint_names = None

        # Load keypoint names if the model is for pose detection
        if "pose" in self.model_name.lower():
            from annolid.utils.config import get_config
            cfg_folder = Path(__file__).resolve().parent.parent
            keypoint_config_file = cfg_folder / 'configs' / 'keypoints.yaml'
            keypoint_config = get_config(keypoint_config_file)
            self.keypoint_names = keypoint_config['KEYPOINTS'].split(" ")
            logger.info("Keypoint names: %s", self.keypoint_names)

    @staticmethod
    def _detect_coreml_export(model_name: str) -> bool:
        """
        Detect whether the provided model reference points to a CoreML export.
        """
        path = Path(model_name)
        suffix = path.suffix.lower()
        if suffix == ".mlpackage":
            return True
        # Handle cases where the path might not yet exist on disk but still ends with .mlpackage
        return str(model_name).lower().endswith(".mlpackage")

    def _load_model(self, class_names: Optional[list] = None):
        """
        Loads the specified model based on the model_type.

        Returns:
            The loaded model instance.
        """
        filtered_classes = []
        if class_names:
            filtered_classes = [
                cls for cls in class_names if isinstance(cls, str) and cls.strip()]

        model_ref = str(self.model_path)
        model_name_lower = model_ref.lower()

        if self.model_type == "yolo":
            if "yoloe" in model_name_lower:
                model = YOLOE(model_ref)
                if filtered_classes:
                    model.set_classes(
                        filtered_classes, model.get_text_pe(filtered_classes))
            else:
                model = YOLO(model_ref)
                if filtered_classes and "pose" not in model_name_lower and "seg" not in model_name_lower:
                    if hasattr(model, "set_classes"):
                        model.set_classes(filtered_classes)
                    else:
                        logger.warning(
                            "Custom class assignment requested, but model '%s' does not support set_classes.",
                            model_ref)
            return model
        if self.model_type == "sam":
            model = SAM(model_ref)
            model.info()
            return model

        raise ValueError("Unsupported model type. Use 'yolo' or 'sam'.")

    def _validate_visual_prompts(self, visual_prompts: dict) -> bool:
        required_keys = {"bboxes", "cls"}
        if not required_keys.issubset(visual_prompts.keys()):
            logger.error(
                "Visual prompts must contain keys: %s", required_keys)
            return False

        bboxes = visual_prompts["bboxes"]
        cls = visual_prompts["cls"]
        if not isinstance(bboxes, np.ndarray) or not isinstance(cls, np.ndarray):
            logger.error(
                "Both 'bboxes' and 'cls' must be numpy arrays after normalization.")
            return False

        if bboxes.shape[0] != cls.shape[0]:
            logger.error("Mismatch: %d bboxes vs %d classes.",
                         bboxes.shape[0], cls.shape[0])
            return False

        return True

    def run_inference(
        self,
        source: str,
        visual_prompts: dict = None,
        *,
        pred_worker=None,
        stop_event=None,
    ) -> str:
        """
        Runs inference on the given video source and saves the results as LabelMe JSON files.

        Args:
            source (str): Path to the video file.

        Returns:
            A string message indicating the completion and frame count.
        """
        output_directory = Path(source).with_suffix("")
        output_directory.mkdir(parents=True, exist_ok=True)

        def should_stop() -> bool:
            try:
                if stop_event is not None and stop_event.is_set():
                    return True
            except Exception:
                pass
            try:
                if pred_worker is not None and hasattr(pred_worker, "is_stopped") and pred_worker.is_stopped():
                    return True
            except Exception:
                pass
            try:
                from qtpy import QtCore  # Optional dependency for GUI runs

                thread = QtCore.QThread.currentThread()
                if thread is not None and thread.isInterruptionRequested():
                    return True
            except Exception:
                pass
            return False

        if visual_prompts is not None:
            try:
                bboxes = np.asarray(visual_prompts.get(
                    "bboxes", []), dtype=float)
                cls = np.asarray(visual_prompts.get("cls", []), dtype=int)
                visual_prompts = {"bboxes": bboxes, "cls": cls}
            except Exception as exc:
                logger.error(
                    "Failed to normalize visual prompts; proceeding without them: %s", exc)
                visual_prompts = None

        if visual_prompts is not None and not self._validate_visual_prompts(visual_prompts):
            logger.error("Invalid visual prompts; proceeding without them.")
            visual_prompts = None

        # Use visual prompts if supported by the model (YOLOE)
        if visual_prompts is not None and 'yoloe' in self.model_name.lower():
            try:
                from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor
                logger.info("Running prediction with visual prompts.")
                results = self.model.predict(
                    source,
                    visual_prompts=visual_prompts,
                    predictor=YOLOEVPSegPredictor,
                    stream=True,
                )
            except Exception as e:
                logger.error("Error during visual prompt prediction: %s", e)
                return f"Error: {e}"
        else:
            if self._is_coreml:
                logger.info(
                    "Detected CoreML export; using predict() instead of track().")
                results = self.model.predict(source, stream=True)
            else:
                results = self.model.track(source, persist=True, stream=True)

        stopped = False
        try:
            for result in results:
                if should_stop():
                    stopped = True
                    break
                if result.boxes and len(result.boxes) > 0:
                    frame_shape = (
                        result.orig_shape[0], result.orig_shape[1], 3)
                    annotations = self.extract_yolo_results(result)
                    self.save_yolo_to_labelme(
                        annotations, frame_shape, output_directory)
                else:
                    self.frame_count += 1
        finally:
            try:
                close = getattr(results, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass

        if stopped:
            return f"Stopped#{self.frame_count}"
        return f"Done#{self.frame_count}"

    def extract_yolo_results(self, detection_result, save_bbox: bool = False, save_track: bool = False) -> list:
        """
        Extracts YOLO results from a single detection result, returning a list of Shape objects.

        Args:
            detection_result: The inference result containing detection data.
            save_bbox (bool): Whether to save bounding boxes.
            save_track (bool): Whether to save tracking history.

        Returns:
            list: A list of Shape objects representing the detections.
        """
        annotations = []

        boxes = detection_result.boxes.xywh.cpu() if detection_result.boxes else []
        track_ids = (detection_result.boxes.id.int().cpu().tolist() if detection_result.boxes
                     and detection_result.boxes.id is not None else ["" for _ in range(len(boxes))])
        masks = detection_result.masks if hasattr(
            detection_result, 'masks') else None
        names = detection_result.names
        cls_ids = [int(box.cls)
                   for box in detection_result.boxes] if detection_result.boxes else []
        keypoints = detection_result.keypoints if hasattr(
            detection_result, 'keypoints') else None

        # Process keypoints if available
        if keypoints is not None:
            for idx, kp in enumerate(keypoints.xy):
                kpt_points = kp.cpu().tolist()
                for kpt_id, kpt in enumerate(kpt_points):
                    kpt_label = str(
                        kpt_id) if self.keypoint_names is None else self.keypoint_names[kpt_id]
                    keypoint_shape = Shape(
                        kpt_label, shape_type='point', description=self.model_type, flags={})
                    keypoint_shape.points = [kpt]
                    annotations.append(keypoint_shape)

        # Process each detection box
        for idx, box in enumerate(boxes):
            x, y, w, h = box.tolist()
            class_name = names[cls_ids[idx]]
            mask = masks[idx] if masks is not None else None
            track_id = track_ids[idx] if save_track else None

            # Update tracking history if enabled
            if save_track and track_id is not None:
                track = self.track_history[track_id]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)

            # Save bounding box
            if save_bbox:
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2
                bbox_shape = Shape(
                    class_name, shape_type='rectangle', description=self.model_type, flags={})
                bbox_shape.points = [[x1, y1], [x2, y2]]
                annotations.append(bbox_shape)

            # Save track polygon if tracking data exists
            if save_track and track_id and len(self.track_history[track_id]) > 1:
                shape_track = Shape(f"track_{track_id}", shape_type="polygon",
                                    description=self.model_type, flags={}, visible=True)
                shape_track.points = np.array(
                    self.track_history[track_id]).tolist()
                annotations.append(shape_track)

            # Process mask if available
            if mask is not None:
                try:
                    polygons = simplify_polygons(mask.xy)
                    for polygon in polygons:
                        contour_points = polygon.tolist()
                        if len(contour_points) > 2:
                            segmentation_shape = Shape(class_name, shape_type='polygon',
                                                       description=self.model_type, flags={}, visible=True)
                            segmentation_shape.points = contour_points
                            annotations.append(segmentation_shape)
                except Exception as e:
                    logger.error("Error processing mask: %s", e)

        return annotations

    def save_yolo_to_labelme(self, annotations: list, frame_shape: tuple, output_dir: Path) -> None:
        """
        Saves YOLO annotations to a LabelMe JSON file.

        Args:
            annotations (list): List of Shape objects.
            frame_shape (tuple): Tuple containing (height, width, channels).
            output_dir (Path): Output directory where JSON will be saved.
        """
        height, width, _ = frame_shape
        json_filename = f"{self.frame_count:09d}.json"
        output_path = output_dir / json_filename
        save_labels(
            filename=str(output_path),
            imagePath="",
            label_list=annotations,
            height=height,
            width=width,
            save_image_to_json=False,
            persist_json=False,
        )
        self.frame_count += 1


if __name__ == "__main__":
    video_path = str(Path.home() / "Downloads" / "people-detection.mp4")
    processor = InferenceProcessor("yolo11n-seg.pt", model_type="yolo")
    result_message = processor.run_inference(video_path)
    logger.info(result_message)
