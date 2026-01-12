from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml
from ultralytics import SAM, YOLO, YOLOE

from annolid.annotation.keypoints import save_labels
from annolid.annotation.polygons import simplify_polygons
from annolid.gui.shape import Shape
from annolid.utils.logger import logger
from annolid.annotation.pose_schema import PoseSchema
from annolid.yolo import configure_ultralytics_cache, resolve_weight_path
from annolid.utils.annotation_store import AnnotationStore, load_labelme_json


class InferenceProcessor:
    def __init__(
        self,
        model_name: str,
        model_type: str,
        class_names: Optional[list] = None,
        *,
        keypoint_names: Optional[list] = None,
        pose_schema_path: Optional[str] = None,
    ) -> None:
        """
        Initializes the InferenceProcessor with a specified model.

        Args:
            model_name (str): Path or identifier for the model file.
            model_type (str): Type of model ('yolo', 'sam', or 'dinokpseg').
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
        self.pose_schema: Optional[PoseSchema] = None
        self._instance_label_to_gid: Dict[str, int] = {}
        self.keypoint_names = self._resolve_keypoint_names(
            keypoint_names=keypoint_names,
            pose_schema_path=pose_schema_path,
        )

    @staticmethod
    def _clean_instance_label(value: object) -> str:
        if value is None:
            return ""
        text = str(value).strip()
        if not text:
            return ""
        return text.rstrip("_-:|")

    @staticmethod
    def _normalize_group_id(value: object) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):  # bool is also int
            return int(value)
        if isinstance(value, int):
            return int(value)
        if isinstance(value, float) and np.isfinite(value):
            return int(value)
        if isinstance(value, str):
            raw = value.strip()
            if raw.isdigit():
                return int(raw)
        return None

    @staticmethod
    def _next_group_id(used: set[int]) -> int:
        candidate = max(used) + 1 if used else 0
        while candidate in used:
            candidate += 1
        return candidate

    def _resolve_keypoint_names(
        self,
        *,
        keypoint_names: Optional[list],
        pose_schema_path: Optional[str],
    ) -> Optional[list]:
        if keypoint_names:
            cleaned = [str(k).strip()
                       for k in keypoint_names if isinstance(k, str) and k.strip()]
            if cleaned:
                return cleaned

        if pose_schema_path:
            try:
                schema = PoseSchema.load(pose_schema_path)
                self.pose_schema = schema
                if schema.keypoints:
                    return list(schema.keypoints)
            except Exception:
                pass

        inferred = self._infer_keypoint_names_from_training_artifacts()
        if inferred:
            return inferred

        inferred = self._infer_keypoint_names_from_model()
        if inferred:
            return inferred

        # Only fall back to COCO-style defaults when the model's keypoint count matches.
        inferred_count = self._infer_keypoint_count()
        if inferred_count:
            default_names = self._load_default_keypoint_names()
            if default_names and len(default_names) == inferred_count:
                return default_names

        return None

    def _infer_keypoint_count(self) -> Optional[int]:
        for candidate in (self.model, getattr(self.model, "model", None)):
            if candidate is None:
                continue
            kpt_shape = getattr(candidate, "kpt_shape", None)
            if isinstance(kpt_shape, (list, tuple)) and kpt_shape:
                try:
                    return int(kpt_shape[0])
                except Exception:
                    continue
        return None

    def _load_default_keypoint_names(self) -> Optional[list]:
        try:
            from annolid.utils.config import get_config

            cfg_folder = Path(__file__).resolve().parent.parent
            keypoint_config_file = cfg_folder / 'configs' / 'keypoints.yaml'
            keypoint_config = get_config(keypoint_config_file)
            names = keypoint_config.get('KEYPOINTS')
            if isinstance(names, str):
                items = [k.strip() for k in names.split(" ") if k.strip()]
                return items or None
            if isinstance(names, list):
                items = [str(k).strip() for k in names if str(k).strip()]
                return items or None
        except Exception:
            return None
        return None

    def _infer_keypoint_names_from_training_artifacts(self) -> Optional[list]:
        model_path = Path(self.model_name)
        candidates = []
        for parent in [model_path.parent, model_path.parent.parent, model_path.parent.parent.parent]:
            args_path = parent / "args.yaml"
            if args_path.exists():
                candidates.append(args_path)
        for args_path in candidates:
            try:
                args = yaml.safe_load(
                    args_path.read_text(encoding="utf-8")) or {}
            except Exception:
                continue
            if not isinstance(args, dict):
                continue
            data_path = args.get("data")
            if not data_path:
                continue
            try:
                data_yaml = Path(str(data_path)).expanduser()
                if not data_yaml.is_absolute():
                    data_yaml = (args_path.parent / data_yaml).resolve()
                if not data_yaml.exists():
                    continue
                return self._infer_keypoint_names_from_data_yaml(data_yaml)
            except Exception:
                continue
        return None

    @staticmethod
    def _infer_keypoint_names_from_data_yaml(data_yaml: Path) -> Optional[list]:
        try:
            payload = yaml.safe_load(
                data_yaml.read_text(encoding="utf-8")) or {}
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None

        kpt_names = payload.get("kpt_names")
        if isinstance(kpt_names, dict) and kpt_names:
            if 0 in kpt_names:
                names = kpt_names.get(0)
            elif "0" in kpt_names:
                names = kpt_names.get("0")
            else:
                names = next(iter(kpt_names.values()))
            if isinstance(names, list):
                cleaned = [str(k).strip() for k in names if str(k).strip()]
                return cleaned or None

        kpt_labels = payload.get("kpt_labels")
        if isinstance(kpt_labels, dict) and kpt_labels:
            items = []
            try:
                for key, value in sorted(kpt_labels.items(), key=lambda kv: int(kv[0])):
                    name = str(value).strip()
                    items.append(name)
            except Exception:
                items = [str(v).strip() for v in kpt_labels.values()]
            cleaned = [k for k in items if k]
            return cleaned or None

        return None

    def _infer_keypoint_names_from_model(self) -> Optional[list]:
        for container in (self.model, getattr(self.model, "model", None)):
            if container is None:
                continue
            for attr in ("names_kpt", "kpt_names", "kpt_labels"):
                value = getattr(container, attr, None)
                if not value:
                    continue
                if isinstance(value, list):
                    cleaned = [str(k).strip() for k in value if str(k).strip()]
                    return cleaned or None
                if isinstance(value, dict):
                    if attr == "kpt_labels":
                        items = []
                        try:
                            for key, val in sorted(value.items(), key=lambda kv: int(kv[0])):
                                items.append(str(val).strip())
                        except Exception:
                            items = [str(v).strip() for v in value.values()]
                        cleaned = [k for k in items if k]
                        return cleaned or None
                    # kpt_names may be {0: [..]}
                    if 0 in value:
                        names = value.get(0)
                    elif "0" in value:
                        names = value.get("0")
                    else:
                        names = next(iter(value.values()))
                    if isinstance(names, list):
                        cleaned = [str(k).strip()
                                   for k in names if str(k).strip()]
                        return cleaned or None
        return None

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
        if self.model_type == "dinokpseg":
            from annolid.segmentation.dino_kpseg import DinoKPSEGPredictor

            return DinoKPSEGPredictor(model_ref)

        raise ValueError(
            "Unsupported model type. Use 'yolo', 'sam', or 'dinokpseg'.")

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
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        skip_existing: bool = True,
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
        start_frame = max(0, int(start_frame))
        end_frame = None if end_frame is None else int(end_frame)
        step = max(1, abs(int(step)))
        skip_existing = bool(skip_existing)
        self.frame_count = 0
        self.track_history.clear()

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

        if self.model_type == "dinokpseg":
            return self._run_dino_kpseg_inference(
                source,
                output_directory=output_directory,
                pred_worker=pred_worker,
                stop_event=stop_event,
                visual_prompts=visual_prompts,
                start_frame=start_frame,
                end_frame=end_frame,
                step=step,
                skip_existing=skip_existing,
            )

        # For GUI-driven video inference we need stable, correct frame indices for:
        # - progress watcher (expects <folder>_000000123.json)
        # - resuming from an arbitrary frame after stopping
        #
        # Ultralytics' video streaming API doesn't provide start-frame support,
        # so use a CV2 loop whenever a non-default window/stride is requested.
        try:
            source_path = Path(source)
        except Exception:
            source_path = None
        if source_path is not None and source_path.is_file() and (
            start_frame != 0 or end_frame is not None or step != 1
        ):
            return self._run_yolo_video_inference_cv2(
                source,
                output_directory=output_directory,
                pred_worker=pred_worker,
                stop_event=stop_event,
                visual_prompts=visual_prompts,
                start_frame=start_frame,
                end_frame=end_frame,
                step=step,
                skip_existing=skip_existing,
            )

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

    def _run_dino_kpseg_inference(
        self,
        source: str,
        *,
        output_directory: Path,
        pred_worker=None,
        stop_event=None,
        visual_prompts: dict = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        skip_existing: bool = True,
    ) -> str:
        import cv2

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

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return f"Error: unable to open {source}"

        stopped = False
        try:
            if int(start_frame) > 0:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(start_frame)))
                except Exception:
                    pass
            frame_index = int(start_frame)
            step = max(1, abs(int(step)))
            while True:
                if should_stop():
                    stopped = True
                    break
                if end_frame is not None and frame_index > int(end_frame):
                    break
                if bool(skip_existing) and self._frame_has_existing_store_record(
                    output_directory, frame_index=int(frame_index)
                ):
                    ok = cap.grab()
                    if not ok:
                        break
                    self.frame_count += 1
                    frame_index += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                frame_shape = (frame.shape[0], frame.shape[1], 3)
                bboxes = None
                if visual_prompts is not None:
                    bboxes = visual_prompts.get("bboxes")
                prompt_shapes = self._load_prompt_shapes(
                    output_directory, frame_index=int(frame_index)
                )
                instance_masks = self._instance_masks_from_shapes(
                    prompt_shapes, frame_hw=(int(frame_shape[0]), int(frame_shape[1]))
                )
                annotations = self.extract_dino_kpseg_results(
                    frame, bboxes=bboxes, instance_masks=instance_masks
                )
                self.save_yolo_to_labelme(
                    annotations,
                    frame_shape,
                    output_directory,
                    frame_index=frame_index,
                )
                self.frame_count += 1

                if int(step) > 1:
                    for _ in range(int(step) - 1):
                        frame_index += 1
                        if end_frame is not None and frame_index > int(end_frame):
                            break
                        if should_stop():
                            stopped = True
                            break
                        ok = cap.grab()
                        if not ok:
                            break
                    if stopped:
                        break
                frame_index += 1
        finally:
            cap.release()

        if stopped:
            return f"Stopped#{self.frame_count}"
        return f"Done#{self.frame_count}"

    def _run_yolo_video_inference_cv2(
        self,
        source: str,
        *,
        output_directory: Path,
        pred_worker=None,
        stop_event=None,
        visual_prompts: dict = None,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        step: int = 1,
        skip_existing: bool = True,
    ) -> str:
        import cv2

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

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return f"Error: unable to open {source}"

        stopped = False
        try:
            if int(start_frame) > 0:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(start_frame)))
                except Exception:
                    pass
            frame_index = int(start_frame)
            step = max(1, abs(int(step)))
            while True:
                if should_stop():
                    stopped = True
                    break
                if end_frame is not None and frame_index > int(end_frame):
                    break
                if bool(skip_existing) and self._frame_has_existing_output(
                    output_directory, frame_index=int(frame_index)
                ):
                    ok = cap.grab()
                    if not ok:
                        break
                    self.frame_count += 1
                    frame_index += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break

                frame_shape = (frame.shape[0], frame.shape[1], 3)
                annotations = []
                try:
                    if visual_prompts is not None and "yoloe" in self.model_name.lower():
                        try:
                            from ultralytics.models.yolo.yoloe import YOLOEVPSegPredictor

                            results = self.model.predict(
                                frame,
                                visual_prompts=visual_prompts,
                                predictor=YOLOEVPSegPredictor,
                                verbose=False,
                            )
                        except Exception:
                            results = self.model.predict(frame, verbose=False)
                    else:
                        if self._is_coreml:
                            results = self.model.predict(frame, verbose=False)
                        elif "pose" in self.model_name.lower():
                            results = self.model.predict(frame, verbose=False)
                        else:
                            results = self.model.track(
                                frame, persist=True, verbose=False
                            )
                    if isinstance(results, (list, tuple)) and results:
                        annotations = self.extract_yolo_results(results[0])
                except Exception as exc:
                    logger.error("YOLO inference failed at frame %s: %s",
                                 frame_index, exc, exc_info=True)
                    annotations = []

                # Always write a JSON to mark the frame as processed (keeps the GUI progress watcher accurate).
                self.save_yolo_to_labelme(
                    annotations,
                    frame_shape,
                    output_directory,
                    frame_index=frame_index,
                )
                self.frame_count += 1

                if step > 1:
                    for _ in range(step - 1):
                        frame_index += 1
                        if end_frame is not None and frame_index > int(end_frame):
                            break
                        if should_stop():
                            stopped = True
                            break
                        ok = cap.grab()
                        if not ok:
                            break
                    if stopped:
                        break
                frame_index += 1
        finally:
            cap.release()

        if stopped:
            return f"Stopped#{self.frame_count}"
        return f"Done#{self.frame_count}"

    @staticmethod
    def _legacy_labelme_json_path(output_dir: Path, *, frame_index: int) -> Path:
        return output_dir / f"{int(frame_index):09d}.json"

    @classmethod
    def _frame_has_existing_output(cls, output_dir: Path, *, frame_index: int) -> bool:
        # Support both the current "<folder>_000000123.json" and the legacy "000000123.json".
        candidates = (
            cls._labelme_json_path(output_dir, frame_index=int(frame_index)),
            cls._legacy_labelme_json_path(
                output_dir, frame_index=int(frame_index)),
        )
        for path in candidates:
            try:
                if path.exists() and path.stat().st_size > 0:
                    return True
            except Exception:
                continue
        return False

    def _frame_has_existing_store_record(self, output_dir: Path, *, frame_index: int) -> bool:
        store = AnnotationStore.for_frame_path(
            self._labelme_json_path(output_dir, frame_index=int(frame_index))
        )
        try:
            return store.get_frame(int(frame_index)) is not None
        except Exception:
            return False

    def _load_prompt_shapes(
        self,
        output_directory: Path,
        *,
        frame_index: int,
    ) -> List[Dict[str, object]]:
        """Load existing shapes to drive per-instance DinoKPSEG inference.

        Preference order:
        1) Per-frame LabelMe JSON (if present)
        2) AnnotationStore record for the frame (NDJSON)
        """
        frame_index = int(frame_index)
        candidates = (
            self._labelme_json_path(output_directory, frame_index=frame_index),
            self._legacy_labelme_json_path(
                output_directory, frame_index=frame_index),
        )
        for path in candidates:
            try:
                if path.exists() and path.stat().st_size > 0:
                    payload = load_labelme_json(path)
                    shapes = payload.get("shapes") or []
                    if isinstance(shapes, list):
                        return [s for s in shapes if isinstance(s, dict)]
            except Exception:
                continue

        store = AnnotationStore.for_frame_path(
            self._labelme_json_path(output_directory, frame_index=frame_index)
        )
        record = store.get_frame(frame_index)
        shapes = record.get("shapes") if isinstance(record, dict) else None
        if isinstance(shapes, list):
            return [s for s in shapes if isinstance(s, dict)]
        return []

    def _instance_masks_from_shapes(
        self,
        shapes: Sequence[Dict[str, object]],
        *,
        frame_hw: Tuple[int, int],
    ) -> List[Tuple[int, np.ndarray]]:
        """Build (instance_id, mask) pairs from existing polygon-like shapes."""
        try:
            import cv2  # type: ignore
        except Exception:
            return []

        height, width = int(frame_hw[0]), int(frame_hw[1])
        if height <= 0 or width <= 0:
            return []

        schema_instances: Dict[str, int] = {}
        if self.pose_schema and getattr(self.pose_schema, "instances", None):
            for idx, name in enumerate(self.pose_schema.instances):
                clean = self._clean_instance_label(name)
                if clean and clean.lower() not in schema_instances:
                    schema_instances[clean.lower()] = int(idx)

        instance_masks: List[Tuple[int, np.ndarray]] = []
        used_gids: set[int] = set()
        for shape in shapes:
            shape_type = str(shape.get("shape_type") or "").strip().lower()
            if shape_type not in ("polygon", "rectangle", "circle"):
                continue
            points = shape.get("points") or []
            if not isinstance(points, list) or not points:
                continue

            flags = shape.get("flags") if isinstance(
                shape.get("flags"), dict) else {}
            gid = self._normalize_group_id(shape.get("group_id"))
            if gid is None:
                gid = self._normalize_group_id(shape.get("instance_id"))
            if gid is None and flags:
                gid = self._normalize_group_id(flags.get("instance_id"))

            label = self._clean_instance_label(shape.get("label"))
            if gid is None and label:
                by_schema = schema_instances.get(label.lower())
                if by_schema is not None:
                    gid = int(by_schema)
            if gid is None and label:
                existing = self._instance_label_to_gid.get(label.lower())
                if existing is not None:
                    gid = int(existing)
            if gid is None:
                gid = self._next_group_id(used_gids)
            used_gids.add(int(gid))
            if label:
                self._instance_label_to_gid.setdefault(label.lower(), int(gid))

            mask = np.zeros((height, width), dtype=np.uint8)
            if shape_type == "polygon":
                poly = []
                for pt in points:
                    if not isinstance(pt, (list, tuple)) or len(pt) < 2:
                        continue
                    poly.append((float(pt[0]), float(pt[1])))
                if len(poly) < 3:
                    continue
                poly_arr = np.rint(np.array(poly, dtype=np.float32)
                                   ).astype(np.int32)
                cv2.fillPoly(mask, [poly_arr], 1)
            elif shape_type == "rectangle":
                if len(points) < 2:
                    continue
                a, b = points[0], points[1]
                if (
                    not isinstance(a, (list, tuple))
                    or not isinstance(b, (list, tuple))
                    or len(a) < 2
                    or len(b) < 2
                ):
                    continue
                x1 = int(round(min(float(a[0]), float(b[0]))))
                y1 = int(round(min(float(a[1]), float(b[1]))))
                x2 = int(round(max(float(a[0]), float(b[0]))))
                y2 = int(round(max(float(a[1]), float(b[1]))))
                x1 = max(0, min(width - 1, x1))
                x2 = max(0, min(width, x2))
                y1 = max(0, min(height - 1, y1))
                y2 = max(0, min(height, y2))
                if x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                cv2.rectangle(mask, (x1, y1), (x2, y2), 1, thickness=-1)
            elif shape_type == "circle":
                if len(points) < 2:
                    continue
                c, e = points[0], points[1]
                if (
                    not isinstance(c, (list, tuple))
                    or not isinstance(e, (list, tuple))
                    or len(c) < 2
                    or len(e) < 2
                ):
                    continue
                cx, cy = float(c[0]), float(c[1])
                ex, ey = float(e[0]), float(e[1])
                r = int(round(((cx - ex) ** 2 + (cy - ey) ** 2) ** 0.5))
                if r <= 0:
                    continue
                cv2.circle(mask, (int(round(cx)), int(round(cy))),
                           r, 1, thickness=-1)

            if not np.any(mask):
                continue
            instance_masks.append((int(gid), mask.astype(bool)))

        instance_masks.sort(key=lambda item: int(item[0]))
        return instance_masks

    def extract_dino_kpseg_results(
        self,
        frame_bgr: np.ndarray,
        *,
        bboxes: Optional[np.ndarray] = None,
        instance_masks: Optional[Sequence[Tuple[int, np.ndarray]]] = None,
    ) -> list:
        annotations = []
        try:
            if instance_masks:
                from annolid.segmentation.dino_kpseg.inference_utils import (
                    build_instance_crops,
                    predict_on_instance_crops,
                )

                crops = build_instance_crops(
                    frame_bgr,
                    list(instance_masks),
                    pad_px=8,
                    use_mask_gate=True,
                )
                predictions = predict_on_instance_crops(
                    self.model,
                    crops,
                    return_patch_masks=False,
                    stabilize_lr=True,
                )
            elif bboxes is not None and len(bboxes) > 0:
                predictions = self.model.predict_instances(
                    frame_bgr,
                    bboxes_xyxy=bboxes,
                    return_patch_masks=False,
                    stabilize_lr=True,
                )
            else:
                # No prompts (no polygons / boxes). Fall back to multi-peak decoding
                # so a single frame can contain multiple "nose" points, etc.
                peaks = self.model.predict_multi_peaks(
                    frame_bgr,
                    threshold=None,
                    topk=5,
                    nms_radius_px=12.0,
                )
                predictions = [(None, self.model.predict(
                    frame_bgr, return_patch_masks=False, stabilize_lr=True))]
        except Exception as exc:
            logger.error("DinoKPSEG inference failed: %s", exc, exc_info=True)
            return annotations

        kp_names = self.keypoint_names or getattr(
            self.model, "keypoint_names", None)
        if not kp_names and predictions:
            first_pred = predictions[0][1]
            kp_names = [str(i)
                        for i in range(len(first_pred.keypoints_xy))]

        # Emit multi-peak points only when no instance separation signals were present.
        if (bboxes is None or len(bboxes) == 0) and not instance_masks:
            if kp_names:
                for kpt_id, channel_peaks in enumerate(peaks):
                    label = kp_names[kpt_id] if kpt_id < len(
                        kp_names) else str(kpt_id)
                    for rank, (x, y, score) in enumerate(channel_peaks):
                        point_shape = Shape(
                            label,
                            shape_type="point",
                            description=self.model_type,
                            flags={},
                            group_id=None,
                        )
                        point_shape.points = [[float(x), float(y)]]
                        point_shape.other_data["score"] = float(score)
                        point_shape.other_data["peak_rank"] = int(rank)
                        point_shape.other_data["multi_peak"] = True
                        annotations.append(point_shape)
                return annotations

        for instance_id, prediction in predictions:
            group_id = int(instance_id) if instance_id is not None else None
            for kpt_id, (xy, score) in enumerate(
                zip(prediction.keypoints_xy, prediction.keypoint_scores)
            ):
                label = kp_names[kpt_id] if kpt_id < len(
                    kp_names) else str(kpt_id)
                x, y = float(xy[0]), float(xy[1])

                point_shape = Shape(
                    label,
                    shape_type="point",
                    description=self.model_type,
                    flags={},
                    group_id=group_id,
                )
                point_shape.points = [[x, y]]
                point_shape.other_data["score"] = float(score)
                if group_id is not None:
                    point_shape.other_data["instance_id"] = int(group_id)
                annotations.append(point_shape)

        return annotations

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

    @staticmethod
    def _labelme_json_path(output_dir: Path, *, frame_index: int) -> Path:
        return output_dir / f"{output_dir.name}_{int(frame_index):09d}.json"

    def save_yolo_to_labelme(
        self,
        annotations: list,
        frame_shape: tuple,
        output_dir: Path,
        *,
        frame_index: Optional[int] = None,
    ) -> None:
        """
        Saves YOLO annotations to a LabelMe JSON file.

        Args:
            annotations (list): List of Shape objects.
            frame_shape (tuple): Tuple containing (height, width, channels).
            output_dir (Path): Output directory where JSON will be saved.
        """
        height, width, _ = frame_shape
        if frame_index is None:
            frame_index = int(self.frame_count)
        output_path = self._labelme_json_path(
            output_dir, frame_index=int(frame_index))
        save_labels(
            filename=str(output_path),
            imagePath="",
            label_list=annotations,
            height=height,
            width=width,
            save_image_to_json=False,
            persist_json=False,
        )
        # `frame_count` is a processed-frame counter; the JSON file encodes the
        # actual frame index for the progress watcher / resume logic.


if __name__ == "__main__":
    video_path = str(Path.home() / "Downloads" / "people-detection.mp4")
    processor = InferenceProcessor("yolo11n-seg.pt", model_type="yolo")
    result_message = processor.run_inference(video_path)
    logger.info(result_message)
