from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

if TYPE_CHECKING:  # pragma: no cover
    pass  # type: ignore

from annolid.annotation.keypoints import save_labels
from annolid.annotation.polygons import simplify_polygons
from annolid.gui.shape import Shape
from annolid.utils.logger import logger
from annolid.annotation.pose_schema import PoseSchema
from annolid.yolo import (
    configure_ultralytics_cache,
    import_ultralytics_symbol,
    is_yoloe_model,
    resolve_weight_path,
)
from annolid.utils.annotation_store import AnnotationStore
from annolid.segmentation.dino_kpseg.inference_bridge import (
    extract_results as extract_dino_kpseg_results,
    instance_masks_from_shapes,
    load_prompt_shapes as load_dino_kpseg_prompt_shapes,
    sanitize_inference_config as sanitize_dino_kpseg_inference_config,
)


class InferenceProcessor:
    def __init__(
        self,
        model_name: str,
        model_type: str,
        class_names: Optional[list] = None,
        *,
        keypoint_names: Optional[list] = None,
        pose_schema_path: Optional[str] = None,
        persist_json: bool = False,
        yoloe_text_prompt: bool = True,
        prompt_class_names: Optional[list] = None,
        dino_kpseg_inference_config: Optional[dict] = None,
        device: Optional[str] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
        half: Optional[bool] = None,
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
        self._yoloe_text_prompt: bool = bool(yoloe_text_prompt)
        self.device: Optional[str] = (
            str(device).strip() if device is not None and str(device).strip() else None
        )
        self.conf: Optional[float] = float(conf) if conf is not None else None
        self.iou: Optional[float] = float(iou) if iou is not None else None
        self.imgsz: Optional[int] = int(imgsz) if imgsz is not None else None
        self.max_det: Optional[int] = int(max_det) if max_det is not None else None
        self.half: Optional[bool] = bool(half) if half is not None else None
        self.class_names: Optional[list] = [
            str(c).strip() for c in (class_names or []) if str(c).strip()
        ] or None
        self.prompt_class_names: Optional[list] = [
            str(c).strip() for c in (prompt_class_names or []) if str(c).strip()
        ] or None
        self._dino_kpseg_inference_config = self._sanitize_dino_kpseg_inference_config(
            dino_kpseg_inference_config
        )
        self.model = self._load_model(class_names)
        self._apply_prompt_names_to_model()
        self.frame_count: int = 0
        self.detection_count: int = 0
        self.track_history = defaultdict(list)
        self.pose_schema: Optional[PoseSchema] = None
        self._instance_label_to_gid: Dict[str, int] = {}
        self.keypoint_names = self._resolve_keypoint_names(
            keypoint_names=keypoint_names,
            pose_schema_path=pose_schema_path,
        )
        self.persist_json: bool = bool(persist_json)

    @staticmethod
    def _sanitize_dino_kpseg_inference_config(
        cfg: Optional[dict],
    ) -> Dict[str, object]:
        return sanitize_dino_kpseg_inference_config(cfg)

    def set_dino_kpseg_inference_config(self, cfg: Optional[dict]) -> None:
        self._dino_kpseg_inference_config = self._sanitize_dino_kpseg_inference_config(
            cfg
        )

    def _apply_prompt_names_to_model(self) -> None:
        """Apply prompt class name mapping to Ultralytics models (used by YOLOE visual prompting)."""
        if not getattr(self, "prompt_class_names", None):
            return
        if "yoloe" not in str(getattr(self, "model_name", "")).lower():
            return

        try:
            names = list(self.prompt_class_names or [])
        except Exception:
            return
        if not names:
            return

        mapping = {int(i): str(name) for i, name in enumerate(names)}
        for target in (
            getattr(self, "model", None),
            getattr(getattr(self, "model", None), "model", None),
        ):
            if target is None:
                continue
            try:
                setattr(target, "names", mapping)
            except Exception:
                pass

    def _resolve_keypoint_names(
        self,
        *,
        keypoint_names: Optional[list],
        pose_schema_path: Optional[str],
    ) -> Optional[list]:
        if keypoint_names:
            cleaned = [
                str(k).strip()
                for k in keypoint_names
                if isinstance(k, str) and k.strip()
            ]
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
            keypoint_config_file = cfg_folder / "configs" / "keypoints.yaml"
            keypoint_config = get_config(keypoint_config_file)
            names = keypoint_config.get("KEYPOINTS")
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
        for parent in [
            model_path.parent,
            model_path.parent.parent,
            model_path.parent.parent.parent,
        ]:
            args_path = parent / "args.yaml"
            if args_path.exists():
                candidates.append(args_path)
        for args_path in candidates:
            try:
                args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
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
            payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
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
                            for key, val in sorted(
                                value.items(), key=lambda kv: int(kv[0])
                            ):
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
                        cleaned = [str(k).strip() for k in names if str(k).strip()]
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
                cls for cls in class_names if isinstance(cls, str) and cls.strip()
            ]

        model_ref = str(self.model_path)
        model_name_lower = model_ref.lower()

        if self.model_type == "yolo":
            if "yoloe" in model_name_lower:
                YOLOE = import_ultralytics_symbol("YOLOE", purpose="YOLOE inference")
                model = YOLOE(model_ref)
                if filtered_classes and bool(getattr(self, "_yoloe_text_prompt", True)):
                    # YOLOE-26 text prompting requires a TorchScript text encoder asset
                    # (e.g. mobileclip2_b.ts). Ultralytics resolves this via
                    # `attempt_download_asset("mobileclip2_b.ts")`, which otherwise downloads into
                    # the current working directory. Prefetch into Annolid's cache to keep runs
                    # reproducible and avoid polluting the project directory.
                    if "yoloe-26" in model_name_lower:
                        from annolid.yolo import ensure_ultralytics_asset_cached

                        ensure_ultralytics_asset_cached("mobileclip2_b.ts")
                    model.set_classes(
                        filtered_classes, model.get_text_pe(filtered_classes)
                    )
            else:
                YOLO = import_ultralytics_symbol("YOLO", purpose="YOLO inference")
                model = YOLO(model_ref)
                if (
                    filtered_classes
                    and "pose" not in model_name_lower
                    and "seg" not in model_name_lower
                ):
                    if hasattr(model, "set_classes"):
                        model.set_classes(filtered_classes)
                    else:
                        logger.warning(
                            "Custom class assignment requested, but model '%s' does not support set_classes.",
                            model_ref,
                        )
            return model
        if self.model_type == "sam":
            SAM = import_ultralytics_symbol(
                "SAM",
                purpose=(
                    "SAM inference via `annolid.segmentation.yolos.InferenceProcessor`"
                ),
            )
            model = SAM(model_ref)
            model.info()
            return model
        if self.model_type == "dinokpseg":
            from annolid.segmentation.dino_kpseg import DinoKPSEGPredictor

            return DinoKPSEGPredictor(model_ref)

        raise ValueError("Unsupported model type. Use 'yolo', 'sam', or 'dinokpseg'.")

    def _validate_visual_prompts(self, visual_prompts: dict) -> bool:
        required_keys = {"bboxes", "cls"}
        if not required_keys.issubset(visual_prompts.keys()):
            logger.error("Visual prompts must contain keys: %s", required_keys)
            return False

        bboxes = visual_prompts["bboxes"]
        cls = visual_prompts["cls"]
        if not isinstance(bboxes, np.ndarray) or not isinstance(cls, np.ndarray):
            logger.error(
                "Both 'bboxes' and 'cls' must be numpy arrays after normalization."
            )
            return False

        if bboxes.shape[0] != cls.shape[0]:
            logger.error(
                "Mismatch: %d bboxes vs %d classes.", bboxes.shape[0], cls.shape[0]
            )
            return False

        instance_ids = visual_prompts.get("instance_ids")
        if instance_ids is not None:
            try:
                num_ids = int(len(instance_ids))
            except Exception:
                logger.error("Visual prompts 'instance_ids' must be array-like.")
                return False
            if num_ids != int(bboxes.shape[0]):
                logger.error(
                    "Mismatch: %d bboxes vs %d instance_ids.",
                    bboxes.shape[0],
                    num_ids,
                )
                return False

        return True

    @staticmethod
    def _should_save_pose_bbox(
        has_keypoints: bool, save_pose_bbox: Optional[bool]
    ) -> bool:
        if not has_keypoints:
            return False
        if save_pose_bbox is None:
            return True
        return bool(save_pose_bbox)

    @classmethod
    def _should_save_yolo_bbox(cls, result, save_pose_bbox: Optional[bool]) -> bool:
        has_keypoints = getattr(result, "keypoints", None) is not None
        if has_keypoints:
            return cls._should_save_pose_bbox(True, save_pose_bbox)
        # Detection models have no mask or keypoint geometry to export.
        return getattr(result, "masks", None) is None

    @staticmethod
    def _make_stop_checker(pred_worker=None, stop_event=None):
        """Return a callable that checks stop_event, pred_worker, and Qt thread interruption."""

        def should_stop() -> bool:
            try:
                if stop_event is not None and stop_event.is_set():
                    return True
            except Exception:
                pass
            try:
                if (
                    pred_worker is not None
                    and hasattr(pred_worker, "is_stopped")
                    and pred_worker.is_stopped()
                ):
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

        return should_stop

    def _build_ultralytics_infer_kwargs(
        self,
        *,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
        device: Optional[str] = None,
        half: Optional[bool] = None,
        stream: bool = False,
    ) -> Dict[str, object]:
        kwargs: Dict[str, object] = {"verbose": False}
        if stream:
            kwargs["stream"] = True

        eff_conf = conf if conf is not None else getattr(self, "conf", None)
        eff_iou = iou if iou is not None else getattr(self, "iou", None)
        eff_imgsz = imgsz if imgsz is not None else getattr(self, "imgsz", None)
        eff_max_det = max_det if max_det is not None else getattr(self, "max_det", None)
        eff_device = (
            str(device).strip()
            if device is not None and str(device).strip()
            else getattr(self, "device", None)
        )
        eff_half = half if half is not None else getattr(self, "half", None)

        if eff_conf is not None:
            kwargs["conf"] = float(eff_conf)
        if eff_iou is not None:
            kwargs["iou"] = float(eff_iou)
        if eff_imgsz is not None:
            kwargs["imgsz"] = int(eff_imgsz)
        if eff_max_det is not None:
            kwargs["max_det"] = int(eff_max_det)
        if eff_device:
            kwargs["device"] = str(eff_device)
        if eff_half is not None:
            dev_lower = str(eff_device or "").lower()
            if bool(eff_half) and (
                bool(getattr(self, "_is_coreml", False)) or dev_lower == "cpu"
            ):
                logger.debug(
                    "Ignoring half=True for CoreML/CPU backend (%s).",
                    eff_device or "coreml",
                )
            else:
                kwargs["half"] = bool(eff_half)
        return kwargs

    @staticmethod
    def _load_yoloe_vp_predictor():
        return import_ultralytics_symbol(
            "YOLOEVPSegPredictor",
            module_name="ultralytics.models.yolo.yoloe",
            purpose="YOLOE visual prompt inference",
        )

    @staticmethod
    def _probe_video_frame_count(source: str) -> int:
        try:
            src_path = Path(str(source))
            if src_path.is_file() and src_path.suffix.lower() in {
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".tif",
                ".tiff",
                ".webp",
            }:
                return 1
        except Exception:
            pass
        try:
            import cv2

            cap = cv2.VideoCapture(str(source))
            try:
                if not cap.isOpened():
                    return 0
                return max(0, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0)
            finally:
                cap.release()
        except Exception:
            return 0

    def _has_any_existing_outputs(self, output_dir: Path) -> bool:
        try:
            store = AnnotationStore.for_frame_path(
                self._labelme_json_path(output_dir, frame_index=0)
            )
            if store.store_path.exists() and store.store_path.stat().st_size > 0:
                return True
        except Exception:
            pass
        try:
            return any(output_dir.glob("*.json"))
        except OSError:
            return False

    def _frame_is_already_processed(
        self, output_dir: Path, *, frame_index: int
    ) -> bool:
        if self._frame_has_existing_output(output_dir, frame_index=int(frame_index)):
            return True
        return self._frame_has_existing_store_record(
            output_dir, frame_index=int(frame_index)
        )

    def _run_ultralytics_call(
        self,
        source_or_frame,
        *,
        infer_kwargs: Dict[str, object],
        enable_tracking: bool,
        tracker: Optional[str],
        visual_prompts: Optional[dict],
        vp_predictor=None,
    ):
        """Dispatch predict() or track() consistently across streaming and frame loops."""
        if visual_prompts is not None and is_yoloe_model(self.model_name):
            predictor = vp_predictor or self._load_yoloe_vp_predictor()
            return self.model.predict(
                source_or_frame,
                visual_prompts=visual_prompts,
                predictor=predictor,
                **infer_kwargs,
            )

        if bool(getattr(self, "_is_coreml", False)) or not enable_tracking:
            return self.model.predict(source_or_frame, **infer_kwargs)

        track_kwargs = dict(infer_kwargs)
        track_kwargs["persist"] = True
        if tracker:
            track_kwargs["tracker"] = tracker
        try:
            return self.model.track(source_or_frame, **track_kwargs)
        except Exception as exc:
            if tracker:
                logger.warning(
                    "Tracker '%s' failed; falling back to default tracker. Error: %s",
                    tracker,
                    exc,
                )
                track_kwargs.pop("tracker", None)
                try:
                    return self.model.track(source_or_frame, **track_kwargs)
                except Exception as fallback_exc:
                    if "pose" in self.model_name.lower():
                        logger.warning(
                            "Tracking unavailable for pose model '%s'; falling back to predict(). Error: %s",
                            self.model_name,
                            fallback_exc,
                        )
                        return self.model.predict(source_or_frame, **infer_kwargs)
                    raise
            if "pose" in self.model_name.lower():
                logger.warning(
                    "Tracking unavailable for pose model '%s'; falling back to predict(). Error: %s",
                    self.model_name,
                    exc,
                )
                return self.model.predict(source_or_frame, **infer_kwargs)
            raise

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
        output_directory: Optional[Path] = None,
        progress_callback=None,
        enable_tracking: bool = True,
        tracker: Optional[str] = None,
        save_pose_bbox: Optional[bool] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
        device: Optional[str] = None,
        half: Optional[bool] = None,
        batch_size: int = 1,
        save_video_path: Optional[Path] = None,
    ) -> str:
        """
        Runs inference on the given source and writes results into Annolid's annotation store.

        When `self.persist_json=True`, per-frame LabelMe-compatible JSON files are also written
        alongside the store entries (without overwriting manually labeled frames).

        Args:
            source (str): Path to the video or image file.

        Returns:
            A string message indicating the completion and frame count.
        """
        output_directory = (
            Path(output_directory) if output_directory else Path(source).with_suffix("")
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        start_frame = max(0, int(start_frame))
        end_frame = None if end_frame is None else int(end_frame)
        step = max(1, abs(int(step)))
        skip_existing = bool(skip_existing)
        batch_size = max(1, int(batch_size or 1))
        self.frame_count = 0
        self.detection_count = 0
        self.track_history.clear()

        should_stop = self._make_stop_checker(
            pred_worker=pred_worker, stop_event=stop_event
        )

        if visual_prompts is not None:
            try:
                raw_prompts = dict(visual_prompts)
                bboxes = np.asarray(visual_prompts.get("bboxes", []), dtype=float)
                cls = np.asarray(visual_prompts.get("cls", []), dtype=int)
                visual_prompts = {"bboxes": bboxes, "cls": cls}
                if "instance_ids" in raw_prompts:
                    visual_prompts["instance_ids"] = np.asarray(
                        raw_prompts.get("instance_ids", []), dtype=object
                    )
            except Exception as exc:
                logger.error(
                    "Failed to normalize visual prompts; proceeding without them: %s",
                    exc,
                )
                visual_prompts = None

        if visual_prompts is not None and not self._validate_visual_prompts(
            visual_prompts
        ):
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
                progress_callback=progress_callback,
            )

        # For GUI-driven video inference we need stable, correct frame indices for:
        # - progress watcher (expects <folder>_000000123.json)
        # - resuming from an arbitrary frame after stopping
        # - batched non-tracking video inference and annotated video export
        try:
            source_path = Path(source)
        except Exception:
            source_path = None

        use_cv2_loop = bool(
            source_path is not None
            and source_path.is_file()
            and (
                start_frame != 0
                or end_frame is not None
                or step != 1
                or batch_size > 1
                or save_video_path is not None
                or (skip_existing and self._has_any_existing_outputs(output_directory))
            )
        )
        if use_cv2_loop:
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
                progress_callback=progress_callback,
                enable_tracking=enable_tracking,
                tracker=tracker,
                save_pose_bbox=save_pose_bbox,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                max_det=max_det,
                device=device,
                half=half,
                batch_size=batch_size,
                save_video_path=save_video_path,
            )

        infer_kwargs = self._build_ultralytics_infer_kwargs(
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            half=half,
            stream=True,
        )

        vp_predictor = None
        if visual_prompts is not None and is_yoloe_model(self.model_name):
            try:
                vp_predictor = self._load_yoloe_vp_predictor()
                logger.info("Running prediction with visual prompts.")
            except Exception as exc:
                logger.error("Error during visual prompt prediction: %s", exc)
                return f"Error: {exc}"

        try:
            results = self._run_ultralytics_call(
                source,
                infer_kwargs=infer_kwargs,
                enable_tracking=enable_tracking,
                tracker=tracker,
                visual_prompts=visual_prompts,
                vp_predictor=vp_predictor,
            )
        except Exception as exc:
            if visual_prompts is not None and is_yoloe_model(self.model_name):
                logger.error("Error during visual prompt prediction: %s", exc)
                return f"Error: {exc}"
            raise

        total_frames = (
            self._probe_video_frame_count(source)
            if (progress_callback and source_path is not None and source_path.is_file())
            else 0
        )

        stopped = False
        try:
            for result in results:
                if should_stop():
                    stopped = True
                    break
                frame_shape = (result.orig_shape[0], result.orig_shape[1], 3)
                annotations = self.extract_yolo_results(
                    result,
                    save_bbox=self._should_save_yolo_bbox(result, save_pose_bbox),
                )
                self.save_yolo_to_labelme(annotations, frame_shape, output_directory)
                self.frame_count += 1
                if progress_callback and total_frames:
                    try:
                        progress_callback(self.frame_count, total_frames)
                    except Exception:
                        pass
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

    run_inference.allow_force_thread_terminate = False

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
        progress_callback=None,
    ) -> str:
        import cv2

        should_stop = self._make_stop_checker(
            pred_worker=pred_worker, stop_event=stop_event
        )

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return f"Error: unable to open {source}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        start_frame = max(0, int(start_frame))
        end_frame = None if end_frame is None else int(end_frame)
        if total_frames > 0 and end_frame is not None:
            end_frame = min(end_frame, total_frames - 1)
        if end_frame is not None:
            max_frames = max(0, end_frame - start_frame + 1)
        elif total_frames > 0:
            max_frames = max(0, total_frames - start_frame)
        else:
            max_frames = 0
        step = max(1, abs(int(step)))
        total_steps = max(1, (max_frames + step - 1) // step) if max_frames else 0
        processed_steps = 0

        stopped = False
        try:
            if int(start_frame) > 0:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(start_frame)))
                except Exception:
                    pass
            frame_index = int(start_frame)
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
                    processed_steps += 1
                    if progress_callback and total_steps:
                        try:
                            progress_callback(processed_steps, total_steps)
                        except Exception:
                            pass
                    if int(step) > 1:
                        for _ in range(int(step) - 1):
                            frame_index += 1
                            if end_frame is not None and frame_index > int(end_frame):
                                break
                            if should_stop():
                                stopped = True
                                break
                            if not cap.grab():
                                break
                        if stopped:
                            break
                    frame_index += 1
                    continue
                ok, frame = cap.read()
                if not ok:
                    break
                frame_shape = (frame.shape[0], frame.shape[1], 3)
                bboxes = None
                instance_ids = None
                if visual_prompts is not None:
                    bboxes = visual_prompts.get("bboxes")
                    instance_ids = visual_prompts.get("instance_ids")
                prompt_shapes = self._load_prompt_shapes(
                    output_directory, frame_index=int(frame_index)
                )
                instance_masks = self._instance_masks_from_shapes(
                    prompt_shapes, frame_hw=(int(frame_shape[0]), int(frame_shape[1]))
                )
                annotations = self.extract_dino_kpseg_results(
                    frame,
                    bboxes=bboxes,
                    instance_masks=instance_masks,
                    instance_ids=instance_ids,
                )
                self.save_yolo_to_labelme(
                    annotations,
                    frame_shape,
                    output_directory,
                    frame_index=frame_index,
                )
                self.frame_count += 1
                processed_steps += 1
                if progress_callback and total_steps:
                    try:
                        progress_callback(processed_steps, total_steps)
                    except Exception:
                        pass

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
        progress_callback=None,
        enable_tracking: bool = True,
        tracker: Optional[str] = None,
        save_pose_bbox: Optional[bool] = None,
        conf: Optional[float] = None,
        iou: Optional[float] = None,
        imgsz: Optional[int] = None,
        max_det: Optional[int] = None,
        device: Optional[str] = None,
        half: Optional[bool] = None,
        batch_size: int = 1,
        save_video_path: Optional[Path] = None,
    ) -> str:
        import cv2

        should_stop = self._make_stop_checker(
            pred_worker=pred_worker, stop_event=stop_event
        )

        cap = cv2.VideoCapture(str(source))
        if not cap.isOpened():
            return f"Error: unable to open {source}"

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if fps <= 0.0:
            fps = 30.0
        start_frame = max(0, int(start_frame))
        end_frame = None if end_frame is None else int(end_frame)
        if total_frames > 0 and end_frame is not None:
            end_frame = min(end_frame, total_frames - 1)
        if end_frame is not None:
            max_frames = max(0, end_frame - start_frame + 1)
        elif total_frames > 0:
            max_frames = max(0, total_frames - start_frame)
        else:
            max_frames = 0
        step = max(1, abs(int(step)))
        total_steps = max(1, (max_frames + step - 1) // step) if max_frames else 0
        processed_steps = 0

        infer_kwargs = self._build_ultralytics_infer_kwargs(
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            max_det=max_det,
            device=device,
            half=half,
            stream=False,
        )

        # Cache visual prompt predictor class once instead of importing every frame.
        vp_predictor = None
        if visual_prompts is not None and is_yoloe_model(self.model_name):
            try:
                vp_predictor = self._load_yoloe_vp_predictor()
            except Exception as exc:
                logger.warning(
                    "YOLOE visual prompt predictor unavailable; falling back to plain predict(): %s",
                    exc,
                )
                visual_prompts = None

        use_batching = (
            int(batch_size or 1) > 1
            and (not enable_tracking or bool(getattr(self, "_is_coreml", False)))
            and visual_prompts is None
        )
        effective_batch = max(1, int(batch_size or 1)) if use_batching else 1

        writer = None
        video_out_path = Path(save_video_path) if save_video_path is not None else None

        def _write_annotated_frame(frame_bgr: np.ndarray, result_obj) -> None:
            nonlocal writer
            if video_out_path is None:
                return
            try:
                rendered = (
                    result_obj.plot()
                    if result_obj is not None and hasattr(result_obj, "plot")
                    else frame_bgr
                )
                if rendered is None:
                    rendered = frame_bgr
                if writer is None:
                    video_out_path.parent.mkdir(parents=True, exist_ok=True)
                    h, w = rendered.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(str(video_out_path), fourcc, fps, (w, h))
                if writer is not None and writer.isOpened():
                    writer.write(rendered)
            except Exception as exc:
                logger.debug("Failed to write annotated video frame: %s", exc)

        def _flush_batch(pending: List[Tuple[int, np.ndarray]]) -> None:
            nonlocal processed_steps
            if not pending:
                return
            batch_frames = [item[1] for item in pending]
            batch_results: List[object] = []
            try:
                raw_results = self.model.predict(batch_frames, **infer_kwargs)
                if isinstance(raw_results, (list, tuple)):
                    batch_results = list(raw_results)
                else:
                    batch_results = list(raw_results or [])
            except Exception as exc:
                logger.error(
                    "Batched YOLO inference failed for frames %s..%s: %s",
                    pending[0][0],
                    pending[-1][0],
                    exc,
                    exc_info=True,
                )
                batch_results = [None] * len(pending)

            for idx_in_batch, (f_idx, f_bgr) in enumerate(pending):
                frame_shape = (f_bgr.shape[0], f_bgr.shape[1], 3)
                res_obj = (
                    batch_results[idx_in_batch]
                    if idx_in_batch < len(batch_results)
                    else None
                )
                annotations = []
                if res_obj is not None:
                    try:
                        annotations = self.extract_yolo_results(
                            res_obj,
                            save_bbox=self._should_save_yolo_bbox(
                                res_obj, save_pose_bbox
                            ),
                        )
                    except Exception as exc:
                        logger.error(
                            "Failed extracting YOLO results at frame %s: %s",
                            f_idx,
                            exc,
                            exc_info=True,
                        )
                _write_annotated_frame(f_bgr, res_obj)
                self.save_yolo_to_labelme(
                    annotations,
                    frame_shape,
                    output_directory,
                    frame_index=f_idx,
                )
                self.frame_count += 1
                processed_steps += 1
                if progress_callback and total_steps:
                    try:
                        progress_callback(processed_steps, total_steps)
                    except Exception:
                        pass
            pending.clear()

        stopped = False
        pending_batch: List[Tuple[int, np.ndarray]] = []
        try:
            if int(start_frame) > 0:
                try:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, float(int(start_frame)))
                except Exception:
                    pass
            frame_index = int(start_frame)
            while True:
                if should_stop():
                    stopped = True
                    break
                if end_frame is not None and frame_index > int(end_frame):
                    break
                if bool(skip_existing) and self._frame_is_already_processed(
                    output_directory, frame_index=int(frame_index)
                ):
                    ok = cap.grab()
                    if not ok:
                        break
                    self.frame_count += 1
                    processed_steps += 1
                    if progress_callback and total_steps:
                        try:
                            progress_callback(processed_steps, total_steps)
                        except Exception:
                            pass
                    if step > 1:
                        for _ in range(step - 1):
                            frame_index += 1
                            if end_frame is not None and frame_index > int(end_frame):
                                break
                            if should_stop():
                                stopped = True
                                break
                            if not cap.grab():
                                break
                        if stopped:
                            break
                    frame_index += 1
                    continue

                ok, frame = cap.read()
                if not ok:
                    break

                if use_batching:
                    pending_batch.append((int(frame_index), frame))
                    if len(pending_batch) >= effective_batch:
                        _flush_batch(pending_batch)
                else:
                    frame_shape = (frame.shape[0], frame.shape[1], 3)
                    annotations = []
                    res_obj = None
                    try:
                        results = self._run_ultralytics_call(
                            frame,
                            infer_kwargs=infer_kwargs,
                            enable_tracking=enable_tracking,
                            tracker=tracker,
                            visual_prompts=visual_prompts,
                            vp_predictor=vp_predictor,
                        )
                        if isinstance(results, (list, tuple)) and results:
                            res_obj = results[0]
                            annotations = self.extract_yolo_results(
                                res_obj,
                                save_bbox=self._should_save_yolo_bbox(
                                    res_obj, save_pose_bbox
                                ),
                            )
                    except Exception as exc:
                        logger.error(
                            "YOLO inference failed at frame %s: %s",
                            frame_index,
                            exc,
                            exc_info=True,
                        )
                        annotations = []

                    _write_annotated_frame(frame, res_obj)
                    # Always write a store record (and optional JSON) to mark the frame as processed.
                    self.save_yolo_to_labelme(
                        annotations,
                        frame_shape,
                        output_directory,
                        frame_index=frame_index,
                    )
                    self.frame_count += 1
                    processed_steps += 1
                    if progress_callback and total_steps:
                        try:
                            progress_callback(processed_steps, total_steps)
                        except Exception:
                            pass

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

            if pending_batch and not stopped:
                _flush_batch(pending_batch)
        finally:
            cap.release()
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

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
            cls._legacy_labelme_json_path(output_dir, frame_index=int(frame_index)),
        )
        for path in candidates:
            try:
                if path.exists() and path.stat().st_size > 0:
                    return True
            except Exception:
                continue
        return False

    def _frame_has_existing_store_record(
        self, output_dir: Path, *, frame_index: int
    ) -> bool:
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
        return load_dino_kpseg_prompt_shapes(
            output_directory,
            frame_index=int(frame_index),
            labelme_json_path=self._labelme_json_path,
            legacy_labelme_json_path=self._legacy_labelme_json_path,
        )

    def _instance_masks_from_shapes(
        self,
        shapes: Sequence[Dict[str, object]],
        *,
        frame_hw: Tuple[int, int],
    ) -> List[Tuple[int, np.ndarray]]:
        return instance_masks_from_shapes(
            shapes,
            frame_hw=(int(frame_hw[0]), int(frame_hw[1])),
            pose_schema=getattr(self, "pose_schema", None),
            instance_label_to_gid=getattr(self, "_instance_label_to_gid", {}),
        )

    def extract_dino_kpseg_results(
        self,
        frame_bgr: np.ndarray,
        *,
        bboxes: Optional[np.ndarray] = None,
        instance_masks: Optional[Sequence[Tuple[int, np.ndarray]]] = None,
        instance_ids: Optional[Sequence[object]] = None,
    ) -> list:
        return extract_dino_kpseg_results(
            frame_bgr=frame_bgr,
            model=self.model,
            model_type=self.model_type,
            keypoint_names=self.keypoint_names,
            inference_config=getattr(self, "_dino_kpseg_inference_config", {}),
            bboxes=bboxes,
            instance_masks=instance_masks,
            instance_ids=instance_ids,
            shape_factory=Shape,
            log=logger,
        )

    def extract_yolo_results(
        self, detection_result, save_bbox: bool = False, save_track: bool = False
    ) -> list:
        """
        Extracts YOLO results from a single detection result, returning a list of Shape objects.

        Args:
            detection_result: The inference result containing detection data.
            save_bbox (bool): Whether to save bounding boxes.
            save_track (bool): Whether to save tracking history.

        Returns:
            list: A list of Shape objects representing the detections.
        """
        annotations: List[Shape] = []

        def _resolve_class_name(cls_id: int, names_obj: object) -> str:
            prompt_names = getattr(self, "prompt_class_names", None)
            if isinstance(prompt_names, list) and 0 <= int(cls_id) < len(prompt_names):
                name = str(prompt_names[int(cls_id)]).strip()
                if name:
                    return name
            if isinstance(names_obj, dict):
                return str(
                    names_obj.get(int(cls_id), names_obj.get(str(int(cls_id)), cls_id))
                )
            if isinstance(names_obj, (list, tuple)) and 0 <= int(cls_id) < len(
                names_obj
            ):
                return str(names_obj[int(cls_id)])
            return str(cls_id)

        boxes_obj = getattr(detection_result, "boxes", None)
        boxes_xywh = boxes_obj.xywh.cpu() if boxes_obj is not None else []
        boxes_xywh_list = None
        if boxes_obj is not None:
            try:
                raw_xywh = boxes_xywh.tolist()
                if isinstance(raw_xywh, list):
                    boxes_xywh_list = raw_xywh
            except Exception:
                boxes_xywh_list = None
        boxes_conf = getattr(boxes_obj, "conf", None) if boxes_obj is not None else None
        boxes_conf_list = None
        if boxes_conf is not None:
            try:
                raw_conf = boxes_conf.cpu().tolist()
                if isinstance(raw_conf, list):
                    boxes_conf_list = raw_conf
                else:
                    boxes_conf_list = [raw_conf]
            except Exception:
                boxes_conf_list = None

        keypoints = getattr(detection_result, "keypoints", None)
        kpts_xy = getattr(keypoints, "xy", None) if keypoints is not None else None
        kpts_conf = getattr(keypoints, "conf", None) if keypoints is not None else None

        num_instances = 0
        try:
            num_instances = (
                len(boxes_xywh_list)
                if boxes_xywh_list is not None
                else int(len(boxes_xywh))
            )
        except Exception:
            num_instances = 0
        if num_instances == 0 and kpts_xy is not None:
            try:
                num_instances = int(kpts_xy.shape[0])
            except Exception:
                num_instances = 0

        group_ids: List[int] = list(range(num_instances))
        if boxes_obj is not None:
            track_tensor = getattr(boxes_obj, "id", None)
            if track_tensor is not None:
                try:
                    track_ids = track_tensor.int().cpu().tolist()
                    if isinstance(track_ids, list) and len(track_ids) == num_instances:
                        group_ids = [int(t) for t in track_ids]
                except Exception:
                    pass

        masks = getattr(detection_result, "masks", None)
        names = getattr(detection_result, "names", {}) or {}
        cls_ids: List[int] = []
        if boxes_obj is not None:
            cls_tensor = getattr(boxes_obj, "cls", None)
            if cls_tensor is not None:
                try:
                    raw_cls = cls_tensor.int().cpu().tolist()
                    if isinstance(raw_cls, list):
                        cls_ids = [int(c) for c in raw_cls]
                    else:
                        cls_ids = [int(raw_cls)]
                except Exception:
                    cls_ids = []
            if not cls_ids:
                try:
                    cls_ids = [int(box.cls) for box in boxes_obj]
                except Exception:
                    cls_ids = []
        class_names: List[str] = []
        for idx in range(num_instances):
            name = ""
            if idx < len(cls_ids):
                name = _resolve_class_name(cls_ids[idx], names)
            class_names.append(name)
        class_name_counts = {k: class_names.count(k) for k in set(class_names) if k}

        # Process each detection instance (bbox/mask/keypoints share the same index).
        for idx in range(num_instances):
            group_id = int(group_ids[idx])
            class_name = class_names[idx] if idx < len(class_names) else ""
            instance_label = class_name or "object"
            if class_name and class_name_counts.get(class_name, 0) > 1:
                instance_label = f"{class_name}_{group_id}"

            if kpts_xy is not None:
                try:
                    xy_row = kpts_xy[idx].cpu().tolist()
                except Exception:
                    xy_row = []
                conf_row = None
                if kpts_conf is not None:
                    try:
                        conf_row = kpts_conf[idx].cpu().tolist()
                    except Exception:
                        conf_row = None

                for kpt_id, kpt in enumerate(xy_row):
                    if not isinstance(kpt, (list, tuple)) or len(kpt) < 2:
                        continue
                    kx, ky = float(kpt[0]), float(kpt[1])
                    conf_val = (
                        float(conf_row[kpt_id])
                        if conf_row is not None and kpt_id < len(conf_row)
                        else None
                    )
                    # Ultralytics encodes undetected keypoints as (0.0, 0.0) with conf == 0.0.
                    # Do not emit them as visible keypoints at the image origin.
                    if (
                        kx == 0.0
                        and ky == 0.0
                        and (conf_val is None or conf_val <= 0.0)
                    ):
                        continue
                    if self.keypoint_names and kpt_id < len(self.keypoint_names):
                        kpt_label = str(self.keypoint_names[kpt_id])
                    else:
                        kpt_label = str(kpt_id)
                    keypoint_shape = Shape(
                        kpt_label,
                        shape_type="point",
                        description=self.model_type,
                        flags={},
                        group_id=group_id,
                    )
                    keypoint_shape.points = [[kx, ky]]
                    if conf_val is not None:
                        keypoint_shape.other_data["score"] = conf_val
                    keypoint_shape.other_data["instance_id"] = int(group_id)
                    keypoint_shape.other_data["instance_label"] = instance_label
                    if class_name:
                        keypoint_shape.other_data["class_name"] = class_name
                    annotations.append(keypoint_shape)

            box_coords = None
            if boxes_xywh_list is not None and idx < len(boxes_xywh_list):
                box_coords = boxes_xywh_list[idx]
            elif idx < len(boxes_xywh):
                box_coords = boxes_xywh[idx].tolist()

            if save_bbox and box_coords is not None:
                x, y, w, h = box_coords
                x1, y1 = x - w / 2, y - h / 2
                x2, y2 = x + w / 2, y + h / 2
                bbox_shape = Shape(
                    class_name,
                    shape_type="rectangle",
                    description=self.model_type,
                    flags={},
                    group_id=group_id,
                )
                bbox_shape.points = [[float(x1), float(y1)], [float(x2), float(y2)]]
                bbox_shape.other_data["instance_id"] = int(group_id)
                bbox_shape.other_data["instance_label"] = instance_label
                if boxes_conf_list is not None and idx < len(boxes_conf_list):
                    bbox_shape.other_data["score"] = float(boxes_conf_list[idx])
                annotations.append(bbox_shape)

            if save_track and box_coords is not None:
                x, y, w, h = box_coords
                track_key = str(group_id)
                track = self.track_history[track_key]
                track.append((float(x), float(y)))
                if len(track) > 30:
                    track.pop(0)
                if len(track) > 1:
                    shape_track = Shape(
                        f"track_{track_key}",
                        shape_type="polygon",
                        description=self.model_type,
                        flags={},
                        group_id=group_id,
                        visible=True,
                    )
                    shape_track.points = np.array(track).tolist()
                    shape_track.other_data["instance_id"] = int(group_id)
                    shape_track.other_data["instance_label"] = instance_label
                    annotations.append(shape_track)

            if masks is not None:
                try:
                    mask = masks[idx]
                except Exception:
                    mask = None
                if mask is not None:
                    try:
                        polygons = simplify_polygons(mask.xy)
                        for polygon in polygons:
                            contour_points = polygon.tolist()
                            if len(contour_points) > 2:
                                segmentation_shape = Shape(
                                    class_name,
                                    shape_type="polygon",
                                    description=self.model_type,
                                    flags={},
                                    group_id=group_id,
                                    visible=True,
                                )
                                segmentation_shape.points = contour_points
                                segmentation_shape.other_data["instance_id"] = int(
                                    group_id
                                )
                                segmentation_shape.other_data["instance_label"] = (
                                    instance_label
                                )
                                if boxes_conf_list is not None and idx < len(
                                    boxes_conf_list
                                ):
                                    segmentation_shape.other_data["score"] = float(
                                        boxes_conf_list[idx]
                                    )
                                annotations.append(segmentation_shape)
                    except Exception as exc:
                        logger.error("Error processing mask: %s", exc)

        self.detection_count = int(getattr(self, "detection_count", 0)) + int(
            num_instances
        )
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
        Append YOLO annotations to the per-folder annotation store, and (optionally) write a LabelMe JSON file.

        Args:
            annotations (list): List of Shape objects.
            frame_shape (tuple): Tuple containing (height, width, channels).
            output_dir (Path): Output directory where JSON will be saved.
        """
        height, width, _ = frame_shape
        if frame_index is None:
            frame_index = int(self.frame_count)
        output_path = self._labelme_json_path(output_dir, frame_index=int(frame_index))
        save_labels(
            filename=str(output_path),
            imagePath="",
            label_list=annotations,
            height=height,
            width=width,
            save_image_to_json=False,
            persist_json=bool(getattr(self, "persist_json", False)),
        )
        # `frame_count` is a processed-frame counter; the JSON file encodes the
        # actual frame index for the progress watcher / resume logic.


if __name__ == "__main__":
    video_path = str(Path.home() / "Downloads" / "people-detection.mp4")
    processor = InferenceProcessor("yolo11n-seg.pt", model_type="yolo")
    result_message = processor.run_inference(video_path)
    logger.info(result_message)
