import torch
import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Dict
import numpy as np
from ultralytics import YOLO
from annolid.utils.logger import logger

# --- Engine Cache to implement Singleton Pattern ---
# This dictionary will cache initialized engines to prevent re-loading the same model
# into memory within the same process. The key will be the model path.
_engine_cache: Dict[str, 'InferenceEngine'] = {}


def get_engine(model_name_base: str = "yolo11n-seg") -> 'InferenceEngine':
    """
    Factory function to get or create an InferenceEngine instance.
    This ensures that for any given model base name, we only create and load
    one instance of the engine, conserving GPU memory.

    Args:
        model_name_base (str): The base name of the model (e.g., "yolo11n-seg").

    Returns:
        An initialized InferenceEngine instance.
    """
    # Determine the best path first, so we can use it as a consistent key
    model_path, _, _ = _select_best_engine_path(model_name_base)

    if model_path in _engine_cache:
        logger.debug(f"Returning cached InferenceEngine for {model_path}")
        return _engine_cache[model_path]

    logger.info(f"Creating new InferenceEngine instance for {model_path}")
    engine = InferenceEngine(model_name_base)
    _engine_cache[model_path] = engine
    return engine


def _select_best_engine_path(model_name_base: str) -> Tuple[str, torch.device, str]:
    """
    Detects available hardware and selects the best available model file and device.
    This is a private helper function for the engine.

    Returns:
        A tuple of (model_path, device, model_type_string).
    """
    # --- Tier 1: Check for TensorRT Engine on NVIDIA GPU ---
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        try:
            import tensorrt
            engine_path = Path(f"{model_name_base}.engine")
            if engine_path.is_file():
                logger.info(
                    "Optimal backend selected: TensorRT on NVIDIA GPU.")
                return str(engine_path), device, "TensorRT (GPU)"
            else:
                logger.warning(f"TensorRT is installed, but '{engine_path}' not found. "
                               "Falling back to PyTorch. For best performance, export the model to TensorRT format first.")
        except ImportError:
            logger.warning(
                "TensorRT not installed. Falling back to standard PyTorch on GPU.")

        # --- Tier 2: Fallback to PyTorch on NVIDIA GPU ---
        pt_path = Path(f"{model_name_base}.pt")
        if pt_path.is_file():
            logger.info("Backend selected: PyTorch on NVIDIA GPU.")
            return str(pt_path), device, "PyTorch (GPU)"
        else:
            raise FileNotFoundError(
                f"GPU detected, but required PyTorch model '{pt_path.name}' was not found. "
                "Place the model in the working directory."
            )

    # --- Tier 3: Fallback to PyTorch on CPU ---
    device = torch.device("cpu")
    pt_path = Path(f"{model_name_base}.pt")
    if pt_path.is_file():
        logger.warning(
            "No compatible NVIDIA GPU found. Falling back to CPU. Performance will be significantly slower.")
        return str(pt_path), device, "PyTorch (CPU)"
    else:
        raise FileNotFoundError(
            f"No PyTorch model '{pt_path.name}' found for CPU operation. "
            "Place the model in the working directory."
        )


class InferenceEngine:
    """
    A hardware-agnostic inference engine for real-time object detection and segmentation.

    This class automatically detects the optimal available backend (TensorRT > PyTorch/GPU > PyTorch/CPU)
    and provides a single, consistent interface for running inference on individual frames.
    It is designed to be instantiated via the `get_engine()` factory function to ensure
    only one instance per model is loaded into memory.
    """

    def __init__(self, model_name_base: str):
        self.model_path, self.device, self.model_type = _select_best_engine_path(
            model_name_base)

        try:
            # The YOLO class from ultralytics can load both .pt and .engine files.
            # It also intelligently handles moving the model to the correct device.
            self.model = YOLO(self.model_path)
            # For non-TensorRT models, an explicit .to(device) is good practice, though often handled by YOLO.
            if "TensorRT" not in self.model_type:
                self.model.to(self.device)

            self.class_names = self.model.names
            logger.info(
                f"InferenceEngine initialized with {self.model_type} on device '{self.device}'.")
            # Perform a single warmup inference to compile kernels and minimize first-frame latency.
            self._warmup()

        except Exception as e:
            logger.critical(
                f"Fatal error during InferenceEngine initialization with model '{self.model_path}': {e}")
            raise

    def _warmup(self):
        """
        Performs a single dummy inference to initialize the model, compile CUDA kernels,
        and ensure the first real inference call is fast.
        """
        logger.info("Performing warmup inference...")
        dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
        try:
            self.infer_single_frame(dummy_image)
            logger.info("Warmup complete. Engine is ready.")
        except Exception as e:
            logger.error(f"Warmup inference failed: {e}")

    def infer_single_frame(self, frame: np.ndarray) -> Optional[Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, np.ndarray]]:
        """
        Runs inference on a single video frame with minimal latency.

        Args:
            frame (np.ndarray): The input image in BGR format (as returned by OpenCV).

        Returns:
            A tuple containing (boxes, masks, class_ids, scores).
            - boxes: (N, 4) numpy array of bounding boxes [x1, y1, x2, y2].
            - masks: (N, H, W) numpy array of segmentation masks, or None.
            - class_ids: (N,) numpy array of integer class IDs.
            - scores: (N,) numpy array of confidence scores.
            Returns (None, None, None, None) if no objects are detected.
        """
        try:
            # Key flags for real-time performance:
            # stream=False: Process a single image, don't treat it as a generator.
            # verbose=False: Suppress console output for every frame.
            results = self.model(frame, stream=False,
                                 verbose=False, device=self.device)

            result = results[0]  # The result for the single image

            if result.boxes is None or len(result.boxes) == 0:
                return None, None, None, None

            # Efficiently extract data and move to CPU
            boxes = result.boxes.xyxy.cpu().numpy()
            masks = result.masks.data.cpu().numpy() if result.masks is not None else None
            class_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
            scores = result.boxes.conf.cpu().numpy()

            return boxes, masks, class_ids, scores

        except Exception as e:
            logger.error(
                f"An error occurred during single-frame inference: {e}", exc_info=True)
            return None, None, None, None
