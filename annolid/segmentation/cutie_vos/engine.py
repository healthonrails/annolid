# cutie_engine.py

from annolid.segmentation.cutie_vos.dependencies import (
    ensure_cutie_runtime_dependencies,
)

ensure_cutie_runtime_dependencies()

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Iterator, Tuple, Union

from omegaconf import open_dict
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Assuming these are from the original Cutie VOS repository or your adaptation
from annolid.segmentation.cutie_vos.model.cutie import CUTIE
from annolid.segmentation.cutie_vos.inference.inference_core import InferenceCore
from annolid.segmentation.cutie_vos.interactive_utils import (
    image_to_torch,
    resize_frame_for_inference,
    torch_prob_to_numpy_mask,
)

from annolid.utils.logger import logger
from annolid.utils.devices import get_device  # For device selection
from annolid.utils.model_assets import ensure_cached_model_asset


class CutieEngine:
    """
    A core processing engine for the Cutie VOS model.
    It loads the model, manages the InferenceCore, and processes video frames
    to yield predicted masks for a given segment.
    It does not handle file I/O for annotations or initial mask creation from shapes.
    """
    _REMOTE_MODEL_URL = "https://github.com/hkchengrex/Cutie/releases/download/v1.0/cutie-base-mega.pth"
    _MD5 = "a6071de6136982e396851903ab4c083a"
    _MODEL_FILENAME = 'cutie-base-mega.pth'

    def __init__(self,
                 # e.g., mem_every, t_max_value
                 cutie_config_overrides: Optional[Dict] = None,
                 device: Optional[Union[torch.device, str]] = None,
                 model_weights_path: Optional[str] = None):

        self.device = device or get_device()
        logger.info(f"CutieEngine: Initializing on device: {self.device}")

        self.cfg = self._load_hydra_config(cutie_config_overrides)
        self.model_weights_path = model_weights_path or self._get_model_weights_path()

        self.cutie_model: Optional[CUTIE] = None
        self.inference_core: Optional[InferenceCore] = None

        try:
            self.cutie_model = self._load_cutie_model()
            self.inference_core = InferenceCore(self.cutie_model, cfg=self.cfg)
            logger.info(
                "CutieEngine: Model and InferenceCore initialized successfully.")
        except Exception as e:
            logger.error(
                f"CutieEngine: Failed to initialize model or InferenceCore: {e}", exc_info=True)
            # Propagate error or handle gracefully
            raise

    def _device_type(self) -> str:
        """Return device type ('cuda', 'mps', 'cpu') for autocast/gating."""
        if isinstance(self.device, torch.device):
            return self.device.type
        device_str = str(self.device).lower()
        if "cuda" in device_str:
            return "cuda"
        if "mps" in device_str:
            return "mps"
        return "cpu"

    def reset_inference_core(self) -> None:
        """Reset inference state (objects + memory) while keeping model weights loaded."""
        if self.cutie_model is None:
            raise RuntimeError(
                "CutieEngine reset requested before model initialization.")
        self.inference_core = InferenceCore(self.cutie_model, cfg=self.cfg)

    def _get_model_weights_path(self) -> str:
        """Determines and potentially downloads the model weights."""
        # Store weights in a known location, e.g., user's cache or package data
        package_dir = Path(__file__).resolve().parent
        # Example: annolid/segmentation/cutie_vos/weights/
        weights_dir = package_dir / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        model_path = weights_dir / self._MODEL_FILENAME

        if not model_path.exists():
            logger.info(
                f"Cutie model weights not found at {model_path}. Downloading...")
            try:
                ensure_cached_model_asset(
                    file_name=model_path.name,
                    url=self._REMOTE_MODEL_URL,
                    expected_md5=self._MD5,
                    cache_dir=weights_dir,
                )
                logger.info(f"Cutie model downloaded to {model_path}")
            except Exception as e:
                logger.error(
                    f"Failed to download Cutie model: {e}", exc_info=True)
                raise FileNotFoundError(
                    f"Cutie model weights could not be downloaded to {model_path}")
        return str(model_path)

    # OmegaConf's DictConfig is often returned
    def _load_hydra_config(self, overrides: Optional[Dict] = None) -> Dict:
        """Loads the base Hydra config for Cutie and applies overrides."""
        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        # Define the relative path to your config directory FROM THE LOCATION OF THIS FILE
        # If 'config' directory is in the same directory as 'cutie_engine.py':
        relative_config_path = "config"
        # If 'config' directory is one level up from 'cutie_engine.py':
        # relative_config_path = "../config_parent_dir_name" # Adjust as needed

        try:
            # The config_path for initialize() should be relative.
            # Hydra will search from the location of the calling Python file if an absolute path
            # isn't established through other means (like being inside a Hydra app).
            initialize(
                version_base='1.3.2',
                config_path=relative_config_path,  # Use the relative path string
                job_name="cutie_engine_config"
            )
            # Loads 'eval_config.yaml' from 'config_path'
            cfg = compose(config_name="eval_config")
        except Exception as e:
            logger.error(f"Hydra initialization failed. Ensure '{relative_config_path}/eval_config.yaml' exists "
                         f"relative to 'cutie_engine.py' or is discoverable by Hydra. Error: {e}", exc_info=True)
            raise  # Re-raise the exception to stop execution if config is critical

        if overrides:
            with open_dict(cfg):  # Allow modifications
                for key, value in overrides.items():
                    try:
                        # For nested keys, use dot notation: OmegaConf.update(cfg, "data.batch_size", value)
                        # For simple top-level keys:
                        if '.' in key:  # Basic check for nested key attempt
                            parts = key.split('.')
                            current_level = cfg
                            for i, part in enumerate(parts[:-1]):
                                if part not in current_level:
                                    logger.warning(
                                        f"Nested override key part '{part}' in '{key}' not found.")
                                    current_level = None
                                    break
                                current_level = current_level[part]
                            if current_level is not None and parts[-1] in current_level:
                                current_level[parts[-1]] = value
                            elif current_level is not None:  # Key not present at leaf
                                logger.warning(
                                    f"Leaf override key part '{parts[-1]}' in '{key}' not found.")
                            # else: already warned
                        elif key in cfg:
                            cfg[key] = value
                        else:
                            logger.warning(
                                f"Override key '{key}' not found in base Cutie config.")
                    except Exception as e:
                        logger.error(
                            f"Error applying override '{key}={value}': {e}")
        return cfg

    def _load_cutie_model(self) -> CUTIE:
        """Loads the CUTIE PyTorch model and its weights."""
        if not self.model_weights_path or not Path(self.model_weights_path).exists():
            raise FileNotFoundError(
                f"Cutie model weights not found at: {self.model_weights_path}")

        with open_dict(self.cfg):  # Ensure weights path in cfg is correct
            self.cfg['weights'] = self.model_weights_path

        model = CUTIE(self.cfg).to(self.device).eval()

        try:
            logger.debug(f"Loading Cutie weights from: {self.cfg.weights}")
            loaded_weights = torch.load(
                self.cfg.weights,
                map_location=self.device,
                weights_only=True,
            )

            # Handle potential nesting of weights (e.g. 'model', 'state_dict')
            if 'model' in loaded_weights:
                weights_to_load = loaded_weights['model']
            elif 'state_dict' in loaded_weights:
                weights_to_load = loaded_weights['state_dict']
            else:
                weights_to_load = loaded_weights

            # Assumes CUTIE class has a load_weights method
            model.load_weights(weights_to_load)
            # Or use model.load_state_dict(weights_to_load) directly
            # if load_weights is just a wrapper.
        except Exception as e:
            logger.error(
                f"Error loading Cutie model weights: {e}", exc_info=True)
            raise
        return model

    def clear_memory(self):
        """Clears the memory of the InferenceCore."""
        if self.inference_core:
            self.inference_core.clear_memory()
            logger.info("CutieEngine: InferenceCore memory cleared.")

    def clear_non_permanent_memory(self):
        """Clears only the non-permanent (working) memory of InferenceCore."""
        if self.inference_core:
            self.inference_core.clear_non_permanent_memory()
            logger.info(
                "CutieEngine: InferenceCore non-permanent memory cleared.")

    def process_frames(self,
                       video_capture: cv2.VideoCapture,
                       start_frame_index: int,    # Actual video frame index to start reading from
                       # NumPy array (H, W), integer object IDs, 0 for bg
                       initial_mask_np: np.ndarray,
                       # Number of distinct objects in initial_mask_np (excluding bg)
                       num_objects_in_mask: int,
                       # How many frames to process *including* the start_frame_index
                       frames_to_propagate: int,
                       pred_worker: Optional[object] = None,
                       reset_core: bool = True,
                       ) -> Iterator[Tuple[int, np.ndarray, np.ndarray]]:  # (frame_idx, frame_bgr, pred_mask_np)
        """
        Processes a sequence of frames from the video_capture.

        Args:
            video_capture: Opened cv2.VideoCapture object for the video.
            start_frame_index: The video frame index from which to start processing.
                               The initial_mask_np corresponds to this frame.
            initial_mask_np: The starting mask (H, W, object IDs).
            num_objects_in_mask: Number of objects in the initial mask.
            frames_to_propagate: Total number of frames to process for this segment.
            pred_worker: Optional object with an `is_stopped()` method for early termination.

        Yields:
            Tuple[int, np.ndarray, np.ndarray]: (current_video_frame_index, frame_bgr, predicted_numpy_mask)
                                    The mask is an object ID mask (H, W).
        """
        if not self.cutie_model:
            logger.error(
                "CutieEngine: Model not initialized.")
            return  # Or raise exception

        if not video_capture.isOpened():
            logger.error("CutieEngine: VideoCapture is not open.")
            return

        if reset_core or self.inference_core is None:
            self.reset_inference_core()

        initial_mask_np = np.asarray(initial_mask_np)
        if initial_mask_np.ndim != 2:
            raise ValueError(
                "CutieEngine initial mask must be a 2D indexed mask; "
                f"received shape {initial_mask_np.shape}."
            )
        initial_object_ids = [
            int(value) for value in np.unique(initial_mask_np) if int(value) != 0
        ]
        if not initial_object_ids:
            raise ValueError("CutieEngine initial mask has no foreground objects.")
        if len(initial_object_ids) != int(num_objects_in_mask):
            logger.warning(
                "CutieEngine initial object count mismatch: mask contains %s, caller reported %s. Using mask ids.",
                len(initial_object_ids),
                num_objects_in_mask,
            )
        uses_temporary_ids_as_object_ids = initial_object_ids == list(
            range(1, len(initial_object_ids) + 1)
        )

        try:
            max_internal_size = int(self.cfg.max_internal_size)
        except (AttributeError, TypeError, ValueError):
            max_internal_size = -1

        total_video_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_processed_in_this_call = 0
        current_video_frame_idx = start_frame_index

        with torch.inference_mode():
            with torch.amp.autocast('cuda', enabled=bool(self.cfg.amp) and self._device_type() == 'cuda'):
                if current_video_frame_idx < 0:
                    current_video_frame_idx = 0
                if current_video_frame_idx >= total_video_frames:
                    logger.info(
                        "CutieEngine: Start frame %s is beyond video length %s.",
                        current_video_frame_idx,
                        total_video_frames,
                    )
                    return

                if int(video_capture.get(cv2.CAP_PROP_POS_FRAMES)) != current_video_frame_idx:
                    video_capture.set(cv2.CAP_PROP_POS_FRAMES,
                                      current_video_frame_idx)

                while True:
                    if pred_worker and pred_worker.is_stopped():
                        logger.info(
                            f"CutieEngine: Processing stopped by worker at frame {current_video_frame_idx}.")
                        break

                    if frames_processed_in_this_call >= frames_to_propagate:
                        logger.info(
                            f"CutieEngine: Reached 'frames_to_propagate' ({frames_to_propagate}). Segment ended at frame {current_video_frame_idx-1}.")
                        break

                    if current_video_frame_idx >= total_video_frames:
                        logger.info(
                            f"CutieEngine: Reached end of video ({current_video_frame_idx}) while processing segment.")
                        break

                    ret, frame_bgr = video_capture.read()

                    if not ret or frame_bgr is None:
                        logger.warning(
                            f"CutieEngine: Could not read frame {current_video_frame_idx}. Ending segment.")
                        break

                    frame_rgb = cv2.cvtColor(
                        frame_bgr, cv2.COLOR_BGR2RGB)  # Cutie expects RGB
                    output_size = tuple(frame_rgb.shape[:2])
                    frame_for_inference = resize_frame_for_inference(
                        frame_rgb,
                        max_internal_size,
                    )
                    frame_torch = image_to_torch(
                        frame_for_inference,
                        device=self.device,
                    )

                    if frames_processed_in_this_call == 0:  # First frame of this segment processing call
                        if tuple(initial_mask_np.shape) != output_size:
                            raise ValueError(
                                "CutieEngine initial mask size does not match its seed frame: "
                                f"mask={tuple(initial_mask_np.shape)}, frame={output_size}."
                            )
                        mask_torch = torch.from_numpy(
                            np.ascontiguousarray(initial_mask_np)
                        )

                        predicted_probs_torch = self.inference_core.step(
                            frame_torch,
                            mask_torch,
                            objects=initial_object_ids,
                            idx_mask=True,
                            force_permanent=True,
                            return_index_mask=True,
                            output_size=output_size,
                        )
                    else:  # Subsequent frames
                        predicted_probs_torch = self.inference_core.step(
                            frame_torch,
                            return_index_mask=True,
                            output_size=output_size,
                        )

                    if uses_temporary_ids_as_object_ids:
                        predicted_object_ids = predicted_probs_torch
                    else:
                        predicted_object_ids = self.inference_core.output_prob_to_mask(
                            predicted_probs_torch
                        )
                    predicted_mask_np = torch_prob_to_numpy_mask(
                        predicted_object_ids)

                    yield current_video_frame_idx, frame_bgr, predicted_mask_np

                    frames_processed_in_this_call += 1
                    current_video_frame_idx += 1

        logger.info(
            f"CutieEngine: Processed {frames_processed_in_this_call} frames for segment starting at {start_frame_index}.")

    def cleanup(self):
        """Releases resources, particularly GPU memory if applicable."""
        logger.debug("CutieEngine: Cleaning up.")
        if hasattr(self, 'inference_core') and self.inference_core is not None:
            del self.inference_core
            self.inference_core = None
        if hasattr(self, 'cutie_model') and self.cutie_model is not None:
            del self.cutie_model
            self.cutie_model = None

        if self._device_type() == 'cuda':
            torch.cuda.empty_cache()
