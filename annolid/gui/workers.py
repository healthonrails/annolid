from qtpy import QtCore, QtGui
from threading import Lock
from collections import deque
import time
import os
import logging
import glob
import qimage2ndarray
import numpy as np
import torch
from annolid.gui.label_file import LabelFile
from pathlib import Path
from annolid.utils.logger import logger
from qtpy.QtCore import Signal, QObject
from annolid.data.videos import extract_frames_from_videos
from qtpy.QtCore import QThread
from annolid.segmentation.SAM.edge_sam_bg import VideoProcessor
from annolid.utils.files import find_manual_labeled_json_files, get_frame_number_from_json
from annolid.annotation.labelme2csv import convert_json_to_csv
import gc


class PredictionWorker(QObject):
    stop_signal = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stopped = False

    def is_stopped(self):
        return self._stopped

    def stop(self):
        self._stopped = True
        self.stop_signal.emit()


class TrackAllWorker(QThread):
    """Worker thread to process videos with Cutie tracking, ensuring one processor per video."""
    progress = Signal(int, str)  # Progress percentage and message
    finished = Signal(str)      # Completion message
    error = Signal(str)         # Error message

    # video_path, output_folder_path
    video_processing_started = Signal(str, str)
    # video_path (or output_folder_path)
    video_processing_finished = Signal(str)

    def __init__(self, video_paths, config=None, parent=None):
        super().__init__(parent)
        self.video_paths = self._validate_video_paths(video_paths)
        self.config = config or {
            'mem_every': 5,
            'epsilon_for_polygon': 2.0,
            't_max_value': 5,
            'use_cpu_only': False,
            'auto_recovery_missing_instances': False,
            'save_video_with_color_mask': False,
            'compute_optical_flow': True,
            'has_occlusion': True
        }
        self.is_running = True
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def _validate_video_paths(self, video_paths):
        """Validate video paths and return a list of valid paths."""
        if not video_paths:
            raise ValueError("No video paths provided.")
        valid_paths = []
        for path in video_paths:
            path = Path(path)
            if path.is_file() and path.suffix.lower() in {'.mp4', '.avi', '.mov'}:
                valid_paths.append(str(path))
            else:
                self.logger.warning(
                    f"Invalid or non-existent video path: {path}")
        if not valid_paths:
            raise ValueError("No valid video files provided.")
        return valid_paths

    def select_device(self):
        """Select the appropriate device (CUDA, MPS, or CPU)."""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def cleanup_processor(self, processor, device):
        """Clean up the processor and free device memory."""
        if processor is not None:
            if processor.cutie_processor is not None:
                processor.cutie_processor = None
            del processor
        gc.collect()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                self.logger.info(f"CUDA cache clearing failed: {e}")
        elif device.type == "mps":
            try:
                torch.mps.empty_cache()
            except AttributeError:
                self.logger.info(
                    "MPS cache clearing not supported in this PyTorch version")

    def log_gpu_memory(self, video_name, stage):
        """Log device memory usage."""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self.logger.info(f"GPU Memory (CUDA - {stage} - {video_name}): "
                             f"Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            allocated = torch.mps.current_allocated_memory() / 1024**2
            self.logger.info(f"GPU Memory (MPS - {stage} - {video_name}): "
                             f"Allocated: {allocated:.2f} MB")
        else:
            self.logger.info(
                f"No GPU available ({stage} - {video_name}): Running on CPU")

    def is_video_finished(self, video_path, total_frames):
        """Check if a video is already processed."""
        video_name = Path(video_path).stem
        output_folder = Path(video_path).with_suffix('')
        csv_pattern = str(output_folder / f"{video_name}*_tracking.csv")
        csv_files = glob.glob(csv_pattern)
        if csv_files:
            self.logger.info(
                f"Found tracking CSV for {video_name}: {csv_files[0]}")
            return True
        last_frame = total_frames - 1
        json_filename = output_folder / f"{video_name}_{last_frame:09d}.json"
        return json_filename.exists()

    def process_single_video(self, video_path, idx, total_videos):
        """Process a single video with a fresh VideoProcessor."""
        video_name = Path(video_path).stem
        output_folder = Path(video_path).with_suffix('')
        output_folder.mkdir(exist_ok=True, parents=True)
        self.logger.info(
            f"Processing {video_name}: Output folder = {output_folder}")

        # --- Emit signal that a video is starting ---
        self.video_processing_started.emit(video_path, str(output_folder))

        # Log device memory before processing
        self.log_gpu_memory(video_name, "Before")

        # Select device
        device = self.select_device()
        self.logger.info(f"Using device: {device} for {video_name}")

        # Check for JSON file
        json_files = find_manual_labeled_json_files(str(output_folder))
        if not json_files:
            self.progress.emit(
                int((idx / total_videos) * 100),
                f"Skipping {video_name}: No JSON label file found."
            )
            return False

        json_file = Path(output_folder) / sorted(json_files)[-1]
        try:
            labeled_frame_number = get_frame_number_from_json(str(json_file))
            label_file = LabelFile(str(json_file), is_video_frame=True)
            valid_shapes = [
                shape for shape in label_file.shapes
                if shape.get('shape_type') == 'polygon' and len(shape.get('points', [])) >= 3
            ]
            if not valid_shapes:
                self.progress.emit(
                    int((idx / total_videos) * 100),
                    f"Skipping {video_name}: JSON file {json_file.name} has no valid polygons (≥3 points)."
                )
                return False
        except Exception as e:
            self.error.emit(f"Invalid JSON file for {video_name}: {str(e)}")
            self.logger.error(
                f"JSON error for {video_name}: {str(e)}", exc_info=True)
            return False

        # Initialize fresh VideoProcessor
        processor = None
        try:
            processor = VideoProcessor(
                video_path=str(video_path),  # Ensure string path
                model_name="Cutie",
                save_image_to_disk=False,
                device=device,
                results_folder=str(output_folder),
                **self.config
            )
            self.logger.info(
                f"Initialized VideoProcessor for {video_name} with video_path: {video_path}")
        except Exception as e:
            self.error.emit(
                f"Failed to initialize VideoProcessor for {video_name}: {str(e)}")
            self.logger.error(f"Initialization error: {str(e)}", exc_info=True)
            return False

        try:
            total_frames = processor.get_total_frames()
            if self.is_video_finished(video_path, total_frames):
                self.progress.emit(
                    int((idx / total_videos) * 100),
                    f"Skipping {video_name}: Video already processed."
                )
                return False

            # Create new PredictionWorker
            pred_worker = PredictionWorker()
            self.logger.info(
                f"Created new pred_worker for {video_name}: {id(pred_worker)}")
            try:
                processor.video_path = str(video_path)  # Ensure string path
                processor.set_pred_worker(pred_worker)
            except Exception as e:
                self.error.emit(
                    f"Failed to set pred_worker for {video_name}: {str(e)}")
                self.logger.error(
                    f"pred_worker error: {str(e)}", exc_info=True)
                return False

            # Reset Cutie processor
            try:
                processor.reset_cutie_processor(
                    mem_every=self.config['mem_every'])
                self.logger.info(f"Reset Cutie processor for {video_name}")
            except AttributeError:
                self.logger.warning(
                    f"reset_cutie_processor not implemented for {video_name}")
                processor.cutie_processor = None

            self.progress.emit(
                int((idx / total_videos) * 100),
                f"Processing {video_name}..."
            )
            # Process video frames
            start_frame = labeled_frame_number + 1
            end_frame = total_frames - 1
            self.logger.info(
                f"Processing {video_name} from frame {start_frame} to {end_frame}")

            with torch.no_grad():
                message = processor.process_video_frames(
                    start_frame=start_frame,
                    end_frame=end_frame,
                    step=1,
                    is_cutie=True,
                    mem_every=self.config['mem_every'],
                    point_tracking=False,
                    has_occlusion=self.config['has_occlusion'],
                    save_video_with_color_mask=self.config['save_video_with_color_mask']
                )
            if "No valid polygon" in message or "No label file" in message:
                self.error.emit(f"Failed to process {video_name}: {message}")
                return False

            # Convert JSON to CSV
            try:
                convert_json_to_csv(str(output_folder))
            except Exception as e:
                self.error.emit(
                    f"Failed to convert JSON to CSV for {video_name}: {str(e)}")
                self.logger.error(
                    f"CSV conversion error: {str(e)}", exc_info=True)
                return False

            self.progress.emit(
                int((idx / total_videos) * 100),
                f"Completed tracking for {video_name}."
            )
            return True

        except Exception as e:
            self.error.emit(f"Failed to process {video_name}: {str(e)}")
            self.logger.error(f"Processing error: {str(e)}", exc_info=True)
            return False
        finally:
            self.cleanup_processor(processor, device)
            self.log_gpu_memory(video_name, "After")
            self.video_processing_finished.emit(video_path)

    def run(self):
        """Process videos sequentially, ensuring one processor per video."""
        try:
            total_videos = len(self.video_paths)
            processed_videos = 0

            for idx, video_path in enumerate(self.video_paths, start=1):
                if not self.is_running:
                    self.finished.emit("Track All stopped by user.")
                    return

                # Process each video independently
                if self.process_single_video(video_path, idx, total_videos):
                    processed_videos += 1

            self.finished.emit(
                f"Track All completed. Processed {processed_videos}/{total_videos} videos."
            )
        except Exception as e:
            self.error.emit(f"Unexpected error: {str(e)}")
            self.logger.error(f"Unexpected error: {str(e)}", exc_info=True)
        finally:
            self.is_running = False

    def stop(self):
        """Stop the worker thread."""
        self.is_running = False


class LoadFrameThread(QtCore.QObject):
    """
    Thread for loading video frames with optimized performance.
    """
    res_frame = QtCore.Signal(QtGui.QImage)
    process = QtCore.Signal()
    video_loader = None

    def __init__(self, parent=None):
        """Initialize the frame loader while maintaining original interface."""
        super().__init__(parent)

        # Replace QMutex with threading.Lock for better performance
        self.working_lock = Lock()

        # Maintain same queue structure but with optimized settings
        # Limit queue size to prevent memory issues
        self.frame_queue = deque(maxlen=30)
        # Increased sample size for better averaging
        self.current_load_times = deque(maxlen=10)

        # Timing management
        self.previous_process_time = time.time()
        self.request_waiting_time = 0.033  # Default to ~30fps for better responsiveness

        # Simple frame cache to avoid reloading recent frames
        self._frame_cache = {}
        self._cache_size = 5
        self._last_frame = None  # Keep last frame for error recovery

        # Timer optimization
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._optimized_load)
        self.timer.start(16)  # ~60fps for smoother playback

    def _optimized_load(self):
        """Optimized version of load() with better error handling and caching."""
        current_time = time.time()
        self.previous_process_time = current_time

        # Quick check without lock
        if not self.frame_queue:
            return

        # Use context manager for automatic lock release
        with self.working_lock:
            if not self.frame_queue:
                return
            frame_number = self.frame_queue.pop()

        # Check cache first
        cached_frame = self._frame_cache.get(frame_number)
        if cached_frame is not None:
            self.res_frame.emit(cached_frame)
            return

        try:
            # Load and time the frame
            t_start = time.time()
            frame = self.video_loader.load_frame(frame_number)

            # Update timing metrics
            load_time = time.time() - t_start
            self.current_load_times.append(load_time)

            # Use numpy for faster average calculation
            self.request_waiting_time = np.mean(self.current_load_times)

            # Convert frame to QImage
            qimage = qimage2ndarray.array2qimage(frame)

            # Update cache
            self._update_cache(frame_number, qimage)

            # Keep last successful frame
            self._last_frame = qimage

            # Emit the frame
            self.res_frame.emit(qimage)

        except KeyError as e:
            logger.error(f"Error loading frame {frame_number}: {e}")
            self._handle_error(frame_number)
        except Exception as e:
            logger.error(f"Unexpected error loading frame {frame_number}: {e}")
            self._handle_error(frame_number)

    def _update_cache(self, frame_number: int, qimage: QtGui.QImage):
        """Update frame cache with size management."""
        self._frame_cache[frame_number] = qimage
        if len(self._frame_cache) > self._cache_size:
            # Remove oldest frame
            oldest = min(self._frame_cache.keys())
            del self._frame_cache[oldest]

    def _handle_error(self, frame_number: int):
        """Handle frame loading errors with fallback."""
        if self._last_frame is not None:
            # Use last successful frame as fallback
            self.res_frame.emit(self._last_frame)

    def request(self, frame_number):
        """Optimized request method with better queue management."""
        with self.working_lock:
            # Clear queue if too many pending requests to prevent lag
            if len(self.frame_queue) > 5:
                self.frame_queue.clear()
            self.frame_queue.appendleft(frame_number)

        t_last = time.time() - self.previous_process_time
        if t_last > self.request_waiting_time:
            self.previous_process_time = time.time()
            self.process.emit()

    # Maintain original interface name for compatibility
    load = _optimized_load


class FlexibleWorker(QtCore.QObject):
    """
    A flexible worker class that runs a given function in a separate thread.
    Provides signals to indicate the start, progress, return value, and completion of the task.
    """

    start_signal = QtCore.Signal()
    finished_signal = QtCore.Signal(object)
    result_signal = QtCore.Signal(object)
    stop_signal = QtCore.Signal()
    progress_signal = QtCore.Signal(int)

    def __init__(self, task_function, *args, **kwargs):
        """
        Initialize the FlexibleWorker with the function to run and its arguments.

        :param task_function: The function to be executed.
        :param args: Positional arguments for the function.
        :param kwargs: Keyword arguments for the function.
        """
        super().__init__()
        self._task_function = task_function
        self._args = args
        self._kwargs = kwargs
        self._is_stopped = False

        # Connect the stop signal to the stop method
        self.stop_signal.connect(self._stop)

    def run(self):
        """
        Executes the task function with the provided arguments.
        Emits signals for result and completion when done.
        """
        self._is_stopped = False
        try:
            result = self._task_function(*self._args, **self._kwargs)
            self.result_signal.emit(result)
            self.finished_signal.emit(result)
        except Exception as e:
            # Optionally handle exceptions and emit an error signal if needed
            self.finished_signal.emit(e)

    def _stop(self):
        """
        Stops the worker by setting the stop flag.
        """
        self._is_stopped = True

    def is_stopped(self):
        """
        Check if the worker has been stopped.

        :return: True if the worker is stopped, otherwise False.
        """
        return self._is_stopped

    def report_progress(self, progress):
        """
        Reports the progress of the task.

        :param progress: An integer representing the progress percentage.
        """
        self.progress_signal.emit(progress)


class FrameExtractorWorker(QThread):
    progress = Signal(int)
    finished = Signal(str)
    error = Signal(str)

    def __init__(self, videos, output_folder, num_frames=5):
        super().__init__()
        self.videos = videos
        self.output_folder = output_folder
        self.num_frames = num_frames

    def run(self):
        try:
            for idx, video in enumerate(self.videos, 1):
                extract_frames_from_videos(
                    input_folder=os.path.dirname(video),
                    output_folder=self.output_folder,
                    num_frames=self.num_frames
                )
                progress_value = int((idx / len(self.videos)) * 100)
                self.progress.emit(progress_value)
            self.finished.emit(self.output_folder)
        except Exception as e:
            self.error.emit(str(e))


class ProcessVideosWorker(QThread):
    progress = Signal(int)  # Signal to update progress
    finished = Signal(str)  # Signal when processing is complete
    error = Signal(str)     # Signal to handle errors

    def __init__(self, videos, agent, parent=None):
        super().__init__(parent)
        self.videos = videos
        self.agent = agent

    def run(self):
        try:
            total_videos = len(self.videos)
            for idx, video_path in enumerate(self.videos, start=1):
                try:
                    # Define user prompt
                    user_prompt = "Describe the main activities in this video."

                    # Process video with the agent
                    from annolid.agents import behavior_agent
                    response = behavior_agent.process_video_with_agent(
                        video_path, user_prompt, self.agent)

                    # Save response to a text file
                    response_file = Path(video_path).with_suffix('.txt')
                    with open(response_file, "w") as f:
                        f.write(response)

                    # Emit progress
                    progress = int((idx / total_videos) * 100)
                    self.progress.emit(progress)

                except Exception as e:
                    self.error.emit(f"Error processing {video_path}: {str(e)}")

            # Notify completion
            self.finished.emit("Processing complete.")
        except Exception as e:
            self.error.emit(str(e))
