from qtpy import QtCore, QtGui
from threading import Lock
from collections import deque
import time
import os
import qimage2ndarray
import numpy as np
from pathlib import Path
from annolid.utils.logger import logger
from qtpy.QtCore import Signal, Qt
from annolid.data.videos import extract_frames_from_videos
from qtpy.QtCore import QThread


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
