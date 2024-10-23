from .transforms import ResizeCenterCropNormalize
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Callable, List, Optional
import logging

logger = logging.getLogger(__name__)

# Configuration (Consider moving these to a separate configuration file)
NUM_FRAMES = 30
CLIP_LEN = 1  # in seconds
FPS = 30


class BehaviorDataset(Dataset):
    def __init__(self, video_folder: str, num_frames: int = NUM_FRAMES, clip_len: float = CLIP_LEN,
                 fps: int = FPS, transform: Optional[Callable] = None, video_ext: str = ".mpg"):
        """
        Initializes the dataset with the folder containing videos and their corresponding annotations.

        :param video_folder: Path to the folder containing video files.
        :param num_frames: Number of frames to extract per video.
        :param clip_len: Length of video clips in seconds.
        :param fps: Frames per second of the video.
        :param transform: Callable transformation to apply to frames.
        :param video_ext: Video file extension (e.g., ".mpg").
        """
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.clip_len = clip_len
        self.fps = fps
        self.video_ext = video_ext
        self.transform = transform or ResizeCenterCropNormalize()
        self.video_files, self.all_annotations = self.load_annotations()
        self.label_mapping = self.create_label_mapping()

    def load_annotations(self) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        """
        Loads video files and their corresponding annotations (CSV).

        :return: A tuple of video files and a dictionary of annotations DataFrames.
        """
        video_files = [f for f in os.listdir(
            self.video_folder) if f.endswith(self.video_ext)]
        all_annotations = {}

        for video_file in video_files.copy():
            csv_file = os.path.splitext(video_file)[0] + ".csv"
            csv_path = os.path.join(self.video_folder, csv_file)

            try:
                all_annotations[video_file] = pd.read_csv(csv_path)
            except FileNotFoundError:
                logger.warning(
                    f"CSV file not found for {video_file}. Skipping.")
                video_files.remove(video_file)
            except pd.errors.ParserError:
                logger.warning(
                    f"Error parsing CSV file for {video_file}. Skipping.")
                video_files.remove(video_file)

        return video_files, all_annotations

    def create_label_mapping(self) -> Dict[str, int]:
        """
        Creates a mapping of behavior labels to integers.

        :return: Dictionary mapping behaviors to indices.
        """
        behaviors = set()
        for annotations in self.all_annotations.values():
            behaviors.update(annotations["Behavior"].unique())
        return {behavior: i for i, behavior in enumerate(behaviors)}

    def __len__(self) -> int:
        """
        Returns the total number of annotation rows across all videos.

        :return: Total number of data points.
        """
        return sum(len(annotations) for annotations in self.all_annotations.values())

    def __getitem__(self, index: int) -> Optional[Tuple[torch.Tensor, int, str]]:
        """
        Retrieves the data (frames, label, and video path) for a given index.

        :param index: Index in the dataset.
        :return: A tuple containing frames, the label, and the video path, or None if an error occurs.
        """
        try:
            video_file, row_index, annotations = self.get_video_and_row_index(
                index)
            video_path = os.path.join(self.video_folder, video_file)

            trial_time = annotations.iloc[row_index,
                                          annotations.columns.get_loc("Trial time")]
            behavior = annotations.iloc[row_index,
                                        annotations.columns.get_loc("Behavior")]
            label = self.label_mapping[behavior]

            frames = self.load_video_frames(video_path, row_index, annotations)

            if frames is None:
                return None

            if self.transform:
                frames = torch.stack([self.transform(frame)
                                     for frame in frames])

            return frames, label, video_path
        except Exception as e:
            logger.error(f"Error fetching item at index {index}: {e}")
            return None

    def load_video_frames(self, video_path: str, row_index: int, annotations: pd.DataFrame) -> Optional[torch.Tensor]:
        """
        Loads the video frames for a given video and row index.

        :param video_path: Path to the video file.
        :param row_index: Row index in the annotations DataFrame.
        :param annotations: DataFrame containing the annotations for the video.
        :return: A tensor containing the frames, or None if an error occurs.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            start_frame = int(
                annotations.iloc[row_index, annotations.columns.get_loc("Trial time")] * self.fps)
        except KeyError as e:
            logger.error(
                f"Missing 'Trial time' column in CSV for {video_path}: {e}")
            return None

        start_frame = max(0, min(start_frame, total_frames -
                          int(self.clip_len * self.fps)))
        end_frame = start_frame + int(self.clip_len * self.fps)

        frames = []
        frame_indices = torch.linspace(
            start_frame, end_frame - 1, self.num_frames, dtype=torch.int).tolist()

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(torch.from_numpy(
                    frame).permute(2, 0, 1).float() / 255.0)
            else:
                logger.warning(f"Failed to read frame {i} from {video_path}")
                cap.release()
                return None

        cap.release()

        if len(frames) != self.num_frames:
            logger.warning(
                f"Expected {self.num_frames} frames, but got {len(frames)} from {video_path}")
            return None

        return torch.stack(frames)

    def get_video_and_row_index(self, index: int) -> Tuple[str, int, pd.DataFrame]:
        """
        Maps the index to a specific video and annotation row.

        :param index: Dataset index.
        :return: A tuple containing the video file name, row index, and annotation DataFrame.
        :raises IndexError: If the index is out of range.
        """
        current_index = 0
        for video_file, annotations in self.all_annotations.items():
            if index < current_index + len(annotations):
                row_index = index - current_index
                return video_file, row_index, annotations
            current_index += len(annotations)

        raise IndexError(f"Index {index} out of range")
