from .transforms import ResizeCenterCropNormalize
import os
import cv2
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Tuple, Dict, Callable, List, Optional
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Configuration (Consider moving these to a separate configuration file)
NUM_FRAMES = 30
CLIP_LEN = 1  # in seconds
FPS = 30


class BehaviorDataset(Dataset):
    def __init__(self, video_folder: str, num_frames: int = NUM_FRAMES, clip_len: float = CLIP_LEN,
                 fps: int = FPS, transform: Optional[Callable] = None, video_ext: str = ".mpg",
                 split: str = 'train', val_ratio: float = 0.2, random_seed: int = 42):
        """
        Initializes the dataset with optional training/validation split.

        :param video_folder: Path to the folder containing video files.
        :param num_frames: Number of frames to extract per video.
        :param clip_len: Length of video clips in seconds.
        :param fps: Frames per second of the video.
        :param transform: Callable transformation to apply to frames.
        :param video_ext: Video file extension (e.g., ".mpg").
        :param split: Either 'train' or 'val' to specify the dataset split.
        :param val_ratio: Ratio of data for validation.
        :param random_seed: Random seed for reproducibility.
        """
        self.video_folder = video_folder
        self.num_frames = num_frames
        self.clip_len = clip_len
        self.fps = fps
        self.video_ext = video_ext
        self.transform = transform or ResizeCenterCropNormalize()
        self.split = split
        self.val_ratio = val_ratio
        self.random_seed = random_seed

        self.video_files, self.all_annotations = self.load_annotations()
        if not self.video_files or not self.all_annotations:
            raise ValueError("No video/annotation files found. Check paths and data.")

        self.label_mapping = self.create_label_mapping()
        self.indices = self.create_split_indices(split, val_ratio, random_seed)

        if len(self.indices) == 0:
            raise ValueError("No samples after split. Check split ratio and data.")


    def create_split_indices(self, split: str, val_ratio: float, random_seed: int) -> List[int]:
        """
        Splits dataset indices for training and validation using stratified sampling.
        """
        np.random.seed(random_seed)
        all_indices = np.arange(sum(len(annotations) for annotations in self.all_annotations.values()))
        labels = []
        for video_file, annotations in self.all_annotations.items():
            for _, row in annotations.iterrows():
                behavior = row.get("Behavior", "unlabeled")
                labels.append(self.label_mapping.get(behavior, self.label_mapping["unlabeled"]))

        train_indices, val_indices = train_test_split(
            all_indices, test_size=val_ratio, stratify=labels, random_state=random_seed
        )

        return train_indices if split == 'train' else val_indices

    def get_num_classes(self) -> int:
        return len(self.label_mapping)


    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        """
        Fetches data for the given index within the split. Handles potential errors.
        """
        index = self.indices[idx]

        video_file, row_index, annotations = self.get_video_and_row_index(index)

        video_path = video_file # Path is already complete 
        behavior = annotations.iloc[row_index].get("Behavior", "unlabeled")
        label = self.label_mapping.get(behavior, self.label_mapping["unlabeled"])
        frames = self.load_video_frames(video_path, row_index, annotations)


        if frames is None:
           raise ValueError(f"Failed to load frames for video {video_path} at row {row_index}")  # Or handle differently

        if self.transform:
            frames = torch.stack([self.transform(frame) for frame in frames])

        return frames, label, video_path

    def fetch_data(self, index: int) -> Optional[Tuple[torch.Tensor, int, str]]:
        try:
            video_file, row_index, annotations = self.get_video_and_row_index(
                index)
            video_path = os.path.join(self.video_folder, video_file)

            # Default to "unlabeled" if "Behavior" column is missing
            behavior = annotations.iloc[row_index]["Behavior"] if "Behavior" in annotations.columns else "unlabeled"
            label = self.label_mapping.get(
                behavior, self.label_mapping["unlabeled"])

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

    def load_annotations(self) -> Tuple[List[str], Dict[str, pd.DataFrame]]:
        video_files = []
        all_annotations = {}

        for root, _, files in os.walk(self.video_folder):
            for file in files:
                if file.endswith(self.video_ext):
                    video_path = os.path.join(root, file)
                    csv_path = os.path.splitext(video_path)[0] + ".csv"
                    video_files.append(video_path)

                    try:
                        all_annotations[video_path] = pd.read_csv(csv_path)
                    except FileNotFoundError:
                        logger.warning(
                            f"CSV not found for {video_path}. Skipping.")
                    except pd.errors.ParserError:
                        logger.warning(
                            f"Error parsing CSV for {video_path}. Skipping.")

        return video_files, all_annotations

    def create_label_mapping(self) -> Dict[str, int]:
        behaviors = set()
        for annotations in self.all_annotations.values():
            behaviors.update(annotations["Behavior"].unique())
        label_mapping = {behavior: i for i, behavior in enumerate(behaviors)}
        label_mapping["unlabeled"] = len(label_mapping)
        return label_mapping

    def load_video_frames(self, video_path: str, row_index: int, annotations: pd.DataFrame) -> Optional[torch.Tensor]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Could not open video: {video_path}")
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        try:
            start_frame = int(
                annotations.iloc[row_index]["Trial time"] * self.fps)
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
        current_index = 0
        for video_file, annotations in self.all_annotations.items():
            if index < current_index + len(annotations):
                row_index = index - current_index
                return video_file, row_index, annotations
            current_index += len(annotations)

        raise IndexError(f"Index {index} out of range")
