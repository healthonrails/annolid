from __future__ import print_function, division
import os
import cv2
import torch

from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms, utils


class VideoFrameDataset(IterableDataset):
    """Video Frame dataset."""

    def __init__(self, video_file, root_dir=None, transform=None):
        """
        Args:
            video_file (string): Path to the video file.
            root_dir (string): Directory with all the videos.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.video_file = video_file
        self.root_dir = root_dir
        self.transform = transform
        self.cap = cv2.VideoCapture(self.video_file)

    def fps(self, cap):
        return round(cap.get(cv2.CAP_PROP_FPS))

    def frame_width(self, cap):
        return round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    def frame_height(self, cap):
        return round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def save(self,
             out_path,
             fps,
             frame_width,
             frame_height
             ):
        out = cv2.VideoWriter(out_path,
                              cv2.VideoWriter_fourcc(*"mp4v"),
                              fps,
                              (frame_width, frame_height))

    def transform_frame(self, frames, transform):
        with torch.no_grad():
            frames = [torch.from_numpy(frame).cuda().float()
                      for frame in frames]
            return frames, transform(torch.stack(frames, 0))

    def __iter__(self):
        cap = self.cap
        ret, old_frame = cap.read()

        num_frames = (int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        for num in range(num_frames - 1):
            ret, frame = cap.read()
            if self.transform:
                x = self.transform(old_frame)
                y = self.transform(frame)
            else:
                x = old_frame
                y = frame
            old_frame = frame.copy()

            yield x, y

    def __exit__(self, exc_type, exc_value, traceback):
        cv2.destroyAllWindows()
        self.cap.release()
