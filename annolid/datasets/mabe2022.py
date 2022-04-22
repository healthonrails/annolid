"""Module to load and handle MABE 2022: MOUSE-TRIPLETS - VIDEO DATA
https://www.aicrowd.com/showcase/getting-started-mouse-triplets-video-data

"""

import os
import cv2
import numpy as np

import torch
import torchvision.transforms as T


def load_user_train(path_to_train='user_train.npy'):
    if os.path.exists(path_to_train):
        user_train_data = np.load(path_to_train,
                                  allow_pickle=True).item()
        return user_train_data
    else:
        return


class MabeVideoDataset(torch.utils.data.Dataset):
    """
    Reads all frames from video files with frame skip
    modified from: 
    https://www.aicrowd.com/showcase/getting-started-mouse-triplets-video-data
    """

    def __init__(self,
                 videofolder,
                 frame_number_map,
                 frame_skip,
                 num_frames_per_clip=1800,
                 image_size=(224, 224)
                 ):
        """
        Initializing the dataset with images and labels
        """
        self.videofolder = videofolder
        self.frame_number_map = frame_number_map
        self.video_keys = list(frame_number_map.keys())
        # For every frame read, skip <frame_skip> frames after that
        self.frame_skip = frame_skip
        self.num_frames_per_clip = num_frames_per_clip
        self.image_size = image_size
        self.num_frames_per_clip = num_frames_per_clip

        assert num_frames_per_clip % (
            frame_skip + 1) == 0, "frame_skip+1 should exactly divide frame number map"
        self.num_frames = num_frames_per_clip // (self.frame_skip + 1)

        self.transform = T.Compose([
            T.ToTensor(),
            # T.Resize(image_size), # Add this if using full sized videos
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.frame_number_map)

    def __getitem__(self, idx):
        video_name = self.video_keys[idx]

        video_path = os.path.join(self.videofolder, video_name + '.avi')
        if not os.path.exists(video_path):
            # raise FileNotFoundError(video_path)
            print("File not found", video_path)
            return torch.zeros((self.num_frames, 3, *self.image_size), dtype=torch.float32)

        cap = cv2.VideoCapture(video_path)
        frame_array = torch.zeros(
            (self.num_frames, 3, *self.image_size), dtype=torch.float32)

        for array_idx, frame_idx in enumerate(range(0, self.num_frames_per_clip, self.frame_skip+1)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if success:
                frame_tensor = self.transform(frame)
                frame_array[array_idx] = frame_tensor

        return frame_array
