"""Module to load and handle MABE 2022: MOUSE-TRIPLETS - VIDEO DATA
https://www.aicrowd.com/showcase/getting-started-mouse-triplets-video-data

"""

from genericpath import exists
import os
import cv2
import numpy as np

import torch
import torchvision.transforms as T
from annolid.data.videos import CV2Video
from annolid.annotation.keypoints import save_labels, get_shapes


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


def pose_to_mask(pose):
    hull_mask = cv2.convexHull(pose)
    return hull_mask


def points_to_labelme(video_file,
                      seq,
                      userTrain_data,
                      dest_dir='/data/label_jsons',
                      skip_num=20,
                      number_to_class=None,
                      min_points=9
                      ):
    """convert pose keypoints to labelme json files for a video

    Args:
        video_file (str): video file path
        seq (str): squence id like xxxxx
        userTrain_data (dict): dict contains the training data
        dest_dir (str, optional): result dir contains label json files. Defaults to '/data/label_jsons'.
        skip_num (int, optional): number of frames to skip. Defaults to 20.
        number_to_class (dict, optional): dict maps numbers to class e.g. {0:lighting, 1:chase}. Defaults to None.
        min_points (int,optional): min number of points to respenset polygon. Defaults to 9
    """
    vfs = CV2Video(video_file)
    width = vfs.get_width()
    height = vfs.get_height()
    if number_to_class is None:
        number_to_class = {i: s for i, s in enumerate(
            userTrain_data['vocabulary'])}
    single_sequence = userTrain_data["sequences"][seq]
    keypoint_sequence = single_sequence['keypoints']
    annos = single_sequence['annotations']
    for i in range(vfs.total_frames()):
        if i % skip_num == 0:
            frame = vfs.load_frame(i)

            try:

                pose = keypoint_sequence[i]
            except IndexError:
                print('out of index', i, video_file)
                continue
            combined_pose = pose.reshape(-1, 2)
            hull_mask = pose_to_mask(combined_pose)
            if annos[0, i] >= 1:
                label = number_to_class[annos[0, i]]
            elif annos[1, i] >= 1:
                label = number_to_class[annos[1, i]]
            else:
                label = 'background'
            points = hull_mask.squeeze()
            if points.shape[0] >= min_points:
                shapes = get_shapes(points, label)
                filename = os.path.join(dest_dir, seq + '_' + str(i) + '.png')
                if not os.path.exists(filename):
                    cv2.imwrite(filename, frame)
                save_labels(filename.replace('.png', '.json'),
                            filename, [shapes], height, width)


def main(user_train_path,
         video_folder,
         dest_dir='/data/label_jsons'):
    userTrain_data = load_user_train(user_train_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    seqs = list(userTrain_data['sequences'].keys())
    for seq in seqs:
        video_file = os.path.join(video_folder, seq + '.avi')
    if os.path.exists(video_file):
        points_to_labelme(video_file, seq, userTrain_data, dest_dir)
    else:
        print('not existing', video_file)
