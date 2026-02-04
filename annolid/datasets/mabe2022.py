"""Module to load and handle Mabe 2022: MOUSE-TRIPLETS - VIDEO DATA
https://www.aicrowd.com/showcase/getting-started-mouse-triplets-video-data

"""

import os
import cv2
import numpy as np

import torch
import torchvision.transforms as T
from annolid.gui.shape import Shape
from annolid.data.videos import CV2Video
from annolid.annotation.keypoints import save_labels


def load_user_train(path_to_train="user_train.npy"):
    if os.path.exists(path_to_train):
        user_train_data = np.load(path_to_train, allow_pickle=True).item()
        return user_train_data
    else:
        return


def load_keypoints(path_to_keypoints="submission_keypoints.npy"):
    if os.path.exists(path_to_keypoints):
        keypoints = np.load(path_to_keypoints, allow_pickle=True).item()
        return keypoints


def get_body_part_dist(submission_keypoints, frame_id, sk, this_part, other_part):
    """Euclidean distances between two body parts in the given frame

    Args:
        submission_keypoints (dict): submission keypoints
        frame_id (int): frame id
        sk (str): sequence key
        this_part (int): body part number
        other_part (int): body part number

    Returns:
        np.array :  array of distances
    """
    kpts = submission_keypoints["sequences"][sk]["keypoints"]
    kpts_this_part = kpts[frame_id, :, this_part, :]
    kpts_other_part = kpts[frame_id, :, other_part, :]
    dist0_1 = np.linalg.norm(kpts_this_part[0, :] - kpts_other_part[1, :])
    dist1_0 = np.linalg.norm(kpts_this_part[1, :] - kpts_other_part[0, :])
    dist0_2 = np.linalg.norm(kpts_this_part[0, :] - kpts_other_part[2, :])
    dist2_0 = np.linalg.norm(kpts_this_part[2, :] - kpts_other_part[0, :])
    dist1_2 = np.linalg.norm(kpts_this_part[1, :] - kpts_other_part[2, :])
    dist2_1 = np.linalg.norm(kpts_this_part[2, :] - kpts_other_part[1, :])
    return np.array([dist0_1, dist1_0, dist0_2, dist2_0, dist1_2, dist2_1])


def keypoints_to_bbox(keypoints, padbbox=50, crop_size=512, scale_factor=224 / 512):
    """
    Estimate bboxes from keypoints
    """
    bboxes = []
    for frame_number in range(len(keypoints)):
        all_coords = np.int32(keypoints[frame_number].reshape(-1, 2))
        min_vals = (
            max(np.min(all_coords[:, 0]) - padbbox, 0),
            max(np.min(all_coords[:, 1]) - padbbox, 0),
        )
        max_vals = (
            min(np.max(all_coords[:, 0]) + padbbox, crop_size),
            min(np.max(all_coords[:, 1]) + padbbox, crop_size),
        )
        bbox = (*min_vals, *max_vals)
        bbox = np.array(bbox)
        bbox = np.int32(bbox * scale_factor)
        bboxes.append(bbox)
    return np.array(bboxes)


def keypoints_dist_from_prev_frame(keypoint_sequence):
    kshape = keypoint_sequence.shape
    kpts_diff = (
        keypoint_sequence[1:, :, :, :] - keypoint_sequence[0 : kshape[0] - 1, :, :, :]
    )
    _kpts_diff = np.zeros(keypoint_sequence.shape)
    _kpts_diff[1:, :, :, :] = kpts_diff
    return _kpts_diff


def number_to_keypoint_names(labels_text="./annolid/annotation/mabe_2022_labels.txt"):
    """load MABA2022 keypoints names

    Args:
        labels_text (str, optional): labels file contains the name of keypoints
        . Defaults to './annolid/annotation/mabe_2022_labels.txt'.

    Returns:
        dict: dict with id and name pairs
    """
    num_to_keypoints = {}
    with open(labels_text) as ml:
        names = list(ml.readlines())
        for i, name in enumerate(names[2:]):
            num_to_keypoints[i] = name.strip()
    return num_to_keypoints


def _polygon_shape_from_points(points, label_name, scale_factor=224 / 512):
    shape = Shape(label=label_name, shape_type="polygon", flags={})
    for x, y in points * scale_factor:
        if x > 0 and y > 0:
            shape.addPoint((int(y), int(x)))
    return shape


def keypoints_to_shapes(
    poses, animal_name="mouse", estimate_animal_mask=True, scale_factor=224 / 512
):
    keypoint_names = number_to_keypoint_names()
    label_list = []
    for aid, pose in enumerate(poses):
        if estimate_animal_mask:
            try:
                hull_points = pose_to_mask(pose)
                if len(hull_points) >= 4:
                    animal_shape = _polygon_shape_from_points(
                        hull_points.squeeze(), "_".join([animal_name, str(aid)])
                    )
                    label_list.append(animal_shape)

            except TypeError as err:
                print(err)

        for pid, point in enumerate(pose):
            shape = Shape(label=keypoint_names[pid], shape_type="point", flags={})
            x = point[1] * scale_factor
            y = point[0] * scale_factor
            shape.addPoint((x, y))
            label_list.append(shape)
    return label_list


class MabeVideoDataset(torch.utils.data.Dataset):
    """
    Reads all frames from video files with frame skip
    modified from:
    https://www.aicrowd.com/showcase/getting-started-mouse-triplets-video-data
    """

    def __init__(
        self,
        videofolder,
        frame_number_map,
        frame_skip,
        num_frames_per_clip=1800,
        image_size=(224, 224),
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

        assert num_frames_per_clip % (frame_skip + 1) == 0, (
            "frame_skip+1 should exactly divide frame number map"
        )
        self.num_frames = num_frames_per_clip // (self.frame_skip + 1)

        self.transform = T.Compose(
            [
                T.ToTensor(),
                # T.Resize(image_size), # Add this if using full sized videos
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.frame_number_map)

    def __getitem__(self, idx):
        video_name = self.video_keys[idx]

        video_path = os.path.join(self.videofolder, video_name + ".avi")
        if not os.path.exists(video_path):
            # raise FileNotFoundError(video_path)
            print("File not found", video_path)
            return torch.zeros(
                (self.num_frames, 3, *self.image_size), dtype=torch.float32
            )

        cap = cv2.VideoCapture(video_path)
        frame_array = torch.zeros(
            (self.num_frames, 3, *self.image_size), dtype=torch.float32
        )

        for array_idx, frame_idx in enumerate(
            range(0, self.num_frames_per_clip, self.frame_skip + 1)
        ):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            success, frame = cap.read()
            if success:
                frame_tensor = self.transform(frame)
                frame_array[array_idx] = frame_tensor

        return frame_array


def pose_to_mask(pose):
    hull_mask = cv2.convexHull(pose)
    return hull_mask


def points_to_labelme(
    video_file,
    seq,
    userTrain_data,
    dest_dir="/data/label_jsons",
    skip_num=20,
    number_to_class=None,
    min_points=9,
):
    """convert pose keypoints to labelme json files for a video

    Args:
        video_file (str): video file path
        seq (str): sequence id like xxxxx
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
        number_to_class = {i: s for i, s in enumerate(userTrain_data["vocabulary"])}
    single_sequence = userTrain_data["sequences"][seq]
    keypoint_sequence = single_sequence["keypoints"]
    annos = single_sequence["annotations"]
    for i in range(vfs.total_frames()):
        if i % skip_num == 0 or annos[0, i] == 1:
            frame = vfs.load_frame(i)
            try:
                pose = keypoint_sequence[i]
            except IndexError:
                print("out of index", i, video_file)
                continue
            label_list = keypoints_to_shapes(pose)
            combined_pose = pose.reshape(-1, 2)
            hull_mask = pose_to_mask(combined_pose)
            if annos[0, i] == 1:
                label = "chase"
            elif annos[1, i] == 1:
                label = "lights"
            else:
                label = "background"
            points = hull_mask.squeeze()
            if points.shape[0] >= min_points:
                shapes = _polygon_shape_from_points(points, label)
                label_list.append(shapes)
                filename = os.path.join(dest_dir, seq + "_" + str(i) + ".png")
                if not os.path.exists(filename):
                    cv2.imwrite(filename, frame)
                img_path = os.path.basename(filename)
                save_labels(
                    filename.replace(".png", ".json"),
                    img_path,
                    label_list,
                    height,
                    width,
                )


def main(user_train_path, video_folder, dest_dir="/data/label_jsons"):
    userTrain_data = load_user_train(user_train_path)
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    seqs = list(userTrain_data["sequences"].keys())
    for seq in seqs:
        video_file = os.path.join(video_folder, seq + ".avi")
    if os.path.exists(video_file):
        points_to_labelme(video_file, seq, userTrain_data, dest_dir)
    else:
        print("not existing", video_file)
