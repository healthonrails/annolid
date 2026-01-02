import torch
import cv2
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from annolid.features import Embedding
from annolid.utils.runs import shared_runs_root
# Temp fix of the no attribute 'get_filesytem' error
# import tensorflow as tf
# import tensorboard as tb
# tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def tensorboard_writer(logdir=None):

    if logdir is None:
        logdir = shared_runs_root()
    writer = SummaryWriter(log_dir=str(logdir))
    return writer


def frame_embeddings(frame):
    embed_vector = Embedding()(frame)
    return embed_vector


def main(video_url=None):
    cap = cv2.VideoCapture(video_url)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_url}")

    writer = tensorboard_writer()
    frame_number = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        embed_vector = frame_embeddings([frame_rgb])
        writer.add_histogram('Frame Embeddings', embed_vector)

        frame_tensor = torch.from_numpy(frame_rgb)
        writer.add_embedding(embed_vector,
                             metadata=[1],
                             label_img=frame_tensor.permute(
                                 2, 0, 1).unsqueeze(0),
                             global_step=frame_number
                             )
        frame_number += 1

    cap.release()
    writer.close()


if __name__ == "__main__":
    main()
