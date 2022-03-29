import argparse
import cv2
import numpy as np
import torch
from raft import RAFT
from annolid.utils import flow_viz
from annolid.motion.utils import InputPadder
from torch.utils.data import DataLoader, IterableDataset


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

    def __iter__(self):

        ret, old_frame = self.cap.read()
        num_frames = (int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        old_frame = cv2.cvtColor(old_frame, cv2.COLOR_BGR2RGB)
        for num in range(num_frames - 1):
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)
    cv2.imshow("Flow", img_flo)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video_path', default=None, help="video path"
    )
    parser.add_argument(
        '--model', default="weights/raft-things.pth", help="restore checkpoint")
    parser.add_argument('--path', default="/content/video",
                        help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', default=False,
                        help='use efficent correlation implementation')
    args = parser.parse_args()

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(
        args.model, map_location=torch.device('cpu')))

    model = model.module
    model.to(DEVICE)
    model.eval()
    video_dataset = VideoFrameDataset(args.video_path)

    loader = DataLoader(video_dataset, batch_size=1)
    counter = 0
    with torch.no_grad():
        for image1, image2 in loader:
            image1 = image1.permute(0, 3, 1, 2).float()
            image2 = image2.permute(0, 3, 1, 2).float()

            image1 = image1.to(DEVICE)
            image2 = image2.to(DEVICE)
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up)
            if counter % 10 == 0:
                print(flow_up)
            counter += 1


if __name__ == '__main__':
    main()
