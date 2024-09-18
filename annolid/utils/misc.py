import os
import cv2
import torch
from tqdm import tqdm
from collections import deque


def _load_frame_as_tensor(frame, image_size):
    """
    Resize the frame and convert it to a normalized tensor.
    """
    img_np = cv2.resize(frame, (image_size, image_size))
    img_np = img_np / 255.0
    img = torch.from_numpy(img_np).permute(2, 0, 1)
    video_height, video_width = frame.shape[:2]  # the original video size
    return img, video_height, video_width


class AsyncVideoFrameLoaderFromVideo:
    """
    A class to load frames asynchronously from a video file without saving them to disk.
    It uses a sliding window to manage memory efficiently,
      loading a limited number of frames at a time.
    """

    def __init__(self, video_path, image_size,
                 offload_video_to_cpu,
                 img_mean, img_std,
                 compute_device,
                 cache_size=10):
        self.video_path = video_path
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.compute_device = compute_device
        self.cache_size = cache_size  # Number of frames to keep in memory
        self.cache = {}  # Frame cache
        # To manage frame eviction order
        self.frame_queue = deque(maxlen=cache_size)
        self.exception = None

        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video file {video_path}")

        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_height, self.video_width = None, None

        # Load the first frame synchronously to get dimensions
        self.__getitem__(0)

    def _load_frame(self, index):
        """
        Load a frame from the video and preprocess it.
        """
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame at index {index}")

        img, video_height, video_width = _load_frame_as_tensor(
            frame, self.image_size)
        self.video_height = video_height
        self.video_width = video_width

        # Normalize by mean and std
        img = (img - self.img_mean) / self.img_std
        if not self.offload_video_to_cpu:

            if img.dtype == torch.float64:
                img = img.float()  # Convert to float32 if necessary

            img = img.to(self.compute_device, non_blocking=True)

        return img

    def __getitem__(self, index):
        """
        Retrieve a frame. If not in the cache, load it and store it.
        """
        if self.exception is not None:
            raise RuntimeError(
                "Failure in frame loading thread") from self.exception

        # If frame is cached, return it
        if index in self.cache:
            return self.cache[index]

        # Load the frame if not already cached
        frame = self._load_frame(index)
        if len(self.frame_queue) >= self.cache_size:
            # Remove the oldest frame from memory
            old_frame_idx = self.frame_queue.popleft()
            del self.cache[old_frame_idx]

        # Cache the new frame
        self.cache[index] = frame
        self.frame_queue.append(index)

        return frame

    def __len__(self):
        return self.num_frames

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()


def load_video_frames_from_video(
    video_path,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    async_loading_frames=False,
    compute_device=torch.device("cuda"),
    cache_size=10  # New parameter for cache size
):
    """
    Load frames directly from a video file asynchronously or synchronously with memory management.
    The frames are resized to `image_size` x `image_size` and are loaded to GPU if
    `offload_video_to_cpu` is `False` or to CPU if `offload_video_to_cpu` is `True`.

    If `async_loading_frames` is set to True, the frames are loaded asynchronously.
    """
    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

    if async_loading_frames:
        lazy_images = AsyncVideoFrameLoaderFromVideo(
            video_path,
            image_size,
            offload_video_to_cpu,
            img_mean,
            img_std,
            compute_device,
            cache_size=cache_size  # Apply cache size for async loading
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    # Synchronous loading
    cap = cv2.VideoCapture(video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    images = torch.zeros(num_frames, 3, image_size,
                         image_size, dtype=torch.float32)

    for n in tqdm(range(num_frames), desc="frame loading (video)"):
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Failed to read frame {n}")
        images[n], video_height, video_width = _load_frame_as_tensor(
            frame, image_size)

    if not offload_video_to_cpu:
        images = images.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    # Normalize by mean and std
    images -= img_mean
    images /= img_std

    return images, video_height, video_width


def test_video_loader(video_path):
    image_size = 224  # Example size (change as needed)
    offload_video_to_cpu = False  # Load frames to GPU if available
    img_mean = (0.485, 0.456, 0.406)
    img_std = (0.229, 0.224, 0.225)
    async_loading = True  # Set to False for synchronous testing
    compute_device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    cache_size = 10  # Keep 10 frames in memory at a time

    # Load video frames asynchronously
    video_frames, video_height, video_width = load_video_frames_from_video(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean,
        img_std,
        async_loading,
        compute_device,
        cache_size=cache_size  # Pass cache size for memory control
    )

    # Check the dimensions of the first frame
    first_frame = video_frames[0]
    print(f"First frame shape: {first_frame.shape}")
    print(f"Video dimensions: {video_height}x{video_width}")

    # Optionally, wait for all frames to load
    if async_loading:
        print("Waiting for all frames to load...")
        video_frames.thread.join()

    # Iterate over frames (print a few details)
    for i, frame in enumerate(video_frames):
        print(f"Frame {i}: {frame.shape}")
        if i == 10:  # Stop after 10 frames
            break


if __name__ == "__main__":
    video_path = os.path.expanduser("~/Downloads/mouse.mp4")
    test_video_loader(video_path)
