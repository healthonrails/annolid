import os
import glob
import sys
import heapq
import cv2
import numpy as np
import random
import subprocess
from typing import Generator, List, Optional, Sequence
from pathlib import Path
from collections import deque
from annolid.segmentation.maskrcnn import inference


def get_video_fps(video_path: str) -> float | None:
    """
    Extracts the frames per second (FPS) of a video file using `ffprobe` if available,
    with a fallback to OpenCV in case of failure.

    Parameters:
        video_path (str): Full path to the video file.

    Returns:
        float | None: The frame rate (FPS) as a float (rounded to 2 decimals), or None if extraction fails.

    Notes:
        - Requires `ffprobe` to be installed and accessible in the system PATH.
        - Fallback via OpenCV may return an approximate value or 0 if not available.
    """
    try:
        # Use ffprobe to get the rational frame rate (e.g., "30000/1001")
        cmd = [
            "ffprobe",
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            rate_str = result.stdout.strip()
            if "/" in rate_str:
                num, denom = map(int, rate_str.split("/"))
                if denom != 0:
                    return round(num / denom, 2)
            elif rate_str.replace('.', '', 1).isdigit():
                return round(float(rate_str), 2)
    except Exception as e:
        print(f"[get_video_fps] ffprobe failed: {e}")

    # Fallback to OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                return round(fps, 2)
    except Exception as e:
        print(f"[get_video_fps] OpenCV fallback failed: {e}")

    return None


def get_keyframe_timestamps(video_path: str) -> List[float]:
    """
    Retrieves presentation timestamps (in seconds) for keyframes using ffprobe.

    Returns an empty list if ffprobe is unavailable or keyframe data cannot be
    extracted.
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-skip_frame", "nokey",
        "-show_frames",
        "-show_entries", "frame=pkt_pts_time",
        "-of", "csv=p=0",
        video_path,
    ]

    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
    except FileNotFoundError:
        return []

    if result.returncode != 0:
        return []

    timestamps: List[float] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        if "," in value:
            value = value.split(",")[-1]
        try:
            ts = float(value)
        except ValueError:
            continue
        if ts >= 0:
            timestamps.append(ts)

    return timestamps


def get_video_files(video_folder):
    """
    Retrieves a list of video files from the specified folder.

    Parameters:
    video_folder (str or Path): Path to the folder containing video files.

    Returns:
    list: A list of paths to video files found in the specified folder.

    Supported video formats: .mp4, .avi, .mov, .mkv, .flv, .wmv, .mpg, .mpeg
    """
    video_extensions = ['*.mp4', '*.avi', '*.mov',
                        '*.mkv', '*.flv', '*.wmv', '*.mpg', '*.mpeg']
    video_files = []
    for ext in video_extensions:
        video_files.extend(glob.glob(os.path.join(video_folder, ext)))
    return video_files


def frame_from_video(video, num_frames):
    attempt = 0
    for i in range(num_frames):
        success, frame = video.read()
        if success:
            yield frame
        else:
            attempt += 1
            if attempt >= 2000:
                break
            else:
                video.set(cv2.CAP_PROP_POS_FRAMES, i+1)
                print('Cannot read this frame:', i)
                continue


def test_cmd(cmd):
    """Test the cmd.
    Modified from moviepy 
    """
    try:
        popen_params = {
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "stdin": subprocess.DEVNULL
        }
        # to remove unwanted window on windows
        # when created the child process
        if os.name == "nt":
            popen_params["creationflags"] = 0x08000000

        proc = subprocess.Popen(cmd, **popen_params)
        proc.communicate()
    except Exception as err:
        return False, err
    else:
        return True, None


def get_ffmpeg_path():
    """Test and find the correct ffmpeg binary file.
    Modified from moviepy.
    """
    ffmpeg_binary = os.getenv('FFMPEG_BINARY', 'ffmpeg-imageio')
    if ffmpeg_binary == 'ffmpeg-imageio':
        from imageio.plugins.ffmpeg import get_exe
        ffmpeg_binary = get_exe()
    elif ffmpeg_binary == 'auto-detect':
        if test_cmd(['ffmpeg'])[0]:
            ffmpeg_binary = 'ffmpeg'
        elif test_cmd(['ffmpeg.exe'])[0]:
            ffmpeg_binary = 'ffmpeg.exe'
        else:
            ffmpeg_binary = 'unset'
    else:
        success, err = test_cmd([ffmpeg_binary])
        if not success:
            raise IOError(
                str(err) +
                " - The path specified for the ffmpeg binary might be wrong")
    return ffmpeg_binary


def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ 
     Makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``.
    Modified from moviepy ffmpeg_tools
    """
    name, ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = "%sSUB%d_%d.%s" % (name, T1, T2, ext)

    ffmpeg_binary = get_ffmpeg_path()
    cmd = [ffmpeg_binary, "-y",
           "-ss", "%0.2f" % t1,
           "-i", filename,
           "-t", "%0.2f" % (t2-t1),
           "-vcodec", "copy",
           "-acodec", "copy", "-strict", "-2", targetname]

    popen_params = {"stdout": subprocess.DEVNULL,
                    "stderr": subprocess.PIPE,
                    "stdin": subprocess.DEVNULL}

    if os.name == "nt":
        popen_params["creationflags"] = 0x08000000

    proc = subprocess.Popen(cmd, **popen_params)

    out, err = proc.communicate()
    proc.stderr.close()

    if proc.returncode:
        raise IOError(err.decode('utf8'))
    del proc


def extract_subclip(video_file,
                    start_time,
                    end_time):
    video_dir = Path(video_file).parent
    out_video_name = Path(video_file).stem + \
        f"_{str(start_time)}_{str(end_time)}.mp4"
    out_video_path = video_dir / out_video_name

    ffmpeg_extract_subclip(
        video_file,
        start_time,
        end_time,
        targetname=str(out_video_path))

    return str(out_video_path)


def key_frames(video_file=None,
               out_dir=None,
               ctx=None,
               sub_clip=False,
               start_seconds=None,
               end_seconds=None,
               ):

    if sub_clip and start_seconds is not None and end_seconds is not None:
        video_file = extract_subclip(video_file,
                                     start_seconds,
                                     end_seconds)

    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {video_file}")

    keyframe_times = get_keyframe_timestamps(video_file)

    extraction_points: List[tuple[str, float]] = []
    if keyframe_times:
        # Use timestamps directly to minimize rounding errors.
        seen_times = set()
        for ts in keyframe_times:
            rounded = round(ts, 6)
            if rounded in seen_times:
                continue
            seen_times.add(rounded)
            extraction_points.append(("time", ts))
    else:
        # Fallback: sample frames roughly every second based on FPS.
        fps = cap.get(cv2.CAP_PROP_FPS) or 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total_frames <= 0:
            extraction_points = [("index", 0)]
        else:
            step = max(int(round(fps)) or 1, 1)
            indices = list(range(0, total_frames, step))
            if indices[-1] != total_frames - 1:
                indices.append(total_frames - 1)
            extraction_points = [("index", idx) for idx in indices]

    video_name = Path(video_file).stem
    video_name = video_name.replace(' ', '_')
    if out_dir is None:
        out_dir = Path(video_file).parent / video_name
        out_dir = out_dir.with_suffix('')
    out_dir = Path(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True)

    saved_indices = set()

    for mode, value in extraction_points:
        if mode == "time":
            cap.set(cv2.CAP_PROP_POS_MSEC, value * 1000.0)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(value))

        ret, frame = cap.read()
        if not ret or frame is None:
            continue

        frame_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        if frame_idx < 0:
            frame_idx = int(value if mode == "index" else round(
                value * (cap.get(cv2.CAP_PROP_FPS) or 0)))

        if frame_idx in saved_indices:
            continue
        saved_indices.add(frame_idx)

        out_frame_file = out_dir / \
            f"{(video_name).replace(' ','_')}_{frame_idx:08}.png"
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(str(out_frame_file), frame_rgb)
        print(f"Saved {str(out_frame_file)}")

    cap.release()

    print(f"Please check your frames located at {out_dir}")
    return out_dir


def download_youtube_video(
    url: str,
    output_dir: str | Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """
    Download a YouTube video and return the local file path.

    The download relies on `yt-dlp` and saves the media as an MP4 file.

    Args:
        url: Public YouTube URL.
        output_dir: Optional directory to store the downloaded file. Defaults to
            ``~/annolid_youtube``.
        overwrite: When False, a previously downloaded copy is reused if present.

    Returns:
        Path: The path to the downloaded (or cached) video file.

    Raises:
        ValueError: If the URL is empty.
        RuntimeError: If `yt-dlp` is missing or the download fails.
    """
    if not url or not url.strip():
        raise ValueError("YouTube URL must not be empty.")

    try:
        from yt_dlp import YoutubeDL
    except ImportError as exc:  # pragma: no cover - defensive path
        raise RuntimeError(
            "The 'yt-dlp' package is required to download YouTube videos."
        ) from exc

    destination_root = Path(output_dir) if output_dir else Path.home() / "annolid_youtube"
    destination_root.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "merge_output_format": "mp4",
        "outtmpl": str(destination_root / "%(title)s-%(id)s.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "overwrites": overwrite,
    }

    with YoutubeDL(ydl_opts) as ydl:
        # Extract metadata first to allow cache reuse without re-downloading.
        info = ydl.extract_info(url, download=False)
        expected_output = Path(ydl.prepare_filename(info)).with_suffix(".mp4")
        if expected_output.exists() and not overwrite:
            return expected_output

        download_info = ydl.extract_info(url, download=True)

    candidate_paths = []

    requested_downloads = download_info.get("requested_downloads")
    if requested_downloads:
        candidate_paths.extend(
            Path(item["filepath"])
            for item in requested_downloads
            if isinstance(item, dict) and item.get("filepath")
        )

    primary_filepath = download_info.get("filepath")
    if primary_filepath:
        candidate_paths.append(Path(primary_filepath))

    legacy_filename = download_info.get("_filename")
    if legacy_filename:
        candidate_paths.append(Path(legacy_filename))

    candidate_paths.append(expected_output)

    valid_suffixes = {".mp4", ".mov", ".mkv", ".avi"}
    for candidate in candidate_paths:
        if not candidate:
            continue
        if candidate.exists():
            if candidate.suffix.lower() in valid_suffixes:
                return candidate
            mp4_candidate = candidate.with_suffix(".mp4")
            if mp4_candidate.exists():
                return mp4_candidate

    raise RuntimeError("Failed to download YouTube video.")


def video_loader(video_file=None):
    """
    Backward-compatible helper that returns a CV2Video instance.
    """
    if video_file is None:
        return None
    if not Path(video_file).exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")
    return CV2Video(video_file)


class CV2Video:
    def __init__(self, video_file, use_decord=False):
        self.video_file = Path(video_file).resolve()
        if not self.video_file.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_file}")
        # `use_decord` is kept for backward compatibility with previous signature.
        # The flag is ignored because OpenCV is now the only backend.
        self.cap = cv2.VideoCapture(str(self.video_file))
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open video: {self.video_file}")
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        self.current_frame_timestamp = None
        self.width = None
        self.height = None
        self.first_frame = None
        self.fps = None
        self._last_frame_index: Optional[int] = None

    def get_first_frame(self):
        if self.first_frame is None:
            self.first_frame = self.load_frame(0)
        return self.first_frame

    def get_width(self):
        if self.width is None:
            self.width = self.first_frame.shape[1]
        return self.width

    def get_height(self):
        if self.height is None:
            self.height = self.first_frame.shape[0]
        return self.height

    def get_fps(self):
        if self.fps is None:
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        return self.fps

    def load_frame(self, frame_number):
        if frame_number < 0 or frame_number >= self.total_frames():
            raise KeyError(f"Frame index out of bounds: {frame_number}")

        expected_next = (
            self._last_frame_index + 1 if self._last_frame_index is not None else None
        )
        if expected_next is None or frame_number != expected_next:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise KeyError(f"Cannot load frame number: {frame_number}")

        self._last_frame_index = frame_number
        self.current_frame_timestamp = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return frame

    def get_time_stamp(self):
        return self.current_frame_timestamp

    def total_frames(self):
        return self.frame_count

    def get_frames_in_batches(self, start_frame: int,
                              end_frame: int,
                              batch_size: int) -> Generator[np.ndarray, None, None]:
        """
        Retrieves frames in batches between start and end frames.

        Args:
            start_frame (int): The starting frame number.
            end_frame (int): The ending frame number.
            batch_size (int): The size of each batch.

        Yields:
            np.ndarray: A batch of video frames.
        """
        if start_frame >= end_frame:
            raise ValueError("Start frame must be less than end frame.")

        total_frames = self.total_frames()
        end_frame = min(end_frame, total_frames)

        if end_frame - start_frame < batch_size:
            batch_size = end_frame - start_frame

        for batch_start in range(start_frame, end_frame, batch_size):
            batch_end = min(batch_start + batch_size, end_frame)
            for frame_number in range(batch_start, batch_end):
                yield self.load_frame(frame_number)

    def get_frames_between(self, start_frame: int,
                           end_frame: int) -> List[np.ndarray]:
        """
        Retrieves all frames between start and end frames.

        Args:
            start_frame (int): The starting frame number.
            end_frame (int): The ending frame number.

        Returns:
            List[np.ndarray]: A list of video frames.
        """
        if start_frame >= end_frame:
            raise ValueError("Start frame must be less than end frame.")

        total_frames = self.total_frames()
        end_frame = min(end_frame, total_frames)

        frames = []
        for frame_number in range(start_frame, end_frame):
            frames.append(self.load_frame(frame_number))

        return np.stack(frames)

    def release(self):
        if getattr(self, "cap", None) is not None and self.cap.isOpened():
            self.cap.release()

    def __del__(self):
        self.release()


def extract_frames(video_file='None',
                   num_frames=100,
                   out_dir=None,
                   show_flow=False,
                   algo='flow',
                   keep_first_frame=False,
                   sub_clip=False,
                   start_seconds=None,
                   end_seconds=None,
                   prediction=True
                   ):
    """
    Extract frames from the given video file. 
    This function saves the wanted number of frames based on
    optical flow by default.
    Or you can save all the frames by providing `num_frames` = -1. 

    """

    if sub_clip and start_seconds is not None and end_seconds is not None:
        video_file = extract_subclip(video_file,
                                     start_seconds,
                                     end_seconds)

    video_name = os.path.splitext(os.path.basename(video_file))[0]
    video_name = video_name.replace(' ', '_')
    if out_dir is None:
        out_dir = os.path.splitext(video_file)[0]
    else:
        out_dir = os.path.join(out_dir, video_name)

    # add extracting methods to folder name
    out_dir = f"{out_dir}_{algo}"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    if algo == 'keyframes':
        out_dir = key_frames(video_file, out_dir,
                             sub_clip=sub_clip,
                             start_seconds=start_seconds,
                             end_seconds=end_seconds)
        yield 100, f"Done. Frames located at {out_dir}."

    cap = cv2.VideoCapture(video_file)
    n_frames = int(cap.get(7))

    current_frame_number = int(cap.get(1))

    subtractor = cv2.createBackgroundSubtractorMOG2()
    keeped_frames = []

    ret, old_frame = cap.read()

    if keep_first_frame:
        # save the first frame
        out_frame_file = f"{out_dir}{os.sep}{video_name}_{current_frame_number:08}.png"
        cv2.imwrite(out_frame_file, old_frame)

    if num_frames < -1 or num_frames > n_frames:
        print(f'The video has {n_frames} number frames in total.')
        print('Please input a valid number of frames!')
        return
    elif num_frames == 1:
        print(f'Please check your first frame here {out_dir}')
        return
    elif num_frames > 2:
        # if save the first frame and the last frame
        # so to make the number of extract frames
        # as the user provided exactly
        if keep_first_frame:
            num_frames -= 1

    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

    # keeped frame index
    ki = 0

    while ret:

        frame_number = int(cap.get(1))
        ret, frame = cap.read()

        progress_percent = ((frame_number + 1) / n_frames) * 100

        if num_frames == -1:
            out_frame_file = f"{out_dir}{os.sep}{video_name}_{frame_number:08}.png"
            if ret:
                cv2.imwrite(
                    out_frame_file, frame)
                print(f'Saved the frame {frame_number}.')
            continue
        if algo == 'random' and num_frames != -1:
            if ret:
                if len(keeped_frames) < num_frames:
                    keeped_frames.append((ki,
                                          frame_number,
                                          frame))
                else:
                    j = random.randrange(ki + 1)
                    if j < num_frames:
                        keeped_frames[j] = ((j,
                                             frame_number,
                                             frame))
                ki += 1
                yield progress_percent, f'Processed {ki} frames.'
            continue
        if algo == 'flow' and num_frames != -1:
            mask = subtractor.apply(frame)
            old_mask = subtractor.apply(old_frame)

            out_frame = cv2.bitwise_and(frame, frame, mask=mask)
            old_out_frame = cv2.bitwise_and(
                old_frame, old_frame, mask=old_mask)
            try:
                out_frame = cv2.cvtColor(out_frame, cv2.COLOR_BGR2GRAY)
                old_out_frame = cv2.cvtColor(old_out_frame, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(
                    old_out_frame, out_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                if show_flow:
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv[..., 0] = ang*180/np.pi/2
                    hsv[..., 2] = cv2.normalize(
                        mag, None, 0, 255, cv2.NORM_MINMAX)
                    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                q_score = int(np.abs(np.sum(flow.reshape(-1))))
                progress_msg = f"precessing frame {frame_number},the diff to prev frame is {q_score}."

                if len(keeped_frames) < num_frames:
                    heapq.heappush(
                        keeped_frames, ((q_score, frame_number, frame)))
                else:
                    heapq.heappushpop(
                        keeped_frames, ((q_score, frame_number, frame)))

                if show_flow:
                    cv2.imshow("Frame", rgb)
                yield progress_percent, progress_msg
            except:
                print('skipping the current frame.')

            old_frame = frame
        key = cv2.waitKey(1)
        if key == 27:
            break

    for kf in keeped_frames:
        s, f, p = kf
        out_img = f"{out_dir}{os.sep}{video_name}_{f:08}_{s}.png"
        cv2.imwrite(out_img, p)
        # default mask rcnn prediction if select less than 5 frames
        if prediction and num_frames <= 5:
            try:
                inference.predict_mask_to_labelme(out_img, 0.7)
            except:
                print("Please install pytorch and torchvision as follows.")
                print("pip install torch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0")
                pass

    cap.release()
    cv2.destroyAllWindows()
    yield 100, f"Done. Frames located at {out_dir}"


def track(video_file=None,
          name="YOLOV5",
          weights=None
          ):

    points = [deque(maxlen=30) for _ in range(1000)]

    if name == "YOLOV5":
        # avoid installing pytorch
        # if the user only wants to use it for
        # extract frames
        # maybe there is a better way to do this
        sys.path.append("detector/yolov5/")
        import torch
        from annolid.detector.yolov5.detect import detect
        from annolid.utils.config import get_config
        cfg = get_config("./configs/yolov5s.yaml")
        from annolid.detector.yolov5.utils.general import strip_optimizer

        opt = cfg
        if weights is not None:
            opt.weights = weights
        opt.source = video_file

        with torch.no_grad():
            if opt.update:  # update all models (to fix SourceChangeWarning)
                for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
                    detect(opt, points=points)
                    strip_optimizer(opt.weights)
            else:
                detect(opt, points=points)
                strip_optimizer(opt.weights)
    else:
        from annolid.tracker.build_deepsort import build_tracker
        from annolid.detector import build_detector
        from annolid.utils.draw import draw_boxes
        if not (os.path.isfile(video_file)):
            print("Please provide a valid video file")
        detector = build_detector()
        class_names = detector.class_names

        cap = cv2.VideoCapture(video_file)

        ret, prev_frame = cap.read()
        deep_sort = build_tracker()

        while ret:
            ret, frame = cap.read()
            if not ret:
                print("Finished tracking.")
                break
            im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            bbox_xywh, cls_conf, cls_ids = detector(im)
            bbox_xywh[:, 3:] *= 1.2
            mask = cls_ids == 0
            cls_conf = cls_conf[mask]

            outputs = deep_sort.update(bbox_xywh, cls_conf, im)

            if len(outputs) > 0:
                bbox_xyxy = outputs[:, :4]
                identities = outputs[:, -1]
                frame = draw_boxes(frame,
                                   bbox_xyxy,
                                   identities,
                                   draw_track=True,
                                   points=points
                                   )

            cv2.imshow("Frame", frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

            prev_frame = frame

        cv2.destroyAllWindows()
        cap.release()


def extract_video_metadata(cap):
    """Extract video metadata

    Args:
        cap (VideoCapture): cv2 VideoCapture object

    Returns:
        dict : dict of video metadata
    """
    meta_data = {
        'frame_width': cap.get(cv2.CAP_PROP_FRAME_WIDTH),
        'frame_height': cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'fps': cap.get(cv2.CAP_PROP_FPS),
        'format': cap.get(cv2.CAP_PROP_FORMAT),
        'frame_count': cap.get(cv2.CAP_PROP_FRAME_COUNT),
        'fourcc': cap.get(cv2.CAP_PROP_FOURCC),
        'mode': cap.get(cv2.CAP_PROP_MODE)
    }
    return meta_data


def extract_frames_from_video(
    video_path: str,
    output_folder: Optional[str] = None,
    *,
    frame_indices: Optional[Sequence[int]] = None,
    num_frames: int = 5,
    image_format: str = "png",
    name_pattern: Optional[str] = None,
) -> List[str]:
    """Extract specific frames (or evenly sampled ones) from a single video.

    Args:
        video_path: Path to the video file.
        output_folder: Directory to store the extracted frames. If omitted, a
            folder named after the video will be created beside the source file.
        frame_indices: Explicit frame indices to extract. If provided, they take
            precedence over the ``num_frames`` sampling logic.
        num_frames: Number of evenly sampled frames when ``frame_indices`` is
            not supplied. Defaults to 5.
        image_format: Image format/extension (e.g., ``"png"``).

    Returns:
        List[str]: Paths to the extracted frame images (may be empty).
    """

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    if output_folder is None:
        output_folder = video_path.with_suffix("" ).as_posix()
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap or not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if frame_count <= 0:
        cap.release()
        return []

    if frame_indices:
        indices = sorted({idx for idx in frame_indices if 0 <= idx < frame_count})
    else:
        if num_frames <= 1:
            indices = [frame_count // 2]
        else:
            num_frames = min(num_frames, frame_count)
            indices = [
                int(round(i * (frame_count - 1) / (num_frames - 1)))
                for i in range(num_frames)
            ] if frame_count > 1 else [0]

    saved_paths: List[str] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        if name_pattern:
            try:
                filename = name_pattern.format(video_stem=video_path.stem, frame=idx)
            except Exception:
                filename = f"{video_path.stem}_{idx:09d}.{image_format}"
        else:
            filename = f"{video_path.stem}_frame{idx:09d}.{image_format}"
        out_path = Path(output_folder) / filename
        if cv2.imwrite(str(out_path), frame):
            saved_paths.append(str(out_path))

    cap.release()
    return saved_paths


def extract_frames_from_videos(input_folder, output_folder=None, num_frames=5):
    """
    Extract specified number of frames from each video in the folder and save as PNG files.

    Args:
        input_folder (str): Path to the folder containing video files.
        output_folder (str): Path to the folder to save extracted frames. Defaults to './extracted_frames'.
        num_frames (int): Number of frames to extract. Defaults to 5.
    """
    # Set default output folder if not provided
    if output_folder is None:
        output_folder = os.path.join(os.getcwd(), "extracted_frames")

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Loop through all video files in the input folder
    for video_file in os.listdir(input_folder):
        video_path = os.path.join(input_folder, video_file)
        if not video_file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.mpg')):
            continue  # Skip non-video files

        try:
            extract_frames_from_video(
                video_path,
                output_folder,
                num_frames=num_frames,
            )
        except Exception as exc:
            print(f"Skipping {video_file}: {exc}")
