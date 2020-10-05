import os
import sys
import heapq
import cv2
import argparse
import numpy as np
import random
from pathlib import Path
import decord as de
from collections import deque

sys.path.append("detector/yolov5/")


def key_frames(video_file=None,
               out_dir=None,
               ctx=de.cpu(0)
               ):
    vr = de.VideoReader(video_file)
    key_idxs = vr.get_key_indices()

    if out_dir is None:
        video_name = Path(video_file).name
        out_dir = Path(video_file).parent / video_name
        out_dir = out_dir.with_suffix('')

    out_dir = Path(out_dir)
    out_dir.mkdir(
        parents=True,
        exist_ok=True)
    for ki in key_idxs:
        frame = vr[ki].asnumpy()
        out_frame_file = out_dir / f"{ki:08}.jpg"
        cv2.imwrite(str(out_frame_file), frame)
    print(f"Please check your frames located at {out_dir}")


def extract_frames(video_file='None',
                   num_frames=100,
                   out_dir=None,
                   show_flow=False,
                   algo='flow',
                   keep_first_frame=False
                   ):
    """
    Extract frames from the given video file. 
    This function saves the wanted number of frames based on
    optical flow by default.
    Or you can save all the frames by providing `num_frames` = -1. 

    """

    if out_dir is None:
        out_dir = os.path.splitext(video_file)[0]
    else:
        video_name = os.path.splitext(os.path.basename(video_file))[0]
        out_dir = os.path.join(out_dir, video_name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    if algo == 'keyframes':
        key_frames(video_file, out_dir)
        return

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(5)
    n_frames = int(cap.get(7))

    current_frame_number = int(cap.get(1))

    subtractor = cv2.createBackgroundSubtractorMOG2()
    keeped_frames = []

    width = cap.get(3)
    height = cap.get(4)
    ret, old_frame = cap.read()

    if keep_first_frame:
        # save the first frame
        out_frame_file = f"{out_dir}{os.sep}{current_frame_number:08}.jpg"
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

        if num_frames == -1:
            out_frame_file = f"{out_dir}{os.sep}{frame_number:08}.jpg"
            if ret:
                cv2.imwrite(
                    out_frame_file, frame)
                print(f'Saved the frame {frame_number}.')
            continue
        if algo == 'random':
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
                print(
                    f"precessing frame {frame_number}, the difference between previous frame is {q_score}.")

                if len(keeped_frames) < num_frames:
                    heapq.heappush(
                        keeped_frames, ((q_score, frame_number, frame)))
                else:
                    heapq.heappushpop(
                        keeped_frames, ((q_score, frame_number, frame)))

                if show_flow:
                    cv2.imshow("Frame", rgb)
            except:
                print('skipping the current frame.')

            old_frame = frame

        key = cv2.waitKey(1)
        if key == 27:
            break

    for kf in keeped_frames:
        s, f, p = kf
        cv2.imwrite(f"{out_dir}{os.sep}{f:08}_{s}.jpg", p)

    cap.release()
    cv2.destroyAllWindows()
    print(f"Please check the extracted frames in folder: {out_dir}")


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
        from annolid.tracker import build_tracker
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
