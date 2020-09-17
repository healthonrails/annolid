import os
import sys
import heapq
import cv2
import argparse
import numpy as np
from collections import deque

sys.path.append("detector/yolov5/")

def extract_frames(video_file='None',
                   num_frames=100,
                   out_dir=None,
                   show_flow=False,
                   algo='flow'
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

    cap = cv2.VideoCapture(video_file)
    fps = cap.get(5)
    n_frames = int(cap.get(7))

    current_frame_number = int(cap.get(1))

    subtractor = cv2.createBackgroundSubtractorMOG2()
    keeped_frames = []

    width = cap.get(3)
    height = cap.get(4)
    ret, old_frame = cap.read()

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
        # always save the first frame and the last frame
        # so to make the number of extract frames
        # as the user provided exactly
        if num_frames % 2 == 0 or n_frames % 2 == 0:
            num_frames -= 2
        else:
            num_frames -= 1

    hsv = np.zeros_like(old_frame)
    hsv[..., 1] = 255

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
        if algo == 'uniform':
            if ret:
                if (frame_number % (n_frames // num_frames) == 0 or
                        frame_number == n_frames - 1):
                    out_frame_file = f"{out_dir}{os.sep}{frame_number:08}.jpg"
                    cv2.imwrite(out_frame_file, frame)
                    print(f'Saved the frame {frame_number}.')
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

                if len(keeped_frames) <= num_frames:
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
        s, f, p = heapq.heappop(keeped_frames)
        cv2.imwrite(f"{out_dir}{os.sep}{f:08}_{s}.jpg", p)
    cap.release()
    cv2.destroyAllWindows
    print(f"Please check the extracted frames in folder: {out_dir}")


def track(video_file=None,
         name="YOLOV5",
         weights=None
         ):
    points = [deque(maxlen=30) for _ in range(1000)]
   
    if name=="YOLOV5":
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
                    detect(opt)
                    strip_optimizer(opt.weights)
            else:
                detect(opt)
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
