import os
import cv2
import pandas as pd
import numpy as np
import ast
import pycocotools.mask as mask_util
from annolid.utils import draw


class FreezingAnalyzer():

    def __init__(self,
                 video_file,
                 tracking_results,
                 iou_threshold=0.9,
                 motion_threshold=2000

                 ):
        self.video_file = video_file
        self.iou_threshold = iou_threshold
        self.motion_threshold = motion_threshold
        if tracking_results:
            self.tracking_results = pd.read_csv(tracking_results)
            try:
                self.tracking_results = self.tracking_results.drop(
                    columns=['Unnamed: 0'])
            except KeyError:
                pass

    def instances(self, frame_number):
        df = self.tracking_results
        _df = df[df.frame_number == frame_number]
        return _df

    def motion(self, prev, cur):
        flow = cv2.calcOpticalFlowFarneback(
            prev, cur, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        return flow

    def is_freezing(self, frame_number, instance_name):
        return True

    def _frame_from_video(self, video):
        while video.isOpened():
            success, frame = video.read()
            if success:
                yield frame
            else:
                break

    def get_video_writer(self,
                         video_file,
                         target_fps,
                         width,
                         height):
        video_name = f"{os.path.splitext(video_file)[0]}_freezing.mp4"
        video_writer = cv2.VideoWriter(video_name,
                                       cv2.VideoWriter_fourcc(*"mp4v"),
                                       target_fps,
                                       (width, height))
        return video_writer

    def instance_names(self):
        return (self.tracking_results.instance_name).unique()

    def mask_iou(self, row):
        x = ast.literal_eval(row.segmentation_x)
        y = ast.literal_eval(row.segmentation_y)
        _iou = mask_util.iou([x], [y], [False, False]).flatten()[0]
        return _iou

    def run(self):
        video = cv2.VideoCapture(self.video_file)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame1 = video.read()
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        prvs = cv2.blur(prvs, (5, 5))
        prvs_instances = self.instances(frame_number)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255

        video_writer = self.get_video_writer(self.video_file,
                                             frames_per_second,
                                             width,
                                             height)

        instance_status = dict((i, 0,) for i in (self.instance_names()))

        while video.isOpened():
            ret, frame2 = video.read()
            frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            current_instances = self.instances(frame_number)
            df_prvs_cur = pd.merge(
                prvs_instances, current_instances, how='inner', on='instance_name')

            try:
                df_prvs_cur['mask_iou'] = df_prvs_cur.apply(
                    self.mask_iou, axis=1)
            except ValueError as ve:
                raise ve

            if ret:
                next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                next = cv2.blur(next, (5, 5))
                flow = self.motion(prvs, next)
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                hsv[..., 0] = ang*180/np.pi/2
                hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                for index, _row in df_prvs_cur.iterrows():

                    _mask = ast.literal_eval(_row.segmentation_y)
                    _mask = mask_util.decode(_mask)[:, :]
                    mask_motion = np.sum(_mask * mag)
                    bgr = draw.draw_binary_masks(
                        bgr,
                        [_mask],
                        [_row.instance_name]
                    )

                    if _row.mask_iou >= self.iou_threshold and mask_motion < self.motion_threshold:
                        instance_status[_row.instance_name] += 1

                        if instance_status[_row.instance_name] >= 5:
                            draw.draw_boxes(
                                bgr,
                                [[_row.x1_y, _row.y1_y, _row.x2_y, _row.y2_y]],
                                identities=[f"Motion: {mask_motion:.2f}"],
                                draw_track=False
                            )
                    else:
                        instance_status[_row.instance_name] -= 3

                dst = cv2.addWeighted(frame2, 1, bgr, 1, 0)
                dst = draw.draw_flow(dst, flow)
                cv2.imshow('frame2', dst)
                video_writer.write(dst)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
                prvs = next
                prvs_instances = current_instances

        video.release()
        cv2.destroyAllWindows()
