import cv2
import pandas as pd
import numpy as np
import ast
import pycocotools.mask as mask_util


class FreezingAnalyzer():

    def __init__(self, video_file, tracking_results):
        self.video_file = video_file
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

    def run(self):
        video = cv2.VideoCapture(self.video_file)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames_per_second = video.get(cv2.CAP_PROP_FPS)
        num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame1 = video.read()
        frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
        prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        prvs_instances = self.instances(frame_number)
        hsv = np.zeros_like(frame1)
        hsv[..., 1] = 255
        while(1):
            ret, frame2 = video.read()
            frame_number = int(video.get(cv2.CAP_PROP_POS_FRAMES))
            current_instances = self.instances(frame_number)
            df_prvs_cur = pd.merge(
                prvs_instances, current_instances, how='inner', on='instance_name')
            seg_x = df_prvs_cur.segmentation_x.apply(ast.literal_eval)
            seg_y = df_prvs_cur.segmentation_y.apply(ast.literal_eval)
            next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            flow = self.motion(prvs, next)
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            hsv[..., 0] = ang*180/np.pi/2
            hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            cv2.imshow('frame2', bgr)
            k = cv2.waitKey(30) & 0xff
            if k == 27:
                break
            prvs = next
            prvs_instances = current_instances

        video.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    fa = FreezingAnalyzer(
        '/Users/chenyang/Downloads/HighF-07_7000_7030.mp4',
        '/Users/chenyang/Downloads/con_01_pointrend_tracking_results_with_segmenation.csv')
    fa.run()
