import cv2
import numpy as np


class InRange:
    """
    Segment the ROIs based on selected
    values in the given range.
    """

    def __init__(self):
        self.max_value = 255
        self.max_value_h = 239
        self.low_h = 0
        self.low_s = 68
        self.low_v = 140
        self.high_h = self.max_value_h
        self.high_s = self.max_value
        self.high_v = self.max_value
        self.window_name = "Threshold Based Segmentation"
        self.low_h_name = "Low H"
        self.low_s_name = "Low S"
        self.low_v_name = "Low V"

        self.high_h_name = "High H"
        self.high_s_name = "High S"
        self.high_v_name = "High V"

        self.location_history = []
        self.area_history = []

    def trackbar_on_low_h_event(self, value):
        self.low_h = value
        self.low_h = min(self.high_h - 1, self.low_h)
        cv2.setTrackbarPos(self.low_h_name, self.window_name, self.low_h)

    def trackbar_on_high_h_event(self, value):
        self.high_h = value
        self.high_h = max(self.high_h, self.low_h + 1)
        cv2.setTrackbarPos(self.high_h_name, self.window_name, self.high_h)

    def trackbar_on_low_s_event(self, value):
        self.low_s = value
        self.low_s = min(self.high_s - 1, self.low_s)
        cv2.setTrackbarPos(self.low_s_name, self.window_name, self.low_s)

    def trackbar_on_high_s_event(self, value):
        self.high_s = value
        self.high_s = max(self.high_s, self.low_s + 1)
        cv2.setTrackbarPos(self.high_s_name, self.window_name, self.high_s)

    def trackbar_on_low_v_event(self, value):
        self.low_v = value
        self.low_v = min(self.high_v - 1, self.low_v)
        cv2.setTrackbarPos(self.low_v_name, self.window_name, self.low_v)

    def trackbar_on_high_v_event(self, value):
        self.high_v = value
        self.high_v = max(self.high_v, self.low_v + 1)
        cv2.setTrackbarPos(self.high_v_name, self.window_name, self.high_v)

    def detect_contours(self, frame, threshold, min_area=0, max_area=10000, draw=True):
        """
        detect contours and threshold them based on areas.

        Parameters
        ----------
        frame: ndarray, shape(n_rows,n_cols,3)
        threshold: ndarray, shape(n_rows,n_cols,1)
        previous_location: float array, instance's center in the previous frame
        current_location: float array, instance's center in the current frame
        min_area: int, min area threshold for the detected instance
        max_area: int, max area threshold for the detected instance

        Returns
        -------
        out_frame: ndarray, shape(n_rows, n_cols, 3)
                output image with contours on it
        contours: list, detected contours
        previous_location: float array, instance's center in the previous frame
        current_location: float array, instance's center in the current frame
        """
        if int(cv2.__version__[0] == 3):
            img, contours, hierarchy = cv2.findContours(
                threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            contours, hierarchy = cv2.findContours(
                threshold.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )

        out = frame.copy()

        i = 0
        current_location = []
        areas = []

        while i < len(contours):
            area = cv2.contourArea(contours[i])
            if area < min_area or area > max_area:
                del contours[i]
            else:
                M = cv2.moments(contours[i])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = 0, 0
                current_location.append([cx, cy])
                areas.append(area)
                i += 1

        return out, contours, current_location, areas

    def draw_contour_history(
        self,
        current_frame,
    ):
        for location, area in zip(self.location_history, self.area_history):
            cx, cy = location[-1]
            cv2.putText(
                current_frame,
                f"{area[-1]}",
                tuple([cx + 2, cy + 2]),
                cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
        return current_frame

    def run(self, video_file, min_area=70, max_area=150, draw_history=False):
        cap = cv2.VideoCapture(video_file)
        cv2.namedWindow(self.window_name)

        cv2.createTrackbar(
            self.low_h_name,
            self.window_name,
            self.low_h,
            self.max_value_h,
            self.trackbar_on_low_h_event,
        )
        cv2.createTrackbar(
            self.high_h_name,
            self.window_name,
            self.high_h,
            self.max_value_h,
            self.trackbar_on_high_h_event,
        )
        cv2.createTrackbar(
            self.low_s_name,
            self.window_name,
            self.low_s,
            self.max_value,
            self.trackbar_on_low_s_event,
        )
        cv2.createTrackbar(
            self.high_s_name,
            self.window_name,
            self.high_s,
            self.max_value,
            self.trackbar_on_high_s_event,
        )
        cv2.createTrackbar(
            self.low_v_name,
            self.window_name,
            self.low_v,
            self.max_value,
            self.trackbar_on_low_v_event,
        )
        cv2.createTrackbar(
            self.high_v_name,
            self.window_name,
            self.high_v,
            self.max_value,
            self.trackbar_on_high_v_event,
        )

        while True:
            ret, frame = cap.read()
            if frame is None:
                break
            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            gray_img = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
            ret3, th3 = cv2.threshold(
                gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            frame_combined = cv2.bitwise_and(frame, frame, mask=th3)
            frame_HSV = cv2.cvtColor(frame_combined, cv2.COLOR_BGR2HSV)
            frame_threshold = cv2.inRange(
                frame_HSV,
                (self.low_h, self.low_s, self.low_v),
                (self.high_h, self.high_s, self.high_v),
            )
            kernel = np.ones((5, 5), np.uint8)
            frame_threshold = cv2.morphologyEx(frame_threshold, cv2.MORPH_OPEN, kernel)

            frame_threshold, contours, current_location, areas = self.detect_contours(
                frame_threshold, frame_threshold, min_area, max_area, draw=True
            )
            if current_location and areas:
                self.location_history.append(current_location)
                self.area_history.append(areas)
            frame_combined = cv2.bitwise_and(
                frame_combined, frame_combined, mask=frame_threshold
            )

            if draw_history:
                frame_combined = self.draw_contour_history(frame_combined)

            cv2.imshow(self.window_name, frame_combined)

            key = cv2.waitKey(30)
            if key == ord("q") or key == 27:
                break

            if key == ord("r"):
                self.location_history = []
                self.area_history = []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="In range Threshold based segmentation."
    )
    parser.add_argument("--video", help="absolute path of the video file.", type=str)
    args = parser.parse_args()

    ir = InRange()
    ir.run(args.video)
