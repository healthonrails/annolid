import cv2
from pathlib import Path
from segment_anything import SegmentAnythingModel
from annolid.data.videos import CV2Video
from annolid.utils.files import find_most_recent_file
import json
from annolid.gui.shape import Shape
from annolid.annotation.keypoints import save_labels
import numpy as np
from collections import deque


class MaxSizeQueue(deque):
    def __init__(self, max_size):
        super().__init__(maxlen=max_size)

    def enqueue(self, item):
        self.append(item)

    def to_numpy(self):
        return np.array(list(self))


def calculate_polygon_center(polygon_vertices):
    x_coords, y_coords = zip(*polygon_vertices)
    center_x = np.mean(x_coords)
    center_y = np.mean(y_coords)
    return np.array([(center_x, center_y)])


class VideoProcessor:
    """
    A class for processing video frames using the Segment-Anything model.
    """

    def __init__(self,
                 video_path,
                 num_center_points=3
                 ):
        """
        Initialize the VideoProcessor.

        Parameters:
        - video_path (str): Path to the video file.
        - num_center_points (int): number of center points for prompt.
        """
        self.video_path = video_path
        self.video_folder = Path(video_path).with_suffix("")
        self.video_loader = CV2Video(video_path)
        self.edge_sam = self.get_model()
        self.num_frames = self.video_loader.total_frames()
        self.center_points = MaxSizeQueue(max_size=num_center_points)

    def get_model(self,
                  encoder_path="edge_sam_3x_encoder.onnx",
                  decoder_path="edge_sam_3x_decoder.onnx",
                  name="Segment-Anything (Edge)"
                  ):
        """
        Load the Segment-Anything model.

        Parameters:
        - encoder_path (str): Path to the encoder model file.
        - decoder_path (str): Path to the decoder model file.
        - name (str): name of the SAM model

        Returns:
        - SegmentAnythingModel: The loaded model.
        """
        model = SegmentAnythingModel(name, encoder_path, decoder_path)
        return model

    def load_json_file(self, json_file_path):
        """
        Load JSON file containing shapes and labels.

        Parameters:
        - json_file_path (str): Path to the JSON file.

        Returns:
        - tuple: A tuple containing two dictionaries (points_dict, point_labels_dict).
        """
        with open(json_file_path, 'r') as json_file:
            data = json.load(json_file)

        points_dict = {}
        point_labels_dict = {}

        for shape in data.get('shapes', []):
            label = shape.get('label')
            points = shape.get('points', [])

            if label and points:
                points_dict[label] = points
                # You can customize this if needed
                point_labels_dict[label] = 1

        return points_dict, point_labels_dict

    def process_frame(self, frame_number):
        """
        Process a single frame of the video.

        Parameters:
        - frame_number (int): Frame number to process.
        """
        cur_frame = self.video_loader.load_frame(frame_number)

        height, width, _ = cur_frame.shape

        points_dict, _ = self.load_json_file(self.get_most_recent_file())
        label_list = []

        # Example usage of predict_polygon_from_points
        for label, points in points_dict.items():
            self.edge_sam.set_image(cur_frame)
            points = calculate_polygon_center(points)
            self.center_points.enqueue(points[0])
            points = self.center_points.to_numpy()
            point_labels = [1] * len(points)
            polygon = self.edge_sam.predict_polygon_from_points(
                points, point_labels)

            # Save the LabelMe JSON to a file
            p_shape = Shape(label, shape_type='polygon', flags={})
            for x, y in polygon:
                # do not add 0,0 to the list
                if x >= 1 and y >= 1:
                    p_shape.addPoint((x, y))
            label_list.append(p_shape)

        filename = self.video_folder / \
            (self.video_folder.name + f"_{frame_number:0>{9}}.json")
        img_filename = str(filename.with_suffix('.png'))
        cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(img_filename, cur_frame)
        save_labels(filename=filename, imagePath=img_filename, label_list=label_list,
                    height=height, width=width)

    def process_video_frames(self, start_frame=0, end_frame=None, step=10):
        """
        Process multiple frames of the video.

        Parameters:
        - start_frame (int): Starting frame number.
        - end_frame (int): Ending frame number.
        - step (int): Step between frames.
        """
        if end_frame is None:
            end_frame = self.num_frames
        for i in range(start_frame, end_frame + 1, step):
            self.process_frame(i)

    def get_most_recent_file(self):
        """
        Find the most recent file in the video folder.

        Returns:
        - str: Path to the most recent file.
        """
        return find_most_recent_file(self.video_folder)


if __name__ == '__main__':
    # Usage
    video_path = "squirrel.mp4"
    video_processor = VideoProcessor(video_path)
    video_processor.process_video_frames()
