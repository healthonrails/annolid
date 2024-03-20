import os
import cv2
from pathlib import Path
from annolid.segmentation.SAM.segment_anything import SegmentAnythingModel
from annolid.data.videos import CV2Video
from annolid.utils.files import find_most_recent_file
import json
from annolid.gui.shape import Shape
from annolid.annotation.keypoints import save_labels
import numpy as np
import labelme
from annolid.annotation.masks import mask_to_polygons
from collections import deque, defaultdict
from shapely.geometry import Point, Polygon
from annolid.segmentation.SAM.sam_hq import SamHQSegmenter
from annolid.gui.shape import MaskShape
from annolid.segmentation.SAM.efficientvit_sam import EfficientViTSAM
from annolid.segmentation.cutie_vos.predict import CutieVideoProcessor
from labelme.utils.shape import shapes_to_label
from annolid.utils.logger import logger
from annolid.tracker.cotracker.track import CoTrackerProcessor


def uniform_points_inside_polygon(polygon, num_points):
    # Get the bounding box of the polygon
    min_x, min_y, max_x, max_y = polygon.bounds

    # Generate random points within the bounding box
    random_points = np.column_stack((np.random.uniform(min_x, max_x, num_points),
                                     np.random.uniform(min_y, max_y, num_points)))

    # Filter points that are inside the polygon
    inside_points = [
        point for point in random_points if Point(point).within(polygon)]

    return np.array(inside_points)


def find_polygon_center(polygon_points):
    # Convert the list of polygon points to a Shapely Polygon
    polygon = Polygon(polygon_points)

    # Find the center of the polygon
    center = polygon.centroid

    return center


def random_sample_near_center(center, num_points, max_distance):
    # Randomly sample points near the center
    sampled_points = []
    for _ in range(num_points):
        # Generate random angle and radius
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, max_distance)

        # Calculate new point coordinates
        x = center.x + radius * np.cos(angle)
        y = center.y + radius * np.sin(angle)

        sampled_points.append((x, y))

    return np.array(sampled_points)


def random_sample_inside_edges(polygon, num_points):
    # Randomly sample points inside the edges of the polygon
    sampled_points = []
    min_x, min_y, max_x, max_y = polygon.bounds

    for _ in range(num_points):
        # Generate random point inside the bounding box
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = Point(x, y)

        # Check if the point is inside the polygon
        if point.within(polygon):
            sampled_points.append((x, y))

    return np.array(sampled_points)


def random_sample_outside_edges(polygon, num_points):
    # Randomly sample points inside the edges of the polygon
    sampled_points = []
    min_x, min_y, max_x, max_y = polygon.bounds

    for _ in range(num_points):
        # Generate random point inside the bounding box
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)
        point = Point(x, y)

        # Check if the point is inside the polygon
        if not point.within(polygon):
            sampled_points.append((x, y))

    return np.array(sampled_points)


def find_bbox_center(polygon_points):
    # Convert the list of polygon points to a NumPy array
    points_array = np.array(polygon_points)

    # Calculate the bounding box
    min_x, min_y = np.min(points_array, axis=0)
    max_x, max_y = np.max(points_array, axis=0)

    # Calculate the center point of the bounding box
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    # Return the center point as a NumPy array
    bbox_center = np.array([(center_x, center_y)])

    return bbox_center


def find_bbox(polygon_points):
    # Convert the list of polygon points to a NumPy array
    points_array = np.array(polygon_points)

    # Calculate the bounding box
    min_x, min_y = np.min(points_array, axis=0)
    max_x, max_y = np.max(points_array, axis=0)

    # Return the top left point and bottom right as a NumPy array
    bbox_ = np.array([min_x, min_y, max_x, max_y])

    return bbox_


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
    sam_hq = None
    cutie_processor = None

    def __init__(self,
                 video_path,
                 num_center_points=3,
                 model_name="Segment-Anything (Edge)",
                 save_image_to_disk=True
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
        self.first_frame = self.video_loader.get_first_frame()
        self.sam_name = model_name
        if model_name == 'sam_hq' and VideoProcessor.sam_hq is None:
            VideoProcessor.sam_hq = SamHQSegmenter()
        elif model_name == "Segment-Anything (Edge)":
            self.edge_sam = self.get_model()
        elif model_name == "efficientvit_sam":
            self.edge_sam = EfficientViTSAM()
        self.num_frames = self.video_loader.total_frames()
        self.most_recent_file = self.get_most_recent_file()
        self.num_points_inside_edges = num_center_points
        self.num_center_points = num_center_points
        self.center_points_dict = defaultdict()
        self.save_image_to_disk = save_image_to_disk
        self.pred_worker = None

    def set_pred_worker(self, pred_worker):
        self.pred_worker = pred_worker

    def load_shapes(self, label_json_file):
        with open(label_json_file, 'r') as json_file:
            data = json.load(json_file)
        shapes = data.get('shapes', [])
        return shapes

    def process_video_with_cutite(self, frames_to_propagate=100,
                                  mem_every=5,
                                  has_occlusion=False,
                                  ):
        self.most_recent_file = self.get_most_recent_file()
        if self.most_recent_file is None:
            message = f"No label file was saved.Please save a frame #0"
            logger.info(message)
            return message
        label_name_to_value = {"_background_": 0}
        frame_number = int(
            Path(self.most_recent_file).stem.split('_')[-1])
        shapes = self.load_shapes(self.most_recent_file)
        shapes = [shape for shape in shapes if len(shape["points"]) >= 3]
        if len(shapes) < 1:
            return f"No valid polygon found in this frame; requires a minimum of 3 points #{frame_number} "
        for shape in sorted(shapes, key=lambda x: x["label"]):
            label_name = shape["label"]
            if label_name in label_name_to_value:
                label_value = label_name_to_value[label_name]
            else:
                label_value = len(label_name_to_value)
                label_name_to_value[label_name] = label_value

        image_size = self.first_frame.shape
        logger.info(f"Loaded the shapes from: {self.most_recent_file}")
        logger.info(f"Frame size: {image_size}")
        mask, _ = shapes_to_label(
            image_size, shapes, label_name_to_value)
        if VideoProcessor.sam_hq is None:
            VideoProcessor.sam_hq = SamHQSegmenter()
        if VideoProcessor.cutie_processor is None:
            VideoProcessor.cutie_processor = CutieVideoProcessor(
                self.video_path, mem_every=mem_every, debug=False)
        VideoProcessor.cutie_processor.set_same_hq(VideoProcessor.sam_hq)
        message = VideoProcessor.cutie_processor.process_video_with_mask(frame_number,
                                                                         mask,
                                                                         frames_to_propagate=frames_to_propagate,
                                                                         visualize_every=20,
                                                                         labels_dict=label_name_to_value,
                                                                         pred_worker=self.pred_worker,
                                                                         has_occlusion=has_occlusion,
                                                                         )
        return message

    def get_model(self,
                  encoder_path="edge_sam_3x_encoder.onnx",
                  decoder_path="edge_sam_3x_decoder.onnx"
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
        name = "Segment-Anything (Edge)"
        current_file_path = os.path.abspath(__file__)
        current_folder = os.path.dirname(current_file_path)
        encoder_path = os.path.join(current_folder, encoder_path)
        decoder_path = os.path.join(current_folder, decoder_path)
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
            mask = labelme.utils.img_b64_to_arr(
                shape["mask"]) if shape.get("mask") else None
            if mask is not None:
                polygons, has_holes = mask_to_polygons(mask)
                polys = polygons[0]
                points = np.array(
                    list(zip(polys[0::2], polys[1::2])))

            if label and points is not None:
                points_dict[label] = points
                point_labels_dict[label] = 1

        return points_dict, point_labels_dict

    def process_frame(self, frame_number):
        """
        Process a single frame of the video.

        Parameters:
        - frame_number (int): Frame number to process.
        """
        cur_frame = self.video_loader.load_frame(frame_number)
        cur_frame = cv2.cvtColor(cur_frame, cv2.COLOR_BGR2RGB)
        if self.sam_name == "Segment-Anything (Edge)":
            self.edge_sam.set_image(cur_frame)
        filename = self.video_folder / \
            (self.video_folder.name + f"_{frame_number:0>{9}}.json")

        height, width, _ = cur_frame.shape
        if self.most_recent_file is None:
            return
        if (str(frame_number) not in str(self.most_recent_file) or
                str(frame_number - 1) not in str(self.most_recent_file)):
            last_frame_annotation = self.video_folder / \
                (self.video_folder.name + f"_{frame_number-1:0>{9}}.json")
            if os.path.exists(last_frame_annotation):
                self.most_recent_file = last_frame_annotation

        points_dict, _ = self.load_json_file(self.most_recent_file)
        label_list = []

        if self.sam_name == 'sam_hq' or self.sam_name == "efficientvit_sam":
            bboxes = []
            for label, points in points_dict.items():
                _bbox = find_bbox(points)
                bboxes.append((_bbox, label))

            _bboxes = [list(box) for box, _ in bboxes]
            if self.sam_name == 'sam_hq':
                masks, scores, input_box = VideoProcessor.sam_hq.segment_objects(
                    cur_frame, _bboxes)
            elif self.sam_name == "efficientvit_sam":
                masks = self.edge_sam.run_inference(cur_frame, _bboxes)

            for i, (box, label) in enumerate(bboxes):

                current_shape = MaskShape(label=label,
                                          flags={},
                                          description='grounding_sam')
                mask = masks[i]
                if self.sam_name == "efficientvit_sam":
                    h, w = mask.shape[-2:]
                    mask = mask.reshape(h, w, 1)
                current_shape.mask = mask
                current_shape = current_shape.toPolygons()[0]
                points = []
                for point in current_shape.points:
                    points.append([point.x(), point.y()])
                current_shape.points = points
                label_list.append(current_shape)
        else:
            # Example usage of predict_polygon_from_points
            for label, points in points_dict.items():
                orig_points = points
                if len(points) == 0:
                    continue
                if len(points) < 4:
                    orig_points = random_sample_near_center(
                        Point(points[0]), 4, 3)
                points = calculate_polygon_center(orig_points)

                polygon = Polygon(orig_points)
                # Randomly sample points inside the edges of the polygon
                points_inside_edges = random_sample_inside_edges(polygon,
                                                                 self.num_points_inside_edges)
                points_outside_edges = random_sample_outside_edges(polygon,
                                                                   self.num_points_inside_edges * 3
                                                                   )
                points_uni = uniform_points_inside_polygon(
                    polygon, self.num_points_inside_edges)
                center_points = self.center_points_dict.get(label,
                                                            MaxSizeQueue(max_size=self.num_center_points))

                center_points.enqueue(points[0])
                points = center_points.to_numpy()
                self.center_points_dict[label] = center_points

                # use other instance's center points as negative point prompts
                other_polygon_center_points = [
                    value for k, v in self.center_points_dict.items() if k != label for value in v]
                other_polygon_center_points = np.array(
                    [(x[0], x[1]) for x in other_polygon_center_points])

                if len(points_inside_edges.shape) > 1:
                    points = np.concatenate(
                        (points, points_inside_edges), axis=0)
                if len(points_uni) > 1:
                    points = np.concatenate(
                        (points, points_uni), axis=0
                    )

                point_labels = [1] * len(points)
                if len(points_outside_edges) > 1:
                    points = np.concatenate(
                        (points, points_outside_edges), axis=0
                    )
                    point_labels += [0] * len(points_outside_edges)

                if len(other_polygon_center_points) > 1:
                    points = np.concatenate(
                        (points, other_polygon_center_points),
                        axis=0
                    )
                    point_labels += [0] * len(other_polygon_center_points)

                polygon = self.edge_sam.predict_polygon_from_points(
                    points, point_labels)

                # Save the LabelMe JSON to a file
                p_shape = Shape(label, shape_type='polygon', flags={})
                for x, y in polygon:
                    # do not add 0,0 to the list
                    if x >= 1 and y >= 1:
                        p_shape.addPoint((x, y))
                label_list.append(p_shape)

        self.most_recent_file = filename
        img_filename = None
        if self.save_image_to_disk:
            img_filename = str(filename.with_suffix('.png'))
            if not Path(img_filename).exists():
                cv2.imwrite(img_filename, cur_frame)

        save_labels(filename=filename, imagePath=img_filename, label_list=label_list,
                    height=height, width=width, save_image_to_json=False)

    def process_video_frames(self,
                             start_frame=0,
                             end_frame=None,
                             step=10,
                             is_cutie=True,
                             mem_every=5,
                             point_tracking=False,
                             has_occlusion=False,
                             ):
        """
        Process multiple frames of the video.

        Parameters:
        - start_frame (int): Starting frame number.
        - end_frame (int): Ending frame number.
        - step (int): Step between frames.
        """
        while not self.pred_worker.is_stopped():
            if is_cutie:
                # always predict to the end of the video
                end_frame = self.num_frames
                message = self.process_video_with_cutite(
                    frames_to_propagate=end_frame,
                    mem_every=mem_every,
                    has_occlusion=has_occlusion,
                )
                return message
            elif point_tracking:
                tracker_processor = CoTrackerProcessor(
                    self.video_path, self.most_recent_file)
                message = tracker_processor.process_video()
                self.pred_worker.stop_signal.emit()
                return message
            elif not point_tracking and not is_cutie:
                if end_frame is None:
                    end_frame = self.num_frames
                for i in range(start_frame, end_frame + 1, step):
                    self.process_frame(i)
            else:
                self.pred_worker.stop_singal.emit()
                return f"Not implemented#404"

    def get_most_recent_file(self):
        """
        Find the most recent file in the video folder.

        Returns:
        - str: Path to the most recent file.
        """
        _recent_file = find_most_recent_file(self.video_folder)
        return _recent_file


if __name__ == '__main__':
    # Usage
    video_path = "squirrel.mp4"
    video_processor = VideoProcessor(video_path)
    video_processor.process_video_frames()
