import copy
import math
import numpy as np
import cv2
from qtpy import QtCore
from qtpy import QtGui
from labelme.logger import logger
import labelme.utils
import skimage.measure
from shapely.geometry import Polygon
from shapely.validation import explain_validity


# TODO(unknown):
# - [opt] Store paths instead of creating new ones at each paint.


DEFAULT_LINE_COLOR = QtGui.QColor(0, 255, 0, 128)  # bf hovering
DEFAULT_FILL_COLOR = QtGui.QColor(0, 255, 0, 128)  # hovering
DEFAULT_SELECT_LINE_COLOR = QtGui.QColor(255, 255, 255)  # selected
DEFAULT_SELECT_FILL_COLOR = QtGui.QColor(0, 255, 0, 155)  # selected
DEFAULT_VERTEX_FILL_COLOR = QtGui.QColor(0, 255, 0, 255)  # hovering
DEFAULT_HVERTEX_FILL_COLOR = QtGui.QColor(255, 255, 255, 255)  # hovering
DEFAULT_NEG_VERTEX_FILL_COLOR = QtGui.QColor(255, 0, 0, 255)


class Shape(object):

    # Render handles as squares
    P_SQUARE = 0

    # Render handles as circles
    P_ROUND = 1

    # Flag for the handles we would move if dragging
    MOVE_VERTEX = 0

    # Flag for all other handles on the current shape
    NEAR_VERTEX = 1

    # The following class variables influence the drawing of all shape objects.
    line_color = DEFAULT_LINE_COLOR
    fill_color = DEFAULT_FILL_COLOR
    select_line_color = DEFAULT_SELECT_LINE_COLOR
    select_fill_color = DEFAULT_SELECT_FILL_COLOR
    vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    hvertex_fill_color = DEFAULT_HVERTEX_FILL_COLOR
    point_type = P_ROUND
    point_size = 8
    scale = 1.0

    def __init__(
        self,
        label=None,
        line_color=None,
        shape_type=None,
        flags=None,
        group_id=None,
        description=None,
        mask=None,
        visible=True,
    ):
        self.label = label
        self.group_id = group_id
        self.points = []
        self.point_labels = []
        self.fill = False
        self.selected = False
        self._shape_raw = None
        self._points_raw = []
        self._shape_type_raw = None
        self.shape_type = shape_type
        self.flags = flags
        self.description = description
        self.other_data = {}
        self.mask = mask
        self.visible = visible

        self._highlightIndex = None
        self._highlightMode = self.NEAR_VERTEX
        self._highlightSettings = {
            self.NEAR_VERTEX: (4, self.P_ROUND),
            self.MOVE_VERTEX: (1.5, self.P_SQUARE),
        }

        self._closed = False

        if line_color is not None:
            # Override the class line_color attribute
            # with an object attribute. Currently this
            # is used for drawing the pending line a different color.
            self.line_color = line_color

    def setShapeRefined(self, shape_type, points, point_labels, mask=None):
        self._shape_raw = (self.shape_type, self.points, self.point_labels)
        self.shape_type = shape_type
        self.points = points
        self.point_labels = point_labels
        self.mask = mask

    def restoreShapeRaw(self):
        if self._shape_raw is None:
            return
        self.shape_type, self.points, self.point_labels = self._shape_raw
        self._shape_raw = None

    @property
    def shape_type(self):
        return self._shape_type

    @shape_type.setter
    def shape_type(self, value):
        if value is None:
            value = "polygon"
        if value not in [
            "polygon",
            "rectangle",
            "point",
            "line",
            "circle",
            "linestrip",
            "multipoints",
            "points",
            "mask",
        ]:
            raise ValueError("Unexpected shape_type: {}".format(value))
        self._shape_type = value

    def close(self):
        self._closed = True

    def addPoint(self, point, label=1):
        if self.points and point == self.points[0]:
            self.close()
        else:
            self.points.append(point)
            self.point_labels.append(label)

    def canAddPoint(self):
        return self.shape_type in ["polygon", "linestrip"]

    def popPoint(self):
        if self.points:
            return self.points.pop()
        return None

    def insertPoint(self, i, point, label=1):
        self.points.insert(i, point)
        self.point_labels.insert(i, label)

    def removePoint(self, i):
        if not self.canAddPoint():
            logger.warning(
                "Cannot remove point from: shape_type=%r",
                self.shape_type,
            )
            return

        if self.shape_type == "polygon" and len(self.points) <= 3:
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return

        if self.shape_type == "linestrip" and len(self.points) <= 2:
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return
        self.points.pop(i)
        self.point_labels.pop(i)

    def isClosed(self):
        return self._closed

    def setOpen(self):
        self._closed = False

    def find_polygon_center(self, points):
        n = len(points)
        x_sum = sum(p.x() for p in points)
        y_sum = sum(p.y() for p in points)
        x_center = x_sum / n
        y_center = y_sum / n
        return (x_center, y_center)

    def getRectFromLine(self, pt1, pt2):
        x1, y1 = pt1.x(), pt1.y()
        x2, y2 = pt2.x(), pt2.y()
        return QtCore.QRectF(x1, y1, x2 - x1, y2 - y1)

    def paint(self, painter):
        if self.mask is None and not self.points:
            return

        color = self.select_line_color if self.selected else self.line_color
        pen = QtGui.QPen(color)
        # Try using integer sizes for smoother drawing(?)
        pen.setWidth(max(1, int(round(2.0 / self.scale))))
        painter.setPen(pen)

        if self.mask is not None:
            image_to_draw = np.zeros(self.mask.shape + (4,), dtype=np.uint8)
            fill_color = (
                self.select_fill_color.getRgb()
                if self.selected
                else self.fill_color.getRgb()
            )
            image_to_draw[self.mask] = fill_color
            qimage = QtGui.QImage.fromData(
                labelme.utils.img_arr_to_data(image_to_draw)
            )
            painter.drawImage(
                int(round(self.points[0].x())),
                int(round(self.points[0].y())),
                qimage,
            )

            line_path = QtGui.QPainterPath()
            contours = skimage.measure.find_contours(
                np.pad(self.mask, pad_width=1)
            )
            for contour in contours:
                contour += [self.points[0].y(), self.points[0].x()]
                line_path.moveTo(contour[0, 1], contour[0, 0])
                for point in contour[1:]:
                    line_path.lineTo(point[1], point[0])
            painter.drawPath(line_path)

        if self.points:
            line_path = QtGui.QPainterPath()
            vrtx_path = QtGui.QPainterPath()
            negative_vrtx_path = QtGui.QPainterPath()

            if self.shape_type in ["rectangle", "mask"]:
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getRectFromLine(*self.points)
                    line_path.addRect(rectangle)
                if self.shape_type == "rectangle":
                    for i in range(len(self.points)):
                        self.drawVertex(vrtx_path, i)
            elif self.shape_type == "circle":
                assert len(self.points) in [1, 2]
                if len(self.points) == 2:
                    rectangle = self.getCircleRectFromLine(self.points)
                    line_path.addEllipse(rectangle)
                for i in range(len(self.points)):
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "linestrip":
                line_path.moveTo(self.points[0])
                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
            elif self.shape_type == "points":
                assert len(self.points) == len(self.point_labels)
                for i, (p, l) in enumerate(
                    zip(self.points, self.point_labels)
                ):
                    if l == 1:
                        self.drawVertex(vrtx_path, i)
                    else:
                        self.drawVertex(negative_vrtx_path, i)
            else:
                line_path.moveTo(self.points[0])
                # Uncommenting the following line will draw 2 paths
                # for the 1st vertex, and make it non-filled, which
                # may be desirable.
                # self.drawVertex(vrtx_path, 0)

                for i, p in enumerate(self.points):
                    line_path.lineTo(p)
                    self.drawVertex(vrtx_path, i)
                if self.isClosed():
                    line_path.lineTo(self.points[0])

            painter.drawPath(line_path)
            if vrtx_path.length() > 0:
                painter.drawPath(vrtx_path)
                painter.fillPath(vrtx_path, self._vertex_fill_color)
            if self.fill and self.mask is None:
                color = (
                    self.select_fill_color
                    if self.selected
                    else self.fill_color
                )
                painter.fillPath(line_path, color)

            pen.setColor(QtGui.QColor(255, 0, 0, 255))
            painter.setPen(pen)
            painter.drawPath(negative_vrtx_path)
            painter.fillPath(negative_vrtx_path, QtGui.QColor(255, 0, 0, 255))

    def drawVertex(self, path, i):
        d = self.point_size / self.scale
        shape = self.point_type
        point = self.points[i]
        if i == self._highlightIndex:
            size, shape = self._highlightSettings[self._highlightMode]
            d *= size
        if self._highlightIndex is not None:
            self._vertex_fill_color = self.hvertex_fill_color
        else:
            self._vertex_fill_color = self.vertex_fill_color
        if shape == self.P_SQUARE:
            path.addRect(point.x() - d / 2, point.y() - d / 2, d, d)
        elif shape == self.P_ROUND:
            path.addEllipse(point, d / 2.0, d / 2.0)
        else:
            assert False, "unsupported vertex shape"

    def nearestVertex(self, point, epsilon):
        min_distance = float("inf")
        min_i = None
        for i, p in enumerate(self.points):
            dist = labelme.utils.distance(p - point)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                min_i = i
        return min_i

    def nearestEdge(self, point, epsilon):
        min_distance = float("inf")
        post_i = None
        for i in range(len(self.points)):
            line = [self.points[i - 1], self.points[i]]
            dist = labelme.utils.distancetoline(point, line)
            if dist <= epsilon and dist < min_distance:
                min_distance = dist
                post_i = i
        return post_i

    def containsPoint(self, point):
        if self.mask is not None:
            y = np.clip(
                int(round(point.y() - self.points[0].y())),
                0,
                self.mask.shape[0] - 1,
            )
            x = np.clip(
                int(round(point.x() - self.points[0].x())),
                0,
                self.mask.shape[1] - 1,
            )
            return self.mask[y, x]

        if not self.makePath():
            return False
        return self.makePath().contains(point)

    def getCircleRectFromLine(self, line):
        """Computes parameters to draw with `QPainterPath::addEllipse`"""
        if len(line) != 2:
            return None
        (c, point) = line
        r = line[0] - line[1]
        d = math.sqrt(math.pow(r.x(), 2) + math.pow(r.y(), 2))
        rectangle = QtCore.QRectF(c.x() - d, c.y() - d, 2 * d, 2 * d)
        return rectangle

    def makePath(self):
        if self.shape_type in ["rectangle", "mask"]:
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getRectFromLine(*self.points)
                path.addRect(rectangle)
        elif self.shape_type == "circle":
            path = QtGui.QPainterPath()
            if len(self.points) == 2:
                rectangle = self.getCircleRectFromLine(self.points)
                path.addEllipse(rectangle)
        else:
            path = QtGui.QPainterPath(self.points[0])
            for p in self.points[1:]:
                path.lineTo(p)
            return path

    def boundingRect(self):
        return self.makePath().boundingRect()

    def moveBy(self, offset):
        self.points = [p + offset for p in self.points]

    def moveVertexBy(self, i, offset):
        self.points[i] = self.points[i] + offset

    def highlightVertex(self, i, action):
        self._highlightIndex = i
        self._highlightMode = action

    def highlightClear(self):
        self._highlightIndex = None

    def copy(self):
        return copy.deepcopy(self)

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points[key]

    def __setitem__(self, key, value):
        self.points[key] = value

    def __hash__(self):
        # Hash based on label, shape_type, and points
        return hash((self.label, self.shape_type,
                     tuple((p.x(), p.y()) for p in self.points)))

    def calculate_iou(self, other):
        """
        Calculate the Intersection over Union (IoU) between two polygons.

        Args:
            other (Shape): The other shape to compare with.

        Returns:
            float: IoU value between the two polygons.
        """
        # Check if the shapes are polygons
        if self.shape_type != 'polygon' or other.shape_type != 'polygon':
            raise ValueError("Both shapes must be polygons to calculate IoU.")

        # Convert the vertices to Shapely Polygon objects
        poly1 = Polygon([(p.x(), p.y()) for p in self.points])
        poly2 = Polygon([(p.x(), p.y()) for p in other.points])
        # Check if the polygons are valid
        if not poly1.is_valid:
            poly1 = poly1.buffer(0)  # Attempt to fix self-intersections

        if not poly2.is_valid:
            poly2 = poly2.buffer(0)  # Attempt to fix self-intersections

        # Calculate intersection area
        intersection_area = poly1.intersection(poly2).area

        # Calculate union area
        union_area = poly1.union(poly2).area

        # Calculate IoU
        if union_area == 0:
            iou = 0  # Set the IOU to 0 if the union area is zero
        else:
            iou = intersection_area / union_area

        return iou

    def __eq__(self, other, epsilon=1e-1,
               iou_threshold=0.90):
        """
        Test if two shapes are equal or similar.

        Args:
            other (Shape): The other shape to compare with.
            epsilon (float): Tolerance for comparing point coordinates.
            iou_threshold (float): Threshold for considering IoU as a match.

        Returns:
            bool: True if the shapes are equal or similar, False otherwise.
        """
        # Check if the other object is None
        if other is None:
            return False

        # Check if the other object is not an instance of the Shape class
        if not isinstance(other, Shape):
            return False

        # Check if the shapes have the same type
        if self.shape_type != other.shape_type:
            return False

        # If shapes are polygons, check IoU
        if self.shape_type == "polygon":
            if len(self.points) >= 4 and len(other.points) >= 4:
                iou = self.calculate_iou(other)
                if iou < iou_threshold:
                    return False

        # Check if the number of points are the same
        if len(self.points) != len(other.points):
            return False

        # Check if other properties such as label, fill, etc. are equal
        if (self.label != other.label or self._closed != other._closed):
            return False

        # Check if each pair of points are similar within the epsilon tolerance
        for point_self, point_other in zip(self.points, other.points):
            # Extract coordinates of points
            x_self, y_self = point_self.x(), point_self.y()
            x_other, y_other = point_other.x(), point_other.y()

            # Calculate distance between points
            distance = ((x_self - x_other) ** 2 +
                        (y_self - y_other) ** 2) ** 0.5

            # Check if distance is within epsilon tolerance
            if distance > epsilon:
                return False

        # If all checks passed, return True
        return True


class MultipoinstShape(Shape):
    """
    Modified from
    https://github.com/originlake/labelme-with-segment-anything/blob/main/labelme/shape.py
    """

    positive_vertex_fill_color = DEFAULT_VERTEX_FILL_COLOR
    negative_vertex_fill_color = DEFAULT_NEG_VERTEX_FILL_COLOR

    def __init__(
        self,
    ):
        super(MultipoinstShape, self).__init__()
        self.labels = []
        self.shape_type = 'multipoints'

    def addPoint(self, point, is_positive=True):
        if not self.points or point != self.points[0]:
            self.points.append(point)
            self.labels.append(is_positive)

    def canAddPoint(self):
        return True

    def popPoint(self):
        if self.points:
            self.labels.pop()
            return self.points.pop()
        return None

    def removePoint(self, i):

        if len(self.points) <= 1:
            logger.warning(
                "Cannot remove point from: shape_type=%r, len(points)=%d",
                self.shape_type,
                len(self.points),
            )
            return
        self.labels.pop(i)
        self.points.pop(i)

    def paint(self, painter):
        if self.points:
            d = self.point_size / self.scale
            shape = self.point_type
            pos_pen = QtGui.QPen(self.positive_vertex_fill_color)
            neg_pen = QtGui.QPen(self.negative_vertex_fill_color)
            # Try using integer sizes for smoother drawing(?)
            pos_pen.setWidth(max(1, int(round(2.0 / self.scale))))
            neg_pen.setWidth(max(1, int(round(2.0 / self.scale))))
            for i, (point, is_positive) in enumerate(zip(self.points, self.labels)):
                if is_positive:
                    painter.setPen(pos_pen)
                else:
                    painter.setPen(neg_pen)

                if shape == self.P_SQUARE:
                    painter.drawRect(point.x() - d / 2,
                                     point.y() - d / 2, d, d)
                elif shape == self.P_ROUND:
                    painter.drawEllipse(point, d / 2.0, d / 2.0)
                else:
                    assert False, "unsupported vertex shape"

    def highlightClear(self):
        """Clear the highlighted point"""
        # self._highlightIndex = None
        pass


class MaskShape(MultipoinstShape):
    """
    Modified from
    https://github.com/originlake/labelme-with-segment-anything/blob/main/labelme/shape.py
    """

    mask_color = np.array([0, 0, 255, 64], np.uint8)
    boundary_color = np.array([0, 0, 255, 128], np.uint8)

    def __init__(self,
                 label=None,
                 group_id=None,
                 flags=None,
                 description=None):
        self.label = label
        self.group_id = group_id
        self.fill = False
        self.selected = False
        self.flags = flags
        self.description = description
        self.other_data = {}
        self.rgba_mask = None
        self.mask = None
        self.logits = None
        self.scale = 1
        self.points = []

    def setScaleMask(self, scale, mask):
        self.scale = scale
        self.mask = mask

    def getQImageMask(self,):
        if self.mask is None:
            return None
        mask = (self.mask * 255).astype(np.uint8)
        mask = cv2.resize(mask, None, fx=1/self.scale, fy=1 /
                          self.scale, interpolation=cv2.INTER_NEAREST)
        if self.rgba_mask is not None and mask.shape[0] == self.rgba_mask.shape[0] and mask.shape[1] == self.rgba_mask.shape[1]:
            self.rgba_mask[:] = 0
        else:
            self.rgba_mask = np.zeros(
                [mask.shape[0], mask.shape[1], 4], dtype=np.uint8)
        self.rgba_mask[mask > 128] = self.mask_color
        kernel = np.ones([5, 5], dtype=np.uint8)
        bound = mask - cv2.erode(mask, kernel, iterations=1)
        self.rgba_mask[bound > 128] = self.boundary_color
        qimage = QtGui.QImage(
            self.rgba_mask.data, self.rgba_mask.shape[1], self.rgba_mask.shape[0], QtGui.QImage.Format_RGBA8888)
        return qimage

    def paint(self, painter):
        qimage = self.getQImageMask()
        if qimage is not None:
            painter.drawImage(QtCore.QPoint(0, 0), qimage)

    def toPolygons(self, epsilon=2.0, merge_contours=False):
        # Fill the holes inside the mask
        filled_mask = cv2.morphologyEx((self.mask*255).astype(
            np.uint8), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        # Find the contours of the filled mask
        contours, hierarchy = cv2.findContours(
            filled_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        shapes = []
        if len(contours) == 0:
            return shapes

        if merge_contours:
            # Merge all the contours into a single contour
            merged_contour = np.concatenate(contours)
            merged_contour = cv2.approxPolyDP(merged_contour, epsilon, True)
            merged_contour = merged_contour[:, 0, :] / self.scale
            merged_contour = np.concatenate(
                [merged_contour, merged_contour[:1, :]], axis=0)
        else:
            # Find the contour with the largest area
            largest_contour = max(contours, key=cv2.contourArea)

            # Approximate the largest contour using Douglas-Peucker algorithm
            epsilon = 0.01 * cv2.arcLength(largest_contour, True)
            approximated_contour = cv2.approxPolyDP(
                largest_contour, epsilon, True)

            # Scale the merged contour if necessary
            scaled_contour = approximated_contour[:, 0, :] / self.scale

            # Close the contour by duplicating its first point at the end
            merged_contour = np.concatenate(
                [scaled_contour, scaled_contour[:1, :]], axis=0)

        # Create a Shape object from the merged contour
        shape = Shape(shape_type="polygon",
                      label=self.label,
                      group_id=self.group_id,
                      flags=self.flags,
                      description=self.description)
        for x, y in merged_contour:
            shape.addPoint(QtCore.QPointF(x, y))
        self.points = shape.points
        shapes.append(shape)
        return shapes

    def copy(self):
        return copy.deepcopy(self)
