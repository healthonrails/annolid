from qtpy import QtCore
from qtpy import QtGui
from qtpy import QtWidgets
from qtpy.QtWidgets import QLabel
from labelme import QT5
import numpy as np
from PIL import Image
import cv2
import os
import imgviz
import labelme.ai
from annolid.utils.logger import logger
import labelme.utils
from annolid.utils.qt2cv import convert_qt_image_to_rgb_cv_image
from annolid.utils.prompts import extract_number_and_remove_digits
from annolid.gui.shape import Shape, MaskShape, MultipoinstShape
from annolid.detector.grounding_dino import GroundingDINO
from annolid.segmentation.SAM.sam_hq import SamHQSegmenter
# TODO(unknown):
# - [maybe] Find optimal epsilon value.


CURSOR_DEFAULT = QtCore.Qt.ArrowCursor
CURSOR_POINT = QtCore.Qt.PointingHandCursor
CURSOR_DRAW = QtCore.Qt.CrossCursor
CURSOR_MOVE = QtCore.Qt.ClosedHandCursor
CURSOR_GRAB = QtCore.Qt.OpenHandCursor

MOVE_SPEED = 5.0


class Canvas(QtWidgets.QWidget):

    zoomRequest = QtCore.Signal(int, QtCore.QPoint)
    scrollRequest = QtCore.Signal(int, int)
    newShape = QtCore.Signal()
    selectionChanged = QtCore.Signal(list)
    shapeMoved = QtCore.Signal()
    drawingPolygon = QtCore.Signal(bool)
    vertexSelected = QtCore.Signal(bool)

    CREATE, EDIT = 0, 1

    # polygon, rectangle, line, or point
    _createMode = "polygon"

    _fill_drawing = False

    def __init__(self, *args, **kwargs):
        self.epsilon = kwargs.pop("epsilon", 10.0)
        self.double_click = kwargs.pop("double_click", "close")
        if self.double_click not in [None, "close"]:
            raise ValueError(
                "Unexpected value for double_click event: {}".format(
                    self.double_click
                )
            )
        self.caption_label = QtWidgets.QTextEdit()
        self.num_backups = kwargs.pop("num_backups", 10)
        self._crosshair = kwargs.pop(
            "crosshair",
            {
                "polygon": False,
                "rectangle": True,
                "grounding_sam": False,
                "circle": False,
                "line": False,
                "point": False,
                "linestrip": False,
                "ai_polygon": False,
                "ai_mask": False,
                "polygonSAM": False,
            },
        )
        self.sam_config = kwargs.pop(
            "sam",
            {
                "maxside": 2048,
                "approxpoly_epsilon": 0.5,
                "weights": "vit_h",
                "device": "cuda"
            }
        )
        super(Canvas, self).__init__(*args, **kwargs)
        # Initialise local state.
        self.mode = self.EDIT
        self.shapes = []
        self.shapesBackups = []
        self.current = None
        self.selectedShapes = []  # save the selected shapes here
        self.selectedShapesCopy = []
        # self.line represents:
        #   - createMode == 'polygon': edge from last point to current
        #   - createMode == 'rectangle': diagonal line of the rectangle
        #   - createMode == 'line': the line
        #   - createMode == 'point': the point
        self.line = Shape()
        self.prevPoint = QtCore.QPoint()
        self.prevMovePoint = QtCore.QPoint()
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.scale = 1.0
        self.pixmap = QtGui.QPixmap()
        self.visible = {}
        self._hideBackround = False
        self.hideBackround = False
        self.hShape = None
        self.prevhShape = None
        self.hVertex = None
        self.prevhVertex = None
        self.hEdge = None
        self.prevhEdge = None
        self.movingShape = False
        self.snapping = True
        self.hShapeIsSelected = False
        self._painter = QtGui.QPainter()
        self._cursor = CURSOR_DEFAULT
        self.mouse_xy_text = ""
        # Collect shapes that need prediction
        self.shapes_to_predict = []

        self.label = QLabel(self)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        # self.label.resize(120, 20)
        # self.label.setStyleSheet("background-color:white")

        # Menus:
        # 0: right-click without selection and dragging of shapes
        # 1: right-click with selection and dragging of shapes
        self.menus = (QtWidgets.QMenu(), QtWidgets.QMenu())
        # Set widget options.
        self.setMouseTracking(True)
        self.setFocusPolicy(QtCore.Qt.WheelFocus)
        self._ai_model = None
        self._ai_model_rect = None
        self.sam_predictor = None
        self.sam_hq_model = None
        self.sam_mask = MaskShape()
        self.behavior_text_position = "top-left"  # Default position
        self.behavior_text_color = QtGui.QColor(255, 255, 255)
        self.behavior_text_background = None  # Optional background color
        self.current_behavior_text = None

        # Patch similarity helpers
        self._patch_similarity_active = False
        self._patch_similarity_callback = None
        self._patch_similarity_pixmap = None
        self._pca_map_pixmap = None

    def fillDrawing(self):
        return self._fill_drawing

    def setFillDrawing(self, value):
        self._fill_drawing = value

    def setCaption(self, text):
        self.caption_label.setText(text)

    def setBehaviorText(self, text):
        self.current_behavior_text = text

    def getCaption(self):
        return self.caption_label.toPlainText()

    # ------------------------------------------------------------------
    # Patch similarity helpers (DINO visualization)
    # ------------------------------------------------------------------
    def enablePatchSimilarityMode(self, callback):
        """Enable click-to-query mode for DINO patch similarity."""
        self._patch_similarity_active = True
        self._patch_similarity_callback = callback
        self.overrideCursor(CURSOR_POINT)

    def disablePatchSimilarityMode(self):
        self._patch_similarity_active = False
        self._patch_similarity_callback = None
        self._patch_similarity_pixmap = None
        self.restoreCursor()
        self.update()

    def setPatchSimilarityOverlay(self, overlay_rgba):
        """Update or clear the rendered heatmap overlay."""
        if overlay_rgba is None:
            self._patch_similarity_pixmap = None
        else:
            h, w, _ = overlay_rgba.shape
            image = QtGui.QImage(
                overlay_rgba.data,
                w,
                h,
                overlay_rgba.strides[0],
                QtGui.QImage.Format_RGBA8888,
            )
            self._patch_similarity_pixmap = QtGui.QPixmap.fromImage(
                image.copy())
        self.update()

    def setPCAMapOverlay(self, overlay_rgba):
        """Update or clear the PCA coloring overlay."""
        if overlay_rgba is None:
            self._pca_map_pixmap = None
        else:
            h, w, _ = overlay_rgba.shape
            image = QtGui.QImage(
                overlay_rgba.data,
                w,
                h,
                overlay_rgba.strides[0],
                QtGui.QImage.Format_RGBA8888,
            )
            self._pca_map_pixmap = QtGui.QPixmap.fromImage(image.copy())
        self.update()

    @property
    def createMode(self):
        return self._createMode

    @createMode.setter
    def createMode(self, value):
        if value not in [
            "polygon",
            "rectangle",
            "circle",
            "line",
            "point",
            "linestrip",
            "polygonSAM",
            "ai_polygon",
            "grounding_sam",
            "ai_mask",
        ]:
            raise ValueError("Unsupported createMode: %s" % value)
        self._createMode = value
        self.sam_mask = MaskShape()
        self.current = None

    def initializeAiModel(self, name, _custom_ai_models=None):
        if (name not in [model.name for model in labelme.ai.MODELS]
                and _custom_ai_models and name not in _custom_ai_models):
            logger.warning("Unsupported ai model: %s" % name)
            model = labelme.ai.MODELS[3]
        else:
            model = labelme.ai.MODELS[3]

        if self._ai_model is not None and self._ai_model.name == model.name:
            logger.debug("AI model is already initialized: %r" % model.name)
        else:
            logger.debug("Initializing AI model: %r" % model.name)
            self._ai_model = model()

        # Check if self.pixmap is None before calling isNull()
        if self.pixmap is None or self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return

        self._ai_model.set_image(
            image=labelme.utils.img_qt_to_arr(self.pixmap.toImage())
        )

    def _ensure_ai_model_initialized(self) -> bool:
        if self._ai_model is not None:
            return True
        try:
            default_model = labelme.ai.MODELS[3].name
        except Exception:
            default_model = None
        if default_model:
            self.initializeAiModel(default_model)
        return self._ai_model is not None

    def auto_mask_generator(self,
                            image_data,
                            label,
                            points_per_side=32,
                            is_polygon_output=True):
        """
        Generate masks or polygons from segmentation annotations.

        Args:
            image_data (numpy.ndarray): The input image data for segmentation.
            label (str): The label to be assigned to generated shapes.
            points_per_side (int or None): The number of points to be sampled
            along one side of the image. The total number of points is
            points_per_side**2. If None, 'point_grids' must provide explicit
            point sampling.
            is_polygon_output (bool, optional):
              Whether to output polygons (True) or masks (False).
                Defaults to True.

        Returns:
            None

        """
        # Segment everything in the image using the SamAutomaticMaskGenerator
        anns = self.sam_hq_model.segment_everything(image_data,
                                                    points_per_side=points_per_side)

        # Iterate over each segmentation annotation
        for i, ann in enumerate(anns):
            # Check if the output format is polygons
            if is_polygon_output:
                # Convert segmentation to polygons
                self.current = MaskShape(label=f"{label}_{i}",
                                         flags={},
                                         description='grounding_sam')
                self.current.mask = ann['segmentation']
                self.current = self.current.toPolygons()[0]
            else:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = ann['bbox']
                # Create shape object with refined shape
                self.current = Shape(label=f"{label}_{i}",
                                     flags={},
                                     description='grounding_sam')
                self.current.setShapeRefined(
                    shape_type="mask",
                    points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                    point_labels=[1, 1],
                    mask=ann['segmentation'][int(y1):int(y2), int(x1):int(x2)],
                )
            # Add stability score as other data
            self.current.other_data['score'] = str(ann['stability_score'])

            # Finalize the process
            self.finalise()

    def predictAiRectangle(self, prompt,
                           rectangle_shapes=None,
                           is_polygon_output=True):
        """
        Predict bounding boxes and then polygons based on the given prompt.

        Args:
            prompt (str): The prompt for prediction.
            is_polygon_output (bool, optional): 
            Whether to output polygons (True).
                Defaults to True.

        Returns:
            None

        """
        # Check if the pixmap is set
        if self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return

        # Convert Qt image to RGB OpenCV image
        qt_image = self.pixmap.toImage()
        image_data = convert_qt_image_to_rgb_cv_image(qt_image)

        # Initialize SAM HQ model if not already initialized
        if self.sam_hq_model is None:
            self.sam_hq_model = SamHQSegmenter()

        # If the prompt contains 'every', segment everything
        if prompt and 'every' in prompt.lower():
            points_per_side, prompt = extract_number_and_remove_digits(prompt)
            if points_per_side < 1 or points_per_side > 100:
                points_per_side = 32

            label = prompt.replace('every', '').strip()
            logger.info(
                f"{points_per_side} points to be sampled along one side of the image")
            self.auto_mask_generator(image_data,
                                     label,
                                     points_per_side=points_per_side,
                                     is_polygon_output=is_polygon_output)
            return

        label = prompt
        if rectangle_shapes is not None:
            _bboxes = self._predict_similar_rectangles(
                rectangle_shapes=rectangle_shapes, prompt=prompt)
        else:
            rectangle_shapes = []
            _bboxes = self._predict_similar_rectangles(
                rectangle_shapes=rectangle_shapes, prompt=prompt)
            # Initialize AI model if not already initialized
            if self._ai_model_rect is None:
                self._ai_model_rect = GroundingDINO()

            # # Predict bounding boxes using the AI model
            bboxes = self._ai_model_rect.predict_bboxes(image_data, prompt)
            gd_bboxes = [list(box) for box, _ in bboxes]
            _bboxes.extend(gd_bboxes)

        # Segment objects using SAM HQ model with predicted bounding boxes
        masks, scores, _bboxes = self.sam_hq_model.segment_objects(
            image_data, _bboxes)

        # Iterate over each predicted bounding box
        for i, box in enumerate(_bboxes):
            # Check if the output format is polygons
            if is_polygon_output:
                # Convert segmentation mask to polygons
                self.current = MaskShape(label=f"{label}_{i}",
                                         flags={},
                                         description='grounding_sam')
                self.current.mask = masks[i]
                self.current = self.current.toPolygons()[0]
            else:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = box
                # Create shape object with refined shape
                self.current = Shape(label=f"{label}_{i}",
                                     flags={},
                                     description='grounding_sam')
                self.current.setShapeRefined(
                    shape_type="mask",
                    points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                    point_labels=[1, 1],
                    mask=masks[i][int(y1):int(y2), int(x1):int(x2)],
                )
            # Add stability score as other data
            self.current.other_data['score'] = str(scores[i])

            # Finalize the process
            self.finalise()

    def loadSamPredictor(self,):
        """
        The code requires Python version 3.8 or higher, along with PyTorch version 1.7 or higher,
        as well as TorchVision version 0.8 or higher. To install these dependencies, 
        kindly refer to the instructions provided here. It's strongly recommended to install both PyTorch and TorchVision with CUDA support.
To install Segment Anything, run the following command in your terminal:
pip install git+https://github.com/facebookresearch/segment-anything.git

        """
        if self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return
        if not self.sam_predictor:
            import torch
            from pathlib import Path
            from segment_anything import sam_model_registry, SamPredictor
            from annolid.utils.devices import has_gpu
            if not has_gpu() and not torch.cuda.is_available():
                QtWidgets.QMessageBox.about(self,
                                            "GPU not available",
                                            """For optimal performance, it is recommended to \
                                                use a GPU for running the Segment Anything model. \
                                                    Running the model on a CPU will result \
                                                        in significantly longer processing times.""")
            here = Path(os.path.abspath(__file__)
                        ).parent.parent.parent / 'segmentation' / 'SAM'
            cachedir = str(here)
            os.makedirs(cachedir, exist_ok=True)
            weight_file = os.path.join(
                cachedir, self.sam_config["weights"] + ".pth")
            weight_urls = {
                "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
            }
            if not os.path.isfile(weight_file):
                torch.hub.download_url_to_file(
                    weight_urls[self.sam_config["weights"]], weight_file)
            sam = sam_model_registry[self.sam_config["weights"]](
                checkpoint=weight_file)
            if self.sam_config["device"] == "cuda" and torch.cuda.is_available():
                sam.to(device="cuda")
            self.sam_predictor = SamPredictor(sam)
        self.samEmbedding()

    def samEmbedding(self,):
        image = self.pixmap.toImage().copy()
        img_size = image.size()
        s = image.bits().asstring(img_size.height() * img_size.width() * image.depth()//8)
        image = np.frombuffer(s, dtype=np.uint8).reshape(
            [img_size.height(), img_size.width(), image.depth()//8])
        image = image[:, :, :3].copy()
        h, w, _ = image.shape
        self.sam_image_scale = self.sam_config["maxside"] / max(h, w)
        self.sam_image_scale = min(1, self.sam_image_scale)
        image = cv2.resize(image, None, fx=self.sam_image_scale,
                           fy=self.sam_image_scale, interpolation=cv2.INTER_LINEAR)
        self.sam_predictor.set_image(image)
        self.update()

    def samPrompt(self, points, labels):
        if self.pixmap.isNull():
            logger.warning("Pixmap is not set yet")
            return
        masks, scores, logits = self.sam_predictor.predict(
            point_coords=points*self.sam_image_scale,
            point_labels=labels,
            mask_input=self.sam_mask.logits[None, :,
                                            :] if self.sam_mask.logits is not None else None,
            multimask_output=False
        )
        self.sam_mask.logits = logits[np.argmax(scores), :, :]
        mask = masks[np.argmax(scores), :, :]
        self.sam_mask.setScaleMask(self.sam_image_scale, mask)

    def storeShapes(self):
        shapesBackup = []
        for shape in self.shapes:
            shapesBackup.append(shape.copy())
        if len(self.shapesBackups) > self.num_backups:
            self.shapesBackups = self.shapesBackups[-self.num_backups - 1:]
        self.shapesBackups.append(shapesBackup)

    @property
    def isShapeRestorable(self):
        # We save the state AFTER each edit (not before) so for an
        # edit to be undoable, we expect the CURRENT and the PREVIOUS state
        # to be in the undo stack.
        if len(self.shapesBackups) < 2:
            return False
        return True

    def restoreShape(self):
        # This does _part_ of the job of restoring shapes.
        # The complete process is also done in app.py::undoShapeEdit
        # and app.py::loadShapes and our own Canvas::loadShapes function.
        if not self.isShapeRestorable:
            return
        self.shapesBackups.pop()  # latest

        # The application will eventually call Canvas.loadShapes which will
        # push this right back onto the stack.
        shapesBackup = self.shapesBackups.pop()
        self.shapes = shapesBackup
        self.selectedShapes = []
        for shape in self.shapes:
            shape.selected = False
        self.update()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.unHighlight()
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def drawing(self):
        return self.mode == self.CREATE

    def editing(self):
        return self.mode == self.EDIT

    def setEditing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if self.mode == self.EDIT:
            # CREATE -> EDIT
            self.repaint()  # clear crosshair
        else:
            # EDIT -> CREATE
            self.unHighlight()
            self.deSelectShape()

    def unHighlight(self):
        if self.hShape:
            self.hShape.highlightClear()
            self.update()
        self.prevhShape = self.hShape
        self.prevhVertex = self.hVertex
        self.prevhEdge = self.hEdge
        self.hShape = self.hVertex = self.hEdge = None

    def selectedVertex(self):
        return self.hVertex is not None

    def selectedEdge(self):
        return self.hEdge is not None

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        try:
            if QT5:
                pos = self.transformPos(ev.localPos())
            else:
                pos = self.transformPos(ev.posF())
        except AttributeError:
            return

        x, y = ev.x(), ev.y()
        self.mouse_xy_text = f'x:{pos.x():.1f},y:{pos.y():.1f}'
        self.label.setText(self.mouse_xy_text)
        self.label.adjustSize()
        label_width = self.label.width()
        label_height = self.label.height()
        self.label.move(int(x - label_width / 2), int(y + label_height))
        self.prevMovePoint = pos
        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier
        self.restoreCursor()
        # Polygon drawing.
        if self.drawing():
            if self.createMode in ["ai_polygon", "ai_mask", "grounding_sam"]:
                self.line.shape_type = "points"
            else:
                self.line.shape_type = self.createMode if "polygonSAM" != self.createMode else "polygon"

            self.overrideCursor(CURSOR_DRAW)
            if not self.current:
                if self.createMode == "polygonSAM":
                    points = np.array([[pos.x(), pos.y()]])
                    labels = np.array([1])
                    self.samPrompt(points, labels)
                self.repaint()  # draw crosshair
                return

            if self.outOfPixmap(pos):
                # Don't allow the user to draw outside the pixmap.
                # Project the point to the pixmap's edges.
                pos = self.intersectionPoint(self.current[-1], pos)
            elif (
                self.snapping
                and len(self.current) > 1
                and self.createMode == "polygon"
                and self.closeEnough(pos, self.current[0])
            ):
                # Attract line to starting point and
                # colorise to alert the user.
                pos = self.current[0]
                self.overrideCursor(CURSOR_POINT)
                self.current.highlightVertex(0, Shape.NEAR_VERTEX)
            if self.createMode in ["polygon", "linestrip"]:
                self.line.points = [self.current[-1], pos]
                self.line.point_labels = [1, 1]
            elif self.createMode in ["ai_polygon", "ai_mask"]:
                self.line.points = [self.current.points[-1], pos]
                self.line.point_labels = [
                    self.current.point_labels[-1],
                    0 if is_shift_pressed else 1,
                ]
            elif self.createMode == "rectangle":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "circle":
                self.line.points = [self.current[0], pos]
                self.line.shape_type = "circle"
            elif self.createMode == "line":
                self.line.points = [self.current[0], pos]
                self.line.close()
            elif self.createMode == "point":
                self.line.points = [self.current[0]]
                self.line.close()
            elif self.createMode == "polygonSAM":
                self.line.points = [pos, pos]
            self.repaint()
            self.current.highlightClear()
            return

        # Polygon copy moving.
        if QtCore.Qt.RightButton & ev.buttons():
            if self.selectedShapesCopy and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapesCopy, pos)
                self.repaint()
            elif self.selectedShapes:
                self.selectedShapesCopy = [
                    s.copy() for s in self.selectedShapes
                ]
                self.repaint()
            return

        # Polygon/Vertex moving.
        if QtCore.Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.repaint()
                self.movingShape = True
            elif self.selectedShapes and self.prevPoint:
                self.overrideCursor(CURSOR_MOVE)
                self.boundedMoveShapes(self.selectedShapes, pos)
                self.repaint()
                self.movingShape = True
            return

        # Just hovering over the canvas, 2 possibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip(self.tr("Image"))
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon / self.scale)
            index_edge = shape.nearestEdge(pos, self.epsilon / self.scale)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex = index
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr(f"Click & drag to move point"))
                self.label.setText(shape.label + "," + self.mouse_xy_text)
                self.label.adjustSize()
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif index_edge is not None and shape.canAddPoint():
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge = index_edge
                self.overrideCursor(CURSOR_POINT)
                self.setToolTip(self.tr("Click to create point"))
                self.label.setText(shape.label + "," + self.mouse_xy_text)
                self.label.adjustSize()
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.prevhVertex = self.hVertex
                self.hVertex = None
                self.prevhShape = self.hShape = shape
                self.prevhEdge = self.hEdge
                self.hEdge = None
                self.label.setText(shape.label + "," + self.mouse_xy_text)
                self.label.adjustSize()
                self.setToolTip(
                    self.tr("Click & drag to move shape '%s'") % shape.label
                )
                self.setStatusTip(self.toolTip())
                self.overrideCursor(CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            self.unHighlight()
        self.vertexSelected.emit(self.hVertex is not None)

    def addPointToEdge(self):
        shape = self.prevhShape
        index = self.prevhEdge
        point = self.prevMovePoint
        if shape is None or index is None or point is None:
            return
        shape.insertPoint(index, point)
        shape.highlightVertex(index, shape.MOVE_VERTEX)
        self.hShape = shape
        self.hVertex = index
        self.hEdge = None
        self.movingShape = True

    def removeSelectedPoint(self):
        shape = self.prevhShape
        index = self.prevhVertex
        if shape is None or index is None:
            return
        shape.removePoint(index)
        shape.highlightClear()
        self.hShape = shape
        self.prevhVertex = None
        self.movingShape = True  # Save changes

    def contextMenuEvent(self, event):
        # If there is at least one selected shape, show the custom context menu.
        if self.selectedShapes:
            menu = QtWidgets.QMenu(self)
            propagate_action = menu.addAction("Propagate Selected Shape")
            propagate_action.triggered.connect(
                lambda: self.propagateSelectedShapeFromCanvas())
            menu.exec_(event.globalPos())
        else:
            # Otherwise, call the base class implementation.
            super(Canvas, self).contextMenuEvent(event)

    def propagateSelectedShapeFromCanvas(self):
        # Since the canvas doesn't directly have frame data,
        # we call the main window's method.
        main_window = self.window()  # Assumes top-level window is AnnolidWindow
        if hasattr(main_window, "propagateSelectedShape"):
            main_window.propagateSelectedShape()

    def mousePressEvent(self, ev):
        if QT5:
            pos = self.transformPos(ev.localPos())
        else:
            pos = self.transformPos(ev.posF())

        is_shift_pressed = ev.modifiers() & QtCore.Qt.ShiftModifier

        if self._patch_similarity_active and ev.button() == QtCore.Qt.LeftButton:
            if not self.outOfPixmap(pos) and self._patch_similarity_callback:
                self._patch_similarity_callback(int(pos.x()), int(pos.y()))
            ev.accept()
            return

        if ev.button() == QtCore.Qt.LeftButton:
            if self.drawing():
                if self.current:
                    # Add point to existing shape.
                    if self.createMode == "polygon":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if self.current.isClosed():
                            self.finalise()
                    elif self.createMode in ["rectangle", "circle", "line"]:
                        assert len(self.current.points) == 1
                        self.current.points = self.line.points
                        self.finalise()
                    elif self.createMode == "linestrip":
                        self.current.addPoint(self.line[1])
                        self.line[0] = self.current[-1]
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode in ["ai_polygon", "ai_mask"]:
                        self.current.addPoint(
                            self.line.points[1],
                            label=self.line.point_labels[1],
                        )
                        self.line.points[0] = self.current.points[-1]
                        self.line.point_labels[0] = self.current.point_labels[
                            -1
                        ]
                        if ev.modifiers() & QtCore.Qt.ControlModifier:
                            self.finalise()
                    elif self.createMode == "polygonSAM":
                        self.current.addPoint(self.line[1], True)
                        points = [[point.x(), point.y()]
                                  for point in self.current.points]
                        labels = [int(label) for label in self.current.labels]
                        self.samPrompt(np.array(points), np.array(labels))
                        if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                            self.finalise()
                elif not self.outOfPixmap(pos):
                    # Create new shape.
                    if self.createMode != "polygonSAM":
                        self.current = Shape(shape_type="points" if self.createMode in [
                                             "ai_polygon", "ai_mask", "grounding_sam"] else self.createMode)
                        self.current.addPoint(
                            pos, label=0 if is_shift_pressed else 1)
                        if self.createMode == "point":
                            self.finalise()
                        elif (
                            self.createMode in [
                                "ai_polygon", "ai_mask", "grounding_sam"]
                            and ev.modifiers() & QtCore.Qt.ControlModifier
                        ):
                            self.finalise()
                        else:
                            if self.createMode == "circle":
                                self.current.shape_type = "circle"
                            self.line.points = [pos, pos]
                            if (
                                self.createMode in [
                                    "ai_polygon", "ai_mask", "grounding_sam"]
                                and is_shift_pressed
                            ):
                                self.line.point_labels = [0, 0]
                            else:
                                self.line.point_labels = [1, 1]
                            self.setHiding()
                            self.drawingPolygon.emit(True)
                            self.update()
                    else:
                        self.current = MultipoinstShape()
                        self.current.addPoint(pos, True)
            elif self.editing():
                if self.selectedEdge():
                    self.addPointToEdge()
                elif (
                    self.selectedVertex()
                    and int(ev.modifiers()) == QtCore.Qt.ShiftModifier
                ):
                    # Delete point if: left-click + SHIFT on a point
                    self.removeSelectedPoint()

                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                self.selectShapePoint(pos, multiple_selection_mode=group_mode)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == QtCore.Qt.RightButton:
            if self.drawing() and self.createMode == "polygonSAM":
                if self.current:
                    self.current.addPoint(self.line[1], False)
                    points = [[point.x(), point.y()]
                              for point in self.current.points]
                    labels = [int(label) for label in self.current.labels]
                    self.samPrompt(np.array(points), np.array(labels))
                    if int(ev.modifiers()) == QtCore.Qt.ControlModifier:
                        self.finalise()
                elif not self.outOfPixmap(pos):
                    self.current = MultipoinstShape()
                    self.current.addPoint(pos, False)
                    self.line.points = [pos, pos]
                    self.setHiding()
                    self.drawingPolygon.emit(True)
                    self.update()
            elif self.editing():
                group_mode = int(ev.modifiers()) == QtCore.Qt.ControlModifier
                if not self.selectedShapes or (
                    self.hShape is not None
                    and self.hShape not in self.selectedShapes
                ):
                    self.selectShapePoint(
                        pos, multiple_selection_mode=group_mode)
                    self.repaint()
                self.prevPoint = pos

    def mouseReleaseEvent(self, ev):
        if ev.button() == QtCore.Qt.RightButton:
            if self.createMode != 'polygonSAM':
                menu = self.menus[len(self.selectedShapesCopy) > 0]
                self.restoreCursor()
                if (
                    not menu.exec_(self.mapToGlobal(ev.pos()))
                    and self.selectedShapesCopy
                ):
                    # Cancel the move by deleting the shadow copy.
                    self.selectedShapesCopy = []
                    self.repaint()
        elif ev.button() == QtCore.Qt.LeftButton:
            if self.editing():
                if (
                    self.hShape is not None
                    and self.hShapeIsSelected
                    and not self.movingShape
                ):
                    self.selectionChanged.emit(
                        [x for x in self.selectedShapes if x != self.hShape]
                    )

        if self.movingShape and self.hShape:
            index = self.shapes.index(self.hShape)
            if (
                self.shapesBackups[-1][index].points
                != self.shapes[index].points
            ):
                self.storeShapes()
                self.shapeMoved.emit()

            self.movingShape = False

    def endMove(self, copy):
        assert self.selectedShapes and self.selectedShapesCopy
        assert len(self.selectedShapesCopy) == len(self.selectedShapes)
        if copy:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.shapes.append(shape)
                self.selectedShapes[i].selected = False
                self.selectedShapes[i] = shape
        else:
            for i, shape in enumerate(self.selectedShapesCopy):
                self.selectedShapes[i].points = shape.points
        self.selectedShapesCopy = []
        self.repaint()
        self.storeShapes()
        return True

    def hideBackroundShapes(self, value):
        self.hideBackround = value
        if self.selectedShapes:
            # Only hide other shapes if there is a current selection.
            # Otherwise the user will not be able to select a shape.
            self.setHiding(True)
            self.update()

    def setHiding(self, enable=True):
        self._hideBackround = self.hideBackround if enable else False

    def canCloseShape(self):
        if self.createMode == "polygonSAM":
            return self.drawing() and self.current and len(self.current) > 0
        return self.drawing() and self.current and len(self.current) > 2

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if self.double_click != "close":
            return
        if (
            self.createMode == "polygon" and self.canCloseShape()
        ) or self.createMode in ["ai_polygon", "ai_mask", "grounding_sam"]:
            self.finalise()
        if (
            self.double_click == "close"
            and self.canCloseShape()
            and len(self.current) > 3
        ):
            self.current.popPoint()
            self.finalise()

    def selectShapes(self, shapes):
        self.setHiding()
        self.selectionChanged.emit(shapes)
        self.update()

    def selectShapePoint(self, point, multiple_selection_mode):
        """Select the first shape created which contains this point."""
        if self.selectedVertex():  # A vertex is marked for selection.
            index, shape = self.hVertex, self.hShape
            shape.highlightVertex(index, shape.MOVE_VERTEX)
        else:
            for shape in reversed(self.shapes):
                if self.isVisible(shape) and shape.containsPoint(point):
                    self.setHiding()
                    if shape not in self.selectedShapes:
                        if multiple_selection_mode:
                            self.selectionChanged.emit(
                                self.selectedShapes + [shape]
                            )
                        else:
                            self.selectionChanged.emit([shape])
                        self.hShapeIsSelected = False
                    else:
                        self.hShapeIsSelected = True
                    self.calculateOffsets(point)
                    return
        self.deSelectShape()

    def calculateOffsets(self, point):
        left = self.pixmap.width() - 1
        right = 0
        top = self.pixmap.height() - 1
        bottom = 0
        for s in self.selectedShapes:
            if s.shape_type == 'mask':
                continue
            rect = s.boundingRect()
            if rect.left() < left:
                left = rect.left()
            if rect.right() > right:
                right = rect.right()
            if rect.top() < top:
                top = rect.top()
            if rect.bottom() > bottom:
                bottom = rect.bottom()

        x1 = left - point.x()
        y1 = top - point.y()
        x2 = right - point.x()
        y2 = bottom - point.y()
        self.offsets = QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)

    def boundedMoveVertex(self, pos):
        index, shape = self.hVertex, self.hShape
        try:
            point = shape[index]
        except IndexError as e:
            logger.info(e)
            return
        if self.outOfPixmap(pos):
            pos = self.intersectionPoint(point, pos)
        shape.moveVertexBy(index, pos - point)

    def boundedMoveShapes(self, shapes, pos):
        if self.outOfPixmap(pos):
            return False  # No need to move
        o1 = pos + self.offsets[0]
        if self.outOfPixmap(o1):
            pos -= QtCore.QPoint(int(min(0, o1.x())), int(min(0, o1.y())))
        o2 = pos + self.offsets[1]
        if self.outOfPixmap(o2):
            pos += QtCore.QPoint(
                int(min(0, self.pixmap.width() - o2.x())),
                int(min(0, self.pixmap.height() - o2.y())),
            )
        # XXX: The next line tracks the new position of the cursor
        # relative to the shape, but also results in making it
        # a bit "shaky" when nearing the border and allows it to
        # go outside of the shape's area for some reason.
        # self.calculateOffsets(self.selectedShapes, pos)
        dp = pos - self.prevPoint
        if dp:
            for shape in shapes:
                shape.moveBy(dp)
            self.prevPoint = pos
            return True
        return False

    def deSelectShape(self):
        if self.selectedShapes:
            self.setHiding(False)
            self.selectionChanged.emit([])
            self.hShapeIsSelected = False
            self.update()

    def deleteSelected(self):
        deleted_shapes = []
        if self.selectedShapes:
            for shape in self.selectedShapes:
                self.shapes.remove(shape)
                deleted_shapes.append(shape)
            self.storeShapes()
            self.selectedShapes = []
            self.update()
        return deleted_shapes

    def deleteShape(self, shape):
        if shape in self.selectedShapes:
            self.selectedShapes.remove(shape)
        if shape in self.shapes:
            self.shapes.remove(shape)
        self.storeShapes()
        self.update()

    def duplicateSelectedShapes(self):
        if self.selectedShapes:
            self.selectedShapesCopy = [s.copy() for s in self.selectedShapes]
            self.boundedShiftShapes(self.selectedShapesCopy)
            self.endMove(copy=True)
        return self.selectedShapes

    def boundedShiftShapes(self, shapes):
        # Try to move in one direction, and if it fails in another.
        # Give up if both fail.
        point = shapes[0][0]
        offset = QtCore.QPointF(2.0, 2.0)
        self.offsets = QtCore.QPoint(), QtCore.QPoint()
        self.prevPoint = point
        if not self.boundedMoveShapes(shapes, point - offset):
            self.boundedMoveShapes(shapes, point + offset)

    def paintEvent(self, event):
        p = self._painter
        p.begin(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)
        p.setRenderHint(QtGui.QPainter.HighQualityAntialiasing)
        p.setRenderHint(QtGui.QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        # Check if pixmap is valid
        if self.pixmap and not self.pixmap.isNull():
            p.drawPixmap(0, 0, self.pixmap)

        self.sam_mask.paint(p)

        if self._patch_similarity_pixmap is not None:
            p.drawPixmap(0, 0, self._patch_similarity_pixmap)

        if self._pca_map_pixmap is not None:
            p.drawPixmap(0, 0, self._pca_map_pixmap)

        if self.current_behavior_text and len(self.current_behavior_text) > 0:
            p.save()
            p.resetTransform()

            if self.pixmap and not self.pixmap.isNull():
                base_size = min(self.pixmap.width(), self.pixmap.height())
            else:
                base_size = min(max(self.width(), 1), max(self.height(), 1))
            font_size = max(20, base_size // 40)

            font = QtGui.QFont()
            font.setPointSize(font_size)
            font.setBold(True)
            p.setFont(font)

            metrics = QtGui.QFontMetrics(font)
            text = self.current_behavior_text
            text_bounds = metrics.boundingRect(text)

            margin = 12
            padding_x = 8
            padding_y = 6

            background_rect = QtCore.QRect(
                margin,
                margin,
                text_bounds.width() + 2 * padding_x,
                text_bounds.height() + 2 * padding_y,
            )

            if self.behavior_text_background is not None:
                bg_color = QtGui.QColor(self.behavior_text_background)
                p.setPen(QtCore.Qt.NoPen)
                p.setBrush(QtGui.QBrush(bg_color))
                p.drawRoundedRect(background_rect, 6, 6)

            text_rect = QtCore.QRect(
                background_rect.left() + padding_x,
                background_rect.top() + padding_y,
                text_bounds.width(),
                text_bounds.height(),
            )

            p.setPen(QtGui.QPen(QtGui.QColor(self.behavior_text_color)))
            p.drawText(
                text_rect,
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                text,
            )

            p.restore()

        # draw crosshair
        if ((not self.createMode == 'grounding_sam')
            and (self._crosshair[self._createMode]
                 and self.drawing())):
            p.setPen(QtGui.QPen(QtGui.QColor(
                255, 255, 255), 1, QtCore.Qt.DotLine))
            p.drawLine(
                0,
                int(self.prevMovePoint.y()),
                self.width() - 1,
                int(self.prevMovePoint.y()),
            )
            p.drawLine(
                int(self.prevMovePoint.x()),
                0,
                int(self.prevMovePoint.x()),
                self.height() - 1,
            )

        Shape.scale = self.scale
        MultipoinstShape.scale = self.scale
        if self.pixmap and not self.pixmap.isNull():
            image_width = self.pixmap.width()
            image_height = self.pixmap.height()
        else:
            image_width = None
            image_height = None

        for shape in self.shapes:
            if (shape.selected or not self._hideBackround) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                try:
                    shape.paint(p, image_width, image_height)
                except SystemError as e:
                    print(e)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapesCopy:
            for s in self.selectedShapesCopy:
                s.paint(p)

        try:
            if (
                self.fillDrawing()
                and self.createMode == "polygon"
                and self.current is not None
                and len(self.current.points) >= 2
            ):
                drawing_shape = self.current.copy()
                drawing_shape.addPoint(self.line[1])
                drawing_shape.fill = True
                drawing_shape.paint(p)
            elif self.createMode == "ai_polygon" and self.current is not None:
                drawing_shape = self.current.copy()
                drawing_shape.addPoint(
                    point=self.line.points[1],
                    label=self.line.point_labels[1],
                )
                try:
                    if not self._ensure_ai_model_initialized():
                        logger.error(
                            "AI polygon model is not initialized; skipping prediction."
                        )
                    else:
                        points = self._ai_model.predict_polygon_from_points(
                            points=[
                                [point.x(), point.y()]
                                for point in drawing_shape.points
                            ],
                            point_labels=drawing_shape.point_labels,
                        )
                        if len(points) > 2:
                            drawing_shape.setShapeRefined(
                                shape_type="polygon",
                                points=[
                                    QtCore.QPointF(point[0], point[1])
                                    for point in points
                                ],
                                point_labels=[1] * len(points),
                            )
                            drawing_shape.fill = self.fillDrawing()
                            drawing_shape.paint(p)
                except Exception as e:
                    logger.error(
                        f"An error occurred during AI polygon prediction: {e}")

            elif self.createMode == "ai_mask" and self.current is not None:
                drawing_shape = self.current.copy()
                drawing_shape.addPoint(
                    point=self.line.points[1],
                    label=self.line.point_labels[1],
                )
                try:
                    if not self._ensure_ai_model_initialized():
                        logger.error(
                            "AI mask model is not initialized; skipping prediction."
                        )
                    else:
                        mask = self._ai_model.predict_mask_from_points(
                            points=[
                                [point.x(), point.y()]
                                for point in drawing_shape.points
                            ],
                            point_labels=drawing_shape.point_labels,
                        )
                        y1, x1, y2, x2 = imgviz.instances.mask_to_bbox(
                            [mask])[0].astype(int)
                        drawing_shape.setShapeRefined(
                            shape_type="mask",
                            points=[QtCore.QPointF(
                                x1, y1), QtCore.QPointF(x2, y2)],
                            point_labels=[1, 1],
                            mask=mask[y1:y2, x1:x2],
                        )
                        drawing_shape.selected = True
                        drawing_shape.paint(p)
                except Exception as e:
                    logger.error(
                        f"An error occurred during AI mask prediction: {e}")

            elif (
                # Check if the current mode is either 'linestrip' or 'line' to handle line segment drawing.
                self.createMode in ["linestrip", "line"]
                and self.drawing()
                and self.current
                and len(self.current.points) > 0
            ):
                # Get starting and end point for the line segment.
                start_point = self.current.points[-1]
                end_point = self.line.points[1]

                # Compute Euclidean distance in pixels.
                dx = end_point.x() - start_point.x()
                dy = end_point.y() - start_point.y()
                distance = (dx**2 + dy**2) ** 0.5

                # Calculate the midpoint of the line segment to place the text.
                mid_point = QtCore.QPointF(
                    (start_point.x() + end_point.x()) / 2,
                    (start_point.y() + end_point.y()) / 2,
                )

                # Adjust the font size with line length. Here we're using a simple scaling
                # factor (distance/10) while clamping the font size between 8 and 20 points.
                # Clamp the font size between 8 and 20 points.
                # Adjust the scaling factor and min/max values as needed.
                font_size = int(max(8, min(20, distance / 10)))

                # Create a font with the computed size.
                font = QtGui.QFont()
                font.setPointSize(font_size)

                # Use the pen already set for drawing lines (this reuses the line color).
                line_pen = p.pen()

                # Set the painter's font and pen.
                p.setFont(font)
                p.setPen(line_pen)
                # Draw the text near the midpoint with localization.
                localized_text = QtCore.QCoreApplication.translate(
                    "Canvas", "{:.1f}px".format(distance)
                )
                p.drawText(mid_point, localized_text)
                p.drawText(mid_point, f"{distance:.1f}px")

        except Exception as e:
            # General error handling for the entire block
            logger.error(f"An error occurred in paintEvent: {e}")

        p.end()

    def transformPos(self, point):
        """Convert from widget-logical coordinates to painter-logical ones."""
        return point / self.scale - self.offsetToCenter()

    def offsetToCenter(self):
        s = self.scale
        area = super(Canvas, self).size()
        aw, ah = area.width(), area.height()
        if self.pixmap is None:
            return QtCore.QPointF(0, 0)
        w, h = self.pixmap.width() * s, self.pixmap.height() * s
        x = (aw - w) / (2 * s) if aw > w else 0
        y = (ah - h) / (2 * s) if ah > h else 0
        return QtCore.QPointF(x, y)

    def outOfPixmap(self, p):
        if self.pixmap is None:
            return True
        w, h = self.pixmap.width(), self.pixmap.height()
        return not (0 <= p.x() <= w - 1 and 0 <= p.y() <= h - 1)

    def finalise(self):
        # assert self.current
        if self.createMode == "rectangle":
            self.current.close()
            rect_shape = self.current
            self.shapes.append(rect_shape)
            self.storeShapes()
            self.current = None
            self.setHiding(False)
            self.newShape.emit()
            self.update()

            # Get the label of the rectangle
            rect_label = rect_shape.label            
            # Launch CountGD if the description is not None and indicates exemplar, count, or starts with '0'
            if rect_shape.description is not None and \
               ("exemplar" in rect_shape.description.lower() or \
                "count" in rect_shape.description.lower() or \
                rect_shape.description.startswith('0')):
                rect_shape.description = f"used_{rect_shape.description}"
                self.createMode = "grounding_sam"
                # Call predictAiRectangle with the rectangle shape and its label
                self.predictAiRectangle(
                    rect_label, [rect_shape], is_polygon_output=True)
            return
        if self.createMode == "ai_polygon":
            # convert points to polygon by an AI model
            if self.current.shape_type != "points":
                return
            if not self._ensure_ai_model_initialized():
                logger.error(
                    "AI polygon model is not initialized; skipping finalisation."
                )
                return
            points = self._ai_model.predict_polygon_from_points(
                points=[
                    [point.x(), point.y()] for point in self.current.points
                ],
                point_labels=self.current.point_labels,
            )
            self.current.setShapeRefined(
                points=[
                    QtCore.QPointF(point[0], point[1]) for point in points
                ],
                point_labels=[1] * len(points),
                shape_type="polygon",
            )
        elif self.createMode == "ai_mask":
            # convert points to mask by an AI model
            assert self.current.shape_type == "points"
            if not self._ensure_ai_model_initialized():
                logger.error(
                    "AI mask model is not initialized; skipping finalisation."
                )
                return
            mask = self._ai_model.predict_mask_from_points(
                points=[
                    [point.x(), point.y()] for point in self.current.points
                ],
                point_labels=self.current.point_labels,
            )
            y1, x1, y2, x2 = imgviz.instances.mask_to_bbox([mask])[0].astype(
                int
            )
            self.current.setShapeRefined(
                shape_type="mask",
                points=[QtCore.QPointF(x1, y1), QtCore.QPointF(x2, y2)],
                point_labels=[1, 1],
                mask=mask[y1:y2, x1:x2],
            )
        if self.createMode != 'grounding_sam':
            self.current.close()
        if self.createMode == 'polygonSAM':
            self.shapes.append(self.sam_mask)
        else:
            self.shapes.append(self.current)
        self.storeShapes()
        self.sam_mask = MaskShape()
        self.current = None
        self.setHiding(False)
        self.newShape.emit()
        self.update()

    def closeEnough(self, p1, p2):
        # d = distance(p1 - p2)
        # m = (p1-p2).manhattanLength()
        # print "d %.2f, m %d, %.2f" % (d, m, d - m)
        # divide by scale to allow more precision when zoomed in
        return labelme.utils.distance(p1 - p2) < (self.epsilon / self.scale)

    def intersectionPoint(self, p1, p2):
        # Cycle through each image edge in clockwise fashion,
        # and find the one intersecting the current line segment.
        # http://paulbourke.net/geometry/lineline2d/
        size = self.pixmap.size()
        points = [
            (0, 0),
            (size.width() - 1, 0),
            (size.width() - 1, size.height() - 1),
            (0, size.height() - 1),
        ]
        # x1, y1 should be in the pixmap, x2, y2 should be out of the pixmap
        x1 = min(max(p1.x(), 0), size.width() - 1)
        y1 = min(max(p1.y(), 0), size.height() - 1)
        x2, y2 = p2.x(), p2.y()
        d, i, (x, y) = min(self.intersectingEdges((x1, y1), (x2, y2), points))
        x3, y3 = points[i]
        x4, y4 = points[(i + 1) % 4]
        if (x, y) == (x1, y1):
            # Handle cases where previous point is on one of the edges.
            if x3 == x4:
                return QtCore.QPointF(x3, min(max(0, y2), max(y3, y4)))
            else:  # y3 == y4
                return QtCore.QPointF(min(max(0, x2), max(x3, x4)), y3)
        return QtCore.QPointF(x, y)

    def intersectingEdges(self, point1, point2, points):
        """Find intersecting edges.

        For each edge formed by `points', yield the intersection
        with the line segment `(x1,y1) - (x2,y2)`, if it exists.
        Also return the distance of `(x2,y2)' to the middle of the
        edge along with its index, so that the one closest can be chosen.
        """
        (x1, y1) = point1
        (x2, y2) = point2
        for i in range(4):
            x3, y3 = points[i]
            x4, y4 = points[(i + 1) % 4]
            denom = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)
            nua = (x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)
            nub = (x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)
            if denom == 0:
                # This covers two cases:
                #   nua == nub == 0: Coincident
                #   otherwise: Parallel
                continue
            ua, ub = nua / denom, nub / denom
            if 0 <= ua <= 1 and 0 <= ub <= 1:
                x = x1 + ua * (x2 - x1)
                y = y1 + ua * (y2 - y1)
                m = QtCore.QPointF((x3 + x4) / 2, (y3 + y4) / 2)
                d = labelme.utils.distance(m - QtCore.QPointF(x2, y2))
                yield d, i, (x, y)

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(Canvas, self).minimumSizeHint()

    def wheelEvent(self, ev):
        if QT5:
            mods = ev.modifiers()
            delta = ev.angleDelta()
            if QtCore.Qt.ControlModifier == int(mods):
                # with Ctrl/Command key
                # zoom
                self.zoomRequest.emit(delta.y(), ev.pos())
            else:
                # scroll
                self.scrollRequest.emit(delta.x(), QtCore.Qt.Horizontal)
                self.scrollRequest.emit(delta.y(), QtCore.Qt.Vertical)
        else:
            if ev.orientation() == QtCore.Qt.Vertical:
                mods = ev.modifiers()
                if QtCore.Qt.ControlModifier == int(mods):
                    # with Ctrl/Command key
                    self.zoomRequest.emit(ev.delta(), ev.pos())
                else:
                    self.scrollRequest.emit(
                        ev.delta(),
                        QtCore.Qt.Horizontal
                        if (QtCore.Qt.ShiftModifier == int(mods))
                        else QtCore.Qt.Vertical,
                    )
            else:
                self.scrollRequest.emit(ev.delta(), QtCore.Qt.Horizontal)
        ev.accept()

    def moveByKeyboard(self, offset):
        if self.selectedShapes:
            self.boundedMoveShapes(
                self.selectedShapes, self.prevPoint + offset
            )
            self.repaint()
            self.movingShape = True

    def keyPressEvent(self, ev):
        modifiers = ev.modifiers()
        key = ev.key()
        if self.drawing():
            if key == QtCore.Qt.Key_Escape and self.current:
                self.current = None
                self.drawingPolygon.emit(False)
                self.update()
            elif key == QtCore.Qt.Key_Return and self.canCloseShape():
                self.finalise()
            elif modifiers == QtCore.Qt.AltModifier:
                self.snapping = False
        elif self.editing():
            if key == QtCore.Qt.Key_Up:
                self.moveByKeyboard(QtCore.QPointF(0.0, -MOVE_SPEED))
            elif key == QtCore.Qt.Key_Down:
                self.moveByKeyboard(QtCore.QPointF(0.0, MOVE_SPEED))
            elif key == QtCore.Qt.Key_Left:
                self.moveByKeyboard(QtCore.QPointF(-MOVE_SPEED, 0.0))
            elif key == QtCore.Qt.Key_Right:
                self.moveByKeyboard(QtCore.QPointF(MOVE_SPEED, 0.0))

    def keyReleaseEvent(self, ev):
        modifiers = ev.modifiers()
        if self.drawing():
            if int(modifiers) == 0:
                self.snapping = True
        elif self.editing():
            if self.movingShape and self.selectedShapes:
                try:
                    index = self.shapes.index(self.selectedShapes[0])
                except ValueError:
                    return
                if (
                    self.shapesBackups[-1][index].points
                    != self.shapes[index].points
                ):
                    self.storeShapes()
                    self.shapeMoved.emit()

                self.movingShape = False

    def setLastLabel(self, text, flags):
        assert text
        self.shapes[-1].label = text
        self.shapes[-1].flags = flags
        polygons = None
        if isinstance(self.shapes[-1], MaskShape):
            mask_shape = self.shapes.pop()
            polygons = mask_shape.toPolygons(
                self.sam_config["approxpoly_epsilon"])
            self.shapes.extend(polygons)
        self.shapesBackups.pop()
        self.storeShapes()

        if isinstance(polygons, list):
            return polygons
        else:
            return self.shapes[-1:]

    def undoLastLine(self):
        assert self.shapes
        self.current = self.shapes.pop()
        if self.createMode != "polygonSAM":
            self.current.setOpen()
        if self.createMode in ["polygon", "linestrip"]:
            self.line.points = [self.current[-1], self.current[0]]
        elif self.createMode in ["rectangle", "line", "circle"]:
            self.current.points = self.current.points[0:1]
        elif self.createMode in ["point", "polygonSAM"]:
            self.current = None
        self.drawingPolygon.emit(True)

    def undoLastPoint(self):
        if not self.current or self.current.isClosed():
            return
        self.current.popPoint()
        if len(self.current) > 0:
            self.line[0] = self.current[-1]
        else:
            self.current = None
            self.drawingPolygon.emit(False)
        self.update()

    def loadPixmap(self, pixmap, clear_shapes=True):
        self.pixmap = pixmap
        if clear_shapes:
            self.shapes = []
            self.sam_mask = MaskShape()
            self.current = None

        if self.createMode == "polygonSAM" and self.pixmap and self.sam_predictor:
            self.samEmbedding()
        self.update()

    def _predict_similar_rectangles(self, rectangle_shapes=None, prompt=None, confidence_threshold=0.23):
        """Predict more rectangle shapes without modifying self.shapes directly here."""
        detected_boxes = []
        if self.pixmap and not self.pixmap.isNull():
            try:
                from annolid.detector.countgd.predict import ObjectCounter
                qt_image = self.pixmap.toImage()
                image_data = convert_qt_image_to_rgb_cv_image(qt_image)
                pil_img = Image.fromarray(image_data)
                object_counter = ObjectCounter()
                exemplar_boxes = []
                for rectangle_shape in rectangle_shapes:

                    x1 = int(
                        min(rectangle_shape.points[0].x(), rectangle_shape.points[1].x()))
                    y1 = int(
                        min(rectangle_shape.points[0].y(), rectangle_shape.points[1].y()))
                    x2 = int(
                        max(rectangle_shape.points[0].x(), rectangle_shape.points[1].x()))
                    y2 = int(
                        max(rectangle_shape.points[0].y(), rectangle_shape.points[1].y()))
                    exemplar_boxes.append([x1, y1, x2, y2])

                detected_boxes = object_counter.count_objects(
                    pil_img,
                    text_prompt=prompt,
                    exemplar_image=pil_img,
                    exemplar_boxes=exemplar_boxes,
                    confidence_threshold=confidence_threshold,
                )

            except ImportError:
                logger.warning("...")
            except Exception as e:
                logger.error(f"Error in CountGD: {e}")
        return detected_boxes

    def loadShapes(self, shapes, replace=True):
        if replace:
            self.shapes = list(shapes)
        else:
            self.shapes.extend(shapes)
        self.storeShapes()
        self.current = None
        self.sam_mask = MaskShape()
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.update()

    def setRealtimeShapes(self, shapes):
        """Replace shapes without touching the undo stack."""
        self.shapes = list(shapes or [])
        self.current = None
        self.hShape = None
        self.hVertex = None
        self.hEdge = None
        self.selectedShapes = []
        self.selectedShapesCopy = []
        self.update()

    def setShapeVisible(self, shape, value):
        self.visible[shape] = value
        self.update()

    def overrideCursor(self, cursor):
        self.restoreCursor()
        self._cursor = cursor
        QtWidgets.QApplication.setOverrideCursor(cursor)

    def restoreCursor(self):
        QtWidgets.QApplication.restoreOverrideCursor()

    def resetState(self):
        self.restoreCursor()
        self.pixmap = None
        self.shapesBackups = []
        self.shapes = []
        self.sam_mask = MaskShape()
        self.update()
