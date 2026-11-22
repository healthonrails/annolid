import base64
import contextlib
import io
import os.path as osp

import PIL.Image

from annolid.annotation.shape_io import deserialize_shape_record
from annolid.utils.annotation_compat import __version__
from annolid.utils.logger import logger
from annolid.utils.annotation_compat import utils

from annolid.infrastructure import AnnotationStoreError, load_labelme_json
from annolid.infrastructure.persistence import atomic_write_json


@contextlib.contextmanager
def _allow_large_pil_images():
    previous_limit = PIL.Image.MAX_IMAGE_PIXELS
    PIL.Image.MAX_IMAGE_PIXELS = None
    try:
        yield
    finally:
        PIL.Image.MAX_IMAGE_PIXELS = previous_limit


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    with io.open(name, mode, encoding="utf-8") as file_obj:
        yield file_obj


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = ".json"

    def __init__(self, filename=None, is_video_frame=None):
        self.shapes = []
        self.flags = {}
        self.otherData = {}
        self.imagePath = None
        self.imageData = None
        self.imageHeight = None
        self.imageWidth = None
        self.is_video_frame = is_video_frame
        self.caption = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            with _allow_large_pil_images():
                image_pil = PIL.Image.open(filename)
                # apply orientation to image according to exif
                image_pil = utils.apply_exif_orientation(image_pil)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if ext in [".jpg", ".jpeg"]:
                format = "JPEG"
            else:
                format = "PNG"
            image_pil.save(f, format=format)
            f.seek(0)
            return f.read()

    def load(self, filename):
        keys = [
            "version",
            "imageData",
            "imagePath",
            "shapes",  # polygonal annotations
            "flags",  # image level flags
            "imageHeight",
            "imageWidth",
            "caption",
        ]
        try:
            data = load_labelme_json(filename)
        except AnnotationStoreError as err:
            raise LabelFileError(err) from err
        except Exception as e:
            raise LabelFileError(e)

        try:
            imageData = None
            imagePath = None
            caption = None

            if data.get("caption") is not None:
                caption = data["caption"]

            if data.get("imageData") is not None:
                imageData = base64.b64decode(data["imageData"])
            elif data.get("imagePath") is not None and data["imagePath"] != "":
                # relative path from label file to relative path from cwd
                resolved_image_path = osp.join(osp.dirname(filename), data["imagePath"])
                if osp.exists(resolved_image_path):
                    imageData = self.load_image_file(resolved_image_path)
                elif not self.is_video_frame:
                    logger.warning(
                        "Referenced image file not found for %s: %s",
                        filename,
                        resolved_image_path,
                    )
            flags = data.get("flags") or {}
            imageHeight = data.get("imageHeight")
            imageWidth = data.get("imageWidth")
            if data.get("imagePath") is not None:
                imagePath = data["imagePath"]
            if not self.is_video_frame and imageData is not None:
                imageHeight, imageWidth = self._check_image_height_and_width(
                    imageData,
                    imageHeight,
                    imageWidth,
                )
            shapes = [deserialize_shape_record(s) for s in data["shapes"]]
        except Exception as e:
            raise LabelFileError(e)

        otherData = {}
        for key, value in data.items():
            if key not in keys:
                otherData[key] = value

        # Only replace data after everything is loaded.
        self.flags = flags
        self.shapes = shapes
        self.imagePath = imagePath
        self.imageData = imageData
        self.imageHeight = imageHeight
        self.imageWidth = imageWidth
        self.filename = filename
        self.otherData = otherData
        self.caption = caption

    def get_caption(self):
        return self.caption

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        try:
            if isinstance(imageData, (bytes, bytearray)):
                raw = bytes(imageData)
            else:
                raw = base64.b64decode(imageData.encode("utf-8"))
            with _allow_large_pil_images():
                with PIL.Image.open(io.BytesIO(raw)) as image:
                    actual_width, actual_height = image.size
        except Exception:
            if isinstance(imageData, (bytes, bytearray)):
                img_arr = utils.img_data_to_arr(bytes(imageData))
            else:
                img_arr = utils.img_b64_to_arr(imageData)
            actual_height, actual_width = img_arr.shape[:2]
        if imageHeight is None:
            imageHeight = actual_height
        elif actual_height != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = actual_height
        if imageWidth is None:
            imageWidth = actual_width
        elif actual_width != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = actual_width
        return imageHeight, imageWidth

    def save(
        self,
        filename,
        shapes,
        imagePath,
        imageHeight,
        imageWidth,
        imageData=None,
        otherData=None,
        flags=None,
        caption=None,
    ):
        if imageData is not None:
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
            imageData = base64.b64encode(imageData).decode("utf-8")
        if otherData is None:
            otherData = {}
        if flags is None:
            flags = {}
        data = dict(
            version=__version__,
            flags=flags,
            shapes=shapes,
            imagePath=imagePath,
            imageData=imageData,
            imageHeight=imageHeight,
            imageWidth=imageWidth,
            caption=caption,
        )
        for key, value in otherData.items():
            if key not in data:
                data[key] = value
        try:
            atomic_write_json(
                filename,
                data,
                indent=2,
                ensure_ascii=False,
            )
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e) from e

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
