import base64
import contextlib
import io
import json
import os.path as osp

import PIL.Image

from annolid.utils.annotation_compat import PY2
from annolid.utils.annotation_compat import QT4
from annolid.utils.annotation_compat import __version__
from annolid.utils.logger import logger
from annolid.utils.annotation_compat import utils

from annolid.utils.annotation_store import AnnotationStoreError, load_labelme_json

PIL.Image.MAX_IMAGE_PIXELS = None


@contextlib.contextmanager
def open(name, mode):
    assert mode in ["r", "w"]
    if PY2:
        mode += "b"
        encoding = None
    else:
        encoding = "utf-8"
    yield io.open(name, mode, encoding=encoding)
    return


class LabelFileError(Exception):
    pass


class LabelFile(object):
    suffix = ".json"

    def __init__(self, filename=None, is_video_frame=None):
        self.shapes = []
        self.imagePath = None
        self.imageData = None
        self.is_video_frame = is_video_frame
        self.caption = None
        if filename is not None:
            self.load(filename)
        self.filename = filename

    @staticmethod
    def load_image_file(filename):
        try:
            image_pil = PIL.Image.open(filename)
        except IOError:
            logger.error("Failed opening image file: {}".format(filename))
            return

        # apply orientation to image according to exif
        image_pil = utils.apply_exif_orientation(image_pil)

        with io.BytesIO() as f:
            ext = osp.splitext(filename)[1].lower()
            if PY2 and QT4:
                format = "PNG"
            elif ext in [".jpg", ".jpeg"]:
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
        shape_keys = [
            "label",
            "points",
            "group_id",
            "shape_type",
            "flags",
            "description",
            "mask",
            "visible",
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
                if PY2 and QT4:
                    imageData = utils.img_data_to_png_data(imageData)
            elif data.get("imagePath") is not None and data["imagePath"] != "":
                # relative path from label file to relative path from cwd
                imagePath = osp.join(osp.dirname(filename), data["imagePath"])
                if osp.exists(imagePath):
                    imageData = self.load_image_file(imagePath)
            flags = data.get("flags") or {}
            if self.is_video_frame is None and data.get("imagePath") is not None:
                imagePath = data["imagePath"]
                self._check_image_height_and_width(
                    base64.b64encode(imageData).decode("utf-8"),
                    data.get("imageHeight"),
                    data.get("imageWidth"),
                )
            shapes = [
                dict(
                    label=s["label"],
                    points=s["points"],
                    shape_type=s.get("shape_type", "polygon"),
                    flags=s.get("flags", {}),
                    description=s.get("description"),
                    group_id=s.get("group_id"),
                    mask=utils.img_b64_to_arr(s["mask"]).astype(bool)
                    if s.get("mask")
                    else None,
                    # Backward-compatible default: missing/null visibility should
                    # keep shapes visible in the canvas and checked in the list.
                    visible=True
                    if s.get("visible") is None
                    else bool(s.get("visible")),
                    other_data={k: v for k, v in s.items() if k not in shape_keys},
                )
                for s in data["shapes"]
            ]
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
        self.filename = filename
        self.otherData = otherData
        self.caption = caption

    def get_caption(self):
        return self.caption

    @staticmethod
    def _check_image_height_and_width(imageData, imageHeight, imageWidth):
        img_arr = utils.img_b64_to_arr(imageData)
        if imageHeight is not None and img_arr.shape[0] != imageHeight:
            logger.error(
                "imageHeight does not match with imageData or imagePath, "
                "so getting imageHeight from actual image."
            )
            imageHeight = img_arr.shape[0]
        if imageWidth is not None and img_arr.shape[1] != imageWidth:
            logger.error(
                "imageWidth does not match with imageData or imagePath, "
                "so getting imageWidth from actual image."
            )
            imageWidth = img_arr.shape[1]
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
            imageData = base64.b64encode(imageData).decode("utf-8")
            imageHeight, imageWidth = self._check_image_height_and_width(
                imageData, imageHeight, imageWidth
            )
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
            with open(filename, "w") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            self.filename = filename
        except Exception as e:
            raise LabelFileError(e)

    @staticmethod
    def is_label_file(filename):
        return osp.splitext(filename)[1].lower() == LabelFile.suffix
