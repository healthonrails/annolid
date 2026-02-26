from __future__ import annotations

import io
import os
import os.path as osp
import re
from pathlib import Path
from typing import Set

import numpy as np
from PIL import Image, ImageQt
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Qt

from annolid.annotation.timestamps import convert_frame_number_to_time
from annolid.gui.label_file import LabelFile, LabelFileError
from annolid.gui.window_base import PY2, utils
from annolid.utils.annotation_store import AnnotationStore
from annolid.utils.logger import logger


class PersistenceLifecycleMixin:
    """Label persistence, title/state updates, and prediction cleanup helpers."""

    def _get_pil_image_from_state(self) -> Image.Image | None:
        if self.imageData is None:
            return None

        if isinstance(self.imageData, bytes):
            try:
                pil_image = Image.open(io.BytesIO(self.imageData))
            except Exception as e:
                logger.error(f"Failed to load PIL Image from bytes: {e}")
                return None
        elif isinstance(self.imageData, Image.Image):
            pil_image = self.imageData
        elif isinstance(
            self.imageData,
            tuple(
                filter(
                    None,
                    (
                        QtGui.QImage,
                        getattr(ImageQt, "ImageQt", None),
                    ),
                )
            ),
        ):
            qimage = QtGui.QImage(self.imageData)
            image_bytes = self._qimage_to_bytes(qimage)
            if image_bytes is None:
                logger.error("Failed to serialize QImage to bytes for saving.")
                return None
            try:
                with io.BytesIO(image_bytes) as buffer:
                    pil_image = Image.open(buffer)
                    pil_image.load()
            except Exception as e:
                logger.error(f"Failed to convert QImage to PIL Image: {e}")
                return None
        else:
            logger.warning(
                f"self.imageData is of an unexpected type ({type(self.imageData)}). "
                "Cannot convert to PIL.Image for saving."
            )
            return None

        if pil_image.mode != "RGB":
            return pil_image.convert("RGB")

        return pil_image

    def saveLabels(self, filename, save_image_data=True):
        lf = LabelFile()
        has_zone_shapes = False

        def with_manual_tag(description: str | None) -> str:
            text = (description or "").strip()
            if not text:
                return "manul"
            lowered_tokens = {token.strip().lower() for token in text.split(",")}
            if "manul" in lowered_tokens:
                return text
            return f"{text}, manul"

        def format_shape(s):
            data = s.other_data.copy()
            if s.description and "zone" in s.description.lower():
                pass
            if len(s.points) <= 1:
                s.shape_type = "point"
            data.update(
                dict(
                    label=s.label.encode("utf-8") if PY2 else s.label,
                    points=[(p.x(), p.y()) for p in s.points],
                    group_id=s.group_id,
                    shape_type=s.shape_type,
                    flags=s.flags,
                    mask=None
                    if s.mask is None
                    else utils.img_arr_to_b64(s.mask.astype(np.uint8)),
                    visible=s.visible,
                    description=with_manual_tag(s.description),
                )
            )
            return data

        shapes = [format_shape(item.shape()) for item in self.labelList]
        flags = {}
        if self.flag_widget:
            flags = {
                _flag: True
                for _flag in self.flag_widget._get_existing_flag_names()
                if self.is_behavior_active(self.frame_number, _flag)
            }

        if self.canvas.current_behavior_text is not None:
            behaviors = self.canvas.current_behavior_text.split(",")
            for behavior in behaviors:
                if len(behavior) > 0:
                    flags[behavior] = True
        try:
            imagePath = osp.relpath(self.imagePath, osp.dirname(filename))
            imageData = None
            if save_image_data and self._config["store_data"]:
                pil_image_to_save = self._get_pil_image_from_state()
                if pil_image_to_save:
                    imageData = utils.img_pil_to_data(pil_image_to_save)

            if osp.dirname(filename) and not osp.exists(osp.dirname(filename)):
                os.makedirs(osp.dirname(filename))
            lf.save(
                filename=filename,
                shapes=shapes,
                imagePath=Path(imagePath).name,
                imageData=imageData,
                imageHeight=self.image.height(),
                imageWidth=self.image.width(),
                otherData=self.otherData,
                flags=flags,
                caption=self.canvas.getCaption(),
            )
            if has_zone_shapes:
                self.zone_path = filename

            self.labelFile = lf
            items = self.fileListWidget.findItems(self.imagePath, Qt.MatchExactly)
            if len(items) > 0:
                if len(items) != 1:
                    raise RuntimeError("There are duplicate files.")
                items[0].setCheckState(Qt.Checked)
                items[0].setData(Qt.UserRole, True)
                items[0].setForeground(QtGui.QBrush())
            if shapes and self.video_file:
                self.video_manager_widget.json_saved.emit(self.video_file, filename)
                logger.debug(
                    f"Emitted VideoManagerWidget.json_saved for video: {self.video_file}, JSON: {filename}"
                )
            return True
        except LabelFileError as e:
            self.errorMessage(
                self.tr("Error saving label data"), self.tr("<b>%s</b>") % e
            )
            return False

    def _saveFile(self, filename):
        json_saved = self.saveLabels(filename)
        if filename and json_saved:
            self.changed_json_stats[filename] = (
                self.changed_json_stats.get(filename, 0) + 1
            )
            if (
                self.changed_json_stats[filename] >= 1
                and self._pred_res_folder_suffix in filename
            ):
                changed_folder = str(Path(filename).parent)
                if "_edited" not in changed_folder:
                    changed_folder = Path(changed_folder + "_edited")
                else:
                    changed_folder = Path(changed_folder)
                if not changed_folder.exists():
                    changed_folder.mkdir(exist_ok=True, parents=True)
                changed_filename = changed_folder / Path(filename).name
                _ = self.saveLabels(str(changed_filename))
                _ = self._saveImageFile(str(changed_filename).replace(".json", ".png"))
            image_filename = self._saveImageFile(filename)
            self._auto_collect_labelme_pair(filename, image_filename)
            self._save_ai_mask_renders(image_filename)
            self.imageList.append(image_filename)
            self.addRecentFile(filename)
            label_file = self._getLabelFile(filename)
            self._addItem(image_filename, label_file)

            if self.caption_widget is not None:
                self.caption_widget.set_image_path(image_filename)

            if self.video_results_folder:
                try:
                    self._refresh_manual_seed_slider_marks(self.video_results_folder)
                except Exception:
                    logger.debug(
                        "Failed to refresh manual seed marks after save.",
                        exc_info=True,
                    )

            self.setClean()

    def _auto_collect_labelme_pair(self, json_path: str, image_path: str) -> None:
        index_file = os.environ.get("ANNOLID_LABEL_INDEX_FILE", "").strip()
        if not index_file:
            index_file = (self.config or {}).get(
                "label_index_file"
            ) or self.settings.value("dataset/label_index_file", "", type=str)
        dataset_root = (
            os.environ.get("ANNOLID_LABEL_COLLECTION_DIR")
            or (self.config or {}).get("label_collection_dir")
            or self.settings.value("dataset/label_collection_dir", "", type=str)
        )
        dataset_root_path = Path(dataset_root).expanduser() if dataset_root else None
        try:
            from annolid.datasets.labelme_collection import (
                default_label_index_path,
                resolve_label_index_path,
            )
        except Exception:
            if not index_file:
                if dataset_root_path is not None:
                    index_file = str(
                        (
                            dataset_root_path.resolve()
                            / "logs"
                            / "label_index"
                            / "annolid_dataset.jsonl"
                        )
                    )
                else:
                    index_file = str(
                        (
                            Path.home()
                            / ".annolid"
                            / "logs"
                            / "label_index"
                            / "annolid_dataset.jsonl"
                        )
                        .expanduser()
                        .resolve()
                    )
            elif (
                dataset_root_path is not None
                and not Path(index_file).expanduser().is_absolute()
            ):
                index_file = str((dataset_root_path.resolve() / index_file).resolve())
        else:
            if not index_file:
                index_file = str(default_label_index_path(dataset_root_path))
            else:
                index_file = str(
                    resolve_label_index_path(Path(index_file), dataset_root_path)
                )

        include_empty_value = (
            os.environ.get("ANNOLID_LABEL_INDEX_INCLUDE_EMPTY", "0").strip().lower()
        )
        include_empty = include_empty_value in {"1", "true", "yes", "on"}

        try:
            from annolid.datasets.labelme_collection import index_labelme_pair

            index_labelme_pair(
                json_path=Path(json_path),
                index_file=Path(index_file),
                image_path=Path(image_path) if image_path else None,
                include_empty=include_empty,
                source="annolid_gui",
            )
        except Exception as exc:
            logger.warning("Auto label indexing failed for %s: %s", json_path, exc)

    def getLabelFile(self):
        if str(self.filename).lower().endswith(".json"):
            label_file = str(self.filename)
        else:
            label_file = osp.splitext(str(self.filename))[0] + ".json"

        return label_file

    def popLabelListMenu(self, point):
        try:
            self.menus.labelList.exec_(self.labelList.mapToGlobal(point))
        except AttributeError:
            return

    def setDirty(self):
        self.actions.undo.setEnabled(self.canvas.isShapeRestorable)
        if self._config["auto_save"] or self.actions.saveAuto.isChecked():
            if self.filename:
                label_file = osp.splitext(self.filename)[0] + ".json"
                if self.output_dir:
                    label_file_without_path = osp.basename(label_file)
                    label_file = osp.join(self.output_dir, label_file_without_path)
                self.saveLabels(label_file)
                self.saveFile()
                return
        self.dirty = True
        self.actions.save.setEnabled(True)
        title = "Annolid"
        if self.filename is not None:
            title = self.getTitle(clean=False)
        self.setWindowTitle(title)

    def getTitle(self, clean=True):
        title = "Annolid"
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(self.filename)
            if getattr(self.caption_widget, "behavior_widget", None) is not None:
                try:
                    self.caption_widget.behavior_widget.set_current_frame(
                        self.frame_number
                    )
                except Exception:
                    pass
        _filename = os.path.basename(self.filename)
        if self.video_loader:
            if self.frame_number:
                self._time_stamp = convert_frame_number_to_time(
                    self.frame_number, self.fps
                )
                if clean:
                    title = f"{title}-Video Timestamp:{self._time_stamp}|Events:{self.behavior_controller.events_count}"
                    title = f"{title}|Frame_number:{self.frame_number}"
                else:
                    title = f"{title}|Video Timestamp:{self._time_stamp}"
                    title = f"{title}|Frame_number:{self.frame_number}*"
            else:
                if clean:
                    title = "{} - {}".format(title, _filename)
                else:
                    title = "{} - {}*".format(title, _filename)
        return title

    def deleteAllFuturePredictions(self):
        if not self.video_loader or not self.video_results_folder:
            return

        prediction_folder = Path(self.video_results_folder)
        if not prediction_folder.exists():
            return

        deleted_files = 0
        protected_frames: Set[int] = set()

        logger.info(f"Scanning for future predictions in: {prediction_folder}")

        for prediction_path in prediction_folder.iterdir():
            if not prediction_path.is_file():
                continue
            if prediction_path.suffix.lower() != ".json":
                continue

            match = re.search(r"(\d+)(?=\.json$)", prediction_path.name)
            if not match:
                logger.debug(
                    "Skipping file with unexpected name format: %s",
                    prediction_path.name,
                )
                continue

            try:
                frame_number = int(float(match.group(1)))
            except (ValueError, IndexError):
                logger.warning(
                    "Could not parse frame number from file: %s",
                    prediction_path.name,
                )
                continue

            image_file_png = prediction_path.with_suffix(".png")
            image_file_jpg = prediction_path.with_suffix(".jpg")
            is_manually_saved = image_file_png.exists() or image_file_jpg.exists()
            if is_manually_saved:
                protected_frames.add(frame_number)

            is_future_frame = frame_number > self.frame_number

            if is_future_frame and not is_manually_saved:
                try:
                    prediction_path.unlink()
                    deleted_files += 1
                except OSError as e:
                    logger.error("Failed to delete file %s: %s", prediction_path, e)

        store_removed = 0
        try:
            store = AnnotationStore.for_frame_path(
                prediction_folder / f"{prediction_folder.name}_000000000.json"
            )
            store_removed = store.remove_frames_after(
                self.frame_number, protected_frames=protected_frames
            )
        except Exception as exc:
            logger.error(
                "Failed to prune annotation store in %s: %s",
                prediction_folder,
                exc,
            )

        if deleted_files or store_removed:
            logger.info(
                "%s future prediction JSON(s) removed and %s store record(s) pruned.",
                deleted_files,
                store_removed,
            )
            if self.seekbar:
                self.seekbar.removeMarksByType("predicted")
                self.seekbar.removeMarksByType("predicted_existing")
                self.seekbar.removeMarksByType("prediction_progress")
            self.last_known_predicted_frame = -1
            self.prediction_start_timestamp = 0.0
            if hasattr(self, "_update_progress_bar"):
                self._update_progress_bar(0)
            try:
                self._scan_prediction_folder(str(prediction_folder))
            except Exception as exc:
                logger.debug(
                    "Failed to rescan prediction folder after deletion: %s", exc
                )
        else:
            logger.info("No future prediction files or store records required removal.")

    def _collect_seed_frames(self, prediction_folder: Path) -> Set[int]:
        seed_frames: Set[int] = set()
        pattern = re.compile(r"(\d+)(?=\.(png|jpg|jpeg)$)", re.IGNORECASE)
        for path in prediction_folder.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in (".png", ".jpg", ".jpeg"):
                continue
            match = pattern.search(path.name)
            if not match:
                continue
            try:
                seed_frames.add(int(match.group(1)))
            except (TypeError, ValueError):
                continue
        return seed_frames

    def deletePredictionsFromSeedToNext(self):
        if not self.video_loader or not self.video_results_folder:
            return False, None, None

        prediction_folder = Path(self.video_results_folder)
        if not prediction_folder.exists():
            return False, None, None

        seed_frames = sorted(self._collect_seed_frames(prediction_folder))
        protected_frames: Set[int] = set(seed_frames)

        current_seed = None
        if seed_frames:
            for seed in seed_frames:
                if seed <= self.frame_number:
                    current_seed = seed
                else:
                    break
        if current_seed is None:
            current_seed = self.frame_number

        next_seed = None
        for seed in seed_frames:
            if seed > current_seed:
                next_seed = seed
                break

        if next_seed is not None and next_seed - 1 < current_seed:
            return False, current_seed, next_seed

        deleted_files = 0
        logger.info(
            "Deleting predictions from seed frame %s to %s.",
            current_seed,
            next_seed if next_seed is not None else "end",
        )

        for prediction_path in prediction_folder.iterdir():
            if not prediction_path.is_file():
                continue
            if prediction_path.suffix.lower() != ".json":
                continue

            match = re.search(r"(\d+)(?=\.json$)", prediction_path.name)
            if not match:
                continue

            try:
                frame_number = int(float(match.group(1)))
            except (ValueError, IndexError):
                continue

            if frame_number < current_seed:
                continue
            if next_seed is not None and frame_number >= next_seed:
                continue
            if frame_number in protected_frames:
                continue

            try:
                prediction_path.unlink()
                deleted_files += 1
            except OSError as e:
                logger.error("Failed to delete file %s: %s", prediction_path, e)

        store_removed = 0
        try:
            store = AnnotationStore.for_frame_path(
                prediction_folder / f"{prediction_folder.name}_000000000.json"
            )
            store_removed = store.remove_frames_in_range(
                current_seed,
                next_seed - 1 if next_seed is not None else None,
                protected_frames=protected_frames,
            )
        except Exception as exc:
            logger.error(
                "Failed to prune annotation store in %s: %s",
                prediction_folder,
                exc,
            )

        if deleted_files or store_removed:
            logger.info(
                "%s prediction JSON(s) removed and %s store record(s) pruned.",
                deleted_files,
                store_removed,
            )
            if self.seekbar:
                self.seekbar.removeMarksByType("predicted")
                self.seekbar.removeMarksByType("predicted_existing")
                self.seekbar.removeMarksByType("prediction_progress")
            self.last_known_predicted_frame = -1
            self.prediction_start_timestamp = 0.0
            if hasattr(self, "_update_progress_bar"):
                self._update_progress_bar(0)
            try:
                self._scan_prediction_folder(str(prediction_folder))
            except Exception as exc:
                logger.debug(
                    "Failed to rescan prediction folder after deletion: %s", exc
                )
        else:
            logger.info(
                "No predicted files required removal for the current seed range."
            )

        return bool(deleted_files or store_removed), current_seed, next_seed

    def deleteFile(self):
        mb = QtWidgets.QMessageBox
        msg = self.tr(
            "You are about to permanently delete this label file, "
            "Or delete predicted label files from the current seed frame "
            "to the next seed frame. "
            "What would you like to do?"
        )
        msg_box = mb(self)
        msg_box.setIcon(mb.Warning)
        msg_box.setText(msg)
        msg_box.setInformativeText(
            self.tr(
                "Yes: delete the current label file. "
                "Yes to All: delete predicted frames for the current seed range."
            )
        )
        msg_box.setStandardButtons(mb.No | mb.Yes | mb.YesToAll)
        msg_box.setDefaultButton(mb.No)
        answer = msg_box.exec_()

        if answer == mb.No:
            return
        elif answer == mb.YesToAll:
            removed, start_seed, next_seed = self.deletePredictionsFromSeedToNext()
            if removed:
                msg = self.tr(
                    "Delete all remaining predicted frames after this seed range?"
                )
                follow_up = mb.question(
                    self,
                    self.tr("Delete All Predictions"),
                    msg,
                    mb.Yes | mb.No,
                    mb.No,
                )
                if follow_up == mb.Yes:
                    self.deleteAllFuturePredictions()
        else:
            label_file = self.getLabelFile()
            if osp.exists(label_file):
                os.remove(label_file)
                img_file = label_file.replace(".json", ".png")
                if osp.exists(img_file):
                    os.remove(img_file)
                logger.info("Label file is removed: {}".format(label_file))

                item = self.fileListWidget.currentItem()
                if item:
                    item.setData(Qt.UserRole, False)
                    item.setForeground(QtGui.QBrush(QtGui.QColor(160, 160, 160)))

                self.resetState()

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.createMode.setEnabled(True)
        self.actions.createRectangleMode.setEnabled(True)
        self.actions.createCircleMode.setEnabled(True)
        self.actions.createLineMode.setEnabled(True)
        self.actions.createPointMode.setEnabled(True)
        self.actions.createLineStripMode.setEnabled(True)
        title = "Annolid"
        if self.filename is not None:
            title = self.getTitle()
        self.setWindowTitle(title)

        if self.hasLabelFile():
            self.actions.deleteFile.setEnabled(True)
        else:
            self.actions.deleteFile.setEnabled(False)

    def save_labels(self):
        file_name, extension = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save labels file",
            str(self.here.parent / "annotation"),
            filter="*.txt",
        )
        _ = extension
        if len(file_name) < 1:
            return

        if Path(file_name).is_file() or Path(file_name).parent.is_dir():
            labels_text_list = ["__ignore__", "_background_"]
            for i in range(self.uniqLabelList.count()):
                label_name = self.uniqLabelList.item(i).data(QtCore.Qt.UserRole)
                labels_text_list.append(label_name)

            with open(file_name, "w") as lt:
                for ltl in labels_text_list:
                    lt.writelines(str(ltl) + "\n")
