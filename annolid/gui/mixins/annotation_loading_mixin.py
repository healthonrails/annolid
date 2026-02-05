from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd
from qtpy import QtCore

from annolid.gui.label_file import LabelFile, LabelFileError
from annolid.gui.shape import Shape
from annolid.postprocessing.quality_control import pred_dict_to_labelme
from annolid.utils.annotation_store import AnnotationStore
from annolid.utils.logger import logger


class AnnotationLoadingMixin:
    """Annotation/label loading workflow for frames and images."""

    def loadLabels(self, shapes):
        s = []
        for shape in shapes:
            label = shape["label"]
            points = shape["points"]
            shape_type = shape["shape_type"]
            flags = shape["flags"]
            group_id = shape["group_id"]
            description = shape.get("description", "")
            other_data = shape["other_data"]
            if "visible" in shape:
                visible = shape["visible"]
            else:
                visible = True

            if not points:
                continue

            shape = Shape(
                label=label,
                shape_type=shape_type,
                group_id=group_id,
                description=description,
                mask=shape["mask"],
                visible=visible,
            )
            for x, y in points:
                shape.addPoint(QtCore.QPointF(x, y))
            shape.close()

            default_flags = {}
            if self._config["label_flags"]:
                for pattern, keys in self._config["label_flags"].items():
                    if re.match(pattern, label):
                        for key in keys:
                            default_flags[key] = False
            shape.flags = default_flags
            shape.flags.update(flags)
            shape.other_data = other_data
            s.append(shape)

        self.loadShapes(s)
        return s

    def loadShapes(self, shapes, replace=True):
        self._noSelectionSlot = True
        if replace:
            self.labelList.clear()
        for shape in shapes:
            if not isinstance(shape.points[0], QtCore.QPointF):
                shape.points = [QtCore.QPointF(x, y) for x, y in shape.points]
            self.addLabel(shape, rebuild_unique=False)
        self.labelList.clearSelection()
        self._noSelectionSlot = False
        self.canvas.loadShapes(shapes, replace=replace)
        self._rebuild_unique_label_list()
        try:
            caption = self.labelFile.get_caption() if self.labelFile else None
        except AttributeError:
            caption = None
        if caption is not None and len(caption) > 0:
            if self.caption_widget is None:
                self.openCaption()
            self.caption_widget.set_caption(caption)
            self.caption_widget.set_image_path(self.filename)

    def update_flags_from_file(self, label_file):
        """Update flags from label file with proper validation and error handling."""
        if not hasattr(label_file, "flags"):
            logger.warning("Label file has no flags attribute")
            return

        try:
            if isinstance(label_file.flags, dict):
                new_flags = label_file.flags.copy()
                flags_in_frame = ",".join(new_flags.keys())
                self.canvas.setBehaviorText(flags_in_frame)
                _existing_flags = self.flag_widget._get_existing_flag_names()
                for _flag in _existing_flags:
                    if _flag not in new_flags:
                        new_flags[_flag] = False
                self.flag_widget.loadFlags(new_flags)
            else:
                logger.error(f"Invalid flags format: {type(label_file.flags)}")
        except Exception as e:
            logger.error(f"Error updating flags: {e}")

    def _annotation_store_has_frame(self, label_json_file: str) -> bool:
        """Return True if the annotation store contains a record for the given label path."""
        try:
            path = Path(label_json_file)
            frame_number = AnnotationStore.frame_number_from_path(path)
            if frame_number is None:
                return False
            store = AnnotationStore.for_frame_path(path)
            if not store.store_path.exists():
                return False
            return store.get_frame(frame_number) is not None
        except Exception:
            return False

    def _annotation_store_frame_count(self) -> int:
        """Return the number of frames currently stored in the annotation store."""
        if not self.video_results_folder:
            return 0
        try:
            store = AnnotationStore.for_frame_path(
                self.video_results_folder
                / f"{self.video_results_folder.name}_000000000.json"
            )
            if not store.store_path.exists():
                return 0
            return len(list(store.iter_frames()))
        except Exception:
            return 0

    def _iter_frame_label_candidates(
        self, frame_number: int, frame_path: Optional[Path]
    ) -> list[Path]:
        """Return possible annotation paths for a given frame."""
        candidates: list[Path] = []

        def _append_candidate(path: Optional[Path]) -> None:
            if path is None:
                return
            if path not in candidates:
                candidates.append(path)

        if frame_path is not None:
            frame_path = Path(frame_path)
            if frame_path.suffix.lower() == ".json":
                _append_candidate(frame_path)
            else:
                _append_candidate(frame_path.with_suffix(".json"))

            stem = frame_path.stem
            if "_" in stem:
                alt_name = f"{stem.split('_')[-1]}.json"
                _append_candidate(frame_path.parent / alt_name)

        frame_tag = f"{int(frame_number):09}"
        base_dir: Optional[Path] = None
        if frame_path is not None:
            base_dir = frame_path.parent
        if self.video_results_folder:
            base_dir = self.video_results_folder

        if base_dir is not None:
            if self.video_results_folder:
                stem_name = self.video_results_folder.name
            elif frame_path is not None:
                stem_name = frame_path.stem.rsplit("_", 1)[0]
            else:
                stem_name = base_dir.name

            if stem_name:
                _append_candidate(base_dir / f"{stem_name}_{frame_tag}.json")
            _append_candidate(base_dir / f"{frame_tag}.json")

        if self.video_results_folder:
            pred_dir = self.video_results_folder / self._pred_res_folder_suffix
            if pred_dir.exists():
                stem_name = self.video_results_folder.name
                _append_candidate(pred_dir / f"{stem_name}_{frame_tag}.json")
                _append_candidate(pred_dir / f"{frame_tag}.json")

        if self.annotation_dir:
            annot_dir = Path(self.annotation_dir)
            stem_name = annot_dir.name
            _append_candidate(annot_dir / f"{stem_name}_{frame_tag}.json")
            _append_candidate(annot_dir / f"{frame_tag}.json")

        return candidates

    def loadPredictShapes(self, frame_number, filename):
        if self.caption_widget is not None:
            self.caption_widget.set_image_path(filename)

        frame_path = Path(filename) if filename else None
        label_candidates = self._iter_frame_label_candidates(frame_number, frame_path)

        seen_candidates: set[Path] = set()
        label_loaded = False
        for candidate in label_candidates:
            if candidate in seen_candidates:
                continue
            seen_candidates.add(candidate)

            candidate_exists = candidate.exists()
            candidate_in_store = self._annotation_store_has_frame(candidate)
            if not candidate_exists and not candidate_in_store:
                continue

            try:
                label_file = LabelFile(
                    str(candidate),
                    is_video_frame=True,
                )
            except LabelFileError as exc:
                logger.error(
                    "Failed to load label file %s: %s",
                    candidate,
                    exc,
                )
                continue
            except Exception as exc:
                logger.error(
                    "Unexpected error loading label file %s: %s",
                    candidate,
                    exc,
                )
                continue

            self.labelFile = label_file
            self.canvas.setBehaviorText(None)
            self.loadLabels(label_file.shapes)
            self.update_flags_from_file(label_file)
            if (
                len(self.canvas.current_behavior_text) > 1
                and "other" not in self.canvas.current_behavior_text.lower()
            ):
                self.add_highlighted_mark(
                    self.frame_number, mark_type=self.canvas.current_behavior_text
                )
            caption = label_file.get_caption()
            if caption is not None and len(caption) > 0:
                if self.caption_widget is None:
                    self.openCaption()
                self.caption_widget.set_caption(caption)
            elif self.caption_widget is not None:
                applied = self._apply_timeline_caption_if_available(
                    frame_number, only_if_empty=False
                )
                if not applied:
                    self.caption_widget.set_caption("")
            label_loaded = True
            break

        if label_loaded:
            return

        if self._df is not None and (frame_path is None or not frame_path.exists()):
            df_cur = self._df[self._df.frame_number == frame_number]
            frame_label_list = []
            pd.options.mode.chained_assignment = None
            for row in df_cur.to_dict(orient="records"):
                if "x1" not in row:
                    row["x1"] = 2
                    row["y1"] = 2
                    row["x2"] = 4
                    row["y2"] = 4
                    row["class_score"] = 1
                    df_cur.drop("frame_number", axis=1, inplace=True)
                    try:
                        instance_names = df_cur.apply(
                            lambda row: df_cur.columns[
                                [i for i in range(len(row)) if row[i] > 0][0]
                            ],
                            axis=1,
                        ).tolist()
                        row["instance_name"] = "_".join(instance_names)
                    except IndexError:
                        row["instance_name"] = "unknown"
                    row["segmentation"] = None
                pred_label_list = pred_dict_to_labelme(row)
                frame_label_list += pred_label_list

            self.loadShapes(frame_label_list)

        if not label_loaded and self.caption_widget is not None:
            applied = self._apply_timeline_caption_if_available(
                frame_number, only_if_empty=False
            )
            if not applied:
                self.caption_widget.set_caption("")
